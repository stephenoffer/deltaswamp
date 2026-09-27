"""The one place an engine's exceptions become this library's.

Every engine a connection routes to is handed out wrapped in a `Boundary`
(the router wraps them as they are registered). A call through it that raises
anything but a `DeltaSwampError` -- delta-rs's DeltaError, a pyarrow error, the
extension's own exception types, a Rust panic -- is translated by the ordered
rule table of that engine, and what no rule matches arrives as `EngineError`
carrying the engine, the operation and the original. The same holds for what
the call returns and raises later: an Arrow stream failing mid-read, a MERGE
builder failing at `execute()`.

Before this, each engine translated at its own call sites (a context manager
around some delta-rs calls, a monkeypatch of every kernel method, a helper
around two PyIceberg writes), so the same wrong input raised a different type
depending on the engine that served it, and anything the sites missed escaped
raw past `except DeltaSwampError`.

Rules match exception types first. The extension sets a stable `kind` code on
each of its exceptions (crates/native/src/error.rs), matched instead of its
messages. delta-rs and PyIceberg raise one type for many failures, so their
rules still read the message; each such rule names the text it matches.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import re
from collections.abc import Callable, Iterator
from typing import Any

from ..capability import Engine as EngineKind
from ..errors import (
    BackfillRequiredError,
    CommitConflictError,
    CredentialError,
    DeltaSwampError,
    EngineLimitError,
    EnginePanicError,
    InvalidArgumentError,
    InvalidReferenceError,
    PreflightError,
    StorageError,
    TransientCommitError,
    UnreachableTableError,
    engine_error,
)

__all__ = [
    "Boundary",
    "GuardedEngines",
    "engine_cause",
    "guard",
    "translate",
    "translating",
]

#: A rule: the translation of `exc`, raised during `what`, or None to pass.
Rule = Callable[[BaseException, str], "BaseException | None"]

#: What each engine is called in messages.
_NAMES = {
    EngineKind.KERNEL: "the kernel",
    EngineKind.DELTARS: "delta-rs",
    EngineKind.SQL: "the SQL warehouse",
    EngineKind.SHARING: "the Delta Sharing engine",
    EngineKind.ICEBERG: "the Iceberg engine",
}


def _first_line(exc: BaseException, limit: int = 300) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:limit] if text else type(exc).__name__


def _detail(exc: BaseException) -> str:
    return " ".join(line.strip() for line in str(exc).splitlines() if line.strip())[:400]


def _during(what: str) -> str:
    """`what` after "failed" or "panicked": ``in files()`` for a method name, else ``to <what>``."""
    return f"in {what}()" if what.isidentifier() else f"to {what}"


def _is_panic(exc: BaseException) -> bool:
    # pyo3_runtime.PanicException derives from BaseException, so it sails
    # past `except Exception` and any retry logic.
    return type(exc).__name__ == "PanicException"


def _native(name: str) -> Any:
    """The extension's exception class `name`, or a type nothing is an instance of."""
    try:
        from deltaswamp import _native
    except ImportError:
        return ()
    return getattr(_native, name, ())


# ---------------------------------------------------------------------------
# Rules every engine shares
# ---------------------------------------------------------------------------


def _panic(kind: EngineKind) -> Rule:
    def rule(exc: BaseException, what: str) -> BaseException | None:
        if not _is_panic(exc):
            return None
        return EnginePanicError(
            f"{_NAMES.get(kind, kind.value)} panicked {_during(what)}: {exc}. This is "
            "a bug in the engine rather than in your input; deltaswamp validates properties "
            "up front to avoid the known cases."
        )

    return rule


def _storage(exc: BaseException, what: str) -> BaseException | None:
    """A storage failure (unreachable, throttled, no such bucket) as StorageError.

    Not a missing file or a refused permission: those keep their builtin class
    (FileNotFoundError, PermissionError) inside the generic EngineError, and
    reads name the missing file as MissingDataFileError.
    """
    if not isinstance(exc, OSError) or isinstance(exc, (FileNotFoundError, PermissionError)):
        return None
    error = StorageError(f"{what}: the table's storage failed the request ({_detail(exc)})")
    error.errno = exc.errno
    return error


# ---------------------------------------------------------------------------
# The kernel (the native extension)
# ---------------------------------------------------------------------------


def conflict_version(message: str) -> int:
    """The version someone else won, if the message names one; -1 otherwise."""
    found = re.search(r"version (\d+)", message)
    return int(found.group(1)) if found else -1


def _kernel_commit(exc: BaseException, what: str) -> BaseException | None:
    """The extension's commit outcomes, as the library types a caller can act on.

    The native ones are RuntimeErrors, and reached callers untranslated -- so
    `except DeltaSwampError` around a commit caught nothing, which is
    precisely the case it exists for.
    """
    if isinstance(exc, _native("BackfillRequiredError")):
        return BackfillRequiredError(str(exc))
    if isinstance(exc, _native("CommitConflictError")):
        message = str(exc)
        if "do not reuse the staged file" in message:
            # Advice for a hand-staged catalog commit, which named a staged
            # file and a txnId to a caller who had neither.
            message = (
                f"another writer committed version {conflict_version(message)} first, and "
                "nothing was committed. Re-read the table and retry the write."
            )
        return CommitConflictError(conflict_version(str(exc)), message)
    if isinstance(exc, _native("RetryableError")):
        return TransientCommitError(str(exc))
    return None


def _kernel_catalog(exc: BaseException, what: str) -> BaseException | None:
    """A catalog's refusal of a commit: nothing was committed."""
    if isinstance(exc, _native("CatalogNotFoundError")):
        return InvalidReferenceError(
            "the catalog no longer has this table: it was dropped (or renamed) "
            f"after it was opened, and the commit was refused. Re-resolve it. ({exc})"
        )
    if isinstance(exc, _native("CatalogPermissionError")):
        # The extension types what the UC client words without a status (a
        # 401 is "Authentication failed").
        if "authentication failed" in str(exc).lower() or "401" in str(exc):
            return CredentialError(
                "the catalog rejected the commit's credentials (expired or invalid "
                f"token); nothing was committed. ({exc})"
            )
        return PreflightError(
            "the catalog refused the commit: the principal may not modify this "
            f"table; nothing was committed. ({exc})"
        )
    if isinstance(exc, _native("CatalogCommitError")):
        # Anything else -- a 400, say -- is still the catalog's refusal.
        return UnreachableTableError(
            "commit to the catalog", f"the catalog refused the commit ({exc})"
        )
    return _uc_commit_http_error(exc)


def _uc_commit_http_error(exc: BaseException) -> BaseException | None:
    """A UC commit API status the extension reports only in its message.

    The extension classifies 409 and 429; any other status (the table dropped
    under the writer, an expired token, a lost privilege) reached callers as a
    bare ValueError, after the data files were written. Matches the UC
    client's "UC update_table error ... status NNN".
    """
    message = str(exc)
    if not isinstance(exc, ValueError) or "UC update_table error" not in message:
        return None
    found = re.search(r"status\D{0,3}(\d{3})", message)
    status = int(found.group(1)) if found else None
    if status == 404:
        return InvalidReferenceError(
            "the catalog no longer has this table: it was dropped (or renamed) after it was "
            f"opened, and the commit was refused. Re-resolve it. ({message})"
        )
    if status == 401:
        return CredentialError(
            "the catalog rejected the commit's credentials (expired or invalid token); "
            f"nothing was committed. ({message})"
        )
    if status == 403:
        return PreflightError(
            "the catalog refused the commit: the principal may not modify this table "
            f"(MODIFY, plus USE SCHEMA and USE CATALOG); nothing was committed. ({message})"
        )
    # A 5xx is left alone: the catalog may have ratified the commit before
    # failing, so "unchanged, retry" (TransientCommitError) would be a guess.
    return None


#: The kernel's words for a binary partition value it cannot write as text.
_BINARY_PARTITION = "binary partition value is not valid UTF-8"


def _kernel_input(exc: BaseException, what: str) -> BaseException | None:
    """Input the extension refused: a column not in the table, data that does not fit.

    `Table` treats anything that is not one of this library's errors as the
    engine breaking, so a typo in `columns=` once warned "kernel failed to
    serve scan", fell back to delta-rs and surfaced as its DeltaError.
    InvalidArgumentError is a ValueError too, so `except ValueError` works.
    """
    code = getattr(exc, "kind", None)
    if isinstance(exc, _native("InvalidInputError")) or code == "invalid_input":
        return InvalidArgumentError(str(exc))
    if what == "commit" and (code == "arrow" or _BINARY_PARTITION in str(exc)):
        # Arrow refusing the rows being committed (a null in a NOT NULL
        # column) is the data not fitting the table.
        return InvalidArgumentError(str(exc))
    return None


# ---------------------------------------------------------------------------
# delta-rs
# ---------------------------------------------------------------------------

#: delta-rs's wording for a commit that lost its race: a conflict found by its
#: checker, a version someone else wrote, or retries used up (the bare number).
_CONFLICT_MESSAGE = re.compile(
    r"concurrent|changed since last commit|existing table version|"
    r"Failed to commit transaction: \d+\s*$",
    re.IGNORECASE,
)


def as_commit_conflict(exc: BaseException) -> CommitConflictError | None:
    """This library's CommitConflictError for a delta-rs lost commit race, else None.

    delta-rs raises its own CommitFailedError, so `except CommitConflictError`
    around a write caught the kernel's lost races but never delta-rs's.
    """
    if type(exc).__name__ != "CommitFailedError" or isinstance(exc, CommitConflictError):
        return None
    message = str(exc)
    if not _CONFLICT_MESSAGE.search(message):
        return None
    found = re.search(r"version:? (\d+)", message)
    return CommitConflictError(
        int(found.group(1)) if found else -1,
        f"another writer committed first: {message}. Re-read the table and retry",
    )


def _deltars_conflict(exc: BaseException, what: str) -> BaseException | None:
    return as_commit_conflict(exc)


def _deltars_fork(exc: BaseException, what: str) -> BaseException | None:
    """delta-rs refusing to run in a forked child of a process that already used it.

    It keeps one tokio runtime per process. Not a bug in the input or in the
    engine's logic; the fix is the caller's start method.
    """
    if not _is_panic(exc) or "Forked process detected" not in str(exc):
        return None
    from .deltars import _FORKED_RUNTIME

    _FORKED_RUNTIME.add(os.getpid())
    return UnreachableTableError(
        what,
        "delta-rs cannot run in a process forked from one that already used it "
        "(its tokio runtime does not survive fork)",
        "start worker processes with multiprocessing's 'spawn' or 'forkserver' "
        "method, or open the table only in the children",
    )


def _deltars_constraint(exc: BaseException, what: str) -> BaseException | None:
    """NOT NULL, CHECK, invariants and generated columns refusing bad rows.

    delta-rs raises one DeltaError, "... failed validation check", for all of
    them; the kernel path raises InvalidArgumentError for the same input.
    """
    if type(exc).__name__ != "DeltaError" or "failed validation check" not in str(exc):
        return None
    return InvalidArgumentError(
        f"the data violates the table's constraints, so nothing was written: {exc}"
    )


#: delta-rs's messages for input that is wrong whatever engine serves it: SQL
#: that ends mid-expression or leaves a quote open, a partitioning that is not
#: the table's, a name the table does not have, nothing to convert.
_DELTARS_BAD_INPUT = (
    re.compile(r"sql parser error: .*found: EOF", re.IGNORECASE),
    re.compile(r"Unterminated string literal"),
    re.compile(r"Specified table partitioning does not match table partitioning"),
    re.compile(r"Constraint with name .* does not exist"),
    re.compile(r"No field with the provided name in the schema"),
    re.compile(r"No parquet file is found in the given location"),
    re.compile(r"Schema error: No field named"),
)

#: DataFusion's words for SQL it could not parse, plan or type -- found before
#: any row is read, so nothing has been written. Spark SQL it cannot run is a
#: limit of delta-rs, not bad input: a MERGE moves on to an engine that can.
_DATAFUSION_SQL_ERRORS = (
    "SQL error: ParserError",
    "Error during planning",
    "This feature is not implemented",
    "type_coercion",
    "Invalid comparison operation",
)


def _deltars_sql(exc: BaseException, what: str) -> BaseException | None:
    """delta-rs's DeltaError for SQL or input it refused, as this library's error."""
    if type(exc).__name__ != "DeltaError":
        return None
    message = str(exc)
    first = _first_line(exc)
    if any(pattern.search(message) for pattern in _DELTARS_BAD_INPUT):
        return InvalidArgumentError(f"cannot {what}: {first}")
    if not any(marker in message for marker in _DATAFUSION_SQL_ERRORS):
        return None
    return EngineLimitError(
        f"{what} with delta-rs",
        f"DataFusion, which evaluates delta-rs's SQL, cannot run it ({first})",
        "ds.connect(..., allow_sql_fallback=True) runs it on a SQL warehouse; or rewrite "
        "the expression",
    )


def _deltars_schema(exc: BaseException, what: str) -> BaseException | None:
    """Data that does not fit the table: SchemaMismatchError, or an Arrow cast error.

    delta-rs raises the cast failure as a bare Exception naming it.
    """
    name = type(exc).__name__
    if name == "SchemaMismatchError":
        return InvalidArgumentError(
            f"{what}: the data does not fit the table's schema ({_detail(exc)})"
        )
    if name in ("DeltaError", "Exception") and re.search(
        r"No field named|Cast error|Schema error", str(exc)
    ):
        return InvalidArgumentError(f"{what}: {_detail(exc)}")
    return None


# ---------------------------------------------------------------------------
# PyIceberg
# ---------------------------------------------------------------------------


def _iceberg_schema(exc: BaseException, what: str) -> BaseException | None:
    """PyIceberg's schema-mismatch ValueError, as a refusal naming the remedy."""
    text = str(exc)
    if not isinstance(exc, ValueError) or not (
        "more columns" in text or "Mismatch in fields" in text
    ):
        return None
    return UnreachableTableError(
        f"{what} an Iceberg table" if what == "overwrite" else f"{what} to an Iceberg table",
        f"the data does not match the table's Iceberg schema: {text}",
        "the Iceberg engine does not evolve schemas (no schema_mode='merge'); "
        "ALTER the table first, or select and cast the data to its schema",
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

#: Rules for the extension's own exception types. Every engine gets them: the
#: Delta Sharing engine and delta-rs call the kernel too.
_NATIVE: tuple[Rule, ...] = (_kernel_commit, _kernel_catalog, _kernel_input)

#: Each engine's rules, tried in order; the first to answer wins.
RULES: dict[EngineKind, tuple[Rule, ...]] = {
    EngineKind.KERNEL: (*_NATIVE, _panic(EngineKind.KERNEL), _storage),
    EngineKind.DELTARS: (
        _deltars_conflict,
        _deltars_fork,
        _panic(EngineKind.DELTARS),
        _deltars_constraint,
        _deltars_sql,
        _deltars_schema,
        *_NATIVE,
        _storage,
    ),
    EngineKind.ICEBERG: (_iceberg_schema, *_NATIVE, _panic(EngineKind.ICEBERG), _storage),
    EngineKind.SQL: (*_NATIVE, _panic(EngineKind.SQL), _storage),
    EngineKind.SHARING: (*_NATIVE, _panic(EngineKind.SHARING), _storage),
}


def translate(kind: EngineKind, what: str, exc: BaseException) -> BaseException:
    """`exc`, raised by engine `kind` during `what`, as a DeltaSwampError.

    A DeltaSwampError is returned as it is. Anything else becomes the answer of
    the first of the engine's rules that has one, else an EngineError. The
    translation carries the engine's exception as `original` (and it is set
    as the `__cause__` where it is raised).
    """
    if isinstance(exc, DeltaSwampError):
        return exc
    for rule in RULES.get(kind, (*_NATIVE, _panic(kind), _storage)):
        translated = rule(exc, what)
        if translated is not None:
            with contextlib.suppress(AttributeError):
                translated.original = exc  # type: ignore[attr-defined]
            return translated
    return engine_error(
        f"{_NAMES.get(kind, kind.value)} failed {_during(what)}: {type(exc).__name__}: {exc}",
        engine=kind.value,
        operation=what,
        original=exc,
    )


def engine_cause(exc: BaseException) -> BaseException:
    """The engine's own exception behind a translated one; `exc` if it is not one."""
    original = getattr(exc, "original", None)
    return original if isinstance(original, BaseException) else exc


def _passes(exc: BaseException) -> bool:
    """Whether `exc` goes through the boundary untranslated."""
    if isinstance(exc, DeltaSwampError):
        return True
    if isinstance(exc, (StopIteration, StopAsyncIteration)):
        return True  # iteration protocol, not a failure
    return not isinstance(exc, Exception) and not _is_panic(exc)


@contextlib.contextmanager
def translating(kind: EngineKind, what: str) -> Iterator[None]:
    """Translate what the block raises, as the boundary would.

    For the few places inside an engine that must act on a typed error before
    the call returns -- a retry loop that re-reads the table on a lost commit
    race -- so they use the boundary's rules rather than a second set.
    """
    try:
        yield
    except BaseException as exc:
        if _passes(exc):
            raise
        raise translate(kind, what, exc) from exc


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


class Boundary:
    """An engine (or an object one returned), with every call's errors translated.

    Attribute reads, writes and deletes go to the wrapped object, and
    `isinstance` sees its class, so a `Boundary` stands in for the engine
    anywhere. A method returns what the wrapped one does, except that an Arrow
    stream, a generator or a MERGE builder comes back wrapped too, since they
    fail only once used; a method returning the object itself (a builder's
    chaining) returns the wrapper.
    """

    __slots__ = ("_boundary_kind", "_boundary_prefix", "_boundary_target")

    def __init__(self, target: Any, kind: EngineKind, prefix: str | None = None) -> None:
        object.__setattr__(self, "_boundary_target", target)
        object.__setattr__(self, "_boundary_kind", kind)
        object.__setattr__(self, "_boundary_prefix", prefix)

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # isinstance() sees the engine
        return type(self._boundary_target)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._boundary_target, name)
        if (
            name.startswith("__")
            or not callable(value)
            or isinstance(value, type)
            or getattr(value, "_boundary_guarded", False)
        ):
            return value
        return self._guarded(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._boundary_target, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._boundary_target, name)

    def __dir__(self) -> list[str]:
        return dir(self._boundary_target)

    def __repr__(self) -> str:
        return repr(self._boundary_target)

    def __eq__(self, other: object) -> bool:
        return bool(self._boundary_target == unwrap(other))

    def __hash__(self) -> int:
        return hash(self._boundary_target)

    def __reduce__(self) -> tuple[Any, ...]:
        # Plans carry their engine to Ray workers: the worker's calls are
        # translated the same way.
        return (Boundary, (self._boundary_target, self._boundary_kind, self._boundary_prefix))

    def _guarded(self, name: str, method: Any) -> Any:
        kind = self._boundary_kind
        what = f"{self._boundary_prefix} ({name})" if self._boundary_prefix else name

        def call(*args: Any, **kwargs: Any) -> Any:
            with translating(kind, what):
                result = method(*args, **kwargs)
            return self._wrapped(result, what)

        call._boundary_guarded = True  # type: ignore[attr-defined]
        call.__name__ = getattr(method, "__name__", name)
        call.__doc__ = getattr(method, "__doc__", None)
        call.__wrapped__ = method  # type: ignore[attr-defined]
        return call

    def _wrapped(self, result: Any, what: str) -> Any:
        target = self._boundary_target
        if result is target:
            return self
        if result is None or isinstance(result, (str, bytes, int, float, dict, list, tuple)):
            return result
        kind = self._boundary_kind
        if hasattr(result, "__arrow_c_stream__") and hasattr(result, "read_next_batch"):
            from .base import translating_stream

            return translating_stream(
                result,
                what,
                lambda exc: None if _passes(exc) else translate(kind, what, exc),
                missing_files=False,
            )
        if inspect.isgenerator(result):
            return _generator(result, kind, what)
        if callable(getattr(result, "execute", None)) and hasattr(result, "when_matched_update"):
            return Boundary(result, kind, prefix=what)
        return result


def _generator(source: Any, kind: EngineKind, what: str) -> Iterator[Any]:
    with translating(kind, what):
        yield from source


def unwrap(value: Any) -> Any:
    """The object behind a Boundary; `value` if it is not one."""
    if type(value) is Boundary:
        return object.__getattribute__(value, "_boundary_target")
    return value


def guard(kind: EngineKind, engine: Any) -> Any:
    """`engine` behind the boundary (once: a Boundary is returned as it is)."""
    if engine is None or type(engine) is Boundary:
        return engine
    return Boundary(engine, kind)


class GuardedEngines(dict):  # type: ignore[type-arg]
    """A router's engines by kind, each put behind the boundary as it is added."""

    def __init__(self, engines: Any = (), /) -> None:
        super().__init__()
        self.update(engines)

    def __setitem__(self, kind: Any, engine: Any) -> None:
        super().__setitem__(kind, guard(kind, engine))

    def update(self, *args: Any, **kwargs: Any) -> None:
        for kind, engine in dict(*args, **kwargs).items():
            self[kind] = engine

    def setdefault(self, kind: Any, engine: Any = None) -> Any:
        if kind not in self:
            self[kind] = engine
        return self[kind]

    def __reduce__(self) -> tuple[Any, ...]:
        return (GuardedEngines, (dict(self),))
