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
import contextvars
import inspect
import os
import re
from collections.abc import Callable, Iterator
from typing import Any

from ..capability import Engine as EngineKind
from ..errors import (
    BackfillRequiredError,
    CommitConflictError,
    CommitRefusedError,
    CorruptTableError,
    CredentialError,
    DeltaSwampError,
    EngineLimitError,
    EnginePanicError,
    InvalidArgumentError,
    InvalidReferenceError,
    MetadataChangedError,
    PreflightError,
    StorageError,
    TableNotFoundError,
    TransientCommitError,
    UnreachableTableError,
    combined,
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


#: The Parquet readers' words for a data file whose bytes are not what the log
#: says: cut short (a footer read past its end) or replaced by another file.
_DAMAGED_FILE = re.compile(
    r"Invalid Parquet file|Corrupt footer|Parquet magic bytes not found|"
    r"Requested range was invalid|out of specified range"
)


def _damaged_file(exc: BaseException, what: str) -> BaseException | None:
    """A truncated or replaced data file, as CorruptTableError.

    It arrived as StorageError ("storage failed the request"), which blamed an
    outage and qualified the read for a fallback that reads the same bytes.
    """
    if not isinstance(exc, Exception) or not _DAMAGED_FILE.search(str(exc)):
        return None
    # Still an instance of the reader's class (an OSError, an ArrowInvalid).
    return combined(
        CorruptTableError,
        exc,
        f"{what}: a data file of the table is damaged: it is shorter than the Delta log "
        f"records, or another file replaced it ({_detail(exc)})",
    )


def _storage(exc: BaseException, what: str) -> BaseException | None:
    """A storage failure (unreachable, throttled, no such bucket) as StorageError.

    Not a missing file or a refused permission: those keep their builtin class
    (FileNotFoundError, PermissionError) inside the generic EngineError, and
    reads name the missing file as MissingDataFileError.
    """
    if not isinstance(exc, OSError) or isinstance(exc, (FileNotFoundError, PermissionError)):
        return None
    # Still a TimeoutError or a ConnectionResetError: `except TimeoutError`,
    # and retry policies keyed on ConnectionError, stopped matching.
    error = combined(
        StorageError, exc, f"{what}: the table's storage failed the request ({_detail(exc)})"
    )
    error.errno = exc.errno
    if exc.filename is not None:
        error.filename = exc.filename
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
    r"concurrent|changed since last commit|existing table version|version \d+ already exists|"
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
    # The kernel's class for the same race: a caller catching
    # MetadataChangedError to re-plan caught it on one engine only.
    kind = (
        MetadataChangedError
        if re.search(r"metadata changed since last commit", message, re.IGNORECASE)
        else CommitConflictError
    )
    return kind(
        int(found.group(1)) if found else -1,
        f"another writer committed first: {message}. Re-read the table and retry",
    )


def _deltars_conflict(exc: BaseException, what: str) -> BaseException | None:
    return as_commit_conflict(exc)


#: delta-rs's words for a table protocol it cannot read or write: a limit of
#: delta-rs, which another engine may not share.
_DELTARS_PROTOCOL = re.compile(
    r"reader features|writer features|features? (?:is |are )?required|"
    r"unsupported (?:reader|writer) ?(?:feature|version)|protocol",
    re.IGNORECASE,
)


def _deltars_data(exc: BaseException, what: str) -> BaseException | None:
    """Data delta-rs cannot take at all: not Arrow, pandas or anything exporting Arrow."""
    if not isinstance(exc, ValueError) or "Expected object with __arrow_c_" not in str(exc):
        return None
    return InvalidArgumentError(
        f"cannot {what}: the data is not a table delta-rs can read ({_first_line(exc)}). Pass "
        "Arrow, pandas or Polars data; wrap an iterator of record batches as "
        "pyarrow.RecordBatchReader.from_batches(schema, batches)"
    )


def _deltars_typed(exc: BaseException, what: str) -> BaseException | None:
    """delta-rs's own exception types, as this library's.

    They derive from Exception alone, so they arrived as a bare EngineError
    that not even an `except` on their category caught.
    """
    name, first = type(exc).__name__, _first_line(exc)
    if name == "TableNotFoundError":
        return TableNotFoundError(f"cannot {what}: there is no Delta table there ({first})")
    if name == "DeltaProtocolError":
        if "Invariant violations" in str(exc):
            return InvalidArgumentError(
                f"the data violates the table's invariants, so nothing was written: {first}"
            )
        # A protocol or a log delta-rs cannot handle (features it does not
        # implement, statistics it cannot parse): the kernel may read it, so a
        # read moves on to the next engine.
        return EngineLimitError(
            f"{what} with delta-rs", f"delta-rs cannot handle the table's protocol or log ({first})"
        )
    if name == "CommitFailedError":
        if _DELTARS_PROTOCOL.search(str(exc)):
            return EngineLimitError(
                f"{what} with delta-rs",
                f"delta-rs cannot commit to this table's protocol ({first})",
            )
        return CommitRefusedError(
            f"delta-rs refused the commit {_during(what)}: {first}",
            engine=EngineKind.DELTARS.value,
            operation=what,
            original=exc,
        )
    return None


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
    re.compile(r"No parquet file is found in the given location"),
)

#: The same for names and types, which only a write or DML is sure to mean as
#: the caller's mistake: on a read the table's own schema raises them where
#: delta-rs cannot follow it, and the next engine may read it.
_DELTARS_BAD_WRITE_INPUT = (
    re.compile(r"No field with the provided name in the schema"),
    re.compile(r"Schema error: No field named"),
)

#: Leading words of the `what` of delta-rs's reads (method names, and the
#: translating() sites that open the table or walk its log).
_DELTARS_READS = frozenset(
    {
        "scan",
        "history",
        "detail",
        "cdf",
        "load",
        "cleaned",
        "files",
        "plan_scan",
        "execute_scan",
        "open",
        "actions",
        "entry",
        "walk",
        "count",
        "metadata_count",
    }
)


def _is_read(what: str) -> bool:
    return (what.split() or [""])[0] in _DELTARS_READS


#: A type in the table's schema delta-rs does not implement: its limit, not the
#: caller's input ("Unsupported Delta table type: 'interval'").
_DELTARS_UNSUPPORTED_TYPE = re.compile(r"Unsupported (?:Delta )?(?:table |data )?type", re.I)

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
    if _DELTARS_UNSUPPORTED_TYPE.search(message):
        return EngineLimitError(
            f"{what} with delta-rs", f"delta-rs does not implement a type of the table ({first})"
        )
    bad_input = (
        _DELTARS_BAD_INPUT if _is_read(what) else (*_DELTARS_BAD_INPUT, *_DELTARS_BAD_WRITE_INPUT)
    )
    if any(pattern.search(message) for pattern in bad_input):
        # All of it: the first line dropped delta-rs's "Valid fields are ...".
        return InvalidArgumentError(f"cannot {what}: {_detail(exc)}")
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
    if _is_read(what):
        # A read has no data of the caller's to not fit: this is delta-rs
        # failing on the table, which the next engine may read.
        return None
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
    EngineKind.KERNEL: (*_NATIVE, _panic(EngineKind.KERNEL), _damaged_file, _storage),
    EngineKind.DELTARS: (
        _deltars_conflict,
        _deltars_fork,
        _panic(EngineKind.DELTARS),
        _deltars_constraint,
        _damaged_file,
        _deltars_sql,
        _deltars_schema,
        _deltars_typed,
        _deltars_data,
        *_NATIVE,
        _storage,
    ),
    EngineKind.ICEBERG: (
        _iceberg_schema,
        *_NATIVE,
        _panic(EngineKind.ICEBERG),
        _damaged_file,
        _storage,
    ),
    EngineKind.SQL: (*_NATIVE, _panic(EngineKind.SQL), _storage),
    EngineKind.SHARING: (*_NATIVE, _panic(EngineKind.SHARING), _damaged_file, _storage),
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


#: The caller's data sources of the boundary call in progress (see `_Sources`).
_CALL_SOURCES: contextvars.ContextVar[_Sources | None] = contextvars.ContextVar(
    "deltaswamp_call_sources", default=None
)


#: What the streams handed to native code inside the `translating()` block in
#: progress raised (see `recorded`).
_STREAM_ERRORS: contextvars.ContextVar[list[BaseException] | None] = contextvars.ContextVar(
    "deltaswamp_stream_errors", default=None
)


def recorded(reader: Any) -> Any:
    """`reader`, read through a wrapper that keeps what it raises for `translating()`.

    Native code pulls a Python stream through Arrow's C interface, which
    carries only an error's text: Ctrl-C inside an OPTIMIZE's batches, or a
    caller's generator raising, came back as "C Data interface error ...
    KeyboardInterrupt", translated to InvalidArgumentError, so an
    `except Exception` retry loop swallowed the interrupt. Recorded here, the
    exception itself is raised again. Outside a `translating()` block the
    reader is returned as it is.
    """
    errors = _STREAM_ERRORS.get()
    if errors is None:
        return reader
    import pyarrow as pa

    if not isinstance(reader, pa.RecordBatchReader):
        return reader  # another Arrow stream is handed on untouched

    def batches() -> Iterator[Any]:
        try:
            yield from reader
        except BaseException as exc:
            errors.append(exc)
            raise

    return pa.RecordBatchReader.from_batches(reader.schema, batches())


@contextlib.contextmanager
def translating(kind: EngineKind, what: str) -> Iterator[None]:
    """Translate what the block raises, as the boundary would.

    For the few places inside an engine that must act on a typed error before
    the call returns -- a retry loop that re-reads the table on a lost commit
    race -- so they use the boundary's rules rather than a second set. What a
    `recorded` stream raised inside the block is raised as itself when it
    would pass the boundary (an interrupt, the library's own error) or is the
    caller's own.
    """
    errors: list[BaseException] = []
    token = _STREAM_ERRORS.set(errors)
    try:
        yield
    except BaseException as exc:
        for error in errors:
            if error is not exc and (_passes(error) or _callers_own(type(error))):
                raise error  # noqa: B904 - the engine's error stays as its context
        if _passes(exc):
            raise
        sources = _CALL_SOURCES.get()
        mine = sources.raised(exc) if sources is not None else None
        if mine is exc:
            raise
        if mine is not None:
            raise mine  # noqa: B904 - the engine's error stays as its context
        raise translate(kind, what, exc) from exc
    finally:
        _STREAM_ERRORS.reset(token)


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

    What the caller's own data raises is not the engine's failure, and passes
    through as it was raised (see `_Sources`).
    """

    __slots__ = (
        "_boundary_cache",
        "_boundary_kind",
        "_boundary_prefix",
        "_boundary_sources",
        "_boundary_target",
    )

    def __init__(
        self,
        target: Any,
        kind: EngineKind,
        prefix: str | None = None,
        sources: _Sources | None = None,
    ) -> None:
        object.__setattr__(self, "_boundary_target", target)
        object.__setattr__(self, "_boundary_kind", kind)
        object.__setattr__(self, "_boundary_prefix", prefix)
        object.__setattr__(self, "_boundary_sources", sources)
        object.__setattr__(self, "_boundary_cache", {})

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
        # One wrapper per method, while the method stays the same: a fresh
        # closure per access made `b.m is b.m` false, and a monkeypatch saved
        # one and restored it as an instance attribute of the engine.
        cached = self._boundary_cache.get(name)
        if cached is not None and _same_method(cached.__wrapped__, value):
            return cached
        guarded = self._guarded(name, value)
        self._boundary_cache[name] = guarded
        return guarded

    def __setattr__(self, name: str, value: Any) -> None:
        target = self._boundary_target
        if getattr(value, "_boundary_guarded", False):
            # One of this wrapper's own methods written back (a monkeypatch
            # undone): the engine gets the method, not the closure, which
            # does not pickle -- and where it is the class's own method, no
            # instance attribute at all.
            value = value.__wrapped__
            if getattr(value, "__self__", None) is target and _same_method(
                getattr(type(target), name, None), getattr(value, "__func__", None)
            ):
                with contextlib.suppress(AttributeError):
                    delattr(target, name)
                return
        setattr(target, name, value)

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
        # translated the same way. A method patched onto the engine instance
        # (a test's monkeypatch) stays behind: a lambda or a closure does not
        # pickle, and failed every plan of the connection.
        return (
            Boundary,
            (_without_patches(self._boundary_target), self._boundary_kind, self._boundary_prefix),
        )

    def _guarded(self, name: str, method: Any) -> Any:
        kind = self._boundary_kind
        what = f"{self._boundary_prefix} ({name})" if self._boundary_prefix else name
        inherited = self._boundary_sources

        def call(*args: Any, **kwargs: Any) -> Any:
            sources = _Sources(inherited)
            args, kwargs = sources.watch(args, kwargs)
            # Seen by the engine's own `translating()` sites too, which
            # translate before the call returns here.
            token = _CALL_SOURCES.set(sources)
            try:
                result = method(*args, **kwargs)
            except BaseException as exc:
                if _passes(exc):
                    raise
                mine = sources.raised(exc)
                if mine is exc:
                    raise
                if mine is not None:
                    raise mine  # noqa: B904 - the engine's error stays as its context
                raise translate(kind, what, exc) from exc
            finally:
                _CALL_SOURCES.reset(token)
            return self._wrapped(result, what, sources)

        call._boundary_guarded = True  # type: ignore[attr-defined]
        call.__name__ = getattr(method, "__name__", name)
        call.__doc__ = getattr(method, "__doc__", None)
        call.__wrapped__ = method  # type: ignore[attr-defined]
        return call

    def _wrapped(self, result: Any, what: str, sources: _Sources | None = None) -> Any:
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
            # The builder reads the MERGE source only at execute(): what the
            # source raises then is still the caller's.
            return Boundary(result, kind, prefix=what, sources=sources)
        return result


def _same_method(a: Any, b: Any) -> bool:
    """Whether `a` and `b` are one method (bound methods are made anew per access)."""
    if a is b:
        return True
    func = getattr(a, "__func__", None)
    return (
        func is not None
        and func is getattr(b, "__func__", None)
        and getattr(a, "__self__", None) is getattr(b, "__self__", None)
    )


def _without_patches(target: Any) -> Any:
    """`target`, or a copy without the callables patched onto the instance."""
    attrs = getattr(target, "__dict__", None)
    if not attrs:
        return target
    patched = [
        name
        for name, value in attrs.items()
        if callable(value) and not isinstance(value, type) and hasattr(type(target), name)
    ]
    if not patched:
        return target
    import copy

    clone = copy.copy(target)
    for name in patched:
        with contextlib.suppress(AttributeError, KeyError):
            del clone.__dict__[name]
    return clone


#: Top-level packages whose objects are not the caller's own data source.
_LIBRARY_PACKAGES = frozenset({"deltaswamp", "deltalake", "pyarrow", "builtins", "_internal"})


def _callers_own(cls: type) -> bool:
    """Whether `cls` is the caller's own: not this library's, an installed package's or Python's."""
    module = cls.__module__ or ""
    if module.split(".")[0] in _LIBRARY_PACKAGES:
        return False
    if module == "__main__":
        return True
    import sys
    import sysconfig

    path = getattr(sys.modules.get(module), "__file__", None)
    if not path:
        return False
    path = os.path.realpath(path)
    installed = {
        os.path.realpath(p)
        for key in ("stdlib", "platstdlib", "purelib", "platlib")
        if (p := sysconfig.get_paths().get(key))
    }
    return not any(path.startswith(root + os.sep) for root in installed)


class _Sources:
    """The caller's data sources handed to one engine call, and what they raised.

    The caller's exception from their own source -- a `__arrow_c_stream__`
    that raises, a reader over a generator that fails -- arrived as the
    engine's failure ("delta-rs failed to append"), so `except MyError`
    stopped matching and the engine was blamed. Two ways it is recognised:
    a traceback frame running a method of an object the caller passed in, and
    an Arrow reader passed in, which is read through a wrapper that keeps what
    it raised (delta-rs and the kernel read a stream through Arrow's C
    interface, which carries only the error's text).
    """

    __slots__ = ("errors", "objects", "parent")

    def __init__(self, parent: _Sources | None = None) -> None:
        self.parent = parent
        self.objects: list[Any] = []
        self.errors: list[BaseException] = []

    def watch(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        if not args and not kwargs:
            return args, kwargs
        new_args = tuple(self._watched(a) for a in args)
        new_kwargs = {k: self._watched(v) for k, v in kwargs.items()} if kwargs else kwargs
        return new_args, new_kwargs

    def _watched(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bytes, int, float, bool, dict, list, tuple)):
            return value
        cls = type(value)
        if cls.__name__ == "RecordBatchReader" and cls.__module__ == "pyarrow.lib":
            import pyarrow as pa

            return pa.RecordBatchReader.from_batches(value.schema, self._read(value))
        if _callers_own(cls):
            self.objects.append(value)
        return value

    def _read(self, reader: Any) -> Iterator[Any]:
        try:
            yield from reader
        except BaseException as exc:
            self.errors.append(exc)
            raise

    def raised(self, exc: BaseException) -> BaseException | None:
        """The caller's exception behind `exc`, if the caller's source raised it."""
        for error in self._errors():
            if (
                isinstance(error, DeltaSwampError)
                or _callers_own(type(error))
                or (not isinstance(error, Exception) and not _is_panic(error))
            ):
                return error
        objects = {id(o) for o in self._objects()}
        if objects:
            tb = exc.__traceback__
            while tb is not None:
                owner = tb.tb_frame.f_locals.get("self")
                if owner is not None and id(owner) in objects:
                    return exc
                tb = tb.tb_next
        return None

    def _errors(self) -> Iterator[BaseException]:
        node: _Sources | None = self
        while node is not None:
            yield from node.errors
            node = node.parent

    def _objects(self) -> Iterator[Any]:
        node: _Sources | None = self
        while node is not None:
            yield from node.objects
            node = node.parent


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

    # dict's own merge and copy build or fill a mapping without __setitem__:
    # `router.engines |= {kind: raw}` put a raw engine past the boundary.
    def __ior__(self, other: Any) -> GuardedEngines:
        self.update(other)
        return self

    def __or__(self, other: Any) -> GuardedEngines:
        merged = GuardedEngines(self)
        merged.update(other)
        return merged

    def __ror__(self, other: Any) -> GuardedEngines:
        merged = GuardedEngines(other)
        merged.update(self)
        return merged

    def copy(self) -> GuardedEngines:
        return GuardedEngines(self)

    def __reduce__(self) -> tuple[Any, ...]:
        return (GuardedEngines, (dict(self),))
