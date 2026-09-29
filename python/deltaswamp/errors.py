"""Exceptions and warnings. Every refusal names the blocker and, where one
exists, the remedy."""

from __future__ import annotations

from typing import Any, TypeVar

#: The remedy for anything only a Databricks SQL warehouse can serve.
SQL_FALLBACK_REMEDY = "ds.connect(..., allow_sql_fallback=True)"


_E = TypeVar("_E", bound=BaseException)


class DeltaSwampError(Exception):
    """Base for every error raised by this library."""


class InvalidReferenceError(DeltaSwampError):
    """A table reference could not be parsed or resolved."""


class TableNotFoundError(InvalidReferenceError):
    """The catalog has no table by that name, or does not let this principal see it.

    A subclass of InvalidReferenceError, which it used to be raised as, so
    existing handlers keep working.
    """


class InvalidArgumentError(DeltaSwampError, ValueError):
    """A call's arguments are malformed or contradict each other.

    A ValueError too, so code that catches the builtin keeps working.
    """


class UnreachableTableError(DeltaSwampError):
    """The table exists but no available engine can serve the request."""

    def __init__(self, operation: str, reason: str, remedy: str | None = None) -> None:
        self.operation = operation
        self.reason = reason
        self.remedy = remedy
        msg = f"cannot {operation}: {reason}"
        if remedy:
            msg += f"\n  remedy: {remedy}"
        super().__init__(msg)

    def __reduce__(self) -> tuple[object, ...]:
        # The default rebuilds from `args` (the one formatted message), which
        # this __init__ cannot take: an error raised in a Ray worker or a
        # multiprocessing child would fail to unpickle on the driver and be
        # replaced by an unrelated TypeError.
        return (type(self), (self.operation, self.reason, self.remedy), self.__dict__)


class FallbackRequiredError(UnreachableTableError):
    """The operation is only servable via the opt-in SQL fallback, which is off."""


class PropertyNotSupportedError(UnreachableTableError):
    """A table property the chosen engine cannot handle."""


class EngineLimitError(UnreachableTableError):
    """The engine serving a call found, only once it had it in hand, that it cannot.

    A limit of that engine, not of the request -- the kernel reads a change feed
    across one schema only; delta-rs mishandles a conditional NOT MATCHED clause
    on a change-feed table, which a MERGE shows only at execute -- so another
    engine that serves the operation may succeed, and `Table` tries it. It is
    raised before anything is written. A refusal about the request itself (no
    such version, a timestamp before the history) is a plain
    `UnreachableTableError`.
    """


class EnginePanicError(DeltaSwampError):
    """An engine panicked across the FFI boundary.

    `pyo3_runtime.PanicException` derives from BaseException and escapes
    `except Exception`, so panics are converted to this.
    """


class EngineError(DeltaSwampError):
    """An engine failed in a way no more specific error of this library describes.

    Every exception an engine raises is translated at one boundary (see
    `deltaswamp.engine.boundary`); what matches no rule arrives as this rather
    than as the engine's own type, so `except DeltaSwampError` catches every
    failure. `engine` names the engine, `operation` the call it failed in,
    and `original` is the engine's exception (also the `__cause__`).

    The error stays an instance of the original's class as well, so code
    catching it keeps working: a raw `OSError` becomes an `EngineError` that
    is also an `OSError`, a `pyarrow.ArrowInvalid` one that is also an
    `ArrowInvalid` (see `combined` for which classes qualify).
    """

    def __init__(
        self,
        message: str,
        *,
        engine: str | None = None,
        operation: str | None = None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.engine = engine
        self.operation = operation
        self.original = original

    def __reduce__(self) -> tuple[object, ...]:
        # A class made by `combined` is not importable by name, and the
        # original may not pickle at all (a Rust exception): rebuild from the
        # class it was combined with and the message, as raised on a Ray
        # worker, with the rest of the state (errno, filename) kept.
        return _reduced(
            self, EngineError, {"engine": self.engine, "operation": self.operation}, ("original",)
        )


#: Modules whose exception classes are never combined with this library's:
#: Rust extensions (delta-rs's `_internal`, this library's `_native`, pyo3's
#: panics, Polars), whose types the translation exists to replace.
_FOREIGN_MODULES = ("_internal", "deltalake", "deltaswamp", "pyo3_runtime", "polars", "daft")

#: Py_TPFLAGS_BASETYPE: the class may be subclassed.
_BASETYPE = 1 << 10

_COMBINED: dict[tuple[type[BaseException], type[BaseException]], type[BaseException] | None] = {}


def _combinable(cls: type) -> bool:
    """Whether `cls` may be a base of a translated error, beside this library's class."""
    if cls in (Exception, BaseException) or not issubclass(cls, Exception):
        return False
    if issubclass(cls, DeltaSwampError) or not cls.__flags__ & _BASETYPE:
        return False
    module = cls.__module__ or ""
    return module.split(".")[0] not in _FOREIGN_MODULES


def _combined_class(base: type[_E], other: type[BaseException]) -> type[_E] | None:
    key = (base, other)
    if key not in _COMBINED:
        try:
            # str() is the translated message: KeyError's own __str__ quoted it.
            cls: type[BaseException] | None = type(
                f"{base.__name__}[{other.__name__}]",
                (base, other),
                {"__str__": BaseException.__str__, "_combined_with": other},
            )
            cls.__module__ = __name__
        except TypeError:  # bases whose layouts cannot be combined
            cls = None
        _COMBINED[key] = cls
    return _COMBINED[key]  # type: ignore[return-value]


def combined(
    base: type[_E],
    like: BaseException | type[BaseException] | None,
    /,
    *args: Any,
    **kwargs: Any,
) -> _E:
    """`base(*args, **kwargs)`, made an instance of `like`'s class too where it can be.

    The rule: `like`'s own class if it qualifies, else the nearest
    ancestor that does. A class qualifies when it is an Exception below
    `Exception` itself, can be subclassed, is not a Rust extension's type (see
    `_FOREIGN_MODULES`) nor this library's, and the combined class can be
    built from the message alone -- UnicodeDecodeError or ExceptionGroup
    cannot, and an ancestor (ValueError) is used instead. With none, the
    error is a plain `base`.
    """
    cls = like if isinstance(like, type) else type(like)
    if like is not None:
        for candidate in cls.__mro__:
            if issubclass(base, candidate):
                break  # `base` is one already (StorageError is an OSError)
            if not _combinable(candidate):
                continue
            made = _combined_class(base, candidate)
            if made is None:
                continue
            try:
                return made(*args, **kwargs)
            except Exception:  # a constructor that needs more than a message
                continue
    return base(*args, **kwargs)


def _reduced(
    error: BaseException, base: type, kwargs: dict[str, Any], dropped: tuple[str, ...] = ()
) -> tuple[object, ...]:
    other = getattr(type(error), "_combined_with", None)
    ref = (other.__module__, other.__qualname__) if other is not None else None
    state = {k: v for k, v in vars(error).items() if k not in dropped}
    if isinstance(error, OSError):
        # Held in slots, not the instance dict: pickling lost them.
        state.update(
            (name, getattr(error, name))
            for name in ("errno", "strerror", "filename")
            if getattr(error, name) is not None
        )
    return (_rebuild_error, (base, ref, error.args[0] if error.args else "", kwargs), state)


def _rebuild_error(
    base: type[_E], ref: tuple[str, str] | None, message: str, kwargs: dict[str, Any]
) -> _E:
    other: Any = None
    if ref is not None:
        import importlib

        try:
            other = importlib.import_module(ref[0])
            for part in ref[1].split("."):
                other = getattr(other, part)
        except Exception:  # not importable where it is unpickled
            other = None
    if not (isinstance(other, type) and issubclass(other, BaseException)):
        other = None
    return combined(base, other, message, **kwargs)


def engine_error(
    message: str, *, engine: str, operation: str, original: BaseException
) -> EngineError:
    """An EngineError that is also an instance of `original`'s class, where it can be."""
    error = combined(
        EngineError, original, message, engine=engine, operation=operation, original=original
    )
    if isinstance(original, OSError) and isinstance(error, OSError):
        error.errno = original.errno
        if original.filename is not None:
            error.filename = original.filename
    return error


def _rebuild_engine_error(
    builtin: str, message: str, engine: str | None, operation: str | None
) -> EngineError:
    # Pickles made before `_reduced`; kept so they still load.
    import builtins

    base = getattr(builtins, builtin, None)
    other = base if isinstance(base, type) and issubclass(base, BaseException) else None
    return combined(EngineError, other, message, engine=engine, operation=operation)


class CommitRefusedError(EngineError):
    """The engine refused a commit for a reason other than a lost race.

    delta-rs's CommitFailedError that is not a conflict: a remove on an
    append-only table, a protocol it will not write, a failure writing the
    commit file. Retrying the same commit meets the same refusal. An
    EngineError, which it used to arrive as.
    """


class DeltaSwampWarning(UserWarning):
    """Base class of every warning this library emits.

    `warnings.filterwarnings("ignore", category=ds.DeltaSwampWarning)` silences
    them all; each subclass can still be filtered on its own.
    """


class CredentialExpiryWarning(DeltaSwampWarning):
    """A read began with a vended credential that is close to expiring.

    Credentials are re-vended between operations, not during one, so a scan
    that outlives its credential fails partway through with a 403.
    """


class EngineFallbackWarning(DeltaSwampWarning):
    """An engine failed on a read it claimed, and the next engine is serving it."""


class SqlFallbackWarning(DeltaSwampWarning):
    """An operation was served by a Databricks SQL warehouse rather than directly."""


class IgnoredPropertyWarning(DeltaSwampWarning):
    """A property will be stored but nothing here acts on it."""


class CredentialError(DeltaSwampError):
    """Credential vending or refresh failed."""


class ExternalWriteNotAllowedError(CredentialError):
    """Unity Catalog vends no write credentials for this table.

    Databricks' ``EXTERNAL_WRITE_NOT_ALLOWED_FOR_TABLE``: a managed table
    accepts writes from outside Databricks only when it uses catalog commits
    (``catalogManaged``). Raised at planning, before any worker runs.
    """


class PreflightError(DeltaSwampError):
    """A required workspace/metastore prerequisite is not satisfied.

    `denied` is True when the catalog refused the principal (its credentials
    rejected, or a privilege missing), as opposed to failing to answer.
    """

    def __init__(self, *args: object, denied: bool = False) -> None:
        super().__init__(*args)
        self.denied = denied


class CommitConflictError(DeltaSwampError):
    """A commit lost its race and must be rebuilt against a fresh snapshot."""

    def __init__(self, version: int, message: str) -> None:
        self.version = version
        super().__init__(message)

    def __reduce__(self) -> tuple[object, ...]:
        # See UnreachableTableError.__reduce__.
        return (type(self), (self.version, str(self)), self.__dict__)


class MetadataChangedError(CommitConflictError):
    """A concurrent commit changed the schema, partitioning or column mapping.

    Delta's MetadataChangedException. Data files written against the old
    layout cannot be committed into the new one -- the table would stop
    reading, or a column's values would land in a column that no longer
    holds them -- so, unlike a plain conflict, retrying the same commit can
    never succeed: re-plan and write the data again.
    """


class ChangeFeedSchemaChangeError(UnreachableTableError):
    """The change feed spans a schema change it cannot read across.

    Spark's DELTA_CHANGE_DATA_FEED_INCOMPATIBLE_SCHEMA_CHANGE. `version` is
    the commit that changed the schema (None if it could not be found), so a
    streaming consumer can restart the feed after it.
    """

    version: int | None = None


class BackfillRequiredError(DeltaSwampError):
    """The catalog is refusing commits until unbackfilled commits are published.

    This is the HTTP 429 from the UC commit API. It is not a rate limit:
    retrying with backoff instead of publishing wedges the table.
    """


class TransientCommitError(DeltaSwampError):
    """A commit failed for a transient reason; retrying the same commit is safe."""


class StorageError(DeltaSwampError, OSError):
    """The table's storage failed a request: unreachable, throttling, no such bucket.

    An engine's raw OSError, which `except DeltaSwampError` did not catch. It
    stays an instance of that error's class (a `TimeoutError` or a
    `ConnectionResetError` too, as `combined` builds it), so code catching the
    builtin, and retry policies keyed on it, keep working.
    """

    # The message, whatever errno or filename are set: OSError's own __str__
    # printed "[Errno None] None: None" once filename was.
    __str__ = BaseException.__str__

    def __reduce__(self) -> tuple[object, ...]:
        return _reduced(self, StorageError, {}, ("original",))


class CorruptTableError(DeltaSwampError):
    """On-disk state failed a correctness check (e.g. DV cardinality mismatch)."""


class MissingDataFileError(CorruptTableError):
    """A data or deletion-vector file the snapshot references is gone from storage.

    `path` is the missing object, as storage reported it.
    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path

    def __reduce__(self) -> tuple[type[MissingDataFileError], tuple[str, str]]:
        # Raised on Ray workers too: the default reduce re-calls __init__ with
        # the message alone, which fails to unpickle.
        return (type(self), (self.path, str(self)))
