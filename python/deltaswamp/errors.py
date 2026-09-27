"""Exceptions and warnings. Every refusal names the blocker and, where one
exists, the remedy."""

from __future__ import annotations

#: The remedy for anything only a Databricks SQL warehouse can serve.
SQL_FALLBACK_REMEDY = "ds.connect(..., allow_sql_fallback=True)"


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

    An engine's builtin error stays an instance of its builtin class as well
    (a raw `OSError` becomes an `EngineError` that is also an `OSError`), so
    code catching the builtin keeps working.
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
        # A class made by `engine_error` is not importable by name, and the
        # original may not pickle at all (a Rust exception): rebuild from the
        # builtin base's name and the message, as raised on a Ray worker.
        builtin = next((c for c in type(self).__mro__ if c.__module__ == "builtins"), Exception)
        return (
            _rebuild_engine_error,
            (builtin.__name__, self.args[0] if self.args else "", self.engine, self.operation),
        )


_ENGINE_ERROR_CLASSES: dict[type[BaseException], type[EngineError]] = {}


def engine_error(
    message: str, *, engine: str, operation: str, original: BaseException
) -> EngineError:
    """An EngineError that is also an instance of `original`'s nearest builtin class."""
    builtin = next((c for c in type(original).__mro__ if c.__module__ == "builtins"), Exception)
    if builtin in (Exception, BaseException) or not issubclass(builtin, Exception):
        cls: type[EngineError] = EngineError
    else:
        cls = _ENGINE_ERROR_CLASSES.get(builtin) or _engine_error_class(builtin)
    error = cls(message, engine=engine, operation=operation, original=original)
    if isinstance(original, OSError) and isinstance(error, OSError):
        error.errno = original.errno
    return error


def _engine_error_class(builtin: type[BaseException]) -> type[EngineError]:
    try:
        cls = type(f"EngineError[{builtin.__name__}]", (EngineError, builtin), {})
    except TypeError:  # a builtin whose layout cannot be combined
        cls = EngineError
    cls.__module__ = __name__
    _ENGINE_ERROR_CLASSES[builtin] = cls
    return cls


def _rebuild_engine_error(
    builtin: str, message: str, engine: str | None, operation: str | None
) -> EngineError:
    import builtins

    base = getattr(builtins, builtin, Exception)
    if not (isinstance(base, type) and issubclass(base, Exception)) or base is Exception:
        return EngineError(message, engine=engine, operation=operation)
    cls = _ENGINE_ERROR_CLASSES.get(base) or _engine_error_class(base)
    return cls(message, engine=engine, operation=operation)


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


class PreflightError(DeltaSwampError):
    """A required workspace/metastore prerequisite is not satisfied."""


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

    An engine's raw OSError, which `except DeltaSwampError` did not catch. An
    OSError too, so code that catches the builtin keeps working.
    """


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
