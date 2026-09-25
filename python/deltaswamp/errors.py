"""Exceptions and warnings. Every refusal names the blocker and, where one
exists, the remedy."""

from __future__ import annotations

#: The remedy for anything only a Databricks SQL warehouse can serve.
SQL_FALLBACK_REMEDY = "ds.connect(..., allow_sql_fallback=True)"


class DeltaSwampError(Exception):
    """Base for every error raised by this library."""


class InvalidReferenceError(DeltaSwampError):
    """A table reference could not be parsed or resolved."""


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


class FallbackRequiredError(UnreachableTableError):
    """The operation is only servable via the opt-in SQL fallback, which is off."""


class PropertyNotSupportedError(UnreachableTableError):
    """A table property the chosen engine cannot handle."""


class EnginePanicError(DeltaSwampError):
    """An engine panicked across the FFI boundary.

    `pyo3_runtime.PanicException` derives from BaseException and escapes
    `except Exception`, so panics are converted to this.
    """


class CredentialExpiryWarning(UserWarning):
    """A read began with a vended credential that is close to expiring.

    Credentials are re-vended between operations, not during one, so a scan
    that outlives its credential fails partway through with a 403.
    """


class EngineFallbackWarning(UserWarning):
    """An engine failed on a read it claimed, and the next engine is serving it."""


class SqlFallbackWarning(UserWarning):
    """An operation was served by a Databricks SQL warehouse rather than directly."""


class IgnoredPropertyWarning(UserWarning):
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


class BackfillRequiredError(DeltaSwampError):
    """The catalog is refusing commits until unbackfilled commits are published.

    This is the HTTP 429 from the UC commit API. It is not a rate limit:
    retrying with backoff instead of publishing wedges the table.
    """


class TransientCommitError(DeltaSwampError):
    """A commit failed for a transient reason; retrying the same commit is safe."""


class CorruptTableError(DeltaSwampError):
    """On-disk state failed a correctness check (e.g. DV cardinality mismatch)."""
