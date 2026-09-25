"""Exception hierarchy.

Design rule: every refusal names the blocker and, where one exists, the remedy.
An opaque "unsupported table feature" error is the thing this library exists to
eliminate, so we never raise one.
"""

from __future__ import annotations


class DeltaSwampError(Exception):
    """Base for every error raised by this library."""


class InvalidReferenceError(DeltaSwampError):
    """A table reference could not be parsed or resolved."""


class UnreachableTableError(DeltaSwampError):
    """The table exists but no available engine can serve the request.

    Carries the reason and, when one exists, the remedy -- so the message is
    actionable rather than merely negative.
    """

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
    """A table property the chosen engine cannot handle.

    Raised instead of letting delta-rs emit its single opaque message for half
    the Delta spec ("Error parsing property"), or panic outright.
    """


class EnginePanicError(DeltaSwampError):
    """An engine panicked across the FFI boundary.

    A Rust panic surfaces in Python as `pyo3_runtime.PanicException`, which
    derives from BaseException and so escapes `except Exception`. We convert it
    so callers can actually handle it.
    """


class CredentialExpiryWarning(UserWarning):
    """A read began with a vended credential that is close to expiring.

    Re-vending happens *between* operations: the kernel builds its object store
    once per snapshot, so a single scan that streams past the credential's
    lifetime fails partway through with an opaque 403 from the storage layer.
    Saying so up front turns that into something actionable.
    """


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

    This is the HTTP 429 from the UC commit API. It is NOT a rate limit --
    retrying with backoff instead of publishing will wedge the table.
    """


class TransientCommitError(DeltaSwampError):
    """A commit failed for a transient reason; the table is unchanged.

    Distinct from a conflict: nobody else won the version, so the very same
    transaction can simply be tried again.
    """


class CorruptTableError(DeltaSwampError):
    """On-disk state failed a correctness check (e.g. DV cardinality mismatch)."""
