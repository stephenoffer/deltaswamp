"""deltaswamp -- one Python connector for every Delta table.

Reads and writes Unity Catalog managed and external tables, Hive Metastore and
Glue tables, and plain object-store paths through a single object, routing each
operation to whichever engine can actually serve it -- or refusing with a reason
that names the blocker and the remedy.
"""

from __future__ import annotations

from .capability import Capability, Engine, Operation, TableFeature
from .credentials import Cloud, CredentialProvider, Credentials
from .distributed import ScanPlan, WritePlan
from .errors import (
    BackfillRequiredError,
    CommitConflictError,
    CorruptTableError,
    CredentialError,
    CredentialExpiryWarning,
    DeltaSwampError,
    EngineFallbackWarning,
    EnginePanicError,
    FallbackRequiredError,
    IgnoredPropertyWarning,
    InvalidReferenceError,
    PreflightError,
    PropertyNotSupportedError,
    TransientCommitError,
    UnreachableTableError,
)
from .identity import RefKind, TableRef, parse_ref
from .predicate import PredicateError
from .router import Router
from .table import Connection, Table, connect

__version__ = "0.1.0"

# The delta_kernel release the native extension is built against. Asserted
# against the compiled module by `check_native()`, so a kernel bump cannot slip
# through silently -- it breaks loudly at import instead of at runtime.
EXPECTED_KERNEL_VERSION = "0.28.0"

__all__ = [
    "EXPECTED_KERNEL_VERSION",
    "BackfillRequiredError",
    "Capability",
    "Cloud",
    "CommitConflictError",
    "Connection",
    "CorruptTableError",
    "CredentialError",
    "CredentialExpiryWarning",
    "CredentialProvider",
    "Credentials",
    "DeltaSwampError",
    "Engine",
    "EngineFallbackWarning",
    "EnginePanicError",
    "FallbackRequiredError",
    "IgnoredPropertyWarning",
    "InvalidReferenceError",
    "Operation",
    "PredicateError",
    "PreflightError",
    "PropertyNotSupportedError",
    "RefKind",
    "Router",
    "ScanPlan",
    "Table",
    "TableFeature",
    "TableRef",
    "TransientCommitError",
    "UnreachableTableError",
    "WritePlan",
    "__version__",
    "check_native",
    "connect",
    "has_native",
    "parse_ref",
]


def has_native() -> bool:
    """True if the compiled kernel extension is importable.

    Pure-Python paths (reference parsing, the conformance matrix, routing
    decisions) work without it, which keeps the test suite fast.
    """
    try:
        import deltaswamp._native  # noqa: F401
    except ImportError:
        return False
    return True


def check_native() -> None:
    """Verify the native extension matches this release's expectations.

    Checks the kernel pin and the multi-threaded-runtime invariant that
    `UCCommitter` depends on -- it calls `block_in_place`, which panics outright
    on a current-thread runtime.
    """
    from . import _native

    if _native.kernel_version() != EXPECTED_KERNEL_VERSION:
        raise RuntimeError(
            f"deltaswamp {__version__} expects delta_kernel "
            f"{EXPECTED_KERNEL_VERSION} but the native extension was built "
            f"against {_native.kernel_version()}. Rebuild the extension."
        )
    if not _native.runtime_is_multithreaded():
        raise RuntimeError(
            "the native extension's Tokio runtime is not multi-threaded; "
            "UCCommitter requires block_in_place and will panic without it"
        )
