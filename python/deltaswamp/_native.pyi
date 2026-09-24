"""Type stubs for the compiled Rust extension.

Kept in the source tree so mypy and editors work whether or not the extension
has been built. The implementation lives in `crates/native/src/`.
"""

from typing import Any

KERNEL_VERSION: str

def kernel_version() -> str:
    """The delta_kernel release this extension was compiled against."""

def native_version() -> str:
    """The deltaswamp-native crate version."""

def runtime_is_multithreaded() -> bool:
    """True if the shared Tokio runtime supports `block_in_place`.

    `UCCommitter` requires it and panics on a current-thread runtime.
    """

class CommitConflictError(RuntimeError):
    """Another writer committed this version first.

    Re-read the snapshot, recompute, and stage a *new* commit. Never reuse a
    staged file: it encodes a version-specific txnId.
    """

class BackfillRequiredError(RuntimeError):
    """The catalog wants staged commits published before accepting more.

    Backpressure, not a rate limit -- retrying with backoff instead of
    publishing will wedge the table.
    """

class RetryableError(RuntimeError):
    """A transient failure; the table is unchanged and a retry is safe."""

class UcCommitConfig:
    """How to reach Unity Catalog to have a commit ratified."""

    def __init__(
        self,
        workspace_url: str,
        token: str,
        table_id: str,
        catalog: str,
        schema: str,
        table: str,
    ) -> None: ...

def create_table(
    table_root: str,
    schema: Any,
    options: dict[str, str] | None = None,
    properties: dict[str, str] | None = None,
    partition_by: list[str] | None = None,
    cluster_by: list[str] | None = None,
    uc: UcCommitConfig | None = None,
    engine_info: str | None = None,
) -> int:
    """Create a Delta table and commit version 0; returns the version.

    Accepts nearly the whole Delta property surface, including `delta.feature.*`
    signals, row tracking, in-commit timestamps and custom non-`delta.` keys,
    all of which delta-rs rejects. Clustering is not a property: pass
    `cluster_by`, which sets it through the kernel's data layout.

    `partition_by` and `cluster_by` are mutually exclusive.
    """

class Snapshot:
    """A resolved Delta table snapshot, backed by delta-kernel-rs."""

    @staticmethod
    def resolve(
        table_root: str,
        options: dict[str, str] | None = None,
        version: int | None = None,
        log_tail: list[tuple[int, str, int, int]] | None = None,
        max_catalog_version: int | None = None,
    ) -> Snapshot:
        """Resolve a snapshot.

        `log_tail` entries are `(version, filename, last_modified_millis, size)`
        and `max_catalog_version` caps the version that may be trusted. Together
        they are what make a `catalogManaged` table readable; omit both for an
        ordinary path-based table.
        """

    @property
    def version(self) -> int: ...
    @property
    def table_root(self) -> str: ...
    @property
    def is_catalog_managed(self) -> bool: ...
    def schema(self) -> Any:
        """The logical Arrow schema (an arro3 Schema, PyCapsule-exporting)."""

    def protocol(self) -> tuple[int, int, list[str], list[str]]:
        """`(min_reader_version, min_writer_version, reader_features, writer_features)`."""

    @property
    def metadata_id(self) -> str:
        """The table id from the Metadata action, for staleness checks."""

    @property
    def partition_columns(self) -> list[str]:
        """Logical partition columns; empty for an unpartitioned table."""

    def table_properties(self) -> dict[str, str]: ...
    def scan(self, columns: list[str] | None = None) -> Any:
        """Read the table as an Arrow stream, with deletion vectors applied."""

    def append(
        self,
        data: Any,
        uc: UcCommitConfig | None = None,
        engine_info: str | None = None,
        operation: str | None = None,
    ) -> int:
        """Append Arrow data as one transaction; returns the committed version.

        Pass `uc` for a catalog-managed table, where the commit is staged and
        then ratified by the catalog rather than written directly.
        """

    def publish(self, uc: UcCommitConfig | None = None) -> int:
        """Publish ratified-but-unpublished commits into `_delta_log/`."""
