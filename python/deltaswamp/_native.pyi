"""Type stubs for the compiled Rust extension.

Kept in the source tree so mypy and editors work whether or not the extension
has been built. The implementation lives in `crates/native/src/`.
"""

from typing import Any

KERNEL_VERSION: str

FEATURES: list[str]
"""Capabilities this build provides, by stable name.

One of: "predicate_skipping", "timestamp_travel", "table_changes", "files",
"metadata_json", "commit_raw", "partitioned_append", "uc_create_table_request",
"checkpoint". Gate on this list, not `hasattr`, so a stale build refuses
cleanly.
"""

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

def table_changes(
    table_root: str,
    options: dict[str, str] | None = None,
    start_version: int | None = None,
    end_version: int | None = None,
    columns: list[str] | None = None,
    predicate: str | None = None,
    start_timestamp_ms: int | None = None,
    end_timestamp_ms: int | None = None,
) -> Any:
    """Read the change data feed as an Arrow stream (arro3 RecordBatchReader).

    Every row carries `_change_type` (insert / delete / update_preimage /
    update_postimage), `_commit_version` and `_commit_timestamp`; they are kept
    even when `columns` projects them away. Deletion vectors are applied by the
    kernel.

    The range is inclusive. `start_version` defaults to 0 and `end_version` to
    the latest version. `start_timestamp_ms` resolves to the first commit at or
    after it and `end_timestamp_ms` to the last commit at or before it; each is
    mutually exclusive with its version counterpart (ValueError). Timestamps are
    matched against published commits (in-commit timestamps when enabled).

    `predicate` is the same JSON AST as `Snapshot.scan` and only skips files.
    Path-based tables only (no catalog log tail). Raises ValueError if CDF was
    not enabled at the range's endpoints or the schema changed across it.
    """

def commit_raw(
    table_root: str,
    version: int,
    actions: list[str],
    options: dict[str, str] | None = None,
) -> int:
    """Write `actions` verbatim as `_delta_log/<version:020>.json`; returns `version`.

    Each action is one single-line JSON object with exactly one key naming the
    action (`{"commitInfo": {...}}`, `{"metaData": {...}}`, `{"protocol": {...}}`,
    `{"domainMetadata": {...}}`, ...); anything else is a ValueError and nothing
    is written. The caller computes `version` (normally snapshot.version + 1)
    and is responsible for the actions being valid against that snapshot.

    The write is an atomic put-if-absent (object_store `PutMode::Create`). If
    the version already exists, raises `CommitConflictError`: re-read,
    recompute, retry at the next version.

    Storage requirements: local, GCS and Azure support put-if-absent natively.
    S3 works with object_store 0.13's default `aws_conditional_put=etag`
    (sends `If-None-Match: *`, supported by AWS S3 since 2024, R2 and MinIO);
    do not set `aws_conditional_put=disabled` -- that makes this raise
    ValueError rather than risk overwriting a commit.

    Path-based tables only: a catalog-managed table's commits must be ratified
    by the catalog, and writing one here would fork its history.
    """

def uc_create_table_request(
    table_root: str,
    table_name: str,
    options: dict[str, str] | None = None,
) -> str:
    """The Unity Catalog `CreateTableRequest` body for version 0, as JSON.

    Resolves the snapshot at version 0 of `table_root` (which must have been
    committed with `uc_required_properties`) and returns the body to POST to
    the UC tables endpoint to finalise a managed table. It carries the schema,
    partition columns, protocol, properties (plus `delta.checkpointPolicy=v2`),
    the `delta.clustering`/`delta.rowTracking` domain metadata and the v0
    commit timestamp. `table_name` is passed through as the request's `name`.
    """

def uc_required_properties(uc_table_id: str) -> dict[str, str]:
    """Table properties a UC catalog-managed table must carry in its v0 commit.

    Includes `delta.feature.catalogManaged=supported`, v2 checkpoints, deletion
    vectors and `io.unitycatalog.tableId=<uc_table_id>`. Merge into
    `create_table(properties=...)`.
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
        timestamp_ms: int | None = None,
    ) -> Snapshot:
        """Resolve a snapshot.

        `log_tail` entries are `(version, filename, last_modified_millis, size)`
        and `max_catalog_version` caps the version that may be trusted. Together
        they are what make a `catalogManaged` table readable; omit both for an
        ordinary path-based table.

        `timestamp_ms` (mutually exclusive with `version`) time-travels to the
        latest version whose commit timestamp is at or before it (in-commit
        timestamps when enabled, else commit-file modification times), limited
        to versions the table can be reconstructed at. With a `log_tail`, the
        latest snapshot is resolved with the tail first, the version found, and
        the snapshot rebuilt at it with the same tail. Raises ValueError if the
        timestamp is before the earliest recreatable commit (version 0 or the
        oldest retained checkpoint).
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
    def scan(
        self,
        columns: list[str] | None = None,
        predicate: str | None = None,
        files: list[str] | None = None,
    ) -> Any:
        """Read the table as an Arrow stream, with deletion vectors applied.

        `predicate` is a JSON string used ONLY to skip files (by statistics and
        partition values); rows that do not match can still be returned, so the
        caller must apply the exact filter. AST:

        - junctions: `{"op": "and"|"or", "args": [p, ...]}`, `{"op": "not", "args": [p]}`
        - comparisons: `{"op": "eq"|"ne"|"lt"|"le"|"gt"|"ge", "args": [e1, e2]}`
        - `{"op": "is_null"|"is_not_null", "args": [e]}`
        - expressions: `{"column": ["a", "b"]}` (nested path, case-insensitive)
          or `{"literal": v, "type": "boolean"|"long"|"double"|"string"|"date"|
          "timestamp"|"timestamp_ntz"|"decimal"|"null"}` (date "YYYY-MM-DD",
          timestamps ISO-8601 or epoch microseconds, decimal as a string).

        Literals are coerced to the column's type only when exact (no rounding
        of doubles to float, decimals to fewer digits, or zoned timestamps to
        NTZ). Anything unconvertible degrades safely: dropped from an AND at
        positive polarity; makes an OR, or anything under a NOT that is not
        fully convertible, "no predicate". Unknown columns and ops never raise;
        only malformed JSON is a ValueError.

        `files` (requires the "file_restricted_scan" feature) restricts the
        read to data files whose path, exactly as `files()` reports it in its
        `path` column (as stored in the log: usually relative, URL-encoded), is
        listed. Other files are dropped before any data or deletion-vector
        I/O. Semantics per file are those of the full scan: deletion vectors
        applied, column mapping and partition values resolved, row order within
        a file preserved, and `predicate` skipping still honoured on top. Paths
        not in this snapshot are ignored; `[]` yields an empty stream with the
        same schema. So scans over a partition of `files()` union to the full
        scan -- the building block for distributed reads.
        """

    def files(self, predicate: str | None = None) -> Any:
        """One row per live data file, as an Arrow table (arro3 Table).

        Columns: `path` (string, as stored in the log: usually relative to the
        table root and URL-encoded), `size` (int64), `modification_time`
        (int64 ms), `partition_values` (map<string, string>; keys are physical
        names under column mapping, null for a NULL partition value), `stats`
        (raw JSON string, nullable), `deletion_vector` (nullable JSON string of
        `storageType`/`pathOrInlineDv`/`offset`/`sizeInBytes`/`cardinality` --
        if present, rows in the file are deleted and must be masked) and
        `num_records` (int64 from stats, nullable; counts rows *before* the
        deletion vector). `predicate` skips files exactly as in `scan`.
        """

    def metadata_json(self) -> str:
        """The current `metaData` action object as Delta-protocol JSON.

        Keys: id, name, description, format{provider, options}, schemaString,
        partitionColumns, configuration, createdTime (unset ones are null).
        Wrap as `{"metaData": ...}` for `commit_raw`.
        """

    def protocol_json(self) -> str:
        """The current `protocol` action object as Delta-protocol JSON.

        minReaderVersion/minWriterVersion, plus readerFeatures (reader 3) and
        writerFeatures (writer 7) only when the versions call for them.
        """

    def app_id_version(self, app_id: str) -> int | None:
        """The last version committed under `app_id` by a `txn` action, or None."""
    def domain_metadata(self, domain: str) -> str | None:
        """The configuration string of `domain`, or None if it has no live entry.

        System domains such as `delta.clustering` and `delta.rowTracking` are
        readable too.
        """

    def timestamp(self) -> int:
        """This version's commit timestamp in epoch milliseconds.

        The in-commit timestamp when ICT is enabled, else the commit file's
        modification time.
        """

    def checkpoint(self) -> bool:
        """Write a checkpoint at this snapshot's version (V1/V2 per table features).

        Returns True if written, False if a checkpoint already existed at this
        version. On a catalog-managed table, every commit up to this version
        must be published first (`publish`); the kernel refuses otherwise with
        ValueError, because a checkpoint over unpublished commits would leave a
        gap in the log for older readers.
        """

    def append(
        self,
        data: Any,
        uc: UcCommitConfig | None = None,
        engine_info: str | None = None,
        operation: str | None = None,
        overwrite: bool = False,
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, str] | None = None,
    ) -> int:
        """Append Arrow data as one transaction; returns the committed version.

        Pass `uc` for a catalog-managed table, where the commit is staged and
        then ratified by the catalog rather than written directly.

        Partitioned tables are supported: `data` must include every partition
        column (matched case-insensitively). Rows are grouped by their distinct
        partition-value tuple and each group is written as its own file with
        the partition columns removed; the kernel serialises the values per the
        Delta protocol (NULL -> null / `__HIVE_DEFAULT_PARTITION__` directory,
        dates as YYYY-MM-DD, timestamps in UTC). A partition column that cannot
        be cast to the table's type is a ValueError.

        `overwrite` removes every file visible in this snapshot in the same
        commit. `txn` is `(app_id, version)` for idempotent writes;
        `commit_metadata` goes into commitInfo.
        """

    def write_files(self, data: Any, uc: UcCommitConfig | None = None) -> bytes:
        """Write data files without committing; returns opaque fragment bytes.

        The worker half of a distributed write. Partitioned tables are handled
        exactly as in `append`. The files are durable when this returns but
        belong to no version until `commit_files` accepts them, so a coordinator
        that abandons the write leaves them behind as garbage.
        """

    def commit_files(
        self,
        fragments: list[bytes],
        uc: UcCommitConfig | None = None,
        engine_info: str | None = None,
        operation: str | None = None,
        overwrite: bool = False,
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, str] | None = None,
    ) -> int:
        """Commit fragments from `write_files` as one transaction.

        Every fragment lands at a single version, so a distributed write is
        atomic. `overwrite` removes every file visible in this snapshot in the
        same commit. Raises the same errors as `append`.
        """

    def publish(self, uc: UcCommitConfig | None = None) -> int:
        """Publish ratified-but-unpublished commits into `_delta_log/`."""
