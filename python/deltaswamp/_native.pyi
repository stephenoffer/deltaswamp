"""Type stubs for the compiled Rust extension.

Kept in the source tree so mypy and editors work whether or not the extension
has been built. The implementation lives in `crates/native/src/`.
"""

from typing import Any

KERNEL_VERSION: str

FEATURES: list[str]
"""Capabilities this build provides, by stable name.

One of: "predicate_skipping", "timestamp_travel", "table_changes", "files",
"metadata_json", "app_id_version", "commit_raw", "partitioned_append",
"uc_create_table_request", "checkpoint", "file_restricted_scan", "legacy_calendar_files",
"distributed_write", "deletion_vector_dml", "materialized_row_ids", "commit_log",
"compaction", "commit_info_patch", "streaming_compaction", "retry_options", "vacuum",
"restore", "row_tracking_compaction", "add_tags", "write_checksum", "incremental_files",
"path_clone", "row_tracking_dml", "check_constraints", "schema_evolution", "log_cleanup",
"symlink_manifest", "fsck", "value_constrained_checkpoint", "commit_timestamps",
"credential_slots". Gate on this list, not `hasattr`, so a stale build refuses cleanly.
"""

def kernel_version() -> str:
    """The delta_kernel release this extension was compiled against."""

def native_version() -> str:
    """The deltaswamp-native crate version."""

def runtime_is_multithreaded() -> bool:
    """True if the shared Tokio runtime supports `block_in_place`.

    `UCCommitter` requires it and panics on a current-thread runtime.
    """

# Every exception the extension raises (these classes, and the builtin
# FileNotFoundError, OSError and ValueError it also raises) carries a `kind`
# attribute: a stable code for what failed, one of "not_found", "storage",
# "kernel", "unsupported", "arrow", "invalid_input", "commit_conflict",
# "backfill_required", "retryable", "catalog_permission", "catalog_not_found"
# and "catalog_rejected". Classify by it, not by the message.

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

class InvalidInputError(ValueError):
    """The extension refused its input: arguments, data or fragments.

    A ValueError subclass, so existing `except ValueError` clauses still work.
    Other extension ValueErrors (kernel, Arrow, URL) stay plain ValueErrors.
    """

class CatalogCommitError(ValueError):
    """The catalog refused a commit (a status other than 409/429/5xx)."""

class CatalogPermissionError(CatalogCommitError):
    """The catalog rejected the commit's credentials or privileges (401/403)."""

class CatalogNotFoundError(CatalogCommitError):
    """The catalog no longer has this table (404)."""

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

    `partition_by` and `cluster_by` are mutually exclusive, and each may name
    a column once. The schema needs at least one column. Unsigned integer
    columns are widened to the next signed type (uint8 -> short, uint16 ->
    integer, uint32/uint64 -> long) so every value fits; uint64 values above
    the long range are refused on write. `table_root` may be a URL or a path
    (relative paths and `~/` are resolved); a URL must not contain an
    unencoded `?` or `#`.
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

def feed_versions(
    table_root: str,
    options: dict[str, str] | None = None,
    start_version: int | None = None,
    end_version: int | None = None,
    start_timestamp_ms: int | None = None,
    end_timestamp_ms: int | None = None,
) -> tuple[int, int | None]:
    """The `(start, end)` versions a change feed's bounds name, as `table_changes` reads them.

    A start timestamp is the first commit at or after it, an end timestamp the
    latest at or before it, by commit times as Delta assigns them (file times
    made monotonic before in-commit timestamps). ValueError when either is
    after the latest commit, an end is before the first, or the range is empty.
    """

def validate_retry_options(options: dict[str, str]) -> None:
    """Refuse retry storage options either engine would refuse or panic on.

    Raises InvalidInputError. Keys are read as delta-rs reads them.
    """

def set_credential_slot(
    slot: str, options: dict[str, str], expires_at: float | None = None
) -> None:
    """Publish a freshly vended credential in `slot`, for stores built with its key."""

def remove_credential_slot(slot: str) -> None:
    """Forget `slot`; stores built from it keep the credential they last saw."""

def probe_put_if_absent(table_root: str, options: dict[str, str] | None = None) -> bool:
    """Whether the store under `table_root` honours put-if-absent.

    Puts one sentinel under ``_delta_log/`` twice with put-if-absent, then
    deletes it: False if the second put succeeded (the store ignores the
    condition) or the store has no conditional put at all.
    """

def absolute_deletion_vector(table_root: str, descriptor: str) -> str:
    """A deletion-vector descriptor (JSON) of a file under `table_root`, made absolute.

    A relative (`u`) vector becomes a `p` one with the vector file's full URL;
    an inline or absolute one is returned unchanged. For shallow clones.
    """

def copy_objects(
    source_root: str,
    target_root: str,
    paths: list[str],
    source_options: dict[str, str] | None = None,
    target_options: dict[str, str] | None = None,
) -> int:
    """Copy `paths` (relative, URL-encoded) from one table root to another; bytes copied."""

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
    the UC tables endpoint to finalize a managed table. It carries the schema,
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
        identify: bool = False,
    ) -> Snapshot:
        """Resolve a snapshot.

        `identify=True` records the strong identity of the commit file the
        snapshot ends at (`commit_identity`), which `refresh` revalidates.

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

    def refresh(self, options: dict[str, str] | None = None, latest: bool = True) -> Snapshot:
        """This snapshot revalidated against storage and brought up to date.

        The commit file it ends at must still have the recorded identity;
        otherwise the table is read afresh (at the same version with
        `latest=False`). Needs a snapshot resolved with `identify=True`.
        Path-based tables only.
        """

    @property
    def commit_identity(self) -> str | None:
        """Opaque identity of the commit file at `version`, or None."""

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
        row_positions: bool = False,
        row_ids: bool = False,
        file_groups: list[int] | None = None,
        row_tracking: bool = False,
    ) -> Any:
        """Read the table as an Arrow stream, with deletion vectors applied.

        `row_positions` appends `__deltaswamp_file` (each row's data file, as
        the log stores its path) and `__deltaswamp_row_index` (its physical
        position in that file): the address a deletion vector uses. `row_ids`
        (which needs `row_positions`, and row tracking enabled) also appends
        `__deltaswamp_row_id`, each row's stable row id. `row_tracking` (with
        `file_groups` or `row_positions`, row tracking enabled) appends `__deltaswamp_row_id` and
        `__deltaswamp_row_commit_version`: each row's id and commit version as
        Databricks' `_metadata` reads them, for a compaction to write back.

        `files` are read in the order given, a few ahead in the background.
        `file_groups` (with `files`, not `row_positions`) splits them into runs
        of that many files each: every batch holds rows of one run only,
        consecutive files' batches merged, tagged by `__deltaswamp_file`
        (dictionary-encoded) -- a compaction's bins.

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
        a file preserved, and `predicate` skipping still honored on top. Paths
        not in this snapshot are ignored; `[]` yields an empty stream with the
        same schema. So scans over a partition of `files()` union to the full
        scan -- the building block for distributed reads.
        """

    def files(self, predicate: str | None = None, tags: bool = False) -> Any:
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
        `tags=True` appends `tags`: each add's tags as a JSON object (nullable).
        """

    def add_actions(self) -> list[str]:
        """Every live file as the `add` action (a JSON object) that restores it.

        All of the add as the log recorded it -- statistics, partition values,
        tags, deletion vector, `baseRowId`, `defaultRowCommitVersion`,
        `clusteringProvider` -- with `dataChange` true. Requires "restore".
        """

    @property
    def deleted_file_retention_ms(self) -> int | None:
        """`delta.deletedFileRetentionDuration` in ms as the kernel parses it; None if unset."""

    def vacuum_plan(
        self,
        cutoff_ms: int,
        lite: bool = False,
        partition_columns: list[str] | None = None,
    ) -> list[tuple[str, str, int, int]]:
        """What a VACUUM would delete with retention cutoff `cutoff_ms` (epoch ms).

        `(key, path, size, modified_ms)` per file: `key` as `delete_files`
        takes it, `path` relative to the table root. Referenced (never listed):
        live files and their deletion vectors, files of removes whose
        `deletionTimestamp` is at or after the cutoff and their vectors, and
        the change-data files of commits written since. A full plan lists the
        table directory (hidden paths skipped as Spark skips them) and keeps
        files modified at or after the cutoff; `lite` lists only what expired
        removes name. `partition_columns` are the names partition directories
        may carry. Requires "vacuum".
        """

    def delete_files(self, keys: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """Delete `keys` from `vacuum_plan` under the root: (deleted, [(key, error)])."""

    @property
    def log_retention_ms(self) -> int:
        """`delta.logRetentionDuration` in ms, or Delta's default (30 days) if unset."""

    def cleanup_log(
        self, cutoff_ms: int, dry_run: bool = False
    ) -> tuple[int | None, list[str], list[tuple[str, str]]]:
        """Delete the log below the newest checkpoint committed at or before `cutoff_ms`.

        Commit, checksum, checkpoint and compacted files below that checkpoint,
        and sidecars no retained v2 checkpoint references that are older than
        the cutoff. Commit times are in-commit timestamps where the table has
        them, else monotonized file modification times. Returns
        `(kept_checkpoint, deleted_keys, [(key, error)])`; `dry_run` returns
        the plan. Refuses a catalog-managed table and checkpointProtection.
        Requires "log_cleanup".
        """

    def write_symlink_manifest(self) -> list[str]:
        """Write `_symlink_format_manifest/[<partition>/]manifest` for the live files.

        As Spark writes them: one decoded absolute path per line, a manifest
        per Hive-escaped partition directory (an empty one for an empty
        unpartitioned table), and the manifests of partitions with no files
        deleted. Returns the manifests written, relative to the table root.
        Requires "symlink_manifest".
        """

    def missing_data_files(self) -> list[str]:
        """The add actions (JSON, as `add_actions`) of live files whose data file is gone.

        Deletion vectors are not looked for; a file outside the table root
        counts as gone. Requires "fsck".
        """

    def missing_files(self, adds: list[str]) -> list[str]:
        """URLs of the data and deletion-vector files `adds` (JSON) name that are gone."""

    def legacy_calendar_files(self, files: list[tuple[str, int]]) -> list[str]:
        """Which of `files` (`(path, size)`, paths as `files()` reports them)
        a reader that does not rebase would misread: written by Spark in its
        legacy hybrid calendar, or storing INT96 timestamps. Requires the
        "legacy_calendar_files" feature; reads only each file's footer.
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

    def domain_metadata(self, domain: str) -> str | None:
        """The configuration string of `domain`, or None if it has no live entry.

        System domains such as `delta.clustering` and `delta.rowTracking` are
        readable too.
        """

    def app_id_version(self, app_id: str) -> int | None:
        """The last `txn` version recorded for `app_id`, or None if none.

        Pair with `append(txn=(app_id, version))` for idempotent writes: skip
        a batch whose version is at or below this. Entries past
        `delta.setTransactionRetentionDuration` read as None.
        """

    def commit_log(self, after: int) -> list[tuple[int, str]]:
        """The raw commit files after version `after` up to this one, ascending.

        `(version, text)`, each text the newline-delimited actions of that
        commit. Published commits only.
        """

    def write_checksum(self, always: bool = False) -> bool:
        """Write `_delta_log/<version>.crc` for this snapshot, best effort.

        Only when cheap (a checksum at most 100 commits back, or a short log
        with no checkpoint) unless `always`; a commit that changed no file
        carries the previous checksum forward. True if one was written; never
        raises.
        """

    def incremental_files(
        self, base_version: int
    ) -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]] | None:
        """The data files added and removed in `(base_version, self.version]`.

        `(live_adds, removes)`, each sorted `(path, dv_unique_id)` pairs with
        paths as stored in the log; the adds are those still live here. None
        when the range's commits are no longer all in the log.
        """

    def timestamp(self) -> int:
        """This version's commit timestamp in epoch milliseconds.

        The in-commit timestamp when ICT is enabled, else the commit file's
        modification time.
        """

    def commit_timestamp(self) -> int:
        """This version's commit timestamp as Delta assigns it, in epoch ms.

        The in-commit timestamp when ICT is enabled, else the commit file's
        modification time made monotonic: no earlier than a millisecond after
        the commit before it, as Spark reports it and time travel compares it.
        """

    def file_commit_timestamps(self) -> list[tuple[int, int]]:
        """`(version, ms)` of each published commit timed by its file.

        Every commit when in-commit timestamps are off, those before their
        enablement when they were turned on later; the times made monotonic.
        """

    def version_at(self, timestamp_ms: int, at_or_after: bool = False) -> tuple[int, int]:
        """`(version, commit ms)` of the published commit a timestamp names.

        The latest at or before it, or with `at_or_after` the first at or
        after it. Raises ValueError ("timestamp out of range") when there is
        none.
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
        operation_parameters: dict[str, str] | None = None,
        blind_append: bool | None = None,
        metadata: str | None = None,
        protocol: str | None = None,
        constraints_checked: bool = False,
    ) -> int:
        """Append Arrow data as one transaction; returns the committed version.

        `operation_parameters` and `blind_append` are written into the commit's
        commitInfo as `operationParameters` and `isBlindAppend`, which kernel
        otherwise writes as `{}` and (unless true) not at all.

        Pass `uc` for a catalog-managed table, where the commit is staged and
        then ratified by the catalog rather than written directly.

        Partitioned tables are supported: `data` must include every partition
        column (matched case-insensitively). Rows are grouped by their distinct
        partition-value tuple and each group is written as its own file with
        the partition columns removed; the kernel serializes the values per the
        Delta protocol (NULL -> null / `__HIVE_DEFAULT_PARTITION__` directory,
        dates as YYYY-MM-DD, timestamps in UTC). A partition column that cannot
        be cast to the table's type without changing a value is a ValueError.

        Columns are matched to the table schema by name (case-insensitively),
        at every struct level, never by position: a missing nullable column is
        written as NULL, a missing NOT NULL one or an extra one is a
        ValueError, and narrower types (int32, ms timestamps, dictionary
        strings) are cast to the table's type when that is lossless.

        `overwrite` removes every file visible in this snapshot in the same
        commit. `txn` is `(app_id, version)` for idempotent writes;
        `commit_metadata` goes into commitInfo, and may not use a key the
        commitInfo action reserves (`operation`, `timestamp`, `txnId`, ...).

        `metadata` and `protocol` (log JSON, "schema_evolution") are the
        table's new metaData and protocol, committed with the rows: the rows
        are conformed to and written under the new schema, and both actions
        are written into the same commit. The metaData must keep the table's
        id. `constraints_checked` ("check_constraints") says the caller
        evaluated every CHECK constraint over every row written; the kernel
        refuses a table with the checkConstraints feature otherwise.
        """

    def write_files(
        self, data: Any, uc: UcCommitConfig | None = None, constraints_checked: bool = False
    ) -> bytes:
        """Write data files without committing; returns opaque fragment bytes.

        The worker half of a distributed write. Partitioned tables are handled
        exactly as in `append`. The files are durable when this returns but
        belong to no version until `commit_files` accepts them, so a coordinator
        that abandons the write leaves them behind as garbage.
        `constraints_checked` is as for `append`.
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
        operation_parameters: dict[str, str] | None = None,
        blind_append: bool | None = None,
        constraints_checked: bool = False,
    ) -> int:
        """Commit fragments from `write_files` as one transaction.

        Every fragment lands at a single version, so a distributed write is
        atomic. `overwrite` removes every file visible in this snapshot in the
        same commit. Raises the same errors as `append`. `constraints_checked`
        is as for `append`: every worker checked the rows it wrote.
        """

    def commit_dml(
        self,
        deletions: Any,
        data: Any | None = None,
        whole_files: list[str] | None = None,
        uc: UcCommitConfig | None = None,
        engine_info: str | None = None,
        operation: str | None = None,
        txn: tuple[str, int] | None = None,
        commit_metadata: dict[str, str] | None = None,
        data_change: bool = True,
        operation_parameters: dict[str, str] | None = None,
        blind_append: bool | None = None,
        add_tags: dict[str, str] | None = None,
        constraints_checked: bool = False,
    ) -> tuple[int, int, int, int]:
        """Commit row-level DML as deletion vectors, in one transaction.

        `whole_files` are removed outright; `data_change=False` commits the
        whole thing as a compaction (OPTIMIZE), the same rows in new files:
        `data` is then pulled one batch at a time and each batch written as
        one file as it arrives, and the table's CHECK constraints, generated
        and identity columns and invariants do not stop it (they constrain
        new values, and a compaction writes the ones it read). On a row-tracked
        table every commit's removes are staged here (kernel refuses them),
        and `__deltaswamp_row_id` / `__deltaswamp_row_commit_version` columns
        in `data` are written to the materialized row-tracking columns (a null
        is a fresh value). `add_tags` are written as the `tags` of every add.
        `constraints_checked` is as for `append`: every CHECK constraint holds
        for the rows in `data`.

        `deletions` is an Arrow stream of `path` and `row_index` columns, as a
        positional scan reports them. Each touched file's new deletions are
        unioned with its existing vector; a file left with no rows is removed.
        `data`, if given, is appended in the same commit, so a copy-on-write
        rewrite is `whole_files` plus their surviving rows in `data`. Returns
        `(version, deleted_rows, deletion_vectors_added, files_removed)`;
        nothing to change commits nothing and returns this snapshot's version.
        """

    @property
    def deletion_vectors_enabled(self) -> bool:
        """Whether the table accepts deletion-vector writes."""

    def publish(self, uc: UcCommitConfig | None = None) -> int:
        """Publish ratified-but-unpublished commits into `_delta_log/`."""
