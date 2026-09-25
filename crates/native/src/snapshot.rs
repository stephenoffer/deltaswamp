//! The Python-facing snapshot: the piece no other Python library has.
//!
//! A `catalogManaged` table cannot be opened by listing `_delta_log/` alone. The
//! catalog is the source of truth: it holds commits that are ratified but not
//! yet published, and it caps the version a reader may trust. Kernel models that
//! as two builder inputs -- `with_log_tail` (the ratified tail, pointing at
//! `_delta_log/_staged_commits/<version>.<uuid>.json`) and
//! `with_max_catalog_version` -- and neither is reachable from Python today.
//! That is what this module exposes.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::history_manager::{latest_version_as_of, HistoryCommitType};
use delta_kernel::snapshot::{CheckpointWriteResult, Snapshot, SnapshotRef};
use delta_kernel::{Engine, LogPath, Version};
use pyo3::prelude::*;
use pyo3_arrow::{PyRecordBatchReader, PySchema, PyTable};
use url::Url;

use crate::commit::{self, SharedEngine, UcCommitConfig};
use crate::dml;
use crate::error::{NativeError, Result};
use crate::files;
use crate::predicate::parse_predicate;
use crate::runtime;
use crate::scan::KernelBatchReader;
use crate::store;

/// One ratified-but-possibly-unpublished commit, as the catalog reports it.
///
/// Tuple form `(version, filename, last_modified_millis, size_bytes)`. The
/// filename is the bare `<version>.<uuid>.json`, not a full path -- kernel
/// resolves it under `_delta_log/_staged_commits/`.
type LogTailEntry = (u64, String, i64, u64);

#[pyclass(module = "deltaswamp._native", name = "Snapshot", frozen)]
pub struct PySnapshot {
    inner: SnapshotRef,
    /// The concrete engine, not `Arc<dyn Engine>`: the Parquet writer used on the
    /// commit path is a `DefaultEngine` method rather than part of the trait.
    engine: SharedEngine,
}

impl PySnapshot {
    /// Normalise a table root: kernel requires a trailing slash and rejects
    /// paths without one, with an error that is hard to act on.
    pub(crate) fn table_root_url(table_root: &str) -> Result<Url> {
        if table_root.trim().is_empty() {
            return Err(NativeError::Invalid(
                "the table root is empty; pass a URL or a filesystem path".to_string(),
            ));
        }
        let normalized = if table_root.ends_with('/') {
            table_root.to_string()
        } else {
            format!("{table_root}/")
        };
        match Url::parse(&normalized) {
            // A one-letter "scheme" is a Windows drive (`C:\t`), not a URL.
            Ok(url) if url.scheme().len() > 1 => {
                if url.query().is_some() || url.fragment().is_some() {
                    // `file:///t/a#b` parses as path `/t/a` plus a fragment,
                    // which silently addresses a different table.
                    return Err(NativeError::Invalid(format!(
                        "table root {table_root:?} contains '?' or '#', which a URL reads as \
                         a query or fragment rather than part of the path; percent-encode \
                         them (%3F, %23) or pass a filesystem path"
                    )));
                }
                Ok(url)
            }
            _ => {
                // Bare filesystem paths are a convenience users expect,
                // relative ones included (resolved against the working dir).
                // `~` is a shell expansion; unexpanded it silently made a
                // directory literally named "~" under the working directory.
                let home = std::env::var_os("HOME").filter(|h| !h.is_empty());
                let expanded = match (normalized.strip_prefix("~/"), home) {
                    (Some(rest), Some(home)) => std::path::Path::new(&home).join(rest),
                    _ => std::path::PathBuf::from(&normalized),
                };
                let path = std::path::absolute(&expanded).map_err(|e| {
                    NativeError::Invalid(format!(
                        "cannot interpret {table_root:?} as a table root: {e}"
                    ))
                })?;
                Url::from_directory_path(&path).map_err(|_| {
                    NativeError::Invalid(format!("cannot interpret {table_root:?} as a table root"))
                })
            }
        }
    }

    fn build_log_tail(table_root: &Url, entries: Vec<LogTailEntry>) -> Result<Vec<LogPath>> {
        // Kernel requires a contiguous ascending run; sorting here means callers
        // can pass the catalog's response through unmodified (UC returns commits
        // newest-first).
        let mut entries = entries;
        entries.sort_by_key(|(version, ..)| *version);

        for pair in entries.windows(2) {
            let (a, b) = (pair[0].0, pair[1].0);
            if b != a + 1 {
                return Err(NativeError::Invalid(format!(
                    "catalog log tail is not contiguous: version {a} is followed by {b}. \
                     Kernel requires an unbroken run of commits; re-fetch the tail."
                )));
            }
        }

        for (version, filename, ..) in &entries {
            // Kernel `join`s the name onto `_staged_commits/`, so a path, an
            // absolute URL or `..` would read a commit from somewhere else
            // entirely -- another table's log, say.
            let bare = !filename.is_empty()
                && filename != "."
                && filename != ".."
                && !filename.contains(['/', '\\', '?', '#', ':', '%']);
            if !bare {
                return Err(NativeError::Invalid(format!(
                    "catalog log tail entry for version {version} names {filename:?}; expected \
                     a bare staged-commit file name such as <version>.<uuid>.json"
                )));
            }
            // The version in the name is the one kernel reads; it must agree
            // with the one the contiguity check above ran on.
            let digits: String = filename.chars().take_while(char::is_ascii_digit).collect();
            if digits.parse::<u64>().ok() != Some(*version) {
                return Err(NativeError::Invalid(format!(
                    "catalog log tail entry for version {version} names {filename:?}, whose \
                     file name is for a different version"
                )));
            }
        }

        entries
            .into_iter()
            .map(|(_, filename, last_modified, size)| {
                LogPath::staged_commit(table_root.clone(), &filename, last_modified, size)
                    .map_err(NativeError::from)
            })
            .collect()
    }

    /// Build one snapshot. The catalog inputs are re-applied on every build so
    /// a timestamp lookup and the final build see the same ratified tail.
    fn build(
        engine: &SharedEngine,
        url: &Url,
        version: Option<Version>,
        log_tail: Option<&[LogTailEntry]>,
        max_catalog_version: Option<Version>,
    ) -> Result<SnapshotRef> {
        let mut builder = Snapshot::builder_for(url.as_str());
        if let Some(v) = version {
            builder = builder.at_version(v);
        }
        if let Some(entries) = log_tail {
            builder = builder.with_log_tail(Self::build_log_tail(url, entries.to_vec())?);
        }
        if let Some(v) = max_catalog_version {
            builder = builder.with_max_catalog_version(v);
        }
        Ok(runtime::block_on(async {
            builder.build(engine.as_ref() as &dyn Engine)
        })?)
    }

    /// Resolve the latest recreatable version as of `timestamp_ms`.
    ///
    /// The history search needs a snapshot to bound it, and on a catalog-managed
    /// table only the catalog's tail says what "latest" is -- so resolve the
    /// latest snapshot with the tail first, search, then rebuild at the found
    /// version with the same tail.
    fn resolve_as_of(
        engine: &SharedEngine,
        url: &Url,
        timestamp_ms: i64,
        log_tail: Option<&[LogTailEntry]>,
        max_catalog_version: Option<Version>,
    ) -> Result<SnapshotRef> {
        let latest = Self::build(engine, url, None, log_tail, max_catalog_version)?;
        let found = latest_version_as_of(
            &latest,
            engine.as_ref() as &dyn Engine,
            timestamp_ms,
            HistoryCommitType::Recreatable,
        )
        .map_err(|e| match e {
            delta_kernel::Error::LogHistory(inner) => NativeError::Invalid(format!(
                "no version of {url} can be reconstructed as of timestamp {timestamp_ms} ms \
                 ({inner}). The timestamp is before the earliest recreatable commit (version 0 \
                 or the oldest retained checkpoint); choose a later timestamp."
            )),
            other => NativeError::from(other),
        })?;
        if found.version == latest.version() {
            return Ok(latest);
        }
        Self::build(
            engine,
            url,
            Some(found.version),
            log_tail,
            max_catalog_version,
        )
    }
}

#[pymethods]
impl PySnapshot {
    /// Resolve a snapshot, optionally pinned to a catalog-supplied tail.
    ///
    /// `log_tail` and `max_catalog_version` are what make a `catalogManaged`
    /// table readable. Omit both for an ordinary path-based table.
    #[staticmethod]
    #[pyo3(signature = (
        table_root,
        options = None,
        version = None,
        log_tail = None,
        max_catalog_version = None,
        timestamp_ms = None,
    ))]
    fn resolve(
        py: Python<'_>,
        table_root: &str,
        options: Option<HashMap<String, String>>,
        version: Option<u64>,
        log_tail: Option<Vec<LogTailEntry>>,
        max_catalog_version: Option<u64>,
        timestamp_ms: Option<i64>,
    ) -> PyResult<Self> {
        if version.is_some() && timestamp_ms.is_some() {
            return Err(
                NativeError::Invalid("pass version or timestamp_ms, not both".to_string()).into(),
            );
        }
        let url = Self::table_root_url(table_root)?;
        let options = options.unwrap_or_default();

        // Log resolution does real I/O, so release the GIL for it.
        let (inner, engine) = py.detach(|| -> Result<(SnapshotRef, SharedEngine)> {
            let object_store = store::build_store(&url, &options)?;
            let engine = commit::new_engine(object_store);
            let tail = log_tail.as_deref();
            let snapshot = match timestamp_ms {
                Some(ts) => Self::resolve_as_of(&engine, &url, ts, tail, max_catalog_version)?,
                None => Self::build(&engine, &url, version, tail, max_catalog_version)?,
            };
            Ok((snapshot, engine))
        })?;

        Ok(Self { inner, engine })
    }

    #[getter]
    fn version(&self) -> u64 {
        self.inner.version()
    }

    #[getter]
    fn table_root(&self) -> String {
        self.inner.table_root().to_string()
    }

    /// The logical Arrow schema, after column-mapping resolution.
    fn schema(&self) -> PyResult<PySchema> {
        use delta_kernel::engine::arrow_conversion::TryIntoArrow;
        let arrow: arrow::datatypes::Schema = self
            .inner
            .schema()
            .as_ref()
            .try_into_arrow()
            .map_err(NativeError::from)?;
        Ok(PySchema::new(Arc::new(arrow)))
    }

    /// `(min_reader_version, min_writer_version, reader_features, writer_features)`.
    ///
    /// Feature names are returned verbatim, including ones this build does not
    /// recognize -- the Python layer decides what to do with them, because an
    /// unknown writer-only feature must not block a read.
    fn protocol(&self) -> (i32, i32, Vec<String>, Vec<String>) {
        let protocol = self.inner.table_configuration().protocol();
        let names = |fs: Option<&[delta_kernel::table_features::TableFeature]>| {
            fs.map(|f| f.iter().map(|x| x.to_string()).collect())
                .unwrap_or_default()
        };
        (
            protocol.min_reader_version(),
            protocol.min_writer_version(),
            names(protocol.reader_features()),
            names(protocol.writer_features()),
        )
    }

    /// Raw `delta.*` table properties as stored in the Metadata action.
    fn table_properties(&self) -> HashMap<String, String> {
        self.inner.metadata_configuration().clone()
    }

    /// The table's own id from the Metadata action.
    ///
    /// Compared against the catalog's table UUID: a table dropped and
    /// re-created under the same name gets a new id, and a stale one silently
    /// reads the wrong table.
    #[getter]
    fn metadata_id(&self) -> String {
        self.inner.table_configuration().metadata().id().to_string()
    }

    /// Logical partition columns, empty for an unpartitioned table.
    #[getter]
    fn partition_columns(&self) -> Vec<String> {
        self.inner
            .table_configuration()
            .logical_partition_columns()
            .to_vec()
    }

    #[getter]
    fn is_catalog_managed(&self) -> bool {
        self.inner.table_configuration().is_catalog_managed()
    }

    /// Read the table, returning an Arrow stream.
    ///
    /// Deletion vectors are applied by kernel and row order is preserved; see
    /// `crate::scan` for why that ordering matters.
    ///
    /// `predicate` (JSON, see `crate::predicate`) only skips files; rows that
    /// do not match can still come back and the caller must filter them.
    ///
    /// `files` restricts the read to data files whose log path (the `path`
    /// column of `files()`: relative, URL-encoded, as stored) is listed.
    /// Unlisted files are dropped before any I/O; unknown paths are ignored;
    /// `[]` gives an empty stream. Output is otherwise identical to the full
    /// scan, so the union of scans over a partition of `files()` equals it.
    ///
    /// `row_positions` appends two columns that address each row the way a
    /// deletion vector does: `__deltaswamp_file` (the data file's log path) and
    /// `__deltaswamp_row_index` (its physical position in that file, counted
    /// before any deletion vector). Rows already deleted are not returned.
    /// `row_ids` (with `row_positions`, on a table with row tracking enabled)
    /// adds `__deltaswamp_row_id`, each row's stable row id.
    #[pyo3(signature = (
        columns = None,
        predicate = None,
        files = None,
        row_positions = false,
        row_ids = false,
    ))]
    fn scan(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        predicate: Option<String>,
        files: Option<Vec<String>>,
        row_positions: bool,
        row_ids: bool,
    ) -> PyResult<PyRecordBatchReader> {
        if row_ids && !row_positions {
            return Err(
                NativeError::Invalid("row_ids=True needs row_positions=True".to_string()).into(),
            );
        }
        let reader = py.detach(|| -> Result<KernelBatchReader> {
            let full = self.inner.schema();
            let predicate = parse_predicate(predicate.as_deref(), full.as_ref())?;
            let mut builder = self
                .inner
                .clone()
                .scan_builder()
                .with_predicate(predicate.map(Arc::new));

            let mut schema = match columns {
                Some(columns) => {
                    let columns = crate::scan::resolve_columns(full.as_ref(), &columns)?;
                    full.project(&columns)?
                }
                None => full.clone(),
            };
            // Reading only partition columns (a projection such as
            // `columns=["region"]`, or a table with no data columns at all)
            // leaves the Parquet read schema empty, and kernel's reader then
            // panics building a zero-field struct -- killing the shared I/O
            // executor. A row-index column keeps the read non-empty and
            // carries each file's row count; it is dropped again below.
            let partition_columns = self
                .inner
                .table_configuration()
                .logical_partition_columns()
                .to_vec();
            let only_partitions = !row_positions
                && schema.num_fields() > 0
                && schema
                    .fields()
                    .all(|f| partition_columns.iter().any(|p| p == f.name()));
            if only_partitions || row_positions {
                schema = Arc::new(schema.add_metadata_column(
                    crate::scan::ROW_COUNT_COLUMN,
                    delta_kernel::schema::MetadataColumnSpec::RowIndex,
                )?);
            }
            if row_ids {
                schema = Arc::new(schema.add_metadata_column(
                    crate::scan::ROW_ID_COLUMN,
                    delta_kernel::schema::MetadataColumnSpec::RowId,
                )?);
            }
            builder = builder.with_schema(schema);

            let scan = builder.build()?;
            let engine = self.engine.clone() as Arc<dyn Engine>;
            if row_positions {
                let paths = files.map(|f| f.into_iter().collect());
                return KernelBatchReader::try_new_positional(&scan, engine, paths);
            }
            let reader = match files {
                Some(files) => KernelBatchReader::try_new_restricted(
                    &scan,
                    engine,
                    files.into_iter().collect(),
                ),
                None => KernelBatchReader::try_new(&scan, engine),
            }?;
            Ok(if only_partitions {
                reader.without_column(crate::scan::ROW_COUNT_COLUMN)
            } else {
                reader
            })
        })?;

        Ok(PyRecordBatchReader::new(Box::new(reader)))
    }

    /// One row per live data file, after predicate-based file skipping.
    ///
    /// Columns: `path` (as stored in the log -- usually relative to the table
    /// root, URL-encoded), `size`, `modification_time`, `partition_values`
    /// (map, keyed by *physical* name under column mapping), `stats` (raw JSON),
    /// `deletion_vector` (JSON descriptor or null) and `num_records`.
    #[pyo3(signature = (predicate = None))]
    fn files(&self, py: Python<'_>, predicate: Option<String>) -> PyResult<PyTable> {
        let batch = py.detach(|| -> Result<arrow::array::RecordBatch> {
            let predicate = parse_predicate(predicate.as_deref(), self.inner.schema().as_ref())?;
            files::list_files(self.inner.clone(), self.engine.as_ref(), predicate)
        })?;
        let schema = batch.schema();
        PyTable::try_new(vec![batch], schema)
    }

    /// The current `metaData` action, as Delta-protocol JSON.
    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.table_configuration().metadata())
            .map_err(|e| NativeError::Invalid(format!("could not serialize metadata: {e}")).into())
    }

    /// The current `protocol` action, as Delta-protocol JSON.
    ///
    /// Feature lists appear only when the versions call for them (reader 3,
    /// writer 7); the kernel serializer omits them otherwise.
    fn protocol_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.table_configuration().protocol())
            .map_err(|e| NativeError::Invalid(format!("could not serialize protocol: {e}")).into())
    }

    /// The configuration string of `domain`, or None if it has no live entry.
    ///
    /// System domains (`delta.clustering`, `delta.rowTracking`, ...) are
    /// readable too; the public kernel accessor refuses them, so this goes
    /// through the internal one.
    fn domain_metadata(&self, py: Python<'_>, domain: &str) -> PyResult<Option<String>> {
        let value = py.detach(|| {
            self.inner
                .get_domain_metadata_internal(domain, self.engine.as_ref())
                .map_err(NativeError::from)
        })?;
        Ok(value)
    }

    /// The last `txn` version recorded for `app_id`, or None if it has none.
    ///
    /// This is what makes `txn=(app_id, version)` writes idempotent: a writer
    /// skips a batch whose version is at or below this. Expired entries
    /// (`delta.setTransactionRetentionDuration`) read as None, as in Spark.
    fn app_id_version(&self, py: Python<'_>, app_id: &str) -> PyResult<Option<i64>> {
        let version = py.detach(|| {
            self.inner
                .get_app_id_version(app_id, self.engine.as_ref())
                .map_err(NativeError::from)
        })?;
        Ok(version)
    }

    /// This snapshot's commit timestamp in milliseconds: the in-commit
    /// timestamp when ICT is enabled, else the commit file's modification time.
    fn timestamp(&self, py: Python<'_>) -> PyResult<i64> {
        let ts = py.detach(|| {
            self.inner
                .get_timestamp(self.engine.as_ref())
                .map_err(NativeError::from)
        })?;
        Ok(ts)
    }

    /// Write a checkpoint at this snapshot's version.
    ///
    /// Returns True if a checkpoint was written, False if one already existed.
    /// On a catalog-managed table every commit up to this version must be
    /// published first; the kernel refuses otherwise, because a checkpoint over
    /// unpublished commits would leave a gap in the log for older readers.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<bool> {
        let written = py.detach(|| -> Result<bool> {
            // Called directly, not inside runtime::block_on: the engine's
            // executor does its own bridging (see commit::SharedEngine).
            let (result, _) = self.inner.checkpoint(self.engine.as_ref(), None)?;
            Ok(matches!(result, CheckpointWriteResult::Written))
        })?;
        Ok(written)
    }

    /// Append Arrow data and commit it as one transaction.
    ///
    /// Pass `uc` for a catalog-managed table: the commit is staged and then
    /// ratified by Unity Catalog rather than written directly. Omit it for a
    /// path-based table, which commits via an atomic object-store put.
    ///
    /// Returns the committed version. Raises `CommitConflictError` if another
    /// writer won the race, or `BackfillRequiredError` if the catalog wants
    /// staged commits published first -- those are different problems and the
    /// second is not fixed by retrying.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        data,
        uc = None,
        engine_info = None,
        operation = None,
        overwrite = false,
        txn = None,
        commit_metadata = None,
    ))]
    fn append(
        &self,
        py: Python<'_>,
        data: PyRecordBatchReader,
        uc: Option<UcCommitConfig>,
        engine_info: Option<String>,
        operation: Option<String>,
        overwrite: bool,
        txn: Option<(String, i64)>,
        commit_metadata: Option<HashMap<String, String>>,
    ) -> PyResult<u64> {
        let reader = data.into_reader()?;

        let version = py.detach(|| {
            // Draining the input is I/O too (it may be a dataset scan), and
            // holding the GIL for it stalled every other Python thread.
            let batches: std::result::Result<Vec<_>, _> = reader.collect();
            let batches = batches.map_err(NativeError::from)?;
            commit::write(
                self.inner.clone(),
                self.engine.clone(),
                batches,
                uc,
                engine_info,
                operation,
                overwrite,
                txn,
                commit_metadata,
            )
        })?;
        Ok(version)
    }

    /// Write data files without committing; returns opaque fragment bytes.
    ///
    /// The worker half of a distributed write. The files are durable when this
    /// returns, but nothing is in the log until `commit_files` runs, so a
    /// coordinator that gives up leaves them as garbage. Decide whether the
    /// commit can succeed *before* the first worker runs -- that is what
    /// `Table.can(...)` is for.
    #[pyo3(signature = (data, uc = None))]
    fn write_files(
        &self,
        py: Python<'_>,
        data: PyRecordBatchReader,
        uc: Option<UcCommitConfig>,
    ) -> PyResult<Vec<u8>> {
        let reader = data.into_reader()?;
        let batches: std::result::Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(NativeError::from)?;
        let result =
            py.detach(|| commit::write_files(self.inner.clone(), self.engine.clone(), batches, uc));
        match result {
            Ok(bytes) => Ok(bytes),
            Err(failure) => {
                if !failure.not_removed.is_empty() {
                    // Logged, never raised: the write's own error is the one
                    // the caller must see.
                    let message = format!(
                        "write_files failed and {} data file(s) it wrote could not be removed \
                         (VACUUM will): {}",
                        failure.not_removed.len(),
                        failure.not_removed.join("; ")
                    );
                    let _ = py
                        .import("logging")
                        .and_then(|m| m.call_method1("getLogger", ("deltaswamp",)))
                        .and_then(|l| l.call_method1("warning", (message,)));
                }
                Err(failure.error.into())
            }
        }
    }

    /// Commit fragments produced by `write_files`, wherever they were written.
    ///
    /// Every fragment lands in one transaction, so a distributed write appears
    /// at a single version and a reader never sees half a job.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        fragments,
        uc = None,
        engine_info = None,
        operation = None,
        overwrite = false,
        txn = None,
        commit_metadata = None,
    ))]
    fn commit_files(
        &self,
        py: Python<'_>,
        fragments: Vec<Vec<u8>>,
        uc: Option<UcCommitConfig>,
        engine_info: Option<String>,
        operation: Option<String>,
        overwrite: bool,
        txn: Option<(String, i64)>,
        commit_metadata: Option<HashMap<String, String>>,
    ) -> PyResult<u64> {
        let version = py.detach(|| {
            commit::commit_files(
                self.inner.clone(),
                self.engine.clone(),
                fragments,
                uc,
                engine_info,
                operation,
                overwrite,
                txn,
                commit_metadata,
            )
        })?;
        Ok(version)
    }

    /// Commit row-level DML as deletion vectors, the way Databricks writes it.
    ///
    /// `deletions` is an Arrow stream with columns `path` and `row_index`: the
    /// rows to delete, addressed as a positional scan (`row_positions=True`)
    /// reports them. `data`, if given, is appended in the same commit -- an
    /// UPDATE's rewritten rows. `whole_files` names files to delete entirely
    /// (every live row) without listing rows. Returns `(version, deleted_rows,
    /// deletion_vectors_added, files_removed)`; a DML that changes nothing
    /// commits nothing and returns this snapshot's version.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        deletions,
        data = None,
        whole_files = None,
        uc = None,
        engine_info = None,
        operation = None,
        txn = None,
        commit_metadata = None,
    ))]
    fn commit_dml(
        &self,
        py: Python<'_>,
        deletions: PyRecordBatchReader,
        data: Option<PyRecordBatchReader>,
        whole_files: Option<Vec<String>>,
        uc: Option<UcCommitConfig>,
        engine_info: Option<String>,
        operation: Option<String>,
        txn: Option<(String, i64)>,
        commit_metadata: Option<HashMap<String, String>>,
    ) -> PyResult<(u64, u64, usize, usize)> {
        let deletions = deletions.into_reader()?;
        let data = data.map(|d| d.into_reader()).transpose()?;
        let outcome = py.detach(|| -> Result<dml::DmlOutcome> {
            let deletions: std::result::Result<Vec<_>, _> = deletions.collect();
            let deletions = dml::deletions_from_batches(&deletions.map_err(NativeError::from)?)?;
            let batches = match data {
                Some(reader) => {
                    let batches: std::result::Result<Vec<_>, _> = reader.collect();
                    batches.map_err(NativeError::from)?
                }
                None => Vec::new(),
            };
            dml::commit_dml(
                self.inner.clone(),
                self.engine.clone(),
                deletions,
                whole_files.unwrap_or_default().into_iter().collect(),
                batches,
                uc,
                engine_info,
                operation,
                txn,
                commit_metadata,
            )
        })?;
        Ok((
            outcome.version,
            outcome.deleted_rows,
            outcome.deletion_vectors_added,
            outcome.files_removed,
        ))
    }

    /// Whether this table accepts deletion-vector writes: the feature on both
    /// sides of the protocol and `delta.enableDeletionVectors=true`.
    #[getter]
    fn deletion_vectors_enabled(&self) -> bool {
        dml::deletion_vectors_enabled(&self.inner)
    }

    /// Publish ratified-but-unpublished commits into `_delta_log/`.
    ///
    /// Not optional housekeeping on a catalog-managed table: the catalog caps how
    /// many unbackfilled commits it holds and refuses writes past the limit, and
    /// checkpoints only run on published versions.
    #[pyo3(signature = (uc = None))]
    fn publish(&self, py: Python<'_>, uc: Option<UcCommitConfig>) -> PyResult<u64> {
        let version = py.detach(|| commit::publish(self.inner.clone(), self.engine.clone(), uc))?;
        Ok(version)
    }
}

/// Create a Delta table and commit version 0.
///
/// Separate from `PySnapshot` because there is no snapshot to resolve yet.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    table_root,
    schema,
    options = None,
    properties = None,
    partition_by = None,
    cluster_by = None,
    uc = None,
    engine_info = None,
))]
pub fn create_table(
    py: Python<'_>,
    table_root: &str,
    schema: PySchema,
    options: Option<HashMap<String, String>>,
    properties: Option<HashMap<String, String>>,
    partition_by: Option<Vec<String>>,
    cluster_by: Option<Vec<String>>,
    uc: Option<UcCommitConfig>,
    engine_info: Option<String>,
) -> PyResult<u64> {
    // Kernel resolves a local table root through `try_parse_uri`, which requires
    // the directory to exist. Object stores have no such notion, but a local
    // create would otherwise fail before it began.
    // The directory comes from the parsed URL, so `file://` percent-escapes
    // (`my%20table`) and `file:/t` forms create the directory kernel will use.
    let url = PySnapshot::table_root_url(table_root)?;
    let mut created_dir = None;
    if url.scheme() == "file" {
        let path = url.to_file_path().map_err(|_| {
            NativeError::Invalid(format!("{table_root:?} is not a local filesystem path"))
        })?;
        if !path.exists() {
            created_dir = Some(path.clone());
        }
        std::fs::create_dir_all(&path).map_err(|e| {
            NativeError::Invalid(format!("could not create table directory {path:?}: {e}"))
        })?;
    }
    let arrow_schema = schema.into_inner();

    let version = py.detach(|| -> Result<u64> {
        let object_store = store::build_store(&url, &options.unwrap_or_default())?;
        let engine = commit::new_engine(object_store);
        commit::create_table(
            url.as_str(),
            arrow_schema,
            engine,
            properties,
            partition_by,
            cluster_by,
            uc,
            engine_info,
        )
    });
    if let (Err(_), Some(dir)) = (&version, created_dir) {
        // Do not leave an empty directory behind for a create that failed
        // validation; `remove_dir` only succeeds while it is still empty.
        let _ = std::fs::remove_dir(dir);
    }
    Ok(version?)
}
