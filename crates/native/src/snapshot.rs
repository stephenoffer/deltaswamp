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

use delta_kernel::history_manager::HistoryCommitType;
use delta_kernel::snapshot::{CheckpointWriteResult, Snapshot, SnapshotRef};
use delta_kernel::{Engine, LogPath, Version};
use pyo3::prelude::*;
use pyo3_arrow::{PyRecordBatchReader, PySchema, PyTable};
use url::Url;

use crate::commit::{self, SharedEngine, UcCommitConfig};
use crate::commit_time::{self, Bound};
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
    /// Identity of the commit file this snapshot ends at (see
    /// [`commit_identity`]), recorded when it was read, so a reused snapshot
    /// can prove the log it was built from is still the one in storage. None
    /// when it was not asked for or storage gives no strong change token.
    identity: Option<String>,
    /// Built by `planned` from a driver's protocol and metadata, with no log
    /// behind it: it reads only the files a plan hands it (`scan_rows=`).
    planned: bool,
}

/// A strong identity for the commit file `snapshot` ends at, or None.
///
/// What a cached snapshot is revalidated against. Size and a whole-second
/// Last-Modified are not enough: a table dropped and re-created at the same
/// path rewrites `00000000000000000001.json` with the same size in the same
/// second, and the old snapshot's schema was then applied to the new table's
/// files. So: the ETag or object version on an object store, and device,
/// inode, nanosecond mtime/ctime and size on a local filesystem. The file must
/// also still match what the log listing reported (size and Last-Modified),
/// so an identity is never paired with a file replaced after it was read.
fn commit_identity(
    store: &delta_kernel::object_store::DynObjectStore,
    snapshot: &SnapshotRef,
) -> Option<String> {
    let commit = snapshot
        .log_segment()
        .listed
        .latest_commit_file
        .as_ref()
        .filter(|c| c.version == snapshot.version())?;
    let listed = &commit.location;
    #[cfg(unix)]
    if listed.location.scheme() == "file" {
        use std::os::unix::fs::MetadataExt;
        let path = listed.location.to_file_path().ok()?;
        let m = std::fs::metadata(path).ok()?;
        if m.size() != listed.size {
            return None;
        }
        return Some(format!(
            "file:{}:{}:{}.{}:{}.{}:{}",
            m.dev(),
            m.ino(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
            m.size()
        ));
    }
    use delta_kernel::object_store::path::Path;
    use delta_kernel::object_store::ObjectStoreExt;
    let path = Path::from_url_path(listed.location.path()).ok()?;
    let meta = runtime::block_on(async { store.head(&path).await }).ok()?;
    if meta.size != listed.size || meta.last_modified.timestamp_millis() != listed.last_modified {
        return None;
    }
    if meta.e_tag.is_none() && meta.version.is_none() {
        return None; // nothing strong to compare: never reused
    }
    Some(format!(
        "object:{:?}:{:?}:{}",
        meta.e_tag, meta.version, meta.size
    ))
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

    /// Build a snapshot of a table that exists only as a template version 0
    /// (see `pending`): the template is the whole log tail.
    fn build_template(
        engine: &SharedEngine,
        url: &Url,
        version: Option<Version>,
        template: (String, i64, u64),
        max_catalog_version: Option<Version>,
    ) -> Result<SnapshotRef> {
        let (location, last_modified, size) = template;
        let tail = crate::pending::template_log_path(url, &location, last_modified, size)?;
        let mut builder = Snapshot::builder_for(url.as_str()).with_log_tail(vec![tail]);
        if let Some(v) = version {
            builder = builder.at_version(v);
        }
        if let Some(v) = max_catalog_version {
            builder = builder.with_max_catalog_version(v);
        }
        Ok(runtime::block_on(async {
            builder.build(engine.as_ref() as &dyn Engine)
        })?)
    }

    /// Resolve the latest recreatable version as of `timestamp_ms`, by commit
    /// times as Delta assigns them (see `commit_time`); a time after the
    /// latest commit is refused.
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
        let (found, _) = commit_time::version_at(
            &latest,
            engine.as_ref() as &dyn Engine,
            timestamp_ms,
            Bound::AtOrBefore,
            HistoryCommitType::Recreatable,
        )
        .map_err(|e| match e {
            NativeError::Invalid(inner) => NativeError::Invalid(format!(
                "no version of {url} can be reconstructed as of timestamp {timestamp_ms} ms \
                 ({inner}). The timestamp is before the earliest recreatable commit (version 0 \
                 or the oldest retained checkpoint); choose a later timestamp."
            )),
            other => other,
        })?;
        if found == latest.version() {
            let last = commit_time::commit_time(&latest, engine.as_ref() as &dyn Engine)?;
            if timestamp_ms > last {
                // Databricks refuses it (DELTA_TIMESTAMP_GREATER_THAN_COMMIT):
                // no version exists at that time yet, and the same timestamp
                // would read different data once the next commit landed.
                return Err(NativeError::Invalid(format!(
                    "timestamp {timestamp_ms} ms is after the latest commit (version {}, at \
                     {last} ms), so no version of {url} exists at that time \
                     (DELTA_TIMESTAMP_GREATER_THAN_COMMIT)",
                    latest.version()
                )));
            }
            return Ok(latest);
        }
        Self::build(engine, url, Some(found), log_tail, max_catalog_version)
    }
}

impl PySnapshot {
    /// The error for a log read on a `planned` snapshot, which has no log.
    fn planned_refusal(&self, what: &str) -> PyErr {
        NativeError::Invalid(format!(
            "cannot {what}: this snapshot was built from a scan plan and has no log behind \
             it; resolve the table to do that"
        ))
        .into()
    }

    /// The object store this snapshot's engine reads through.
    fn store(&self) -> Result<Arc<delta_kernel::object_store::DynObjectStore>> {
        self.engine
            .get_object_store_for_url(self.inner.table_root())
            .ok_or_else(|| NativeError::Invalid("the engine has no object store".into()))
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
        identify = false,
        template = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        py: Python<'_>,
        table_root: &str,
        options: Option<HashMap<String, String>>,
        version: Option<u64>,
        log_tail: Option<Vec<LogTailEntry>>,
        max_catalog_version: Option<u64>,
        timestamp_ms: Option<i64>,
        identify: bool,
        template: Option<(String, i64, u64)>,
    ) -> PyResult<Self> {
        if version.is_some() && timestamp_ms.is_some() {
            return Err(
                NativeError::Invalid("pass version or timestamp_ms, not both".to_string()).into(),
            );
        }
        if template.is_some() && (log_tail.is_some() || timestamp_ms.is_some() || identify) {
            // A template is the whole log of a table that does not exist yet.
            return Err(NativeError::Invalid(
                "a template snapshot takes no log tail, timestamp or identity".to_string(),
            )
            .into());
        }
        let url = Self::table_root_url(table_root)?;
        let options = options.unwrap_or_default();

        // Log resolution does real I/O, so release the GIL for it.
        type Resolved = (SnapshotRef, SharedEngine, Option<String>);
        let (inner, engine, identity) = py.detach(|| -> Result<Resolved> {
            let object_store = store::build_store(&url, &options)?;
            let engine = commit::new_engine(object_store.clone());
            let tail = log_tail.as_deref();
            let snapshot = match (template, timestamp_ms) {
                (Some(template), _) => {
                    Self::build_template(&engine, &url, version, template, max_catalog_version)?
                }
                (None, Some(ts)) => {
                    Self::resolve_as_of(&engine, &url, ts, tail, max_catalog_version)?
                }
                (None, None) => Self::build(&engine, &url, version, tail, max_catalog_version)?,
            };
            let identity = if identify {
                commit_identity(object_store.as_ref(), &snapshot)
            } else {
                None
            };
            Ok((snapshot, engine, identity))
        })?;

        Ok(Self {
            inner,
            engine,
            identity,
            planned: false,
        })
    }

    /// A snapshot of `table_root` at `version`, built from the protocol and
    /// metadata a driver planned with, without reading the log.
    ///
    /// A worker reads planned files through it: `scan(files=..., scan_rows=...)`
    /// with the scan rows `files(scan_rows=True)` listed on the driver. No log
    /// listing, no replay, no catalog tail -- so a read task costs only its
    /// own files, and a catalog-managed table still reads after the staged
    /// commits it was planned from were published and removed. Everything
    /// else that reads the log is refused on it.
    #[staticmethod]
    #[pyo3(signature = (table_root, version, protocol_json, metadata_json, options = None))]
    fn planned(
        py: Python<'_>,
        table_root: &str,
        version: u64,
        protocol_json: &str,
        metadata_json: &str,
        options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        use delta_kernel::actions::{Metadata, Protocol};
        use delta_kernel::log_segment::LogSegment;
        use delta_kernel::log_segment_files::LogSegmentFiles;
        use delta_kernel::path::ParsedLogPath;
        use delta_kernel::table_configuration::TableConfiguration;
        use delta_kernel::FileMeta;

        let url = Self::table_root_url(table_root)?;
        let options = options.unwrap_or_default();
        let invalid = |what: &str, e: serde_json::Error| {
            NativeError::Invalid(format!("the planned {what} is not valid JSON: {e}"))
        };
        let protocol: Protocol =
            serde_json::from_str(protocol_json).map_err(|e| invalid("protocol", e))?;
        let metadata: Metadata =
            serde_json::from_str(metadata_json).map_err(|e| invalid("metadata", e))?;
        let (inner, engine) = py.detach(|| -> Result<(SnapshotRef, SharedEngine)> {
            let engine = commit::new_engine(store::build_store(&url, &options)?);
            let config = TableConfiguration::try_new(metadata, protocol, url.clone(), version)?;
            // The segment names the planned version's commit so the snapshot
            // is well formed; nothing reads it (see `scan`'s scan_rows).
            let log_root = url.join("_delta_log/").map_err(NativeError::from)?;
            let location = log_root
                .join(&format!("{version:020}.json"))
                .map_err(NativeError::from)?;
            let commit = ParsedLogPath::try_from(FileMeta {
                location,
                last_modified: 0,
                size: 0,
            })?
            .ok_or_else(|| NativeError::Invalid("no commit path for the version".into()))?;
            let files = LogSegmentFiles {
                ascending_commit_files: vec![commit.clone()],
                latest_commit_file: Some(commit),
                ..Default::default()
            };
            let segment = LogSegment::try_new(files, log_root, Some(version), None)?;
            Ok((Arc::new(Snapshot::new(segment, config)?), engine))
        })?;
        Ok(Self {
            inner,
            engine,
            identity: None,
            planned: true,
        })
    }

    /// This snapshot revalidated against storage, and brought up to date.
    ///
    /// Only a snapshot resolved with `identify=True` can be refreshed. The
    /// commit file it ends at is checked first, with a real read of its
    /// strong identity through a store built from `options` (so a re-vended
    /// credential is used, and a connection that cannot reach storage fails
    /// here rather than being handed what another connection read). If it is
    /// gone or replaced -- the table was deleted and re-created at the same
    /// path -- the table is read afresh at the same version (or the latest).
    /// Otherwise a pinned version (`latest=False`) is reused as it is, and the
    /// latest is brought forward by kernel's incremental update, which lists
    /// `_delta_log/` from this snapshot's version and replays only newer
    /// commits: one listing when nothing changed, where `resolve` replays
    /// everything since the last checkpoint.
    ///
    /// Path-based tables only: a catalog-managed table's latest version is
    /// whatever the catalog ratified, which a listing cannot see.
    #[pyo3(signature = (options = None, latest = true))]
    fn refresh(
        &self,
        py: Python<'_>,
        options: Option<HashMap<String, String>>,
        latest: bool,
    ) -> PyResult<Self> {
        let Some(recorded) = self.identity.clone() else {
            return Err(NativeError::Invalid(
                "this snapshot recorded no identity for its commit file, so it cannot be \
                 revalidated; resolve the table afresh"
                    .to_string(),
            )
            .into());
        };
        let url = self.inner.table_root().clone();
        let options = options.unwrap_or_default();
        let existing = self.inner.clone();
        type Refreshed = (SnapshotRef, SharedEngine, Option<String>);
        let (inner, engine, identity) = py.detach(|| -> Result<Refreshed> {
            let object_store = store::build_store(&url, &options)?;
            let engine = commit::new_engine(object_store.clone());
            let pinned = (!latest).then(|| existing.version());
            if commit_identity(object_store.as_ref(), &existing).as_deref()
                != Some(recorded.as_str())
            {
                // Replaced, gone, or unreadable with these options: read the
                // table afresh, as `resolve` would, with its own errors.
                let snapshot = Self::build(&engine, &url, pinned, None, None)?;
                let identity = commit_identity(object_store.as_ref(), &snapshot);
                return Ok((snapshot, engine, identity));
            }
            if !latest {
                return Ok((existing, engine, Some(recorded)));
            }
            let snapshot = match runtime::block_on(async {
                Snapshot::builder_from(existing.clone()).build(engine.as_ref() as &dyn Engine)
            }) {
                Ok(snapshot) => snapshot,
                // Anything unexpected (the log shrank under it): read the
                // table afresh.
                Err(_) => Self::build(&engine, &url, None, None, None)?,
            };
            let identity = if snapshot.version() == existing.version() {
                Some(recorded)
            } else {
                commit_identity(object_store.as_ref(), &snapshot)
            };
            Ok((snapshot, engine, identity))
        })?;
        Ok(Self {
            inner,
            engine,
            identity,
            planned: false,
        })
    }

    /// The identity recorded for the commit file this snapshot ends at, or
    /// None: what `refresh` revalidates. Opaque; compare for equality only.
    #[getter]
    fn commit_identity(&self) -> Option<String> {
        self.identity.clone()
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
    ///
    /// `row_tracking` (with `file_groups` or `row_positions`, on a table with
    /// row tracking enabled) adds `__deltaswamp_row_id` and `__deltaswamp_row_commit_version`,
    /// each row's id and commit version as Databricks' `_metadata` reads them:
    /// what a compaction writes back into the materialized columns.
    #[pyo3(signature = (
        columns = None,
        predicate = None,
        files = None,
        row_positions = false,
        row_ids = false,
        file_groups = None,
        row_tracking = false,
        scan_rows = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn scan(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        predicate: Option<String>,
        files: Option<Vec<String>>,
        row_positions: bool,
        row_ids: bool,
        file_groups: Option<Vec<usize>>,
        row_tracking: bool,
        scan_rows: Option<Vec<String>>,
    ) -> PyResult<PyRecordBatchReader> {
        if scan_rows.is_some() && (row_positions || file_groups.is_some() || row_tracking) {
            return Err(NativeError::Invalid(
                "scan_rows=... reads planned files only; it takes no row_positions, \
                 file_groups or row_tracking"
                    .to_string(),
            )
            .into());
        }
        if self.planned && scan_rows.is_none() {
            return Err(self.planned_refusal("scan without the planned scan rows"));
        }
        if row_tracking && file_groups.is_none() && !row_positions {
            return Err(NativeError::Invalid(
                "row_tracking=True takes file_groups=... or row_positions=True".to_string(),
            )
            .into());
        }
        if file_groups.is_some() && (row_positions || files.is_none()) {
            return Err(NativeError::Invalid(
                "file_groups=... takes files=... and not row_positions=True".to_string(),
            )
            .into());
        }
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
            if row_ids || row_tracking {
                schema = Arc::new(schema.add_metadata_column(
                    crate::scan::ROW_ID_COLUMN,
                    delta_kernel::schema::MetadataColumnSpec::RowId,
                )?);
            }
            let commit_versions = if row_tracking {
                let column = self
                    .inner
                    .metadata_configuration()
                    .get("delta.rowTracking.materializedRowCommitVersionColumnName")
                    .cloned()
                    .ok_or_else(|| {
                        NativeError::Invalid(
                            "the table names no \
                             delta.rowTracking.materializedRowCommitVersionColumnName, so its \
                             rows' commit versions cannot be read"
                                .to_string(),
                        )
                    })?;
                Some(crate::scan::CommitVersions::new(column))
            } else {
                None
            };
            builder = builder.with_schema(schema);

            let scan = builder.build()?;
            // Rebases Parquet files Spark wrote in its legacy hybrid calendar.
            // Confined to the table root: a log path that climbs out of it is
            // refused, not read with the table's credentials (confine.rs).
            let engine = crate::confine::confined(
                crate::rebase::reading_engine(&self.engine, self.inner.table_root()),
                self.inner.table_root(),
            );
            if row_positions {
                let paths = files.map(|f| f.into_iter().collect());
                return KernelBatchReader::try_new_positional(
                    &scan,
                    engine,
                    paths,
                    commit_versions,
                );
            }
            if let (Some(files), Some(groups)) = (&files, file_groups) {
                let reader = KernelBatchReader::try_new_grouped(
                    &scan,
                    engine,
                    files.clone(),
                    groups,
                    commit_versions,
                )?;
                return Ok(if only_partitions {
                    reader.without_column(crate::scan::ROW_COUNT_COLUMN)
                } else {
                    reader
                });
            }
            if let Some(rows) = scan_rows {
                let reader = KernelBatchReader::try_new_planned(
                    &scan,
                    engine,
                    &crate::scan::PlannedFiles(rows),
                    files,
                )?;
                return Ok(if only_partitions {
                    reader.without_column(crate::scan::ROW_COUNT_COLUMN)
                } else {
                    reader
                });
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
    /// `tags=True` adds `tags`, each add's tags as a JSON object.
    ///
    /// `scan_rows=True` adds `scan_row`, each file's kernel scan row as JSON:
    /// what a worker passes back to `scan(scan_rows=...)` to read the file with
    /// no log replay.
    #[pyo3(signature = (predicate = None, tags = false, scan_rows = false))]
    fn files(
        &self,
        py: Python<'_>,
        predicate: Option<String>,
        tags: bool,
        scan_rows: bool,
    ) -> PyResult<PyTable> {
        if self.planned {
            return Err(self.planned_refusal("list its files"));
        }
        let batch = py.detach(|| -> Result<arrow::array::RecordBatch> {
            let predicate = parse_predicate(predicate.as_deref(), self.inner.schema().as_ref())?;
            files::list_files(
                self.inner.clone(),
                self.engine.as_ref(),
                predicate,
                tags,
                scan_rows,
            )
        })?;
        let schema = batch.schema();
        PyTable::try_new(vec![batch], schema)
    }

    /// Every live file as the `add` action that restores it (JSON objects).
    fn add_actions(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(py.detach(|| files::add_actions(self.inner.clone(), self.engine.as_ref()))?)
    }

    /// `delta.deletedFileRetentionDuration` in milliseconds, as the kernel
    /// parses it; None when the table does not set it.
    #[getter]
    fn deleted_file_retention_ms(&self) -> Option<u64> {
        self.inner
            .table_properties()
            .deleted_file_retention_duration
            .map(|d| d.as_millis() as u64)
    }

    /// What a VACUUM would delete with retention cutoff `cutoff_ms` (epoch
    /// ms): `(key, path, size, modified_ms)` per file, `key` as `delete_files`
    /// takes it and `path` relative to the table root.
    #[pyo3(signature = (cutoff_ms, lite = false, partition_columns = None))]
    fn vacuum_plan(
        &self,
        py: Python<'_>,
        cutoff_ms: i64,
        lite: bool,
        partition_columns: Option<Vec<String>>,
    ) -> PyResult<Vec<(String, String, u64, i64)>> {
        let plan = py.detach(|| -> Result<Vec<crate::vacuum::Candidate>> {
            crate::vacuum::plan(
                &self.inner,
                self.engine.as_ref(),
                self.store()?,
                cutoff_ms,
                lite,
                &partition_columns.unwrap_or_default(),
            )
        })?;
        Ok(plan
            .into_iter()
            .map(|c| (c.key, c.path, c.size, c.modified_ms))
            .collect())
    }

    /// Delete the files `vacuum_plan` reported, by key. Returns the keys
    /// deleted (or already gone) and `(key, error)` for each that failed.
    fn delete_files(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<crate::vacuum::Deletion> {
        Ok(py.detach(|| crate::vacuum::delete(self.store()?, self.inner.table_root(), keys))?)
    }

    /// `delta.logRetentionDuration` in milliseconds, as the kernel parses it,
    /// or Delta's default (30 days) when the table does not set it.
    #[getter]
    fn log_retention_ms(&self) -> u64 {
        self.inner
            .table_properties()
            .log_retention_duration
            .map(|d| d.as_millis() as u64)
            .unwrap_or(crate::logclean::DEFAULT_LOG_RETENTION_MS)
    }

    /// Clean up the log below the newest checkpoint committed at or before
    /// `cutoff_ms` (see `crate::logclean`). Returns `(kept_checkpoint,
    /// deleted, failed)`: the checkpoint the retained history starts from,
    /// the keys deleted (relative to the table root) and `(key, error)` for
    /// each that was not. `dry_run` deletes nothing and returns the plan.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (cutoff_ms, dry_run = false))]
    fn cleanup_log(
        &self,
        py: Python<'_>,
        cutoff_ms: i64,
        dry_run: bool,
    ) -> PyResult<(Option<u64>, Vec<String>, Vec<(String, String)>)> {
        Ok(py.detach(
            || -> Result<(Option<u64>, Vec<String>, Vec<(String, String)>)> {
                let store = self.store()?;
                let plan = crate::logclean::plan(
                    &self.inner,
                    self.engine.as_ref(),
                    store.clone(),
                    cutoff_ms,
                )?;
                if dry_run {
                    return Ok((plan.kept_checkpoint, plan.keys, Vec::new()));
                }
                let (deleted, failed) =
                    crate::logclean::delete(store, self.inner.table_root(), plan.keys)?;
                Ok((plan.kept_checkpoint, deleted, failed))
            },
        )?)
    }

    /// Write the symlink format manifests of this snapshot (see
    /// `crate::manifest`), deleting those of partitions with no files.
    /// Returns the manifests written, relative to the table root.
    fn write_symlink_manifest(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(py.detach(|| crate::manifest::write(&self.inner, self.engine.as_ref(), self.store()?))?)
    }

    /// Of the data and deletion-vector files `adds` (add actions as JSON)
    /// reference, the ones missing from storage, as URLs.
    fn missing_files(&self, py: Python<'_>, adds: Vec<String>) -> PyResult<Vec<String>> {
        Ok(py.detach(|| -> Result<Vec<String>> {
            let root = self.inner.table_root();
            let mut urls = Vec::new();
            for add in &adds {
                let add: serde_json::Value = serde_json::from_str(add)
                    .map_err(|e| NativeError::Invalid(format!("bad add action: {e}")))?;
                let path = add
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| NativeError::Invalid("an add action without a path".into()))?;
                urls.push(match url::Url::parse(path) {
                    Ok(url) => url,
                    Err(_) => root.join(path)?,
                });
                if let Some(dv) = add.get("deletionVector").filter(|dv| !dv.is_null()) {
                    if let Some(url) = crate::vacuum::deletion_vector_url_json(root, dv)? {
                        urls.push(url);
                    }
                }
            }
            crate::vacuum::missing(self.store()?, root, urls)
        })?)
    }

    /// The live files whose data file is gone from storage (FSCK REPAIR):
    /// their add actions as JSON, as `add_actions` gives them. A deletion
    /// vector is not looked for; a file outside the table root counts as
    /// gone, so callers refuse shallow clones.
    fn missing_data_files(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(py.detach(|| -> Result<Vec<String>> {
            let root = self.inner.table_root();
            let adds = files::add_actions(self.inner.clone(), self.engine.as_ref())?;
            let mut by_url: HashMap<String, Vec<String>> = HashMap::new();
            let mut urls = Vec::new();
            for add in adds {
                let parsed: serde_json::Value = serde_json::from_str(&add)
                    .map_err(|e| NativeError::Invalid(format!("bad add action: {e}")))?;
                let path = parsed
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| NativeError::Invalid("an add action without a path".into()))?;
                let url = match url::Url::parse(path) {
                    Ok(url) => url,
                    Err(_) => root.join(path)?,
                };
                by_url.entry(url.to_string()).or_default().push(add);
                urls.push(url);
            }
            let missing = crate::vacuum::missing(self.store()?, root, urls)?;
            let mut out = Vec::new();
            for url in missing {
                out.extend(by_url.remove(&url).unwrap_or_default());
            }
            Ok(out)
        })?)
    }

    /// Which of `files` (`(path, size)` pairs, paths as `files()` reports
    /// them) hold values a reader that does not rebase would misread: written
    /// by Spark in its legacy hybrid calendar, or storing INT96 timestamps.
    /// Reads only each file's footer.
    fn legacy_calendar_files(
        &self,
        py: Python<'_>,
        files: Vec<(String, u64)>,
    ) -> PyResult<Vec<String>> {
        Ok(py.detach(|| -> Result<Vec<String>> {
            Ok(crate::rebase::legacy_calendar_files(
                &self.engine,
                self.inner.table_root(),
                &files,
            )?)
        })?)
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

    /// The raw commit files `_delta_log/<v>.json` for `after < v <= self.version`,
    /// as `(version, text)` pairs in ascending order.
    ///
    /// What a DML that lost a commit race needs to apply Delta's conflict
    /// rules: whether each winning commit was a blind append, and which files
    /// it added. A commit the snapshot's log segment holds is read from the
    /// file it names -- on a catalog-managed table, the ratified but
    /// unpublished `_delta_log/_staged_commits/<v>.<uuid>.json` -- and any
    /// other from its published path.
    ///
    /// `until` stops at that version instead, so a long range is read in
    /// bounded chunks.
    #[pyo3(signature = (after, until = None))]
    fn commit_log(
        &self,
        py: Python<'_>,
        after: u64,
        until: Option<u64>,
    ) -> PyResult<Vec<(u64, String)>> {
        if self.planned {
            return Err(self.planned_refusal("read its commits"));
        }
        let end = until.map_or(self.inner.version(), |u| u.min(self.inner.version()));
        let root = self.inner.table_root().clone();
        let texts = py.detach(|| -> Result<Vec<(u64, String)>> {
            if after >= end {
                return Ok(Vec::new());
            }
            let versions: Vec<u64> = ((after + 1)..=end).collect();
            let listed: std::collections::HashMap<u64, url::Url> = self
                .inner
                .log_segment()
                .listed
                .ascending_commit_files
                .iter()
                .map(|p| (p.version, p.location.location.clone()))
                .collect();
            let urls = versions
                .iter()
                .map(|v| match listed.get(v) {
                    Some(url) => Ok((url.clone(), None)),
                    None => root
                        .join(&format!("_delta_log/{v:020}.json"))
                        .map(|url| (url, None))
                        .map_err(|e| NativeError::Invalid(format!("bad commit path: {e}"))),
                })
                .collect::<Result<Vec<_>>>()?;
            let bytes = self.engine.storage_handler().read_files(urls)?;
            versions
                .into_iter()
                .zip(bytes)
                .map(|(v, data)| {
                    let data = data?;
                    let text = String::from_utf8(data.to_vec()).map_err(|e| {
                        NativeError::Invalid(format!("commit {v} is not UTF-8: {e}"))
                    })?;
                    Ok((v, text))
                })
                .collect()
        })?;
        Ok(texts)
    }

    /// Write `_delta_log/<version>.crc` for this snapshot, best effort.
    ///
    /// For a commit another writer made (delta-rs writes none). Only when it
    /// is cheap -- a checksum at most `checksum::MAX_TAIL` commits back, or a
    /// short log with no checkpoint -- unless `always`. True if one was
    /// written; False if it was not worth it, already existed, or failed.
    #[pyo3(signature = (always = false))]
    fn write_checksum(&self, py: Python<'_>, always: bool) -> bool {
        py.detach(|| crate::checksum::write(&self.inner, self.engine.as_ref(), always))
    }

    /// The data files added and removed in `(base_version, self.version]`.
    ///
    /// `(live_adds, removes)`, each a sorted list of `(path, dv_unique_id)`
    /// with paths as stored in the log: the adds still live at this version,
    /// and every remove in the range. From the kernel's incremental scan,
    /// which walks this snapshot's commit list (a catalog-managed table's
    /// ratified tail included) rather than listing storage. None when those
    /// commits are no longer all in the log (cleaned up past a checkpoint).
    #[allow(clippy::type_complexity)]
    fn incremental_files(
        &self,
        py: Python<'_>,
        base_version: u64,
    ) -> PyResult<Option<(Vec<(String, Option<String>)>, Vec<(String, Option<String>)>)>> {
        type Keys = Vec<(String, Option<String>)>;
        let diff = py.detach(|| -> Result<Option<(Keys, Keys)>> {
            if base_version >= self.inner.version() {
                return Ok(Some((Vec::new(), Vec::new())));
            }
            let Some(stream) = self
                .inner
                .clone()
                .incremental_scan_builder(base_version)
                .build(self.engine.as_ref())?
            else {
                return Ok(None);
            };
            let summary = stream.into_summary()?;
            let keys =
                |set: std::collections::HashSet<delta_kernel::log_replay::FileActionKey>| -> Keys {
                    let mut out: Keys = set
                        .iter()
                        .map(|k| (k.path().to_string(), k.dv_unique_id().map(str::to_string)))
                        .collect();
                    out.sort();
                    out
                };
            Ok(Some((keys(summary.live_adds), keys(summary.removes))))
        })?;
        Ok(diff)
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

    /// This snapshot's commit timestamp in milliseconds as Delta assigns it:
    /// the in-commit timestamp when ICT is enabled, else the commit file's
    /// modification time made monotonic -- no earlier than a millisecond
    /// after the commit before it, as Spark's history reports it and time
    /// travel compares it. Lists the log when ICT is off.
    fn commit_timestamp(&self, py: Python<'_>) -> PyResult<i64> {
        Ok(py.detach(|| commit_time::commit_time(&self.inner, self.engine.as_ref()))?)
    }

    /// `(version, ms)` of every commit timed by its file (all of them when
    /// in-commit timestamps are off, those before their enablement when they
    /// were turned on later), the times made monotonic as `commit_timestamp`.
    fn file_commit_timestamps(&self, py: Python<'_>) -> PyResult<Vec<(u64, i64)>> {
        Ok(py.detach(|| commit_time::file_commit_times(&self.inner, self.engine.as_ref()))?)
    }

    /// The commit `timestamp_ms` names among the published commits, as
    /// `(version, commit ms)`: the latest at or before it, or with
    /// `at_or_after` the first at or after it -- what a change feed's bounds
    /// resolve to. A time before the first commit (at or before) or after the
    /// latest (at or after) raises ValueError.
    #[pyo3(signature = (timestamp_ms, at_or_after = false))]
    fn version_at(
        &self,
        py: Python<'_>,
        timestamp_ms: i64,
        at_or_after: bool,
    ) -> PyResult<(u64, i64)> {
        let bound = if at_or_after {
            Bound::AtOrAfter
        } else {
            Bound::AtOrBefore
        };
        Ok(py.detach(|| {
            commit_time::version_at(
                &self.inner,
                self.engine.as_ref(),
                timestamp_ms,
                bound,
                HistoryCommitType::Published,
            )
        })?)
    }

    /// Write a checkpoint at this snapshot's version.
    ///
    /// Returns True if a checkpoint was written, False if one already existed.
    /// On a catalog-managed table every commit up to this version must be
    /// published first; the kernel refuses otherwise, because a checkpoint over
    /// unpublished commits would leave a gap in the log for older readers.
    ///
    /// A table carrying a feature that binds only the values of rows written
    /// (CHECK constraints, generated or identity columns, invariants) is
    /// checkpointed from a restated snapshot (`restate::log_writing_snapshot`):
    /// a checkpoint writes no row, and holds the protocol and metadata from
    /// the log, those features included. Its checksum is written with it --
    /// past a table's first checkpoint the commits' own often cannot be.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<bool> {
        let written = py.detach(|| -> Result<bool> {
            let engine = self.engine.as_ref();
            let snapshot = crate::restate::log_writing_snapshot(&self.inner)?;
            // Called directly, not inside runtime::block_on: the engine's
            // executor does its own bridging (see commit::SharedEngine).
            let (result, checkpointed) = snapshot.checkpoint(engine, None)?;
            let written = matches!(result, CheckpointWriteResult::Written);
            if written && !Arc::ptr_eq(&snapshot, &self.inner) {
                crate::checksum::write_at_checkpoint(&checkpointed, engine);
            }
            Ok(written)
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
        operation_parameters = None,
        blind_append = None,
        metadata = None,
        protocol = None,
        constraints_checked = false,
        values_checked = None,
        domain_metadata = None,
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
        operation_parameters: Option<HashMap<String, String>>,
        blind_append: Option<bool>,
        metadata: Option<String>,
        protocol: Option<String>,
        constraints_checked: bool,
        values_checked: Option<Vec<String>>,
        domain_metadata: Option<HashMap<String, String>>,
    ) -> PyResult<u64> {
        let constraints_checked =
            crate::restate::Checked::from_args(constraints_checked, values_checked)?;
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
                commit::CommitInfoPatch {
                    operation_parameters,
                    blind_append,
                    domains: domain_metadata,
                    ..Default::default()
                },
                &crate::restate::Restatement {
                    metadata,
                    protocol,
                    constraints_checked,
                },
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
    #[pyo3(signature = (data, uc = None, constraints_checked = false, values_checked = None))]
    fn write_files(
        &self,
        py: Python<'_>,
        data: PyRecordBatchReader,
        uc: Option<UcCommitConfig>,
        constraints_checked: bool,
        values_checked: Option<Vec<String>>,
    ) -> PyResult<Vec<u8>> {
        let constraints_checked =
            crate::restate::Checked::from_args(constraints_checked, values_checked)?;
        let reader = data.into_reader()?;
        let batches: std::result::Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(NativeError::from)?;
        let result = py.detach(|| {
            commit::write_files(
                self.inner.clone(),
                self.engine.clone(),
                batches,
                uc,
                constraints_checked,
            )
        });
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
        operation_parameters = None,
        blind_append = None,
        constraints_checked = false,
        values_checked = None,
        domain_metadata = None,
        create_template = None,
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
        operation_parameters: Option<HashMap<String, String>>,
        blind_append: Option<bool>,
        constraints_checked: bool,
        values_checked: Option<Vec<String>>,
        domain_metadata: Option<HashMap<String, String>>,
        create_template: Option<Vec<String>>,
    ) -> PyResult<u64> {
        let constraints_checked =
            crate::restate::Checked::from_args(constraints_checked, values_checked)?;
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
                commit::CommitInfoPatch {
                    operation_parameters,
                    blind_append,
                    domains: domain_metadata,
                    create_template: create_template.map(Arc::new),
                    ..Default::default()
                },
                constraints_checked,
            )
        })?;
        Ok(version)
    }

    /// `(version, n)` for each commit after `after` (up to this snapshot)
    /// that adds any of `paths`, `n` being how many. Reads only those commits,
    /// a catalog-managed table's ratified tail included; raises when one of
    /// them can no longer be read.
    fn commits_adding(
        &self,
        py: Python<'_>,
        after: u64,
        paths: Vec<String>,
    ) -> PyResult<Vec<(u64, usize)>> {
        Ok(py.detach(|| {
            crate::landed::commits_adding(&self.inner, self.engine.as_ref(), after, &paths)
        })?)
    }

    /// Delete data files `write_files` wrote that no commit references.
    ///
    /// The caller settles that they are unreferenced (`commits_adding`).
    /// Returns the paths that could not be deleted, with why.
    fn delete_uncommitted(&self, py: Python<'_>, paths: Vec<String>) -> PyResult<Vec<String>> {
        Ok(py.detach(|| crate::landed::delete_uncommitted(&self.inner, &self.engine, &paths))?)
    }

    /// Commit row-level DML as deletion vectors, the way Databricks writes it.
    ///
    /// `deletions` is an Arrow stream with columns `path` and `row_index`: the
    /// rows to delete, addressed as a positional scan (`row_positions=True`)
    /// reports them. `data`, if given, is appended in the same commit -- an
    /// UPDATE's rewritten rows. `whole_files` names files to delete entirely
    /// (every live row) without listing rows. `data_change=False` commits
    /// it as a compaction (OPTIMIZE): every add and remove says the rows did
    /// not change, only the files holding them. Returns `(version, deleted_rows,
    /// deletion_vectors_added, files_removed)`; a DML that changes nothing
    /// commits nothing and returns this snapshot's version. `add_tags` are
    /// written as the `tags` of every add the commit makes (a Z-order's
    /// `ZCUBE_*` tags).
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
        data_change = true,
        operation_parameters = None,
        blind_append = None,
        add_tags = None,
        constraints_checked = false,
        values_checked = None,
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
        data_change: bool,
        operation_parameters: Option<HashMap<String, String>>,
        blind_append: Option<bool>,
        add_tags: Option<HashMap<String, String>>,
        constraints_checked: bool,
        values_checked: Option<Vec<String>>,
    ) -> PyResult<(u64, u64, usize, usize)> {
        let constraints_checked =
            crate::restate::Checked::from_args(constraints_checked, values_checked)?;
        let deletions = deletions.into_reader()?;
        let data = data.map(|d| d.into_reader()).transpose()?;
        let outcome = py.detach(|| -> Result<dml::DmlOutcome> {
            let deletions: std::result::Result<Vec<_>, _> = deletions.collect();
            let deletions = dml::deletions_from_batches(&deletions.map_err(NativeError::from)?)?;
            let data = match data {
                // A compaction's rows are pulled as they are written: its
                // input can be far larger than memory.
                Some(reader) if !data_change => dml::DmlData::Stream(Box::new(
                    reader.map(|batch| batch.map_err(NativeError::from)),
                )),
                Some(reader) => {
                    let batches: std::result::Result<Vec<_>, _> = reader.collect();
                    dml::DmlData::Batches(batches.map_err(NativeError::from)?)
                }
                None => dml::DmlData::Batches(Vec::new()),
            };
            dml::commit_dml(
                self.inner.clone(),
                self.engine.clone(),
                deletions,
                whole_files.unwrap_or_default().into_iter().collect(),
                data,
                uc,
                engine_info,
                operation,
                txn,
                commit_metadata,
                data_change,
                constraints_checked,
                commit::CommitInfoPatch {
                    operation_parameters,
                    blind_append,
                    add_tags,
                    ..Default::default()
                },
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

    /// Commit raw `add`/`remove`/`txn`/`domainMetadata` actions on this
    /// snapshot, through the catalog when `uc` is given; the new version.
    ///
    /// See `commit::commit_actions`: kernel writes the commitInfo, and a
    /// concurrent commit of the next version raises CommitConflictError.
    #[pyo3(signature = (
        actions,
        uc = None,
        engine_info = None,
        operation = None,
        operation_parameters = None,
        commit_metadata = None,
        blind_append = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn commit_actions(
        &self,
        py: Python<'_>,
        actions: Vec<String>,
        uc: Option<UcCommitConfig>,
        engine_info: Option<String>,
        operation: Option<String>,
        operation_parameters: Option<HashMap<String, String>>,
        commit_metadata: Option<HashMap<String, String>>,
        blind_append: bool,
    ) -> PyResult<u64> {
        let version = py.detach(|| {
            commit::commit_actions(
                self.inner.clone(),
                self.engine.clone(),
                uc,
                &actions,
                engine_info,
                operation,
                operation_parameters,
                commit_metadata,
                blind_append,
            )
        })?;
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
