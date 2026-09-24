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

use delta_kernel::snapshot::{Snapshot, SnapshotRef};
use delta_kernel::{Engine, LogPath};
use delta_kernel_default_engine::DefaultEngine;
use pyo3::prelude::*;
use pyo3_arrow::{PyRecordBatchReader, PySchema};
use url::Url;

use crate::commit::{self, SharedEngine, UcCommitConfig};
use crate::error::{NativeError, Result};
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
    fn table_root_url(table_root: &str) -> Result<Url> {
        let normalised = if table_root.ends_with('/') {
            table_root.to_string()
        } else {
            format!("{table_root}/")
        };
        Url::parse(&normalised).or_else(|_| {
            // Bare filesystem paths are a convenience users expect.
            Url::from_directory_path(&normalised).map_err(|_| {
                NativeError::Invalid(format!("cannot interpret {table_root:?} as a table root"))
            })
        })
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

        entries
            .into_iter()
            .map(|(_, filename, last_modified, size)| {
                LogPath::staged_commit(table_root.clone(), &filename, last_modified, size)
                    .map_err(NativeError::from)
            })
            .collect()
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
    ))]
    fn resolve(
        py: Python<'_>,
        table_root: &str,
        options: Option<HashMap<String, String>>,
        version: Option<u64>,
        log_tail: Option<Vec<LogTailEntry>>,
        max_catalog_version: Option<u64>,
    ) -> PyResult<Self> {
        let url = Self::table_root_url(table_root)?;
        let options = options.unwrap_or_default();

        // Log resolution does real I/O, so release the GIL for it.
        let (inner, engine) = py.detach(|| -> Result<(SnapshotRef, SharedEngine)> {
            let object_store = store::build_store(&url, &options)?;
            let engine: SharedEngine = Arc::new(DefaultEngine::builder(object_store).build());

            let mut builder = Snapshot::builder_for(url.as_str());
            if let Some(v) = version {
                builder = builder.at_version(v);
            }
            if let Some(entries) = log_tail {
                builder = builder.with_log_tail(Self::build_log_tail(&url, entries)?);
            }
            if let Some(v) = max_catalog_version {
                builder = builder.with_max_catalog_version(v);
            }

            let snapshot =
                runtime::block_on(async { builder.build(engine.as_ref() as &dyn Engine) })?;
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
    /// recognise -- the Python layer decides what to do with them, because an
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
    #[pyo3(signature = (columns = None))]
    fn scan(&self, py: Python<'_>, columns: Option<Vec<String>>) -> PyResult<PyRecordBatchReader> {
        let reader = py.detach(|| -> Result<KernelBatchReader> {
            let mut builder = self.inner.clone().scan_builder();

            if let Some(columns) = columns {
                let full = self.inner.schema();
                let projected = full.project(&columns)?;
                builder = builder.with_schema(projected);
            }

            let scan = builder.build()?;
            KernelBatchReader::try_new(&scan, self.engine.clone() as Arc<dyn Engine>)
        })?;

        Ok(PyRecordBatchReader::new(Box::new(reader)))
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
        let batches: std::result::Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(NativeError::from)?;

        let version = py.detach(|| {
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
    if !table_root.contains("://") || table_root.starts_with("file://") {
        let path = table_root.strip_prefix("file://").unwrap_or(table_root);
        std::fs::create_dir_all(path).map_err(|e| {
            NativeError::Invalid(format!("could not create table directory {path:?}: {e}"))
        })?;
    }

    let url = PySnapshot::table_root_url(table_root)?;
    let arrow_schema = schema.into_inner();

    let version = py.detach(|| -> Result<u64> {
        let object_store = store::build_store(&url, &options.unwrap_or_default())?;
        let engine: SharedEngine = Arc::new(DefaultEngine::builder(object_store).build());
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
    })?;
    Ok(version)
}
