//! Module-level functions: the operations that do not start from a snapshot.
//!
//! Each one either predates a snapshot (a raw commit that creates a version),
//! spans several (a change feed over a version range), or is a pure helper for
//! the Unity Catalog managed-table creation flow.

use std::collections::HashMap;

use delta_kernel::snapshot::Snapshot;
use delta_kernel::Engine;
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatchReader;

use crate::changes::{self, Range};
use crate::commit;
use crate::error::{NativeError, Result};
use crate::runtime;
use crate::snapshot::PySnapshot;
use crate::store;

/// Read the change data feed over a version (or timestamp) range.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    table_root,
    options = None,
    start_version = None,
    end_version = None,
    columns = None,
    predicate = None,
    start_timestamp_ms = None,
    end_timestamp_ms = None,
))]
pub fn table_changes(
    py: Python<'_>,
    table_root: &str,
    options: Option<HashMap<String, String>>,
    start_version: Option<u64>,
    end_version: Option<u64>,
    columns: Option<Vec<String>>,
    predicate: Option<String>,
    start_timestamp_ms: Option<i64>,
    end_timestamp_ms: Option<i64>,
) -> PyResult<PyRecordBatchReader> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    let reader = py.detach(|| {
        changes::table_changes(
            &url,
            &options,
            Range {
                start_version,
                end_version,
                start_timestamp_ms,
                end_timestamp_ms,
            },
            columns,
            predicate.as_deref(),
        )
    })?;
    Ok(PyRecordBatchReader::new(Box::new(reader)))
}

/// The version range a change feed's bounds name, as `table_changes` reads it.
///
/// A start timestamp is the first commit at or after it, an end timestamp the
/// latest at or before it, both by commit times as Delta assigns them; either
/// after the latest commit raises ValueError, as Databricks refuses it
/// (DELTA_TIMESTAMP_GREATER_THAN_COMMIT). For an engine that reads the feed
/// itself but resolves timestamps by raw file times.
#[pyfunction]
#[pyo3(signature = (
    table_root,
    options = None,
    start_version = None,
    end_version = None,
    start_timestamp_ms = None,
    end_timestamp_ms = None,
))]
pub fn feed_versions(
    py: Python<'_>,
    table_root: &str,
    options: Option<HashMap<String, String>>,
    start_version: Option<u64>,
    end_version: Option<u64>,
    start_timestamp_ms: Option<i64>,
    end_timestamp_ms: Option<i64>,
) -> PyResult<(u64, Option<u64>)> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    let range = py.detach(|| {
        let store = store::build_store(&url, &options)?;
        let engine = commit::new_engine(store);
        changes::resolve_range(
            &url,
            &engine,
            Range {
                start_version,
                end_version,
                start_timestamp_ms,
                end_timestamp_ms,
            },
        )
    })?;
    Ok(range)
}

/// Write `actions` verbatim as commit `version`, put-if-absent.
#[pyfunction]
#[pyo3(signature = (table_root, version, actions, options = None))]
pub fn commit_raw(
    py: Python<'_>,
    table_root: &str,
    version: u64,
    actions: Vec<String>,
    options: Option<HashMap<String, String>>,
) -> PyResult<u64> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    let committed = py.detach(|| commit::commit_raw(&url, &options, version, &actions))?;
    Ok(committed)
}

/// Whether the object store under `table_root` honours put-if-absent.
///
/// Writes (and deletes) a sentinel under `_delta_log/`; see
/// `store::probe_put_if_absent`.
#[pyfunction]
#[pyo3(signature = (table_root, options = None))]
pub fn probe_put_if_absent(
    py: Python<'_>,
    table_root: &str,
    options: Option<HashMap<String, String>>,
) -> PyResult<bool> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    Ok(py.detach(|| store::probe_put_if_absent(&url, &options))?)
}

/// Refuse retry storage options (`max_retries`, `retry_timeout`,
/// `backoff_config.*`) that either engine would refuse or panic on.
///
/// `connect()` calls this, so a value delta-rs rejects is refused before any
/// engine is chosen, rather than accepted by the kernel's reads and failing
/// the first call routed to delta-rs.
#[pyfunction]
pub fn validate_retry_options(options: HashMap<String, String>) -> PyResult<()> {
    store::retry_config(&options)?;
    Ok(())
}

/// The UC `CreateTableRequest` body for a freshly committed version 0, as JSON.
///
/// Step three of the managed-table creation flow: after staging the table in
/// UC and committing v0 with `uc_required_properties`, send this to the UC
/// `tables` endpoint to finalize registration.
#[pyfunction]
#[pyo3(signature = (table_root, table_name, options = None))]
pub fn uc_create_table_request(
    py: Python<'_>,
    table_root: &str,
    table_name: &str,
    options: Option<HashMap<String, String>>,
) -> PyResult<String> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    let body = py.detach(|| -> Result<String> {
        let engine = commit::new_engine(store::build_store(&url, &options)?);
        let engine_ref = engine.as_ref() as &dyn Engine;
        // A catalog-managed table cannot be loaded without a trusted version
        // cap, and any other table refuses one. Version 0 is the only version
        // either can have here, so try plain first and cap at 0 if required.
        let build = |capped: bool| {
            runtime::block_on(async {
                let builder = Snapshot::builder_for(url.as_str()).at_version(0);
                let builder = if capped {
                    builder.with_max_catalog_version(0)
                } else {
                    builder
                };
                builder.build(engine_ref)
            })
        };
        let snapshot = match build(false) {
            Err(e) if e.to_string().contains("Max catalog version is required") => build(true)?,
            other => other?,
        };
        let request = delta_kernel_unity_catalog::build_uc_create_table_request(
            &snapshot, engine_ref, table_name,
        )?;
        serde_json::to_string(&request)
            .map_err(|e| NativeError::Invalid(format!("could not serialize the UC request: {e}")))
    })?;
    Ok(body)
}

/// The table properties a UC catalog-managed table must carry in its v0 commit.
#[pyfunction]
pub fn uc_required_properties(uc_table_id: &str) -> HashMap<String, String> {
    delta_kernel_unity_catalog::get_required_properties_for_disk(uc_table_id)
}

/// A deletion-vector descriptor (Delta-protocol JSON, as `files()` reports
/// it) of a file in the table at `table_root`, made absolute.
///
/// What a shallow clone's add actions carry: a relative (`u`) vector names a
/// file under the *source* table, which the clone's root would resolve
/// elsewhere, so it becomes a `p` descriptor with the vector's full URL. An
/// inline or already-absolute one is returned as it is.
#[pyfunction]
pub fn absolute_deletion_vector(table_root: &str, descriptor: &str) -> PyResult<String> {
    use delta_kernel::actions::deletion_vector::{
        DeletionVectorDescriptor, DeletionVectorStorageType,
    };

    let root = PySnapshot::table_root_url(table_root)?;
    let mut value: serde_json::Value = serde_json::from_str(descriptor)
        .map_err(|e| NativeError::Invalid(format!("not a deletion vector descriptor: {e}")))?;
    let field = |name: &str| value.get(name).cloned().unwrap_or(serde_json::Value::Null);
    if field("storageType").as_str() != Some("u") {
        return Ok(descriptor.to_string());
    }
    let path = field("pathOrInlineDv")
        .as_str()
        .unwrap_or_default()
        .to_string();
    let offset = field("offset").as_i64().map(|o| o as i32);
    let size = field("sizeInBytes").as_i64().unwrap_or_default() as i32;
    let cardinality = field("cardinality").as_i64().unwrap_or_default();
    let dv = DeletionVectorDescriptor::try_new(
        DeletionVectorStorageType::PersistedRelative,
        path,
        offset,
        size,
        cardinality,
    )
    .map_err(NativeError::from)?;
    let absolute = dv
        .absolute_path(&root)
        .map_err(NativeError::from)?
        .ok_or_else(|| NativeError::Invalid("a relative vector has a path".to_string()))?;
    value["storageType"] = "p".into();
    value["pathOrInlineDv"] = absolute.to_string().into();
    Ok(value.to_string())
}

/// Copy `paths` (relative to both roots, URL-encoded as the log stores them)
/// from the table at `source_root` to `target_root`; returns bytes copied.
///
/// A deep clone's data and deletion-vector files. Read and written through
/// each root's own store and options, so the two may be different accounts
/// or clouds. Each object is held in memory while it is copied.
#[pyfunction]
#[pyo3(signature = (source_root, target_root, paths, source_options = None, target_options = None))]
pub fn copy_objects(
    py: Python<'_>,
    source_root: &str,
    target_root: &str,
    paths: Vec<String>,
    source_options: Option<HashMap<String, String>>,
    target_options: Option<HashMap<String, String>>,
) -> PyResult<u64> {
    use delta_kernel::object_store::path::Path;
    use delta_kernel::object_store::{ObjectStoreExt, PutPayload};

    let source = PySnapshot::table_root_url(source_root)?;
    let target = PySnapshot::table_root_url(target_root)?;
    let copied = py.detach(|| -> Result<u64> {
        let from = store::build_store(&source, &source_options.unwrap_or_default())?;
        let to = store::build_store(&target, &target_options.unwrap_or_default())?;
        let mut total = 0u64;
        for relative in &paths {
            let locate = |root: &url::Url| -> Result<Path> {
                let url = root
                    .join(relative)
                    .map_err(|e| NativeError::Invalid(format!("bad path {relative:?}: {e}")))?;
                Ok(Path::from_url_path(url.path())
                    .map_err(delta_kernel::object_store::Error::from)?)
            };
            let (src, dst) = (locate(&source)?, locate(&target)?);
            let bytes = runtime::block_on(async { from.get(&src).await?.bytes().await })?;
            total += bytes.len() as u64;
            runtime::block_on(async { to.put(&dst, PutPayload::from(bytes)).await })?;
        }
        Ok(total)
    })?;
    Ok(copied)
}
