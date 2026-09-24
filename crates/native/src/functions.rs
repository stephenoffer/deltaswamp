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

/// The UC `CreateTableRequest` body for a freshly committed version 0, as JSON.
///
/// Step three of the managed-table creation flow: after staging the table in
/// UC and committing v0 with `uc_required_properties`, send this to the UC
/// `tables` endpoint to finalise registration.
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
            .map_err(|e| NativeError::Invalid(format!("could not serialise the UC request: {e}")))
    })?;
    Ok(body)
}

/// The table properties a UC catalog-managed table must carry in its v0 commit.
#[pyfunction]
pub fn uc_required_properties(uc_table_id: &str) -> HashMap<String, String> {
    delta_kernel_unity_catalog::get_required_properties_for_disk(uc_table_id)
}
