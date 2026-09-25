//! The change data feed, read through the kernel.
//!
//! `TableChanges` resolves a version range, checks CDF was enabled at both
//! ends and that the schema did not change across it, and its scan yields rows
//! with `_change_type`, `_commit_version` and `_commit_timestamp` appended.
//! Like the table scan, the kernel applies deletion vectors itself (including
//! the add/remove DV pairs that encode a row-level update) and preserves file
//! row order; this module only hands its batches onward.
//!
//! Timestamp bounds are converted with the history manager using *published*
//! commits: CDF needs the commits in the range to exist in the log, not a
//! checkpoint to rebuild the table from. Asking for "recreatable" versions
//! instead would silently move the start forward to the earliest checkpoint and
//! drop the changes before it.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::history_manager::{first_version_after, latest_version_as_of, HistoryCommitType};
use delta_kernel::snapshot::Snapshot;
use delta_kernel::table_changes::TableChanges;
use delta_kernel::Engine;
use url::Url;

use crate::commit::SharedEngine;
use crate::error::{NativeError, Result};
use crate::predicate::parse_predicate;
use crate::runtime;
use crate::scan::KernelBatchReader;

/// The metadata columns every change-feed row carries.
pub const CDF_COLUMNS: [&str; 3] = ["_change_type", "_commit_version", "_commit_timestamp"];

/// A change-feed request's version range, as given by the caller.
pub struct Range {
    pub start_version: Option<u64>,
    pub end_version: Option<u64>,
    pub start_timestamp_ms: Option<i64>,
    pub end_timestamp_ms: Option<i64>,
}

/// Resolve timestamp bounds to versions, refusing ambiguous combinations.
fn resolve_range(url: &Url, engine: &SharedEngine, range: Range) -> Result<(u64, Option<u64>)> {
    if range.start_version.is_some() && range.start_timestamp_ms.is_some() {
        return Err(NativeError::Invalid(
            "pass start_version or start_timestamp_ms, not both".to_string(),
        ));
    }
    if range.end_version.is_some() && range.end_timestamp_ms.is_some() {
        return Err(NativeError::Invalid(
            "pass end_version or end_timestamp_ms, not both".to_string(),
        ));
    }
    if range.start_timestamp_ms.is_none() && range.end_timestamp_ms.is_none() {
        return Ok((range.start_version.unwrap_or(0), range.end_version));
    }

    let engine_ref = engine.as_ref() as &dyn Engine;
    let latest =
        runtime::block_on(async { Snapshot::builder_for(url.as_str()).build(engine_ref) })?;
    let start = match range.start_timestamp_ms {
        Some(ts) => {
            first_version_after(&latest, engine_ref, ts, HistoryCommitType::Published)
                .map_err(|e| {
                    NativeError::Invalid(format!(
                        "no commit at or after start timestamp {ts} ms: {e}"
                    ))
                })?
                .version
        }
        None => range.start_version.unwrap_or(0),
    };
    let end = match range.end_timestamp_ms {
        Some(ts) => Some(
            latest_version_as_of(&latest, engine_ref, ts, HistoryCommitType::Published)
                .map_err(|e| {
                    NativeError::Invalid(format!(
                        "no commit at or before end timestamp {ts} ms: {e}"
                    ))
                })?
                .version,
        ),
        None => range.end_version,
    };
    if let Some(end) = end {
        if end < start {
            return Err(NativeError::Invalid(format!(
                "the requested range resolves to start version {start} after end version {end}; \
                 no commit falls inside it"
            )));
        }
    }
    Ok((start, end))
}

/// Build a change-feed reader for `url` over the requested range.
pub fn table_changes(
    url: &Url,
    options: &HashMap<String, String>,
    range: Range,
    columns: Option<Vec<String>>,
    predicate: Option<&str>,
) -> Result<KernelBatchReader> {
    let store = crate::store::build_store(url, options)?;
    let engine = crate::commit::new_engine(store);
    let (start, end) = resolve_range(url, &engine, range)?;

    let changes = runtime::block_on(async {
        TableChanges::try_new(url.clone(), engine.as_ref() as &dyn Engine, start, end)
    })?;

    let predicate = parse_predicate(predicate, changes.schema())?;
    let projected = match columns {
        Some(columns) => {
            let mut columns = if columns.is_empty() {
                columns
            } else {
                crate::scan::resolve_columns(changes.schema(), &columns)?
            };
            // The change metadata columns are what make a row a *change*;
            // always keep them.
            for c in CDF_COLUMNS {
                if !columns.iter().any(|x| x == c) {
                    columns.push(c.to_string());
                }
            }
            changes.schema().project(&columns)?
        }
        None => Arc::new(changes.schema().clone()),
    };
    // A projection of partition and change columns only (`columns=["p"]`,
    // `columns=[]`) leaves the Parquet read schema empty, and kernel's reader
    // panics on that, killing the shared I/O executor. A row-index column
    // keeps every read non-empty; it is dropped from the output below.
    let projected = Some(Arc::new(projected.add_metadata_column(
        crate::scan::ROW_COUNT_COLUMN,
        delta_kernel::schema::MetadataColumnSpec::RowIndex,
    )?));

    let scan = Arc::new(changes)
        .scan_builder()
        .with_schema(projected)
        .with_predicate(predicate.map(Arc::new))
        .build()?;
    let schema = scan.logical_schema().clone();
    let iter = scan.execute(engine.clone() as Arc<dyn Engine>)?;
    Ok(KernelBatchReader::from_parts(schema.as_ref(), iter)?
        .without_column(crate::scan::ROW_COUNT_COLUMN))
}
