//! Change data files for a DML commit on a table with the change data feed.
//!
//! A commit that removes or rewrites data on such a table must say what
//! changed: an UPDATE's rows before and after (`update_preimage`,
//! `update_postimage`), a copy-on-write DELETE's deleted rows, a MERGE's
//! inserts too. Readers take those from the commit's `cdc` actions, which name
//! Parquet files under `_change_data/` holding the changed rows in the table's
//! physical layout plus a `_change_type` column -- and then ignore the
//! commit's adds and removes for the feed. delta-kernel 0.28 writes no such
//! files, and refuses a commit that both adds and removes on a change-feed
//! table for that reason; these are written here, and the commit carries them
//! (see `dml::commit_dml`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use delta_kernel::engine::arrow_conversion::TryFromArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::ObjectStoreExt;
use delta_kernel::parquet::basic::Compression;
use delta_kernel::schema::{SchemaRef, StructType};
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::transaction::{BoundWriteContext, Transaction};
use delta_kernel::{Engine, FileMeta};
use delta_kernel_default_engine::parquet::DataFileMetadata;
use delta_kernel_default_engine::stats::collect_stats;

use crate::commit::SharedEngine;
use crate::error::{NativeError, Result};
use crate::{partition, runtime};

/// The column a change file names each row's kind of change in.
pub const CHANGE_TYPE: &str = "_change_type";

/// The kinds of change a change file may hold.
const CHANGE_TYPES: &[&str] = &["insert", "delete", "update_preimage", "update_postimage"];

/// Write `batches` -- the table's logical columns plus [`CHANGE_TYPE`] -- as
/// change files under `_change_data/`, one per batch and partition.
///
/// Returns each file's `cdc` action, as one line of log JSON; the relative
/// path of every file written goes into `written`, for the caller to take
/// back out if the commit does not land.
pub fn write_change_files(
    snapshot: &SnapshotRef,
    engine: &SharedEngine,
    transaction: &Transaction,
    batches: Vec<RecordBatch>,
    codec: Compression,
    written: &mut Vec<String>,
) -> Result<Vec<String>> {
    let partition_columns = snapshot
        .table_configuration()
        .logical_partition_columns()
        .to_vec();
    let table_schema: SchemaRef = snapshot.schema();
    let write_state = transaction.write_state()?;
    let mut actions = Vec::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let (rows, kinds) = take_change_type(batch)?;
        // Aligned by name, typed as the table's, as data rows are; the
        // change types keep their row order through it.
        let rows = partition::conform_to_table(&rows, table_schema.as_ref(), &partition_columns)?;
        let rows = with_column(&rows, CHANGE_TYPE, kinds)?;
        if partition_columns.is_empty() {
            let context = write_state.unpartitioned_write_context()?;
            actions.push(write_one(snapshot, engine, rows, &context, codec, written)?);
        } else {
            for group in
                partition::split_by_partition(&rows, &partition_columns, table_schema.as_ref())?
            {
                let context = write_state.partitioned_write_context(group.values)?;
                actions.push(write_one(
                    snapshot, engine, group.data, &context, codec, written,
                )?);
            }
        }
    }
    Ok(actions)
}

/// `batch` without its [`CHANGE_TYPE`] column, and that column as text,
/// checked to hold only the protocol's kinds of change.
fn take_change_type(batch: RecordBatch) -> Result<(RecordBatch, ArrayRef)> {
    let mut rows = batch;
    let index = rows.schema().index_of(CHANGE_TYPE).map_err(|_| {
        NativeError::Invalid(format!(
            "change rows need a {CHANGE_TYPE} column naming each row's kind of change"
        ))
    })?;
    let kinds = arrow::compute::cast(rows.column(index), &DataType::Utf8)?;
    let text = kinds.as_string::<i32>();
    for i in 0..text.len() {
        if text.is_null(i) || !CHANGE_TYPES.contains(&text.value(i)) {
            return Err(NativeError::Invalid(format!(
                "{CHANGE_TYPE} must be one of {CHANGE_TYPES:?}, not {:?}",
                (!text.is_null(i)).then(|| text.value(i))
            )));
        }
    }
    rows.remove_column(index);
    Ok((rows, kinds))
}

/// `batch` with `column` appended as a non-null text column named `name`.
fn with_column(batch: &RecordBatch, name: &str, column: ArrayRef) -> Result<RecordBatch> {
    let mut fields: Vec<arrow::datatypes::FieldRef> = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(name, DataType::Utf8, false)));
    let mut columns = batch.columns().to_vec();
    columns.push(column);
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            fields,
            batch.schema().metadata().clone(),
        )),
        columns,
    )?)
}

/// Write one partition's change rows (logical, no partition columns, plus
/// [`CHANGE_TYPE`]) and return its `cdc` action.
fn write_one(
    snapshot: &SnapshotRef,
    engine: &SharedEngine,
    rows: RecordBatch,
    context: &BoundWriteContext,
    codec: Compression,
    written: &mut Vec<String>,
) -> Result<String> {
    let (logical, kinds) = take_change_type(rows)?;
    // The table's logical-to-physical transform, as a data file gets it:
    // physical names and field ids under column mapping. The change type is
    // not a table column; it goes in after, under its own name.
    let input_schema = StructType::try_from_arrow(logical.schema().as_ref())?;
    let evaluator = engine.evaluation_handler().new_expression_evaluator(
        Arc::new(input_schema),
        context.logical_to_physical(),
        context.physical_schema().clone().into(),
    )?;
    let physical = evaluator
        .evaluate(&ArrowEngineData::new(logical))?
        .try_into_record_batch()?;
    let stats = collect_stats(
        &physical,
        context.stats_columns(),
        context.physical_schema().as_ref(),
    )?;
    let physical = with_column(&physical, CHANGE_TYPE, kinds)?;
    let buffer = crate::writer::encode(&physical, codec)?;
    let size = u64::try_from(buffer.len())
        .map_err(|_| NativeError::Invalid("a change file too large to size".to_string()))?;

    let root = snapshot.table_root();
    // The data file's directory (its partition, or random prefix) under
    // `_change_data/`, as Spark lays change files out.
    let relative = context
        .write_dir()
        .path()
        .strip_prefix(root.path())
        .unwrap_or("")
        .to_string();
    let url = root
        .join("_change_data/")?
        .join(&relative)?
        .join(&format!("cdc-{}.c000.snappy.parquet", uuid::Uuid::new_v4()))?;
    let store = engine
        .get_object_store_for_url(&url)
        .ok_or_else(|| NativeError::Invalid(format!("no object store is registered for {url}")))?;
    let location = Path::from_url_path(url.path())
        .map_err(|e| NativeError::Invalid(format!("bad change file path {url}: {e}")))?;
    runtime::block_on(async { store.put(&location, buffer.into()).await })?;
    let path = url
        .as_str()
        .strip_prefix(root.as_str())
        .unwrap_or(url.as_str())
        .to_string();
    written.push(path);

    // Partition values as a data file of this partition records them: by
    // physical name, serialized as the protocol says.
    let metadata = delta_kernel_default_engine::build_add_file_metadata(
        DataFileMetadata::new(FileMeta::new(url.clone(), 0, size), stats),
        context,
    )?;
    let metadata = metadata.try_into_record_batch()?;
    let partition_values = partition_values_json(&metadata)?;
    let action = serde_json::json!({
        "cdc": {
            "path": written.last().cloned().unwrap_or_default(),
            "partitionValues": partition_values,
            "size": size,
            "dataChange": false,
        }
    });
    Ok(action.to_string())
}

/// The `partitionValues` of the single add-file metadata row in `batch`.
fn partition_values_json(batch: &RecordBatch) -> Result<serde_json::Value> {
    let mut out = serde_json::Map::new();
    let Some(column) = batch.column_by_name("partitionValues") else {
        return Ok(serde_json::Value::Object(out));
    };
    let Some(map) = column.as_map_opt() else {
        return Ok(serde_json::Value::Object(out));
    };
    if map.is_empty() || map.is_null(0) {
        return Ok(serde_json::Value::Object(out));
    }
    let entries = map.value(0);
    let keys = arrow::compute::cast(entries.column(0), &DataType::Utf8)?;
    let values = arrow::compute::cast(entries.column(1), &DataType::Utf8)?;
    let keys: &StringArray = keys.as_string::<i32>();
    let values: &StringArray = values.as_string::<i32>();
    for i in 0..keys.len() {
        let value = if values.is_null(i) {
            serde_json::Value::Null
        } else {
            values.value(i).into()
        };
        out.insert(keys.value(i).to_string(), value);
    }
    Ok(serde_json::Value::Object(out))
}
