//! Listing a snapshot's live data files as an Arrow table.
//!
//! Kernel's convenience visitor (`ScanMetadata::visit_scan_files`) throws away
//! exactly what a caller planning its own reads needs: it parses `stats` down to
//! `numRecords` and keeps the deletion-vector descriptor crate-private. So we
//! read the scan-metadata batches ourselves, as Arrow, and apply the selection
//! vector with `filter`. The kernel's column names (`path`, `size`,
//! `modificationTime`, `stats`, `deletionVector`,
//! `fileConstantValues.partitionValues`) are its documented scan-row schema.
//!
//! This is a listing, not a read: nothing here touches row order or applies a
//! deletion vector. It reports the descriptor so a caller that reads the file
//! itself knows it *must* apply one.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Int64Array, Int64Builder, RecordBatch, StringArray,
    StringBuilder, StructArray,
};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use delta_kernel::engine::arrow_data::EngineDataArrowExt;
use delta_kernel::expressions::Predicate;
use delta_kernel::scan::StatsOptions;
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::Engine;

use crate::error::{NativeError, Result};

fn missing(name: &str) -> NativeError {
    NativeError::Invalid(format!(
        "kernel scan metadata has no {name:?} column; the pinned kernel's scan-row \
         schema changed, so the file listing must be updated"
    ))
}

/// One row per live data file, with optional predicate-based file skipping.
///
/// With `tags`, a column `tags` holds each add's tags as a JSON object. With
/// `scan_rows`, a last column `scan_row` holds each file's kernel scan row as
/// JSON (see [`scan_row_json`]): what a worker reads the file by.
pub fn list_files(
    snapshot: SnapshotRef,
    engine: &dyn Engine,
    predicate: Option<Predicate>,
    tags: bool,
    scan_rows: bool,
) -> Result<RecordBatch> {
    // Struct stats are requested only so the kernel reads a checkpoint's
    // `stats_parsed` and re-serializes it into `stats`. With the default
    // (JSON only) a checkpoint written with writeStatsAsJson=false -- every
    // Databricks managed table -- listed every file with no statistics.
    // A scan plan (`scan_rows`) needs no statistics of its own, and parsing
    // every file's into structs was most of planning's memory: it keeps the
    // log's JSON only. File skipping is the same either way.
    let stats = if scan_rows {
        StatsOptions::json_only()
    } else {
        StatsOptions::all()
    };
    let scan = snapshot
        .scan_builder()
        .with_predicate(predicate.map(Arc::new))
        .with_stats(stats)
        .without_row_transforms()
        .build()?;

    let mut batches = Vec::new();
    for metadata in scan.scan_metadata(engine)? {
        let (data, selection) = metadata?.scan_files.into_parts();
        let batch = data.try_into_record_batch()?;
        // A selection vector shorter than the data means "the rest are selected".
        let mask: BooleanArray = (0..batch.num_rows())
            .map(|i| Some(selection.get(i).copied().unwrap_or(true)))
            .collect();
        let selected = filter_record_batch(&batch, &mask)?;
        if selected.num_rows() > 0 {
            batches.push(project(&selected, tags, scan_rows)?);
        }
    }

    let schema = schema_for(tags, scan_rows);
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }
    Ok(arrow::compute::concat_batches(&schema, &batches)?)
}

fn partition_values_field() -> Field {
    Field::new_map(
        "partition_values",
        "key_value",
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, true),
        false,
        true,
    )
}

pub fn output_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("size", DataType::Int64, false),
        Field::new("modification_time", DataType::Int64, true),
        partition_values_field(),
        Field::new("stats", DataType::Utf8, true),
        Field::new("deletion_vector", DataType::Utf8, true),
        Field::new("num_records", DataType::Int64, true),
    ]))
}

fn schema_for(tags: bool, scan_rows: bool) -> Arc<Schema> {
    let schema = output_schema();
    if !tags && !scan_rows {
        return schema;
    }
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    if tags {
        fields.push(Field::new("tags", DataType::Utf8, true));
    }
    if scan_rows {
        fields.push(Field::new("scan_row", DataType::Utf8, false));
    }
    Arc::new(Schema::new(fields))
}

/// Reshape one selected scan-metadata batch into the listing schema.
fn project(batch: &RecordBatch, with_tags: bool, scan_rows: bool) -> Result<RecordBatch> {
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .cloned()
            .ok_or_else(|| missing(name))
    };

    let path = arrow::compute::cast(&column("path")?, &DataType::Utf8)?;
    let size = arrow::compute::cast(&column("size")?, &DataType::Int64)?;
    let modification_time = arrow::compute::cast(&column("modificationTime")?, &DataType::Int64)?;
    let stats = arrow::compute::cast(&column("stats")?, &DataType::Utf8)?;

    let constants = column("fileConstantValues")?;
    let constants = constants
        .as_struct_opt()
        .ok_or_else(|| missing("fileConstantValues"))?;
    let partition_values = constants
        .column_by_name("partitionValues")
        .ok_or_else(|| missing("fileConstantValues.partitionValues"))?;
    let partition_values = normalize_map(partition_values)?;

    let dv = column("deletionVector")?;
    let dv = dv
        .as_struct_opt()
        .ok_or_else(|| missing("deletionVector"))?;
    let deletion_vector = deletion_vector_json(dv)?;

    let stats_strings = stats.as_string::<i32>();
    let num_records = num_records(stats_strings);

    let mut columns: Vec<ArrayRef> = vec![
        path,
        size,
        modification_time,
        partition_values,
        stats,
        Arc::new(deletion_vector),
        Arc::new(num_records),
    ];
    if with_tags {
        let tags = match constants.column_by_name("tags") {
            Some(tags) => tags_json(tags)?,
            None => StringArray::new_null(batch.num_rows()),
        };
        columns.push(Arc::new(tags));
    }
    if scan_rows {
        columns.push(Arc::new(scan_row_json(batch)?));
    }
    Ok(RecordBatch::try_new(schema_for(with_tags, scan_rows), columns)?)
}

/// Each selected scan row as a JSON object in the kernel's scan-row schema,
/// without its statistics: what `scan::PlannedFiles` hands back to the kernel
/// on a worker. Statistics are left out because the plan already skipped by
/// them (a worker's scan keeps a file with none), and they are most of a
/// row's size; the other fields decide how the file reads.
pub fn scan_row_json(batch: &RecordBatch) -> Result<StringArray> {
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .cloned()
            .ok_or_else(|| missing(name))
    };
    let path = arrow::compute::cast(&column("path")?, &DataType::Utf8)?;
    let path = path.as_string::<i32>();
    let size = arrow::compute::cast(&column("size")?, &DataType::Int64)?;
    let size = size.as_primitive::<Int64Type>();
    let modified = arrow::compute::cast(&column("modificationTime")?, &DataType::Int64)?;
    let modified = modified.as_primitive::<Int64Type>();
    let dv = column("deletionVector")?;
    let dv = deletion_vector_json(
        dv.as_struct_opt()
            .ok_or_else(|| missing("deletionVector"))?,
    )?;
    let constants = column("fileConstantValues")?;
    let constants = constants
        .as_struct_opt()
        .ok_or_else(|| missing("fileConstantValues"))?;
    let field = |name: &str| {
        constants
            .column_by_name(name)
            .cloned()
            .ok_or_else(|| missing(&format!("fileConstantValues.{name}")))
    };
    let partitions = normalize_map(&field("partitionValues")?)?;
    let base_row_id = arrow::compute::cast(&field("baseRowId")?, &DataType::Int64)?;
    let base_row_id = base_row_id.as_primitive::<Int64Type>();
    let commit_version =
        arrow::compute::cast(&field("defaultRowCommitVersion")?, &DataType::Int64)?;
    let commit_version = commit_version.as_primitive::<Int64Type>();
    let mut out = StringBuilder::new();
    for i in 0..batch.num_rows() {
        let optional = |array: &Int64Array| -> serde_json::Value {
            if array.is_null(i) {
                serde_json::Value::Null
            } else {
                array.value(i).into()
            }
        };
        let deletion_vector = if dv.is_null(i) {
            serde_json::Value::Null
        } else {
            serde_json::from_str(dv.value(i))
                .map_err(|e| NativeError::Invalid(format!("bad deletion vector: {e}")))?
        };
        let row = serde_json::json!({
            "path": path.value(i),
            "size": size.value(i),
            "modificationTime": optional(modified),
            "deletionVector": deletion_vector,
            "fileConstantValues": {
                "partitionValues": map_json(&partitions, i)
                    .unwrap_or_else(|| serde_json::json!({})),
                "baseRowId": optional(base_row_id),
                "defaultRowCommitVersion": optional(commit_version),
            },
        });
        out.append_value(row.to_string());
    }
    Ok(out.finish())
}

/// Each add's `tags` map as a JSON object (null where it has none): what an
/// incremental Z-order reads its `ZCUBE_*` tags from.
fn tags_json(array: &ArrayRef) -> Result<StringArray> {
    let map = array
        .as_map_opt()
        .ok_or_else(|| missing("fileConstantValues.tags (as a map)"))?;
    let keys = arrow::compute::cast(map.keys(), &DataType::Utf8)?;
    let values = arrow::compute::cast(map.values(), &DataType::Utf8)?;
    let (keys, values) = (keys.as_string::<i32>(), values.as_string::<i32>());
    let mut out = StringBuilder::new();
    for i in 0..map.len() {
        if map.is_null(i) {
            out.append_null();
            continue;
        }
        let (start, end) = (
            map.value_offsets()[i] as usize,
            map.value_offsets()[i + 1] as usize,
        );
        let mut object = serde_json::Map::new();
        for j in start..end {
            let value = if values.is_null(j) {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(values.value(j).to_string())
            };
            object.insert(keys.value(j).to_string(), value);
        }
        out.append_value(serde_json::Value::Object(object).to_string());
    }
    Ok(out.finish())
}

/// Give the map column our field names, so the output schema is stable no
/// matter how kernel names its map entries internally.
fn normalize_map(array: &ArrayRef) -> Result<ArrayRef> {
    let map = array
        .as_map_opt()
        .ok_or_else(|| missing("fileConstantValues.partitionValues (as a map)"))?;
    let DataType::Map(entries_field, _) = partition_values_field().data_type().clone() else {
        unreachable!("partition_values_field is a map");
    };
    let DataType::Struct(fields) = entries_field.data_type().clone() else {
        unreachable!("map entries are a struct");
    };
    let keys = arrow::compute::cast(map.keys(), &DataType::Utf8)?;
    let values = arrow::compute::cast(map.values(), &DataType::Utf8)?;
    let entries = StructArray::try_new(fields, vec![keys, values], None)?;
    let rebuilt = arrow::array::MapArray::try_new(
        entries_field,
        map.offsets().clone(),
        entries,
        map.nulls().cloned(),
        false,
    )?;
    Ok(Arc::new(rebuilt))
}

/// Render each deletion-vector descriptor as the Delta-protocol JSON object.
fn deletion_vector_json(dv: &StructArray) -> Result<StringArray> {
    let str_col = |name: &str| -> Result<StringArray> {
        let col = dv
            .column_by_name(name)
            .ok_or_else(|| missing(&format!("deletionVector.{name}")))?;
        Ok(arrow::compute::cast(col, &DataType::Utf8)?
            .as_string::<i32>()
            .clone())
    };
    let int_col = |name: &str| -> Result<Int64Array> {
        let col = dv
            .column_by_name(name)
            .ok_or_else(|| missing(&format!("deletionVector.{name}")))?;
        Ok(arrow::compute::cast(col, &DataType::Int64)?
            .as_primitive::<Int64Type>()
            .clone())
    };
    let storage_type = str_col("storageType")?;
    let path_or_inline = str_col("pathOrInlineDv")?;
    let offset = int_col("offset")?;
    let size_in_bytes = int_col("sizeInBytes")?;
    let cardinality = int_col("cardinality")?;

    let mut out = StringBuilder::new();
    for i in 0..dv.len() {
        // A null struct, or a struct whose required storageType is null, means
        // the file has no deletion vector.
        if dv.is_null(i) || storage_type.is_null(i) {
            out.append_null();
            continue;
        }
        let mut obj = serde_json::Map::new();
        obj.insert("storageType".into(), storage_type.value(i).into());
        obj.insert(
            "pathOrInlineDv".into(),
            if path_or_inline.is_null(i) {
                serde_json::Value::Null
            } else {
                path_or_inline.value(i).into()
            },
        );
        if !offset.is_null(i) {
            obj.insert("offset".into(), offset.value(i).into());
        }
        obj.insert(
            "sizeInBytes".into(),
            if size_in_bytes.is_null(i) {
                serde_json::Value::Null
            } else {
                size_in_bytes.value(i).into()
            },
        );
        obj.insert(
            "cardinality".into(),
            if cardinality.is_null(i) {
                serde_json::Value::Null
            } else {
                cardinality.value(i).into()
            },
        );
        out.append_value(serde_json::Value::Object(obj).to_string());
    }
    Ok(out.finish())
}

/// `numRecords` from the stats JSON, where present and well-formed.
fn num_records(stats: &StringArray) -> Int64Array {
    let mut out = Int64Builder::with_capacity(stats.len());
    for value in stats.iter() {
        let parsed = value
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .and_then(|v| v.get("numRecords").and_then(serde_json::Value::as_i64));
        out.append_option(parsed);
    }
    out.finish()
}

/// Every live file as the `add` action that would add it back, as JSON.
///
/// For RESTORE: a restored file must come back exactly as it was at the
/// target version -- statistics, partition values, tags, and under row
/// tracking the `baseRowId` and `defaultRowCommitVersion` its rows' ids are
/// derived from. The kernel's scan-row schema carries all of it; `dataChange`
/// is set true, as a restore changes the table's data.
pub fn add_actions(snapshot: SnapshotRef, engine: &dyn Engine) -> Result<Vec<String>> {
    let scan = snapshot
        .scan_builder()
        .with_stats(StatsOptions::all())
        .without_row_transforms()
        .build()?;
    let mut out = Vec::new();
    for metadata in scan.scan_metadata(engine)? {
        let (data, selection) = metadata?.scan_files.into_parts();
        let batch = data.try_into_record_batch()?;
        let mask: BooleanArray = (0..batch.num_rows())
            .map(|i| Some(selection.get(i).copied().unwrap_or(true)))
            .collect();
        let batch = filter_record_batch(&batch, &mask)?;
        if batch.num_rows() == 0 {
            continue;
        }
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .cloned()
                .ok_or_else(|| missing(name))
        };
        let path = arrow::compute::cast(&column("path")?, &DataType::Utf8)?;
        let path = path.as_string::<i32>();
        let size = arrow::compute::cast(&column("size")?, &DataType::Int64)?;
        let size = size.as_primitive::<Int64Type>();
        let modified = arrow::compute::cast(&column("modificationTime")?, &DataType::Int64)?;
        let modified = modified.as_primitive::<Int64Type>();
        let stats = arrow::compute::cast(&column("stats")?, &DataType::Utf8)?;
        let stats = stats.as_string::<i32>();
        let dv = column("deletionVector")?;
        let dv = deletion_vector_json(
            dv.as_struct_opt()
                .ok_or_else(|| missing("deletionVector"))?,
        )?;
        let constants = column("fileConstantValues")?;
        let constants = constants
            .as_struct_opt()
            .ok_or_else(|| missing("fileConstantValues"))?;
        let field = |name: &str| {
            constants
                .column_by_name(name)
                .cloned()
                .ok_or_else(|| missing(&format!("fileConstantValues.{name}")))
        };
        let partitions = normalize_map(&field("partitionValues")?)?;
        let tags = field("tags")?;
        let tags = normalize_map(&tags)?;
        let base_row_id = arrow::compute::cast(&field("baseRowId")?, &DataType::Int64)?;
        let base_row_id = base_row_id.as_primitive::<Int64Type>();
        let commit_version =
            arrow::compute::cast(&field("defaultRowCommitVersion")?, &DataType::Int64)?;
        let commit_version = commit_version.as_primitive::<Int64Type>();
        let provider = arrow::compute::cast(&field("clusteringProvider")?, &DataType::Utf8)?;
        let provider = provider.as_string::<i32>();
        for i in 0..batch.num_rows() {
            let mut add = serde_json::Map::new();
            add.insert("path".into(), path.value(i).into());
            add.insert(
                "partitionValues".into(),
                map_json(&partitions, i).unwrap_or_else(|| serde_json::json!({})),
            );
            add.insert("size".into(), size.value(i).into());
            add.insert(
                "modificationTime".into(),
                if modified.is_null(i) {
                    0
                } else {
                    modified.value(i)
                }
                .into(),
            );
            add.insert("dataChange".into(), true.into());
            if !stats.is_null(i) {
                add.insert("stats".into(), stats.value(i).into());
            }
            if let Some(tags) = map_json(&tags, i) {
                add.insert("tags".into(), tags);
            }
            if !dv.is_null(i) {
                let parsed: serde_json::Value = serde_json::from_str(dv.value(i))
                    .map_err(|e| NativeError::Invalid(format!("bad deletion vector: {e}")))?;
                add.insert("deletionVector".into(), parsed);
            }
            if !base_row_id.is_null(i) {
                add.insert("baseRowId".into(), base_row_id.value(i).into());
            }
            if !commit_version.is_null(i) {
                add.insert(
                    "defaultRowCommitVersion".into(),
                    commit_version.value(i).into(),
                );
            }
            if !provider.is_null(i) {
                add.insert("clusteringProvider".into(), provider.value(i).into());
            }
            out.push(serde_json::Value::Object(add).to_string());
        }
    }
    Ok(out)
}

/// Row `i` of a string map column as a JSON object, or None when null.
fn map_json(map: &ArrayRef, i: usize) -> Option<serde_json::Value> {
    let map = map.as_map_opt()?;
    if map.is_null(i) {
        return None;
    }
    let entries = map.value(i);
    let keys = entries.column(0).as_string::<i32>();
    let values = entries.column(1).as_string::<i32>();
    let mut out = serde_json::Map::new();
    for j in 0..entries.len() {
        let value = if values.is_null(j) {
            serde_json::Value::Null
        } else {
            values.value(j).into()
        };
        out.insert(keys.value(j).to_string(), value);
    }
    Some(serde_json::Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_records_tolerates_missing_and_malformed_stats() {
        let stats = StringArray::from(vec![
            Some(r#"{"numRecords":7,"minValues":{}}"#),
            None,
            Some("not json"),
            Some(r#"{"minValues":{}}"#),
        ]);
        let n = num_records(&stats);
        assert_eq!(n.value(0), 7);
        assert!(n.is_null(1) && n.is_null(2) && n.is_null(3));
    }

    #[test]
    fn empty_listing_has_the_full_schema() {
        let batch = RecordBatch::new_empty(output_schema());
        let names: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            names,
            [
                "path",
                "size",
                "modification_time",
                "partition_values",
                "stats",
                "deletion_vector",
                "num_records"
            ]
        );
    }
}
