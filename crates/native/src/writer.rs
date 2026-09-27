//! Writing data files, with the footer Spark needs to read their dates right.
//!
//! Spark decides per file whether to rebase DATE and TIMESTAMP values from
//! its legacy hybrid Julian/Gregorian calendar (`DataSourceUtils.
//! datetimeRebaseSpec`). A file whose footer names a Spark version of 3.0 or
//! later, and no `org.apache.spark.legacyDateTime`, is read as written. A file
//! that names no Spark version at all -- everything arrow-rs writes, the
//! kernel's default engine included -- is read in the mode
//! `spark.sql.parquet.datetimeRebaseModeInRead` gives, and Databricks SQL
//! warehouses read such files LEGACY: `DATE '0001-01-01'` written here came
//! back as 0001-01-03, and `1500-06-15` as 1500-06-05, timestamps alike, while
//! every other reader saw the values written. Every data file deltaswamp
//! writes (appends, overwrites, the rows a deletion-vector UPDATE or MERGE
//! rewrites, compaction, distributed workers) goes through here and says it is
//! proleptic Gregorian the one way Spark reads, by naming a Spark 3 version.
//!
//! Otherwise this is `DefaultEngine::write_parquet`: the logical-to-physical
//! transform, the kernel's statistics, a PUT and a HEAD, and the kernel's own
//! add-file metadata, which `build_add_file_metadata` is public for.

use std::sync::Arc;

use arrow::array::RecordBatch;
use delta_kernel::engine::arrow_conversion::TryFromArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::ObjectStoreExt;
use delta_kernel::parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use delta_kernel::parquet::file::metadata::KeyValue;
use delta_kernel::parquet::file::properties::WriterProperties;
use delta_kernel::schema::StructType;
use delta_kernel::transaction::BoundWriteContext;
use delta_kernel::{DeltaResult, Engine, EngineData, Error, FileMeta};
use delta_kernel_default_engine::parquet::DataFileMetadata;
use delta_kernel_default_engine::stats::collect_stats;

use crate::commit::SharedEngine;

/// The footer key Spark reads its writer's version from.
pub const SPARK_VERSION_KEY: &str = "org.apache.spark.version";
/// The Spark version every data file written here names. Any version from
/// 3.1.0 on means "proleptic Gregorian, INT96 included" to Spark (it compares
/// the strings); without `legacyDateTime` or `legacyINT96` alongside, nothing
/// is rebased, in any session time zone.
pub const SPARK_VERSION_VALUE: &str = "3.5.0";

/// The writer options for a data file: the kernel's (no embedded Arrow
/// schema), plus the Spark version key.
fn writer_options() -> ArrowWriterOptions {
    let properties = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            SPARK_VERSION_KEY.to_string(),
            SPARK_VERSION_VALUE.to_string(),
        )]))
        .build();
    ArrowWriterOptions::new()
        .with_skip_arrow_metadata(true)
        .with_properties(properties)
}

/// `batch` as the bytes of one Parquet file with the footer described above.
pub fn encode(batch: &RecordBatch) -> DeltaResult<Vec<u8>> {
    let mut buffer = vec![];
    let mut writer =
        ArrowWriter::try_new_with_options(&mut buffer, batch.schema(), writer_options())?;
    writer.write(batch)?;
    writer.close()?; // the footer is written on close
    Ok(buffer)
}

/// `DefaultEngine::write_parquet`, writing the file here: `data` is logical
/// (no partition columns), and comes back as add-file metadata for
/// `Transaction::add_files`.
pub async fn write_parquet(
    engine: &SharedEngine,
    data: &ArrowEngineData,
    write_context: &BoundWriteContext,
) -> DeltaResult<Box<dyn EngineData>> {
    let input_schema = StructType::try_from_arrow(data.record_batch().schema().as_ref())?;
    let evaluator = engine.evaluation_handler().new_expression_evaluator(
        Arc::new(input_schema),
        write_context.logical_to_physical(),
        write_context.physical_schema().clone().into(),
    )?;
    let physical = evaluator.evaluate(data)?.try_into_record_batch()?;
    write_physical(engine, &physical, write_context).await
}

/// `DefaultParquetHandler::write_parquet_file`, writing the file here:
/// `batch` is already in the physical schema.
pub async fn write_physical(
    engine: &SharedEngine,
    batch: &RecordBatch,
    write_context: &BoundWriteContext,
) -> DeltaResult<Box<dyn EngineData>> {
    let stats = collect_stats(
        batch,
        write_context.stats_columns(),
        write_context.physical_schema().as_ref(),
    )?;
    let buffer = encode(batch)?;
    let size = u64::try_from(buffer.len())
        .map_err(|_| Error::generic("unable to convert usize to u64"))?;

    let dir = write_context.write_dir();
    if !dir.path().ends_with('/') {
        return Err(Error::generic(format!(
            "Path must end with a trailing slash: {dir}"
        )));
    }
    let url = dir.join(&format!("{}.parquet", uuid::Uuid::new_v4()))?;
    let store = engine
        .get_object_store_for_url(&url)
        .ok_or_else(|| Error::generic(format!("no object store is registered for {url}")))?;
    let location = Path::from_url_path(url.path())?;
    store.put(&location, buffer.into()).await?;
    let head = store.head(&location).await?;
    if head.size != size {
        return Err(Error::generic(format!(
            "Size mismatch after writing parquet file: expected {size}, got {}",
            head.size
        )));
    }
    let file_meta = FileMeta::new(url, head.last_modified.timestamp_millis(), size);
    delta_kernel_default_engine::build_add_file_metadata(
        DataFileMetadata::new(file_meta, stats),
        write_context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Date32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use delta_kernel::parquet::file::reader::{FileReader, SerializedFileReader};

    #[test]
    fn every_file_names_a_spark_3_version_and_no_legacy_calendar() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("d", DataType::Date32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                // 0001-01-01, proleptic Gregorian.
                Arc::new(Date32Array::from(vec![-719162])),
            ],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("ds-footer-{}.parquet", uuid::Uuid::new_v4()));
        std::fs::write(&path, encode(&batch).unwrap()).unwrap();
        let reader = SerializedFileReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        let kv = reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .cloned()
            .unwrap_or_default();
        let keys: Vec<(&str, Option<&str>)> = kv
            .iter()
            .map(|kv| (kv.key.as_str(), kv.value.as_deref()))
            .collect();
        assert_eq!(keys, vec![(SPARK_VERSION_KEY, Some(SPARK_VERSION_VALUE))]);

        // And deltaswamp's own reader leaves such a file alone.
        let spec = crate::rebase::RebaseSpec::from_footer(reader.metadata(), "f").unwrap();
        assert!(
            !spec.dates && !spec.timestamps && !spec.int96_micros,
            "{spec:?}"
        );
    }
}
