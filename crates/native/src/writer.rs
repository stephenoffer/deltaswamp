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

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Schema};
use delta_kernel::engine::arrow_conversion::TryFromArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::ObjectStoreExt;
use delta_kernel::parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use delta_kernel::parquet::arrow::ArrowSchemaConverter;
use delta_kernel::parquet::basic::{Compression, Type as PhysicalType};
use delta_kernel::parquet::file::metadata::KeyValue;
use delta_kernel::parquet::file::properties::{EnabledStatistics, WriterProperties};
use delta_kernel::schema::StructType;
use delta_kernel::transaction::BoundWriteContext;
use delta_kernel::{DeltaResult, Engine, EngineData, Error, FileMeta};
use delta_kernel_default_engine::parquet::DataFileMetadata;
use delta_kernel_default_engine::stats::collect_stats;

use crate::commit::SharedEngine;
use url::Url;

/// The footer key Spark reads its writer's version from.
pub const SPARK_VERSION_KEY: &str = "org.apache.spark.version";
/// The Spark version every data file written here names. Any version from
/// 3.1.0 on means "proleptic Gregorian, INT96 included" to Spark (it compares
/// the strings); without `legacyDateTime` or `legacyINT96` alongside, nothing
/// is rebased, in any session time zone.
pub const SPARK_VERSION_VALUE: &str = "3.5.0";

/// The table property naming the codec data files are written with.
pub const CODEC_PROPERTY: &str = "delta.parquet.compression.codec";

/// The codec to write `snapshot`'s data files with.
pub fn codec_for(snapshot: &delta_kernel::snapshot::SnapshotRef) -> Compression {
    codec_named(
        snapshot
            .metadata_configuration()
            .get(CODEC_PROPERTY)
            .map(String::as_str),
    )
}

/// The codec a `delta.parquet.compression.codec` value names; snappy when unset.
///
/// arrow-rs's writer defaults to UNCOMPRESSED, and so did every file the
/// kernel wrote: an OPTIMIZE of ten snappy files (10.9 MB) wrote one
/// uncompressed file of 19.4 MB. Spark and delta-rs write snappy unless the
/// table says otherwise. LZO, which arrow-rs cannot write, falls back to
/// snappy, as does a value no codec answers to (the property is a writer's
/// preference, and every reader reads every codec).
pub fn codec_named(name: Option<&str>) -> Compression {
    let name = name.map(|n| n.trim().to_ascii_lowercase());
    match name.as_deref() {
        Some("uncompressed") | Some("none") => Compression::UNCOMPRESSED,
        Some("gzip") => Compression::GZIP(Default::default()),
        Some("zstd") => Compression::ZSTD(Default::default()),
        Some("brotli") => Compression::BROTLI(Default::default()),
        // Spark's "lz4" is the Hadoop-framed LZ4 codec, which arrow-rs's
        // LZ4 writes; LZ4_RAW is the newer, unframed one.
        Some("lz4") => Compression::LZ4,
        Some("lz4_raw") | Some("lz4raw") => Compression::LZ4_RAW,
        _ => Compression::SNAPPY,
    }
}

/// The writer options for a data file: the kernel's (no embedded Arrow
/// schema), plus the Spark version key, compressed with `codec`, and no
/// statistics for the FLOAT/DOUBLE leaves of the columns in `nan_columns`.
fn writer_options(
    codec: Compression,
    schema: &Schema,
    nan_columns: &HashSet<String>,
) -> DeltaResult<ArrowWriterOptions> {
    let mut builder = WriterProperties::builder()
        .set_compression(codec)
        .set_key_value_metadata(Some(vec![KeyValue::new(
            SPARK_VERSION_KEY.to_string(),
            SPARK_VERSION_VALUE.to_string(),
        )]));
    if !nan_columns.is_empty() {
        let descriptor = ArrowSchemaConverter::new().convert(schema)?;
        for column in descriptor.columns() {
            let float = matches!(
                column.physical_type(),
                PhysicalType::FLOAT | PhysicalType::DOUBLE
            );
            let top = column.path().parts().first();
            if float && top.is_some_and(|name| nan_columns.contains(name)) {
                builder = builder
                    .set_column_statistics_enabled(column.path().clone(), EnabledStatistics::None);
            }
        }
    }
    Ok(ArrowWriterOptions::new()
        .with_skip_arrow_metadata(true)
        .with_properties(builder.build()))
}

/// The top-level columns of `batch` with a NaN in some FLOAT or DOUBLE leaf.
///
/// arrow-rs leaves NaN out of a column chunk's min/max, and Spark, for which
/// NaN is greater than every other value, writes no min/max for a float
/// column holding one. Databricks prunes row groups on the footer's bounds, so
/// a file written here with max 3.0 beside a NaN answered `f = 'NaN'`,
/// `f > 100` and `NOT (f < 100)` without its NaN rows. Such columns get no
/// footer statistics (the file is read, never wrongly skipped); every other
/// column, and every NaN-free file, keeps them. The Delta stats need nothing:
/// the kernel's collector counts NaN as the maximum and writes it as null.
fn columns_with_nan(batch: &RecordBatch) -> HashSet<String> {
    fn has_nan(array: &dyn Array) -> bool {
        match array.data_type() {
            DataType::Float32 => array
                .as_primitive::<Float32Type>()
                .iter()
                .any(|v| v.is_some_and(f32::is_nan)),
            DataType::Float64 => array
                .as_primitive::<Float64Type>()
                .iter()
                .any(|v| v.is_some_and(f64::is_nan)),
            // A child's slots under a null parent, or past a list's offsets,
            // count too: at worst a column loses statistics it could keep.
            DataType::Struct(_) => array.as_struct().columns().iter().any(|c| has_nan(c)),
            DataType::List(_) => has_nan(array.as_list::<i32>().values()),
            DataType::LargeList(_) => has_nan(array.as_list::<i64>().values()),
            DataType::FixedSizeList(_, _) => has_nan(array.as_fixed_size_list().values()),
            DataType::Map(_, _) => {
                let map = array.as_map();
                has_nan(map.keys()) || has_nan(map.values())
            }
            _ => false,
        }
    }
    batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .filter(|(_, column)| has_nan(column.as_ref()))
        .map(|(field, _)| field.name().clone())
        .collect()
}

/// `batch` as the bytes of one Parquet file with the footer described above.
#[cfg(test)]
pub fn encode(batch: &RecordBatch, codec: Compression) -> DeltaResult<Vec<u8>> {
    encode_typed(batch, codec, None)
}

/// [`encode`], with the geospatial columns of `physical` (the table's
/// physical schema, as the batch is laid out) typed GEOMETRY or GEOGRAPHY in
/// Parquet (see `crate::geo`).
pub fn encode_typed(
    batch: &RecordBatch,
    codec: Compression,
    physical: Option<&StructType>,
) -> DeltaResult<Vec<u8>> {
    let mut buffer = vec![];
    let mut options = writer_options(codec, &batch.schema(), &columns_with_nan(batch))?;
    if let Some(physical) = physical {
        let converted = ArrowSchemaConverter::new().convert(&batch.schema())?;
        if let Some(typed) = crate::geo::parquet_schema(physical, &converted)
            .map_err(|e| Error::generic(e.to_string()))?
        {
            options = options.with_parquet_schema(typed);
        }
    }
    let mut writer = ArrowWriter::try_new_with_options(&mut buffer, batch.schema(), options)?;
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
    codec: Compression,
) -> DeltaResult<Box<dyn EngineData>> {
    let input_schema = StructType::try_from_arrow(data.record_batch().schema().as_ref())?;
    let evaluator = engine.evaluation_handler().new_expression_evaluator(
        Arc::new(input_schema),
        write_context.logical_to_physical(),
        write_context.physical_schema().clone().into(),
    )?;
    let physical = evaluator.evaluate(data)?.try_into_record_batch()?;
    write_physical(engine, &physical, write_context, codec).await
}

/// [`write_parquet`] of `batch`, whose columns named in `carried` (by the
/// first of each pair) are not the table's: they are taken out before the
/// logical-to-physical transform and written after it under the second name
/// -- a compaction's row ids and commit versions, into the table's
/// materialized row-tracking columns. They get no statistics, as in Spark.
pub async fn write_parquet_carrying(
    engine: &SharedEngine,
    batch: RecordBatch,
    carried: &[(String, String)],
    write_context: &BoundWriteContext,
    codec: Compression,
) -> DeltaResult<Box<dyn EngineData>> {
    let mut logical = batch;
    let mut extra = Vec::new();
    for (name, physical) in carried {
        if let Ok(index) = logical.schema().index_of(name) {
            extra.push((physical.clone(), logical.column(index).clone()));
            logical.remove_column(index);
        }
    }
    if extra.is_empty() {
        return write_parquet(engine, &ArrowEngineData::new(logical), write_context, codec).await;
    }
    let input_schema = StructType::try_from_arrow(logical.schema().as_ref())?;
    let evaluator = engine.evaluation_handler().new_expression_evaluator(
        Arc::new(input_schema),
        write_context.logical_to_physical(),
        write_context.physical_schema().clone().into(),
    )?;
    let physical = evaluator
        .evaluate(&ArrowEngineData::new(logical))?
        .try_into_record_batch()?;
    let mut fields: Vec<arrow::datatypes::FieldRef> =
        physical.schema().fields().iter().cloned().collect();
    let mut columns = physical.columns().to_vec();
    for (name, column) in extra {
        let column = arrow::compute::cast(&column, &DataType::Int64)?;
        fields.push(Arc::new(arrow::datatypes::Field::new(
            name,
            DataType::Int64,
            column.null_count() > 0,
        )));
        columns.push(column);
    }
    let physical = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            fields,
            physical.schema().metadata().clone(),
        )),
        columns,
    )?;
    write_physical(engine, &physical, write_context, codec).await
}

/// `DefaultParquetHandler::write_parquet_file`, writing the file here:
/// `batch` is already in the physical schema.
pub async fn write_physical(
    engine: &SharedEngine,
    batch: &RecordBatch,
    write_context: &BoundWriteContext,
    codec: Compression,
) -> DeltaResult<Box<dyn EngineData>> {
    let stats = collect_stats(
        batch,
        write_context.stats_columns(),
        write_context.physical_schema().as_ref(),
    )?;
    let buffer = encode_typed(batch, codec, Some(write_context.physical_schema().as_ref()))?;
    let size = u64::try_from(buffer.len())
        .map_err(|_| Error::generic("unable to convert usize to u64"))?;

    let dir = lowercase_prefix(write_context.write_dir(), write_context.table_root_dir());
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
    // Past the PUT the file exists, but no add-file metadata will name it if
    // this fails: taken back out, best effort, or it is nobody's to clean up.
    let head = match store.head(&location).await {
        Ok(head) => head,
        Err(err) => {
            let _ = store.delete(&location).await;
            return Err(err.into());
        }
    };
    if head.size != size {
        let _ = store.delete(&location).await;
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

/// `dir` with the kernel's random directory prefix lowercased.
///
/// The kernel draws the prefix from mixed-case letters, as Spark does. On a
/// case-insensitive filesystem (macOS and Windows by default) `nT/` and `nt/`
/// are one directory that lists under whichever spelling created it, so a
/// file the log names `nt/x.parquet` lists as `nT/x.parquet` and looks like
/// an orphan to an exact-match VACUUM. Only a single alphanumeric segment
/// under the root is a prefix; Hive partition directories are left alone.
fn lowercase_prefix(dir: Url, root: &Url) -> Url {
    let Some(rest) = dir.path().strip_prefix(root.path()) else {
        return dir;
    };
    let segment = rest.strip_suffix('/').unwrap_or(rest);
    if segment.is_empty()
        || segment.contains('/')
        || !segment.bytes().all(|b| b.is_ascii_alphanumeric())
        || !segment.bytes().any(|b| b.is_ascii_uppercase())
    {
        return dir;
    }
    let mut lowered = dir.clone();
    lowered.set_path(&format!("{}{}/", root.path(), segment.to_ascii_lowercase()));
    lowered
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Date32Array, Int32Array};
    use arrow::datatypes::Field;
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
        std::fs::write(&path, encode(&batch, Compression::SNAPPY).unwrap()).unwrap();
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

    #[test]
    fn float_columns_holding_nan_get_no_footer_statistics() {
        use arrow::array::{Float64Array, ListArray, StructArray};
        use arrow::datatypes::{Fields, Float64Type};

        let inner = Fields::from(vec![Field::new("x", DataType::Float64, true)]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float64, true),
            Field::new("g", DataType::Float64, true),
            Field::new("st", DataType::Struct(inner.clone()), true),
            Field::new_list("l", Field::new("item", DataType::Float64, true), true),
            Field::new("i", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, f64::NAN, 3.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(StructArray::new(
                    inner,
                    vec![Arc::new(Float64Array::from(vec![f64::NAN, 1.0, 2.0]))],
                    None,
                )),
                Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
                    Some(vec![Some(1.0), Some(f64::NAN)]),
                    None,
                    Some(vec![]),
                ])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("ds-nan-{}.parquet", uuid::Uuid::new_v4()));
        std::fs::write(&path, encode(&batch, Compression::SNAPPY).unwrap()).unwrap();
        let reader = SerializedFileReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        let group = reader.metadata().row_group(0);
        let with_stats: Vec<(String, bool)> = group
            .columns()
            .iter()
            .map(|c| (c.column_path().string(), c.statistics().is_some()))
            .collect();
        assert_eq!(
            with_stats,
            vec![
                ("f".to_string(), false),
                ("g".to_string(), true),
                ("st.x".to_string(), false),
                ("l.list.item".to_string(), false),
                ("i".to_string(), true),
            ]
        );
    }

    #[test]
    fn files_are_compressed_with_the_tables_codec_and_snappy_by_default() {
        assert_eq!(codec_named(None), Compression::SNAPPY);
        assert_eq!(codec_named(Some("lzo")), Compression::SNAPPY);
        assert_eq!(
            codec_named(Some(" ZSTD ")),
            Compression::ZSTD(Default::default())
        );
        assert_eq!(codec_named(Some("none")), Compression::UNCOMPRESSED);
        assert_eq!(codec_named(Some("lz4_raw")), Compression::LZ4_RAW);

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from((0..1000).collect::<Vec<i32>>()))],
        )
        .unwrap();
        for codec in [Compression::SNAPPY, Compression::ZSTD(Default::default())] {
            let path =
                std::env::temp_dir().join(format!("ds-codec-{}.parquet", uuid::Uuid::new_v4()));
            std::fs::write(&path, encode(&batch, codec).unwrap()).unwrap();
            let reader = SerializedFileReader::new(std::fs::File::open(&path).unwrap()).unwrap();
            std::fs::remove_file(&path).unwrap();
            let written = reader.metadata().row_group(0).column(0).compression();
            assert_eq!(written, codec);
        }
    }
    #[test]
    fn random_prefixes_are_lowercased_and_partitions_kept() {
        let root = Url::parse("file:///t/tbl/").unwrap();
        let at = |p: &str| lowercase_prefix(Url::parse(p).unwrap(), &root).to_string();
        assert_eq!(at("file:///t/tbl/nT/"), "file:///t/tbl/nt/");
        assert_eq!(at("file:///t/tbl/ab/"), "file:///t/tbl/ab/");
        assert_eq!(at("file:///t/tbl/"), "file:///t/tbl/");
        assert_eq!(at("file:///t/tbl/P=Ab/"), "file:///t/tbl/P=Ab/");
        assert_eq!(at("file:///t/tbl/A=1/B=2/"), "file:///t/tbl/A=1/B=2/");
    }
}
