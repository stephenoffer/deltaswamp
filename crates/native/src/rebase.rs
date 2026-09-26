//! Spark's legacy hybrid-calendar dates and timestamps, rebased on read.
//!
//! Spark 2.x -- and Spark 3 / Databricks with `datetimeRebaseModeInWrite =
//! LEGACY`, which Photon writes today -- stores DATE and TIMESTAMP values in
//! the hybrid Julian/Gregorian calendar of `java.util.Date`. The Delta
//! protocol, Arrow and every other reader use the proleptic Gregorian
//! calendar, so before 1582-10-15 the stored numbers name different days:
//! `DATE '0001-01-01'` is on disk as -719164 days, which a plain reader shows
//! as 0000-12-30 (and which Python's `date` cannot hold). The file's footer
//! says so: `org.apache.spark.legacyDateTime` (and `legacyINT96` for INT96
//! timestamps), or a writer version before Spark 3.0.
//!
//! Spark reads such a file back by rebasing each value from the Julian to the
//! proleptic Gregorian calendar (`RebaseDateTime.rebaseJulianToGregorianDays`
//! and `...Micros`); this module does the same to every batch the kernel's
//! Parquet reader returns for such a file, before the kernel sees it. The
//! file statistics in the Delta log were computed from the in-memory
//! (proleptic) values, so after the rebase the rows agree with them and data
//! skipping stays correct.
//!
//! Scope, following Spark exactly:
//!
//! * DATE (INT32) and zoned TIMESTAMP (INT64 micros/millis, or INT96) columns
//!   are rebased, nested ones included. TIMESTAMP_NTZ never is: Spark writes
//!   it without rebasing.
//! * Values on or after the last calendar switch (1582-10-15) are unchanged,
//!   which is nearly every value in practice.
//! * A timestamp's rebase depends on the time zone the writer used
//!   (`org.apache.spark.timeZone`). It is computed here for UTC -- what
//!   Databricks writes -- and a file written in any other zone that holds a
//!   timestamp before the switch is refused rather than shifted by a guess.

use std::future::Future;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, LargeListArray, ListArray, MapArray, RecordBatch, StructArray,
};
use arrow::datatypes::{DataType, Date32Type, TimeUnit, TimestampMicrosecondType};
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::DynObjectStore;
// ParquetObjectReader is deprecated upstream in favour of a hand-written
// AsyncFileReader, but it is what the kernel's own default engine reads with.
#[allow(deprecated)]
use delta_kernel::parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use delta_kernel::parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use delta_kernel::parquet::file::metadata::ParquetMetaData;
use delta_kernel::schema::{DataType as KernelType, PrimitiveType, SchemaRef};
use delta_kernel::{
    DeltaResult, Engine, EngineData, Error, EvaluationHandler, FileDataReadResultIterator,
    FileMeta, JsonHandler, ParquetFooter, ParquetHandler, PredicateRef, StorageHandler,
};

use crate::commit::SharedEngine;

/// Julian days since the epoch at which the Julian - Gregorian difference
/// changes, and that difference (in days) from each switch to the next. From
/// Spark's `RebaseDateTime` (`julianGregDiffSwitchDay`, `julianGregDiffs`).
/// Days before the first switch take the first difference, as in Spark.
const SWITCH_DAYS: [i32; 14] = [
    -719164, -682945, -646420, -609895, -536845, -500320, -463795, -390745, -354220, -317695,
    -244645, -208120, -171595, -141427,
];
const DIFF_DAYS: [i32; 14] = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 0];
/// 1582-10-15: from here on both calendars agree.
const LAST_SWITCH_DAY: i32 = -141427;
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Footer keys Spark writes (see Spark's `package.scala` in `sql`).
const SPARK_VERSION_KEY: &str = "org.apache.spark.version";
const LEGACY_DATETIME_KEY: &str = "org.apache.spark.legacyDateTime";
const LEGACY_INT96_KEY: &str = "org.apache.spark.legacyINT96";
const TIME_ZONE_KEY: &str = "org.apache.spark.timeZone";

/// Spark's `rebaseJulianToGregorianDays`.
pub fn julian_to_gregorian_days(days: i32) -> i32 {
    if days >= LAST_SWITCH_DAY {
        return days;
    }
    let mut i = SWITCH_DAYS.len() - 1;
    while i > 0 && days < SWITCH_DAYS[i] {
        i -= 1;
    }
    days + DIFF_DAYS[i]
}

/// Spark's `rebaseJulianToGregorianMicros` for a file written in UTC: the
/// local date-time is the UTC one, so the date is rebased and the time of
/// day kept.
pub fn julian_to_gregorian_micros_utc(micros: i64) -> i64 {
    let day = micros.div_euclid(MICROS_PER_DAY);
    if day >= i64::from(LAST_SWITCH_DAY) {
        return micros;
    }
    // Any i64 of micros is within +-2^27 days, so this cannot truncate.
    let rebased = julian_to_gregorian_days(day as i32);
    i64::from(rebased) * MICROS_PER_DAY + micros.rem_euclid(MICROS_PER_DAY)
}

/// How one file's values must be rebased.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RebaseSpec {
    pub dates: bool,
    pub timestamps: bool,
    /// The writer's time zone when it is not UTC (timestamps only).
    pub zone: Option<String>,
}

impl RebaseSpec {
    fn any(&self) -> bool {
        self.dates || self.timestamps
    }

    /// What Spark's `DataSourceUtils.datetimeRebaseSpec` / `int96RebaseSpec`
    /// decide for a file with this footer.
    pub fn from_footer(metadata: &ParquetMetaData, location: &str) -> DeltaResult<Self> {
        let file = metadata.file_metadata();
        let kv = |key: &str| {
            file.key_value_metadata()
                .and_then(|kvs| kvs.iter().find(|kv| kv.key == key))
                .map(|kv| kv.value.clone().unwrap_or_default())
        };
        // No Spark version: not written by Spark, and so proleptic Gregorian.
        let Some(version) = kv(SPARK_VERSION_KEY) else {
            return Ok(Self::default());
        };
        // Spark compares the version strings lexicographically; so do we.
        let legacy_datetime = version.as_str() < "3.0.0" || kv(LEGACY_DATETIME_KEY).is_some();
        let legacy_int96 = version.as_str() < "3.1.0" || kv(LEGACY_INT96_KEY).is_some();

        let (mut dates, mut int96, mut int64_ts) = (false, false, false);
        for column in metadata.file_metadata().schema_descr().columns() {
            match column.physical_type() {
                PhysicalType::INT96 => int96 = true,
                PhysicalType::INT32 => {
                    dates |= matches!(column.logical_type_ref(), Some(LogicalType::Date))
                        || column.converted_type() == ConvertedType::DATE;
                }
                PhysicalType::INT64 => {
                    int64_ts |= match column.logical_type_ref() {
                        Some(LogicalType::Timestamp(t)) => t.is_adjusted_to_u_t_c,
                        _ => matches!(
                            column.converted_type(),
                            ConvertedType::TIMESTAMP_MICROS | ConvertedType::TIMESTAMP_MILLIS
                        ),
                    };
                }
                _ => {}
            }
        }
        if int96 && int64_ts && legacy_int96 != legacy_datetime {
            // One Arrow type (a zoned timestamp) for both, so the batch cannot
            // tell which column needs which rebase.
            return Err(Error::generic(format!(
                "{location} holds both INT96 and INT64 timestamps, written with different \
                 calendar rebase modes (Spark {version}); deltaswamp cannot rebase them apart. \
                 Read it through the SQL warehouse"
            )));
        }
        let timestamps = (int96 && legacy_int96) || (int64_ts && legacy_datetime);
        let zone = kv(TIME_ZONE_KEY).filter(|z| !is_utc(z));
        Ok(Self {
            dates: dates && legacy_datetime,
            timestamps,
            zone,
        })
    }
}

fn is_utc(zone: &str) -> bool {
    matches!(
        zone.trim(),
        "UTC"
            | "Etc/UTC"
            | "GMT"
            | "Etc/GMT"
            | "Etc/GMT0"
            | "Etc/GMT+0"
            | "Etc/GMT-0"
            | "GMT0"
            | "UCT"
            | "Etc/UCT"
            | "Universal"
            | "Etc/Universal"
            | "Zulu"
            | "Etc/Zulu"
            | "Z"
            | "+00:00"
            | "-00:00"
            | ""
    )
}

/// Whether `schema` has a column a rebase could change.
fn needs_footer(schema: &SchemaRef) -> bool {
    fn walk(t: &KernelType) -> bool {
        match t {
            KernelType::Primitive(PrimitiveType::Date | PrimitiveType::Timestamp) => true,
            KernelType::Struct(s) => s.fields().any(|f| walk(f.data_type())),
            KernelType::Array(a) => walk(a.element_type()),
            KernelType::Map(m) => walk(m.key_type()) || walk(m.value_type()),
            _ => false,
        }
    }
    schema.fields().any(|f| walk(f.data_type()))
}

/// `array` with its DATE and zoned TIMESTAMP values rebased per `spec`.
fn rebase_array(array: &ArrayRef, spec: &RebaseSpec, location: &str) -> DeltaResult<ArrayRef> {
    Ok(match array.data_type() {
        DataType::Date32 if spec.dates => Arc::new(
            array
                .as_primitive::<Date32Type>()
                .unary::<_, Date32Type>(julian_to_gregorian_days),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) if spec.timestamps => {
            let values = array.as_primitive::<TimestampMicrosecondType>();
            if spec.zone.is_some() {
                // Rebasing depends on the zone's offset rules; values after
                // 1582-10-16 UTC need none in any zone (offsets are < 1 day).
                let limit = (i64::from(LAST_SWITCH_DAY) + 1) * MICROS_PER_DAY;
                if values.iter().flatten().any(|v| v < limit) {
                    return Err(Error::generic(format!(
                        "{location} stores timestamps before 1582-10-15 in Spark's legacy hybrid \
                         calendar, written in time zone {:?}; deltaswamp rebases those only for \
                         UTC. Read the table through the SQL warehouse",
                        spec.zone.as_deref().unwrap_or_default()
                    )));
                }
                return Ok(array.clone());
            }
            Arc::new(
                values
                    .unary::<_, TimestampMicrosecondType>(julian_to_gregorian_micros_utc)
                    .with_timezone(tz.clone()),
            )
        }
        DataType::Struct(fields) => {
            let s = array.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(|c| rebase_array(c, spec, location))
                .collect::<DeltaResult<Vec<_>>>()?;
            Arc::new(StructArray::try_new(
                fields.clone(),
                columns,
                s.nulls().cloned(),
            )?)
        }
        DataType::List(field) => {
            let l = array.as_list::<i32>();
            let values = rebase_array(l.values(), spec, location)?;
            Arc::new(ListArray::try_new(
                field.clone(),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?)
        }
        DataType::LargeList(field) => {
            let l = array.as_list::<i64>();
            let values = rebase_array(l.values(), spec, location)?;
            Arc::new(LargeListArray::try_new(
                field.clone(),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?)
        }
        DataType::Map(field, ordered) => {
            let m = array.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = rebase_array(&entries, spec, location)?;
            Arc::new(MapArray::try_new(
                field.clone(),
                m.offsets().clone(),
                entries.as_struct().clone(),
                m.nulls().cloned(),
                *ordered,
            )?)
        }
        _ => array.clone(),
    })
}

/// One physical batch of a file, rebased per `spec`.
pub fn rebase_batch(
    batch: RecordBatch,
    spec: &RebaseSpec,
    location: &str,
) -> DeltaResult<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .map(|c| rebase_array(c, spec, location))
        .collect::<DeltaResult<Vec<_>>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Run `fut` to completion from sync code, inside or outside the runtime.
fn block<F: Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        // A kernel callback already on a runtime worker: our runtime is
        // multi-threaded, so another worker drives the future meanwhile.
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => crate::runtime::block_on(fut),
    }
}

/// A Parquet handler that rebases files Spark wrote in the hybrid calendar.
struct RebasingParquet {
    inner: Arc<dyn ParquetHandler>,
    store: Arc<DynObjectStore>,
}

impl RebasingParquet {
    #[allow(deprecated)]
    fn spec(&self, file: &FileMeta) -> DeltaResult<RebaseSpec> {
        // Read only the footer: one extra range request per file with a
        // date/timestamp column, which is what it costs to know.
        let path = Path::from_url_path(file.location.path())?;
        let mut reader = ParquetObjectReader::new(self.store.clone(), path);
        if file.size != 0 {
            reader = reader.with_file_size(file.size);
        }
        let metadata = block(async move { reader.get_metadata(None).await })?;
        RebaseSpec::from_footer(&metadata, file.location.as_str())
    }
}

impl ParquetHandler for RebasingParquet {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        if !needs_footer(&physical_schema) || files.is_empty() {
            return self
                .inner
                .read_parquet_files(files, physical_schema, predicate);
        }
        let mut specs = Vec::with_capacity(files.len());
        for file in files {
            specs.push(self.spec(file)?);
        }
        if specs.iter().all(|s| !s.any()) {
            return self
                .inner
                .read_parquet_files(files, physical_schema, predicate);
        }
        // Read file by file, so each batch is rebased with its own file's
        // spec. No row-group pushdown for a rebased file: its Parquet
        // statistics are in the Julian calendar too.
        let mut out: Vec<FileDataReadResultIterator> = Vec::with_capacity(files.len());
        for (file, spec) in files.iter().zip(specs) {
            if !spec.any() {
                out.push(self.inner.read_parquet_files(
                    std::slice::from_ref(file),
                    physical_schema.clone(),
                    predicate.clone(),
                )?);
                continue;
            }
            let location = file.location.to_string();
            let batches = self.inner.read_parquet_files(
                std::slice::from_ref(file),
                physical_schema.clone(),
                None,
            )?;
            out.push(Box::new(batches.map(move |data| {
                let batch = data?.try_into_record_batch()?;
                let rebased = rebase_batch(batch, &spec, &location)?;
                Ok(Box::new(ArrowEngineData::new(rebased)) as Box<dyn EngineData>)
            })));
        }
        Ok(Box::new(out.into_iter().flatten()))
    }

    fn write_parquet_file(
        &self,
        location: url::Url,
        data: FileDataReadResultIterator,
    ) -> DeltaResult<()> {
        self.inner.write_parquet_file(location, data)
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> DeltaResult<ParquetFooter> {
        self.inner.read_parquet_footer(file)
    }
}

/// The shared engine, reading Parquet through [`RebasingParquet`].
struct RebasingEngine {
    inner: SharedEngine,
    parquet: Arc<dyn ParquetHandler>,
}

impl Engine for RebasingEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.inner.evaluation_handler()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.inner.storage_handler()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.inner.json_handler()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet.clone()
    }
}

/// The engine a data read should use: `engine`, with legacy-calendar Parquet
/// files rebased as Spark rebases them.
pub fn reading_engine(engine: &SharedEngine, table_root: &url::Url) -> Arc<dyn Engine> {
    let Some(store) = engine.get_object_store_for_url(table_root) else {
        return engine.clone() as Arc<dyn Engine>;
    };
    Arc::new(RebasingEngine {
        inner: engine.clone(),
        parquet: Arc::new(RebasingParquet {
            inner: engine.parquet_handler(),
            store,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_rebase_as_spark_does() {
        // Values the warehouse reports for rows Photon wrote in LEGACY mode.
        assert_eq!(julian_to_gregorian_days(-719164), -719162); // 0001-01-01
        assert_eq!(julian_to_gregorian_days(-171489), -171499); // 1500-06-15
        assert_eq!(julian_to_gregorian_days(-682945), -682944); // 0100-03-01
                                                                // Before the first switch: the first difference, as in Spark.
        assert_eq!(julian_to_gregorian_days(-800000), -799998);
        // From 1582-10-15 on, nothing moves.
        assert_eq!(julian_to_gregorian_days(-141427), -141427);
        assert_eq!(julian_to_gregorian_days(19723), 19723);
        // Just before the last switch: the 10-day gap.
        assert_eq!(julian_to_gregorian_days(-141428), -141438);
    }

    #[test]
    fn utc_micros_keep_the_time_of_day() {
        assert_eq!(
            julian_to_gregorian_micros_utc(-62135769600000000),
            -62135596800000000
        );
        assert_eq!(
            julian_to_gregorian_micros_utc(-14816604303211000),
            -14817468303211000
        );
        assert_eq!(
            julian_to_gregorian_micros_utc(-59006538000000000),
            -59006365200000000
        );
        assert_eq!(
            julian_to_gregorian_micros_utc(1704067200000000),
            1704067200000000
        );
        assert_eq!(
            julian_to_gregorian_micros_utc(-12219292800000000),
            -12219292800000000
        );
    }

    #[test]
    fn nested_dates_rebase_and_ntz_does_not() {
        use arrow::array::{Date32Array, TimestampMicrosecondArray};
        use arrow::datatypes::{Field, Fields, Schema};

        let inner = Fields::from(vec![Field::new("d", DataType::Date32, true)]);
        let st = StructArray::try_new(
            inner.clone(),
            vec![Arc::new(Date32Array::from(vec![Some(-719164), None])) as ArrayRef],
            None,
        )
        .unwrap();
        let ntz = TimestampMicrosecondArray::from(vec![Some(-62135596800000000), None]);
        let ts = TimestampMicrosecondArray::from(vec![Some(-62135769600000000), None])
            .with_timezone("UTC");
        let schema = Arc::new(Schema::new(vec![
            Field::new("st", DataType::Struct(inner), true),
            Field::new("ntz", ntz.data_type().clone(), true),
            Field::new("ts", ts.data_type().clone(), true),
        ]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(st), Arc::new(ntz), Arc::new(ts)]).unwrap();
        let spec = RebaseSpec {
            dates: true,
            timestamps: true,
            zone: None,
        };
        let out = rebase_batch(batch, &spec, "f").unwrap();
        let d = out
            .column(0)
            .as_struct()
            .column(0)
            .as_primitive::<Date32Type>();
        assert_eq!(d.value(0), -719162);
        assert!(d.is_null(1));
        let ntz = out.column(1).as_primitive::<TimestampMicrosecondType>();
        assert_eq!(ntz.value(0), -62135596800000000);
        let ts = out.column(2).as_primitive::<TimestampMicrosecondType>();
        assert_eq!(ts.value(0), -62135596800000000);
    }

    #[test]
    fn other_zones_refuse_ancient_timestamps_only() {
        use arrow::array::TimestampMicrosecondArray;
        use arrow::datatypes::{Field, Schema};

        let spec = RebaseSpec {
            dates: false,
            timestamps: true,
            zone: Some("America/Los_Angeles".into()),
        };
        let modern = TimestampMicrosecondArray::from(vec![1704067200000000]).with_timezone("UTC");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            modern.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(modern)]).unwrap();
        assert!(rebase_batch(batch, &spec, "f").is_ok());
        let old = TimestampMicrosecondArray::from(vec![-62135769600000000]).with_timezone("UTC");
        let batch = RecordBatch::try_new(schema, vec![Arc::new(old)]).unwrap();
        let err = rebase_batch(batch, &spec, "f").unwrap_err();
        assert!(err.to_string().contains("America/Los_Angeles"), "{err}");
    }
}
