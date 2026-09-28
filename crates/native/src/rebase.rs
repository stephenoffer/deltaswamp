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
//!   Databricks writes. In any other zone Spark rebases with per-zone switch
//!   tables that run until about 1900 (local mean time, which
//!   `java.util.TimeZone` does not model), so a file written in another zone,
//!   or one that does not record its zone at all (Spark reads those in the
//!   reader's session zone), is refused if it holds a timestamp before
//!   1900-01-01T00:00:00Z -- the same bound Spark's own warning names --
//!   rather than shifted by a guess.
//!
//! INT96 timestamps are also read here at microsecond precision, whatever
//! wrote them. Arrow decodes INT96 as nanoseconds by default, and an `i64` of
//! nanoseconds overflows before 1677-09-21: a year-1500 value came back as
//! 2085, so neither the value nor its rebase could be right.

use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, LargeListArray, ListArray, MapArray, RecordBatch, StructArray,
};
use arrow::datatypes::{
    DataType, Date32Type, Field, FieldRef, Schema, TimeUnit, TimestampMicrosecondType,
};
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::engine::arrow_utils::{
    fixup_parquet_read, ordering_needs_row_indexes, parquet_read_plan, ReorderIndex,
    RowIndexBuilder,
};
use delta_kernel::engine::reader_options;
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::DynObjectStore;
// ParquetObjectReader is deprecated upstream in favour of a hand-written
// AsyncFileReader, but it is what the kernel's own default engine reads with.
use delta_kernel::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReader};
#[allow(deprecated)]
use delta_kernel::parquet::arrow::async_reader::{
    AsyncFileReader, ParquetObjectReader, ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder,
};
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
/// 1900-01-01T00:00:00Z. Before it Spark's rebase of a timestamp written in a
/// zone other than UTC follows that zone's own switch table; from it on, no
/// zone's rebase moves a value (Spark warns of "timestamps before
/// 1900-01-01T00:00:00Z" for the same reason).
const ZONED_REBASE_LIMIT_MICROS: i64 = -2_208_988_800_000_000;
/// Rows per batch of a file read here rather than by the kernel's reader.
const BATCH_SIZE: usize = 1024;

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

/// The time zone a legacy file's timestamps were written in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum WriterZone {
    /// UTC, whose rebase is computed here.
    #[default]
    Utc,
    /// Any other zone, by the name the footer gives.
    Named(String),
    /// The footer names none (Spark before 3.2): Spark rebases these in the
    /// reader's session time zone, which a file cannot tell us.
    Unknown,
}

/// How one file's values must be rebased.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RebaseSpec {
    pub dates: bool,
    pub timestamps: bool,
    /// The writer's time zone (timestamps only).
    pub zone: WriterZone,
    /// The file stores timestamps as INT96, which must be decoded at
    /// microsecond precision to survive values before 1677.
    pub int96_micros: bool,
}

impl RebaseSpec {
    fn any(&self) -> bool {
        self.dates || self.timestamps
    }

    /// Whether the file must be read here rather than by the kernel's reader.
    fn own_read(&self) -> bool {
        self.any() || self.int96_micros
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

        let (mut dates, mut int96, mut int64_ts, mut int64_nanos) = (false, false, false, false);
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
                    int64_nanos |= matches!(
                        column.logical_type_ref(),
                        Some(LogicalType::Timestamp(t))
                            if matches!(t.unit, delta_kernel::parquet::basic::TimeUnit::NANOS)
                    );
                }
                _ => {}
            }
        }
        // INT96 is read at micros by retyping the nanosecond timestamps Arrow
        // infers for it; an INT64 nanosecond column would be retyped too, so a
        // file with one is left to the kernel's reader (no Spark writes one).
        let int96_micros = int96 && !int64_nanos;

        // No Spark version: not written by Spark, and so proleptic Gregorian.
        let Some(version) = kv(SPARK_VERSION_KEY) else {
            return Ok(Self {
                int96_micros,
                ..Self::default()
            });
        };
        // Spark compares the version strings lexicographically; so do we.
        let legacy_datetime = version.as_str() < "3.0.0" || kv(LEGACY_DATETIME_KEY).is_some();
        let legacy_int96 = version.as_str() < "3.1.0" || kv(LEGACY_INT96_KEY).is_some();

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
        let zone = match kv(TIME_ZONE_KEY) {
            None => WriterZone::Unknown,
            Some(z) if is_utc(&z) => WriterZone::Utc,
            Some(z) => WriterZone::Named(z),
        };
        Ok(Self {
            dates: dates && legacy_datetime,
            timestamps,
            zone,
            int96_micros,
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
            if spec.zone != WriterZone::Utc {
                // Rebasing depends on the zone's offset rules, down to its
                // local mean time before about 1900 (Los Angeles moved a value
                // of 1850 by 422 seconds); from 1900 on no zone's does.
                if values
                    .iter()
                    .flatten()
                    .any(|v| v < ZONED_REBASE_LIMIT_MICROS)
                {
                    let zone = match &spec.zone {
                        WriterZone::Named(z) => format!("written in time zone {z:?}"),
                        _ => "whose footer does not record the writer's time zone (Spark \
                              reads those in its session time zone)"
                            .to_string(),
                    };
                    return Err(Error::generic(format!(
                        "{location} stores timestamps before 1900-01-01T00:00:00Z in Spark's \
                         legacy hybrid calendar, {zone}; deltaswamp rebases those only for \
                         UTC. Read the table through the SQL warehouse"
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

/// `schema` with every nanosecond timestamp retyped to microseconds: the read
/// hint that makes Arrow decode INT96 without overflowing.
///
/// Arrow infers INT96 without a zone, and the hint gives it UTC, the type the
/// kernel asks for a TIMESTAMP in (INT96 is only ever Spark's TIMESTAMP).
/// Left zoneless, the read needed a cast to UTC, and the kernel plans a cast
/// of a list's element as a cast of the whole list: every ARRAY<TIMESTAMP>
/// Databricks wrote failed with "Cannot cast LIST to non-list data type".
fn int96_as_micros(schema: &Schema) -> Schema {
    fn retype(t: &DataType) -> DataType {
        match t {
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => DataType::Timestamp(
                TimeUnit::Microsecond,
                Some(tz.clone().unwrap_or_else(|| "UTC".into())),
            ),
            DataType::Struct(fields) => DataType::Struct(fields.iter().map(field).collect()),
            DataType::List(f) => DataType::List(field(f)),
            DataType::LargeList(f) => DataType::LargeList(field(f)),
            DataType::Map(f, ordered) => DataType::Map(field(f), *ordered),
            other => other.clone(),
        }
    }
    fn field(f: &FieldRef) -> FieldRef {
        Arc::new(Field::clone(f).with_data_type(retype(f.data_type())))
    }
    Schema::new_with_metadata(
        schema.fields().iter().map(field).collect::<Vec<_>>(),
        schema.metadata().clone(),
    )
}

/// One file read row group by row group with INT96 decoded at microseconds,
/// then shaped to the requested schema exactly as the kernel's reader does.
#[allow(deprecated)]
struct Int96MicrosRead {
    stream: ParquetRecordBatchStream<ParquetObjectReader>,
    current: Option<ParquetRecordBatchReader>,
    ordering: Vec<ReorderIndex>,
    row_indexes: Option<std::iter::Flatten<std::vec::IntoIter<Range<i64>>>>,
    schema: SchemaRef,
    location: String,
    done: bool,
}

impl Int96MicrosRead {
    #[allow(deprecated)]
    fn open(store: Arc<DynObjectStore>, file: &FileMeta, schema: SchemaRef) -> DeltaResult<Self> {
        let path = Path::from_url_path(file.location.path())?;
        let mut reader = ParquetObjectReader::new(store, path);
        if file.size != 0 {
            reader = reader.with_file_size(file.size);
        }
        let requested = schema.clone();
        let (stream, ordering, row_indexes) = block(async move {
            // Spelled out: parquet errors convert into more than one type.
            let fail = |e: delta_kernel::parquet::errors::ParquetError| Error::from(e);
            let inferred = ArrowReaderMetadata::load_async(&mut reader, reader_options())
                .await
                .map_err(fail)?;
            let hinted = Arc::new(int96_as_micros(inferred.schema()));
            let metadata = ArrowReaderMetadata::try_new(
                inferred.metadata().clone(),
                reader_options().with_schema(hinted),
            )
            .map_err(fail)?;
            let (ordering, mask) = parquet_read_plan(&requested, &metadata)?;
            let row_indexes = ordering_needs_row_indexes(&ordering)
                .then(|| RowIndexBuilder::new(metadata.metadata().row_groups()).build())
                .transpose()?;
            let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata)
                .with_batch_size(BATCH_SIZE);
            if let Some(mask) = mask {
                builder = builder.with_projection(mask);
            }
            Ok::<_, Error>((builder.build().map_err(fail)?, ordering, row_indexes))
        })?;
        Ok(Self {
            stream,
            current: None,
            ordering,
            row_indexes,
            schema,
            location: file.location.to_string(),
            done: false,
        })
    }
}

impl Iterator for Int96MicrosRead {
    type Item = DeltaResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(reader) = self.current.as_mut() {
                match reader.next() {
                    Some(Ok(batch)) => {
                        return Some(
                            fixup_parquet_read(
                                batch,
                                &self.ordering,
                                self.row_indexes.as_mut(),
                                Some(&self.location),
                                Some(&self.schema),
                            )
                            .map(RecordBatch::from),
                        );
                    }
                    Some(Err(e)) => {
                        self.done = true;
                        self.current = None;
                        return Some(Err(e.into()));
                    }
                    None => self.current = None,
                }
            }
            if self.done {
                return None;
            }
            match block(self.stream.next_row_group()) {
                Ok(Some(reader)) => self.current = Some(reader),
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e.into()));
                }
            }
        }
    }
}

/// The footer of one data file.
#[allow(deprecated)]
fn footer(store: Arc<DynObjectStore>, file: &FileMeta) -> DeltaResult<Arc<ParquetMetaData>> {
    // Read only the footer: one extra range request per file with a
    // date/timestamp column, which is what it costs to know.
    let path = Path::from_url_path(file.location.path())?;
    let mut reader = ParquetObjectReader::new(store, path);
    if file.size != 0 {
        reader = reader.with_file_size(file.size);
    }
    Ok(block(async move { reader.get_metadata(None).await })?)
}

/// Which of `files` (`(path, size)`, the path relative to `table_root` as the
/// log spells it) a reader that does not rebase would misread: files Spark
/// wrote in its legacy hybrid calendar, and files storing INT96 timestamps,
/// which such a reader decodes as nanoseconds. delta-rs is such a reader, and
/// its DML and compaction write what it read back into new files.
pub fn legacy_calendar_files(
    engine: &SharedEngine,
    table_root: &url::Url,
    files: &[(String, u64)],
) -> DeltaResult<Vec<String>> {
    let Some(store) = engine.get_object_store_for_url(table_root) else {
        return Err(Error::generic(format!(
            "no object store is registered for {table_root}"
        )));
    };
    let mut out = Vec::new();
    for (path, size) in files {
        let location = table_root.join(path).map_err(|e| {
            Error::generic(format!("{path:?} is not a path under {table_root}: {e}"))
        })?;
        let file = FileMeta::new(location, 0, *size);
        let metadata = footer(store.clone(), &file)?;
        match RebaseSpec::from_footer(&metadata, file.location.as_str()) {
            Ok(spec) if !spec.own_read() => {}
            // A file the kernel cannot rebase either (INT96 and INT64
            // timestamps in two modes) is no less legacy.
            _ => out.push(path.clone()),
        }
    }
    Ok(out)
}

/// A Parquet handler that rebases files Spark wrote in the hybrid calendar.
struct RebasingParquet {
    inner: Arc<dyn ParquetHandler>,
    store: Arc<DynObjectStore>,
}

impl ParquetHandler for RebasingParquet {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        let rebase = needs_footer(&physical_schema);
        let maps = has_map_of_structs(&physical_schema);
        if !(rebase || maps) || files.is_empty() {
            return self
                .inner
                .read_parquet_files(files, physical_schema, predicate);
        }
        let mut plans = Vec::with_capacity(files.len());
        for file in files {
            let metadata = footer(self.store.clone(), file)?;
            let spec = if rebase {
                RebaseSpec::from_footer(&metadata, file.location.as_str())?
            } else {
                RebaseSpec::default()
            };
            let ordered = if maps {
                file_ordered(&physical_schema, &metadata)?
            } else {
                None
            };
            plans.push((spec, ordered));
        }
        if plans.iter().all(|(s, o)| !s.own_read() && o.is_none()) {
            return self
                .inner
                .read_parquet_files(files, physical_schema, predicate);
        }
        // Read file by file, so each batch is rebased with its own file's
        // spec. No row-group pushdown for a rebased file: its Parquet
        // statistics are in the Julian calendar too.
        let mut out: Vec<FileDataReadResultIterator> = Vec::with_capacity(files.len());
        for (file, (spec, ordered)) in files.iter().zip(plans) {
            if !spec.own_read() && ordered.is_none() {
                out.push(self.inner.read_parquet_files(
                    std::slice::from_ref(file),
                    physical_schema.clone(),
                    predicate.clone(),
                )?);
                continue;
            }
            let location = file.location.to_string();
            let read_schema = ordered.clone().unwrap_or_else(|| physical_schema.clone());
            let batches: Box<dyn Iterator<Item = DeltaResult<RecordBatch>> + Send> =
                if spec.int96_micros {
                    Box::new(Int96MicrosRead::open(
                        self.store.clone(),
                        file,
                        read_schema,
                    )?)
                } else {
                    Box::new(
                        self.inner
                            .read_parquet_files(
                                std::slice::from_ref(file),
                                read_schema,
                                if spec.any() { None } else { predicate.clone() },
                            )?
                            .map(|data| data?.try_into_record_batch()),
                    )
                };
            let requested = ordered.is_some().then(|| physical_schema.clone());
            out.push(Box::new(batches.map(move |batch| {
                let mut batch = batch?;
                if let Some(requested) = &requested {
                    batch = requested_order(batch, requested)?;
                }
                if spec.any() {
                    batch = rebase_batch(batch, &spec, &location)?;
                }
                Ok(Box::new(ArrowEngineData::new(batch)) as Box<dyn EngineData>)
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

// ------------------------------------------------------------------------
// Maps of structs, read in the file's field order.
//
// The kernel's Parquet reader rebuilds a MAP column whose value struct it has
// to reorder (`reorder_map` in delta_kernel's arrow_utils) with the map
// *entries* field's nullability -- always false -- for the column itself, so
// the first NULL map in a batch fails with "Found unmasked nulls for
// non-nullable StructArray field". A VARIANT always needs that reorder when
// Spark or Databricks wrote it: the file stores `value, metadata`, the kernel
// asks for `metadata, value`. So a MAP<STRING, VARIANT> with a NULL row could
// not be read at all. For such a file the structs under a map are requested
// in the order the file stores them (a VARIANT as the plain struct it is on
// disk), which leaves the kernel nothing to reorder there, and each batch is
// put back into the requested order here.

/// Whether `schema` has a MAP with a struct (or VARIANT) somewhere under it.
fn has_map_of_structs(schema: &SchemaRef) -> bool {
    fn under_map(t: &KernelType) -> bool {
        match t {
            KernelType::Struct(_) | KernelType::Variant(_) => true,
            KernelType::Array(a) => under_map(a.element_type()),
            KernelType::Map(m) => under_map(m.key_type()) || under_map(m.value_type()),
            _ => false,
        }
    }
    fn walk(t: &KernelType) -> bool {
        match t {
            KernelType::Struct(s) => s.fields().any(|f| walk(f.data_type())),
            KernelType::Array(a) => walk(a.element_type()),
            KernelType::Map(m) => under_map(m.key_type()) || under_map(m.value_type()),
            _ => false,
        }
    }
    schema.fields().any(|f| walk(f.data_type()))
}

/// `requested` with every struct under a map in `metadata`'s field order, or
/// None when that is already its order (or the file does not say).
fn file_ordered(
    requested: &SchemaRef,
    metadata: &ParquetMetaData,
) -> DeltaResult<Option<SchemaRef>> {
    let file = metadata.file_metadata();
    let arrow = delta_kernel::parquet::arrow::parquet_to_arrow_schema(
        file.schema_descr(),
        file.key_value_metadata(),
    )?;
    let mut changed = false;
    let mut fields = Vec::with_capacity(requested.num_fields());
    for field in requested.fields() {
        let mut field = field.clone();
        if let Ok(on_disk) = arrow.field_with_name(field.name()) {
            if let Some(t) = order_type(&field.data_type, on_disk.data_type(), false)? {
                field.data_type = t;
                changed = true;
            }
        }
        fields.push(field);
    }
    Ok(if changed {
        Some(Arc::new(delta_kernel::schema::StructType::try_new(fields)?))
    } else {
        None
    })
}

/// `want` in the order of `on_disk` where it is under a map; None if unchanged.
fn order_type(
    want: &KernelType,
    on_disk: &DataType,
    in_map: bool,
) -> DeltaResult<Option<KernelType>> {
    use delta_kernel::schema::{ArrayType, MapType, StructType};
    Ok(match (want, on_disk) {
        (KernelType::Map(m), DataType::Map(entries, _)) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Ok(None);
            };
            if kv.len() != 2 {
                return Ok(None);
            }
            let key = order_type(m.key_type(), kv[0].data_type(), true)?;
            let value = order_type(m.value_type(), kv[1].data_type(), true)?;
            if key.is_none() && value.is_none() {
                return Ok(None);
            }
            Some(KernelType::Map(Box::new(MapType::new(
                key.unwrap_or_else(|| m.key_type().clone()),
                value.unwrap_or_else(|| m.value_type().clone()),
                m.value_contains_null(),
            ))))
        }
        (KernelType::Array(a), DataType::List(f) | DataType::LargeList(f)) => {
            order_type(a.element_type(), f.data_type(), in_map)?.map(|element| {
                KernelType::Array(Box::new(ArrayType::new(element, a.contains_null())))
            })
        }
        (KernelType::Variant(s), DataType::Struct(disk)) if in_map => {
            // Only the unshredded form: anything else is left for the
            // kernel's reader to refuse as shredded.
            let names: Vec<&str> = disk.iter().map(|f| f.name().as_str()).collect();
            if *want != KernelType::unshredded_variant() || names != ["value", "metadata"] {
                return Ok(None);
            }
            let fields = ["value", "metadata"]
                .iter()
                .filter_map(|n| s.fields().find(|f| f.name() == n).cloned());
            Some(KernelType::Struct(Box::new(StructType::try_new(fields)?)))
        }
        (KernelType::Struct(s), DataType::Struct(disk)) => {
            let mut changed = false;
            let mut fields = Vec::with_capacity(s.num_fields());
            for field in s.fields() {
                let mut field = field.clone();
                if let Some(d) = disk.iter().find(|d| d.name() == field.name()) {
                    if let Some(t) = order_type(&field.data_type, d.data_type(), in_map)? {
                        field.data_type = t;
                        changed = true;
                    }
                }
                fields.push(field);
            }
            if in_map {
                let position = |name: &str| disk.iter().position(|d| d.name() == name);
                let before: Vec<String> = fields.iter().map(|f| f.name().clone()).collect();
                // Fields the file lacks keep their place after those it has.
                fields.sort_by_key(|f| position(f.name()).unwrap_or(usize::MAX));
                changed |= fields.iter().map(|f| f.name()).ne(before.iter());
            }
            if !changed {
                return Ok(None);
            }
            Some(KernelType::Struct(Box::new(StructType::try_new(fields)?)))
        }
        _ => None,
    })
}

/// `batch`, read with [`file_ordered`]'s schema, with its structs back in the
/// order of `requested`.
fn requested_order(batch: RecordBatch, requested: &SchemaRef) -> DeltaResult<RecordBatch> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(schema.fields().len());
    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        let column = match requested.field(field.name()) {
            Some(want) => reorder_array(column, want.data_type())?,
            None => column.clone(),
        };
        fields.push(Arc::new(
            Field::clone(field).with_data_type(column.data_type().clone()),
        ));
        columns.push(column);
    }
    let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
}

fn reorder_array(array: &ArrayRef, want: &KernelType) -> DeltaResult<ArrayRef> {
    let retyped = |f: &FieldRef, a: &ArrayRef| -> FieldRef {
        Arc::new(Field::clone(f).with_data_type(a.data_type().clone()))
    };
    Ok(match (want, array.data_type()) {
        (KernelType::Struct(s) | KernelType::Variant(s), DataType::Struct(have)) => {
            let sa = array.as_struct();
            let mut fields = Vec::with_capacity(s.num_fields());
            let mut columns = Vec::with_capacity(s.num_fields());
            for want_field in s.fields() {
                let Some(i) = have.iter().position(|f| f.name() == want_field.name()) else {
                    return Err(Error::generic(format!(
                        "field {:?} is missing from a struct read in file order",
                        want_field.name()
                    )));
                };
                let column = reorder_array(sa.column(i), want_field.data_type())?;
                fields.push(retyped(&have[i], &column));
                columns.push(column);
            }
            Arc::new(StructArray::try_new(
                fields.into(),
                columns,
                sa.nulls().cloned(),
            )?)
        }
        (KernelType::Array(a), DataType::List(f)) => {
            let l = array.as_list::<i32>();
            let values = reorder_array(l.values(), a.element_type())?;
            Arc::new(ListArray::try_new(
                retyped(f, &values),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?)
        }
        (KernelType::Array(a), DataType::LargeList(f)) => {
            let l = array.as_list::<i64>();
            let values = reorder_array(l.values(), a.element_type())?;
            Arc::new(LargeListArray::try_new(
                retyped(f, &values),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )?)
        }
        (KernelType::Map(m), DataType::Map(entries_field, ordered)) => {
            let map = array.as_map();
            let entries = map.entries();
            let DataType::Struct(kv) = entries_field.data_type() else {
                return Ok(array.clone());
            };
            let key = reorder_array(entries.column(0), m.key_type())?;
            let value = reorder_array(entries.column(1), m.value_type())?;
            let kv_fields = vec![retyped(&kv[0], &key), retyped(&kv[1], &value)];
            let entries =
                StructArray::try_new(kv_fields.into(), vec![key, value], entries.nulls().cloned())?;
            Arc::new(MapArray::try_new(
                Arc::new(Field::clone(entries_field).with_data_type(entries.data_type().clone())),
                map.offsets().clone(),
                entries,
                map.nulls().cloned(),
                *ordered,
            )?)
        }
        _ => array.clone(),
    })
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
            ..RebaseSpec::default()
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
            zone: WriterZone::Named("America/Los_Angeles".into()),
            ..RebaseSpec::default()
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
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(old)]).unwrap();
        let err = rebase_batch(batch, &spec, "f").unwrap_err();
        assert!(err.to_string().contains("America/Los_Angeles"), "{err}");

        // After the calendar switch but before 1900 the zone's own offsets
        // still differ (Spark stores 1850-06-01T19:00Z 422 s early in LA).
        let lmt = TimestampMicrosecondArray::from(vec![-3773710378000000]).with_timezone("UTC");
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(lmt)]).unwrap();
        assert!(rebase_batch(batch, &spec, "f").is_err());
        let from_1900 =
            TimestampMicrosecondArray::from(vec![ZONED_REBASE_LIMIT_MICROS]).with_timezone("UTC");
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(from_1900)]).unwrap();
        assert!(rebase_batch(batch, &spec, "f").is_ok());

        // No zone recorded: Spark would use its session zone, unknown here.
        let unknown = RebaseSpec {
            timestamps: true,
            zone: WriterZone::Unknown,
            ..RebaseSpec::default()
        };
        let old = TimestampMicrosecondArray::from(vec![-14816604303211000]).with_timezone("UTC");
        let batch = RecordBatch::try_new(schema, vec![Arc::new(old)]).unwrap();
        let err = rebase_batch(batch, &unknown, "f").unwrap_err();
        assert!(err.to_string().contains("does not record"), "{err}");
    }

    #[test]
    fn int96_hint_retypes_nanoseconds_only() {
        use arrow::datatypes::Fields;

        let nanos = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let schema = Schema::new(vec![
            Field::new("ts", nanos.clone(), true),
            Field::new(
                "st",
                DataType::Struct(Fields::from(vec![Field::new("t", nanos.clone(), true)])),
                true,
            ),
            Field::new_list("l", Field::new("item", nanos, true), true),
            Field::new("d", DataType::Date32, true),
        ]);
        let micros = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let hinted = int96_as_micros(&schema);
        assert_eq!(hinted.field(0).data_type(), &micros);
        assert_eq!(
            hinted.field(1).data_type(),
            &DataType::Struct(Fields::from(vec![Field::new("t", micros.clone(), true)]))
        );
        assert_eq!(
            hinted.field(2).data_type(),
            &DataType::List(Arc::new(Field::new("item", micros, true)))
        );
        assert_eq!(hinted.field(3).data_type(), &DataType::Date32);
    }
}
