//! Turning a kernel scan into an Arrow stream.
//!
//! Correctness rule, enforced by construction: **rows are never reordered.**
//! A deletion vector is a positional keep-mask over a file's rows in physical
//! order, so any repartitioning, limit pushdown or pre-filtering applied before
//! the mask lands silently returns the right *number* of rows made of the wrong
//! records. `Scan::execute` applies DVs and the physical->logical transform
//! itself; this module's only job is to hand the resulting batches onward in
//! the order kernel produced them.
//!
//! A *file-restricted* scan ([`KernelBatchReader::try_new_restricted`]) is the
//! one place this module drives the read itself. Kernel's `Scan::execute` has
//! no hook for choosing files, so the restricted path is a line-for-line
//! mirror of it with one extra step: scan files whose path is not in the
//! caller's set are dropped from the planned list before any data or
//! deletion-vector I/O. Everything after that -- DV mask, physical->logical
//! transform, per-file batch order -- is exactly what `execute` does, so a
//! restricted scan over every file equals the full scan.
//!
//! A *positional* scan ([`KernelBatchReader::try_new_positional`]) is the same
//! restricted read with each surviving row tagged by its data file's log path
//! and its physical row index within that file. Those two values are exactly
//! what a deletion vector addresses, so DML computes its DVs from them.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatchReader;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::scan::state::{transform_to_logical, ScanFile};
use delta_kernel::scan::{Scan, ScanMetadata};
use delta_kernel::schema::SchemaRef;
use delta_kernel::{DeltaResult, Engine, EngineData, Error, FileMeta};
use url::Url;

use crate::error::Result;

/// Streams a kernel scan as Arrow record batches, preserving order.
pub struct KernelBatchReader {
    schema: ArrowSchemaRef,
    iter: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>,
    /// Set once the stream has failed; nothing is read after an error.
    failed: bool,
}

/// Map requested column names onto the schema's own spelling.
///
/// Delta column names are case-insensitive, and kernel's `project` matches
/// exactly, so `["ID"]` against a column `id` used to fail with the bare
/// message "ID". An empty list is refused outright: kernel cannot build a
/// zero-column batch (it panics, which aborts the process mid-stream).
pub fn resolve_columns(
    schema: &delta_kernel::schema::StructType,
    columns: &[String],
) -> Result<Vec<String>> {
    if columns.is_empty() {
        return Err(crate::error::NativeError::Invalid(
            "columns=[] selects no columns; pass None to read every column, or name at least one"
                .to_string(),
        ));
    }
    let mut out: Vec<String> = Vec::with_capacity(columns.len());
    for name in columns {
        let field = schema.field(name).or_else(|| {
            let mut matches = schema
                .fields()
                .filter(|f| f.name().eq_ignore_ascii_case(name));
            let first = matches.next();
            // Ambiguous only if the schema itself differs just by case.
            first.filter(|_| matches.next().is_none())
        });
        let Some(field) = field else {
            let known: Vec<&str> = schema.fields().map(|f| f.name().as_str()).collect();
            return Err(crate::error::NativeError::Invalid(format!(
                "column {name:?} is not in the table schema; columns are {known:?}"
            )));
        };
        if out.iter().any(|c| c == field.name()) {
            return Err(crate::error::NativeError::Invalid(format!(
                "column {name:?} is requested more than once"
            )));
        }
        out.push(field.name().clone());
    }
    Ok(out)
}

/// Name of the row-index column added to a partition-columns-only read so
/// the Parquet read schema is never empty; it never reaches the caller.
pub const ROW_COUNT_COLUMN: &str = "__deltaswamp_row_index";

/// Name of the column a positional scan adds: each row's data file, as its
/// path is stored in the log (relative and URL-encoded, as `files()` shows it).
pub const FILE_PATH_COLUMN: &str = "__deltaswamp_file";

/// Name of the column a positional scan adds on request: each row's stable
/// row id, on a table with row tracking enabled.
pub const ROW_ID_COLUMN: &str = "__deltaswamp_row_id";

impl KernelBatchReader {
    /// This reader with the top-level column `name` removed from its schema
    /// and from every batch (row counts are kept).
    pub fn without_column(mut self, name: &str) -> Self {
        let Ok(index) = self.schema.index_of(name) else {
            return self;
        };
        let keep: Vec<usize> = (0..self.schema.fields().len())
            .filter(|i| *i != index)
            .collect();
        let Ok(schema) = self.schema.project(&keep) else {
            return self;
        };
        self.schema = Arc::new(schema);
        let inner = std::mem::replace(&mut self.iter, Box::new(std::iter::empty()));
        self.iter = Box::new(inner.map(move |item| {
            let data = item?;
            let batch = data.try_into_record_batch()?;
            let batch = batch.project(&keep)?;
            Ok(Box::new(ArrowEngineData::new(batch)) as Box<dyn EngineData>)
        }));
        self
    }

    pub fn try_new(scan: &Scan, engine: Arc<dyn Engine>) -> Result<Self> {
        let iter = scan.execute(engine)?;
        Self::from_parts(scan.logical_schema().as_ref(), iter)
    }

    /// Read only the data files whose log path (as stored in the `add`
    /// action, i.e. as `files()` reports it) is in `paths`.
    ///
    /// Paths not in the snapshot are ignored; an empty set yields an empty
    /// stream with the scan's logical schema. Predicate-based file skipping
    /// still applies on top of the restriction. Files are read in the order
    /// given (a path given twice is read once), after one log replay.
    pub fn try_new_restricted(
        scan: &Scan,
        engine: Arc<dyn Engine>,
        paths: Vec<String>,
    ) -> Result<Self> {
        let iter = RestrictedScan::new(scan, engine, Some(paths), false, false)?;
        Self::from_parts(scan.logical_schema().as_ref(), iter)
    }

    /// Read with every row tagged by its data file ([`FILE_PATH_COLUMN`]).
    ///
    /// The scan's schema must already carry a row-index metadata column; with
    /// it, each row is addressed by `(file, physical row index)`, which is what
    /// a deletion vector records. Rows a deletion vector already removed are
    /// not returned, but the survivors keep their physical indexes.
    pub fn try_new_positional(
        scan: &Scan,
        engine: Arc<dyn Engine>,
        paths: Option<Vec<String>>,
    ) -> Result<Self> {
        let iter = RestrictedScan::new(scan, engine, paths, true, false)?;
        let mut reader = Self::from_parts(scan.logical_schema().as_ref(), iter)?;
        let mut fields: Vec<arrow::datatypes::FieldRef> =
            reader.schema.fields().iter().cloned().collect();
        fields.push(Arc::new(arrow::datatypes::Field::new(
            FILE_PATH_COLUMN,
            arrow::datatypes::DataType::Utf8,
            false,
        )));
        reader.schema = Arc::new(ArrowSchema::new_with_metadata(
            fields,
            reader.schema.metadata().clone(),
        ));
        Ok(reader)
    }

    /// Read `paths` in order, as runs of `groups[i]` files each (a
    /// compaction's bins), every row tagged by its file
    /// ([`FILE_PATH_COLUMN`], dictionary-encoded) and the batches of a run
    /// merged into fewer, larger ones; see [`Coalesced`].
    pub fn try_new_grouped(
        scan: &Scan,
        engine: Arc<dyn Engine>,
        paths: Vec<String>,
        groups: Vec<usize>,
    ) -> Result<Self> {
        if groups.iter().sum::<usize>() != paths.len() {
            return Err(crate::error::NativeError::Invalid(format!(
                "file_groups sizes add up to {}, but {} files were given",
                groups.iter().sum::<usize>(),
                paths.len()
            )));
        }
        let mut group_of = HashMap::new();
        let mut next = paths.iter();
        for (group, size) in groups.into_iter().enumerate() {
            for path in next.by_ref().take(size) {
                group_of.insert(path.clone(), group);
            }
        }
        let iter = RestrictedScan::new(scan, engine, Some(paths), true, true)?;
        let mut reader = Self::from_parts(
            scan.logical_schema().as_ref(),
            Coalesced {
                inner: iter,
                group_of,
                held: Vec::new(),
                held_rows: 0,
                held_bytes: 0,
                held_group: None,
                done: false,
            },
        )?;
        let mut fields: Vec<arrow::datatypes::FieldRef> =
            reader.schema.fields().iter().cloned().collect();
        fields.push(Arc::new(arrow::datatypes::Field::new(
            FILE_PATH_COLUMN,
            arrow::datatypes::DataType::Dictionary(
                Box::new(arrow::datatypes::DataType::Int32),
                Box::new(arrow::datatypes::DataType::Utf8),
            ),
            false,
        )));
        reader.schema = Arc::new(ArrowSchema::new_with_metadata(
            fields,
            reader.schema.metadata().clone(),
        ));
        Ok(reader)
    }

    /// Wrap any kernel data iterator (a table scan or a change-feed scan).
    pub fn from_parts(
        logical_schema: &delta_kernel::schema::StructType,
        iter: impl Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send + 'static,
    ) -> Result<Self> {
        let schema: ArrowSchema = logical_schema.try_into_arrow()?;
        Ok(Self {
            schema: Arc::new(schema),
            iter: Box::new(iter),
            failed: false,
        })
    }
}

impl Iterator for KernelBatchReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // A 1:1 pass-through. Do not batch, coalesce, reorder or
        // filter here; see the module docs.
        if self.failed {
            return None;
        }
        // This runs inside the Arrow C stream callback, which cannot unwind:
        // a kernel panic here would abort the whole Python process. Turn it
        // into a stream error instead.
        let iter = &mut self.iter;
        let item = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            iter.next().map(|data| {
                data.try_into_record_batch().map_err(|e| {
                    // Kept as an I/O error so pyarrow raises OSError (EIO)
                    // rather than ArrowInvalid: a dropped connection or a
                    // vacuumed file mid-read is not bad input.
                    if is_io_error(&e) {
                        let msg = e.to_string();
                        ArrowError::IoError(msg.clone(), std::io::Error::other(msg))
                    } else {
                        ArrowError::ExternalError(Box::new(e))
                    }
                })
            })
        }))
        .unwrap_or_else(|panic| {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Some(Err(ArrowError::ExternalError(
                format!("the kernel panicked while reading: {msg}").into(),
            )))
        });
        match item {
            Some(Err(e)) => {
                self.failed = true;
                // pyo3-arrow copies the message into a CString with `expect`,
                // so a NUL byte (e.g. from a path in a corrupt log) would
                // panic inside the C callback and abort the process.
                let clean = |m: String| m.replace('\0', "\\0");
                Some(Err(match e {
                    ArrowError::IoError(msg, io) => ArrowError::IoError(clean(msg), io),
                    ArrowError::ExternalError(inner) => {
                        ArrowError::ExternalError(clean(inner.to_string()).into())
                    }
                    other => ArrowError::ExternalError(clean(other.to_string()).into()),
                }))
            }
            other => other,
        }
    }
}

impl RecordBatchReader for KernelBatchReader {
    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }
}

/// True if `err`, or anything in its source chain, is a storage failure.
///
/// The Parquet reader wraps object-store errors in Arrow errors, so the
/// top-level kernel variant alone does not tell.
fn is_io_error(err: &Error) -> bool {
    if matches!(
        err,
        Error::ObjectStore(_) | Error::IOError(_) | Error::FileNotFound(_) | Error::Reqwest(_)
    ) {
        return true;
    }
    let mut source: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(err);
    while let Some(e) = source {
        if e.is::<delta_kernel::object_store::Error>() || e.is::<std::io::Error>() {
            return true;
        }
        if let Some(ArrowError::ExternalError(inner)) = e.downcast_ref::<ArrowError>() {
            let inner: &(dyn std::error::Error + 'static) = inner.as_ref();
            if inner.is::<delta_kernel::object_store::Error>() || inner.is::<std::io::Error>() {
                return true;
            }
        }
        source = e.source();
    }
    false
}

type DataIter = Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>;

/// The file currently being read: its Parquet batches, the part of its DV
/// keep-mask not yet consumed, and its physical->logical transform.
struct OpenFile {
    /// The file's path as the log stores it.
    path: String,
    batches: DataIter,
    selection: Option<Vec<bool>>,
    transform: Option<delta_kernel::ExpressionRef>,
}

/// `Scan::execute`, restricted to a set of file paths. See the module docs.
struct RestrictedScan {
    metadata: Box<dyn Iterator<Item = DeltaResult<ScanMetadata>> + Send>,
    engine: Arc<dyn Engine>,
    table_root: Url,
    physical_schema: SchemaRef,
    logical_schema: SchemaRef,
    /// The files to read; `None` reads every file the scan plans.
    paths: Option<HashSet<String>>,
    /// The order to read `paths` in, until the replay that finds them all.
    order: Option<Vec<String>>,
    /// Append [`FILE_PATH_COLUMN`] to every batch.
    tag_path: bool,
    /// ... dictionary-encoded (a grouped scan), rather than as strings.
    dictionary: bool,
    /// Files in flight before any has finished (then the budget decides).
    learned: bool,
    pending: VecDeque<ScanFile>,
    current: Option<OpenFile>,
    finished: bool,
    /// Read the files ahead of the consumer, in order (an ordered scan).
    prefetch: bool,
    /// Files being read in the background, with their decoded-size estimates.
    inflight: VecDeque<(tokio::task::JoinHandle<DeltaResult<Loaded>>, u64)>,
    inflight_bytes: u64,
    /// The most a file has grown when decoded (decoded / file bytes) so far.
    decoded_ratio: f64,
    /// A prefetched file's batches, finished, not yet handed on.
    ready: VecDeque<Box<dyn EngineData>>,
}

/// Files read ahead at most, and their estimated decoded bytes at most (one
/// file is always read, whatever its size).
///
/// Reading one small file at a time left an OPTIMIZE of 20,000 files waiting
/// on each read in turn (5 s of its 7); delta-rs reads them concurrently.
const PREFETCH_FILES: usize = 16;
const PREFETCH_BYTES: u64 = 256 << 20;

/// A file read in full in the background: its physical batches, in order.
struct Loaded {
    /// Logical, masked (and tagged) batches, as the scan hands them on.
    batches: Vec<Box<dyn EngineData>>,
    /// Their Arrow memory, and the file's size, to learn the decoded ratio.
    decoded: usize,
    size: u64,
}

/// Open one file exactly as `Scan::execute` does.
fn open_file(
    engine: &dyn Engine,
    table_root: &Url,
    physical_schema: &SchemaRef,
    file: ScanFile,
) -> DeltaResult<OpenFile> {
    let location = table_root.join(&file.path)?;
    let selection = file.dv_info.get_selection_vector(engine, table_root)?;
    let meta = FileMeta {
        last_modified: 0,
        size: file
            .size
            .try_into()
            .map_err(|_| Error::generic("Unable to convert scan file size into FileSize"))?,
        location,
    };
    // No predicate pushdown into the reader: row-level filtering before
    // the DV mask would misalign it (kernel disables it for the same reason).
    let mut batches = engine
        .parquet_handler()
        .read_parquet_files(&[meta], physical_schema.clone(), None)?
        .peekable();
    let expect_data = file.stats.as_ref().is_some_and(|s| s.num_records > 0);
    if expect_data && batches.peek().is_none() {
        return Err(Error::internal_error(format!(
            "ParquetHandler returned no data for file '{}' although its stats report rows",
            file.path
        )));
    }
    Ok(OpenFile {
        path: file.path,
        batches: Box::new(batches),
        selection,
        transform: file.transform,
    })
}

impl RestrictedScan {
    fn new(
        scan: &Scan,
        engine: Arc<dyn Engine>,
        order: Option<Vec<String>>,
        tag_path: bool,
        dictionary: bool,
    ) -> Result<Self> {
        let paths = order.as_ref().map(|o| o.iter().cloned().collect());
        let prefetch = order.is_some();
        Ok(Self {
            metadata: Box::new(scan.scan_metadata(engine.as_ref())?),
            engine,
            table_root: scan.snapshot().table_root().clone(),
            physical_schema: scan.physical_schema().clone(),
            logical_schema: scan.logical_schema().clone(),
            paths,
            order,
            tag_path,
            dictionary,
            learned: false,
            pending: VecDeque::new(),
            current: None,
            finished: false,
            prefetch,
            inflight: VecDeque::new(),
            inflight_bytes: 0,
            decoded_ratio: 4.0,
            ready: VecDeque::new(),
        })
    }

    /// Queue the wanted files of the next scan-metadata batch. Returns false
    /// when the log replay is exhausted.
    fn plan_next(&mut self) -> DeltaResult<bool> {
        fn collect(files: &mut Vec<ScanFile>, file: ScanFile) {
            files.push(file);
        }
        if let (Some(order), Some(paths)) = (self.order.take(), self.paths.as_ref()) {
            // Read in the caller's order: every file is found first (one
            // replay), so a caller reading bins of files in turn gets each
            // bin's rows together and can stream them. Scan files are small;
            // the data and vectors are still read only as each is opened.
            let mut found: HashMap<String, ScanFile> = HashMap::new();
            for metadata in self.metadata.by_ref() {
                for file in metadata?.visit_scan_files(Vec::new(), collect)? {
                    if paths.contains(&file.path) {
                        found.insert(file.path.clone(), file);
                    }
                }
            }
            self.pending
                .extend(order.iter().filter_map(|path| found.remove(path)));
            return Ok(true);
        }
        let Some(metadata) = self.metadata.next() else {
            return Ok(false);
        };
        // Visiting only decodes the scan rows; the DV is not read until the
        // file is opened, so dropping a file here costs no I/O.
        let files = metadata?.visit_scan_files(Vec::new(), collect)?;
        let paths = &self.paths;
        self.pending.extend(
            files
                .into_iter()
                .filter(|file| paths.as_ref().is_none_or(|p| p.contains(&file.path))),
        );
        Ok(true)
    }

    /// Open one file exactly as `Scan::execute` does.
    fn open(&self, file: ScanFile) -> DeltaResult<OpenFile> {
        open_file(
            self.engine.as_ref(),
            &self.table_root,
            &self.physical_schema,
            file,
        )
    }

    /// Start reading queued files in the background, in order, while the
    /// estimated decoded bytes in flight stay under [`PREFETCH_BYTES`].
    fn fill_prefetch(&mut self) {
        // Until a file has been read, nothing says how much one decodes to:
        // a file of 120 KB held 96 MB of rows, and sixteen of them at once
        // took gigabytes. So one runs ahead until then.
        let limit = if self.learned { PREFETCH_FILES } else { 2 };
        while self.inflight.len() < limit {
            let Some(file) = self.pending.front() else {
                return;
            };
            let estimate = (file.size.max(0) as f64 * self.decoded_ratio) as u64;
            if !self.inflight.is_empty() && self.inflight_bytes + estimate > PREFETCH_BYTES {
                return;
            }
            let Some(file) = self.pending.pop_front() else {
                return;
            };
            let size = file.size.max(1) as u64;
            let engine = self.engine.clone();
            let root = self.table_root.clone();
            let physical = self.physical_schema.clone();
            let logical = self.logical_schema.clone();
            let tag_path = self.tag_path;
            let dictionary = self.dictionary;
            // The whole file -- read, physical->logical transform, DV mask --
            // runs on a worker, so files are finished concurrently and
            // handed on in order.
            let handle = crate::runtime::runtime().spawn_blocking(move || {
                let mut open = open_file(engine.as_ref(), &root, &physical, file)?;
                let mut batches = Vec::new();
                let mut decoded = 0usize;
                while let Some(item) = open.batches.next() {
                    let batch =
                        Self::finish_batch(engine.as_ref(), &physical, &logical, &mut open, item?)?;
                    let batch = if tag_path {
                        Self::tag_with_path(batch, &open.path, dictionary)?
                    } else {
                        batch
                    };
                    decoded += batch
                        .as_ref()
                        .any_ref()
                        .downcast_ref::<ArrowEngineData>()
                        .map_or(0, |b| b.record_batch().get_array_memory_size());
                    batches.push(batch);
                }
                Ok(Loaded {
                    batches,
                    decoded,
                    size,
                })
            });
            self.inflight.push_back((handle, estimate));
            self.inflight_bytes += estimate;
        }
    }

    /// The next prefetched file's finished batches; None when none is in flight.
    fn next_prefetched(&mut self) -> Option<DeltaResult<Vec<Box<dyn EngineData>>>> {
        let (handle, estimate) = self.inflight.pop_front()?;
        self.inflight_bytes = self.inflight_bytes.saturating_sub(estimate);
        let loaded = match crate::runtime::block_on(handle) {
            Ok(Ok(loaded)) => loaded,
            Ok(Err(e)) => return Some(Err(e)),
            Err(e) => {
                return Some(Err(Error::generic(format!(
                    "reading a data file failed: {e}"
                ))))
            }
        };
        // Learn how much the table's files grow when decoded, so the next
        // files in flight are budgeted by what they will hold in memory.
        self.learned = true;
        let ratio = loaded.decoded as f64 / loaded.size as f64;
        if ratio > self.decoded_ratio {
            self.decoded_ratio = ratio;
        }
        Some(Ok(loaded.batches))
    }

    fn next_batch(&mut self) -> Option<DeltaResult<Box<dyn EngineData>>> {
        loop {
            if let Some(batch) = self.ready.pop_front() {
                return Some(Ok(batch));
            }
            if let Some(open) = self.current.as_mut() {
                match open.batches.next() {
                    Some(Ok(physical)) => {
                        let batch = Self::finish_batch(
                            self.engine.as_ref(),
                            &self.physical_schema,
                            &self.logical_schema,
                            open,
                            physical,
                        );
                        return Some(if self.tag_path {
                            batch.and_then(|b| Self::tag_with_path(b, &open.path, self.dictionary))
                        } else {
                            batch
                        });
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => self.current = None,
                }
            }
            if self.prefetch {
                self.fill_prefetch();
                if let Some(next) = self.next_prefetched() {
                    match next {
                        Ok(batches) => self.ready.extend(batches),
                        Err(e) => return Some(Err(e)),
                    }
                    continue;
                }
            }
            if let Some(file) = self.pending.pop_front() {
                match self.open(file) {
                    Ok(open) => self.current = Some(open),
                    Err(e) => return Some(Err(e)),
                }
                continue;
            }
            match self.plan_next() {
                Ok(true) => continue,
                Ok(false) => return None,
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Transform one physical batch to logical form, then apply the slice of
    /// the file's DV mask that covers it. Mirrors `Scan::execute`.
    fn finish_batch(
        engine: &dyn Engine,
        physical_schema: &SchemaRef,
        logical_schema: &SchemaRef,
        open: &mut OpenFile,
        physical: Box<dyn EngineData>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let logical = transform_to_logical(
            engine,
            physical,
            physical_schema,
            logical_schema,
            open.transform.clone(),
        )?;
        let len = logical.len();
        // The mask may be shorter than the file (trailing rows are kept); it
        // is consumed front to back as batches arrive in file order.
        match open.selection.take() {
            Some(mut mask) => {
                if len < mask.len() {
                    open.selection = Some(mask.split_off(len));
                }
                logical.apply_selection_vector(mask)
            }
            None => Ok(logical),
        }
    }
}

impl RestrictedScan {
    /// `data` with a constant [`FILE_PATH_COLUMN`] of `path` appended.
    ///
    /// `dictionary` encodes it as one dictionary value and four bytes a row:
    /// written out as a string, a path of sixty bytes on each row of a file
    /// that compresses to nothing took more memory than its data.
    fn tag_with_path(
        data: Box<dyn EngineData>,
        path: &str,
        dictionary: bool,
    ) -> DeltaResult<Box<dyn EngineData>> {
        use arrow::array::Array as _;
        use arrow::datatypes::{DataType, Int32Type};

        let batch = data.try_into_record_batch()?;
        let (paths, kind): (arrow::array::ArrayRef, DataType) = if dictionary {
            let keys = arrow::array::Int32Array::from(vec![0; batch.num_rows()]);
            let values = Arc::new(arrow::array::StringArray::from(vec![path]));
            let array = arrow::array::DictionaryArray::<Int32Type>::try_new(keys, values)?;
            let kind = array.data_type().clone();
            (Arc::new(array), kind)
        } else {
            (
                Arc::new(arrow::array::StringArray::from(vec![
                    path;
                    batch.num_rows()
                ])),
                DataType::Utf8,
            )
        };
        let mut fields: Vec<arrow::datatypes::FieldRef> =
            batch.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(arrow::datatypes::Field::new(
            FILE_PATH_COLUMN,
            kind,
            false,
        )));
        let mut columns = batch.columns().to_vec();
        columns.push(paths);
        let schema = Arc::new(ArrowSchema::new(fields));
        let tagged = RecordBatch::try_new(schema, columns)?;
        Ok(Box::new(ArrowEngineData::new(tagged)))
    }
}

/// Rows and Arrow bytes a coalesced batch reaches before it is handed on.
const COALESCE_ROWS: usize = 128 * 1024;
const COALESCE_BYTES: usize = 32 << 20;

/// A positional scan's batches, merged across consecutive files of one group.
///
/// A compaction reads its bins' files in turn, and 20,000 one-file batches
/// cost as much to hand across to Python as reading them did. Batches are
/// merged only within a group (a bin), so every batch still belongs to one;
/// rows keep their order, and each keeps its file tag.
struct Coalesced {
    inner: RestrictedScan,
    group_of: HashMap<String, usize>,
    held: Vec<RecordBatch>,
    held_rows: usize,
    held_bytes: usize,
    held_group: Option<usize>,
    done: bool,
}

impl Coalesced {
    fn flush(&mut self) -> Option<DeltaResult<Box<dyn EngineData>>> {
        if self.held.is_empty() {
            return None;
        }
        let held = std::mem::take(&mut self.held);
        self.held_rows = 0;
        self.held_bytes = 0;
        let merged = if held.len() == 1 {
            Ok(held.into_iter().next().expect("one batch"))
        } else {
            arrow::compute::concat_batches(&held[0].schema(), &held).map_err(Error::from)
        };
        Some(merged.map(|batch| Box::new(ArrowEngineData::new(batch)) as Box<dyn EngineData>))
    }
}

impl Iterator for Coalesced {
    type Item = DeltaResult<Box<dyn EngineData>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let data = match self.inner.next() {
                Some(Ok(data)) => data,
                Some(Err(e)) => {
                    self.done = true;
                    return Some(Err(e));
                }
                None => {
                    self.done = true;
                    return self.flush();
                }
            };
            let batch = match data.try_into_record_batch() {
                Ok(batch) => batch,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            if batch.num_rows() == 0 {
                continue;
            }
            let group = batch
                .column_by_name(FILE_PATH_COLUMN)
                .and_then(|c| {
                    c.as_any()
                        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>(
                        )
                })
                .and_then(|paths| {
                    let values = paths
                        .values()
                        .as_any()
                        .downcast_ref::<arrow::array::StringArray>()?;
                    let key = paths.keys().value(0);
                    self.group_of.get(values.value(key as usize)).copied()
                });
            let bytes = batch.get_array_memory_size();
            let fits = group.is_some()
                && group == self.held_group
                && self.held_rows + batch.num_rows() <= COALESCE_ROWS
                && self.held_bytes + bytes <= COALESCE_BYTES;
            let out = if fits { None } else { self.flush() };
            self.held_group = group;
            self.held_rows += batch.num_rows();
            self.held_bytes += bytes;
            self.held.push(batch);
            if out.is_some() {
                return out;
            }
        }
    }
}

impl Iterator for RestrictedScan {
    type Item = DeltaResult<Box<dyn EngineData>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let item = self.next_batch();
        // Stop after the first error rather than reading on past it.
        if !matches!(item, Some(Ok(_))) {
            self.finished = true;
        }
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::PySnapshot;
    use delta_kernel::schema::{DataType, StructField, StructType};

    fn schema() -> StructType {
        StructType::try_new([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable("Name", DataType::STRING),
        ])
        .unwrap()
    }

    #[test]
    fn columns_resolve_case_insensitively_to_the_schema_spelling() {
        let cols = resolve_columns(&schema(), &["ID".into(), "name".into()]).unwrap();
        assert_eq!(cols, ["id", "Name"]);
    }

    #[test]
    fn empty_unknown_and_duplicate_columns_are_clear_errors() {
        let err = resolve_columns(&schema(), &[]).unwrap_err();
        assert!(err.to_string().contains("columns=[]"), "{err}");
        let err = resolve_columns(&schema(), &["nope".into()]).unwrap_err();
        assert!(err.to_string().contains("not in the table schema"), "{err}");
        let err = resolve_columns(&schema(), &["id".into(), "ID".into()]).unwrap_err();
        assert!(err.to_string().contains("more than once"), "{err}");
    }

    #[test]
    fn a_panicking_kernel_iterator_becomes_a_stream_error() {
        let iter =
            std::iter::from_fn(|| -> Option<DeltaResult<Box<dyn EngineData>>> { panic!("boom") });
        let mut reader = KernelBatchReader::from_parts(&schema(), iter).unwrap();
        let err = reader.next().unwrap().unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
        assert!(reader.next().is_none(), "nothing is read after an error");
    }

    #[test]
    fn stream_error_messages_carry_no_nul_bytes() {
        let iter = std::iter::once(Err(Error::generic("bad path a\0b")));
        let mut reader = KernelBatchReader::from_parts(&schema(), iter).unwrap();
        let err = reader.next().unwrap().unwrap_err();
        assert!(!err.to_string().contains('\0'), "{err}");
    }

    #[test]
    fn table_roots_parse_paths_and_refuse_fragments() {
        let url = PySnapshot::table_root_url("/tmp/my table").unwrap();
        assert_eq!(url.as_str(), "file:///tmp/my%20table/");
        let url = PySnapshot::table_root_url("rel/t").unwrap();
        assert!(url.path().ends_with("/rel/t/"), "{url}");
        assert!(url.scheme() == "file");
        let err = PySnapshot::table_root_url("file:///tmp/a#b").unwrap_err();
        assert!(err.to_string().contains("fragment"), "{err}");
        assert!(PySnapshot::table_root_url("s3://bucket/t?x=1").is_err());
        assert!(PySnapshot::table_root_url("").is_err());
        let url = PySnapshot::table_root_url("s3://bucket/t").unwrap();
        assert_eq!(url.as_str(), "s3://bucket/t/");
        // A path containing '#' is fine: only a URL reads it as a fragment.
        let url = PySnapshot::table_root_url("/tmp/a#b").unwrap();
        assert_eq!(url.path(), "/tmp/a%23b/");
        if let Some(home) = std::env::var_os("HOME") {
            let url = PySnapshot::table_root_url("~/t").unwrap();
            let want = url::Url::from_directory_path(std::path::Path::new(&home).join("t"));
            assert_eq!(Some(url), want.ok());
        }
    }
}
