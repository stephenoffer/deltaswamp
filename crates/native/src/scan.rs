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

use std::collections::{HashSet, VecDeque};
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
    /// still applies on top of the restriction.
    pub fn try_new_restricted(
        scan: &Scan,
        engine: Arc<dyn Engine>,
        paths: HashSet<String>,
    ) -> Result<Self> {
        let iter = RestrictedScan::new(scan, engine, Some(paths), false)?;
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
        paths: Option<HashSet<String>>,
    ) -> Result<Self> {
        let iter = RestrictedScan::new(scan, engine, paths, true)?;
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
    /// Append [`FILE_PATH_COLUMN`] to every batch.
    tag_path: bool,
    pending: VecDeque<ScanFile>,
    current: Option<OpenFile>,
    finished: bool,
}

impl RestrictedScan {
    fn new(
        scan: &Scan,
        engine: Arc<dyn Engine>,
        paths: Option<HashSet<String>>,
        tag_path: bool,
    ) -> Result<Self> {
        Ok(Self {
            metadata: Box::new(scan.scan_metadata(engine.as_ref())?),
            engine,
            table_root: scan.snapshot().table_root().clone(),
            physical_schema: scan.physical_schema().clone(),
            logical_schema: scan.logical_schema().clone(),
            paths,
            tag_path,
            pending: VecDeque::new(),
            current: None,
            finished: false,
        })
    }

    /// Queue the wanted files of the next scan-metadata batch. Returns false
    /// when the log replay is exhausted.
    fn plan_next(&mut self) -> DeltaResult<bool> {
        fn collect(files: &mut Vec<ScanFile>, file: ScanFile) {
            files.push(file);
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
        let location = self.table_root.join(&file.path)?;
        let selection = file
            .dv_info
            .get_selection_vector(self.engine.as_ref(), &self.table_root)?;
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
        let mut batches = self
            .engine
            .parquet_handler()
            .read_parquet_files(&[meta], self.physical_schema.clone(), None)?
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

    fn next_batch(&mut self) -> Option<DeltaResult<Box<dyn EngineData>>> {
        loop {
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
                            batch.and_then(|b| Self::tag_with_path(b, &open.path))
                        } else {
                            batch
                        });
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => self.current = None,
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
    fn tag_with_path(data: Box<dyn EngineData>, path: &str) -> DeltaResult<Box<dyn EngineData>> {
        let batch = data.try_into_record_batch()?;
        let paths = arrow::array::StringArray::from(vec![path; batch.num_rows()]);
        let mut fields: Vec<arrow::datatypes::FieldRef> =
            batch.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(arrow::datatypes::Field::new(
            FILE_PATH_COLUMN,
            arrow::datatypes::DataType::Utf8,
            false,
        )));
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(paths));
        let schema = Arc::new(ArrowSchema::new(fields));
        let tagged = RecordBatch::try_new(schema, columns)?;
        Ok(Box::new(ArrowEngineData::new(tagged)))
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
