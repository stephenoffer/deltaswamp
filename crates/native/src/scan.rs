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

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatchReader;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::EngineDataArrowExt;
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
}

impl KernelBatchReader {
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
        let iter = RestrictedScan {
            metadata: Box::new(scan.scan_metadata(engine.as_ref())?),
            engine,
            table_root: scan.snapshot().table_root().clone(),
            physical_schema: scan.physical_schema().clone(),
            logical_schema: scan.logical_schema().clone(),
            paths,
            pending: VecDeque::new(),
            current: None,
            finished: false,
        };
        Self::from_parts(scan.logical_schema().as_ref(), iter)
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
        })
    }
}

impl Iterator for KernelBatchReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Deliberately a 1:1 pass-through. Do not batch, coalesce, reorder or
        // filter here; see the module docs.
        self.iter.next().map(|data| {
            data.try_into_record_batch()
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))
        })
    }
}

impl RecordBatchReader for KernelBatchReader {
    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }
}

type DataIter = Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>;

/// The file currently being read: its Parquet batches, the part of its DV
/// keep-mask not yet consumed, and its physical->logical transform.
struct OpenFile {
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
    paths: HashSet<String>,
    pending: VecDeque<ScanFile>,
    current: Option<OpenFile>,
    finished: bool,
}

impl RestrictedScan {
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
        self.pending.extend(
            files
                .into_iter()
                .filter(|file| self.paths.contains(&file.path)),
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
                        return Some(Self::finish_batch(
                            self.engine.as_ref(),
                            &self.physical_schema,
                            &self.logical_schema,
                            open,
                            physical,
                        ))
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
