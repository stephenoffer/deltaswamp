//! Turning a kernel scan into an Arrow stream.
//!
//! Correctness rule, enforced by construction: **rows are never reordered.**
//! A deletion vector is a positional keep-mask over a file's rows in physical
//! order, so any repartitioning, limit pushdown or pre-filtering applied before
//! the mask lands silently returns the right *number* of rows made of the wrong
//! records. `Scan::execute` applies DVs and the physical->logical transform
//! itself; this module's only job is to hand the resulting batches onward in
//! the order kernel produced them.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatchReader;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::EngineDataArrowExt;
use delta_kernel::scan::Scan;
use delta_kernel::{DeltaResult, Engine, EngineData};

use crate::error::Result;

/// Streams a kernel scan as Arrow record batches, preserving order.
pub struct KernelBatchReader {
    schema: ArrowSchemaRef,
    iter: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>,
}

impl KernelBatchReader {
    pub fn try_new(scan: &Scan, engine: Arc<dyn Engine>) -> Result<Self> {
        let schema: ArrowSchema = scan.logical_schema().as_ref().try_into_arrow()?;
        let iter = scan.execute(engine)?;
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
