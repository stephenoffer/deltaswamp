//! Committing to Delta tables, including catalog-managed ones.
//!
//! For a path-based table the commit is an atomic put of `<version>.json` and
//! `FileSystemCommitter` handles it. For a `catalogManaged` table the catalog is
//! the arbiter: the writer stages a commit at
//! `_delta_log/_staged_commits/<version>.<uuid>.json`, then asks Unity Catalog
//! to ratify it. Multiple writers may stage different UUIDs for the same
//! version and the catalog picks the winner, so conflict detection is
//! server-side and positional.
//!
//! Two response codes mean very different things and kernel currently collapses
//! both into a generic error, so we recover the distinction here:
//!
//! * **409** -- you lost the race. Re-read the snapshot, recompute, and stage a
//!   *new* UUID at the next version. Never reuse a staged file: it encodes
//!   version-dependent state and a `txnId` that must be unique per commit.
//! * **429** -- "maximum unbackfilled commits reached". This is backpressure,
//!   not rate limiting. You owe a *publish*, and retrying with backoff instead
//!   of publishing will wedge the table.

use std::sync::Arc;

use delta_kernel::committer::{Committer, FileSystemCommitter};
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::transaction::{CommitResult, Transaction};
use delta_kernel_default_engine::executor::tokio::TokioBackgroundExecutor;
use delta_kernel_default_engine::DefaultEngine;
use pyo3::prelude::*;
use unity_catalog_delta_rest_client::TableIdentifier;

use unity_catalog_delta_rest_client::{ClientConfig, UCUpdateTableRestClient};

use crate::error::{NativeError, Result};
use crate::runtime;

pub type SharedEngine = Arc<DefaultEngine<TokioBackgroundExecutor>>;

/// How to reach Unity Catalog in order to have a commit ratified.
// `from_py_object` is explicit because this class is passed *into* Rust as an
// argument; pyo3 0.29 deprecated deriving that implicitly from Clone.
#[pyclass(
    module = "deltaswamp._native",
    name = "UcCommitConfig",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct UcCommitConfig {
    workspace_url: String,
    token: String,
    table_id: String,
    catalog: String,
    schema: String,
    table: String,
}

#[pymethods]
impl UcCommitConfig {
    #[new]
    #[pyo3(signature = (workspace_url, token, table_id, catalog, schema, table))]
    fn new(
        workspace_url: String,
        token: String,
        table_id: String,
        catalog: String,
        schema: String,
        table: String,
    ) -> Self {
        Self {
            workspace_url,
            token,
            table_id,
            catalog,
            schema,
            table,
        }
    }

    fn __repr__(&self) -> String {
        // Never render the token.
        format!(
            "UcCommitConfig(workspace_url={:?}, table={}.{}.{}, table_id={:?})",
            self.workspace_url, self.catalog, self.schema, self.table, self.table_id
        )
    }
}

impl UcCommitConfig {
    fn committer(&self) -> Result<Box<dyn Committer>> {
        let config = ClientConfig::build(self.workspace_url.clone(), self.token.clone())
            .build()
            .map_err(|e| NativeError::Invalid(format!("invalid UC client config: {e}")))?;
        let client = UCUpdateTableRestClient::new(config)
            .map_err(|e| NativeError::Invalid(format!("could not build UC client: {e}")))?;
        let identifier = TableIdentifier {
            catalog: self.catalog.clone(),
            schema: self.schema.clone(),
            table: self.table.clone(),
        };
        Ok(Box::new(delta_kernel_unity_catalog::UCCommitter::new(
            Arc::new(client),
            self.table_id.clone(),
            identifier,
        )))
    }
}

/// Classify a commit failure so callers can react correctly.
///
/// Kernel surfaces UC errors as `Error::Generic`, so the status code has to be
/// recovered from the message. Crude, but the alternative is treating a
/// backfill demand as a transient error and wedging the table.
pub fn classify_commit_error(message: &str) -> NativeError {
    let lowered = message.to_lowercase();
    if lowered.contains("429") || lowered.contains("unbackfilled") {
        return NativeError::BackfillRequired(message.to_string());
    }
    if lowered.contains("409") || lowered.contains("conflict") {
        return NativeError::CommitConflict(message.to_string());
    }
    NativeError::Invalid(message.to_string())
}

/// Append or overwrite Arrow batches, committing them as one transaction.
///
/// With `overwrite`, every file visible in the snapshot is removed in the same
/// commit that adds the new ones, which is what makes a full replace atomic.
/// This is the only route to overwriting a catalog-managed table, since
/// delta-rs cannot open one at all.
///
/// Returns the committed version.
#[allow(clippy::too_many_arguments)]
pub fn write(
    snapshot: SnapshotRef,
    engine: SharedEngine,
    batches: Vec<arrow::array::RecordBatch>,
    uc: Option<UcCommitConfig>,
    engine_info: Option<String>,
    operation: Option<String>,
    overwrite: bool,
    txn: Option<(String, i64)>,
    commit_metadata: Option<std::collections::HashMap<String, String>>,
) -> Result<u64> {
    let committer: Box<dyn Committer> = match &uc {
        Some(config) => config.committer()?,
        None => Box::new(FileSystemCommitter::new()),
    };

    // Clone before the transaction consumes it; the overwrite path needs to
    // scan the same snapshot to learn which files to remove.
    let scan_source = snapshot.clone();
    let mut transaction = snapshot.transaction(committer, engine.as_ref())?;
    if let Some(info) = engine_info {
        transaction = transaction.with_engine_info(info);
    }
    if let Some(op) = operation {
        transaction = transaction.with_operation(op);
    }
    if let Some((app_id, version)) = txn {
        transaction = transaction.with_transaction_id(app_id, version);
    }
    if let Some(metadata) = commit_metadata {
        transaction = apply_commit_metadata(transaction, metadata)?;
    }

    if overwrite {
        // Remove everything the snapshot can see, in this same commit.
        let scan = scan_source.scan_builder().build()?;
        let scan_metadata = runtime::block_on(async { scan.scan_metadata(engine.as_ref()) })?;
        for filtered in Transaction::scan_metadata_to_engine_data(scan_metadata) {
            transaction.remove_files(filtered?);
        }
    }

    let mut txn = transaction;
    let write_state = txn.write_state()?;
    let write_context = write_state.unpartitioned_write_context()?;

    for batch in batches {
        let data = ArrowEngineData::new(batch);
        let metadata =
            runtime::block_on(async { engine.write_parquet(&data, &write_context).await })?;
        txn.add_files(metadata);
    }

    match txn.commit(engine.as_ref()) {
        Ok(CommitResult::CommittedTransaction(committed)) => Ok(committed.commit_version()),
        Ok(CommitResult::ConflictedTransaction(conflicted)) => {
            let version = conflicted.conflict_version();
            Err(NativeError::CommitConflict(format!(
                "another writer committed version {version} first. Re-read the snapshot, \
                 recompute the write, and stage a new commit -- do not reuse the staged file, \
                 because it encodes a version-specific txnId."
            )))
        }
        Ok(CommitResult::RetryableTransaction(_)) => Err(NativeError::Retryable(
            "the commit failed with a retryable I/O error; the table state is unchanged, \
             so the same transaction may be retried"
                .to_string(),
        )),
        Err(err) => Err(classify_commit_error(&err.to_string())),
    }
}

/// Attach arbitrary commit metadata, the way `userMetadata` works elsewhere.
fn apply_commit_metadata(
    transaction: Transaction,
    metadata: std::collections::HashMap<String, String>,
) -> Result<Transaction> {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use delta_kernel::engine::arrow_conversion::TryFromArrow;
    use delta_kernel::engine::arrow_data::ArrowEngineData;
    use delta_kernel::schema::Schema;

    let mut keys: Vec<&str> = metadata.keys().map(String::as_str).collect();
    keys.sort_unstable();

    let fields: Vec<Field> = keys
        .iter()
        .map(|k| Field::new(*k, DataType::Utf8, true))
        .collect();
    let columns: Vec<arrow::array::ArrayRef> = keys
        .iter()
        .map(|k| Arc::new(StringArray::from(vec![metadata[*k].as_str()])) as arrow::array::ArrayRef)
        .collect();

    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    let batch = arrow::array::RecordBatch::try_new(arrow_schema.clone(), columns)?;
    let kernel_schema = Arc::new(Schema::try_from_arrow(arrow_schema.as_ref())?);

    Ok(transaction.with_commit_info(Box::new(ArrowEngineData::new(batch)), kernel_schema))
}

/// Create a table, committing version 0.
///
/// The kernel accepts nearly the whole Delta property surface here, including
/// `delta.feature.*` signals, row tracking, in-commit timestamps and custom
/// non-`delta.` keys, all of which delta-rs rejects outright. Clustering is the
/// one thing that is *not* a property: the kernel takes clustering columns
/// through its data layout and deliberately refuses
/// `delta.feature.clustering = supported`.
#[allow(clippy::too_many_arguments)]
pub fn create_table(
    table_root: &str,
    schema: arrow::datatypes::SchemaRef,
    engine: SharedEngine,
    properties: Option<std::collections::HashMap<String, String>>,
    partition_by: Option<Vec<String>>,
    cluster_by: Option<Vec<String>>,
    uc: Option<UcCommitConfig>,
    engine_info: Option<String>,
) -> Result<u64> {
    use delta_kernel::engine::arrow_conversion::TryFromArrow;
    use delta_kernel::schema::Schema;
    use delta_kernel::transaction::create_table::create_table as kernel_create_table;
    use delta_kernel::transaction::data_layout::DataLayout;

    if partition_by.as_ref().is_some_and(|p| !p.is_empty())
        && cluster_by.as_ref().is_some_and(|c| !c.is_empty())
    {
        return Err(NativeError::Invalid(
            "a table is either partitioned or clustered, not both".to_string(),
        ));
    }

    let kernel_schema = Schema::try_from_arrow(schema.as_ref())?;
    let info = engine_info.unwrap_or_else(|| "deltaswamp".to_string());
    let mut builder = kernel_create_table(table_root, Arc::new(kernel_schema), info);

    if let Some(properties) = properties {
        builder = builder.with_table_properties(properties);
    }
    if let Some(columns) = partition_by.filter(|c| !c.is_empty()) {
        builder = builder.with_data_layout(DataLayout::partitioned(columns));
    } else if let Some(columns) = cluster_by.filter(|c| !c.is_empty()) {
        builder = builder.with_data_layout(DataLayout::clustered(columns));
    }

    let committer: Box<dyn Committer> = match &uc {
        Some(config) => config.committer()?,
        None => Box::new(FileSystemCommitter::new()),
    };

    let txn = builder
        .build(engine.as_ref(), committer)
        .map_err(|e| classify_commit_error(&e.to_string()))?;

    match txn.commit(engine.as_ref()) {
        Ok(CommitResult::CommittedTransaction(committed)) => Ok(committed.commit_version()),
        Ok(CommitResult::ConflictedTransaction(_)) => Err(NativeError::CommitConflict(
            "another writer created this table first".to_string(),
        )),
        Ok(CommitResult::RetryableTransaction(_)) => Err(NativeError::Retryable(
            "the create failed with a retryable I/O error".to_string(),
        )),
        Err(err) => Err(classify_commit_error(&err.to_string())),
    }
}

/// Publish ratified-but-unpublished commits into `_delta_log/`.
///
/// Catalog-managed tables accumulate staged commits until someone publishes
/// them. This is not optional housekeeping: the catalog caps how many
/// unbackfilled commits it will hold and starts refusing writes past the limit,
/// and checkpoints only run on published versions.
pub fn publish(
    snapshot: SnapshotRef,
    engine: SharedEngine,
    uc: Option<UcCommitConfig>,
) -> Result<u64> {
    let committer: Box<dyn Committer> = match &uc {
        Some(config) => config.committer()?,
        None => Box::new(FileSystemCommitter::new()),
    };
    let published =
        runtime::block_on(async { snapshot.publish(engine.as_ref(), committer.as_ref()) })
            .map_err(|e| classify_commit_error(&e.to_string()))?;
    Ok(published.version())
}
