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
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::transaction::{CommitResult, Transaction};
use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;
use delta_kernel_default_engine::DefaultEngine;
use pyo3::prelude::*;
use unity_catalog_delta_rest_client::TableIdentifier;

use unity_catalog_delta_rest_client::{ClientConfig, UCUpdateTableRestClient};

use crate::error::{NativeError, Result};
use crate::partition;
use crate::runtime;

/// The engine every binding uses.
///
/// Built on [`TokioMultiThreadExecutor`] over our shared runtime, NOT the
/// default `TokioBackgroundExecutor`. The background executor runs every
/// kernel I/O future on one current-thread runtime in one thread. Checkpointing
/// hands `write_parquet_file` an iterator that itself does log I/O, and pulls
/// it from *inside* the future running on that thread: the nested
/// `block_on` enqueues a second future on the same single thread and then
/// blocks it waiting for the result, so it deadlocks every time, on or off our
/// runtime. The multi-thread executor bridges a nested `block_on` with
/// `block_in_place` and another worker picks the inner future up. The kernel
/// documents this requirement on `Snapshot::checkpoint`.
pub type SharedEngine = Arc<DefaultEngine<TokioMultiThreadExecutor>>;

/// Build a [`SharedEngine`] over `store`, sharing the process runtime.
pub fn new_engine(store: Arc<DynObjectStore>) -> SharedEngine {
    let executor = Arc::new(TokioMultiThreadExecutor::new(
        runtime::runtime().handle().clone(),
    ));
    Arc::new(
        DefaultEngine::builder(store)
            .with_task_executor(executor)
            .build(),
    )
}

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
    // Clone before the transaction consumes it; the overwrite path needs to
    // scan the same snapshot to learn which files to remove.
    let scan_source = snapshot.clone();
    let partition_columns = snapshot
        .table_configuration()
        .logical_partition_columns()
        .to_vec();
    let table_schema = snapshot.schema();
    let mut transaction = begin_transaction(
        snapshot,
        &engine,
        &uc,
        engine_info,
        operation,
        txn,
        commit_metadata,
    )?;

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

    if partition_columns.is_empty() {
        let write_context = write_state.unpartitioned_write_context()?;
        for batch in batches {
            let data = ArrowEngineData::new(batch);
            let metadata =
                runtime::block_on(async { engine.write_parquet(&data, &write_context).await })?;
            txn.add_files(metadata);
        }
    } else {
        // One write context per distinct partition tuple; see crate::partition.
        for batch in batches {
            for group in
                partition::split_by_partition(&batch, &partition_columns, table_schema.as_ref())?
            {
                let write_context = write_state.partitioned_write_context(group.values)?;
                let data = ArrowEngineData::new(group.data);
                let metadata =
                    runtime::block_on(async { engine.write_parquet(&data, &write_context).await })?;
                txn.add_files(metadata);
            }
        }
    }

    // UCCommitter looks up the current Tokio handle and bridges its HTTP calls
    // with block_in_place, so the commit must run inside the shared
    // multi-threaded runtime rather than on a bare Python thread.
    match runtime::block_on(async { txn.commit(engine.as_ref()) }) {
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

    // UCCommitter looks up the current Tokio handle and bridges its HTTP calls
    // with block_in_place, so the commit must run inside the shared
    // multi-threaded runtime rather than on a bare Python thread.
    match runtime::block_on(async { txn.commit(engine.as_ref()) }) {
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

/// Atomically write `actions` as commit `version`, verbatim.
///
/// For metadata-only commits authored in Python (property changes, protocol
/// upgrades, domain metadata) that the kernel transaction API cannot express.
/// The commit is a put-if-absent of `_delta_log/<version>.json`: if the file
/// already exists another writer won, and the caller must re-read and
/// recompute -- exactly the contract `FileSystemCommitter` gives.
///
/// Path-based tables only. A catalog-managed table's commits must be ratified
/// by the catalog; writing one here would fork the table's history.
pub fn commit_raw(
    table_root: &url::Url,
    options: &std::collections::HashMap<String, String>,
    version: u64,
    actions: &[String],
) -> Result<u64> {
    use delta_kernel::object_store::path::Path;
    use delta_kernel::object_store::{PutMode, PutOptions, PutPayload};

    let body = raw_commit_body(actions)?;
    let store = crate::store::build_store(table_root, options)?;
    let root =
        Path::from_url_path(table_root.path()).map_err(delta_kernel::object_store::Error::from)?;
    let location = root.join("_delta_log").join(format!("{version:020}.json"));

    let result = runtime::block_on(async {
        store
            .put_opts(
                &location,
                PutPayload::from(body.into_bytes()),
                PutOptions::from(PutMode::Create),
            )
            .await
    });
    match result {
        Ok(_) => Ok(version),
        Err(delta_kernel::object_store::Error::AlreadyExists { .. }) => {
            Err(NativeError::CommitConflict(format!(
                "version {version} already exists at {table_root}: another writer committed \
                 it first. Re-read the snapshot, recompute the actions against the new \
                 state, and commit at the next version."
            )))
        }
        Err(delta_kernel::object_store::Error::NotImplemented { .. }) => {
            Err(NativeError::Invalid(format!(
                "the object store for {table_root} cannot do an atomic put-if-absent, so a \
                 raw commit could silently overwrite another writer's. On S3 do not set \
                 aws_conditional_put=disabled (the default, etag, sends If-None-Match)."
            )))
        }
        Err(e) => Err(e.into()),
    }
}

/// Validate and join raw actions into a newline-delimited commit body.
///
/// Each action must be one JSON object with exactly one key (the action name),
/// on one line: a newline inside an action would split it into two log lines
/// and corrupt the commit for every reader.
fn raw_commit_body(actions: &[String]) -> Result<String> {
    if actions.is_empty() {
        return Err(NativeError::Invalid(
            "a commit needs at least one action".to_string(),
        ));
    }
    let mut body = String::new();
    for (i, action) in actions.iter().enumerate() {
        let action = action.trim();
        if action.contains(['\n', '\r']) {
            return Err(NativeError::Invalid(format!(
                "action {i} spans several lines; each action must be single-line JSON"
            )));
        }
        let parsed: serde_json::Value = serde_json::from_str(action)
            .map_err(|e| NativeError::Invalid(format!("action {i} is not valid JSON: {e}")))?;
        match parsed.as_object() {
            Some(obj) if obj.len() == 1 => {}
            _ => {
                return Err(NativeError::Invalid(format!(
                    "action {i} must be a JSON object with exactly one key naming the action, \
                     e.g. {{\"commitInfo\": {{...}}}}"
                )))
            }
        }
        body.push_str(action);
        body.push('\n');
    }
    Ok(body)
}

// ---------------------------------------------------------------- distributed

/// Build a transaction with the commit-level metadata applied.
///
/// Shared by the single-shot path and the distributed one so the two cannot
/// drift on which committer a catalog-managed table gets.
#[allow(clippy::too_many_arguments)]
fn begin_transaction(
    snapshot: SnapshotRef,
    engine: &SharedEngine,
    uc: &Option<UcCommitConfig>,
    engine_info: Option<String>,
    operation: Option<String>,
    txn: Option<(String, i64)>,
    commit_metadata: Option<std::collections::HashMap<String, String>>,
) -> Result<Transaction> {
    let committer: Box<dyn Committer> = match uc {
        Some(config) => config.committer()?,
        None => Box::new(FileSystemCommitter::new()),
    };
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
    Ok(transaction)
}

/// The add-action metadata the engine produced, as an Arrow `RecordBatch`.
fn add_metadata_batch(
    data: Box<dyn delta_kernel::EngineData>,
) -> Result<arrow::array::RecordBatch> {
    let arrow = data.into_any().downcast::<ArrowEngineData>().map_err(|_| {
        NativeError::Invalid(
            "the engine returned add-file metadata that is not Arrow-backed".to_string(),
        )
    })?;
    Ok((*arrow).into())
}

/// Serialise add-action metadata as Arrow IPC.
///
/// IPC rather than JSON because the schema kernel expects from `add_files` is
/// nested, partly table-dependent (the stats struct follows the data schema)
/// and gains columns under row tracking. Round-tripping the Arrow data keeps
/// whatever kernel produced byte-exact, instead of re-deriving a schema here
/// that would silently drift from `Transaction::add_files_schema`.
/// Schema-metadata keys identifying the table a fragment was written for.
///
/// A fragment names data files by a path relative to its table root, so
/// committing one into a different table writes an add action pointing at a
/// file that is not there: the commit succeeds and the table is unreadable from
/// then on. Nothing in the add action itself records which table it came from,
/// so the binding is carried here and checked before the commit.
const FRAGMENT_TABLE_ROOT: &str = "deltaswamp.table_root";
const FRAGMENT_METADATA_ID: &str = "deltaswamp.metadata_id";

fn batches_to_ipc(
    batches: &[arrow::array::RecordBatch],
    table_root: &str,
    metadata_id: &str,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    if let Some(first) = batches.first() {
        let mut metadata = first.schema().metadata().clone();
        metadata.insert(FRAGMENT_TABLE_ROOT.to_string(), table_root.to_string());
        metadata.insert(FRAGMENT_METADATA_ID.to_string(), metadata_id.to_string());
        let schema = Arc::new(first.schema().as_ref().clone().with_metadata(metadata));

        let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut buffer, schema.as_ref())?;
        for batch in batches {
            let stamped =
                arrow::array::RecordBatch::try_new(schema.clone(), batch.columns().to_vec())?;
            writer.write(&stamped)?;
        }
        writer.finish()?;
    }
    Ok(buffer)
}

/// Decode a fragment, refusing one written for a different table.
fn ipc_to_batches(
    bytes: &[u8],
    table_root: &str,
    metadata_id: &str,
) -> Result<Vec<arrow::array::RecordBatch>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let reader = arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)?;
    let schema = reader.schema();
    let metadata = schema.metadata();

    let fragment_root = metadata.get(FRAGMENT_TABLE_ROOT).map(String::as_str);
    let fragment_id = metadata.get(FRAGMENT_METADATA_ID).map(String::as_str);
    match (fragment_root, fragment_id) {
        (Some(root), Some(id)) => {
            if root != table_root || id != metadata_id {
                return Err(NativeError::Invalid(format!(
                    "this fragment was written for a different table ({root}, metadata id \
                     {id}) and is being committed to {table_root} (metadata id \
                     {metadata_id}). Its files are named relative to the table they were \
                     written under, so committing it here would add files that are not \
                     there and leave this table unreadable."
                )));
            }
        }
        _ => {
            return Err(NativeError::Invalid(
                "this fragment carries no table identity, so it cannot be checked against \
                 the table being committed to. It was not produced by write_files."
                    .to_string(),
            ));
        }
    }

    let mut out = Vec::new();
    for batch in reader {
        out.push(batch?);
    }
    Ok(out)
}

/// Write Arrow batches as Parquet data files **without committing**.
///
/// This is the worker half of a distributed write: it produces data files and
/// returns the add-action metadata describing them, which travels back to
/// whoever is coordinating and is committed there by [`commit_files`]. The
/// transaction built here exists only to obtain a write context and is dropped
/// unread, so nothing is added to the log by this call.
///
/// The files are real and durable the moment this returns. A coordinator that
/// never commits leaves them behind as garbage -- which is exactly what happens
/// today when a connector writes for an hour and only then discovers the commit
/// will be refused, so callers should settle whether the commit can succeed
/// before the first worker runs.
pub fn write_files(
    snapshot: SnapshotRef,
    engine: SharedEngine,
    batches: Vec<arrow::array::RecordBatch>,
    uc: Option<UcCommitConfig>,
) -> Result<Vec<u8>> {
    let partition_columns = snapshot
        .table_configuration()
        .logical_partition_columns()
        .to_vec();
    let table_schema = snapshot.schema();
    let table_root = snapshot.table_root().to_string();
    let metadata_id = snapshot.table_configuration().metadata().id().to_string();
    // Built with the same committer the coordinator will use: a catalog-managed
    // table validates differently, and a worker must not discover that late.
    let txn = begin_transaction(snapshot, &engine, &uc, None, None, None, None)?;
    let write_state = txn.write_state()?;

    let mut metadata_batches = Vec::new();
    if partition_columns.is_empty() {
        let write_context = write_state.unpartitioned_write_context()?;
        for batch in batches {
            let data = ArrowEngineData::new(batch);
            let metadata =
                runtime::block_on(async { engine.write_parquet(&data, &write_context).await })?;
            metadata_batches.push(add_metadata_batch(metadata)?);
        }
    } else {
        for batch in batches {
            for group in
                partition::split_by_partition(&batch, &partition_columns, table_schema.as_ref())?
            {
                let write_context = write_state.partitioned_write_context(group.values)?;
                let data = ArrowEngineData::new(group.data);
                let metadata =
                    runtime::block_on(async { engine.write_parquet(&data, &write_context).await })?;
                metadata_batches.push(add_metadata_batch(metadata)?);
            }
        }
    }

    batches_to_ipc(&metadata_batches, &table_root, &metadata_id)
}

/// Commit add-action metadata produced by [`write_files`], possibly elsewhere.
///
/// This is the coordinator half. Every fragment is added to one transaction, so
/// a distributed write lands as a single atomic commit at one version -- readers
/// never see half a job. With `overwrite`, the files visible in this snapshot
/// are removed in that same commit.
#[allow(clippy::too_many_arguments)]
pub fn commit_files(
    snapshot: SnapshotRef,
    engine: SharedEngine,
    fragments: Vec<Vec<u8>>,
    uc: Option<UcCommitConfig>,
    engine_info: Option<String>,
    operation: Option<String>,
    overwrite: bool,
    txn: Option<(String, i64)>,
    commit_metadata: Option<std::collections::HashMap<String, String>>,
) -> Result<u64> {
    let scan_source = snapshot.clone();
    let table_root = snapshot.table_root().to_string();
    let metadata_id = snapshot.table_configuration().metadata().id().to_string();
    let mut transaction = begin_transaction(
        snapshot,
        &engine,
        &uc,
        engine_info,
        operation,
        txn,
        commit_metadata,
    )?;

    if overwrite {
        let scan = scan_source.scan_builder().build()?;
        let scan_metadata = runtime::block_on(async { scan.scan_metadata(engine.as_ref()) })?;
        for filtered in Transaction::scan_metadata_to_engine_data(scan_metadata) {
            transaction.remove_files(filtered?);
        }
    }

    for fragment in &fragments {
        for batch in ipc_to_batches(fragment, &table_root, &metadata_id)? {
            transaction.add_files(Box::new(ArrowEngineData::new(batch)));
        }
    }

    finish_commit(transaction, &engine)
}

/// Run a prepared transaction's commit and classify the outcome.
fn finish_commit(txn: Transaction, engine: &SharedEngine) -> Result<u64> {
    match runtime::block_on(async { txn.commit(engine.as_ref()) }) {
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

#[cfg(test)]
mod raw_commit_tests {
    use super::*;

    #[test]
    fn body_is_newline_delimited_and_terminated() {
        let body = raw_commit_body(&[
            r#"{"commitInfo":{"a":1}}"#.to_string(),
            r#" {"protocol":{"minReaderVersion":1,"minWriterVersion":2}} "#.to_string(),
        ])
        .unwrap();
        assert_eq!(
            body,
            "{\"commitInfo\":{\"a\":1}}\n{\"protocol\":{\"minReaderVersion\":1,\"minWriterVersion\":2}}\n"
        );
    }

    #[test]
    fn malformed_actions_are_refused() {
        assert!(raw_commit_body(&[]).is_err());
        assert!(raw_commit_body(&["not json".to_string()]).is_err());
        assert!(raw_commit_body(&[r#"{"a":1,"b":2}"#.to_string()]).is_err());
        assert!(raw_commit_body(&["{\"a\":\n1}".to_string()]).is_err());
        assert!(raw_commit_body(&["[1]".to_string()]).is_err());
    }

    #[test]
    fn put_if_absent_conflicts_on_an_existing_version() {
        let dir = std::env::temp_dir().join(format!("ds-raw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let url = url::Url::from_directory_path(&dir).unwrap();
        let opts = std::collections::HashMap::new();
        let actions = [r#"{"commitInfo":{}}"#.to_string()];
        assert_eq!(commit_raw(&url, &opts, 3, &actions).unwrap(), 3);
        let written =
            std::fs::read_to_string(dir.join("_delta_log/00000000000000000003.json")).unwrap();
        assert_eq!(written, "{\"commitInfo\":{}}\n");
        let err = commit_raw(&url, &opts, 3, &actions).unwrap_err();
        assert!(matches!(err, NativeError::CommitConflict(_)), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
