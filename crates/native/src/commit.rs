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
use delta_kernel::{DeltaResult, FilteredEngineData};
use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;
use delta_kernel_default_engine::DefaultEngine;
use pyo3::prelude::*;
use unity_catalog_delta_rest_client::TableIdentifier;

use unity_catalog_delta_rest_client::{ClientConfig, UCUpdateTableRestClient};

use crate::error::{NativeError, Result};
use crate::partition;
use crate::runtime;
use delta_kernel::parquet::basic::Compression;

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
    // The UC client renders "HTTP error (status NNN): ...". When a status is
    // present it is authoritative: a 400 whose text mentions "conflicting
    // properties" is not a lost race, and retrying it would loop forever.
    if let Some(status) = http_status(&lowered) {
        return match status {
            429 => NativeError::BackfillRequired(message.to_string()),
            409 => NativeError::CommitConflict(message.to_string()),
            // The Python layer words these for the caller; the type is what
            // it needs, so the message stays the catalog's own.
            401 | 403 => NativeError::CatalogPermission(message.to_string()),
            404 => NativeError::CatalogNotFound(message.to_string()),
            // The version is positional: re-sending the same commit cannot land
            // twice, which is what makes these retryable.
            500..=599 => NativeError::Retryable(format!(
                "the catalog failed before confirming the commit, which may or may not have \
                 been ratified; re-read the table before committing again: {message}"
            )),
            _ => NativeError::CatalogRejected(message.to_string()),
        };
    }
    // The UC client renders some statuses without one: a 401 is its
    // `AuthenticationFailed` ("Authentication failed"), and its API errors
    // name a missing table or the unpublished-commit cap in words.
    if lowered.contains("uc update_table error") {
        if lowered.contains("authentication failed") {
            return NativeError::CatalogPermission(message.to_string());
        }
        if lowered.contains("table not found") {
            return NativeError::CatalogNotFound(message.to_string());
        }
        if lowered.contains("max unpublished commits") {
            return NativeError::BackfillRequired(message.to_string());
        }
    }
    if lowered.contains("timed out") || lowered.contains("timeout") {
        return NativeError::Retryable(format!(
            "the commit timed out before it was confirmed, and may or may not have landed; \
             re-read the table before committing again: {message}"
        ));
    }
    // Otherwise look for the codes as whole numbers only: a staged-commit
    // UUID or a version like `...00429.json` must not read as a status.
    if has_code(&lowered, "429") || lowered.contains("unbackfilled") {
        return NativeError::BackfillRequired(message.to_string());
    }
    if has_code(&lowered, "409") || lowered.contains("conflict") {
        return NativeError::CommitConflict(message.to_string());
    }
    NativeError::Invalid(message.to_string())
}

/// The status in an `(status NNN)` fragment, if the message carries one.
fn http_status(lowered: &str) -> Option<u16> {
    let rest = &lowered[lowered.find("status ")? + "status ".len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    (digits.len() == 3).then(|| digits.parse().ok()).flatten()
}

/// True if `code` occurs in `text` not adjacent to another alphanumeric.
fn has_code(text: &str, code: &str) -> bool {
    text.match_indices(code).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + code.len()..].chars().next();
        !before.is_some_and(|c| c.is_ascii_alphanumeric())
            && !after.is_some_and(|c| c.is_ascii_alphanumeric())
    })
}

/// Classify a kernel commit error, keeping I/O failures as I/O errors.
///
/// Flattening an object-store failure to its message made a network error or
/// a 403 during commit surface as `ValueError`, indistinguishable from bad
/// input.
pub fn classify_kernel_commit_error(err: delta_kernel::Error) -> NativeError {
    match err {
        // object_store 0.13 can report a lost put-if-absent on Azure as a 412
        // Precondition rather than AlreadyExists (object_store#829, fixed in
        // 0.14). The kernel only maps AlreadyExists to a conflict, so a lost
        // race surfaced as a raw I/O error and was never retried. Every put a
        // commit makes is put-if-absent, so a failed precondition here means
        // another writer took the version.
        delta_kernel::Error::ObjectStore(delta_kernel::object_store::Error::Precondition {
            ..
        }) => NativeError::CommitConflict(format!(
            "the version already exists: another writer committed it first (the store \
             refused the put-if-absent: {err}). Re-read the snapshot and retry."
        )),
        delta_kernel::Error::ObjectStore(_)
        | delta_kernel::Error::IOError(_)
        | delta_kernel::Error::FileNotFound(_) => NativeError::Kernel(err),
        other => classify_commit_error(&other.to_string()),
    }
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
    info: CommitInfoPatch,
) -> Result<u64> {
    // Clone before the transaction consumes it; the overwrite path needs to
    // scan the same snapshot to learn which files to remove.
    let scan_source = snapshot.clone();
    let codec = crate::writer::codec_for(&snapshot);
    let partition_columns = snapshot
        .table_configuration()
        .logical_partition_columns()
        .to_vec();
    let table_schema = snapshot.schema();
    let batches = batches
        .iter()
        .map(|b| partition::conform_to_table(b, table_schema.as_ref(), &partition_columns))
        .collect::<Result<Vec<_>>>()?;
    let batches = partition::coalesce(batches)?;
    let mut transaction = begin_transaction(
        snapshot,
        &engine,
        &uc,
        engine_info,
        operation,
        txn,
        commit_metadata,
        info,
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
    stage_batches(
        &mut txn,
        &engine,
        &partition_columns,
        &table_schema,
        batches,
        codec,
    )?;

    // UCCommitter looks up the current Tokio handle and bridges its HTTP calls
    // with block_in_place, so the commit must run inside the shared
    // multi-threaded runtime rather than on a bare Python thread.
    finish_commit(txn, &engine)
}

/// Align `batches` with the table schema and coalesce them, as every write does.
pub(crate) fn prepare_batches(
    snapshot: &SnapshotRef,
    batches: Vec<arrow::array::RecordBatch>,
) -> Result<Vec<arrow::array::RecordBatch>> {
    let partition_columns = snapshot
        .table_configuration()
        .logical_partition_columns()
        .to_vec();
    let table_schema = snapshot.schema();
    let batches = batches
        .iter()
        .map(|b| partition::conform_to_table(b, table_schema.as_ref(), &partition_columns))
        .collect::<Result<Vec<_>>>()?;
    partition::coalesce(batches)
}

/// Merge small batches of one schema, as every write does before writing.
pub(crate) fn coalesce(
    batches: Vec<arrow::array::RecordBatch>,
) -> Result<Vec<arrow::array::RecordBatch>> {
    partition::coalesce(batches)
}

/// Write prepared `batches` as Parquet and add them to `txn`.
///
/// A partitioned table gets one write context per distinct partition tuple;
/// see `crate::partition`.
pub(crate) fn stage_batches(
    txn: &mut Transaction,
    engine: &SharedEngine,
    partition_columns: &[String],
    table_schema: &delta_kernel::schema::SchemaRef,
    batches: Vec<arrow::array::RecordBatch>,
    codec: Compression,
) -> Result<()> {
    let write_state = txn.write_state()?;
    let mut staged = Vec::new();
    if partition_columns.is_empty() {
        let write_context = write_state.unpartitioned_write_context()?;
        for batch in batches {
            let data = ArrowEngineData::new(batch);
            let metadata = runtime::block_on(crate::writer::write_parquet(
                engine,
                &data,
                &write_context,
                codec,
            ))?;
            staged.push(metadata);
        }
    } else {
        for batch in batches {
            for group in
                partition::split_by_partition(&batch, partition_columns, table_schema.as_ref())?
            {
                let write_context = write_state.partitioned_write_context(group.values)?;
                let data = ArrowEngineData::new(group.data);
                let metadata = runtime::block_on(crate::writer::write_parquet(
                    engine,
                    &data,
                    &write_context,
                    codec,
                ))?;
                staged.push(metadata);
            }
        }
    }
    for metadata in staged {
        txn.add_files(metadata);
    }
    Ok(())
}

/// Output files written at once by [`stage_stream`] at most, and the Arrow
/// bytes they hold at most (one is always written, whatever its size).
///
/// One file at a time left an OPTIMIZE writing its 200 output files in turn;
/// the bytes bound keeps what waits to be written -- beside the batch the
/// stream is building -- to about one large file.
const WRITE_FILES: usize = 16;
const WRITE_BYTES: usize = 256 << 20;

/// [`stage_batches`] over a stream: each batch (a prepared group of them) is
/// written as it arrives, several at once, and added to `txn` in order.
pub(crate) fn stage_stream(
    txn: &mut Transaction,
    engine: &SharedEngine,
    partition_columns: &[String],
    table_schema: &delta_kernel::schema::SchemaRef,
    batches: impl Iterator<Item = Result<Vec<arrow::array::RecordBatch>>>,
    codec: Compression,
) -> Result<()> {
    type Write = tokio::task::JoinHandle<DeltaResult<Box<dyn delta_kernel::EngineData>>>;
    let write_state = txn.write_state()?;
    let mut inflight: std::collections::VecDeque<(Write, usize)> = Default::default();
    let mut inflight_bytes = 0usize;
    let finish = |txn: &mut Transaction, handle: Write| -> Result<()> {
        let metadata = runtime::block_on(handle)
            .map_err(|e| NativeError::Invalid(format!("writing a data file failed: {e}")))??;
        txn.add_files(metadata);
        Ok(())
    };
    for prepared in batches {
        for batch in prepared? {
            let groups = if partition_columns.is_empty() {
                vec![(None, batch)]
            } else {
                partition::split_by_partition(&batch, partition_columns, table_schema.as_ref())?
                    .into_iter()
                    .map(|group| (Some(group.values), group.data))
                    .collect()
            };
            for (values, data) in groups {
                let context = match values {
                    None => write_state.unpartitioned_write_context()?,
                    Some(values) => write_state.partitioned_write_context(values)?,
                };
                let bytes = data.get_array_memory_size();
                while !inflight.is_empty()
                    && (inflight.len() >= WRITE_FILES || inflight_bytes + bytes > WRITE_BYTES)
                {
                    let (handle, size) = inflight.pop_front().expect("not empty");
                    inflight_bytes -= size;
                    finish(txn, handle)?;
                }
                let engine = engine.clone();
                let handle = runtime::runtime().spawn(async move {
                    let data = ArrowEngineData::new(data);
                    // The writer that marks the footer with the Spark version,
                    // as every other write here uses: without it Databricks
                    // read a compacted file's early dates shifted.
                    crate::writer::write_parquet(&engine, &data, &context, codec).await
                });
                inflight.push_back((handle, bytes));
                inflight_bytes += bytes;
            }
        }
    }
    while let Some((handle, _)) = inflight.pop_front() {
        finish(txn, handle)?;
    }
    Ok(())
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
    // Kernel writes these itself and silently drops a caller's value for
    // them, so a user's "operation" or "timestamp" just vanished.
    const RESERVED: &[&str] = &[
        "timestamp",
        "inCommitTimestamp",
        "operation",
        "operationParameters",
        "operationMetrics",
        "kernelVersion",
        "isBlindAppend",
        "engineInfo",
        "txnId",
    ];
    if let Some(bad) = keys
        .iter()
        .find(|k| k.is_empty() || RESERVED.iter().any(|r| r.eq_ignore_ascii_case(k)))
    {
        return Err(NativeError::Invalid(format!(
            "commit_metadata key {bad:?} is empty or reserved by the Delta commitInfo action \
             (reserved: {RESERVED:?}); use a different key, e.g. userMetadata"
        )));
    }

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

    if schema.fields().is_empty() {
        // Kernel creates it, then refuses every scan of it.
        return Err(NativeError::Invalid(
            "a table needs at least one column".to_string(),
        ));
    }
    for columns in [&partition_by, &cluster_by].into_iter().flatten() {
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = columns.iter().find(|c| !seen.insert(c.to_lowercase())) {
            return Err(NativeError::Invalid(format!(
                "column {dup:?} is listed twice in the table layout"
            )));
        }
    }
    let widened = arrow::datatypes::Schema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(|f| {
                f.as_ref()
                    .clone()
                    .with_data_type(partition::widen_unsigned(f.data_type()))
            })
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    );
    let kernel_schema = Schema::try_from_arrow(&widened)?;
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
        .map_err(classify_kernel_commit_error)?;

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
        Err(err) => Err(classify_kernel_commit_error(err)),
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
            .map_err(classify_kernel_commit_error)?;
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
    use delta_kernel::object_store::{ObjectStoreExt, PutMode, PutOptions, PutPayload};

    let body = raw_commit_body(actions)?;
    let store = crate::store::build_store(table_root, options)?;
    let root =
        Path::from_url_path(table_root.path()).map_err(delta_kernel::object_store::Error::from)?;
    let location = root
        .clone()
        .join("_delta_log")
        .join(format!("{version:020}.json"));

    if version > 0 {
        // Put-if-absent only stops two writers taking the same version; it
        // does not stop a skipped one. `N.json` without `N-1.json` is a gap
        // that makes the log unreadable, so require the predecessor.
        let previous = root
            .join("_delta_log")
            .join(format!("{:020}.json", version - 1));
        match runtime::block_on(async { store.head(&previous).await }) {
            Ok(_) => {}
            Err(delta_kernel::object_store::Error::NotFound { .. }) => {
                return Err(NativeError::Invalid(format!(
                    "cannot commit version {version} at {table_root}: version {} does not \
                     exist, so the log would have a gap. Commit at the version after the \
                     snapshot's.",
                    version - 1
                )))
            }
            Err(e) => return Err(e.into()),
        }
    }

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
        Err(e) => Err(raw_put_error(e, version, table_root)),
    }
}

/// What a failed put-if-absent of commit `version` means.
fn raw_put_error(
    err: delta_kernel::object_store::Error,
    version: u64,
    table_root: &url::Url,
) -> NativeError {
    use delta_kernel::object_store::Error;
    match err {
        // Precondition: Azure's answer to a lost put-if-absent under
        // object_store 0.13 (object_store#829); see classify_kernel_commit_error.
        Error::AlreadyExists { .. } | Error::Precondition { .. } => {
            NativeError::CommitConflict(format!(
                "version {version} already exists at {table_root}: another writer committed \
                 it first. Re-read the snapshot, recompute the actions against the new \
                 state, and commit at the next version."
            ))
        }
        Error::NotImplemented { .. } => NativeError::Invalid(format!(
            "the object store for {table_root} cannot do an atomic put-if-absent, so a \
             raw commit could silently overwrite another writer's. On S3 do not set \
             aws_conditional_put=disabled (the default, etag, sends If-None-Match)."
        )),
        e => e.into(),
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

// ---------------------------------------------------------------- commitInfo

/// The `commitInfo` fields kernel 0.28 writes itself and gives no way to set.
///
/// It always writes `operationParameters` as an empty map and `isBlindAppend`
/// only when true, and overwrites whatever a caller's commit info says for
/// either. Both matter beyond display: a concurrent writer's conflict check
/// (Spark's, delta-rs's, this library's) treats a winner as a blind append --
/// one whose added rows nobody's read could have depended on -- only when
/// `isBlindAppend` says so, and a replaceWhere written with only adds looked
/// exactly like an append without it. Databricks' DESCRIBE HISTORY shows the
/// parameters (mode, predicate, zOrderBy).
#[derive(Debug, Clone, Default)]
pub struct CommitInfoPatch {
    /// Written as `operationParameters`, replacing kernel's empty map.
    pub operation_parameters: Option<std::collections::HashMap<String, String>>,
    /// Written as `isBlindAppend`, true or false (kernel writes only true).
    pub blind_append: Option<bool>,
}

impl CommitInfoPatch {
    fn is_empty(&self) -> bool {
        self.operation_parameters.is_none() && self.blind_append.is_none()
    }

    /// `data` with the patched fields, when it is the `commitInfo` action.
    fn apply(&self, data: FilteredEngineData) -> DeltaResult<FilteredEngineData> {
        use arrow::array::{Array, ArrayRef, BooleanArray, MapArray, StringArray, StructArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::{DataType, Field};

        let (data, selection) = data.into_parts();
        if data
            .as_ref()
            .any_ref()
            .downcast_ref::<ArrowEngineData>()
            .is_none()
        {
            // Not Arrow-backed, so not the commitInfo kernel builds: pass it on.
            return FilteredEngineData::try_new(data, selection);
        }
        let batch: arrow::array::RecordBatch = (*data
            .into_any()
            .downcast::<ArrowEngineData>()
            .map_err(|_| delta_kernel::Error::generic("action data is not Arrow"))?)
        .into();
        let Ok(index) = batch.schema().index_of("commitInfo") else {
            return FilteredEngineData::try_new(Box::new(ArrowEngineData::new(batch)), selection);
        };
        let column = batch.column(index);
        let Some(info) = column.as_any().downcast_ref::<StructArray>() else {
            return FilteredEngineData::try_new(Box::new(ArrowEngineData::new(batch)), selection);
        };
        let rows = info.len();
        let DataType::Struct(struct_fields) = info.data_type().clone() else {
            unreachable!("a StructArray has a struct type");
        };
        let mut fields: Vec<arrow::datatypes::FieldRef> = struct_fields.iter().cloned().collect();
        let mut columns: Vec<ArrayRef> = info.columns().to_vec();
        let mut put = |name: &str, field: Field, array: ArrayRef| match fields
            .iter()
            .position(|f| f.name() == name)
        {
            Some(i) => {
                fields[i] = Arc::new(field);
                columns[i] = array;
            }
            None => {
                fields.push(Arc::new(field));
                columns.push(array);
            }
        };
        if let Some(parameters) = &self.operation_parameters {
            let mut entries: Vec<(&String, &String)> = parameters.iter().collect();
            entries.sort();
            let keys: Vec<&str> = entries
                .iter()
                .cycle()
                .take(entries.len() * rows)
                .map(|(k, _)| k.as_str())
                .collect();
            let values: Vec<&str> = entries
                .iter()
                .cycle()
                .take(entries.len() * rows)
                .map(|(_, v)| v.as_str())
                .collect();
            let entry_fields = arrow::datatypes::Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Utf8, true),
            ]);
            let entry_struct = StructArray::try_new(
                entry_fields.clone(),
                vec![
                    Arc::new(StringArray::from(keys)) as ArrayRef,
                    Arc::new(StringArray::from(values)) as ArrayRef,
                ],
                None,
            )?;
            let entries_field = Arc::new(Field::new(
                "key_value",
                DataType::Struct(entry_fields),
                false,
            ));
            let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(entries.len(), rows));
            let map = MapArray::try_new(entries_field.clone(), offsets, entry_struct, None, false)?;
            put(
                "operationParameters",
                Field::new(
                    "operationParameters",
                    DataType::Map(entries_field, false),
                    true,
                ),
                Arc::new(map),
            );
        }
        if let Some(blind) = self.blind_append {
            put(
                "isBlindAppend",
                Field::new("isBlindAppend", DataType::Boolean, true),
                Arc::new(BooleanArray::from(vec![blind; rows])),
            );
        }
        let patched = StructArray::try_new(fields.into(), columns, info.nulls().cloned())?;
        let mut outer: Vec<arrow::datatypes::FieldRef> =
            batch.schema().fields().iter().cloned().collect();
        outer[index] = Arc::new(Field::new(
            "commitInfo",
            patched.data_type().clone(),
            outer[index].is_nullable(),
        ));
        let mut outer_columns = batch.columns().to_vec();
        outer_columns[index] = Arc::new(patched);
        let batch = arrow::array::RecordBatch::try_new(
            Arc::new(arrow::datatypes::Schema::new_with_metadata(
                outer,
                batch.schema().metadata().clone(),
            )),
            outer_columns,
        )?;
        FilteredEngineData::try_new(Box::new(ArrowEngineData::new(batch)), selection)
    }
}

/// A committer that writes [`CommitInfoPatch`] into the `commitInfo` action,
/// then hands every action to the real committer unchanged otherwise.
struct PatchingCommitter {
    inner: Box<dyn Committer>,
    info: CommitInfoPatch,
}

impl Committer for PatchingCommitter {
    fn commit(
        &self,
        engine: &dyn delta_kernel::Engine,
        actions: delta_kernel::DeltaResultIterator<'_, FilteredEngineData>,
        commit_metadata: delta_kernel::committer::CommitMetadata,
    ) -> DeltaResult<delta_kernel::committer::CommitResponse> {
        let info = &self.info;
        let patched = actions.map(move |item| item.and_then(|data| info.apply(data)));
        self.inner
            .commit(engine, Box::new(patched), commit_metadata)
    }

    fn is_catalog_committer(&self) -> bool {
        self.inner.is_catalog_committer()
    }

    fn publish(
        &self,
        engine: &dyn delta_kernel::Engine,
        publish_metadata: delta_kernel::committer::PublishMetadata,
    ) -> DeltaResult<()> {
        self.inner.publish(engine, publish_metadata)
    }
}

// ---------------------------------------------------------------- distributed

/// Build a transaction with the commit-level metadata applied.
///
/// Shared by the single-shot path and the distributed one so the two cannot
/// drift on which committer a catalog-managed table gets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn begin_transaction(
    snapshot: SnapshotRef,
    engine: &SharedEngine,
    uc: &Option<UcCommitConfig>,
    engine_info: Option<String>,
    operation: Option<String>,
    txn: Option<(String, i64)>,
    commit_metadata: Option<std::collections::HashMap<String, String>>,
    info: CommitInfoPatch,
) -> Result<Transaction> {
    let committer: Box<dyn Committer> = match uc {
        Some(config) => config.committer()?,
        None => Box::new(FileSystemCommitter::new()),
    };
    let committer: Box<dyn Committer> = if info.is_empty() {
        committer
    } else {
        Box::new(PatchingCommitter {
            inner: committer,
            info,
        })
    };
    let mut transaction = snapshot.transaction(committer, engine.as_ref())?;
    // Kernel leaves column defaults to the connector and refuses to write
    // until told they are handled. They are: every batch passes through
    // `partition::conform_to_table`, which refuses one that leaves out a
    // column with a default, and the Python side fills a literal default in.
    transaction.ack_column_defaults();
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
    // Junk bytes surfaced as "arrow error: Ipc error: Expected schema
    // message", which says nothing about where they came from.
    let not_a_fragment = |e: arrow::error::ArrowError| {
        NativeError::Invalid(format!(
            "this is not a fragment produced by write_files (it does not decode: {e}); \
             pass the bytes plan.write() returned, unchanged"
        ))
    };
    let reader = arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)
        .map_err(not_a_fragment)?;
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
        out.push(batch.map_err(not_a_fragment)?);
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
) -> std::result::Result<Vec<u8>, Box<WriteFilesError>> {
    let root = snapshot.table_root().clone();
    let mut written = Vec::new();
    match write_files_tracked(snapshot, &engine, batches, uc, &mut written) {
        Ok(bytes) => Ok(bytes),
        Err(error) => {
            // The files this attempt already wrote belong to no fragment, so
            // nobody would ever commit or find them: an orphan per failed
            // task. Remove them, best effort, and keep the original error.
            let not_removed = remove_written(&engine, &root, &written);
            Err(Box::new(WriteFilesError { error, not_removed }))
        }
    }
}

/// A failed [`write_files`]: the original error, and any data files it wrote
/// that could not be removed again (with why), for the caller to report.
#[derive(Debug)]
pub struct WriteFilesError {
    pub error: NativeError,
    pub not_removed: Vec<String>,
}

fn write_files_tracked(
    snapshot: SnapshotRef,
    engine: &SharedEngine,
    batches: Vec<arrow::array::RecordBatch>,
    uc: Option<UcCommitConfig>,
    written: &mut Vec<String>,
) -> Result<Vec<u8>> {
    let partition_columns = snapshot
        .table_configuration()
        .logical_partition_columns()
        .to_vec();
    let table_schema = snapshot.schema();
    let codec = crate::writer::codec_for(&snapshot);
    let table_root = snapshot.table_root().to_string();
    let metadata_id = snapshot.table_configuration().metadata().id().to_string();
    // Built with the same committer the coordinator will use: a catalog-managed
    // table validates differently, and a worker must not discover that late.
    // Same preprocessing as `write`: align columns by name, conform physical
    // types, validate partition values, and coalesce tiny batches.
    let batches = batches
        .iter()
        .map(|b| partition::conform_to_table(b, table_schema.as_ref(), &partition_columns))
        .collect::<Result<Vec<_>>>()?;
    let batches = partition::coalesce(batches)?;
    let txn = begin_transaction(
        snapshot,
        engine,
        &uc,
        None,
        None,
        None,
        None,
        CommitInfoPatch::default(),
    )?;
    let write_state = txn.write_state()?;

    let mut metadata_batches = Vec::new();
    let mut record = |batch: arrow::array::RecordBatch| -> Result<()> {
        written.extend(batch_paths(&batch)?);
        metadata_batches.push(batch);
        Ok(())
    };
    if partition_columns.is_empty() {
        let write_context = write_state.unpartitioned_write_context()?;
        for batch in batches {
            let data = ArrowEngineData::new(batch);
            let metadata = runtime::block_on(crate::writer::write_parquet(
                engine,
                &data,
                &write_context,
                codec,
            ))?;
            record(add_metadata_batch(metadata)?)?;
        }
    } else {
        for batch in batches {
            for group in
                partition::split_by_partition(&batch, &partition_columns, table_schema.as_ref())?
            {
                let write_context = write_state.partitioned_write_context(group.values)?;
                let data = ArrowEngineData::new(group.data);
                let metadata = runtime::block_on(crate::writer::write_parquet(
                    engine,
                    &data,
                    &write_context,
                    codec,
                ))?;
                record(add_metadata_batch(metadata)?)?;
            }
        }
    }

    batches_to_ipc(&metadata_batches, &table_root, &metadata_id)
}

/// The `path` column of an add-metadata batch.
fn batch_paths(batch: &arrow::array::RecordBatch) -> Result<Vec<String>> {
    use arrow::array::{Array, StringArray};

    let Some(column) = batch.column_by_name("path") else {
        return Ok(Vec::new());
    };
    let column = arrow::compute::cast(column, &arrow::datatypes::DataType::Utf8)?;
    let Some(paths) = column.as_any().downcast_ref::<StringArray>() else {
        return Ok(Vec::new());
    };
    Ok((0..paths.len())
        .filter(|i| !paths.is_null(*i))
        .map(|i| paths.value(i).to_string())
        .collect())
}

/// Delete data files named relative to `root`; returns the ones that failed.
pub(crate) fn remove_written(
    engine: &SharedEngine,
    root: &url::Url,
    written: &[String],
) -> Vec<String> {
    use delta_kernel::object_store::path::Path;
    use delta_kernel::object_store::ObjectStoreExt;

    if written.is_empty() {
        return Vec::new();
    }
    let Some(store) = engine.get_object_store_for_url(root) else {
        return written.to_vec();
    };
    let mut failed = Vec::new();
    for relative in written {
        let location = url::Url::parse(relative)
            .or_else(|_| root.join(relative))
            .map_err(|e| e.to_string())
            .and_then(|url| Path::from_url_path(url.path()).map_err(|e| e.to_string()));
        let outcome = match location {
            Ok(path) => {
                runtime::block_on(async { store.delete(&path).await }).map_err(|e| e.to_string())
            }
            Err(e) => Err(e),
        };
        match outcome {
            Ok(()) => {}
            Err(e) if e.contains("not found") || e.contains("NotFound") => {}
            Err(e) => failed.push(format!("{relative}: {e}")),
        }
    }
    failed
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
    info: CommitInfoPatch,
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
        info,
    )?;

    if overwrite {
        let scan = scan_source.scan_builder().build()?;
        let scan_metadata = runtime::block_on(async { scan.scan_metadata(engine.as_ref()) })?;
        for filtered in Transaction::scan_metadata_to_engine_data(scan_metadata) {
            transaction.remove_files(filtered?);
        }
    }

    // Decode and check every fragment before adding any, so a bad one cannot
    // leave a half-built transaction behind.
    let mut seen = std::collections::HashSet::new();
    let mut decoded = Vec::new();
    for fragment in &fragments {
        for batch in ipc_to_batches(fragment, &table_root, &metadata_id)? {
            refuse_duplicate_paths(&batch, &mut seen)?;
            decoded.push(batch);
        }
    }
    for batch in decoded {
        transaction.add_files(Box::new(ArrowEngineData::new(batch)));
    }

    finish_commit(transaction, &engine)
}

/// Refuse a data file that is added twice in one commit.
///
/// The same fragment passed twice (a retried task whose first result was also
/// kept, or a list concatenated with itself) produced two `add` actions for one
/// path in one commit. The protocol forbids that; readers that replay the log
/// file-by-file count the rows twice, and the change feed reports them twice.
fn refuse_duplicate_paths(
    batch: &arrow::array::RecordBatch,
    seen: &mut std::collections::HashSet<String>,
) -> Result<()> {
    use arrow::array::{Array, StringArray};

    let Some(column) = batch.column_by_name("path") else {
        return Err(NativeError::Invalid(
            "this fragment's add-file metadata has no path column; it was not produced by \
             write_files"
                .to_string(),
        ));
    };
    let column = arrow::compute::cast(column, &arrow::datatypes::DataType::Utf8)?;
    let paths = column
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| NativeError::Invalid("fragment paths are not strings".to_string()))?;
    for i in 0..paths.len() {
        if paths.is_null(i) {
            return Err(NativeError::Invalid(
                "a fragment names a data file with a null path".to_string(),
            ));
        }
        let path = paths.value(i);
        if !seen.insert(path.to_string()) {
            return Err(NativeError::Invalid(format!(
                "data file {path:?} appears in more than one fragment (or twice in one). \
                 Committing it twice would add the same file twice in one version; pass each \
                 worker's fragment exactly once"
            )));
        }
    }
    Ok(())
}

/// Run a prepared transaction's commit and classify the outcome.
pub(crate) fn finish_commit(txn: Transaction, engine: &SharedEngine) -> Result<u64> {
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
        Err(err) => Err(classify_kernel_commit_error(err)),
    }
}

#[cfg(test)]
mod commit_info_tests {
    use super::*;
    use arrow::array::{Array, BooleanArray, MapArray, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field};

    fn commit_info_batch() -> arrow::array::RecordBatch {
        let operation = Arc::new(StringArray::from(vec!["WRITE"])) as arrow::array::ArrayRef;
        let fields = vec![Arc::new(Field::new("operation", DataType::Utf8, true))];
        let info = StructArray::try_new(fields.into(), vec![operation], None).unwrap();
        let schema = arrow::datatypes::Schema::new(vec![Field::new(
            "commitInfo",
            info.data_type().clone(),
            true,
        )]);
        arrow::array::RecordBatch::try_new(Arc::new(schema), vec![Arc::new(info)]).unwrap()
    }

    #[test]
    fn the_patch_writes_parameters_and_the_blind_append_flag() {
        let patch = CommitInfoPatch {
            operation_parameters: Some([("mode".to_string(), "Overwrite".to_string())].into()),
            blind_append: Some(false),
        };
        let data = FilteredEngineData::with_all_rows_selected(Box::new(ArrowEngineData::new(
            commit_info_batch(),
        )));
        let (data, _) = patch.apply(data).unwrap().into_parts();
        let batch: arrow::array::RecordBatch =
            (*data.into_any().downcast::<ArrowEngineData>().unwrap()).into();
        let info = batch
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let blind = info
            .column_by_name("isBlindAppend")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(!blind.value(0) && blind.is_valid(0));
        let parameters = info
            .column_by_name("operationParameters")
            .unwrap()
            .as_any()
            .downcast_ref::<MapArray>()
            .unwrap();
        let entries = parameters.value(0);
        let keys = entries
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values = entries
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!((keys.value(0), values.value(0)), ("mode", "Overwrite"));
        assert!(
            info.column_by_name("operation").is_some(),
            "other fields kept"
        );
    }

    #[test]
    fn other_actions_pass_through_unchanged() {
        let patch = CommitInfoPatch {
            operation_parameters: None,
            blind_append: Some(true),
        };
        let column = Arc::new(StringArray::from(vec!["x"])) as arrow::array::ArrayRef;
        let schema = arrow::datatypes::Schema::new(vec![Field::new("add", DataType::Utf8, true)]);
        let batch = arrow::array::RecordBatch::try_new(Arc::new(schema), vec![column]).unwrap();
        let data = FilteredEngineData::with_all_rows_selected(Box::new(ArrowEngineData::new(
            batch.clone(),
        )));
        let (data, _) = patch.apply(data).unwrap().into_parts();
        let out: arrow::array::RecordBatch =
            (*data.into_any().downcast::<ArrowEngineData>().unwrap()).into();
        assert_eq!(out, batch);
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
        assert_eq!(commit_raw(&url, &opts, 0, &actions).unwrap(), 0);
        let written =
            std::fs::read_to_string(dir.join("_delta_log/00000000000000000000.json")).unwrap();
        assert_eq!(written, "{\"commitInfo\":{}}\n");
        let err = commit_raw(&url, &opts, 0, &actions).unwrap_err();
        assert!(matches!(err, NativeError::CommitConflict(_)), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_raw_commit_that_would_leave_a_gap_is_refused() {
        let dir = std::env::temp_dir().join(format!("ds-raw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let url = url::Url::from_directory_path(&dir).unwrap();
        let opts = std::collections::HashMap::new();
        let actions = [r#"{"commitInfo":{}}"#.to_string()];
        let err = commit_raw(&url, &opts, 3, &actions).unwrap_err();
        assert!(err.to_string().contains("gap"), "{err}");
        assert!(!dir.join("_delta_log/00000000000000000003.json").exists());
        commit_raw(&url, &opts, 0, &actions).unwrap();
        assert_eq!(commit_raw(&url, &opts, 1, &actions).unwrap(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn status_codes_are_read_from_the_status_not_from_uuids_or_versions() {
        // A UUID or zero-padded version containing 429/409 is not a status.
        let msg = "UC update_table error: HTTP error (status 400): bad staged commit \
                   _delta_log/_staged_commits/00000000000000000429.a429b-409c.json";
        assert!(matches!(
            classify_commit_error(msg),
            NativeError::CatalogRejected(_)
        ));
        let msg = "HTTP error (status 400): conflicting table properties";
        assert!(matches!(
            classify_commit_error(msg),
            NativeError::CatalogRejected(_)
        ));
        let msg = "UC update_table error: HTTP error (status 409): version exists";
        assert!(matches!(
            classify_commit_error(msg),
            NativeError::CommitConflict(_)
        ));
        let msg = "UC update_table error: HTTP error (status 429): too many";
        assert!(matches!(
            classify_commit_error(msg),
            NativeError::BackfillRequired(_)
        ));
        // No status: whole-number codes and keywords still classify.
        assert!(matches!(
            classify_commit_error("got 409 back"),
            NativeError::CommitConflict(_)
        ));
        assert!(matches!(
            classify_commit_error("file 00000000000000000409.json is bad"),
            NativeError::Invalid(_)
        ));
        assert!(matches!(
            classify_commit_error("max unbackfilled commits reached"),
            NativeError::BackfillRequired(_)
        ));
    }

    #[test]
    fn a_file_added_twice_in_one_commit_is_refused() {
        use arrow::array::{RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![Field::new("path", DataType::Utf8, true)]));
        let batch = |paths: Vec<&str>| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(paths))]).unwrap()
        };
        let mut seen = std::collections::HashSet::new();
        refuse_duplicate_paths(&batch(vec!["a.parquet", "b.parquet"]), &mut seen).unwrap();
        let err =
            refuse_duplicate_paths(&batch(vec!["c.parquet", "a.parquet"]), &mut seen).unwrap_err();
        assert!(err.to_string().contains("a.parquet"), "{err}");
        let mut fresh = std::collections::HashSet::new();
        assert!(refuse_duplicate_paths(&batch(vec!["x", "x"]), &mut fresh).is_err());
    }

    #[test]
    fn every_catalog_status_is_classified() {
        let status = |code: u16| {
            classify_commit_error(&format!(
                "UC update_table error: HTTP error (status {code}): nope"
            ))
        };
        assert!(matches!(status(401), NativeError::CatalogPermission(_)));
        assert!(matches!(status(403), NativeError::CatalogPermission(_)));
        assert!(matches!(status(404), NativeError::CatalogNotFound(_)));
        assert!(matches!(status(500), NativeError::Retryable(_)));
        assert!(matches!(status(503), NativeError::Retryable(_)));
        assert!(matches!(status(400), NativeError::CatalogRejected(_)));
        assert!(matches!(status(409), NativeError::CommitConflict(_)));
        assert!(matches!(status(429), NativeError::BackfillRequired(_)));
        let words = |text: &str| classify_commit_error(&format!("UC update_table error: {text}"));
        assert!(matches!(
            words("Authentication failed"),
            NativeError::CatalogPermission(_)
        ));
        assert!(matches!(
            words("Table not found: t1"),
            NativeError::CatalogNotFound(_)
        ));
        assert!(matches!(
            words("Max unpublished commits exceeded (max: 50)"),
            NativeError::BackfillRequired(_)
        ));
        assert!(matches!(
            classify_commit_error("error sending request: operation timed out"),
            NativeError::Retryable(_)
        ));
        // A kernel refusal with no status is still plain invalid input.
        assert!(matches!(
            classify_commit_error("Invalid transaction state: append-only"),
            NativeError::Invalid(_)
        ));
    }

    #[test]
    fn a_failed_put_precondition_is_a_commit_conflict() {
        // object_store#829: Azure answers a lost put-if-absent with 412.
        let err =
            delta_kernel::Error::ObjectStore(delta_kernel::object_store::Error::Precondition {
                path: "t/_delta_log/00000000000000000001.json".to_string(),
                source: "412 Precondition Failed".into(),
            });
        assert!(matches!(
            classify_kernel_commit_error(err),
            NativeError::CommitConflict(_)
        ));
    }

    #[test]
    fn commit_raw_maps_a_failed_precondition_to_a_conflict() {
        let root = url::Url::parse("az://c/t/").unwrap();
        let precondition = delta_kernel::object_store::Error::Precondition {
            path: "t/_delta_log/00000000000000000001.json".to_string(),
            source: "412 Precondition Failed".into(),
        };
        assert!(matches!(
            raw_put_error(precondition, 1, &root),
            NativeError::CommitConflict(_)
        ));
        let exists = delta_kernel::object_store::Error::AlreadyExists {
            path: "p".to_string(),
            source: "exists".into(),
        };
        assert!(matches!(
            raw_put_error(exists, 1, &root),
            NativeError::CommitConflict(_)
        ));
    }

    #[test]
    fn io_failures_during_commit_stay_io_errors() {
        let err =
            delta_kernel::Error::IOError(std::io::Error::other("connection reset (status 409)"));
        assert!(matches!(
            classify_kernel_commit_error(err),
            NativeError::Kernel(_)
        ));
    }
}
