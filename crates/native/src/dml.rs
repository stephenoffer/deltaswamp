//! Row-level DML through deletion vectors, the way Databricks writes it.
//!
//! A DELETE marks rows deleted rather than rewriting the files that hold them:
//! each touched data file gets a new deletion vector (a RoaringBitmap of the
//! physical row indexes it no longer serves), and the commit swaps the file's
//! `add` for one carrying that vector. An UPDATE is the same delete plus the
//! updated rows appended as new files, in one transaction.
//!
//! The caller says *which* rows to delete, as `(file, physical row index)`
//! pairs from a positional scan ([`crate::scan::FILE_PATH_COLUMN`]). Everything
//! protocol-shaped happens here:
//!
//! * The new deletions are unioned with the file's existing vector, so a
//!   second DELETE never resurrects the rows the first one removed.
//! * A file whose every row is now deleted is removed outright, as Spark does,
//!   unless the table tracks row ids (the kernel cannot stage removes there),
//!   in which case it keeps a vector covering every row -- equally valid.
//! * All vectors go into one `deletion_vector_<uuid>.bin` at the table root,
//!   in the on-disk format the Delta protocol specifies, referenced by
//!   relative (`u`) descriptors. The file is written before the commit and
//!   removed again, best effort, if the commit fails.
//! * The commit is staged against the snapshot the rows were read from, so a
//!   concurrent writer makes it conflict instead of being lost.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StringArray};
use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use delta_kernel::actions::deletion_vector_writer::{
    KernelDeletionVector, StreamingDeletionVectorWriter,
};
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::scan::state::ScanFile;
use delta_kernel::scan::ScanMetadata;
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::table_features::TableFeature;
use delta_kernel::FilteredEngineData;
use roaring::RoaringTreemap;

use crate::commit::{self, SharedEngine, UcCommitConfig};
use crate::error::{NativeError, Result};
use crate::runtime;

/// What a DML commit did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmlOutcome {
    pub version: u64,
    /// Rows newly deleted by this commit (rows already deleted are not counted).
    pub deleted_rows: u64,
    /// Data files that received a new deletion vector.
    pub deletion_vectors_added: usize,
    /// Data files removed because every row in them was deleted.
    pub files_removed: usize,
}

/// Whether the table permits DV writes: the feature supported on both sides of
/// the protocol and `delta.enableDeletionVectors=true`, which is what the kernel
/// checks before it stages any DV update.
pub fn deletion_vectors_enabled(snapshot: &SnapshotRef) -> bool {
    snapshot
        .table_configuration()
        .is_feature_enabled(&TableFeature::DeletionVectors)
}

/// Whether the kernel will stage `remove` actions on this table.
///
/// Mirrors the kernel's own check (`validate_feature_support_for_remove`),
/// which is crate-private: row tracking supported and not suspended, or
/// icebergCompatV3 enabled, refuses every remove at commit time.
fn removes_allowed(snapshot: &SnapshotRef) -> bool {
    let config = snapshot.table_configuration();
    let suspended = snapshot
        .metadata_configuration()
        .get("delta.rowTracking.suspended")
        .is_some_and(|v| v.eq_ignore_ascii_case("true"));
    let row_tracking = config.is_feature_supported(&TableFeature::RowTracking) && !suspended;
    !row_tracking && !config.is_feature_enabled(&TableFeature::IcebergCompatV3)
}

/// The random directory prefix for a new DV file, honoring
/// `delta.randomizeFilePrefixes` / `delta.randomPrefixLength` as Spark does.
fn random_prefix(snapshot: &SnapshotRef) -> String {
    let properties = snapshot.metadata_configuration();
    let randomize = properties
        .get("delta.randomizeFilePrefixes")
        .is_some_and(|v| v.eq_ignore_ascii_case("true"));
    if !randomize {
        return String::new();
    }
    let len = properties
        .get("delta.randomPrefixLength")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| (1..=16).contains(n))
        .unwrap_or(2);
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let bytes = uuid::Uuid::new_v4().into_bytes();
    (0..len)
        .map(|i| CHARSET[bytes[i % bytes.len()] as usize % CHARSET.len()] as char)
        .collect()
}

/// A new DV file's location: the log's encoded form and the path relative to
/// the table root, per the protocol's "Derived Fields" for storage type `u`.
struct DvFileName {
    encoded: String,
    relative: String,
}

impl DvFileName {
    fn new(prefix: &str) -> Self {
        let uuid = uuid::Uuid::new_v4();
        let encoded = format!("{prefix}{}", z85::encode(uuid.as_bytes()));
        let file = format!("deletion_vector_{uuid}.bin");
        let relative = if prefix.is_empty() {
            file
        } else {
            format!("{prefix}/{file}")
        };
        Self { encoded, relative }
    }
}

/// One live file the DML touches, and what becomes of it.
struct Touched {
    /// Every deleted row after this commit: the old vector plus the new rows.
    deleted: RoaringTreemap,
    newly_deleted: u64,
    remove: bool,
}

/// The rows a DML commit adds.
pub enum DmlData {
    /// Collected up front: an UPDATE's or MERGE's rewritten rows.
    Batches(Vec<RecordBatch>),
    /// Pulled one batch at a time and written as each arrives: a compaction's
    /// output files, each one batch.
    Stream(Box<dyn Iterator<Item = Result<RecordBatch>> + Send>),
}

/// Features that constrain only the values a commit writes.
///
/// A compaction writes back exactly the values it read (`dataChange=false`),
/// already checked when they were first written: a CHECK constraint, an
/// invariant or a generation expression holds for them as it did, and an
/// identity column's values and high-water mark are untouched. delta-kernel
/// refuses any transaction on a table carrying one of these -- a writer
/// version 3 or 4 table implies two of them, so every table delta-rs gave a
/// change data feed was refused -- which left those tables to delta-rs's
/// OPTIMIZE, whose commit duplicates rows under a concurrent compaction.
const VALUE_CONSTRAINTS: &[&str] = &[
    "checkConstraints",
    "generatedColumns",
    "identityColumns",
    "invariants",
];

/// The writer features a legacy writer version implies, in protocol order.
fn legacy_writer_features(version: i32) -> Vec<&'static str> {
    let mut features = Vec::new();
    if version >= 2 {
        features.extend(["appendOnly", "invariants"]);
    }
    if version >= 3 {
        features.push("checkConstraints");
    }
    if version >= 4 {
        features.extend(["changeDataFeed", "generatedColumns"]);
    }
    if version >= 5 {
        features.push("columnMapping");
    }
    if version >= 6 {
        features.push("identityColumns");
    }
    features
}

/// `snapshot` as a compaction's transaction sees it: the same log segment,
/// metadata and version, with [`VALUE_CONSTRAINTS`] left out of the protocol
/// kernel checks the write against.
///
/// Only the transaction's pre-commit check ever sees this protocol. A
/// compaction's commit carries no protocol action (the table's own protocol
/// stays in force, for every reader and writer after it), its conflict check
/// is the object store's put-if-absent of the next version, and the post-
/// commit snapshot kernel builds from it is dropped: checkpoints are written
/// from a snapshot read back from storage.
pub(crate) fn compaction_snapshot(snapshot: &SnapshotRef) -> Result<SnapshotRef> {
    use delta_kernel::actions::Protocol;
    use delta_kernel::snapshot::Snapshot;
    use delta_kernel::table_configuration::TableConfiguration;

    let config = snapshot.table_configuration();
    let protocol = config.protocol();
    let writer = protocol.min_writer_version();
    let listed: Vec<String> = match protocol.writer_features() {
        Some(features) => features.iter().map(|f| f.to_string()).collect(),
        None => legacy_writer_features(writer)
            .into_iter()
            .map(str::to_string)
            .collect(),
    };
    if !listed
        .iter()
        .any(|f| VALUE_CONSTRAINTS.contains(&f.as_str()))
    {
        return Ok(snapshot.clone());
    }
    let kept: Vec<&String> = listed
        .iter()
        .filter(|f| !VALUE_CONSTRAINTS.contains(&f.as_str()))
        .collect();
    let reader = protocol.min_reader_version();
    let reader_features: Option<Vec<String>> = protocol
        .reader_features()
        .map(|features| features.iter().map(|f| f.to_string()).collect());
    // A legacy writer version becomes the same features listed explicitly
    // (writer version 7), the only form in which some can be left out.
    let protocol: Protocol = serde_json::from_value(serde_json::json!({
        "minReaderVersion": reader,
        "minWriterVersion": 7,
        "readerFeatures": reader_features,
        "writerFeatures": kept,
    }))
    .map_err(|e| {
        NativeError::Invalid(format!(
            "could not restate the table's protocol for its compaction: {e}"
        ))
    })?;
    let config = TableConfiguration::try_new(
        config.metadata().clone(),
        protocol,
        snapshot.table_root().clone(),
        snapshot.version(),
    )?;
    Ok(Arc::new(Snapshot::new(
        snapshot.log_segment().clone(),
        config,
    )?))
}

/// Commit a DELETE (and, with `batches`, an UPDATE's new rows) as deletion vectors.
///
/// `deletions` maps a data file's log path to the physical row indexes to
/// delete from it. Indexes already deleted are ignored; an index past the
/// file's `numRecords`, or a path that is not a live file of this snapshot, is
/// refused before anything is written.
///
/// `whole_files` names files every live row of which is deleted, without
/// listing the rows: a DELETE with no predicate, or a file the caller is
/// rewriting (its surviving rows are in `batches`). Such a file is removed; a
/// full-file vector stands in where row tracking forbids removes, which needs
/// the file's `numRecords`. Where an add carries no such statistic, the row
/// count comes from the file's Parquet footer.
#[allow(clippy::too_many_arguments)]
pub fn commit_dml(
    snapshot: SnapshotRef,
    engine: SharedEngine,
    deletions: HashMap<String, RoaringTreemap>,
    whole_files: HashSet<String>,
    data: DmlData,
    uc: Option<UcCommitConfig>,
    engine_info: Option<String>,
    operation: Option<String>,
    txn: Option<(String, i64)>,
    commit_metadata: Option<HashMap<String, String>>,
    data_change: bool,
    info: commit::CommitInfoPatch,
) -> Result<DmlOutcome> {
    let deletions: HashMap<String, RoaringTreemap> = deletions
        .into_iter()
        .filter(|(path, rows)| !rows.is_empty() && !whole_files.contains(path))
        .collect();
    if !deletions.is_empty() && !deletion_vectors_enabled(&snapshot) {
        return Err(NativeError::Invalid(
            "the table does not have deletion vectors enabled (the deletionVectors reader and \
             writer feature, and delta.enableDeletionVectors=true), so rows cannot be deleted \
             by deletion vector"
                .to_string(),
        ));
    }
    let table_root = snapshot.table_root().clone();
    let codec = crate::writer::codec_for(&snapshot);

    // Log replay once: the scan files to join the deletions against, kept
    // whole because the kernel's DV update and remove both take them as is.
    let scan = snapshot.clone().scan_builder().build()?;
    let metadata: Vec<ScanMetadata> =
        runtime::block_on(async { scan.scan_metadata(engine.as_ref()) })?
            .collect::<std::result::Result<_, _>>()?;

    let wanted: HashSet<&str> = deletions
        .keys()
        .chain(whole_files.iter())
        .map(String::as_str)
        .collect();
    let mut live: HashMap<String, ScanFile> = HashMap::new();
    for item in &metadata {
        fn collect(files: &mut Vec<ScanFile>, file: ScanFile) {
            files.push(file);
        }
        for file in item.visit_scan_files(Vec::new(), collect)? {
            if wanted.contains(file.path.as_str()) {
                live.insert(file.path.clone(), file);
            }
        }
    }

    // Decide every file's fate before any I/O, so bad input writes nothing.
    let allow_remove = removes_allowed(&snapshot);
    // Files whose add carries no `numRecords` in its JSON stats, with the row
    // count read from their Parquet footer instead. Databricks writes such
    // adds as a matter of course: its tables default to
    // `delta.checkpoint.writeStatsAsJson=false`, so every add in a checkpoint
    // has only `stats_parsed`, which the kernel's scan files do not carry.
    let mut unsized_files: HashMap<String, u64> = HashMap::new();
    let mut touched: BTreeMap<String, Touched> = BTreeMap::new();
    for path in &whole_files {
        let file = live.get(path).ok_or_else(|| {
            NativeError::Invalid(format!(
                "data file {path:?} is not live in version {} of the table; rows can be deleted \
                 only from the snapshot they were read from",
                snapshot.version()
            ))
        })?;
        let already = match file.dv_info.get_row_indexes(engine.as_ref(), &table_root)? {
            Some(rows) => rows.len() as u64,
            None => 0,
        };
        // Without row tracking an unsized file is simply removed, so its row
        // count is not worth a footer read; a full-file vector needs it.
        let num_records = match file.stats.as_ref() {
            Some(stats) => Some(stats.num_records),
            None if allow_remove => None,
            None => {
                let n = footer_num_rows(&engine, &table_root, file)?;
                unsized_files.insert(path.clone(), n);
                Some(n)
            }
        };
        match num_records {
            Some(n) if n == already => continue, // nothing left to delete
            Some(n) => {
                let mut deleted = RoaringTreemap::new();
                deleted.insert_range(0..n);
                touched.insert(
                    path.clone(),
                    Touched {
                        deleted,
                        newly_deleted: n - already,
                        remove: allow_remove,
                    },
                );
            }
            None if allow_remove => {
                touched.insert(
                    path.clone(),
                    Touched {
                        deleted: RoaringTreemap::new(),
                        // Unknown without statistics; the caller counts these.
                        newly_deleted: 0,
                        remove: true,
                    },
                );
            }
            None => {
                return Err(NativeError::Invalid(format!(
                    "data file {path:?} has no numRecords statistic and the table tracks row \
                     ids, so it can neither take a deletion vector nor be removed"
                )))
            }
        }
    }
    for (path, rows) in deletions {
        let file = live.get(&path).ok_or_else(|| {
            NativeError::Invalid(format!(
                "data file {path:?} is not live in version {} of the table; rows can be deleted \
                 only from the snapshot they were read from",
                snapshot.version()
            ))
        })?;
        let num_records = match file.stats.as_ref() {
            Some(stats) => stats.num_records,
            None => {
                let n = footer_num_rows(&engine, &table_root, file)?;
                unsized_files.insert(path.clone(), n);
                n
            }
        };
        if let Some(max) = rows.max() {
            if max >= num_records {
                return Err(NativeError::Invalid(format!(
                    "row index {max} is out of range for data file {path:?}, which holds \
                     {num_records} rows"
                )));
            }
        }
        let mut deleted = RoaringTreemap::new();
        if let Some(existing) = file.dv_info.get_row_indexes(engine.as_ref(), &table_root)? {
            deleted.extend(existing);
        }
        let before = deleted.len();
        deleted |= rows;
        let newly_deleted = deleted.len() - before;
        if newly_deleted == 0 {
            continue;
        }
        let remove = allow_remove && deleted.len() == num_records;
        touched.insert(
            path,
            Touched {
                deleted,
                newly_deleted,
                remove,
            },
        );
    }

    let deleted_rows: u64 = touched.values().map(|t| t.newly_deleted).sum();
    let (batches, stream) = match data {
        DmlData::Batches(batches) => (batches, None),
        DmlData::Stream(_) if data_change => {
            return Err(NativeError::Invalid(
                "only a compaction (data_change=False) streams its rows".to_string(),
            ))
        }
        // Nothing to remove: whatever the stream holds is all there is to commit.
        DmlData::Stream(stream) if touched.is_empty() => {
            (stream.collect::<Result<Vec<_>>>()?, None)
        }
        DmlData::Stream(stream) => (Vec::new(), Some(stream)),
    };
    if touched.is_empty() && batches.is_empty() {
        // Nothing to change: no commit, as Spark writes none for a no-op DELETE.
        return Ok(DmlOutcome {
            version: snapshot.version(),
            deleted_rows: 0,
            deletion_vectors_added: 0,
            files_removed: 0,
        });
    }

    let partition_columns = snapshot
        .table_configuration()
        .logical_partition_columns()
        .to_vec();
    let table_schema = snapshot.schema();
    let materialized_row_ids = materialized_row_id_column(&snapshot, &batches)?;
    let batches = if data_change {
        prepare_with_row_ids(&snapshot, batches)?
    } else {
        // A compaction sized each batch as one output file; coalescing them
        // would ignore the table's target file size.
        let mut prepared = Vec::with_capacity(batches.len());
        for batch in batches {
            prepared.extend(commit::prepare_batches(&snapshot, vec![batch])?);
        }
        prepared
    };

    // Serialize every new vector into one file, in the protocol's format.
    let mut descriptors: HashMap<String, DeletionVectorDescriptor> = HashMap::new();
    let mut dv_file: Option<(DvFileName, Vec<u8>)> = None;
    if touched.values().any(|t| !t.remove) {
        let name = DvFileName::new(&random_prefix(&snapshot));
        let mut buffer = Vec::new();
        {
            let mut writer = StreamingDeletionVectorWriter::new(&mut buffer);
            for (path, t) in touched.iter().filter(|(_, t)| !t.remove) {
                let mut dv = KernelDeletionVector::new();
                dv.add_deleted_row_indexes(t.deleted.iter());
                let written = writer.write_deletion_vector(dv)?;
                let descriptor = DeletionVectorDescriptor::try_new(
                    DeletionVectorStorageType::PersistedRelative,
                    name.encoded.clone(),
                    Some(written.offset),
                    written.size_in_bytes,
                    written.cardinality,
                )?;
                descriptors.insert(path.clone(), descriptor);
            }
            writer.finalize()?;
        }
        dv_file = Some((name, buffer));
    }
    let removals: HashSet<&str> = touched
        .iter()
        .filter(|(_, t)| t.remove)
        .map(|(p, _)| p.as_str())
        .collect();

    // A compaction commits on the table's own snapshot with the features
    // that only constrain the values a commit writes set aside; see
    // `compaction_snapshot`.
    let committing = if data_change {
        snapshot.clone()
    } else {
        compaction_snapshot(&snapshot)?
    };
    let mut transaction = commit::begin_transaction(
        committing,
        &engine,
        &uc,
        engine_info,
        operation,
        txn,
        commit_metadata,
        info,
    )?;
    if !data_change {
        // A compaction: the same rows, moved from the removed files into the
        // new ones. Readers of the change feed and streams skip it.
        transaction = transaction.with_data_change(false);
    }

    // Split each scan-file batch into the rows whose vector changes and the
    // rows removed outright. Both halves share the same data.
    let mut dv_files = Vec::new();
    for item in metadata {
        let (data, selection) = item.scan_files.into_parts();
        let batch: RecordBatch =
            (*data.into_any().downcast::<ArrowEngineData>().map_err(|_| {
                NativeError::Invalid(
                    "the kernel returned scan files that are not Arrow".to_string(),
                )
            })?)
            .into();
        let paths = scan_file_paths(&batch)?;
        let selected = |i: usize| selection.get(i).copied().unwrap_or(true);
        if !removals.is_empty() {
            let remove: Vec<bool> = (0..batch.num_rows())
                .map(|i| selected(i) && !paths.is_null(i) && removals.contains(paths.value(i)))
                .collect();
            if remove.iter().any(|r| *r) {
                transaction.remove_files(FilteredEngineData::try_new(
                    Box::new(ArrowEngineData::new(batch.clone())),
                    remove,
                )?);
            }
        }
        if !descriptors.is_empty() {
            let update: Vec<bool> = (0..batch.num_rows())
                .map(|i| {
                    selected(i) && !paths.is_null(i) && descriptors.contains_key(paths.value(i))
                })
                .collect();
            if update.iter().any(|u| *u) {
                // The kernel's DV update rewrites each add's JSON stats (it
                // widens tightBounds) and refuses an add without numRecords,
                // so an unsized file gets the count its footer gave.
                let batch = if unsized_files.is_empty() {
                    batch
                } else {
                    with_num_records(&batch, &paths, &unsized_files)?
                };
                // The kernel widens by parsing the stats into a
                // serde_json::Value and writing it back, which turns every
                // number into an f64: a DECIMAL(38,18) max of
                // 12345678901234567890.123456789012345678 came back as
                // 1.2345678901234567e+19, below the real value, and
                // Databricks skipped the file for `dec = <that value>`.
                // Stats already wide are kept verbatim, so they are widened
                // here, textually.
                let batch = with_wide_stats(&batch)?;
                dv_files.push(Ok(FilteredEngineData::try_new(
                    Box::new(ArrowEngineData::new(batch)),
                    update,
                )?));
            }
        }
    }
    let deletion_vectors_added = descriptors.len();
    if !descriptors.is_empty() {
        transaction.update_deletion_vectors(descriptors, dv_files.into_iter())?;
    }

    // Upload the vectors only once the transaction has accepted them, and
    // take them back out if the commit does not land.
    let dv_relative = match dv_file {
        Some((name, bytes)) => {
            put_new_file(&engine, &table_root, &name.relative, bytes)?;
            Some(name.relative)
        }
        None => None,
    };
    let mut written = Vec::new();
    let staged = match (&materialized_row_ids, stream) {
        (_, Some(stream)) => commit::stage_stream(
            &mut transaction,
            &engine,
            &partition_columns,
            &table_schema,
            stream.map(|batch| {
                // One batch is one file: each is conformed on its own and
                // written as it arrives, so the rows of a whole compaction
                // step are never in memory at once.
                batch.and_then(|b| commit::prepare_batches(&snapshot, vec![b]))
            }),
            codec,
            &mut written,
        ),
        (None, None) => commit::stage_batches(
            &mut transaction,
            &engine,
            &partition_columns,
            &table_schema,
            batches,
            codec,
            &mut written,
        ),
        (Some(column), None) => stage_with_row_ids(
            &mut transaction,
            &engine,
            &partition_columns,
            &table_schema,
            batches,
            column,
            codec,
            &mut written,
        ),
    };
    let staging_failed = staged.is_err();
    let result = staged.and_then(|()| commit::finish_commit(transaction, &engine));
    if let Err(err) = &result {
        if staging_failed || commit::never_committed(err) {
            // Rows a lost race or a failed step wrote, which no commit names:
            // orphans until VACUUM otherwise.
            let _ = commit::remove_written(&engine, &table_root, &written);
        }
    }
    match result {
        Ok(version) => Ok(DmlOutcome {
            version,
            deleted_rows,
            deletion_vectors_added,
            files_removed: removals.len(),
        }),
        Err(err) => {
            if let Some(relative) = dv_relative {
                let _ = commit::remove_written(&engine, &table_root, &[relative]);
            }
            Err(err)
        }
    }
}

/// The physical column that carries materialized row ids, when `batches` bring them.
///
/// Rows an UPDATE rewrites must keep their row ids on a table with row
/// tracking enabled. A new file's rows would otherwise get fresh ids from its
/// `baseRowId`, so the old ids are written into the column the table names in
/// `delta.rowTracking.materializedRowIdColumnName`, which readers prefer over
/// the computed id. Their commit version is left unmaterialized, which reads
/// as this commit's -- correct for a row this commit changed.
fn materialized_row_id_column(
    snapshot: &SnapshotRef,
    batches: &[RecordBatch],
) -> Result<Option<String>> {
    let carries = batches
        .iter()
        .any(|b| b.schema().index_of(crate::scan::ROW_ID_COLUMN).is_ok());
    if !carries {
        return Ok(None);
    }
    snapshot
        .metadata_configuration()
        .get("delta.rowTracking.materializedRowIdColumnName")
        .cloned()
        .map(Some)
        .ok_or_else(|| {
            NativeError::Invalid(
                "the data carries row ids, but the table names no \
                 delta.rowTracking.materializedRowIdColumnName to write them to"
                    .to_string(),
            )
        })
}

/// `prepare_batches`, carrying a row-id column through untouched.
///
/// Conforming to the table schema refuses unknown columns, so the row ids are
/// set aside and put back afterwards; conforming keeps rows in order.
fn prepare_with_row_ids(
    snapshot: &SnapshotRef,
    batches: Vec<RecordBatch>,
) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        let Ok(index) = batch.schema().index_of(crate::scan::ROW_ID_COLUMN) else {
            out.extend(commit::prepare_batches(snapshot, vec![batch])?);
            continue;
        };
        let row_ids = batch.column(index).clone();
        let mut rest = batch.clone();
        rest.remove_column(index);
        let conformed = commit::prepare_batches(snapshot, vec![rest])?;
        let Some(conformed) = conformed.into_iter().next() else {
            continue;
        };
        let row_ids = arrow::compute::cast(&row_ids, &arrow::datatypes::DataType::Int64)?;
        out.push(with_column(
            &conformed,
            crate::scan::ROW_ID_COLUMN,
            row_ids,
        )?);
    }
    commit::coalesce(out)
}

/// `batch` with `array` appended as the nullable column `name`.
fn with_column(
    batch: &RecordBatch,
    name: &str,
    array: arrow::array::ArrayRef,
) -> Result<RecordBatch> {
    let mut fields: Vec<arrow::datatypes::FieldRef> =
        batch.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(arrow::datatypes::Field::new(
        name,
        array.data_type().clone(),
        true,
    )));
    let mut columns = batch.columns().to_vec();
    columns.push(array);
    Ok(RecordBatch::try_new(
        Arc::new(arrow::datatypes::Schema::new_with_metadata(
            fields,
            batch.schema().metadata().clone(),
        )),
        columns,
    )?)
}

/// `commit::stage_batches`, writing each batch's row ids to `column`.
///
/// Mirrors `writer::write_parquet` -- the logical-to-physical transform, then
/// the Parquet write -- with the materialized row-id column appended to
/// the physical data in between. Batches without row ids are written normally.
#[allow(clippy::too_many_arguments)]
fn stage_with_row_ids(
    txn: &mut delta_kernel::transaction::Transaction,
    engine: &SharedEngine,
    partition_columns: &[String],
    table_schema: &delta_kernel::schema::SchemaRef,
    batches: Vec<RecordBatch>,
    column: &str,
    codec: delta_kernel::parquet::basic::Compression,
    written: &mut Vec<String>,
) -> Result<()> {
    use delta_kernel::engine::arrow_conversion::TryFromArrow;
    use delta_kernel::engine::arrow_data::EngineDataArrowExt;
    use delta_kernel::schema::StructType;
    use delta_kernel::Engine;

    let write_state = txn.write_state()?;
    let mut groups: Vec<(
        RecordBatch,
        Option<HashMap<String, delta_kernel::expressions::Scalar>>,
    )> = Vec::new();
    for batch in batches {
        if partition_columns.is_empty() {
            groups.push((batch, None));
        } else {
            for group in crate::partition::split_by_partition(
                &batch,
                partition_columns,
                table_schema.as_ref(),
            )? {
                groups.push((group.data, Some(group.values)));
            }
        }
    }
    let mut staged = Vec::new();
    for (batch, values) in groups {
        let write_context = match values {
            None => write_state.unpartitioned_write_context()?,
            Some(values) => write_state.partitioned_write_context(values)?,
        };
        let Ok(index) = batch.schema().index_of(crate::scan::ROW_ID_COLUMN) else {
            let data = ArrowEngineData::new(batch);
            let metadata = runtime::block_on(crate::writer::write_parquet(
                engine,
                &data,
                &write_context,
                codec,
            ))?;
            staged.push(commit::track_written(metadata, written)?);
            continue;
        };
        let row_ids = batch.column(index).clone();
        let mut logical = batch;
        logical.remove_column(index);
        let input_schema = StructType::try_from_arrow(logical.schema().as_ref())?;
        let evaluator = engine.evaluation_handler().new_expression_evaluator(
            Arc::new(input_schema),
            write_context.logical_to_physical(),
            write_context.physical_schema().clone().into(),
        )?;
        let physical = evaluator
            .evaluate(&ArrowEngineData::new(logical))?
            .try_into_record_batch()?;
        let physical = with_column(&physical, column, row_ids)?;
        let metadata = runtime::block_on(crate::writer::write_physical(
            engine,
            &physical,
            &write_context,
            codec,
        ))?;
        staged.push(commit::track_written(metadata, written)?);
    }
    for metadata in staged {
        txn.add_files(metadata);
    }
    Ok(())
}

/// The `path` column of a scan-file batch.
fn scan_file_paths(batch: &RecordBatch) -> Result<StringArray> {
    let column = batch.column_by_name("path").ok_or_else(|| {
        NativeError::Invalid("the kernel's scan files carry no path column".to_string())
    })?;
    let column = arrow::compute::cast(column, &arrow::datatypes::DataType::Utf8)?;
    column
        .as_any()
        .downcast_ref::<StringArray>()
        .cloned()
        .ok_or_else(|| NativeError::Invalid("scan-file paths are not strings".to_string()))
}

/// A data file's row count, from its Parquet footer.
///
/// For adds whose JSON stats lack `numRecords`; one ranged read of the footer.
fn footer_num_rows(engine: &SharedEngine, root: &url::Url, file: &ScanFile) -> Result<u64> {
    use delta_kernel::object_store::path::Path;
    use delta_kernel::object_store::ObjectStoreExt;
    use delta_kernel::parquet::file::metadata::{FooterTail, ParquetMetaDataReader};

    let unreadable = |why: String| {
        NativeError::Invalid(format!(
            "data file {:?} has no numRecords statistic, and its Parquet footer could not \
             be read for the row count a deletion vector requires: {why}",
            file.path
        ))
    };
    let url = root.join(&file.path)?;
    crate::confine::check(root, &url)?;
    let store = engine
        .get_object_store_for_url(&url)
        .ok_or_else(|| NativeError::Invalid(format!("no object store is registered for {url}")))?;
    let location = Path::from_url_path(url.path())
        .map_err(|e| NativeError::Invalid(format!("invalid data file path {url}: {e}")))?;
    runtime::block_on(async {
        let size = match u64::try_from(file.size) {
            Ok(size) if size > 0 => size,
            _ => store.head(&location).await?.size,
        };
        if size < 8 {
            return Err(unreadable(format!("the file is only {size} bytes")));
        }
        let tail = store.get_range(&location, size - 8..size).await?;
        let tail: [u8; 8] = tail[..]
            .try_into()
            .map_err(|_| unreadable("short read".to_string()))?;
        let footer = FooterTail::try_new(&tail).map_err(|e| unreadable(e.to_string()))?;
        if footer.is_encrypted_footer() {
            return Err(unreadable("the footer is encrypted".to_string()));
        }
        let length = footer.metadata_length() as u64;
        if length + 8 > size {
            return Err(unreadable("the footer length exceeds the file".to_string()));
        }
        let bytes = store
            .get_range(&location, size - 8 - length..size - 8)
            .await?;
        let metadata = ParquetMetaDataReader::decode_metadata(&bytes)
            .map_err(|e| unreadable(e.to_string()))?;
        u64::try_from(metadata.file_metadata().num_rows())
            .map_err(|_| unreadable("the footer reports a negative row count".to_string()))
    })
}

/// `batch` (scan files) with `{"numRecords": n}` as the stats of each file in
/// `counts` that has none.
fn with_num_records(
    batch: &RecordBatch,
    paths: &StringArray,
    counts: &HashMap<String, u64>,
) -> Result<RecordBatch> {
    let schema = batch.schema();
    let index = schema.index_of("stats")?;
    let original = batch.column(index);
    let stats = arrow::compute::cast(original, &arrow::datatypes::DataType::Utf8)?;
    let stats = stats
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| NativeError::Invalid("scan-file stats are not strings".to_string()))?;
    let patched: StringArray = (0..batch.num_rows())
        .map(|i| {
            if !stats.is_null(i) {
                return Some(stats.value(i).to_string());
            }
            if paths.is_null(i) {
                return None;
            }
            counts
                .get(paths.value(i))
                .map(|n| format!("{{\"numRecords\":{n}}}"))
        })
        .collect();
    let patched = arrow::compute::cast(&patched, original.data_type())?;
    let mut columns = batch.columns().to_vec();
    columns[index] = patched;
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// `batch` (scan files) with `"tightBounds":false` in every file's stats,
/// every other value kept as the log spells it.
fn with_wide_stats(batch: &RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();
    let index = schema.index_of("stats")?;
    let original = batch.column(index);
    let stats = arrow::compute::cast(original, &arrow::datatypes::DataType::Utf8)?;
    let stats = stats
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| NativeError::Invalid("scan-file stats are not strings".to_string()))?;
    let widened: StringArray = stats
        .iter()
        .map(|value| value.map(|s| wide_stats(s).unwrap_or_else(|| s.to_string())))
        .collect();
    let widened = arrow::compute::cast(&widened, original.data_type())?;
    let mut columns = batch.columns().to_vec();
    columns[index] = widened;
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// `stats` (one add's JSON stats) with `tightBounds` false, or None when it
/// is not a JSON object (the kernel then reports it). Each value is copied as
/// raw JSON text, so no number goes through a float.
fn wide_stats(stats: &str) -> Option<String> {
    use serde::de::{Deserializer, MapAccess, Visitor};
    use serde_json::value::RawValue;

    struct Entries(Vec<(String, Box<RawValue>)>);
    impl<'de> serde::Deserialize<'de> for Entries {
        fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Entries;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a JSON object")
                }
                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> std::result::Result<Entries, A::Error> {
                    let mut out = Vec::new();
                    while let Some(entry) = map.next_entry::<String, Box<RawValue>>()? {
                        out.push(entry);
                    }
                    Ok(Entries(out))
                }
            }
            d.deserialize_map(V)
        }
    }

    const TIGHT_BOUNDS: &str = "tightBounds";
    let Entries(entries) = serde_json::from_str(stats).ok()?;
    let mut out = String::from("{");
    let mut seen = false;
    for (key, value) in &entries {
        if out.len() > 1 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(key).ok()?);
        out.push(':');
        if key == TIGHT_BOUNDS {
            seen = true;
            out.push_str("false");
        } else {
            out.push_str(value.get());
        }
    }
    if !seen {
        if out.len() > 1 {
            out.push(',');
        }
        out.push_str("\"tightBounds\":false");
    }
    out.push('}');
    Some(out)
}

/// Write a new, uniquely named file under the table root.
fn put_new_file(
    engine: &SharedEngine,
    root: &url::Url,
    relative: &str,
    bytes: Vec<u8>,
) -> Result<()> {
    use delta_kernel::object_store::path::Path;
    use delta_kernel::object_store::{ObjectStoreExt, PutPayload};

    let store = engine
        .get_object_store_for_url(root)
        .ok_or_else(|| NativeError::Invalid(format!("no object store is registered for {root}")))?;
    let url = root.join(relative)?;
    let path = Path::from_url_path(url.path())
        .map_err(|e| NativeError::Invalid(format!("invalid deletion vector path {url}: {e}")))?;
    // A plain put, as the kernel's own Parquet writes use: the name carries a
    // fresh UUID, and a conditional put is refused by S3 stores configured
    // without conditional-write support.
    runtime::block_on(async { store.put(&path, PutPayload::from(bytes)).await })?;
    Ok(())
}

/// Group `(path, row_index)` columns into per-file row sets.
pub fn deletions_from_batches(batches: &[RecordBatch]) -> Result<HashMap<String, RoaringTreemap>> {
    let mut out: HashMap<String, RoaringTreemap> = HashMap::new();
    for batch in batches {
        let paths = batch
            .column_by_name("path")
            .ok_or_else(|| NativeError::Invalid("deletions need a `path` column".to_string()))?;
        let rows = batch.column_by_name("row_index").ok_or_else(|| {
            NativeError::Invalid("deletions need a `row_index` column".to_string())
        })?;
        let paths = arrow::compute::cast(paths, &arrow::datatypes::DataType::Utf8)?;
        let paths = paths
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| NativeError::Invalid("deletion paths are not strings".to_string()))?;
        let rows = arrow::compute::cast(rows, &arrow::datatypes::DataType::Int64)?;
        let rows = rows
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .ok_or_else(|| NativeError::Invalid("row indexes are not integers".to_string()))?;
        for i in 0..batch.num_rows() {
            if paths.is_null(i) || rows.is_null(i) {
                return Err(NativeError::Invalid(
                    "a deletion has a null path or row index".to_string(),
                ));
            }
            let row = u64::try_from(rows.value(i)).map_err(|_| {
                NativeError::Invalid(format!("row index {} is negative", rows.value(i)))
            })?;
            out.entry(paths.value(i).to_string())
                .or_default()
                .insert(row);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {

    #[test]
    fn legacy_writer_versions_imply_their_features() {
        assert_eq!(super::legacy_writer_features(1), Vec::<&str>::new());
        assert_eq!(
            super::legacy_writer_features(4),
            [
                "appendOnly",
                "invariants",
                "checkConstraints",
                "changeDataFeed",
                "generatedColumns"
            ]
        );
        assert!(super::legacy_writer_features(6).contains(&"identityColumns"));
    }

    use super::*;

    #[test]
    fn widened_stats_keep_every_number_as_written() {
        let exact = r#"{"numRecords":3,"minValues":{"dec":-12345678901234567890.123456789012345678,"big":-9007199254740995},"maxValues":{"dec":12345678901234567890.123456789012345678,"f":1e300},"nullCount":{"dec":0},"tightBounds":true}"#;
        assert_eq!(
            wide_stats(exact).unwrap(),
            exact.replace(r#""tightBounds":true"#, r#""tightBounds":false"#)
        );
        assert_eq!(
            wide_stats(r#"{"numRecords":1, "maxValues":{"d":0.10000000000000000001}}"#).unwrap(),
            r#"{"numRecords":1,"maxValues":{"d":0.10000000000000000001},"tightBounds":false}"#
        );
        assert_eq!(wide_stats("{}").unwrap(), r#"{"tightBounds":false}"#);
        assert_eq!(wide_stats("[1]"), None);
        assert_eq!(wide_stats("not json"), None);
    }

    #[test]
    fn dv_file_names_follow_the_protocol_derivation() {
        let name = DvFileName::new("");
        assert_eq!(name.encoded.len(), 20);
        assert!(name.relative.starts_with("deletion_vector_"));
        assert!(name.relative.ends_with(".bin"));
        // The kernel decodes the log form back to the same relative path.
        let descriptor = DeletionVectorDescriptor::try_new(
            DeletionVectorStorageType::PersistedRelative,
            name.encoded.clone(),
            Some(1),
            10,
            1,
        )
        .unwrap();
        let root = url::Url::parse("file:///tmp/t/").unwrap();
        let resolved = descriptor.absolute_path(&root).unwrap().unwrap();
        assert_eq!(resolved, root.join(&name.relative).unwrap());

        let prefixed = DvFileName::new("ab");
        assert!(prefixed.encoded.starts_with("ab"));
        assert!(prefixed.relative.starts_with("ab/deletion_vector_"));
        let descriptor = DeletionVectorDescriptor::try_new(
            DeletionVectorStorageType::PersistedRelative,
            prefixed.encoded.clone(),
            Some(1),
            10,
            1,
        )
        .unwrap();
        let resolved = descriptor.absolute_path(&root).unwrap().unwrap();
        assert_eq!(resolved, root.join(&prefixed.relative).unwrap());
    }

    #[test]
    fn deletions_group_by_file_and_refuse_bad_rows() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![
            Field::new("path", DataType::Utf8, true),
            Field::new("row_index", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "a", "a"])),
                Arc::new(Int64Array::from(vec![3, 0, 1, 3])),
            ],
        )
        .unwrap();
        let grouped = deletions_from_batches(&[batch]).unwrap();
        assert_eq!(grouped["a"].iter().collect::<Vec<_>>(), [1, 3]);
        assert_eq!(grouped["b"].iter().collect::<Vec<_>>(), [0]);

        let negative = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a"])),
                Arc::new(Int64Array::from(vec![-1])),
            ],
        )
        .unwrap();
        assert!(deletions_from_batches(&[negative]).is_err());
    }
}
