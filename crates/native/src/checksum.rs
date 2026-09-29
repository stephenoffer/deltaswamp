//! Version checksum files (`_delta_log/<version>.crc`) after a commit.
//!
//! Spark and Databricks write one after every commit, and read it to load a
//! snapshot faster and to check the state they reconstructed (numFiles,
//! tableSizeBytes, the protocol and metadata). A table this library wrote had
//! none past the one kernel writes at CREATE, so the chain Databricks keeps
//! broke at the first append (delta-rs#4190, delta-kernel-rs#1781).
//!
//! The kernel builds the checksum (`Snapshot::write_checksum`); what is
//! decided here is when it is cheap enough to be worth it. A post-commit
//! snapshot whose read had a checksum at its version carries the next one in
//! memory -- no I/O but the put. A checksum a few commits back is advanced
//! over the tail. Beyond that the kernel would replay a checkpoint or the
//! whole log, which a write must not pay for, so the chain is not started on
//! a long history that never had one.
//!
//! Best effort throughout: a checksum is an optimization, and a commit that
//! worked is never failed over one. Nor is a catalog-managed table's: its
//! commits are ratified by the catalog, and the catalog's own writer keeps
//! its checksums.

use delta_kernel::snapshot::{ChecksumWriteResult, SnapshotRef};
use delta_kernel::Engine;

/// The most commits a stale checksum (or a log with no checkpoint) is
/// advanced over to write the next one.
pub const MAX_TAIL: u64 = 100;

/// Whether writing `snapshot`'s checksum costs no more than a short tail replay.
pub fn worth_writing(snapshot: &SnapshotRef) -> bool {
    if snapshot.table_configuration().is_catalog_managed() {
        return false;
    }
    if snapshot.get_file_stats_if_present().is_some() {
        return true; // computed in memory from the previous version's
    }
    let segment = snapshot.log_segment();
    let end = snapshot.version();
    let floor = end.saturating_sub(MAX_TAIL);
    if let Some(crc) = segment.listed.latest_crc_file.as_ref() {
        if crc.version == end {
            return false; // already there
        }
        // The kernel advances a checksum from at or above the checkpoint only.
        if crc.version >= floor && segment.checkpoint_version.is_none_or(|c| crc.version >= c) {
            return true;
        }
    }
    segment.checkpoint_version.is_none() && end <= MAX_TAIL
}

/// Write `snapshot`'s checksum when [`worth_writing`]; true if one was written.
///
/// Never fails: every error, and a panic inside the kernel, is swallowed.
pub fn write_best_effort(snapshot: &SnapshotRef, engine: &dyn Engine) -> bool {
    write(snapshot, engine, false)
}

/// Whether the kernel resolves `snapshot`'s checksum cheaply from a restated
/// copy of it (see [`write`]): by replaying a short log that has no checkpoint.
///
/// A restated snapshot carries no checksum in memory, so neither the one a
/// post-commit snapshot holds nor a stale one a few commits back can root it:
/// past the first checkpoint the kernel would read the checkpoint on every
/// commit. A commit is carried from the checksum just before it instead
/// ([`carry_forward`]), and the checksum at each checkpoint is written with
/// it ([`write_at_checkpoint`]), which starts the chain again after a commit
/// neither could count.
pub fn worth_replaying(snapshot: &SnapshotRef) -> bool {
    if snapshot.table_configuration().is_catalog_managed() {
        return false;
    }
    let segment = snapshot.log_segment();
    let end = snapshot.version();
    if segment
        .listed
        .latest_crc_file
        .as_ref()
        .is_some_and(|crc| crc.version == end)
    {
        return false; // already there
    }
    segment.checkpoint_version.is_none() && end <= MAX_TAIL
}

/// As [`write_best_effort`]; `always` skips the cost check (an explicit call).
///
/// A table the kernel refuses to write for a feature binding only the values
/// of rows written (CHECK constraints, generated or identity columns,
/// invariants) has its checksum resolved from a restated snapshot
/// (`restate::log_writing_snapshot`): the kernel builds a checksum's protocol
/// and metadata from the log, so the file holds the table's own, and writes
/// no row, so nothing is left unchecked.
pub fn write(snapshot: &SnapshotRef, engine: &dyn Engine, always: bool) -> bool {
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if snapshot.table_configuration().is_catalog_managed() {
            return false;
        }
        let target =
            crate::restate::log_writing_snapshot(snapshot).unwrap_or_else(|_| snapshot.clone());
        let kernel = |target: &SnapshotRef| match target.write_checksum(engine) {
            Ok((ChecksumWriteResult::Written, _)) => Some(true),
            Ok((ChecksumWriteResult::AlreadyExists, _)) => Some(false),
            Err(_) => None,
        };
        if std::sync::Arc::ptr_eq(&target, snapshot) {
            if always || worth_writing(snapshot) {
                if let Some(written) = kernel(snapshot) {
                    return written;
                }
            }
            if snapshot.version() == 0 {
                return first_commit(snapshot, engine).unwrap_or(false);
            }
            return carry_forward(snapshot, engine).unwrap_or(false);
        }
        // Restated: the kernel would replay what the previous checksum
        // already holds (see `worth_replaying`), so that is carried first.
        if snapshot.version() == 0 {
            if first_commit(snapshot, engine).unwrap_or(false) {
                return true;
            }
        } else if carry_forward(snapshot, engine).unwrap_or(false) {
            return true;
        }
        if always || worth_replaying(snapshot) {
            return kernel(&target).unwrap_or(false);
        }
        false
    }));
    attempt.unwrap_or(false)
}

/// Write the checksum of `checkpointed`, a snapshot whose log segment ends
/// in the checkpoint just written at its version; true if one was written.
///
/// The kernel builds it from that checkpoint and the commit's in-commit
/// timestamp. Called after a checkpoint of a restated table, whose commits
/// past its first checkpoint get no checksum of their own (see
/// [`worth_replaying`]); never fails, as [`write_best_effort`].
pub fn write_at_checkpoint(checkpointed: &SnapshotRef, engine: &dyn Engine) -> bool {
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if checkpointed.table_configuration().is_catalog_managed()
            || checkpointed.log_segment().checkpoint_version != Some(checkpointed.version())
        {
            return false;
        }
        matches!(
            checkpointed.write_checksum(engine),
            Ok((ChecksumWriteResult::Written, _))
        )
    }));
    attempt.unwrap_or(false)
}

/// The previous version's checksum, carried over one commit.
///
/// The kernel counts file stats only across operations it knows are
/// incremental (WRITE, MERGE, DELETE, ...), and gives up on the rest: a SET
/// TBLPROPERTIES, ADD COLUMNS or delta-rs's VACUUM START/END ended the chain,
/// and every checksum after it. A commit with no add, remove or cdc action
/// leaves every file statistic as it was, so the next checksum is the last
/// one with this commit's metadata, protocol, domain metadata, transactions
/// and in-commit timestamp applied -- as the Delta spec defines each field.
fn carry_forward(snapshot: &SnapshotRef, engine: &dyn Engine) -> crate::Result<bool> {
    use serde_json::Value;

    let version = snapshot.version();
    if version == 0 {
        return Ok(false);
    }
    let log = snapshot.table_root().join("_delta_log/")?;
    let previous = log.join(&format!("{:020}.crc", version - 1))?;
    let commit = log.join(&format!("{version:020}.json"))?;
    let target = log.join(&format!("{version:020}.crc"))?;
    let storage = engine.storage_handler();
    let mut texts = storage.read_files(vec![(previous, None), (commit, None)])?;
    let (Some(previous), Some(commit)) = (texts.next(), texts.next()) else {
        return Ok(false);
    };
    let (Ok(previous), Ok(commit)) = (previous, commit) else {
        return Ok(false); // no checksum to carry (or no published commit)
    };
    let Ok(Value::Object(mut crc)) = serde_json::from_slice::<Value>(&previous) else {
        return Ok(false);
    };
    let text = String::from_utf8_lossy(&commit);
    let mut actions = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(Value::Object(action)) = serde_json::from_str::<Value>(line) else {
            return Ok(false);
        };
        actions.push(action);
    }
    let files_changed = actions
        .iter()
        .any(|a| a.contains_key("add") || a.contains_key("remove"));
    if files_changed && !count_files(&mut crc, &actions) {
        return Ok(false);
    }
    // A transaction id names one commit; the carried checksum is another's.
    crc.remove("txnId");
    let mut ict = None;
    for action in &actions {
        if let Some(metadata) = action.get("metaData") {
            crc.insert("metadata".into(), metadata.clone());
        }
        if let Some(protocol) = action.get("protocol") {
            crc.insert("protocol".into(), protocol.clone());
        }
        if let Some(info) = action.get("commitInfo") {
            ict = info.get("inCommitTimestamp").cloned();
        }
        if let Some(domain) = action.get("domainMetadata") {
            upsert(&mut crc, "domainMetadata", "domain", domain, |d| {
                d.get("removed").and_then(Value::as_bool) == Some(true)
            });
        }
        if let Some(txn) = action.get("txn") {
            upsert(&mut crc, "setTransactions", "appId", txn, |_| false);
        }
    }
    let ict_enabled = crc
        .get("metadata")
        .and_then(|m| m.get("configuration"))
        .and_then(|c| c.get("delta.enableInCommitTimestamps"))
        .and_then(Value::as_str)
        == Some("true");
    match ict {
        Some(ts) => {
            crc.insert("inCommitTimestampOpt".into(), ts);
        }
        None if ict_enabled => return Ok(false),
        None => {
            crc.remove("inCommitTimestampOpt");
        }
    }
    let body = serde_json::to_vec(&Value::Object(crc))
        .map_err(|e| crate::NativeError::Invalid(format!("could not serialize a checksum: {e}")))?;
    Ok(storage.put(&target, body.into(), false).is_ok())
}

/// The checksum of version 0, counted from its own actions.
///
/// The kernel counts only operations it knows (CREATE TABLE, WRITE, ...), so
/// a table whose first commit is a CLONE -- every live file added at once --
/// had none. Version 0 holds the whole state, so the counts are its adds.
fn first_commit(snapshot: &SnapshotRef, engine: &dyn Engine) -> crate::Result<bool> {
    use serde_json::{json, Map, Value};

    let log = snapshot.table_root().join("_delta_log/")?;
    let commit = log.join(&format!("{:020}.json", 0))?;
    let target = log.join(&format!("{:020}.crc", 0))?;
    let storage = engine.storage_handler();
    let Some(Ok(bytes)) = storage.read_files(vec![(commit, None)])?.next() else {
        return Ok(false);
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut crc = Map::new();
    let (mut files, mut size) = (0u64, 0u64);
    let (mut domains, mut txns) = (Vec::new(), Vec::new());
    let mut ict = None;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(Value::Object(action)) = serde_json::from_str::<Value>(line) else {
            return Ok(false);
        };
        if let Some(add) = action.get("add") {
            let Some(bytes) = add.get("size").and_then(Value::as_u64) else {
                return Ok(false); // a size the spec requires is missing
            };
            files += 1;
            size += bytes;
        } else if action.contains_key("remove") || action.contains_key("cdc") {
            return Ok(false);
        } else if let Some(metadata) = action.get("metaData") {
            crc.insert("metadata".into(), metadata.clone());
        } else if let Some(protocol) = action.get("protocol") {
            crc.insert("protocol".into(), protocol.clone());
        } else if let Some(domain) = action.get("domainMetadata") {
            if domain.get("removed").and_then(Value::as_bool) != Some(true) {
                domains.push(domain.clone());
            }
        } else if let Some(txn) = action.get("txn") {
            txns.push(txn.clone());
        } else if let Some(info) = action.get("commitInfo") {
            ict = info.get("inCommitTimestamp").cloned();
        }
    }
    if !crc.contains_key("metadata") || !crc.contains_key("protocol") {
        return Ok(false);
    }
    let ict_enabled = crc["metadata"]
        .get("configuration")
        .and_then(|c| c.get("delta.enableInCommitTimestamps"))
        .and_then(Value::as_str)
        == Some("true");
    match ict {
        Some(ts) => {
            crc.insert("inCommitTimestampOpt".into(), ts);
        }
        None if ict_enabled => return Ok(false),
        None => {}
    }
    crc.insert("tableSizeBytes".into(), json!(size));
    crc.insert("numFiles".into(), json!(files));
    crc.insert("numMetadata".into(), json!(1));
    crc.insert("numProtocol".into(), json!(1));
    crc.insert("setTransactions".into(), Value::Array(txns));
    crc.insert("domainMetadata".into(), Value::Array(domains));
    let body = serde_json::to_vec(&Value::Object(crc))
        .map_err(|e| crate::NativeError::Invalid(format!("could not serialize a checksum: {e}")))?;
    Ok(storage.put(&target, body.into(), false).is_ok())
}

/// The operations whose adds and removes net to the change in the table's
/// files: delta-kernel's own list (`crc::file_stats::INCREMENTAL_SAFE_OPS`,
/// crate-private). Any other -- ANALYZE STATS re-adding live files with new
/// statistics, a RESTORE, an unknown engine's -- is not counted.
const INCREMENTAL_SAFE_OPS: &[&str] = &[
    "WRITE",
    "STREAMING UPDATE",
    "MERGE",
    "UPDATE",
    "DELETE",
    "OPTIMIZE",
    "CREATE TABLE",
    "REPLACE TABLE",
    "CREATE TABLE AS SELECT",
    "REPLACE TABLE AS SELECT",
    "CREATE OR REPLACE TABLE AS SELECT",
];

/// Advance `crc`'s file statistics over one commit's `actions`, as the
/// kernel's replay does (`LogSegment::build_crc_from_base`): every add counts
/// a file of its size in, every remove one out, in `numFiles`,
/// `tableSizeBytes` and the bins of `fileSizeHistogram`.
///
/// False -- nothing is written -- where the kernel would give up too: an
/// operation not incremental-safe, an add or remove without its size, or a
/// count that would go negative (a checksum or a commit that is not what it
/// says). The deletion-vector counts are dropped where a file with a vector
/// comes or goes (they are optional; nothing here counts them), and so is a
/// file list (`allFiles`, optional too).
fn count_files(
    crc: &mut serde_json::Map<String, serde_json::Value>,
    actions: &[serde_json::Map<String, serde_json::Value>],
) -> bool {
    use serde_json::{json, Value};

    let operation = actions
        .iter()
        .find_map(|a| a.get("commitInfo"))
        .and_then(|info| info.get("operation"))
        .and_then(Value::as_str);
    if !operation.is_some_and(|op| INCREMENTAL_SAFE_OPS.contains(&op)) {
        return false;
    }
    let (Some(mut files), Some(mut bytes)) = (
        crc.get("numFiles").and_then(Value::as_i64),
        crc.get("tableSizeBytes").and_then(Value::as_i64),
    ) else {
        return false;
    };
    let mut histogram = match crc.get("fileSizeHistogram") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let list = |key: &str| -> Option<Vec<i64>> {
                value
                    .get(key)?
                    .as_array()?
                    .iter()
                    .map(Value::as_i64)
                    .collect()
            };
            match (
                list("sortedBinBoundaries"),
                list("fileCounts"),
                list("totalBytes"),
            ) {
                (Some(bounds), Some(counts), Some(sizes))
                    if !bounds.is_empty()
                        && bounds[0] == 0
                        && counts.len() == bounds.len()
                        && sizes.len() == bounds.len() =>
                {
                    Some((bounds, counts, sizes))
                }
                _ => return false, // a histogram we cannot keep true
            }
        }
    };
    let mut vectors_changed = false;
    for action in actions {
        let (file, sign) = match (action.get("add"), action.get("remove")) {
            (Some(add), _) => (add, 1),
            (None, Some(remove)) => (remove, -1),
            _ => continue,
        };
        let Some(size) = file.get("size").and_then(Value::as_i64).filter(|s| *s >= 0) else {
            return false;
        };
        if file.get("deletionVector").is_some_and(|dv| !dv.is_null()) {
            vectors_changed = true;
        }
        files += sign;
        bytes += sign * size;
        if let Some((bounds, counts, sizes)) = histogram.as_mut() {
            let bin = match bounds.binary_search(&size) {
                Ok(i) => i,
                Err(i) => i - 1, // bounds[0] is 0 and size >= 0, so i >= 1
            };
            counts[bin] += sign;
            sizes[bin] += sign * size;
        }
    }
    if files < 0 || bytes < 0 {
        return false;
    }
    if let Some((bounds, counts, sizes)) = histogram {
        if counts.iter().chain(&sizes).any(|n| *n < 0) {
            return false;
        }
        crc.insert(
            "fileSizeHistogram".into(),
            json!({"sortedBinBoundaries": bounds, "fileCounts": counts, "totalBytes": sizes}),
        );
    }
    crc.insert("numFiles".into(), json!(files));
    crc.insert("tableSizeBytes".into(), json!(bytes));
    crc.remove("allFiles");
    if vectors_changed {
        for key in [
            "numDeletedRecordsOpt",
            "numDeletionVectorsOpt",
            "deletedRecordCountsHistogramOpt",
        ] {
            crc.remove(key);
        }
    }
    true
}

/// Replace (or drop, when `removed`) the entry of `list` whose `key` matches `entry`'s.
fn upsert(
    crc: &mut serde_json::Map<String, serde_json::Value>,
    list: &str,
    key: &str,
    entry: &serde_json::Value,
    removed: impl Fn(&serde_json::Value) -> bool,
) {
    use serde_json::Value;

    let Some(Value::Array(items)) = crc.get_mut(list) else {
        // Absent means "not tracked" (optional in the spec): leave it so.
        return;
    };
    let id = entry.get(key).cloned();
    items.retain(|item| item.get(key).cloned() != id);
    if !removed(entry) {
        items.push(entry.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::{count_files, upsert};
    use serde_json::json;

    fn actions(values: &[serde_json::Value]) -> Vec<serde_json::Map<String, serde_json::Value>> {
        values
            .iter()
            .map(|v| v.as_object().unwrap().clone())
            .collect()
    }

    #[test]
    fn a_commit_s_files_are_counted_into_the_previous_checksum() {
        let mut crc = json!({
            "numFiles": 2, "tableSizeBytes": 300, "allFiles": [],
            "numDeletionVectorsOpt": 0, "numDeletedRecordsOpt": 0,
            "fileSizeHistogram": {
                "sortedBinBoundaries": [0, 150], "fileCounts": [1, 1], "totalBytes": [100, 200]
            },
        })
        .as_object()
        .unwrap()
        .clone();
        let commit = actions(&[
            json!({"commitInfo": {"operation": "DELETE"}}),
            json!({"remove": {"path": "a", "size": 100}}),
            json!({"add": {"path": "c", "size": 150}}),
            json!({"add": {"path": "d", "size": 10}}),
        ]);
        assert!(count_files(&mut crc, &commit));
        assert_eq!(crc["numFiles"], json!(3));
        assert_eq!(crc["tableSizeBytes"], json!(360));
        assert_eq!(
            crc["fileSizeHistogram"],
            json!({"sortedBinBoundaries": [0, 150], "fileCounts": [1, 2], "totalBytes": [10, 350]})
        );
        assert!(!crc.contains_key("allFiles"));
        // No file with a deletion vector came or went: its counts still hold.
        assert_eq!(crc["numDeletionVectorsOpt"], json!(0));

        let vector = actions(&[
            json!({"commitInfo": {"operation": "DELETE"}}),
            json!({"remove": {"path": "c", "size": 150}}),
            json!({"add": {"path": "c", "size": 150, "deletionVector": {"cardinality": 1}}}),
        ]);
        assert!(count_files(&mut crc, &vector));
        assert_eq!(crc["numFiles"], json!(3));
        assert!(!crc.contains_key("numDeletionVectorsOpt"));
        assert!(!crc.contains_key("numDeletedRecordsOpt"));
    }

    #[test]
    fn a_commit_the_kernel_would_not_count_is_not_counted() {
        let crc = json!({"numFiles": 1, "tableSizeBytes": 100})
            .as_object()
            .unwrap()
            .clone();
        let add = json!({"add": {"path": "b", "size": 5}});
        for commit in [
            // Not incremental-safe: ANALYZE re-adds live files.
            vec![
                json!({"commitInfo": {"operation": "ANALYZE STATS"}}),
                add.clone(),
            ],
            vec![add.clone()], // no commitInfo at all
            vec![
                json!({"commitInfo": {"operation": "WRITE"}}),
                json!({"remove": {"path": "a"}}), // no size
            ],
            vec![
                json!({"commitInfo": {"operation": "DELETE"}}),
                json!({"remove": {"path": "a", "size": 100}}),
                json!({"remove": {"path": "z", "size": 100}}), // more than the table holds
            ],
        ] {
            let mut copy = crc.clone();
            assert!(!count_files(&mut copy, &actions(&commit)), "{commit:?}");
        }
        let mut broken = json!({
            "numFiles": 1, "tableSizeBytes": 100,
            "fileSizeHistogram": {"sortedBinBoundaries": [0], "fileCounts": [], "totalBytes": []},
        })
        .as_object()
        .unwrap()
        .clone();
        let commit = actions(&[json!({"commitInfo": {"operation": "WRITE"}}), add]);
        assert!(!count_files(&mut broken, &commit));
    }

    #[test]
    fn a_domain_is_replaced_or_dropped_and_an_untracked_list_left_alone() {
        let mut crc = json!({
            "domainMetadata": [{"domain": "a", "configuration": "1", "removed": false}],
        })
        .as_object()
        .unwrap()
        .clone();
        let removed = |d: &serde_json::Value| d["removed"] == json!(true);
        upsert(
            &mut crc,
            "domainMetadata",
            "domain",
            &json!({"domain": "a", "configuration": "2", "removed": false}),
            removed,
        );
        assert_eq!(
            crc["domainMetadata"],
            json!([{"domain": "a", "configuration": "2", "removed": false}])
        );
        upsert(
            &mut crc,
            "domainMetadata",
            "domain",
            &json!({"domain": "a", "configuration": "", "removed": true}),
            removed,
        );
        assert_eq!(crc["domainMetadata"], json!([]));
        upsert(
            &mut crc,
            "setTransactions",
            "appId",
            &json!({"appId": "x", "version": 1}),
            |_| false,
        );
        assert!(!crc.contains_key("setTransactions"));
    }
}
