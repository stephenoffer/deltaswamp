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

/// As [`write_best_effort`]; `always` skips the cost check (an explicit call).
pub fn write(snapshot: &SnapshotRef, engine: &dyn Engine, always: bool) -> bool {
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if snapshot.table_configuration().is_catalog_managed() {
            return false;
        }
        if always || worth_writing(snapshot) {
            match snapshot.write_checksum(engine) {
                Ok((ChecksumWriteResult::Written, _)) => return true,
                Ok((ChecksumWriteResult::AlreadyExists, _)) => return false,
                Err(_) => {}
            }
        }
        carry_forward(snapshot, engine).unwrap_or(false)
    }));
    attempt.unwrap_or(false)
}

/// The previous version's checksum, carried over a commit that changed no file.
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
        .any(|a| a.contains_key("add") || a.contains_key("remove") || a.contains_key("cdc"));
    if files_changed {
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
    use super::upsert;
    use serde_json::json;

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
