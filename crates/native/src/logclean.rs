//! Expired log cleanup (`cleanup_metadata`) planned by the kernel.
//!
//! delta-rs cannot open a table with in-commit timestamps for writing, so it
//! refused to clean up such a table's log, and on the rest it goes by file
//! modification times alone and knows nothing of v2 checkpoints' sidecars.
//! This follows Spark's `MetadataCleanup`, with the retained history always
//! reconstructable:
//!
//! * the boundary is `delta.logRetentionDuration` (30 days by default) before
//!   now, and a version's time is its commit timestamp: the in-commit
//!   timestamp where the table has them, else the commit file's modification
//!   time, made monotonic (`history_manager`, as timestamp time travel reads
//!   it). So every version at or below the latest one at the boundary is
//!   expired, and none above it.
//! * the cleanup keeps the newest complete checkpoint at or below that
//!   version (one the kernel builds a snapshot from) and deletes only what
//!   lies below it: commit files, `.crc` files, checkpoints of every kind
//!   (classic, multi-part, v2) and compacted log files ending below it. Every
//!   version from that checkpoint on still reads; without one nothing goes.
//! * sidecar files no retained v2 checkpoint references are deleted once
//!   they are older than the boundary themselves (a checkpoint being written
//!   now has written its sidecars, not yet the file naming them).
//!
//! Only files directly in `_delta_log/` and in `_delta_log/_sidecars/` are
//! considered: `_staged_commits/` is the catalog's, and a catalog-managed
//! table is refused outright (its catalog owns the log), as is a table with
//! `checkpointProtection`, whose history below
//! `delta.requireCheckpointProtectionBeforeVersion` may only be truncated as
//! a whole. Everything deleted is under the table root.
//!
//! Files are deleted in version order, oldest first, so an interrupted
//! cleanup leaves a log that still starts somewhere and runs without a gap.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::DataType as ArrowType;
use delta_kernel::engine::arrow_data::EngineDataArrowExt;
use delta_kernel::history_manager::error::LogHistoryError;
use delta_kernel::history_manager::{latest_version_as_of, HistoryCommitType};
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::{DynObjectStore, ObjectMeta};
use delta_kernel::schema::{DataType, StructField, StructType};
use delta_kernel::snapshot::{Snapshot, SnapshotRef};
use delta_kernel::{Engine, FileMeta, Version};
use futures::StreamExt;
use url::Url;

use crate::confine;
use crate::error::{NativeError, Result};
use crate::runtime;
use crate::vacuum;

/// How many deletes are in flight at once; the next batch waits for these.
const BATCH: usize = 32;

/// The retention Delta applies when a table sets no `delta.logRetentionDuration`.
pub const DEFAULT_LOG_RETENTION_MS: u64 = 30 * 24 * 3600 * 1000;

/// What a file directly in `_delta_log/` is, by its name.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LogFile {
    Commit,
    Checksum,
    Checkpoint,
    /// Part `part` of a `parts`-part checkpoint.
    CheckpointPart {
        part: u32,
        parts: u32,
    },
    /// A v2 checkpoint (`<v>.checkpoint.<uuid>.json|parquet`).
    V2Checkpoint {
        json: bool,
    },
    /// A compacted log file covering versions up to `end`.
    Compacted {
        end: Version,
    },
}

fn digits(text: &str, len: usize) -> Option<u64> {
    (text.len() == len && text.bytes().all(|b| b.is_ascii_digit()))
        .then(|| text.parse().ok())
        .flatten()
}

fn is_uuid(text: &str) -> bool {
    text.len() == 36
        && text.char_indices().all(|(i, c)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                c == '-'
            } else {
                c.is_ascii_hexdigit()
            }
        })
}

/// The version and kind of a log file name, or None for anything else
/// (`_last_checkpoint`, a writer's temporary file, a stranger).
fn parse(name: &str) -> Option<(Version, LogFile)> {
    let version = digits(name.get(..20)?, 20)?;
    let rest = name.get(20..)?;
    let kind = match rest {
        ".json" => LogFile::Commit,
        ".crc" => LogFile::Checksum,
        ".checkpoint.parquet" => LogFile::Checkpoint,
        _ => {
            if let Some(tail) = rest.strip_prefix(".checkpoint.") {
                if let Some(uuid) = tail.strip_suffix(".json") {
                    is_uuid(uuid).then_some(LogFile::V2Checkpoint { json: true })?
                } else {
                    let stem = tail.strip_suffix(".parquet")?;
                    if is_uuid(stem) {
                        LogFile::V2Checkpoint { json: false }
                    } else {
                        let (part, parts) = stem.split_once('.')?;
                        let part = digits(part, 10)? as u32;
                        let parts = digits(parts, 10)? as u32;
                        if part == 0 || part > parts {
                            return None;
                        }
                        LogFile::CheckpointPart { part, parts }
                    }
                }
            } else {
                let end = rest.strip_prefix('.')?.strip_suffix(".compacted.json")?;
                let end = digits(end, 20)?;
                if end < version {
                    return None;
                }
                LogFile::Compacted { end }
            }
        }
    };
    Some((version, kind))
}

/// One listed log file.
#[derive(Debug, Clone)]
struct Listed {
    name: String,
    version: Version,
    kind: LogFile,
    meta: ObjectMeta,
}

/// What a cleanup deletes, oldest first, and the checkpoint it keeps.
#[derive(Debug, Default)]
pub struct Plan {
    /// The checkpoint every retained version is reconstructed from; None
    /// when there is none old enough, and nothing below any is deleted.
    pub kept_checkpoint: Option<Version>,
    /// Object keys relative to the table root, in the order to delete them.
    pub keys: Vec<String>,
}

fn path_error(e: delta_kernel::object_store::path::Error) -> NativeError {
    delta_kernel::object_store::Error::from(e).into()
}

/// `location`'s key under the root, in the store's spelling.
fn key_under(root_path: &Path, location: &Path) -> Option<String> {
    let parts: Vec<String> = location
        .prefix_match(root_path)?
        .map(|p| p.as_ref().to_string())
        .collect();
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// Why the log of this table must not be cleaned up here, if it must not.
fn refusal(snapshot: &SnapshotRef) -> Option<String> {
    if snapshot.table_configuration().is_catalog_managed() {
        return Some(
            "the table is catalog-managed: its catalog owns the log, and it refuses changes \
             to it from external engines"
                .into(),
        );
    }
    let protocol = serde_json::to_value(snapshot.table_configuration().protocol()).ok()?;
    let protected = ["readerFeatures", "writerFeatures"].iter().any(|key| {
        protocol
            .get(key)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|features| {
                features
                    .iter()
                    .any(|f| f.as_str() == Some("checkpointProtection"))
            })
    });
    protected.then(|| {
        "the table has checkpointProtection: its history before \
         delta.requireCheckpointProtectionBeforeVersion may only be truncated as a whole, by a \
         writer supporting every feature it ever had"
            .into()
    })
}

/// The sidecar files the v2 checkpoint at `file` references, as store paths.
fn sidecars_of(engine: &dyn Engine, log_root: &Url, file: &Listed) -> Result<Vec<Path>> {
    let url = log_root.join(&file.name)?;
    let sidecar_root = log_root.join("_sidecars/")?;
    let mut names: Vec<String> = Vec::new();
    if matches!(file.kind, LogFile::V2Checkpoint { json: true }) {
        for data in engine.storage_handler().read_files(vec![(url, None)])? {
            let data = data?;
            for line in data.split(|b| *b == b'\n') {
                if !line.windows(10).any(|w| w == b"\"sidecar\":") {
                    continue;
                }
                let action: serde_json::Value = serde_json::from_slice(line).map_err(|e| {
                    NativeError::Invalid(format!("bad action in {}: {e}", file.name))
                })?;
                if let Some(path) = action
                    .get("sidecar")
                    .and_then(|s| s.get("path"))
                    .and_then(serde_json::Value::as_str)
                {
                    names.push(path.to_string());
                }
            }
        }
    } else {
        let sidecar = StructType::try_new([StructField::nullable("path", DataType::STRING)])?;
        let schema = Arc::new(StructType::try_new([StructField::nullable(
            "sidecar",
            DataType::Struct(Box::new(sidecar)),
        )])?);
        let meta = FileMeta::new(
            url,
            file.meta.last_modified.timestamp_millis(),
            file.meta.size,
        );
        for batch in engine
            .parquet_handler()
            .read_parquet_files(&[meta], schema, None)?
        {
            let batch: RecordBatch = batch?.try_into_record_batch()?;
            let Some(sidecar) = batch
                .column_by_name("sidecar")
                .and_then(|c| c.as_struct_opt())
            else {
                continue;
            };
            let Some(path) = sidecar.column_by_name("path") else {
                continue;
            };
            let path = arrow::compute::cast(path, &ArrowType::Utf8)?;
            let path = path.as_string::<i32>();
            for i in 0..sidecar.len() {
                if !sidecar.is_null(i) && !path.is_null(i) {
                    names.push(path.value(i).to_string());
                }
            }
        }
    }
    names
        .into_iter()
        .map(|name| {
            // A bare file name, as writers store it, or a URI.
            let url = match Url::parse(&name) {
                Ok(url) => url,
                Err(url::ParseError::RelativeUrlWithoutBase) => sidecar_root.join(&name)?,
                Err(e) => return Err(e.into()),
            };
            Path::from_url_path(url.path()).map_err(path_error)
        })
        .collect()
}

/// Whether the checkpoint at `version` is whole: a classic one, every part
/// of a multi-part one, or a v2 one whose sidecars are all there.
fn complete(
    engine: &dyn Engine,
    log_root: &Url,
    files: &[&Listed],
    present: &HashSet<Path>,
) -> bool {
    if files.iter().any(|f| f.kind == LogFile::Checkpoint) {
        return true;
    }
    let mut parts: BTreeMap<u32, HashSet<u32>> = BTreeMap::new();
    for f in files {
        if let LogFile::CheckpointPart { part, parts: n } = f.kind {
            parts.entry(n).or_default().insert(part);
        }
    }
    if parts.iter().any(|(n, seen)| seen.len() == *n as usize) {
        return true;
    }
    files.iter().any(|f| {
        matches!(f.kind, LogFile::V2Checkpoint { .. })
            && sidecars_of(engine, log_root, f)
                .is_ok_and(|sidecars| sidecars.iter().all(|s| present.contains(s)))
    })
}

fn list(store: &DynObjectStore, prefix: &Path, recursive: bool) -> Result<Vec<ObjectMeta>> {
    runtime::block_on(async {
        if !recursive {
            return Ok(store.list_with_delimiter(Some(prefix)).await?.objects);
        }
        let mut listing = store.list(Some(prefix));
        let mut out = Vec::new();
        while let Some(meta) = listing.next().await {
            out.push(meta?);
        }
        Ok::<_, NativeError>(out)
    })
}

/// The latest of `commits` (ascending `(version, modification ms)`) whose
/// time, made monotonic as time travel makes it, is at or before `cutoff_ms`.
fn latest_at_or_before(commits: &[(Version, i64)], cutoff_ms: i64) -> Option<Version> {
    let mut found = None;
    let mut previous = i64::MIN;
    for &(version, raw) in commits {
        let time = raw.max(previous.saturating_add(1));
        if time > cutoff_ms {
            break;
        }
        found = Some(version);
        previous = time;
    }
    found
}

/// What cleaning up the log of `snapshot` (the latest) deletes, with the
/// retention boundary at `cutoff_ms` (epoch ms).
pub fn plan(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    store: Arc<DynObjectStore>,
    cutoff_ms: i64,
) -> Result<Plan> {
    if let Some(reason) = refusal(snapshot) {
        return Err(NativeError::Invalid(format!(
            "cannot clean up the log: {reason}"
        )));
    }
    let root = snapshot.table_root().clone();
    let root_path = Path::from_url_path(root.path()).map_err(path_error)?;
    let log_root = root.join("_delta_log/")?;
    let log_path = Path::from_url_path(log_root.path()).map_err(path_error)?;
    let sidecar_path =
        Path::from_url_path(log_root.join("_sidecars/")?.path()).map_err(path_error)?;

    let mut listed: Vec<Listed> = Vec::new();
    for meta in list(store.as_ref(), &log_path, false)? {
        let Some(name) = meta.location.filename().map(str::to_string) else {
            continue;
        };
        if let Some((version, kind)) = parse(&name) {
            listed.push(Listed {
                name,
                version,
                kind,
                meta,
            });
        }
    }
    listed.sort_by(|a, b| a.name.cmp(&b.name));
    let sidecars = list(store.as_ref(), &sidecar_path, true)?;
    let present: HashSet<Path> = sidecars.iter().map(|m| m.location.clone()).collect();

    // The latest version committed at or before the boundary.
    let ict = snapshot
        .table_properties()
        .enable_in_commit_timestamps
        .unwrap_or(false);
    let expired = if ict {
        match latest_version_as_of(snapshot, engine, cutoff_ms, HistoryCommitType::Published) {
            Ok(commit) => Some(commit.version.min(snapshot.version())),
            Err(delta_kernel::Error::LogHistory(e))
                if matches!(*e, LogHistoryError::TimestampOutOfRange { .. }) =>
            {
                None // every commit is newer than the boundary
            }
            Err(e) => return Err(e.into()),
        }
    } else {
        // Not the history manager: it answers any time after the latest
        // commit's raw modification time with the latest version, and so
        // expired every version after one whose file looked newer.
        let commits: Vec<(Version, i64)> = listed
            .iter()
            .filter(|f| f.kind == LogFile::Commit && f.version <= snapshot.version())
            .map(|f| (f.version, f.meta.last_modified.timestamp_millis()))
            .collect();
        latest_at_or_before(&commits, cutoff_ms)
    };

    let mut checkpoints: BTreeMap<Version, Vec<&Listed>> = BTreeMap::new();
    for f in &listed {
        if matches!(
            f.kind,
            LogFile::Checkpoint | LogFile::CheckpointPart { .. } | LogFile::V2Checkpoint { .. }
        ) {
            checkpoints.entry(f.version).or_default().push(f);
        }
    }
    let mut kept = None;
    if let Some(expired) = expired {
        for (&version, files) in checkpoints.range(..=expired).rev() {
            if !complete(engine, &log_root, files, &present) {
                continue;
            }
            // The kernel must build the table at that version from it.
            let from_it = Snapshot::builder_for(root.as_str())
                .at_version(version)
                .build(engine)
                .is_ok_and(|s| s.log_segment().checkpoint_version == Some(version));
            if from_it {
                kept = Some(version);
                break;
            }
        }
    }

    let mut keys = Vec::new();
    if let Some(kept) = kept {
        for f in &listed {
            let last = match f.kind {
                LogFile::Compacted { end } => end,
                _ => f.version,
            };
            if last < kept {
                if let Some(key) = key_under(&root_path, &f.meta.location) {
                    keys.push(key);
                }
            }
        }
    }

    // Sidecars no retained v2 checkpoint references, older than the boundary.
    if !sidecars.is_empty() {
        let mut referenced: HashSet<Path> = HashSet::new();
        for f in &listed {
            if matches!(f.kind, LogFile::V2Checkpoint { .. })
                && kept.is_none_or(|kept| f.version >= kept)
            {
                referenced.extend(sidecars_of(engine, &log_root, f)?);
            }
        }
        let mut doomed: Vec<&ObjectMeta> = sidecars
            .iter()
            .filter(|m| {
                m.last_modified.timestamp_millis() < cutoff_ms && !referenced.contains(&m.location)
            })
            .collect();
        doomed.sort_by(|a, b| a.location.cmp(&b.location));
        keys.extend(
            doomed
                .iter()
                .filter_map(|m| key_under(&root_path, &m.location)),
        );
    }
    // Never anything outside the log (a listing is of the log alone, but the
    // keys are handed to a delete that takes whatever it is given).
    keys.retain(|key| {
        key.starts_with("_delta_log/")
            && !key.starts_with("_delta_log/_staged_commits/")
            && root
                .join(key)
                .is_ok_and(|url| confine::is_under(&log_root, &url))
    });
    Ok(Plan {
        kept_checkpoint: kept,
        keys,
    })
}

/// Delete what `plan` listed, in its order, a batch at a time; stops at the
/// first batch with a failure so the log never gains a gap. Returns the keys
/// deleted and `(key, error)` for each failure or key not attempted.
pub fn delete(
    store: Arc<DynObjectStore>,
    root: &Url,
    keys: Vec<String>,
) -> Result<vacuum::Deletion> {
    let mut deleted = Vec::new();
    let mut failed = Vec::new();
    let mut batches = keys.chunks(BATCH);
    for batch in batches.by_ref() {
        let (ok, bad) = vacuum::delete(store.clone(), root, batch.to_vec())?;
        deleted.extend(ok);
        if !bad.is_empty() {
            failed.extend(bad);
            break;
        }
    }
    for batch in batches {
        failed.extend(batch.iter().map(|key| {
            (
                key.clone(),
                "not attempted: an older file failed first".into(),
            )
        }));
    }
    Ok((deleted, failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_file_names_parse_by_kind() {
        let v = |n: u64| format!("{n:020}");
        assert_eq!(parse(&format!("{}.json", v(3))), Some((3, LogFile::Commit)));
        assert_eq!(
            parse(&format!("{}.crc", v(3))),
            Some((3, LogFile::Checksum))
        );
        assert_eq!(
            parse(&format!("{}.checkpoint.parquet", v(10))),
            Some((10, LogFile::Checkpoint))
        );
        assert_eq!(
            parse(&format!(
                "{}.checkpoint.0000000002.0000000003.parquet",
                v(10)
            )),
            Some((10, LogFile::CheckpointPart { part: 2, parts: 3 }))
        );
        let uuid = "3a0d65cd-4056-49b8-937b-95f9e3ee90e5";
        assert_eq!(
            parse(&format!("{}.checkpoint.{uuid}.json", v(10))),
            Some((10, LogFile::V2Checkpoint { json: true }))
        );
        assert_eq!(
            parse(&format!("{}.checkpoint.{uuid}.parquet", v(10))),
            Some((10, LogFile::V2Checkpoint { json: false }))
        );
        assert_eq!(
            parse(&format!("{}.{}.compacted.json", v(4), v(9))),
            Some((4, LogFile::Compacted { end: 9 }))
        );
        for other in [
            "_last_checkpoint".to_string(),
            format!("{}.json.tmp", v(3)),
            format!(".{}.json.crc", v(3)),
            format!("{}.checkpoint.0000000004.0000000003.parquet", v(1)),
            format!("{}.{}.compacted.json", v(9), v(4)),
            format!("{}.checkpoint.not-a-uuid.json", v(1)),
            "0001.json".to_string(),
        ] {
            assert_eq!(parse(&other), None, "{other}");
        }
    }

    #[test]
    fn modification_times_are_made_monotonic() {
        let commits = [(0, 10), (1, 20), (2, 90), (3, 30), (4, 40)];
        assert_eq!(latest_at_or_before(&commits, 50), Some(1));
        assert_eq!(latest_at_or_before(&commits, 91), Some(3));
        assert_eq!(latest_at_or_before(&commits, 92), Some(4));
        assert_eq!(latest_at_or_before(&commits, 5), None);
        assert_eq!(latest_at_or_before(&[], 5), None);
    }

    #[test]
    fn keys_are_relative_to_the_root() {
        let root = Path::from("t");
        assert_eq!(
            key_under(&root, &Path::from("t/_delta_log/x.json")).as_deref(),
            Some("_delta_log/x.json")
        );
        assert_eq!(key_under(&root, &Path::from("u/_delta_log/x.json")), None);
    }
}
