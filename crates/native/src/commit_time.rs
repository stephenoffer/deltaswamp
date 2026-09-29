//! Commit timestamps as Delta assigns them, and timestamp -> version.
//!
//! A commit's timestamp is its in-commit timestamp once those are enabled, and
//! before that its commit file's modification time -- made monotonic, as
//! Spark's `DeltaHistoryManager` makes it: a commit whose file looks no newer
//! than the one before it is taken to be a millisecond after it. File times
//! go out of order whenever a log is copied, rewritten, or written from
//! machines whose clocks disagree.
//!
//! The kernel's history manager monotonizes too, but first compares the
//! timestamp with the *raw* modification time of the latest commit: any time
//! after it resolved to the latest version (and "first commit at or after" it
//! found none), so a log whose last file looked old read the latest version
//! as of every time since then. So every timestamp -> version lookup, and
//! every commit time this extension reports, goes through this module; the
//! kernel's search is used only inside the in-commit-timestamp region, where
//! times come from the commits themselves and are monotonic by protocol.

use delta_kernel::history_manager::{
    first_version_after, get_earliest_commit, latest_version_as_of, HistoryCommitType,
};
use delta_kernel::snapshot::Snapshot;
use delta_kernel::table_features::TableFeature;
use delta_kernel::{Engine, Version};

use crate::error::{NativeError, Result};

/// Which commit a timestamp names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// The latest commit at or before it (time travel, a feed's end).
    AtOrBefore,
    /// The first commit at or after it (a feed's start).
    AtOrAfter,
}

/// Where in-commit timestamps apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ict {
    /// Every commit's time is its file's modification time.
    Off,
    /// Enabled when the table was created: every commit has one.
    FromCreation,
    /// Enabled at `version`, whose in-commit timestamp is `timestamp`;
    /// commits before it are timed by their files.
    Since { version: Version, timestamp: i64 },
}

fn ict(snapshot: &Snapshot) -> Result<Ict> {
    if !snapshot
        .table_configuration()
        .is_feature_enabled(&TableFeature::InCommitTimestamp)
    {
        return Ok(Ict::Off);
    }
    let props = snapshot.table_properties();
    match (
        props.in_commit_timestamp_enablement_version,
        props.in_commit_timestamp_enablement_timestamp,
    ) {
        (Some(version), Some(timestamp)) => Ok(Ict::Since { version, timestamp }),
        (None, None) => Ok(Ict::FromCreation),
        _ => Err(NativeError::Invalid(
            "in-commit timestamps are enabled, but only one of \
             delta.inCommitTimestampEnablementVersion and \
             delta.inCommitTimestampEnablementTimestamp is set"
                .to_string(),
        )),
    }
}

/// `(version, modification ms)` of every published commit up to the
/// snapshot's version: the unbroken run of commit files that ends there.
fn listed_commits(snapshot: &Snapshot, engine: &dyn Engine) -> Result<Vec<(Version, i64)>> {
    let log_root = snapshot.table_root().join("_delta_log/")?;
    let start = log_root.join(&format!("{:020}", 0))?;
    let mut commits = Vec::new();
    for meta in engine.storage_handler().list_from(&start)? {
        let meta = meta?;
        let Some(name) = meta
            .location
            .path_segments()
            .and_then(|mut s| s.next_back())
        else {
            continue;
        };
        let Some(digits) = name.strip_suffix(".json") else {
            continue;
        };
        if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(version) = digits.parse::<Version>() else {
            continue;
        };
        if version > snapshot.version() {
            break;
        }
        commits.push((version, meta.last_modified));
    }
    commits.sort_unstable();
    // Only the run that is unbroken up to the latest: a gap is a commit log
    // cleanup removed, and the times before it no longer order anything.
    let mut start = commits.len();
    while start > 0 {
        let contiguous = start == commits.len() || commits[start - 1].0 + 1 == commits[start].0;
        if !contiguous {
            break;
        }
        start -= 1;
    }
    Ok(commits.split_off(start))
}

/// `commits` with each time made strictly greater than the one before.
fn monotonize(commits: &[(Version, i64)]) -> Vec<(Version, i64)> {
    let mut previous = i64::MIN;
    commits
        .iter()
        .map(|&(version, raw)| {
            let time = raw.max(previous.saturating_add(1));
            previous = time;
            (version, time)
        })
        .collect()
}

/// The commits timed by their files, their times made monotonic: every
/// published commit before in-commit timestamps were enabled (all of them
/// when they never were).
pub fn file_commit_times(snapshot: &Snapshot, engine: &dyn Engine) -> Result<Vec<(Version, i64)>> {
    let limit = match ict(snapshot)? {
        Ict::FromCreation => return Ok(Vec::new()),
        Ict::Since { version, .. } => Some(version),
        Ict::Off => None,
    };
    let mut listed = listed_commits(snapshot, engine)?;
    if let Some(limit) = limit {
        listed.retain(|&(v, _)| v < limit);
    }
    Ok(monotonize(&listed))
}

/// The snapshot's commit timestamp as Delta assigns it (ms): its in-commit
/// timestamp, or its file's modification time made monotonic.
pub fn commit_time(snapshot: &Snapshot, engine: &dyn Engine) -> Result<i64> {
    if ict(snapshot)? != Ict::Off {
        // In-commit timestamps are on at this version: the commit's own.
        return Ok(snapshot.get_timestamp(engine)?);
    }
    let times = file_commit_times(snapshot, engine)?;
    match times.last() {
        Some(&(version, time)) if version == snapshot.version() => Ok(time),
        _ => Ok(snapshot.get_timestamp(engine)?),
    }
}

/// Why a timestamp names no commit.
pub struct OutOfRange(String);

impl OutOfRange {
    fn new(timestamp: i64, bound: Bound, detail: &str) -> Self {
        let what = match bound {
            Bound::AtOrBefore => "at or before",
            Bound::AtOrAfter => "at or after",
        };
        Self(format!(
            "timestamp out of range: no commit {what} {timestamp} ms ({detail})"
        ))
    }
}

impl From<OutOfRange> for NativeError {
    fn from(e: OutOfRange) -> Self {
        NativeError::Invalid(e.0)
    }
}

/// The commit a timestamp names, and its (Delta-assigned) time.
///
/// `commit_type` is the kernel's: `Recreatable` only answers a version the
/// table can be rebuilt at (time travel), `Published` any commit in the log
/// (the change data feed). A time after the latest commit resolves to the
/// latest under `AtOrBefore`; refusing it is the caller's call (see
/// [`commit_time`]).
pub fn version_at(
    snapshot: &Snapshot,
    engine: &dyn Engine,
    timestamp: i64,
    bound: Bound,
    commit_type: HistoryCommitType,
) -> Result<(Version, i64)> {
    Ok(lookup(snapshot, engine, timestamp, bound, commit_type)??)
}

/// [`version_at`], with a timestamp naming no commit told apart from a
/// failure to read the log.
pub fn lookup(
    snapshot: &Snapshot,
    engine: &dyn Engine,
    timestamp: i64,
    bound: Bound,
    commit_type: HistoryCommitType,
) -> Result<std::result::Result<(Version, i64), OutOfRange>> {
    let ict = ict(snapshot)?;
    let in_ict_region = match ict {
        Ict::FromCreation => true,
        Ict::Since { timestamp: t, .. } => timestamp >= t,
        Ict::Off => false,
    };
    if in_ict_region {
        let found = match bound {
            Bound::AtOrBefore => latest_version_as_of(snapshot, engine, timestamp, commit_type),
            Bound::AtOrAfter => first_version_after(snapshot, engine, timestamp, commit_type),
        };
        return match found {
            Ok(commit) => Ok(Ok((commit.version, commit.timestamp))),
            Err(delta_kernel::Error::LogHistory(e)) => {
                Ok(Err(OutOfRange::new(timestamp, bound, &e.to_string())))
            }
            Err(e) => Err(e.into()),
        };
    }
    let times = file_commit_times(snapshot, engine)?;
    let earliest = match commit_type {
        HistoryCommitType::Recreatable => {
            let log_root = snapshot.table_root().join("_delta_log/")?;
            match get_earliest_commit(engine, &log_root, None, commit_type) {
                Ok(v) => v,
                Err(delta_kernel::Error::LogHistory(e)) => {
                    return Ok(Err(OutOfRange::new(timestamp, bound, &e.to_string())))
                }
                Err(e) => return Err(e.into()),
            }
        }
        HistoryCommitType::Published => times.first().map_or(0, |&(v, _)| v),
    };
    let found = match bound {
        Bound::AtOrBefore => times
            .iter()
            .take_while(|&&(_, t)| t <= timestamp)
            .last()
            .copied(),
        Bound::AtOrAfter => times.iter().find(|&&(_, t)| t >= timestamp).copied(),
    };
    let none = "the log holds no commit file timed by its modification time";
    Ok(match (found, bound, ict) {
        (Some((version, _)), Bound::AtOrBefore, _) if version < earliest => Err(OutOfRange::new(
            timestamp,
            bound,
            &format!(
                "it is before the earliest recreatable commit, version {earliest}, which the \
                 log's retained checkpoints can rebuild"
            ),
        )),
        (Some(found), _, _) => Ok(found),
        // Before in-commit timestamps and after every file-timed commit: the
        // commit that enabled them is the first after it.
        (None, Bound::AtOrAfter, Ict::Since { version, timestamp }) => Ok((version, timestamp)),
        (None, Bound::AtOrBefore, _) => Err(OutOfRange::new(
            timestamp,
            bound,
            &match times.first() {
                Some(&(v, t)) => {
                    format!("it is before the earliest commit in the log, version {v} at {t} ms")
                }
                None => none.into(),
            },
        )),
        (None, Bound::AtOrAfter, _) => Err(OutOfRange::new(
            timestamp,
            bound,
            &match times.last() {
                Some(&(v, t)) => format!("it is after the latest commit, version {v} at {t} ms"),
                None => none.into(),
            },
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_are_made_strictly_increasing() {
        let raw = [(0, 10), (1, 30), (2, 15), (3, 20), (4, 40)];
        assert_eq!(
            monotonize(&raw),
            vec![(0, 10), (1, 30), (2, 31), (3, 32), (4, 40)]
        );
        assert_eq!(monotonize(&[(0, 5), (1, 5)]), vec![(0, 5), (1, 6)]);
        assert!(monotonize(&[]).is_empty());
    }
}
