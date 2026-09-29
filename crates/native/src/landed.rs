//! Whether a distributed write's files already landed, and removing them if not.
//!
//! A commit reported as failed can still have landed (a put or a catalog
//! ratification that timed out after it took effect), and a driver that
//! restarts may commit its saved fragments a second time. The check reads
//! only the commits made since the write was planned -- O(new commits), not a
//! listing of every live file -- and a catalog-managed table's ratified tail
//! is read from where the catalog says it is.

use std::collections::{HashMap, HashSet};

use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::Engine;
use percent_encoding::percent_decode_str;

use crate::commit::SharedEngine;
use crate::error::{NativeError, Result};

/// Commit files are read this many at a time.
const CHUNK: usize = 64;

fn decoded(path: &str) -> String {
    percent_decode_str(path).decode_utf8_lossy().into_owned()
}

/// `(version, n)` for each commit in `(after, snapshot.version]` that adds any
/// of `paths`, `n` being how many of them it adds.
///
/// Paths compare percent-decoded, as the log may escape what the writer did
/// not. A commit in the range that can no longer be read (cleaned up past a
/// checkpoint) is an error: the answer would be a guess.
pub fn commits_adding(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    after: u64,
    paths: &[String],
) -> Result<Vec<(u64, usize)>> {
    let end = snapshot.version();
    if after >= end || paths.is_empty() {
        return Ok(Vec::new());
    }
    let wanted: HashSet<String> = paths.iter().map(|p| decoded(p)).collect();
    // Past the checkpoint the segment names each commit file, the catalog's
    // staged ones included; before it they are the published log files.
    let listed: HashMap<u64, url::Url> = snapshot
        .log_segment()
        .listed
        .ascending_commit_files
        .iter()
        .map(|c| (c.version, c.location.location.clone()))
        .collect();
    let root = snapshot.table_root();
    let versions: Vec<u64> = ((after + 1)..=end).collect();
    let mut out = Vec::new();
    for chunk in versions.chunks(CHUNK) {
        let urls = chunk
            .iter()
            .map(|v| match listed.get(v) {
                Some(url) => Ok((url.clone(), None)),
                None => root
                    .join(&format!("_delta_log/{v:020}.json"))
                    .map(|url| (url, None))
                    .map_err(|e| NativeError::Invalid(format!("bad commit path: {e}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        let contents = engine.storage_handler().read_files(urls)?;
        for (version, data) in chunk.iter().zip(contents) {
            let data = data.map_err(|e| {
                NativeError::Invalid(format!(
                    "commit {version} cannot be read ({e}), so whether these files were \
                     already committed cannot be told"
                ))
            })?;
            let n = adds_among(&data, &wanted)?;
            if n > 0 {
                out.push((*version, n));
            }
        }
    }
    Ok(out)
}

/// How many `add` actions in the commit `data` name a path in `wanted`.
fn adds_among(data: &[u8], wanted: &HashSet<String>) -> Result<usize> {
    let mut n = 0;
    for line in data.split(|b| *b == b'\n') {
        // Cheap pre-filter: most lines of a large commit are adds, but the
        // commitInfo, removes and domain metadata need no parse.
        if line.len() < 8 || !line.starts_with(b"{\"add\"") && !contains(line, b"\"add\":") {
            continue;
        }
        let value: serde_json::Value = serde_json::from_slice(line)
            .map_err(|e| NativeError::Invalid(format!("a commit line is not JSON: {e}")))?;
        if let Some(path) = value
            .get("add")
            .and_then(|a| a.get("path"))
            .and_then(|p| p.as_str())
        {
            if wanted.contains(&decoded(path)) {
                n += 1;
            }
        }
    }
    Ok(n)
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Delete data files a distributed write produced and never committed.
///
/// Only plain relative paths under the table root are touched: an absolute
/// URL, a `..` segment or anything under `_delta_log` is refused outright,
/// since a fragment is caller input. Returns the paths that could not be
/// deleted, with why; a file already gone counts as deleted.
pub fn delete_uncommitted(
    snapshot: &SnapshotRef,
    engine: &SharedEngine,
    paths: &[String],
) -> Result<Vec<String>> {
    for path in paths {
        let plain = decoded(path);
        let bad = path.is_empty()
            || url::Url::parse(path).is_ok()
            || plain.starts_with('/')
            || plain.split('/').any(|s| s == ".." || s == ".")
            || plain.starts_with("_delta_log");
        if bad {
            return Err(NativeError::Invalid(format!(
                "{path:?} is not a data file path relative to the table root, so it is not \
                 one a distributed write produced; nothing was deleted"
            )));
        }
    }
    Ok(crate::commit::remove_written(
        engine,
        snapshot.table_root(),
        paths,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_adds_of_the_wanted_paths() {
        let wanted: HashSet<String> = ["a%20b.parquet", "c.parquet"]
            .iter()
            .map(|p| decoded(p))
            .collect();
        let commit = b"{\"commitInfo\":{\"operation\":\"WRITE\"}}\n\
{\"add\":{\"path\":\"a b.parquet\",\"size\":1}}\n\
{\"remove\":{\"path\":\"c.parquet\"}}\n\
{\"add\":{\"path\":\"d.parquet\",\"size\":1}}\n";
        assert_eq!(adds_among(commit, &wanted).unwrap(), 1);
    }
}
