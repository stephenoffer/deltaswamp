//! A table that does not exist yet, as a distributed write sees it.
//!
//! Creating a table by name for a distributed write must not make the table
//! visible before the job's data can land: an empty table left behind by a
//! failed job then refuses the retry as "already exists", and a catalog would
//! list it. So the driver writes the table's version 0 -- protocol and
//! metaData -- as a *template*, a commit file under
//! `<root>/_deltaswamp_pending/<plan>/_delta_log/`, where no reader looks
//! (it is not the table's `_delta_log/`). Workers resolve the table from that
//! template, supplied to the kernel as a one-entry log tail, and write their
//! data files under the real root with the real layout. The commit publishes
//! version 0 and then commits the files.

use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::ObjectStoreExt;
use delta_kernel::{FileMeta, LogPath};
use pyo3::prelude::*;
use std::collections::HashMap;
use url::Url;

use crate::error::{NativeError, Result};
use crate::runtime;
use crate::snapshot::PySnapshot;
use crate::store;

/// Where templates live, under the table root. Not `_delta_log/`: a listing
/// of the log (recursive on object stores) must never find one.
pub const PENDING_DIR: &str = "_deltaswamp_pending/";

/// The template commit at `location` as a log tail entry for `table_root`.
///
/// Only a version-0 commit file under the table's own pending directory is
/// accepted, so a template can never point the kernel at another table's log.
pub fn template_log_path(
    table_root: &Url,
    location: &str,
    last_modified: i64,
    size: u64,
) -> Result<LogPath> {
    let url = Url::parse(location)
        .map_err(|e| NativeError::Invalid(format!("template {location:?} is not a URL: {e}")))?;
    let prefix = format!("{}{PENDING_DIR}", table_root.as_str());
    let inside = url.as_str().strip_prefix(prefix.as_str()).unwrap_or("");
    let segments: Vec<&str> = inside.split('/').collect();
    let well_formed = segments.len() == 3
        && !segments[0].is_empty()
        && segments[0] != "."
        && segments[0] != ".."
        && segments[1] == "_delta_log"
        && segments[2] == format!("{:020}.json", 0)
        && url.query().is_none()
        && url.fragment().is_none();
    if !well_formed {
        return Err(NativeError::Invalid(format!(
            "template {location:?} is not a version-0 commit under {prefix}<plan>/_delta_log/"
        )));
    }
    Ok(LogPath::try_new(FileMeta {
        location: url,
        last_modified,
        size,
    })?)
}

/// Delete a table's version 0 if it is still the only version and is `metadata_id`'s.
///
/// Undoes a create whose data commit failed for certain. Returns whether it
/// deleted anything: nothing when version 1 exists (someone committed on
/// top, so the table is theirs too now), or when version 0 belongs to another
/// table (another writer created it).
pub fn rollback_create(
    table_root: &Url,
    options: &HashMap<String, String>,
    metadata_id: &str,
) -> Result<bool> {
    let store = store::build_store(table_root, options)?;
    let root =
        Path::from_url_path(table_root.path()).map_err(delta_kernel::object_store::Error::from)?;
    let log = root.clone().join("_delta_log");
    let first = log.clone().join(format!("{:020}.json", 0));
    let second = log.clone().join(format!("{:020}.json", 1));
    runtime::block_on(async {
        match store.head(&second).await {
            Ok(_) => return Ok(false),
            Err(delta_kernel::object_store::Error::NotFound { .. }) => {}
            Err(e) => return Err(NativeError::from(e)),
        }
        let body = match store.get(&first).await {
            Ok(result) => result.bytes().await?,
            Err(delta_kernel::object_store::Error::NotFound { .. }) => return Ok(false),
            Err(e) => return Err(NativeError::from(e)),
        };
        if !is_metadata_of(&body, metadata_id) {
            return Ok(false);
        }
        store.delete(&first).await?;
        // A checksum written after version 0, if any; best effort.
        let _ = store
            .delete(&log.clone().join(format!("{:020}.crc", 0)))
            .await;
        Ok(true)
    })
}

/// Whether a commit's `metaData` action has id `metadata_id`.
fn is_metadata_of(body: &[u8], metadata_id: &str) -> bool {
    String::from_utf8_lossy(body).lines().any(|line| {
        serde_json::from_str::<serde_json::Value>(line)
            .ok()
            .and_then(|action| {
                action
                    .get("metaData")
                    .and_then(|m| m.get("id"))
                    .and_then(|id| id.as_str().map(|s| s == metadata_id))
            })
            .unwrap_or(false)
    })
}

/// The directory a plan's template lives in: `<root>/_deltaswamp_pending/<plan>/`.
fn template_root(table_root: &Url, plan_id: &str) -> Result<Url> {
    let bare = !plan_id.is_empty()
        && plan_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !bare {
        return Err(NativeError::Invalid(format!(
            "plan id {plan_id:?} must be letters, digits, '-' or '_'"
        )));
    }
    table_root
        .join(&format!("{PENDING_DIR}{plan_id}/"))
        .map_err(|e| {
            NativeError::Invalid(format!("cannot place a template under {table_root}: {e}"))
        })
}

/// Write `actions` as plan `plan_id`'s template version 0, put-if-absent.
///
/// Returns `(url, last_modified_millis, size)`: what `Snapshot.resolve`
/// takes as `template=`, with the URL spelled as the kernel spells the root.
pub fn write_template(
    table_root: &Url,
    options: &HashMap<String, String>,
    plan_id: &str,
    actions: &[String],
) -> Result<(String, i64, u64)> {
    let root = template_root(table_root, plan_id)?;
    crate::commit::commit_raw(&root, options, 0, actions)?;
    let location = root
        .join(&format!("_delta_log/{:020}.json", 0))
        .map_err(|e| NativeError::Invalid(e.to_string()))?;
    let store = store::build_store(table_root, options)?;
    let path =
        Path::from_url_path(location.path()).map_err(delta_kernel::object_store::Error::from)?;
    let meta = runtime::block_on(async { store.head(&path).await })?;
    Ok((
        location.to_string(),
        meta.last_modified.timestamp_millis(),
        meta.size,
    ))
}

/// Delete plan `plan_id`'s template; true if it was there.
pub fn delete_template(
    table_root: &Url,
    options: &HashMap<String, String>,
    plan_id: &str,
) -> Result<bool> {
    let root = template_root(table_root, plan_id)?;
    let location = root
        .join(&format!("_delta_log/{:020}.json", 0))
        .map_err(|e| NativeError::Invalid(e.to_string()))?;
    let store = store::build_store(table_root, options)?;
    let path =
        Path::from_url_path(location.path()).map_err(delta_kernel::object_store::Error::from)?;
    match runtime::block_on(async { store.delete(&path).await }) {
        Ok(()) => Ok(true),
        Err(delta_kernel::object_store::Error::NotFound { .. }) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Whether the table's version 0 is there: false only when storage says it is not.
pub fn published(table_root: &Url, options: &HashMap<String, String>) -> Result<bool> {
    let store = store::build_store(table_root, options)?;
    let root =
        Path::from_url_path(table_root.path()).map_err(delta_kernel::object_store::Error::from)?;
    let first = root.join("_delta_log").join(format!("{:020}.json", 0));
    match runtime::block_on(async { store.head(&first).await }) {
        Ok(_) => Ok(true),
        Err(delta_kernel::object_store::Error::NotFound { .. }) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// `published` for Python.
#[pyfunction]
#[pyo3(signature = (table_root, options = None))]
pub fn create_published(
    py: Python<'_>,
    table_root: &str,
    options: Option<HashMap<String, String>>,
) -> PyResult<bool> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    Ok(py.detach(|| published(&url, &options))?)
}

/// `write_template` for Python.
#[pyfunction]
#[pyo3(signature = (table_root, plan_id, actions, options = None))]
pub fn write_create_template(
    py: Python<'_>,
    table_root: &str,
    plan_id: &str,
    actions: Vec<String>,
    options: Option<HashMap<String, String>>,
) -> PyResult<(String, i64, u64)> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    Ok(py.detach(|| write_template(&url, &options, plan_id, &actions))?)
}

/// `delete_template` for Python.
#[pyfunction]
#[pyo3(signature = (table_root, plan_id, options = None))]
pub fn delete_create_template(
    py: Python<'_>,
    table_root: &str,
    plan_id: &str,
    options: Option<HashMap<String, String>>,
) -> PyResult<bool> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    Ok(py.detach(|| delete_template(&url, &options, plan_id))?)
}

/// `rollback_create` for Python.
#[pyfunction]
#[pyo3(signature = (table_root, metadata_id, options = None))]
pub fn rollback_create_table(
    py: Python<'_>,
    table_root: &str,
    metadata_id: &str,
    options: Option<HashMap<String, String>>,
) -> PyResult<bool> {
    let url = PySnapshot::table_root_url(table_root)?;
    let options = options.unwrap_or_default();
    Ok(py.detach(|| rollback_create(&url, &options, metadata_id))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> Url {
        Url::parse("s3://bucket/t/").unwrap()
    }

    #[test]
    fn accepts_a_template_under_the_pending_directory() {
        let path = template_log_path(
            &root(),
            "s3://bucket/t/_deltaswamp_pending/abc/_delta_log/00000000000000000000.json",
            0,
            10,
        );
        assert!(path.is_ok(), "{path:?}");
    }

    #[test]
    fn refuses_anything_else() {
        for location in [
            "s3://bucket/t/_delta_log/00000000000000000000.json",
            "s3://bucket/other/_deltaswamp_pending/abc/_delta_log/00000000000000000000.json",
            "s3://bucket/t/_deltaswamp_pending/abc/_delta_log/00000000000000000001.json",
            "s3://bucket/t/_deltaswamp_pending/../_delta_log/00000000000000000000.json",
            "s3://bucket/t/_deltaswamp_pending/a/b/_delta_log/00000000000000000000.json",
            "not a url",
        ] {
            assert!(
                template_log_path(&root(), location, 0, 10).is_err(),
                "{location} was accepted"
            );
        }
    }

    #[test]
    fn a_plan_id_cannot_leave_the_pending_directory() {
        for bad in ["", "..", "a/b", "a%2Fb", "."] {
            assert!(template_root(&root(), bad).is_err(), "{bad:?} was accepted");
        }
        assert_eq!(
            template_root(&root(), "abc-1").unwrap().as_str(),
            "s3://bucket/t/_deltaswamp_pending/abc-1/"
        );
    }

    #[test]
    fn matches_the_metadata_id() {
        let body = b"{\"commitInfo\":{}}\n{\"metaData\":{\"id\":\"abc\"}}\n";
        assert!(is_metadata_of(body, "abc"));
        assert!(!is_metadata_of(body, "abd"));
    }
}
