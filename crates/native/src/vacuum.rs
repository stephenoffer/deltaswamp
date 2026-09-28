//! VACUUM planned from the kernel's log replay.
//!
//! delta-rs cannot commit to a table carrying clustering, row tracking,
//! in-commit timestamps, type widening, `vacuumProtocolCheck` and the rest, so
//! it refuses to VACUUM one -- and those are most tables Databricks creates.
//! Nothing about which files a VACUUM may delete depends on those features,
//! only on which files the table still references. This module works that out
//! the way Spark's `VacuumCommand` does:
//!
//! * referenced: every live data file and its deletion-vector file; every file
//!   a `remove` still protects (its `deletionTimestamp` at or after the
//!   cutoff) and that remove's deletion-vector file; and the `_change_data`
//!   files of commits written since the cutoff.
//! * a full VACUUM lists the table directory, skips hidden paths (a name that
//!   starts with `_` or `.`, other than `_change_data`, `_delta_index` and a
//!   partition directory `<column>=`; and Iceberg's `metadata`), and returns
//!   every unreferenced file last modified before the cutoff.
//! * VACUUM LITE lists nothing: it returns the files that expired removes
//!   name and nothing references, which still exist.
//!
//! Tombstones come from the snapshot's log segment -- its checkpoint (where
//! writers keep the removes `delta.deletedFileRetentionDuration` still
//! protects) and the commits after it -- read by the kernel, so v2 checkpoints
//! with sidecars and compacted logs count too. A remove older than the
//! checkpoint and past that retention is no longer in the log, which is
//! Delta's own rule: nothing protects its file any more.
//!
//! Deleting is separate (`delete`), so the caller can vet the list first.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch, StructArray};
use arrow::datatypes::{DataType as ArrowType, Int32Type, Int64Type};
use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use delta_kernel::engine::arrow_data::EngineDataArrowExt;
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::{DynObjectStore, ObjectStore, ObjectStoreExt};
use delta_kernel::schema::{DataType, StructField, StructType};
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::Engine;
use futures::stream::{self, StreamExt};
use percent_encoding::percent_decode_str;
use url::Url;

use crate::confine;
use crate::error::{NativeError, Result};
use crate::files;
use crate::runtime;

/// How many storage requests (HEAD, DELETE) are in flight at once.
const CONCURRENCY: usize = 32;

/// A file a VACUUM would delete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The object's key relative to the table root, as the store spells it:
    /// what `delete` takes back.
    pub key: String,
    /// The path relative to the table root, as a person reads it.
    pub path: String,
    pub size: u64,
    pub modified_ms: i64,
}

/// Files the table still references, relative to its root.
///
/// Each file is recorded in two spellings: the object store's own (percent-
/// encoding a few characters, `%` among them, on a local filesystem) and the
/// plain decoded key (what S3 lists). A listed file is referenced if either
/// spelling matches, so a key with `%` in it is never deleted for being
/// spelled differently -- the error that way is keeping an orphan.
///
/// Keys also match regardless of case. On a case-insensitive filesystem
/// (macOS and Windows by default) `nt/` and `nT/` are one directory, which
/// lists under the spelling that created it; Spark and the kernel pick
/// random mixed-case prefixes, so a live `nt/x.parquet` can list as
/// `nT/x.parquet`, and an exact match deleted it. On a case-sensitive store
/// the cost is keeping an orphan that differs from a live file only by case.
#[derive(Default)]
struct Referenced {
    keys: HashSet<String>,
}

impl Referenced {
    fn insert(&mut self, root: &Url, root_path: &Path, url: &Url) {
        if !confine::is_under(root, url) {
            return;
        }
        if let Some(key) = store_key(root_path, url) {
            self.keys.insert(key.to_lowercase());
        }
        if let Some(key) = decoded_key(root, url) {
            self.keys.insert(key.to_lowercase());
        }
    }

    fn contains(&self, key: &str) -> bool {
        self.keys.contains(&key.to_lowercase()) || self.keys.contains(&decode(key).to_lowercase())
    }
}

fn decode(text: &str) -> String {
    percent_decode_str(text).decode_utf8_lossy().into_owned()
}

/// `url`'s key under the root, in the object store's spelling.
fn store_key(root_path: &Path, url: &Url) -> Option<String> {
    let path = Path::from_url_path(url.path()).ok()?;
    let parts: Vec<String> = path
        .prefix_match(root_path)?
        .map(|part| part.as_ref().to_string())
        .collect();
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// `url`'s key under the root, percent-decoded.
fn decoded_key(root: &Url, url: &Url) -> Option<String> {
    let full = decode(url.path());
    let base = decode(root.path());
    let rest = full.strip_prefix(base.trim_end_matches('/'))?;
    let rest = rest.strip_prefix('/')?;
    (!rest.is_empty()).then(|| rest.to_string())
}

/// A path from the log (relative and URL-encoded, or an absolute URI) as a URL.
fn resolve(root: &Url, path: &str) -> Result<Url> {
    match Url::parse(path) {
        Ok(url) => Ok(url),
        Err(url::ParseError::RelativeUrlWithoutBase) => Ok(root.join(path)?),
        Err(e) => Err(e.into()),
    }
}

/// The deletion-vector file a descriptor names, or None for an inline one.
pub fn deletion_vector_url(
    root: &Url,
    storage_type: &str,
    path_or_inline: &str,
    offset: Option<i32>,
    size_in_bytes: i32,
    cardinality: i64,
) -> Result<Option<Url>> {
    let kind: DeletionVectorStorageType = storage_type.parse()?;
    if kind == DeletionVectorStorageType::Inline {
        return Ok(None);
    }
    let descriptor = DeletionVectorDescriptor::try_new(
        kind,
        path_or_inline,
        offset,
        size_in_bytes,
        cardinality,
    )?;
    Ok(descriptor.absolute_path(root)?)
}

/// The deletion-vector file named by a descriptor in its Delta JSON form.
pub fn deletion_vector_url_json(root: &Url, dv: &serde_json::Value) -> Result<Option<Url>> {
    let text = |key: &str| dv.get(key).and_then(serde_json::Value::as_str);
    let int = |key: &str| dv.get(key).and_then(serde_json::Value::as_i64);
    let (Some(kind), Some(path)) = (text("storageType"), text("pathOrInlineDv")) else {
        return Err(NativeError::Invalid(format!(
            "a deletion vector descriptor without storageType or pathOrInlineDv: {dv}"
        )));
    };
    deletion_vector_url(
        root,
        kind,
        path,
        int("offset").map(|o| o as i32),
        int("sizeInBytes").unwrap_or(0) as i32,
        int("cardinality").unwrap_or(0),
    )
}

/// The deletion-vector file of row `i` of a `deletionVector` struct column.
fn struct_dv_url(root: &Url, dv: &StructArray, i: usize) -> Result<Option<Url>> {
    if dv.is_null(i) {
        return Ok(None);
    }
    let text = |name: &str| -> Option<String> {
        let column = dv.column_by_name(name)?;
        let column = arrow::compute::cast(column, &ArrowType::Utf8).ok()?;
        let column = column.as_string::<i32>();
        (!column.is_null(i)).then(|| column.value(i).to_string())
    };
    let int32 = |name: &str| -> Option<i32> {
        let column = dv.column_by_name(name)?;
        let column = arrow::compute::cast(column, &ArrowType::Int32).ok()?;
        let column = column.as_primitive::<Int32Type>();
        (!column.is_null(i)).then(|| column.value(i))
    };
    let int64 = |name: &str| -> Option<i64> {
        let column = dv.column_by_name(name)?;
        let column = arrow::compute::cast(column, &ArrowType::Int64).ok()?;
        let column = column.as_primitive::<Int64Type>();
        (!column.is_null(i)).then(|| column.value(i))
    };
    let (Some(kind), Some(path)) = (text("storageType"), text("pathOrInlineDv")) else {
        return Ok(None);
    };
    deletion_vector_url(
        root,
        &kind,
        &path,
        int32("offset"),
        int32("sizeInBytes").unwrap_or(0),
        int64("cardinality").unwrap_or(0),
    )
}

/// The projection of `remove` actions the plan reads from the log.
fn remove_schema() -> Result<Arc<StructType>> {
    let dv = StructType::try_new([
        StructField::nullable("storageType", DataType::STRING),
        StructField::nullable("pathOrInlineDv", DataType::STRING),
        StructField::nullable("offset", DataType::INTEGER),
        StructField::nullable("sizeInBytes", DataType::INTEGER),
        StructField::nullable("cardinality", DataType::LONG),
    ])?;
    let remove = StructType::try_new([
        StructField::nullable("path", DataType::STRING),
        StructField::nullable("deletionTimestamp", DataType::LONG),
        StructField::nullable("size", DataType::LONG),
        StructField::nullable("deletionVector", DataType::Struct(Box::new(dv))),
    ])?;
    Ok(Arc::new(StructType::try_new([StructField::nullable(
        "remove",
        DataType::Struct(Box::new(remove)),
    )])?))
}

/// One `remove` from the log: its file, deletion vector and timestamp.
struct Tombstone {
    file: Url,
    vector: Option<Url>,
    deleted_ms: i64,
    size: Option<i64>,
}

/// Every `remove` in the snapshot's log segment (checkpoint and commits).
fn tombstones(snapshot: &SnapshotRef, engine: &dyn Engine) -> Result<Vec<Tombstone>> {
    let root = snapshot.table_root();
    let mut out = Vec::new();
    for batch in snapshot
        .log_segment()
        .read_actions(engine, remove_schema()?)?
    {
        let batch: RecordBatch = batch?.actions.try_into_record_batch()?;
        let Some(remove) = batch.column_by_name("remove") else {
            continue;
        };
        let Some(remove) = remove.as_struct_opt() else {
            continue;
        };
        let path = remove
            .column_by_name("path")
            .map(|c| arrow::compute::cast(c, &ArrowType::Utf8))
            .transpose()?;
        let Some(path) = path else { continue };
        let path = path.as_string::<i32>();
        let deleted = remove
            .column_by_name("deletionTimestamp")
            .map(|c| arrow::compute::cast(c, &ArrowType::Int64))
            .transpose()?;
        let size = remove
            .column_by_name("size")
            .map(|c| arrow::compute::cast(c, &ArrowType::Int64))
            .transpose()?;
        let dv = remove.column_by_name("deletionVector");
        for i in 0..remove.len() {
            if remove.is_null(i) || path.is_null(i) {
                continue;
            }
            let file = resolve(root, path.value(i))?;
            let vector = match dv.and_then(|c| c.as_struct_opt()) {
                Some(dv) => struct_dv_url(root, dv, i)?,
                None => None,
            };
            // A remove without a timestamp protects nothing, as in Spark
            // (`delTimestamp` defaults to 0).
            let deleted_ms = deleted
                .as_ref()
                .map(|c| c.as_primitive::<Int64Type>())
                .filter(|c| !c.is_null(i))
                .map(|c| c.value(i))
                .unwrap_or(0);
            let size = size
                .as_ref()
                .map(|c| c.as_primitive::<Int64Type>())
                .filter(|c| !c.is_null(i))
                .map(|c| c.value(i));
            out.push(Tombstone {
                file,
                vector,
                deleted_ms,
                size,
            });
        }
    }
    Ok(out)
}

/// The `_change_data` files of the segment's commits written at or after `cutoff_ms`.
///
/// Change-data files appear only in commits, never in checkpoints; a commit
/// before the latest checkpoint is older than it, and its files are left to
/// the modification-time rule, as Spark leaves every change-data file.
fn recent_change_data(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    cutoff_ms: i64,
) -> Result<Vec<Url>> {
    let root = snapshot.table_root();
    let commits: Vec<Url> = snapshot
        .log_segment()
        .listed
        .ascending_commit_files
        .iter()
        .filter(|commit| commit.location.last_modified >= cutoff_ms)
        .map(|commit| commit.location.location.clone())
        .collect();
    if commits.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let contents = engine
        .storage_handler()
        .read_files(commits.into_iter().map(|url| (url, None)).collect())?;
    for data in contents {
        let data = data?;
        for line in data.split(|b| *b == b'\n') {
            // Only lines naming a cdc action are parsed.
            if !line.windows(6).any(|w| w == b"\"cdc\":") {
                continue;
            }
            let Ok(action) = serde_json::from_slice::<serde_json::Value>(line) else {
                continue;
            };
            if let Some(path) = action
                .get("cdc")
                .and_then(|cdc| cdc.get("path"))
                .and_then(serde_json::Value::as_str)
            {
                out.push(resolve(root, path)?);
            }
        }
    }
    Ok(out)
}

/// Whether one name in a path hides it from VACUUM, as Spark's
/// `DeltaTableUtils.isHiddenDirectory` decides for directories and files.
fn hidden(name: &str, partition_columns: &[String]) -> bool {
    if name == "metadata" {
        // Reserved for UniForm's Iceberg metadata.
        return true;
    }
    (name.starts_with('_') || name.starts_with('.'))
        && !name.starts_with("_delta_index")
        && !name.starts_with("_change_data")
        && !partition_columns
            .iter()
            .any(|column| name.starts_with(&format!("{column}=")))
}

/// What a VACUUM of `snapshot` would delete, with a retention cutoff of `cutoff_ms`.
///
/// `partition_columns` are the names partition directories may carry
/// (logical and physical). `lite` plans VACUUM LITE.
pub fn plan(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    store: Arc<DynObjectStore>,
    cutoff_ms: i64,
    lite: bool,
    partition_columns: &[String],
) -> Result<Vec<Candidate>> {
    let root = snapshot.table_root().clone();
    let root_path = Path::from_url_path(root.path()).map_err(path_error)?;
    let mut referenced = Referenced::default();

    let live = files::list_files(snapshot.clone(), engine, None, false)?;
    let paths = arrow::compute::cast(
        live.column_by_name("path")
            .ok_or_else(|| NativeError::Invalid("file listing has no path".into()))?,
        &ArrowType::Utf8,
    )?;
    let paths = paths.as_string::<i32>();
    let vectors = arrow::compute::cast(
        live.column_by_name("deletion_vector")
            .ok_or_else(|| NativeError::Invalid("file listing has no deletion_vector".into()))?,
        &ArrowType::Utf8,
    )?;
    let vectors = vectors.as_string::<i32>();
    for i in 0..live.num_rows() {
        referenced.insert(&root, &root_path, &resolve(&root, paths.value(i))?);
        if !vectors.is_null(i) {
            let dv: serde_json::Value = serde_json::from_str(vectors.value(i))
                .map_err(|e| NativeError::Invalid(format!("bad deletion vector JSON: {e}")))?;
            if let Some(url) = deletion_vector_url_json(&root, &dv)? {
                referenced.insert(&root, &root_path, &url);
            }
        }
    }

    let mut expired = Vec::new();
    for tombstone in tombstones(snapshot, engine)? {
        if tombstone.deleted_ms >= cutoff_ms {
            referenced.insert(&root, &root_path, &tombstone.file);
            if let Some(vector) = &tombstone.vector {
                referenced.insert(&root, &root_path, vector);
            }
        } else {
            expired.push(tombstone);
        }
    }
    for url in recent_change_data(snapshot, engine, cutoff_ms)? {
        referenced.insert(&root, &root_path, &url);
    }

    let local = root.scheme() == "file";
    let shown = |key: &str| if local { decode(key) } else { key.to_string() };

    if lite {
        // Each file once, and only those under the root: a remove naming a
        // file elsewhere never lets this delete it.
        let mut seen = HashSet::new();
        let mut wanted: Vec<(String, Option<i64>)> = Vec::new();
        for tombstone in &expired {
            let urls = std::iter::once((&tombstone.file, tombstone.size))
                .chain(tombstone.vector.iter().map(|url| (url, None)));
            for (url, size) in urls {
                if !confine::is_under(&root, url) {
                    continue;
                }
                let Some(key) = store_key(&root_path, url) else {
                    continue;
                };
                if referenced.contains(&key) || !seen.insert(key.clone()) {
                    continue;
                }
                wanted.push((key, size));
            }
        }
        let heads = runtime::block_on(async {
            stream::iter(wanted.into_iter().map(|(key, _)| {
                let store = store.clone();
                let location = child(&root_path, &key);
                async move {
                    let location = location?;
                    match store.head(&location).await {
                        Ok(meta) => Ok(Some((
                            key,
                            meta.size,
                            meta.last_modified.timestamp_millis(),
                        ))),
                        Err(delta_kernel::object_store::Error::NotFound { .. }) => Ok(None),
                        Err(e) => Err(NativeError::from(e)),
                    }
                }
            }))
            .buffer_unordered(CONCURRENCY)
            .collect::<Vec<_>>()
            .await
        });
        let mut out = Vec::new();
        for head in heads {
            if let Some((key, size, modified_ms)) = head? {
                out.push(Candidate {
                    path: shown(&key),
                    key,
                    size,
                    modified_ms,
                });
            }
        }
        out.sort_by(|a, b| a.path.cmp(&b.path));
        return Ok(out);
    }

    let listed = runtime::block_on(async {
        let mut listing = store.list(Some(&root_path));
        let mut out = Vec::new();
        while let Some(meta) = listing.next().await {
            out.push(meta?);
        }
        Ok::<_, NativeError>(out)
    })?;
    let mut out = Vec::new();
    for meta in listed {
        let Some(parts) = meta.location.prefix_match(&root_path) else {
            continue;
        };
        let parts: Vec<String> = parts.map(|p| p.as_ref().to_string()).collect();
        if parts.is_empty()
            || parts
                .iter()
                .any(|part| hidden(&decode(part), partition_columns))
        {
            continue;
        }
        let key = parts.join("/");
        let modified_ms = meta.last_modified.timestamp_millis();
        if modified_ms >= cutoff_ms || referenced.contains(&key) {
            continue;
        }
        out.push(Candidate {
            path: shown(&key),
            key,
            size: meta.size,
            modified_ms,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn path_error(e: delta_kernel::object_store::path::Error) -> NativeError {
    delta_kernel::object_store::Error::from(e).into()
}

/// The object at `key` under the root, keeping the key's own spelling.
fn child(root_path: &Path, key: &str) -> Result<Path> {
    let full = if root_path.as_ref().is_empty() {
        key.to_string()
    } else {
        format!("{}/{key}", root_path.as_ref())
    };
    Path::parse(full).map_err(path_error)
}

/// The keys deleted (or already gone), and `(key, error)` for each that was not.
pub type Deletion = (Vec<String>, Vec<(String, String)>);

/// Delete `keys` (as `plan` reported them) under the table root.
///
/// Returns the keys deleted (or already gone) and, for every other key, the
/// storage error; a failure does not stop the rest.
pub fn delete(store: Arc<DynObjectStore>, root: &Url, keys: Vec<String>) -> Result<Deletion> {
    let root_path = Path::from_url_path(root.path()).map_err(path_error)?;
    let results = runtime::block_on(async {
        stream::iter(keys.into_iter().map(|key| {
            let store = store.clone();
            let location = child(&root_path, &key);
            async move {
                let outcome = match location {
                    Ok(location) => match store.delete(&location).await {
                        Ok(()) | Err(delta_kernel::object_store::Error::NotFound { .. }) => Ok(()),
                        Err(e) => Err(e.to_string()),
                    },
                    Err(e) => Err(e.to_string()),
                };
                (key, outcome)
            }
        }))
        .buffer_unordered(CONCURRENCY)
        .collect::<Vec<_>>()
        .await
    });
    let mut deleted = Vec::new();
    let mut failed = Vec::new();
    for (key, outcome) in results {
        match outcome {
            Ok(()) => deleted.push(key),
            Err(message) => failed.push((key, message)),
        }
    }
    deleted.sort();
    Ok((deleted, failed))
}

/// The files among `urls` that are not in storage, as their URLs.
pub fn missing(store: Arc<DynObjectStore>, root: &Url, urls: Vec<Url>) -> Result<Vec<String>> {
    let root_path = Path::from_url_path(root.path()).map_err(path_error)?;
    let results = runtime::block_on(async {
        stream::iter(urls.into_iter().map(|url| {
            let store = store.clone();
            let root_path = root_path.clone();
            async move {
                // A file outside the table is not looked up: the confined
                // reader would refuse it anyway.
                let Some(key) = store_key(&root_path, &url) else {
                    return Ok(Some(url.to_string()));
                };
                let location = child(&root_path, &key)?;
                match store.head(&location).await {
                    Ok(_) => Ok(None),
                    // A case-insensitive filesystem (macOS and Windows by
                    // default) holds `nt/x` as `nT/x` once another file
                    // created the directory that way; the local store then
                    // reports the live file missing. The filesystem is asked.
                    Err(delta_kernel::object_store::Error::NotFound { .. })
                        if url.to_file_path().is_ok_and(|path| path.exists()) =>
                    {
                        Ok(None)
                    }
                    Err(delta_kernel::object_store::Error::NotFound { .. }) => {
                        Ok(Some(url.to_string()))
                    }
                    Err(e) => Err(NativeError::from(e)),
                }
            }
        }))
        .buffer_unordered(CONCURRENCY)
        .collect::<Vec<_>>()
        .await
    });
    let mut out = Vec::new();
    for result in results {
        if let Some(url) = result? {
            out.push(url);
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_follows_spark() {
        let parts = vec!["_p".to_string()];
        assert!(hidden("_delta_log", &parts));
        assert!(hidden(".part-0.parquet.crc", &parts));
        assert!(hidden("_SUCCESS", &parts));
        assert!(hidden("metadata", &parts));
        assert!(!hidden("_change_data", &parts));
        assert!(!hidden("_delta_index", &parts));
        assert!(!hidden("_p=1", &parts));
        assert!(!hidden("part-0.parquet", &parts));
        assert!(!hidden("ab", &parts));
    }

    #[test]
    fn relative_vectors_decode_with_and_without_prefix() {
        let root = Url::parse("file:///t/").unwrap();
        let url = deletion_vector_url(&root, "u", "ab^-aqEH.-t@S}K{vb[*k^", Some(1), 34, 1)
            .unwrap()
            .unwrap();
        assert_eq!(
            url.as_str(),
            "file:///t/ab/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin"
        );
        let url = deletion_vector_url(&root, "u", "vBn[lx{q8@P<9BNH/isA", Some(1), 34, 1)
            .unwrap()
            .unwrap();
        assert_eq!(
            url.as_str(),
            "file:///t/deletion_vector_61d16c75-6994-46b7-a15b-8b538852e50e.bin"
        );
        assert!(deletion_vector_url(
            &root,
            "i",
            "wi5b=000010000siXQKl0rr91000f55c8Xg0@@D72lkbi5=-{L",
            None,
            40,
            6
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn both_spellings_of_a_key_are_referenced() {
        let root = Url::parse("file:///t/").unwrap();
        let root_path = Path::from_url_path(root.path()).unwrap();
        let mut referenced = Referenced::default();
        let url = resolve(&root, "p=a%253Ab/part-0.parquet").unwrap();
        referenced.insert(&root, &root_path, &url);
        // As a local listing spells it, and as S3 lists the raw key.
        assert!(referenced.contains("p=a%253Ab/part-0.parquet"));
        assert!(referenced.contains("p=a%3Ab/part-0.parquet"));
        assert!(!referenced.contains("p=a/part-0.parquet"));
    }

    #[test]
    fn a_key_listed_in_another_case_is_referenced() {
        // A case-insensitive filesystem lists `nt/` as `nT/` once another
        // file created the directory under that spelling.
        let root = Url::parse("file:///t/").unwrap();
        let root_path = Path::from_url_path(root.path()).unwrap();
        let mut referenced = Referenced::default();
        let url = resolve(&root, "nt/9eb12f17.parquet").unwrap();
        referenced.insert(&root, &root_path, &url);
        assert!(referenced.contains("nT/9eb12f17.parquet"));
        assert!(referenced.contains("NT/9EB12F17.parquet"));
        assert!(!referenced.contains("nt/other.parquet"));
    }

    #[test]
    fn files_outside_the_root_are_never_referenced_keys() {
        let root = Url::parse("file:///t/").unwrap();
        let root_path = Path::from_url_path(root.path()).unwrap();
        let mut referenced = Referenced::default();
        referenced.insert(
            &root,
            &root_path,
            &resolve(&root, "../u/x.parquet").unwrap(),
        );
        assert!(referenced.keys.is_empty());
    }
}
