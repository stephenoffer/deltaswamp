//! Symlink format manifests (`GENERATE symlink_format_manifest`), as Spark writes them.
//!
//! delta-rs cannot open a table with clustering, row tracking, in-commit
//! timestamps, type widening or column defaults for writing, so it refused to
//! generate a manifest for one, though a manifest depends on none of them:
//! it is the list of the snapshot's live files, for engines (Presto, Athena,
//! Redshift Spectrum) that read that instead of the log. Spark's
//! `GenerateSymlinkManifest` writes:
//!
//! * `_symlink_format_manifest/manifest` for an unpartitioned table (empty
//!   when the table is), and `_symlink_format_manifest/<fragment>/manifest` per
//!   partition otherwise, the fragment being `col=value/...` with Hive's
//!   escaping and `__HIVE_DEFAULT_PARTITION__` for a null or empty value;
//! * one absolute path per line, decoded (a Hadoop path, not a URL);
//! * and deletes the manifests of partitions that no longer have files.
//!
//! Tables whose files a manifest cannot describe (deletion vectors, column
//! mapping) are refused by the caller, as Spark refuses them.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::Engine;
use futures::StreamExt;
use percent_encoding::{percent_decode_str, utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use url::Url;

use crate::error::{NativeError, Result};
use crate::files;
use crate::runtime;
use crate::vacuum;

/// The directory manifests are written under, relative to the table root.
pub const MANIFEST_DIR: &str = "_symlink_format_manifest";
/// The partition name of a null or empty partition value.
const DEFAULT_PARTITION: &str = "__HIVE_DEFAULT_PARTITION__";

/// Hive's `escapePathName`, as Spark applies it to partition directory names.
fn escape(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        let escaped = matches!(
            c,
            '\u{01}'
                ..='\u{1F}'
                    | '"'
                    | '#'
                    | '%'
                    | '\''
                    | '*'
                    | '/'
                    | ':'
                    | '='
                    | '?'
                    | '\\'
                    | '\u{7F}'
                    | '{'
                    | '['
                    | ']'
                    | '^'
        );
        if escaped {
            out.push_str(&format!("%{:02X}", c as u32));
        } else {
            out.push(c);
        }
    }
    out
}

/// The partition directory of one file: `col=value/...` in partition-column order.
fn fragment(columns: &[String], values: &serde_json::Map<String, serde_json::Value>) -> String {
    columns
        .iter()
        .map(|column| {
            let value = values
                .get(column)
                .and_then(serde_json::Value::as_str)
                .filter(|v| !v.is_empty());
            let value = value.map_or_else(|| DEFAULT_PARTITION.to_string(), escape);
            format!("{}={value}", escape(column))
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// A file's location as a manifest line: its URL with the path decoded.
fn line(url: &Url) -> String {
    if url.scheme() == "file" {
        if let Ok(path) = url.to_file_path() {
            return path.to_string_lossy().into_owned();
        }
    }
    let prefix = &url[..url::Position::BeforePath];
    let path = percent_decode_str(url.path()).decode_utf8_lossy();
    format!("{prefix}{path}")
}

/// Characters a path segment of the manifest URL must encode.
const SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'=');

fn path_error(e: delta_kernel::object_store::path::Error) -> NativeError {
    delta_kernel::object_store::Error::from(e).into()
}

/// Write the manifests of `snapshot`, replacing the ones there, and delete
/// those of partitions with no files. Returns the manifests written, as
/// paths relative to the table root.
pub fn write(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    store: Arc<DynObjectStore>,
) -> Result<Vec<String>> {
    let root = snapshot.table_root().clone();
    let columns: Vec<String> = snapshot
        .table_configuration()
        .metadata()
        .partition_columns()
        .to_vec();
    let mut manifests: BTreeMap<String, Vec<String>> = BTreeMap::new();
    if columns.is_empty() {
        manifests.insert(String::new(), Vec::new());
    }
    for add in files::add_actions(snapshot.clone(), engine)? {
        let add: serde_json::Value = serde_json::from_str(&add)
            .map_err(|e| NativeError::Invalid(format!("bad add action: {e}")))?;
        let path = add
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| NativeError::Invalid("an add action without a path".into()))?;
        let url = match Url::parse(path) {
            Ok(url) => url,
            Err(url::ParseError::RelativeUrlWithoutBase) => root.join(path)?,
            Err(e) => return Err(e.into()),
        };
        let values = add
            .get("partitionValues")
            .and_then(serde_json::Value::as_object)
            .cloned()
            .unwrap_or_default();
        manifests
            .entry(fragment(&columns, &values))
            .or_default()
            .push(line(&url));
    }

    let storage = engine.storage_handler();
    let mut written = Vec::new();
    let mut kept: HashSet<Path> = HashSet::new();
    for (fragment, mut lines) in manifests {
        lines.sort();
        let mut relative = String::from(MANIFEST_DIR);
        for segment in fragment.split('/').filter(|s| !s.is_empty()) {
            relative.push('/');
            relative.push_str(&utf8_percent_encode(segment, SEGMENT).to_string());
        }
        relative.push_str("/manifest");
        let url = root.join(&relative)?;
        let body = if lines.is_empty() {
            String::new()
        } else {
            lines.join("\n") + "\n"
        };
        storage.put(&url, Bytes::from(body), true)?;
        kept.insert(Path::from_url_path(url.path()).map_err(path_error)?);
        written.push(if fragment.is_empty() {
            format!("{MANIFEST_DIR}/manifest")
        } else {
            format!("{MANIFEST_DIR}/{fragment}/manifest")
        });
    }

    // The manifests of partitions that are gone.
    let root_path = Path::from_url_path(root.path()).map_err(path_error)?;
    let dir =
        Path::from_url_path(root.join(&format!("{MANIFEST_DIR}/"))?.path()).map_err(path_error)?;
    let listed = runtime::block_on(async {
        let mut listing = store.list(Some(&dir));
        let mut out = Vec::new();
        while let Some(meta) = listing.next().await {
            out.push(meta?);
        }
        Ok::<_, NativeError>(out)
    })?;
    let stale: Vec<String> = listed
        .into_iter()
        .filter(|meta| meta.location.filename() == Some("manifest"))
        .filter(|meta| !kept.contains(&meta.location))
        .filter_map(|meta| {
            let parts: Vec<String> = meta
                .location
                .prefix_match(&root_path)?
                .map(|p| p.as_ref().to_string())
                .collect();
            Some(parts.join("/"))
        })
        .collect();
    if !stale.is_empty() {
        let (_, failed) = vacuum::delete(store, &root, stale)?;
        if let Some((key, message)) = failed.first() {
            return Err(NativeError::Invalid(format!(
                "wrote the manifests, but could not delete the stale one {key}: {message}"
            )));
        }
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_names_are_escaped_as_hive_escapes_them() {
        assert_eq!(escape("a b"), "a b");
        assert_eq!(escape("x/y"), "x%2Fy");
        assert_eq!(escape("é=%"), "é%3D%25");
        assert_eq!(escape("a:b#c?d"), "a%3Ab%23c%3Fd");
        assert_eq!(escape("\u{1}"), "%01");
    }

    #[test]
    fn fragments_follow_partition_column_order() {
        let values: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(r#"{"h": "1", "g": null, "e": ""}"#).unwrap();
        let columns = ["g".to_string(), "h".to_string(), "e".to_string()];
        assert_eq!(
            fragment(&columns, &values),
            "g=__HIVE_DEFAULT_PARTITION__/h=1/e=__HIVE_DEFAULT_PARTITION__"
        );
    }

    #[test]
    fn lines_are_decoded_paths() {
        let url = Url::parse("s3://bucket/t/g=a%20b/part%2D0.parquet").unwrap();
        assert_eq!(line(&url), "s3://bucket/t/g=a b/part-0.parquet");
        let url = Url::parse("file:///tmp/t%20t/g=x%252Fy/p.parquet").unwrap();
        assert_eq!(line(&url), "/tmp/t t/g=x%2Fy/p.parquet");
    }
}
