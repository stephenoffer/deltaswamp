//! Reads confined to the table root.
//!
//! A data file's path in the log is relative to the table root, or an
//! absolute URI. The kernel resolves it with `Url::join`, which follows `..`
//! (and `%2E%2E`, which the url crate reads as the same segment), and hands the
//! result to the table's object store -- so an `add` of `../other/x.parquet`,
//! or `s3://<same bucket>/<other prefix>/x.parquet`, read an object outside
//! the table with the connection's credentials, and compaction then copied its
//! rows into the table. A URI in another bucket was looked up in the table's
//! own bucket under the same key. delta-rs refuses both. Whoever can write a
//! table's `_delta_log` should reach nothing but the table.
//!
//! Every read of a scan goes through the engine returned here, which refuses
//! a Parquet file or a deletion vector whose location is not under the root.
//! Shallow clones, whose files are the source table's by absolute path, are
//! refused before a kernel read is planned (`KernelEngine.capability`).

use std::sync::Arc;

use delta_kernel::{
    CancellationTokenRef, DeltaResult, Engine, EngineData, Error, EvaluationHandler,
    FileDataReadResultIterator, FileMeta, FileSlice, JsonHandler, ParquetFooter, ParquetHandler,
    PredicateRef, StorageHandler,
};
use url::Url;

/// Whether `location` is `root` or below it: same scheme, authority
/// (container, account, bucket, port) and a path under the root's.
pub fn is_under(root: &Url, location: &Url) -> bool {
    if location.scheme() != root.scheme()
        || location.host_str() != root.host_str()
        || location.port_or_known_default() != root.port_or_known_default()
        || location.username() != root.username()
    {
        return false;
    }
    let base = root.path();
    let base = if base.ends_with('/') {
        base.to_string()
    } else {
        format!("{base}/")
    };
    let path = location.path();
    // A segment that still decodes to `..` (a double-encoded one) is never
    // under the root.
    if path
        .split('/')
        .any(|s| s == ".." || s.eq_ignore_ascii_case("%2e%2e"))
    {
        return false;
    }
    path.starts_with(&base)
}

/// `Err` naming the file unless `location` is under `root`.
pub fn check(root: &Url, location: &Url) -> DeltaResult<()> {
    if is_under(root, location) {
        return Ok(());
    }
    Err(Error::generic(format!(
        "the log names the file {location}, which is outside the table root {root}; \
         refusing to read it (a data file or deletion vector must be inside the table)"
    )))
}

/// `engine`, with every Parquet and deletion-vector read confined to `root`.
pub fn confined(engine: Arc<dyn Engine>, root: &Url) -> Arc<dyn Engine> {
    let root = root.clone();
    Arc::new(ConfinedEngine {
        parquet: Arc::new(ConfinedParquet {
            inner: engine.parquet_handler(),
            root: root.clone(),
        }),
        storage: Arc::new(ConfinedStorage {
            inner: engine.storage_handler(),
            root,
        }),
        inner: engine,
    })
}

struct ConfinedEngine {
    inner: Arc<dyn Engine>,
    parquet: Arc<dyn ParquetHandler>,
    storage: Arc<dyn StorageHandler>,
}

impl Engine for ConfinedEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.inner.evaluation_handler()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.storage.clone()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.inner.json_handler()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet.clone()
    }
}

struct ConfinedParquet {
    inner: Arc<dyn ParquetHandler>,
    root: Url,
}

impl ConfinedParquet {
    fn check_all(&self, files: &[FileMeta]) -> DeltaResult<()> {
        files
            .iter()
            .try_for_each(|f| check(&self.root, &f.location))
    }
}

impl ParquetHandler for ConfinedParquet {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: delta_kernel::schema::SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        self.check_all(files)?;
        self.inner
            .read_parquet_files(files, physical_schema, predicate)
    }

    fn read_parquet_files_with_cancellation(
        &self,
        files: &[FileMeta],
        physical_schema: delta_kernel::schema::SchemaRef,
        predicate: Option<PredicateRef>,
        cancellation_token: Option<CancellationTokenRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        self.check_all(files)?;
        self.inner.read_parquet_files_with_cancellation(
            files,
            physical_schema,
            predicate,
            cancellation_token,
        )
    }

    fn write_parquet_file(
        &self,
        location: Url,
        data: delta_kernel::DeltaResultIterator<'static, Box<dyn EngineData>>,
    ) -> DeltaResult<()> {
        self.inner.write_parquet_file(location, data)
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> DeltaResult<ParquetFooter> {
        check(&self.root, &file.location)?;
        self.inner.read_parquet_footer(file)
    }
}

struct ConfinedStorage {
    inner: Arc<dyn StorageHandler>,
    root: Url,
}

type Listing = Box<dyn Iterator<Item = DeltaResult<FileMeta>>>;
type Reads = Box<dyn Iterator<Item = DeltaResult<bytes::Bytes>>>;

impl StorageHandler for ConfinedStorage {
    fn list_from(&self, path: &Url) -> DeltaResult<Listing> {
        self.inner.list_from(path)
    }

    fn list_from_with_cancellation(
        &self,
        path: &Url,
        cancellation_token: Option<CancellationTokenRef>,
    ) -> DeltaResult<Listing> {
        self.inner
            .list_from_with_cancellation(path, cancellation_token)
    }

    fn read_files(&self, files: Vec<FileSlice>) -> DeltaResult<Reads> {
        files.iter().try_for_each(|(u, _)| check(&self.root, u))?;
        self.inner.read_files(files)
    }

    fn read_files_with_cancellation(
        &self,
        files: Vec<FileSlice>,
        cancellation_token: Option<CancellationTokenRef>,
    ) -> DeltaResult<Reads> {
        files.iter().try_for_each(|(u, _)| check(&self.root, u))?;
        self.inner
            .read_files_with_cancellation(files, cancellation_token)
    }

    fn copy_atomic(&self, src: &Url, dest: &Url) -> DeltaResult<()> {
        self.inner.copy_atomic(src, dest)
    }

    fn put(&self, path: &Url, data: bytes::Bytes, overwrite: bool) -> DeltaResult<()> {
        self.inner.put(path, data, overwrite)
    }

    fn head(&self, path: &Url) -> DeltaResult<FileMeta> {
        self.inner.head(path)
    }

    fn delete(&self, path: &Url) -> DeltaResult<()> {
        self.inner.delete(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_that_climb_out_or_leave_the_bucket_are_not_under_the_root() {
        let root = Url::parse("s3://bucket/tables/t/").unwrap();
        let inside = [
            "part-0.parquet",
            "a=1/part-0.parquet",
            "deletion_vector_x.bin",
        ];
        for path in inside {
            assert!(is_under(&root, &root.join(path).unwrap()), "{path}");
        }
        let outside = [
            "../other/secret.parquet",
            "%2E%2E/other/secret.parquet",
            "a/../../x.parquet",
            "s3://bucket/tables/other/x.parquet",
            "s3://victim/tables/t/x.parquet",
            "file:///etc/passwd",
            "/tables/other/x.parquet",
        ];
        for path in outside {
            assert!(!is_under(&root, &root.join(path).unwrap()), "{path}");
        }
        assert!(is_under(
            &root,
            &Url::parse("s3://bucket/tables/t/x.parquet").unwrap()
        ));
        let azure = Url::parse("abfss://c@acct.dfs.core.windows.net/t/").unwrap();
        let other = Url::parse("abfss://d@acct.dfs.core.windows.net/t/x.parquet").unwrap();
        assert!(!is_under(&azure, &other));
    }
}
