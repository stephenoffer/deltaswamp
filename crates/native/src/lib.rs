//! `deltaswamp._native` -- PyO3 bindings over delta-kernel-rs.
//!
//! Scope note: Arrow crosses this boundary only via the Arrow C Data Interface
//! (pyo3-arrow), never as shared Rust types. That is what makes it safe for this
//! extension to coexist in one process with the `deltalake` wheel, which links
//! its own (forked) build of the kernel.

mod changes;
mod checksum;
mod commit;
mod confine;
mod dml;
mod error;
mod files;
mod functions;
mod logclean;
mod manifest;
mod partition;
mod predicate;
mod rebase;
mod restate;
mod runtime;
mod scan;
mod snapshot;
mod store;
mod vacuum;
mod writer;

pub use error::{NativeError, Result};

use pyo3::prelude::*;

use commit::UcCommitConfig;
use error::{
    BackfillRequiredError, CatalogCommitError, CatalogNotFoundError, CatalogPermissionError,
    CommitConflictError, InvalidInputError, RetryableError,
};
use snapshot::{create_table, PySnapshot};

/// The delta_kernel version this extension is pinned to.
///
/// delta_kernel exports no VERSION constant of its own, so we record the pin
/// here and assert it against Cargo.lock in a test. Python asserts against the
/// same string, so a kernel bump fails loudly at three layers rather than
/// silently changing behavior.
pub const KERNEL_VERSION: &str = "0.28.0";

/// Capabilities this build provides, by stable name.
///
/// Python gates each feature on this list rather than on `hasattr`, so a stale
/// extension (built before a feature landed) refuses cleanly instead of
/// failing with an `AttributeError` or, worse, a changed signature.
pub const FEATURES: &[&str] = &[
    "predicate_skipping",
    "timestamp_travel",
    "table_changes",
    "files",
    "metadata_json",
    "app_id_version",
    "commit_raw",
    "partitioned_append",
    "uc_create_table_request",
    "checkpoint",
    "file_restricted_scan",
    "legacy_calendar_files",
    // Distributed writes: workers produce data files, a coordinator commits
    // them as one transaction.
    "distributed_write",
    // Row-level DML as deletion vectors: positional scans and `commit_dml`.
    "deletion_vector_dml",
    // UPDATE writes rewritten rows' ids to the materialized row-id column.
    "materialized_row_ids",
    // Raw commit files, for applying Delta's conflict rules on a lost race.
    "commit_log",
    // `commit_dml(data_change=False)`: compactions committed by the kernel.
    "compaction",
    // `operation_parameters=` and `blind_append=` on every commit, written
    // into its commitInfo.
    "commit_info_patch",
    // A compaction streams its rows (`commit_dml` pulls them as it writes),
    // commits past the value-constraint features, and file-restricted scans
    // read files in the order given.
    "streaming_compaction",
    // `validate_retry_options`: retry storage options read as delta-rs reads them.
    "retry_options",
    // `vacuum_plan`/`delete_files`: VACUUM from the kernel's log replay.
    "vacuum",
    // `add_actions`/`missing_files`: RESTORE committed from the target's adds.
    "restore",
    // Compactions and overwrites of row-tracked tables: removes staged by
    // hand, and `scan(row_tracking=True)` rows written back with their ids
    // and commit versions in the materialized columns.
    "row_tracking_compaction",
    // `commit_dml(add_tags=)`: tags on every add (a Z-order's ZCUBE_* tags),
    // and files() lists each file's tags.
    "add_tags",
    // `<version>.crc` after every kernel commit, and `write_checksum()` for
    // one another writer committed.
    "write_checksum",
    // `incremental_files(base_version)`: the file diff between two versions.
    "incremental_files",
    // `absolute_deletion_vector` and `copy_objects`: shallow and deep clones
    // of path tables.
    "path_clone",
    // Copy-on-write DML of row-tracked tables: `commit_dml` stages the
    // removes by hand and writes the rows' ids and commit versions it is
    // given into the materialized columns, and `scan(row_positions=True,
    // row_tracking=True)` reads both beside each row's position.
    "row_tracking_dml",
    // `constraints_checked=` on every data write: CHECK constraints evaluated
    // by the caller over the rows written, and the write committed past the
    // kernel's refusal of the checkConstraints feature.
    "check_constraints",
    // `append(metadata=, protocol=)`: a schema-evolving write, its rows and
    // the table's new metaData in one commit.
    "schema_evolution",
    // `cleanup_log`: expired log cleanup below a retained checkpoint, by
    // commit timestamps (in-commit timestamps where the table has them).
    "log_cleanup",
    // `write_symlink_manifest`: GENERATE symlink_format_manifest.
    "symlink_manifest",
    // `missing_data_files`: FSCK REPAIR from the kernel's file listing.
    "fsck",
    // `checkpoint()` and `write_checksum()` on a table carrying CHECK
    // constraints, generated or identity columns, or invariants: written from
    // a snapshot whose checked protocol sets those aside, with the table's own
    // protocol and metadata in the files.
    "value_constrained_checkpoint",
];

#[pyfunction]
fn kernel_version() -> &'static str {
    KERNEL_VERSION
}

#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// True if the shared runtime supports `block_in_place`.
///
/// `UCCommitter` bridges async UC calls with `block_in_place`, which panics on a
/// current-thread runtime. This is the probe Python uses to assert the
/// invariant holds in the built wheel, not just in our tests.
#[pyfunction]
fn runtime_is_multithreaded() -> bool {
    runtime::block_on(async { tokio::task::block_in_place(|| true) })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("KERNEL_VERSION", KERNEL_VERSION)?;
    m.add("FEATURES", FEATURES.to_vec())?;
    m.add_class::<PySnapshot>()?;
    m.add_class::<UcCommitConfig>()?;
    m.add(
        "CommitConflictError",
        m.py().get_type::<CommitConflictError>(),
    )?;
    m.add(
        "BackfillRequiredError",
        m.py().get_type::<BackfillRequiredError>(),
    )?;
    m.add("RetryableError", m.py().get_type::<RetryableError>())?;
    m.add("InvalidInputError", m.py().get_type::<InvalidInputError>())?;
    m.add(
        "CatalogCommitError",
        m.py().get_type::<CatalogCommitError>(),
    )?;
    m.add(
        "CatalogPermissionError",
        m.py().get_type::<CatalogPermissionError>(),
    )?;
    m.add(
        "CatalogNotFoundError",
        m.py().get_type::<CatalogNotFoundError>(),
    )?;
    m.add_function(wrap_pyfunction!(create_table, m)?)?;
    m.add_function(wrap_pyfunction!(functions::table_changes, m)?)?;
    m.add_function(wrap_pyfunction!(functions::commit_raw, m)?)?;
    m.add_function(wrap_pyfunction!(functions::absolute_deletion_vector, m)?)?;
    m.add_function(wrap_pyfunction!(functions::copy_objects, m)?)?;
    m.add_function(wrap_pyfunction!(functions::probe_put_if_absent, m)?)?;
    m.add_function(wrap_pyfunction!(functions::validate_retry_options, m)?)?;
    m.add_function(wrap_pyfunction!(functions::uc_create_table_request, m)?)?;
    m.add_function(wrap_pyfunction!(functions::uc_required_properties, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_version, m)?)?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(runtime_is_multithreaded, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::KERNEL_VERSION;

    /// The pin in KERNEL_VERSION must match what Cargo actually resolved.
    /// Kernel ships breaking changes roughly every three weeks; this is the
    /// tripwire that makes a bump impossible to do by accident.
    #[test]
    fn kernel_pin_matches_lockfile() {
        let lock = include_str!("../../../Cargo.lock");
        let mut found = None;
        let mut lines = lock.lines().peekable();
        while let Some(line) = lines.next() {
            if line.trim() == r#"name = "delta_kernel""# {
                for next in lines.by_ref() {
                    if let Some(v) = next.trim().strip_prefix(r#"version = ""#) {
                        found = Some(v.trim_end_matches('"').to_string());
                        break;
                    }
                }
                break;
            }
        }
        let resolved = found.expect("delta_kernel not present in Cargo.lock");
        assert_eq!(
            resolved, KERNEL_VERSION,
            "Cargo.lock resolved delta_kernel {resolved} but KERNEL_VERSION pins {KERNEL_VERSION}. \
             Bumping the kernel is a deliberate act: update the pin, re-read the kernel \
             CHANGELOG for breaking changes, and re-run the cross-engine tests."
        );
    }
}
