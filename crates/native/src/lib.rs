//! `deltaswamp._native` -- PyO3 bindings over delta-kernel-rs.
//!
//! Scope note: Arrow crosses this boundary only via the Arrow C Data Interface
//! (pyo3-arrow), never as shared Rust types. That is what makes it safe for this
//! extension to coexist in one process with the `deltalake` wheel, which links
//! its own (forked) build of the kernel.

mod changes;
mod commit;
mod error;
mod files;
mod functions;
mod partition;
mod predicate;
mod runtime;
mod scan;
mod snapshot;
mod store;

pub use error::{NativeError, Result};

use pyo3::prelude::*;

use commit::UcCommitConfig;
use error::{BackfillRequiredError, CommitConflictError, RetryableError};
use snapshot::{create_table, PySnapshot};

/// The delta_kernel version this extension is pinned to.
///
/// delta_kernel exports no VERSION constant of its own, so we record the pin
/// here and assert it against Cargo.lock in a test. Python asserts against the
/// same string, so a kernel bump fails loudly at three layers rather than
/// silently changing behaviour.
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
    m.add_function(wrap_pyfunction!(create_table, m)?)?;
    m.add_function(wrap_pyfunction!(functions::table_changes, m)?)?;
    m.add_function(wrap_pyfunction!(functions::commit_raw, m)?)?;
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
