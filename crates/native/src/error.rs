//! Error translation from kernel errors into Python exceptions.

use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyValueError};
use pyo3::types::PyAnyMethods;
use pyo3::{PyErr, Python};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NativeError {
    #[error("delta kernel error: {0}")]
    Kernel(#[from] delta_kernel::Error),

    #[error("object store error: {0}")]
    ObjectStore(#[from] delta_kernel::object_store::Error),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("invalid url: {0}")]
    Url(#[from] url::ParseError),

    #[error("{0}")]
    Invalid(String),

    /// Another writer won the race for this version.
    #[error("{0}")]
    CommitConflict(String),

    /// The catalog is refusing commits until staged ones are published.
    /// This is backpressure, not a rate limit -- backing off will not help.
    #[error("{0}")]
    BackfillRequired(String),

    /// Transient I/O failure; the table is unchanged and a retry is safe.
    #[error("{0}")]
    Retryable(String),

    /// The catalog refused the commit's credentials or privileges (401/403).
    #[error("{0}")]
    CatalogPermission(String),

    /// The catalog no longer has the table (404): dropped or renamed.
    #[error("{0}")]
    CatalogNotFound(String),

    /// Any other refusal of the commit by the catalog.
    #[error("{0}")]
    CatalogRejected(String),
}

pyo3::create_exception!(
    _native,
    CommitConflictError,
    pyo3::exceptions::PyRuntimeError,
    "Another writer committed this version first."
);
pyo3::create_exception!(
    _native,
    BackfillRequiredError,
    pyo3::exceptions::PyRuntimeError,
    "The catalog requires staged commits to be published before accepting more."
);
pyo3::create_exception!(
    _native,
    RetryableError,
    pyo3::exceptions::PyRuntimeError,
    "A transient failure; the table is unchanged and the operation may be retried."
);
// Input the extension refuses. A ValueError subclass, so every existing
// `except ValueError` keeps working while callers can tell the extension's
// own input errors apart from any other ValueError.
pyo3::create_exception!(
    _native,
    InvalidInputError,
    pyo3::exceptions::PyValueError,
    "The extension refused its input: bad arguments, data or fragments."
);
// Catalog refusals of a commit. ValueErrors for the same compatibility reason:
// they reached Python as bare ValueErrors before, and code matched on that.
pyo3::create_exception!(
    _native,
    CatalogCommitError,
    pyo3::exceptions::PyValueError,
    "The catalog refused the commit; nothing was committed."
);
pyo3::create_exception!(
    _native,
    CatalogPermissionError,
    CatalogCommitError,
    "The catalog rejected the commit's credentials or privileges (HTTP 401/403)."
);
pyo3::create_exception!(
    _native,
    CatalogNotFoundError,
    CatalogCommitError,
    "The catalog no longer has this table (HTTP 404)."
);

impl NativeError {
    /// A stable code for what failed, set on every exception the extension
    /// raises as its `kind` attribute.
    ///
    /// Python classified these errors by searching their messages for phrases
    /// ("that are not in the table schema", "Found unmasked nulls"), which
    /// break whenever a message is reworded, here or in a dependency. The
    /// codes are part of the extension's interface: add new ones, never
    /// rename one.
    pub fn kind(&self) -> &'static str {
        match self {
            NativeError::ObjectStore(delta_kernel::object_store::Error::NotFound { .. }) => {
                "not_found"
            }
            NativeError::ObjectStore(_) => "storage",
            NativeError::Kernel(err) => kernel_kind(err),
            NativeError::Arrow(_) => "arrow",
            NativeError::Url(_) | NativeError::Invalid(_) => "invalid_input",
            NativeError::CommitConflict(_) => "commit_conflict",
            NativeError::BackfillRequired(_) => "backfill_required",
            NativeError::Retryable(_) => "retryable",
            NativeError::CatalogPermission(_) => "catalog_permission",
            NativeError::CatalogNotFound(_) => "catalog_not_found",
            NativeError::CatalogRejected(_) => "catalog_rejected",
        }
    }
}

/// `NativeError::kind` for an error the kernel raised.
fn kernel_kind(err: &delta_kernel::Error) -> &'static str {
    use delta_kernel::Error as K;
    match err {
        K::Backtraced { source, .. } => kernel_kind(source),
        K::FileNotFound(_) => "not_found",
        K::ObjectStore(_) | K::IOError(_) | K::Reqwest(_) => "storage",
        K::Arrow(_) => "arrow",
        K::Unsupported(_)
        | K::ChangeDataFeedUnsupported(_)
        | K::RowTrackingChangeFeedUnsupported(_)
        | K::ChecksumWriteUnsupported(_) => "unsupported",
        _ => "kernel",
    }
}

impl From<NativeError> for PyErr {
    fn from(err: NativeError) -> PyErr {
        let kind = err.kind();
        let error = py_error(err);
        // Best effort: an exception that cannot take an attribute is still
        // raised, only without its code.
        Python::attach(|py| {
            let _ = error.value(py).setattr("kind", kind);
        });
        error
    }
}

/// The Python exception class for `err`.
fn py_error(err: NativeError) -> PyErr {
    let message = err.to_string();
    match err {
        // A missing object (e.g. a data file removed by VACUUM) is an
        // I/O failure, not bad input: callers retrying on OSError, or
        // telling "not found" apart, need the right class.
        NativeError::ObjectStore(delta_kernel::object_store::Error::NotFound { .. })
        | NativeError::Kernel(delta_kernel::Error::FileNotFound(_)) => {
            PyFileNotFoundError::new_err(message)
        }
        NativeError::ObjectStore(_)
        | NativeError::Kernel(
            delta_kernel::Error::ObjectStore(_)
            | delta_kernel::Error::IOError(_)
            | delta_kernel::Error::Reqwest(_),
        ) => PyIOError::new_err(message),
        NativeError::CommitConflict(_) => CommitConflictError::new_err(message),
        NativeError::BackfillRequired(_) => BackfillRequiredError::new_err(message),
        NativeError::Retryable(_) => RetryableError::new_err(message),
        NativeError::CatalogPermission(_) => CatalogPermissionError::new_err(message),
        NativeError::CatalogNotFound(_) => CatalogNotFoundError::new_err(message),
        NativeError::CatalogRejected(_) => CatalogCommitError::new_err(message),
        NativeError::Invalid(_) => InvalidInputError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

pub type Result<T> = std::result::Result<T, NativeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_are_stable_codes() {
        assert_eq!(NativeError::Invalid("x".into()).kind(), "invalid_input");
        assert_eq!(
            NativeError::CommitConflict("x".into()).kind(),
            "commit_conflict"
        );
        assert_eq!(
            NativeError::Kernel(delta_kernel::Error::FileNotFound("f".into())).kind(),
            "not_found"
        );
        assert_eq!(
            NativeError::Kernel(delta_kernel::Error::Unsupported("u".into())).kind(),
            "unsupported"
        );
        assert_eq!(
            NativeError::Arrow(arrow::error::ArrowError::ComputeError("c".into())).kind(),
            "arrow"
        );
        assert_eq!(
            NativeError::Kernel(delta_kernel::Error::generic("g")).kind(),
            "kernel"
        );
    }
}
