//! Error translation from kernel errors into Python exceptions.

use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyValueError};
use pyo3::PyErr;
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

impl From<NativeError> for PyErr {
    fn from(err: NativeError) -> PyErr {
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
            _ => PyValueError::new_err(message),
        }
    }
}

pub type Result<T> = std::result::Result<T, NativeError>;
