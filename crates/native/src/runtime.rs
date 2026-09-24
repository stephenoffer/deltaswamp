//! The shared Tokio runtime for all kernel calls.
//!
//! This MUST be multi-threaded. `UCCommitter` bridges its async UC calls with
//! `tokio::task::block_in_place`, which panics outright on a current-thread
//! runtime. Building the runtime in one place makes that non-negotiable.

use std::sync::OnceLock;
use tokio::runtime::{Builder, Runtime};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Returns the process-wide multi-threaded runtime, building it on first use.
pub fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .enable_all()
            .thread_name("deltaswamp")
            .build()
            .expect("failed to build the deltaswamp Tokio runtime")
    })
}

/// Run a future to completion on the shared runtime.
pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    runtime().block_on(fut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_is_multi_threaded() {
        // block_in_place is only legal on a multi-threaded runtime; it panics
        // on a current-thread one. This asserts the invariant UCCommitter needs.
        block_on(async {
            tokio::task::block_in_place(|| {});
        });
    }
}
