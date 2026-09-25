//! The shared Tokio runtime for all kernel calls.
//!
//! This MUST be multi-threaded. `UCCommitter` bridges its async UC calls with
//! `tokio::task::block_in_place`, which panics outright on a current-thread
//! runtime. Building the runtime in one place makes that non-negotiable.

use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
use std::sync::Mutex;
use tokio::runtime::{Builder, Runtime};

/// The runtime, and the id of the process that built it.
///
/// A forked child (Python `multiprocessing` with the default "fork" start
/// method on Linux) inherits the runtime's memory but none of its worker
/// threads, so every `block_on` in the child waited forever on tasks no
/// thread would ever run. The runtime is therefore per *process*: a child
/// that finds a parent's runtime builds its own (the parent's is leaked, as
/// dropping it would try to join threads that do not exist).
static RUNTIME: AtomicPtr<Runtime> = AtomicPtr::new(std::ptr::null_mut());
static OWNER_PID: AtomicU32 = AtomicU32::new(0);
static BUILD: Mutex<()> = Mutex::new(());

/// Returns the process-wide multi-threaded runtime, building it on first use.
pub fn runtime() -> &'static Runtime {
    let pid = std::process::id();
    // Pid first: seeing this pid guarantees seeing the runtime stored before it.
    let owner = OWNER_PID.load(Ordering::Acquire);
    let current = RUNTIME.load(Ordering::Acquire);
    if !current.is_null() && owner == pid {
        // SAFETY: the pointer came from `Box::leak` and is never freed.
        return unsafe { &*current };
    }
    let _guard = BUILD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let owner = OWNER_PID.load(Ordering::Acquire);
    let current = RUNTIME.load(Ordering::Acquire);
    if !current.is_null() && owner == pid {
        // SAFETY: as above.
        return unsafe { &*current };
    }
    let built: &'static Runtime = Box::leak(Box::new(
        Builder::new_multi_thread()
            .enable_all()
            .thread_name("deltaswamp")
            .build()
            .expect("failed to build the deltaswamp Tokio runtime"),
    ));
    // Publish the runtime before the pid, so a reader that sees this pid
    // also sees this runtime.
    RUNTIME.store(built as *const Runtime as *mut Runtime, Ordering::Release);
    OWNER_PID.store(pid, Ordering::Release);
    built
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
