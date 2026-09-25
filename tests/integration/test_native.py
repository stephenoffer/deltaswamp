"""The compiled extension.

Skipped when the extension has not been built, so the pure-Python suite stays
runnable without a Rust toolchain.
"""

from __future__ import annotations

import deltaswamp as ds
import pytest

pytestmark = pytest.mark.skipif(
    not ds.has_native(), reason="native extension not built (run `maturin develop`)"
)


def test_check_native_passes() -> None:
    ds.check_native()


def test_kernel_pin_agrees_with_python() -> None:
    """The kernel version is pinned in three places -- Cargo.lock, the Rust
    crate, and Python. A bump that updates only some of them must fail loudly."""
    from deltaswamp import _native

    assert _native.kernel_version() == ds.EXPECTED_KERNEL_VERSION


def test_runtime_is_multithreaded() -> None:
    """UCCommitter bridges async UC calls with `block_in_place`, which panics
    outright on a current-thread runtime. This asserts the built wheel holds
    that invariant, not merely our Rust unit tests."""
    from deltaswamp import _native

    assert _native.runtime_is_multithreaded() is True


def test_native_version_matches_package() -> None:
    from deltaswamp import _native

    assert _native.native_version() == ds.__version__
