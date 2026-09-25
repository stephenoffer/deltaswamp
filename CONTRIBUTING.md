# Contributing

## Setup

```bash
make venv      # .venv plus the dev dependencies
make build     # compile and install the native extension
make check     # everything CI runs
```

You need a Rust toolchain; `rust-toolchain.toml` pins 1.88, delta-kernel's
MSRV. Python 3.11 is the floor because `pyo3-arrow` needs `Py_buffer` in the
stable ABI. `make test-unit` needs no Rust at all.

## Capabilities must not lie

`capabilities()` says in advance what can be done with a table. A verdict of
`ok=True` followed by a runtime failure is the worst bug this library can have.
Two tests in `tests/unit/test_router.py` guard it: every operation an engine is
routed must exist on that engine, and every operation in the matrix must be
reachable from `Table` or `Connection`. Add an operation and they tell you what
is missing.

Every refusal carries a reason, and a remedy where one exists:
`raise UnreachableTableError("do the thing", "because X", "try Y")`.

## Behavior lives in data

`python/deltaswamp/capability.py` holds the feature matrix (`FEATURE_SUPPORT`)
and the routing table (`OPERATION_ENGINES`); `python/deltaswamp/properties.py`
holds `PROPERTY_SUPPORT`.
`tests/integration/test_properties.py` re-probes the installed engines against
the property table, so an engine upgrade that changes behavior fails a test.
`tests/unit/test_capability.py` checks `docs/conformance.md` against the
matrices.

Test the outcome, not the call. A change can read correctly in the diff and do
nothing, so assert on the data the operation leaves behind.

## Bumping delta-kernel

The version is pinned in `Cargo.lock`, `KERNEL_VERSION` in
`crates/native/src/lib.rs` and `EXPECTED_KERNEL_VERSION` in
`python/deltaswamp/__init__.py`, and a test asserts they agree. Dependabot
leaves it alone. To bump it, update all three, read the kernel changelog for
breaking changes, and re-run the cross-engine tests (delta-rs writes and the
kernel reads, and the reverse). [Architecture](docs/architecture.md#version-pinning)
explains why the kernel comes from git.

## Correctness rules

[Architecture](docs/architecture.md#correctness-invariants) lists the invariants
that would return wrong data if broken. The three most often at risk:

- never reorder rows before applying a deletion vector;
- bound deletion-vector concurrency by a constant, never by file count;
- a 409 from the catalog means restage at the next version, and a 429 means
  publish. Retrying a 429 with backoff wedges the table.

## Adding a catalog

Catalogs load from the `deltaswamp.catalogs` entry-point group, so they can ship
out of tree. Implement `from_uri(uri, **kwargs)` and the `Catalog` protocol; see
`python/deltaswamp/catalog/registry.py`.

## Tests

[Testing](docs/testing.md) covers the three tiers. `tests/fake_uc.py` is a Unity
Catalog server that speaks the real `/delta/v1` protocol, so the catalog-managed
path is testable without an account, and it can return a 409 or 429 on demand.
