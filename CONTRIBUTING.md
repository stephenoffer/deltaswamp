# Contributing

## Getting set up

```bash
make venv      # .venv plus the dev dependencies
make build     # compile and install the native extension
make check     # everything CI runs
```

You need a Rust toolchain. `rust-toolchain.toml` pins 1.88, which is
delta-kernel's MSRV. Python 3.11 is the floor, because `pyo3-arrow` needs
`Py_buffer` and that only entered Python's stable ABI in 3.11.

`make test-unit` skips the extension entirely, so pure-Python work needs no Rust.

## The rule that matters most

**A capability that says yes must work.** `capabilities()` is the product: it
reports, in advance, what can and cannot be done with a table and why. A verdict
of `ok=True` followed by a runtime failure is the worst bug this library can
have, worse than a missing feature.

Two tests enforce it, in `tests/unit/test_router.py`:

- every operation an engine is routed must exist as a method on that engine;
- every operation in the matrix must be reachable from `Table` or `Connection`.

If you add an operation, those tests tell you what you forgot.

The matching rule for refusals: every one carries a reason, and a remedy where
one exists. `raise UnreachableTableError("do the thing", "because X", "try Y")`,
never a bare exception.

## Where behavior is recorded

`python/deltaswamp/capability.py` is data, not prose:

| Table | Contents |
|---|---|
| `FEATURE_SUPPORT` | 34 table features x {kernel, delta-rs} x {read, write} |
| `OPERATION_ENGINES` | 33 operations -> engines in preference order |
| `PROPERTY_SUPPORT` | 30 properties x {create, set} x engine |

`tests/integration/test_properties.py` re-probes the installed engines against
`PROPERTY_SUPPORT`, so an engine upgrade that changes behavior fails loudly
instead of drifting. `docs/conformance.md` is asserted against the matrices by
`tests/unit/test_capability.py`, so it cannot go stale either.

## Verify by behavior, not by diff

This is worth stating because it has bitten repeatedly. `ruff format` reformats
as it goes, so a scripted edit whose anchor was valid a moment ago may silently
match nothing. Several changes here looked applied, read correctly in the diff,
and did nothing:

- `cluster_by` was dropped between the router and the engine, producing an
  unclustered table with no error;
- a test fixture's setup call was never inserted, so the test asserted against
  the wrong kind of table.

Both were caught by re-running the behavior, not by reading the change. Assert
the outcome.

## Bumping delta-kernel

Deliberate, never routine. Kernel has shipped breaking changes roughly every
three weeks, and the version is pinned in three places that a test asserts agree:
`Cargo.lock`, `KERNEL_VERSION` in `crates/native/src/lib.rs`, and
`EXPECTED_KERNEL_VERSION` in `python/deltaswamp/__init__.py`.

Dependabot is configured to leave it alone. To bump it: update all three, read
the kernel CHANGELOG for breaking changes, and re-run the cross-engine tests --
the ones where delta-rs writes and the kernel reads, and the reverse.

Note the kernel resolves from git rather than crates.io, via `[patch.crates-io]`.
That is forced, not preferred: the Unity Catalog crates live only inside the
delta-kernel-rs repository and depend on the kernel by path, and Cargo treats a
path dependency as a different crate from the crates.io one. Mixing them yields
two kernels and two incompatible `Committer` traits.

## Correctness rules

`docs/architecture.md` lists the invariants that are enforced in code rather than
documented as advice. Each one is a silent-wrong-data bug rather than a crash,
which is why they are tests. The sharpest:

- never reorder rows before applying a deletion vector;
- bound deletion-vector concurrency by a constant, never by file count;
- a 409 from the catalog means restage at the next version; a 429 means publish.
  Retrying a 429 with backoff wedges the table.

## Adding a catalog

Catalogs load from the `deltaswamp.catalogs` entry-point group, so a third party
can ship one out of tree. The contract is `from_uri(uri, **kwargs)` plus the
`Catalog` protocol. See `python/deltaswamp/catalog/registry.py`.

## Tests

| Tier | Needs | Command |
|---|---|---|
| Unit | nothing | `make test-unit` |
| Integration | the built extension | `make test` |
| Live Databricks | a workspace and a PAT | `make test-live` |

The live suite is opt-in and creates and drops tables in a schema you nominate.
`docs/testing.md` has the setup, and `python -m tests.live_preflight` checks it
in a few seconds.

`tests/fake_uc.py` is a Unity Catalog server speaking the real `/delta/v1`
protocol. It exists because the catalog-managed path is the headline capability
and would otherwise be untestable without an account -- and because it can
produce a 409 or a 429 on demand, which a live server will not do to order.
