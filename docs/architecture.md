# Architecture

How deltaswamp is built, and which constraints forced each decision.

## Layers

```
identity.py      parse a reference   ->  TableRef
catalog/         resolve a TableRef  ->  ResolvedTable  (+ credential provider)
                 databricks | unity (OSS) | hive | glue | sharing | filesystem
credentials/     mint short-lived, per-table credentials
capability.py    the conformance matrix, as data
router.py        (table, operation)  ->  engine, or a refusal with a reason
engine/          kernel | deltars | sql | sharing | iceberg: the actual work
engine/metadata  metadata-only commits written by this library
predicate.py     one SQL predicate -> kernel file skipping + exact row filter
governance.py    grants, tags, lineage and the rest of Unity Catalog's API
_native          Rust: delta-kernel-rs through PyO3
connection.py    Connection: the entry point
table.py         Table: one table, every operation
```

Each engine wraps a library that already does its job: delta-kernel-rs,
delta-rs, the Databricks Statement Execution API, the `delta-sharing` client and
PyIceberg. deltaswamp's own code sits in the gaps: routing, credential
lifetimes, the kernel binding, metadata commits, predicate evaluation for the
kernel, warehouse staging flows and Unity Catalog's managed-table creation.

A read goes like this. `conn.table("main.sales.orders")` parses the name, asks
the catalog to resolve it and returns a `Table`. The first operation adds the
protocol from the Delta log, the router picks an engine, and the engine returns
an Arrow stream.

## Two engines, opposite gaps

delta-kernel checks feature support per operation, ignores writer-only features
when reading, and writes several features delta-rs cannot. delta-rs also
ignores writer-only features on read (deltalake 1.6.5 opens row-tracked,
liquid-clustered and in-commit-timestamp tables), but it refuses seven
reader-writer features the kernel reads: `catalogManaged` and its preview, type
widening and shredded variants (two spellings each), and `vacuumProtocolCheck`.
On the write side there are eleven writer features delta-rs reads but will not
write, and the kernel writes several of them.

The other way round, delta-rs wins outright. The kernel has no MERGE, no
`replaceWhere`, no schema evolution on write, no OPTIMIZE, Z-ORDER, VACUUM,
RESTORE, FSCK, CONVERT or manifest generation, and its log compaction is a stub.

So the kernel is the default reader and delta-rs the default for DML and
maintenance. Two rows break the pattern: for `checkConstraints` and
`generatedColumns`, delta-rs is the more capable engine.

## Resolution happens in two stages

The catalog answers first, before the log is opened. That is the only point
where a shallow clone can be caught: its add actions point at the source
table's files by absolute path, and once the log is open they look like the
clone's own. Views and materialized views have no files to read at all. The
capability manifest from `ListTables(include_manifest_capabilities=true)`
decides external eligibility, because a row filter removes an ordinary managed
table from credential vending and nothing in `table_type` or the protocol shows
it.

The log answers second. Reader and writer feature lists exist only there, so
the first operation on a `Table` opens a snapshot and merges the real protocol
into the resolved table. That step bypasses the router, since routing depends
on the features it is discovering.

A `catalogManaged` table has commits the catalog has ratified but not yet
published, and a cap on the version a reader may trust. Resolution fetches both
from the UC Delta API (`/api/2.1/unity-catalog/delta/v1`) and hands them to the
kernel. They are captured once, so a long-lived `Table` does not see later
commits made elsewhere until it is re-opened.

## Routing

The router asks each engine in the order `OPERATION_ENGINES` gives and takes
the first that accepts. Engines never raise for an unsupported combination;
they return a `Capability` with a reason. When nothing accepts, the router
raises with every reason it collected. `capabilities()` runs the same logic for
every operation without performing any.

The SQL warehouse sits late in most chains and is skipped unless the connection
opted in. `DROP FEATURE`, `REORG`, `CLONE`, `ANALYZE`, `REFRESH` and UniForm
metadata sync reach only the warehouse, because no open-source engine
implements them.

The shape of a request narrows the candidates. A predicate, a timestamp, a
schema merge, `OPTIMIZE FULL` or `CLUSTER BY AUTO` each maps to a
`supports_<shape>` flag, and an engine without the flag is skipped before it can
accept the call and fail halfway.

A Delta Sharing table has one way in, the sharing engine. So does an Iceberg
table, through the catalog's Iceberg REST endpoint.

## Credentials

Two clocks run independently. The catalog token is OAuth against
`/oidc/v1/token`, and the SDK refreshes it. The vended storage credential is
per-table, carries its own `expiration_time`, and must be re-vended. Databricks
publishes no TTL, so nothing here assumes an hour. Mixing the two up is how a
job dies after about an hour while authentication still looks fine.

Re-vending happens between operations. The kernel builds its object store once
per snapshot, so a single scan that streams past its credential's lifetime
fails. Refreshing inside Rust needs an `object_store::CredentialProvider` that
calls back into Python, which is not built. Until then, a scan that starts with
less than `KernelEngine.expiry_warning_seconds` of credential life raises
`CredentialExpiryWarning`. `plan_scan()` and `to_ray_dataset()` avoid the
problem, since each worker vends its own credential for its own slice.

Providers are picklable and credentials are not. `__getstate__` drops the live
client, the cached credential and the lock, so a worker receives configuration,
never a token.

Vending is per-table with no batch endpoint. Discovery therefore makes one
`ListTables` call per schema rather than one lookup per table.

Credentials are keyed by table and path prefix, not by bucket. Azure vends a
user-delegation SAS scoped to a path, so keying on the storage root would hand
one table another table's signature and get a `403 AuthenticationFailed`.

## Rust extension

`crates/native` wraps delta-kernel-rs through PyO3 and ships as an abi3 wheel:
one binary per platform covers Python 3.11 and later. Arrow crosses the boundary
only through the C Data Interface, which lets the extension share a process with
the `deltalake` wheel and its own copy of the kernel.

It exposes snapshot resolution (with the catalog's log tail, at a version or a
timestamp), scans with predicate-based file skipping, file listing, protocol,
metadata and domain metadata, the change data feed, appends and overwrites
through `FileSystemCommitter` or `UCCommitter`, publishing, checkpoints, a
put-if-absent commit of raw actions, and Unity Catalog's managed-table creation.

Five kernel constraints shape the code:

- **Multi-threaded runtime and executor.** `UCCommitter` bridges async catalog
  calls with `block_in_place`, which panics on a current-thread runtime. The
  default engine's `TokioBackgroundExecutor` also deadlocks inside the
  checkpoint writer, which waits on its own thread. Every engine here runs on
  `TokioMultiThreadExecutor` over one shared runtime.
- **Predicates only skip files.** The kernel turns off Parquet row filtering so
  row positions, and with them deletion vectors, stay valid. `predicate.py`
  applies the exact filter from the same parse. A conjunct the kernel cannot
  express is dropped only at positive polarity, never beneath a NOT.
- **No metadata changes through `UCCommitter` after version 0.** Schema
  evolution, column renames, feature toggles and constraint DDL on a
  catalog-managed table are closed to every external client, Spark included.
- **The Parquet writer lives on `DefaultEngine`**, not the `Engine` trait, so the
  snapshot holds the concrete type.
- **Unity Catalog flattens commit failures into one error.** The status is
  recovered from the message, and 409 and 429 become different exceptions:
  one means restage, the other means publish.

## Correctness invariants

Each of these would return wrong data rather than crash, so each is enforced in
code and covered by a test.

- Rows are never reordered before a deletion vector is applied. A DV is a
  positional mask over a file's rows in physical order.
- Deletion-vector concurrency is bounded by a constant, never by file count.
  Unbounded `spawn_blocking` exhausts the pool and deadlocks the runtime.
- The catalog's log tail must be contiguous. A gap means a stale or partial
  response, and it is reported as one.
- Staged commits are never reused across versions. A staged file encodes
  version-specific state and a unique `txnId`.
- Writes to tables read as Iceberg are refused. They would need
  `MSCK REPAIR TABLE ... SYNC METADATA`, which only Databricks can run.
- Partitioned kernel appends split each batch by partition value in Rust and
  write each group through its own write context.
- Metadata-only commits are put-if-absent. Losing the race recomputes against
  the new state and never overwrites the winner.
- A copy-on-write rewrite commits against the snapshot it read, so a concurrent
  writer causes a conflict, not a lost commit.
- Unknown feature names never raise, or the first table to adopt a new
  writer-only feature would become unreadable.

## Distributed reads and writes

A scan plan is a list of per-file splits pinned to one snapshot version. A
worker re-resolves that version and reads its files through a file-restricted
kernel scan, which applies deletion vectors, column mapping and partition values
exactly as a full scan does. Only the kernel engine plans; other engines read
on the driver.

Writes run the same way in reverse. Workers write Parquet and return add-action
metadata, which the driver commits in one transaction at a single version. The
metadata travels as Arrow IPC, not JSON, because the schema the kernel expects
is nested, follows the table's schema for statistics and grows under row
tracking. Re-deriving it here would drift from `Transaction::add_files_schema`.

`plan_write` decides on the driver whether the commit can succeed, before any
worker runs. It reads the full protocol: a legacy `minWriterVersion` implies
features it never names, and a check that reads only the named list would
accept writes it cannot perform.

## Extension points

Third-party catalogs register under the `deltaswamp.catalogs` entry-point group,
or load from a dotted `module:Class` path. The six built-in catalogs are listed
in `BUILTIN_CATALOGS` and resolve without installed metadata, which source
checkouts, vendored copies, zipapps and freezers lose. Entry points are checked
first, so a plugin can shadow a built-in name. A test keeps the map in step with
`pyproject.toml`.

A catalog is one classmethod plus the `Catalog` protocol:

```python
class MyCatalog:
    name = "mine"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs) -> "MyCatalog": ...

    def resolve(self, ref: TableRef) -> ResolvedTable: ...
    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]: ...
```

Engines and credential providers are structural protocols too. A new engine
needs `supports`, `scan` and whichever operations it claims.

## Version pinning

The kernel is pinned to 0.28.0 in `Cargo.lock`, a Rust constant and a Python
constant, and a test asserts all three agree. The kernel has shipped breaking
changes about every three weeks, so a bump must be deliberate.

It comes from git through `[patch.crates-io]`, not from crates.io. The Unity
Catalog crates exist only in the delta-kernel-rs repository and depend on
`delta_kernel` by path, and Cargo treats a path dependency as a different crate
from the registry one. Mixing them yields two kernels and two incompatible
`Committer` traits. The patch points at the v0.28.0 tag, so the code is the same.

## Not built

- Deletion-vector authoring. Kernel 0.28 has `update_deletion_vectors` only as
  an internal API, and it is not bound. DML on kernel-only tables is a
  whole-table rewrite bounded by `KernelEngine.rewrite_max_bytes`, refused on
  row-tracked tables; MERGE on them needs the warehouse.
- The change feed of a catalog-managed table. The kernel's `TableChanges`
  lists the log itself and would miss unpublished commits.
- Incremental reads through the kernel's `incremental_scan`. `Table.changes()`
  follows the change data feed instead.
- Credential refresh inside a single long read.
- Databricks server-side behavior: predictive optimization, auto compaction,
  row-level concurrency, UniForm metadata generation. The
  [feature map](features.md) shows how each is reached.
