# Architecture

Why this library is shaped the way it is, and which constraints forced each
decision.

- [The problem](#the-problem)
- [Layers](#layers)
- [The finding that drives everything](#the-finding-that-drives-everything)
- [Resolution happens in two stages](#resolution-happens-in-two-stages)
- [Routing](#routing)
- [Credentials](#credentials)
- [Rust extension](#rust-extension)
- [Correctness invariants](#correctness-invariants)
- [Extension points](#extension-points)
- [Version pinning](#version-pinning)
- [Deliberately not built](#deliberately-not-built)

## The problem

Delta tables live behind a dozen different access paths, and which one you need
depends on facts about the table you often cannot see until you try. Path-based
tables go through `deltalake`. Some Unity Catalog tables work through an
undocumented `uc://` scheme. Managed tables need a SQL warehouse. Anything
carrying `catalogManaged` needs the catalog itself to ratify commits. Hive
Metastore wants a Thrift client, Glue wants boto3, managed Iceberg wants
PyIceberg. Each brings separate authentication, a separate credential lifetime,
and its own opaque unsupported-feature error.

The job is to collapse that into one object without lying about what it can do.

## Layers

```
identity.py      parse a reference   ->  TableRef
catalog/         resolve a TableRef  ->  ResolvedTable  (+ credential provider)
                 databricks | unity (OSS) | hive | glue | sharing | filesystem
governance.py    what Unity Catalog says about a table: grants, tags, lineage
credentials/     mint short-lived, per-table credentials
capability.py    the conformance matrix, as data
router.py        (table, operation)  ->  engine, or a refusal with a reason
engine/          kernel | deltars | sql | sharing | iceberg   do the actual work
engine/metadata  metadata-only commits written by this library
predicate.py     one SQL predicate -> kernel file skipping + exact row filter
_native          Rust: delta-kernel-rs through PyO3
table.py         Connection and Table              the only public surface
```

Each engine wraps a library that already does its job well: delta-kernel-rs,
delta-rs, the Databricks Statement Execution API, the `delta-sharing` client,
PyIceberg. deltaswamp's own code sits in the gaps between them. That covers
routing, credential lifetimes, the kernel binding, metadata commits, predicate
evaluation for the kernel, the staging flows that let a warehouse write, and
Unity Catalog's managed-table creation.

A read flows down and back up: `conn.table("main.sales.orders")` parses the
name, asks the catalog to resolve it, and returns a `Table`. The first operation
enriches that resolution with the protocol from the log, the router picks an
engine, and the engine produces an Arrow stream.

`capability.py` is data, not code, and deliberately so. A support matrix written
in prose drifts from reality within a release; here every claim is a row a test
asserts against.

## The finding that drives everything

The two open-source engines fail in opposite directions, so this library
composes them per operation instead of choosing one.

delta-kernel evaluates feature support per operation and ignores writer-only
features when reading. delta-rs does a flat set-difference against a hardcoded
list, and its write check calls its read check first. The practical consequence
is large: there are 19 table features the kernel reads happily and delta-rs
refuses to even open, including every `catalogManaged` table and anything
carrying `vacuumProtocolCheck`, which blocks reads because it is a ReaderWriter
feature despite the name.

Run it the other way and delta-rs wins outright. The kernel has no MERGE, no
`replaceWhere`, no schema evolution on write, no OPTIMIZE, no Z-ORDER, no
VACUUM, no RESTORE, no FSCK, no CONVERT, no manifest generation, and its log
compaction is a stub that does nothing.

So the kernel is the default read engine and delta-rs the default engine for DML
and maintenance. Neither is "better"; the routing table encodes which one is
right for each operation, and two rows run against the general grain because
`checkConstraints` and `generatedColumns` are cases where delta-rs is more
capable.

## Resolution happens in two stages

Some decisions have to be made before the Delta log is opened, and others cannot
be made until afterward.

The catalog answers first. A shallow clone must be caught here, because its add
actions reference the source table's files by absolute path and the kernel
resolves those to absolute URLs before a connector sees them; once the log is
open, borrowed files are indistinguishable from owned ones. Views and
materialized views have no readable file surface at all. Then the decisive one:
the capability manifest from `ListTables(include_manifest_capabilities=true)` is
authoritative: a row filter silently removes an otherwise ordinary managed Delta
table from the set eligible for credential vending, and nothing in `table_type`
or the protocol reveals that.

The log answers second. Reader and writer feature lists exist only there, so the
first operation on a `Table` opens a snapshot, merges the real protocol into the
resolved table, and everything afterward routes on the complete picture. That
enrichment deliberately bypasses the router, since routing depends on the
features it is trying to discover.

For a `catalogManaged` table there is no log to list in the usual sense. The
catalog holds commits that are ratified but not yet published, and it caps the
version a reader may trust, so resolution also fetches the commit tail from the
UC Delta API at `/api/2.1/unity-catalog/delta/v1` and hands both to the kernel.

## Routing

The router asks each candidate engine, in the preference order the matrix
defines, and takes the first that says yes. An engine never raises for an
unsupported combination. It returns a `Capability` carrying the reason. When
nothing can serve the request, the router raises with every reason it collected.

The SQL warehouse sits late in most chains and is skipped unless the
connection opted in. `DROP FEATURE`, `REORG`, `CLONE`, `ANALYZE`, `REFRESH` and
UniForm metadata sync reach it alone, because no open-source engine implements
them at all.

A request's shape narrows the candidates too. A predicate, a timestamp, a schema
merge, `OPTIMIZE FULL` or `CLUSTER BY AUTO` each names a `supports_<shape>`
flag. An engine without the flag is skipped before it can accept the call and
fail halfway.

Two kinds of table have exactly one way in. A table reached through a Delta
Sharing profile is served by the sharing engine, whose verdict is final. An
Iceberg table is served by the Iceberg engine through the catalog's Iceberg
REST endpoint.

`capabilities()` runs the same logic over every operation without performing
any, which is what makes checking cheaper than catching.

## Credentials

Two clocks run independently and conflating them is the classic production
failure: a job dies after roughly an hour while authentication still looks fine.

The catalog token is OAuth against `/oidc/v1/token` and the SDK refreshes it.
The vended storage credential is per-table, carries its own `expiration_time`,
and has to be re-vended. Databricks publishes no TTL, so that field is the only
authority and nothing here hardcodes an hour.

The boundary worth knowing: re-vending happens *between* operations. Every call
resolves a snapshot and mints credentials when the cached ones are near expiry,
but the kernel builds its object store once per snapshot, so one scan that
streams past the credential's lifetime will fail. Pushing refresh down into Rust
needs an `object_store::CredentialProvider` that calls back into the Python
provider, and that is not built yet.

Until it is, the condition is at least announced rather than discovered: a scan
that starts with less than `KernelEngine.expiry_warning_seconds` of credential
life raises `CredentialExpiryWarning` naming the remedy, because the failure it
precedes surfaces as a bare 403 from the storage layer several frames down.
`plan_scan()` and `to_ray_dataset()` sidestep it entirely -- each worker vends
its own credential for its own slice, so no single read has to outlive one.

Providers are picklable and credentials are not. A provider holds configuration
and mints on demand, so a distributed worker receives no secret and re-vends
when its own copy expires. `__getstate__` drops the live client, the cached
credential and the lock. Sending a credential instead would put a token in task
payloads and logs and freeze it at submission time, which is exactly what breaks
long jobs.

Vending is strictly per-table with no batch endpoint, so N tables means N calls.
Discovery therefore goes through one `ListTables` call per schema instead of N
per-table lookups.

Storage credentials are keyed by table identity and path prefix, not by bucket.
Azure vends a user-delegation SAS scoped to a path, so a registry keyed on the
storage root hands the second table in a container the first table's
signature and earns a `403 AuthenticationFailed`.

## Rust extension

`crates/native` wraps delta-kernel-rs through PyO3 and ships as an abi3 wheel,
so one binary per platform covers every Python from 3.11 up.

Arrow crosses the boundary only through the Arrow C Data Interface, never as
shared Rust types. That is what lets this extension coexist in one process with
the `deltalake` wheel, which links its own separate build of the kernel.

What it exposes: snapshot resolution with the catalog's log tail and maximum
ratified version (at a version or a timestamp), scans that return an Arrow
stream with predicate-based file skipping, file listing, the raw protocol,
metadata and domain metadata, the change data feed, appends and overwrites
(partitioned or not) committed through either `FileSystemCommitter` or
`UCCommitter`, publishing staged commits, checkpoints, an atomic put-if-absent
commit of raw actions for metadata-only changes, and the helpers of Unity
Catalog's managed-table creation.

Five constraints from the kernel's own source shape the code:

The runtime must be multi-threaded, and so must the engine's executor.
`UCCommitter` bridges its async catalog calls with `block_in_place`, which
panics outright on a current-thread runtime and needs a Tokio handle in scope.
So commits run inside the shared runtime. The default engine's
`TokioBackgroundExecutor` runs all I/O on one current-thread runtime, and the
checkpoint writer pulls log reads from inside a future on that same thread, so
it deadlocks every time. Every engine here is built on `TokioMultiThreadExecutor`
over the shared runtime instead, which turns the nested wait into
`block_in_place` on another worker. That is what makes checkpointing a
catalog-managed table possible.

Predicates only skip files. The kernel turns Parquet row filtering off so that
row positions, and so deletion vectors, stay valid. The exact filter is applied
in Python from the same parse (`predicate.py`). A predicate the kernel cannot
express is weakened safely: a conjunct is dropped only at positive polarity,
never beneath a NOT.

`UCCommitter` rejects protocol, metadata and clustering changes at version 1 and
above. Schema evolution on a catalog-managed table has to go through the
catalog's own APIs, and column rename, feature toggles and constraint DDL are
unavailable to external clients entirely, Spark included.

The Parquet writer lives on the concrete `DefaultEngine` rather than the
`Engine` trait, so the snapshot holds the concrete type.

Unity Catalog collapses commit failures into a generic error, so the status code
is recovered from the message. A 409 and a 429 become distinct Python exception
types, because one means restage and the other means publish.

## Correctness invariants

Each of these is a silent-wrong-data bug rather than a crash, which is why they
are enforced in code rather than documented as advice.

Rows are never reordered before a deletion vector is applied. A DV is a
positional keep-mask over a file's rows in physical order, so repartitioning,
limit pushdown or pre-filtering ahead of it returns the right *number* of rows
made of the wrong records.

Deletion-vector concurrency is bounded by a constant, never by file count.
Unbounded `spawn_blocking` exhausts the pool and deadlocks the runtime.

The catalog's log tail must be contiguous. A gap means the response was stale or
partial, so we say so, instead of failing deep inside log replay.

Staged commits are never reused across versions. A staged file encodes
version-dependent state and a `txnId` that must be unique per commit, so losing
a race means writing a fresh UUID at the next version.

Writes to Iceberg-reads tables are refused. They would require
`MSCK REPAIR TABLE ... SYNC METADATA` afterward, which only Databricks can run,
so writing anyway leaves the Iceberg view silently stale.

Partitioned appends on the kernel path split each batch by partition value in
Rust and write each group through its own partitioned write context. A single
unpartitioned context would put every row in the table root with no partition
values, which reads back as wrong data rather than an error.

Metadata-only commits are put-if-absent. The change is a pure function of the
protocol and metadata it was computed from, so losing the race means
recomputing against the new state, never overwriting the winner. `commitInfo`
comes first and carries a strictly increasing in-commit timestamp where the
table uses them.

A copy-on-write rewrite commits against the snapshot it read. Kernel DELETE,
UPDATE and replaceWhere read, transform and commit through one snapshot, so a
concurrent writer produces a conflict (a 409 from the catalog, or a lost put),
never a silently dropped commit.

Unknown feature names never raise. The kernel tolerates unknown writer-only
features when reading and so must this library, or the first table to adopt
something new becomes unreadable.

## Extension points

Catalogs are discovered through the `deltaswamp.catalogs` entry-point group, so
a third party can ship `deltaswamp-gravitino` out of tree and have it work with
no change here. A dotted `module:Class` path is accepted too, for a class that
is not installed as a distribution.

Entry points are the extension mechanism, not how deltaswamp finds its own
modules. The six built-in catalogs are listed in `BUILTIN_CATALOGS` and resolve
without consulting installed metadata, because a source checkout, a vendored
copy, a zipapp and several freezers all lose that metadata, and losing it used
to leave every catalog unregistered and the library unusable. Lookups consult
entry points first, so a plugin can still shadow a built-in name deliberately;
the built-in map is the fallback, and a test asserts it agrees with
`pyproject.toml`.

The contract is one classmethod plus the `Catalog` protocol:

```python
class MyCatalog:
    name = "mine"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs) -> "MyCatalog": ...

    def resolve(self, ref: TableRef) -> ResolvedTable: ...
    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]: ...
```

PyIceberg solves the same problem with configuration alone and people work
around it; entry points plus a dotted-path override covers both cases.

Engines and credential providers are structural protocols too, so a new engine
needs `supports`, `scan` and whichever operations it claims.

## Version pinning

The kernel is pinned exactly at 0.28.0 in three places: `Cargo.lock`, a Rust
constant, and a Python constant. A test asserts all three agree. Kernel has
shipped breaking changes on a roughly three-week cadence, so a bump has to be a
deliberate act, never something that slips through.

The kernel also comes from git, not crates.io, through
`[patch.crates-io]`, and that is forced on us, not preferred. The Unity Catalog
crates exist only inside the delta-kernel-rs repository and depend on
`delta_kernel` by path. Cargo treats a path dependency as a different crate
instance from the crates.io one, so mixing them produces two copies of the
kernel and two incompatible `Committer` traits. The patch points at the same
v0.28.0 tag, so the code is identical and only the source differs.

## Deliberately not built

Deletion-vector authoring is blocked upstream, not merely unbuilt.
`Transaction::update_deletion_vectors` and `ack_row_tracking_preservation` are
both absent from delta-kernel 0.28. So DML on tables only the kernel can write
is a whole-table rewrite bounded by `KernelEngine.rewrite_max_bytes`. It is
refused on row-tracked tables, and MERGE on such tables needs the warehouse.

The change feed of a catalog-managed table: the kernel's `TableChanges` lists
the log itself and takes no catalog commit tail, so it would miss unpublished
commits.

Incremental reads over the kernel's `incremental_scan`, i.e. reading only the
files added since a version without a change feed. `Table.changes()` follows
the change data feed instead.

Distributed planning is built for the kernel engine only. delta-rs, sharing,
Iceberg and the warehouse read on the driver. A plan is a list of per-file
splits pinned to one snapshot version. A worker re-resolves that version and
reads only its files through a file-restricted kernel scan, which applies
deletion vectors, column mapping and partition values exactly as the full scan
does.

Writes take the same shape in reverse, and the kernel's own write API is
already built for it: a connector writes Parquet and hands back add-action
metadata, which the kernel commits. Splitting that across machines needs the
metadata to travel, so it moves as Arrow IPC rather than JSON -- the schema
kernel expects is nested, follows the table's own schema for statistics, and
gains columns under row tracking, so re-deriving it here would drift from
`Transaction::add_files_schema` silently. Every fragment joins one transaction,
so a distributed write appears at a single version.

The ordering matters more than the mechanism. `plan_write` settles on the driver
whether the commit can succeed before any worker runs, because the way a
distributed Delta write normally fails is to find out at commit time, after the
compute is spent, leaving orphaned Parquet that nobody owns. That check reads
the protocol in full: a legacy `minWriterVersion` implies features it never
names, and a connector that reads only the named list sees a featureless table
and accepts a write it cannot perform.

Credential refresh inside a single long read.

Catalog-managed state is captured once at resolution. `log_tail` and
`max_catalog_version` come from the catalog when the table is resolved, so a
long-lived `Table` will not see commits made after that point by other writers.
Re-open the table to pick them up.

Server-side Databricks behaviour stays server-side: predictive optimization,
auto compaction, row-level concurrency, UniForm metadata generation.
[ecosystem-audit.md](ecosystem-audit.md) maps each of these to the route that
reaches it.
