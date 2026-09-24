# Usage guide

Everything the public API does, with the limits stated where they exist.

- [Install](#install)
- [Connecting](#connecting)
- [Opening a table](#opening-a-table)
- [Reading](#reading)
- [Writing](#writing)
- [Changing rows](#changing-rows)
- [Maintenance](#maintenance)
- [Schema and table features](#schema-and-table-features)
- [Metadata](#metadata)
- [Asking what is possible](#asking-what-is-possible)
- [Errors](#errors)
- [Credentials](#credentials)
- [SQL fallback](#sql-fallback)
- [Known limits](#known-limits)

## Install

```bash
pip install deltaswamp
```

The base install pulls in `deltalake` and `databricks-sdk` and nothing else.
Arrow leaves this library through the PyCapsule interface, so pyarrow is
optional even though Arrow is the native currency.

| Extra | Adds | You need it for |
|---|---|---|
| `pyarrow` | pyarrow | `to_arrow`, `to_pandas`, `to_pyarrow_dataset` |
| `pandas` | pandas | `to_pandas` |
| `polars` | Polars | `to_polars` |
| `sql` | `databricks-sql-connector` | the SQL warehouse fallback |
| `hms` | `pymetastore` | Hive Metastore catalogs |
| `glue` | boto3 | AWS Glue catalogs |
| `ossuc` | `unitycatalog-client` | open-source Unity Catalog |
| `iceberg` | PyIceberg | managed Iceberg tables |
| `sharing` | `delta-sharing` | Delta Sharing |
| `ray` | Ray Data | `to_ray_dataset` |

`pip install 'deltaswamp[pyarrow,sql]'` combines them. Note the absence of
`databricks-connect`: it pins `requires-python == 3.12.*` and conflicts with
pyspark, so it is deliberately not reachable from any extra here.

## Connecting

`connect()` with no arguments talks to Databricks and resolves authentication
through the SDK's normal precedence: explicit arguments, then environment, then
`~/.databrickscfg`, then cloud-native sources. This library never reimplements
that logic. `databricks.sdk.core.Config` already handles PAT, OAuth U2M and M2M,
Azure CLI, MSI and service principals, GCP service accounts, GitHub OIDC, and
in-cluster notebook auth, so the arguments are passed straight through.

```python
import deltaswamp as ds

conn = ds.connect()  # environment or ~/.databrickscfg
conn = ds.connect(profile="prod")  # a named profile
conn = ds.connect(host="https://adb-123.4.azuredatabricks.net", config=my_config)
```

Prefer OAuth M2M for anything long-running. A PAT cannot be refreshed, so a job
outliving its token simply stops.

Other catalogs are selected by URI scheme:

```python
conn = ds.connect("uc://http://localhost:8080")  # open-source Unity Catalog
conn = ds.connect("hms://thrift://metastore:9083")  # Hive Metastore
conn = ds.connect("glue://")  # AWS Glue, ambient region
conn = ds.connect("glue://123456789012")  # a specific Glue catalog id
```

Useful keyword arguments:

| Argument | Effect |
|---|---|
| `allow_sql_fallback=True` | permits routing through a SQL warehouse; off by default |
| `warehouse_id="..."` | which warehouse the fallback uses |
| `storage_options={...}` | extra object-store settings, merged under vended credentials |
| `default_catalog`, `default_schema` | let you write `conn.table("orders")` |
| `catalog=MyCatalog()` | supply a catalog object directly and skip URI dispatch |

Run `conn.preflight()` once against a new workspace. It returns a list of
problems, empty when ready, and it checks the settings that block everything
else: external data access on the metastore, which is off by default and needs
an account admin.

## Opening a table

```python
t = conn.table("main.sales.orders")  # three-level name
t = conn.table("sales.orders")  # needs default_catalog
t = conn.table("orders")  # needs default_catalog and default_schema
t = conn.table("main.sales.orders", version=12)

t = conn.open_table("s3://bucket/tables/orders")  # straight to storage
```

Names may be backtick-quoted, so a dot inside an identifier works:
``conn.table("main.`my.schema`.orders")`` is three parts, not four.

These reference forms all resolve:

| Form | Goes to |
|---|---|
| `catalog.schema.table` | the connection's catalog |
| `uc://catalog.schema.table` | Unity Catalog |
| `hms://host:9083/db/table` | Hive Metastore |
| `glue://db.table`, `glue://<id>/db.table` | Glue |
| `s3://`, `gs://`, `abfss://`, `file://`, `/local/path` | storage directly |

`dbfs:/` and `/mnt/...` paths parse and are then refused with an explanation.
They are unreachable from outside Databricks, and a vague missing-file error
would send you hunting in the wrong place.

## Reading

`scan()` is the primitive. It returns an Arrow stream that exports
`__arrow_c_stream__`, so Polars, DuckDB, pandas and pyarrow can all consume it
without this library depending on any of them.

```python
stream = t.scan()
stream = t.scan(columns=["id", "amount"])
stream = t.scan(predicate="amount > 100")  # delta-rs engine only
stream = t.scan(version=3)
stream = t.scan(timestamp="2026-01-01T00:00:00Z")  # delta-rs engine only
```

Convenience wrappers sit on top:

```python
t.to_arrow()  # pyarrow.Table
t.to_pandas()
t.to_polars()
t.to_pyarrow_dataset()
t.head(10)
t.count()
```

Predicates and timestamp travel currently route through delta-rs. Ask for either
on a table only the kernel can open and you get a refusal naming the reason
instead of a silent full scan. Version-based travel works on both engines.

Change feeds come from `cdf()`:

```python
for batch in t.cdf(starting_version=5, ending_version=9):
    ...
```

Two guards fire before any CDF read. A table without
`delta.enableChangeDataFeed` gets a refusal, because enabling it is not
retroactive and the range you asked for was never recorded. Column mapping earns
one too, since CDF does not support it. A third guard catches the case where
`delta.deletedFileRetentionDuration` is shorter than
`delta.logRetentionDuration`, which lets data files be vacuumed while their
commits survive; the read would otherwise die on a missing file deep in the
reader.

## Writing

```python
t.append(df)  # anything Arrow-shaped
t.overwrite(df)
t.overwrite(df, predicate="region = 'eu'")  # replaceWhere
```

`df` can be a pyarrow Table or RecordBatchReader, a Polars DataFrame, a pandas
DataFrame, or anything else exporting the Arrow PyCapsule interface.

Extra keyword arguments pass through to the engine, so delta-rs options such as
`partition_by=` and `schema_mode="merge"` work on `append`.

Creating a table:

```python
import pyarrow as pa

schema = pa.schema([("id", pa.int64()), ("city", pa.string())])
t = conn.create_table("s3://bucket/tables/new", schema)
t = conn.create_table("main.sales.new", schema, location="s3://bucket/tables/new")
```

A catalog name without `location=` will not go through. Creating a *managed* table means
asking the catalog to allocate storage through its staging-table API and then
finalising through its create-table API, and that flow is not built yet. The
error says so instead of writing something into a location nobody asked for.

## Changing rows

```python
t.delete("id = 42")
t.delete()  # every row
t.update({"status": "'archived'"}, predicate="age > 365")
t.merge(source, predicate="target.id = source.id")
```

`merge` returns delta-rs's `TableMerger`, so the clause API is theirs:

```python
(
    t.merge(source, "t.id = s.id", source_alias="s", target_alias="t")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute()
)
```

All three are copy-on-write on delta-rs, which rewrites whole Parquet files
rather than emitting deletion vectors. On a large table that is real write
amplification, and worth knowing before you run a narrow `DELETE` against
billions of rows.

## Maintenance

```python
t.optimize()  # bin-packing compaction
t.z_order(["customer_id"])
t.vacuum(retention_hours=168, dry_run=True)
t.restore(3)  # or a datetime
t.repair()  # FSCK
```

All of these run through delta-rs; the kernel implements none of them. Three
refuse instead of misbehaving. `vacuum` on a shallow clone stops early, because
the clone borrows the source's files and vacuuming it risks deleting data the
source still owns. `restore` on a table with deletion vectors stops too, because
delta-rs reports success and leaves the rows deleted, so the table reads wrong
afterward. Databricks forbids `OPTIMIZE`, `VACUUM` and `ANALYZE` on UC managed
tables from every external client, so those route to SQL or fail.

## Schema and table features

```python
from deltalake import Schema
import pyarrow as pa

field = Schema.from_arrow(pa.schema([("region", pa.string())])).fields[0]
t.add_column([field])
t.set_properties({"delta.deletedFileRetentionDuration": "interval 30 days"})
t.add_feature("deletionVectors")
t.add_constraint({"id_positive": "id > 0"})
```

These run through delta-rs. Four more exist only in Databricks, so they need the
SQL fallback and refuse without it: `drop_column`, `rename_column`,
`drop_feature` and `reorg`.

```python
t.drop_column("region")
t.rename_column("city", "town")
t.drop_feature("deletionVectors", truncate_history=True)
t.reorg(purge=True)
t.clone("main.sales.orders_copy", shallow=True)
```

Log upkeep:

```python
t.checkpoint()  # write a checkpoint
t.compact_logs()  # aggregate commits; a no-op on a single-commit table
t.generate()  # symlink manifests, for Presto and Athena
t.publish()  # staged catalog commits -> _delta_log
```

Converting an existing Parquet directory in place:

```python
t = conn.convert_to_delta("s3://bucket/parquet-dir")
```

## Metadata

```python
t.schema()  # Arrow schema
t.protocol()  # (min_reader_version, min_writer_version)
t.features()  # frozenset of table feature names, verbatim
t.properties()  # delta.* table properties
t.detail()  # version, location, protocol, properties
t.history(limit=10)
t.version
t.location
t.table_type  # MANAGED, EXTERNAL, VIEW, ...
t.securable_kind  # the finer-grained UC discriminator
t.is_catalog_managed
```

Metadata reflects the table as of the last call. Any write or ALTER through this
object refreshes it, so `properties()` after `set_properties()` shows the new
value rather than the old one.

`features()` returns names exactly as the log holds them, including ones this
release does not recognize. That is deliberate: an unknown writer-only feature
must not block a read, or the first table adopting something new becomes
unreadable.

## Asking what is possible

`capabilities()` reports every operation, whether it can be served, by which
engine, and why not when it cannot.

```python
>>> t.capabilities()[ds.Operation.DELETE]
Capability(ok=False, engine=None,
  reason="table has row filters; UC credential vending refuses it",
  remedy="ds.connect(..., allow_sql_fallback=True)")

>>> t.can("scan")
Capability(ok=True, engine=Engine.KERNEL, ...)

>>> if t.can("merge"):
...     t.merge(source, "t.id = s.id")
```

`Capability` is truthy when `ok`, so it reads naturally in a condition. Every
refusal carries a reason, and a remedy whenever one exists. Checking first is
cheaper than catching, since the verdict comes from catalog metadata and the
protocol, not from attempting the work.

## Errors

All inherit from `DeltaSwampError`.

| Error | Means |
|---|---|
| `InvalidReferenceError` | the reference could not be parsed or resolved |
| `UnreachableTableError` | no available engine can serve the request |
| `FallbackRequiredError` | only the SQL fallback could serve it, and it is off |
| `CredentialError` | vending or refresh failed |
| `PreflightError` | a workspace prerequisite is not satisfied |
| `CommitConflictError` | another writer took that version first |
| `BackfillRequiredError` | the catalog wants staged commits published |
| `CorruptTableError` | on-disk state failed a correctness check |

The last two are separate on purpose. A conflict means re-read the snapshot,
recompute, then stage again at the next version. A backfill demand is
backpressure, not a rate limit: retrying it with exponential backoff and no
publish will wedge the table.

```python
from deltaswamp import BackfillRequiredError, CommitConflictError

try:
    t.append(df)
except CommitConflictError:
    t = conn.table(name)  # fresh snapshot, then redo the work
    t.append(df)
except BackfillRequiredError:
    ...  # publish staged commits, do not just retry
```

## Credentials

Unity Catalog vends short-lived, per-table credentials. Two independent clocks
are involved, and conflating them causes the classic failure where a job dies
after an hour with authentication that still looks healthy.

The catalog token, OAuth against `/oidc/v1/token`, is refreshed by the SDK. The
vended storage credential carries its own `expiration_time` and has to be
re-vended. Databricks publishes no TTL for it, so that field is the only
authority and nothing here assumes an hour.

Re-vending happens between operations, not during one. Each call resolves a
fresh snapshot and mints credentials if the cached ones are near expiry, so a
long-lived `Table` keeps working. A *single* scan that streams for longer than
the credential's lifetime can still fail, because the kernel builds its object
store once when the snapshot opens. Split very long reads, or retry them.

```python
creds = t.credentials()  # or credentials(write=True)
creds.expires_at  # epoch seconds, or None
creds.expires_within(300)  # True if it dies within five minutes
creds.cloud  # Cloud.AWS, AZURE, GCP, R2
creds.redacted()  # safe to log
```

`repr` never renders secret material, so a credential in a traceback leaks
nothing.

Credential *providers* are picklable; credentials are not, by design. A provider
holds configuration and mints on demand, so shipping one to a distributed worker
sends no secret and the worker re-vends when its own copy expires. Sending a
credential instead would put a token in task payloads and logs and freeze it at
submission time.

Two cloud-specific notes. Azure user-delegation SAS is scoped to a *path*, so
credentials are keyed by table identity, not by bucket. Sharing one
across sibling tables in a container produces `403 AuthenticationFailed`. Azure
also always gets an explicit endpoint, because account-name inference happens to
work on `*.blob.core.windows.net` and silently breaks Azurite, private-link DNS
and the sovereign clouds.

## SQL fallback

Some operations have no open-source implementation at all: `DROP FEATURE`,
`REORG`, `CLONE`, column rename and drop. Views, materialized views and
row-filtered tables are likewise unreachable by direct storage access. A
Databricks SQL warehouse can serve all of it, because the work happens
server-side.

It is off by default:

```python
conn = ds.connect(allow_sql_fallback=True, warehouse_id="abc123")
```

Rerouting a scan through a warehouse changes latency, egress and DBU cost by
orders of magnitude. A silent reroute would turn that cliff into a mystery, so
you opt in, and a `SqlFallbackWarning` fires whenever the fallback actually runs.

## Known limits

Read [conformance.md](conformance.md) for the full matrix. The gaps worth
knowing before you start:

Catalog-managed tables can be read, appended to and fully overwritten. What they
cannot do is partial replacement: `DELETE`, `UPDATE`, `MERGE` and a predicate
overwrite all need deletion vectors, and `Transaction::update_deletion_vectors`
does not exist in delta-kernel 0.28, so there is nothing to build on. Those route
to the SQL fallback.

They also cannot be checkpointed here. Binding the kernel's checkpoint writer
deadlocks against the default engine's executor, and delta-rs cannot open the
table, so the log grows unbounded. Run a checkpoint from Databricks periodically.

Creating a *managed* table is still refused: it needs the catalog to allocate
storage through its staging-table API, which cannot be verified without a live
catalog.
Distributed scan planning has its seams in place but no Ray integration yet.
Kernel-side appends handle unpartitioned tables only, which matters solely for
catalog-managed tables, since path-based ones route to delta-rs anyway.

A `Table` captures catalog-managed commit state when it resolves, so it will not
see later commits, including its own. Re-open it to pick them up. Credential
re-vending happens between operations and not inside a single streaming read.

On the Databricks side, external write to UC managed Delta is Public Preview,
and external access to catalog-commit tables is Beta behind a workspace preview
an admin must enable. `preflight()` failing on a fresh workspace is the expected
first result, not a bug.
