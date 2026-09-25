# Usage guide

The whole public API, with its limits.

- [Install](#install)
- [Connecting](#connecting)
- [Opening a table](#opening-a-table)
- [Reading](#reading)
- [Writing](#writing)
- [Creating and registering tables](#creating-and-registering-tables)
- [Changing rows](#changing-rows)
- [Schema, properties and features](#schema-properties-and-features)
- [Maintenance](#maintenance)
- [Metadata](#metadata)
- [Governance](#governance)
- [Delta Sharing](#delta-sharing)
- [Iceberg](#iceberg)
- [Handing off to other engines](#handing-off-to-other-engines)
- [Distributed reads and writes](#distributed-reads-and-writes)
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
Data leaves through the Arrow PyCapsule interface, so pyarrow is optional.

| Extra | Adds | You need it for |
|---|---|---|
| `pyarrow` | pyarrow | `to_arrow`, predicates on the kernel path, the SQL fallback |
| `pandas` | pandas | `to_pandas` |
| `polars` | Polars | `to_polars`, `Connection.sql(engine="polars")` |
| `duckdb` | DuckDB | `to_duckdb`, `Connection.sql` |
| `daft` | Daft | `to_daft` |
| `ray` | Ray Data | `to_ray_dataset` |
| `sql` | pyarrow | the SQL warehouse fallback (the warehouse itself is reached through `databricks-sdk`) |
| `iceberg` | PyIceberg | Iceberg tables through Unity Catalog's Iceberg REST endpoint |
| `sharing` | `delta-sharing` | Delta Sharing |
| `hms` | `pymetastore` | Hive Metastore catalogs |
| `glue` | boto3 | AWS Glue catalogs |
| `all` | all of the above except Ray and Daft | |

No extra installs `databricks-connect`: it pins `requires-python == 3.12.*`
and conflicts with pyspark.

## Connecting

`connect()` with no arguments talks to Databricks and hands authentication to
`databricks-sdk`, with its usual precedence: explicit arguments, environment,
`~/.databrickscfg`, then cloud-native sources. Every method the SDK supports
works, from PATs and OAuth to Azure MSI, GCP service accounts and GitHub OIDC.

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
conn = ds.connect("sharing:///path/to/config.share")  # a Delta Sharing profile
```

Useful keyword arguments:

| Argument | Effect |
|---|---|
| `allow_sql_fallback=True` | permits routing through a SQL warehouse; off by default |
| `warehouse_id="..."` | which warehouse the fallback uses; chosen automatically when omitted |
| `staging_volume="cat.schema.vol"` | lets the warehouse serve writes and MERGE, staging data in that volume |
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

| Form | Goes to |
|---|---|
| `catalog.schema.table` | the connection's catalog |
| `uc://catalog.schema.table` | Unity Catalog |
| `hms://host:9083/db/table` | Hive Metastore |
| `glue://db.table`, `glue://<id>/db.table` | Glue |
| `share.schema.table` on a sharing connection | Delta Sharing |
| `s3://`, `gs://`, `abfss://`, `file://`, `/local/path` | storage directly |

`dbfs:/` and `/mnt/...` paths are refused with an explanation, since they are
unreachable from outside Databricks.

## Reading

`scan()` is the primitive. It returns an Arrow stream that exports
`__arrow_c_stream__`, so Polars, DuckDB, pandas and pyarrow can all consume it
without this library depending on any of them.

```python
stream = t.scan()
stream = t.scan(columns=["id", "amount"])
stream = t.scan(predicate="amount > 100 AND region IN ('eu', 'us')")
stream = t.scan(version=3)
stream = t.scan(timestamp="2026-01-01T00:00:00Z")
```

Predicates and timestamp travel work on every engine, catalog-managed tables
included. On the kernel path one parse of the predicate both skips files by
their statistics and filters the rows exactly. The predicate language is the boolean subset of Spark SQL:
comparisons, `AND`/`OR`/`NOT`, `IN`, `BETWEEN`, `LIKE`, `IS [NOT] NULL`, `<=>`,
nested columns (`addr.zip`) and typed literals (`DATE '2026-01-01'`).
Arithmetic and function calls are refused with a message; delta-rs and the
warehouse accept full SQL.

Timestamps honor in-commit timestamps where a table has them, and a timestamp
earlier than the oldest reconstructable version is refused rather than
silently clamped.

Convenience wrappers sit on top:

```python
t.to_arrow()  # pyarrow.Table
t.to_pandas()
t.to_polars()  # or to_polars(lazy=True)
t.to_pyarrow_dataset()
t.head(10)
t.count()  # exact; count(predicate="...") too
t.files()  # live data files with sizes, partition values and statistics
```

Change feeds come from `cdf()`, by version or by timestamp:

```python
for batch in t.cdf(starting_version=5, ending_version=9):
    ...
t.cdf(starting_timestamp="2026-09-01T00:00:00Z", predicate="region = 'eu'")
```

delta-rs serves CDF on tables it can open and the kernel serves the others.
A catalog-managed table's feed needs the warehouse, since the kernel's change
feed cannot take the catalog's commit tail. Guards fire before any read.
A table without `delta.enableChangeDataFeed` is refused, because enabling it is
not retroactive. A table whose `delta.deletedFileRetentionDuration` is shorter
than its `delta.logRetentionDuration` is refused too, because files could be
vacuumed while their commits survive.

## Writing

```python
t.append(df)  # anything Arrow-shaped
t.overwrite(df)
t.overwrite(df, predicate="region = 'eu'")  # replaceWhere
t.overwrite(df, partition_overwrite="dynamic")  # replace the partitions in df
t.append(df, schema_mode="merge")  # widen the schema to fit
t.replace(df)  # new contents and schema (RTAS)
t.append(df, txn=("nightly-load", batch_id))  # idempotent
```

`df` can be a pyarrow Table or RecordBatchReader, a Polars DataFrame, a pandas
DataFrame, or anything else exporting the Arrow PyCapsule interface.

Catalog-managed tables append through the kernel, which commits via the
catalog. Partitioned tables are handled: rows are split by partition value and
written to their own partitions. `conn.write_table(name, df, mode=...)` adds
Spark's save modes (`error`, `ignore`, `append`, `overwrite`), creating the
table when it is absent.

## Creating and registering tables

```python
import pyarrow as pa

schema = pa.schema([("id", pa.int64()), ("city", pa.string())])

t = conn.create_table("s3://bucket/tables/new", schema)  # a path
t = conn.create_table("main.sales.new", schema)  # managed, catalog-managed
t = conn.create_table("main.sales.ext", schema, location="s3://bucket/ext")  # external
t = conn.create_table("main.sales.c", schema, cluster_by=["city"], comment="...")
t = conn.register_table("main.sales.old", "s3://bucket/existing-delta-table")
```

A managed table goes through the catalog's staging-table flow. The catalog
allocates the id and the storage, deltaswamp writes version 0 there with the
id and protocol the catalog requires, and the catalog then finalizes the
registration. An external table is written with path credentials the catalog
vends for creating tables, then registered, so the catalog and the log agree.
`register_table` takes the schema, partitioning and properties from the log.

Both catalog shapes need Unity Catalog (Databricks or open source). On any
other catalog they are refused, because writing a log alone would leave it
orphaned while the call appeared to succeed.

`properties=` accepts nearly the whole Delta property surface, including
`delta.feature.*` signals, row tracking and in-commit timestamps. When delta-rs
would reject a property, the create falls through to the kernel.

## Changing rows

```python
t.delete("id = 42")
t.delete()  # every row
t.update({"status": "'archived'"}, predicate="age > 365")  # SQL expressions
t.update(new_values={"status": "archived"}, predicate="age > 365")  # plain values
(
    t.merge(source, "t.id = s.id", source_alias="s", target_alias="t")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute()
)
```

`merge` returns a builder with delta-rs's clause API, whichever engine serves
it. When the warehouse serves it, the builder generates one `MERGE INTO`
statement, with the source staged in a volume.

On a table with deletion vectors enabled (`delta.enableDeletionVectors`, the
Databricks default), all three are written as deletion vectors, as Databricks
writes them: the matching rows are marked deleted in a small
`deletion_vector_<uuid>.bin`, rewritten and inserted rows go to new files, and
nothing else is copied. This is the path for catalog-managed tables, and it
works on row-tracked tables too: surviving rows keep their `baseRowId`, and
rows an UPDATE or MERGE rewrites keep their row ids through the table's
materialized row-id column. Only the files the predicate cannot skip are read,
and the commit is staged against the snapshot that was read, so a concurrent
writer makes it conflict rather than be lost. A MERGE evaluates its clauses
with DuckDB (`pip install 'deltaswamp[duckdb]'`), skips target files using the
source's join keys, and refuses a target row matched by more than one source
row, as Spark does.

Without deletion vectors, delta-rs serves DML as copy-on-write, rewriting the
Parquet files that hold matching rows. On a table only the kernel can write,
`delete`, `update` and predicate overwrites then fall back to rewriting the
whole table in one commit, bounded by `KernelEngine.rewrite_max_bytes` (1 GiB
by default) and refused on row-tracked tables; on this path `update` takes
plain values, or SQL that is a literal or a column name, and MERGE needs the
warehouse. On a table with the change data feed enabled, the kernel serves only
DELETE, because UPDATE and MERGE need CDC files it cannot write.

## Schema, properties and features

```python
t.add_column(pa.schema([("region", pa.string())]))  # or {"region": "string"}
t.set_properties({"delta.enableChangeDataFeed": "true"})
t.unset_properties(["owner.team"])
t.add_feature("deletionVectors")
t.add_constraint({"id_positive": "id > 0"})
t.drop_constraint("id_positive")
t.set_comment("orders, one row per line item")
t.set_column_comment("city", "shipping city")
t.set_not_null("id")  # checks existing rows first
t.drop_not_null("id")
t.alter_column_type("qty", "bigint")  # widening; needs delta.enableTypeWidening
t.cluster_by(["city"])  # liquid clustering keys; None clears them
t.rename_column("city", "town")  # needs column mapping
t.drop_column("region")  # needs column mapping
```

delta-rs serves what it can. Everything else on a path table is a
metadata-only commit that deltaswamp writes itself. That covers renames and
drops under column mapping, type widening, SET NOT NULL, clustering keys, the
properties delta-rs rejects on ALTER, and any change to a table delta-rs cannot
write, such as a liquid-clustered one. The change is computed from the table's
current protocol and metadata, and it raises the protocol when a value implies
a feature. It is committed as a put-if-absent of the next log file, so a
concurrent writer causes a recompute rather than an overwrite.

Enable column mapping first to rename or drop:
`t.set_properties({"delta.columnMapping.mode": "name"})`. The existing Parquet
column names become the physical names, so no data is rewritten.

These are refused with the reason, because they need more than a metadata
commit:

- enabling row tracking on an existing table (needs a backfill)
- UniForm / Iceberg compatibility (needs Iceberg metadata generated)
- changing column mapping other than none -> name
- a catalog-managed table's metadata, which the catalog refuses from external
  writers after version 0

Databricks-only operations need the SQL fallback:

```python
t.drop_feature("deletionVectors", truncate_history=True)
t.reorg(purge=True)
t.clone("main.sales.orders_copy", shallow=True)
t.analyze(delta_statistics=True)
t.sync_iceberg()  # regenerate UniForm metadata
t.refresh()  # a materialized view or streaming table
t.cluster_by("auto")
```

## Maintenance

```python
t.optimize()  # bin-packing compaction
t.optimize(zorder_by=["customer_id"])  # or t.z_order([...])
t.optimize(full=True)  # OPTIMIZE FULL; warehouse
t.vacuum(retention_hours=168)  # a dry run by default
t.vacuum(retention_hours=168, dry_run=False, lite=True)
t.restore(3)  # or a datetime
t.repair()  # FSCK
t.checkpoint()
t.compact_logs()
t.cleanup_metadata()  # delete logs past delta.logRetentionDuration
t.generate()  # symlink manifests, for Presto and Athena
t.publish()  # staged catalog commits -> _delta_log
conn.convert_to_delta("s3://bucket/parquet-dir")
```

`vacuum` defaults to a dry run because the real thing deletes files. `lite=True`
considers only files the log records as removed. Three operations refuse rather
than misbehave:

- `vacuum` on a shallow clone, which borrows the source's files.
- `restore` through delta-rs on a table with deletion vectors, where delta-rs
  reports success and leaves the rows deleted.
- OPTIMIZE and VACUUM on UC managed tables. Databricks forbids them from
  external clients, so they route to the warehouse.

Checkpoints work on catalog-managed tables. The kernel publishes staged commits
first, since it checkpoints only published versions. Commits made here also
checkpoint on their own at the table's `delta.checkpointInterval`.

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
t.is_catalog_managed
```

Metadata reflects the table as of the last call, and any write or ALTER through
this object refreshes it. `features()` returns names exactly as the log holds
them, including ones this release does not recognize, because an unknown
writer-only feature must not block a read.

## Governance

On Unity Catalog:

```python
info = t.info()  # owner, comment, columns, row filter, masks, predictive optimization
t.grants(); t.effective_grants()
t.grant("analysts", ["SELECT"]); t.revoke("analysts", "SELECT")
t.tags(); t.set_tags({"pii": "true"}, column="email"); t.unset_tags(["pii"])
t.set_owner("data-platform")
t.lineage(); t.column_lineage("amount")
t.add_primary_key("pk", ["id"]); t.add_foreign_key("fk", ["cust"], "main.sales.customers", ["id"])
t.drop_key_constraint("fk")
t.set_row_filter("main.sec.only_eu", ["region"])  # warehouse
t.set_column_mask("email", "main.sec.mask_email")  # warehouse

conn.create_catalog("sandbox"); conn.create_schema("sandbox.scratch")
conn.grant("sandbox.scratch", "analysts", ["USE_SCHEMA"])
conn.search_tables("main", table_pattern="ord%")
conn.list_volumes("main.raw"); conn.list_functions("main.udfs")
vol = conn.volume("main.raw.landing")
vol.write("drop/2026-09-24.csv", data); vol.read("drop/2026-09-24.csv"); vol.list("drop")
conn.undrop_table("main.sales.orders")  # warehouse
```

Open-source Unity Catalog lacks tags, lineage, key constraints, ownership
changes and the Files API. Those calls come back as refusals that name the
gap. Other catalogs have no governance API, and say so.

## Delta Sharing

```python
conn = ds.connect("sharing:///path/to/config.share")
conn.list_catalogs()  # shares
t = conn.table("share.schema.table")
t.to_arrow(predicate="year = 2026")
t.scan(version=12)
t.cdf(starting_version=10)  # when the provider shares history
```

Shared files are read with pyarrow, keeping Delta's types. Tables with
deletion vectors or column mapping use the client's delta-format path.
Predicates are sent as hints and then applied exactly. Shares are read-only,
and writes are refused with that reason.

## Iceberg

Iceberg tables in Unity Catalog (managed or foreign) are served through the
catalog's Iceberg REST endpoint with PyIceberg: reads, time travel by snapshot
id or timestamp, history, appends, and overwrites by predicate. UniForm Delta
tables can also be read as Iceberg, but the Delta path remains the default for
them. External writes to UniForm tables are refused, because they would leave
the Iceberg metadata stale; `t.sync_iceberg()` regenerates it on Databricks.

## Handing off to other engines

```python
t.to_duckdb(name="orders")  # a DuckDB relation, optionally a view
t.to_polars(lazy=True)
t.to_ray_dataset()  # parallel: a Ray Data datasource over a scan plan
t.to_daft()

for version, batch in t.changes(start, poll_interval=30):  # a CDF stream
    ...

conn.sql(
    "SELECT o.region, sum(o.amount) FROM o JOIN c ON o.cust = c.id GROUP BY 1",
    tables={"o": "main.sales.orders", "c": "glue://crm.customers"},
)
conn.sql("SELECT * FROM system.access.audit LIMIT 10", engine="warehouse")
```

Each hand-off reads through this library, so it works on tables the target
engine's own Delta reader cannot open. `Connection.sql` runs on DuckDB by
default (or `engine="polars"`), and can join tables from different catalogs.
`engine="warehouse"` sends the query to Databricks as it stands.

## Distributed reads and writes

`plan_scan` and `plan_write` split the work between a driver and its workers.
Everything is decided on the driver, so a refusal arrives before any compute
is spent.

```python
plan = t.plan_scan(columns=["id"], predicate="day >= '2026-09-01'")
for group in plan.partitions(8):  # byte-balanced; pickle and ship each
    part = plan.read(group)  # on a worker: same version, own credentials

plan = t.plan_write(mode="append")  # driver: raises now if the table refuses it
fragment = plan.write(batch)  # worker: durable, uncommitted files -> bytes
version = plan.commit(fragments)  # driver: every fragment in one commit
```

`WritePlan` and `ScanPlan` are picklable, and what crosses a process boundary
carries the table's short-lived, table-scoped *storage* credential (vended on
the driver when the plan is pickled) and its expiry -- never the catalog's
credentials (a PAT, an OAuth client secret, a UC bearer token) and never the
catalog itself. The driver's own plan object keeps full catalog access, so
`commit()` runs on the driver; a worker's copy cannot commit. A worker whose
shipped credential is within a minute of expiry raises `CredentialError`
saying to re-plan on the driver. For jobs that outlive the credential, plan
with `ship_catalog_auth=True` to ship the catalog's credential provider instead,
so workers re-vend on their own -- at the cost of the catalog token travelling
in every task payload. `to_ray_dataset()` is built on `plan_scan`, and
catalog-managed tables work too: the commit goes through the catalog's committer.

Concurrency follows what the mode means. An **append** commits against the table
as it is then, so a writer that arrived while the job ran is not a conflict --
both sets of rows survive. An **overwrite** removes what it finds, so committing
against a table that has moved on would discard that writer silently; that is
refused, and `allow_concurrent_overwrite=True` is how you say last-writer-wins.
Where a catalog does arbitrate and rejects the commit, the fragments stay valid
-- they describe data files, which carry no version -- so the same fragments can
be committed again against a fresh snapshot, which `retries=` does for tables
this library commits itself (an append on a path table retries by default, as
`Table.append` does).

A guarded overwrite commits against the planned snapshot itself, so a writer
that lands between the check and the commit makes it conflict rather than
vanish. An idempotent `txn=` is checked again at commit, so two runs of the same
job cannot both land, and a retry refuses fragments whose files an earlier,
seemingly failed attempt already committed. Pass each fragment exactly once: a
data file named twice in one commit is refused.

Which tables accept a distributed write depends on the protocol version. A
legacy writer version implies a whole feature set, and versions 3 to 6 imply
`checkConstraints`, which the kernel refuses whether or not a constraint
exists. Enabling change data feed alone puts a table at version 4.

| Table | Distributed write |
|---|---|
| writer version 1-2 (plain legacy) | yes |
| writer version 3-6 (legacy CDF, legacy column mapping) | no: the version implies `checkConstraints` |
| writer version 7 (feature-based) | yes, for the features the kernel writes |

At version 7, change data feed, column mapping, row tracking, in-commit
timestamps, deletion vectors and type widening all write fine. DELETE, UPDATE,
MERGE and maintenance run on the driver.

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

>>> t.can("create", properties={"delta.enableRowTracking": "true"})
```

`Capability` is truthy when `ok`. Every refusal carries a reason, and a remedy
whenever one exists.

## Errors

All inherit from `DeltaSwampError`.

| Error | Means |
|---|---|
| `InvalidReferenceError` | the reference could not be parsed or resolved |
| `InvalidArgumentError` | a call's arguments are malformed or contradict each other (also a `ValueError`) |
| `UnreachableTableError` | no available engine can serve the request |
| `FallbackRequiredError` | only the SQL fallback could serve it, and it is off |
| `PropertyNotSupportedError` | a property the chosen engine cannot handle |
| `EngineLimitError` | an `UnreachableTableError`: the engine serving a read hit a limit of its own once it read the log, so the next capable engine is tried first |
| `CredentialError` | vending or refresh failed |
| `PreflightError` | a workspace prerequisite is not satisfied |
| `CommitConflictError` | another writer took that version first |
| `TransientCommitError` | a commit failed for a transient reason; the table is unchanged, so retry it as is |
| `BackfillRequiredError` | the catalog wants staged commits published |
| `CorruptTableError` | on-disk state failed a correctness check |
| `MissingDataFileError` | a `CorruptTableError`: a file the snapshot references was removed (VACUUM, manual delete); `.path` names it |
| `PredicateError` | a predicate uses SQL that cannot be evaluated outside a SQL engine |
| `EnginePanicError` | an engine panicked across the FFI boundary |

A conflict means re-read the snapshot, recompute, then stage again at the next
version. A backfill demand is backpressure, not a rate limit: retrying it with
exponential backoff and no publish will wedge the table.

## Credentials

Unity Catalog vends short-lived, per-table storage credentials. They expire
on their own clock, separate from the catalog token the SDK refreshes, and are
re-vended between operations. A single scan that streams past its
credential's lifetime can still fail; `plan_scan()` avoids that by vending per
worker. [Architecture](architecture.md#credentials) has the details.

```python
creds = t.credentials()  # or credentials(write=True)
creds.expires_at  # epoch seconds, or None
creds.expires_within(300)
creds.redacted()  # safe to log
```

Credential *providers* are picklable; credentials are not. Pickling a provider
(or a `Table` or `Connection` holding one) carries its catalog configuration,
which for a token-authenticated workspace includes the token, so a worker can
re-vend. Distributed plans do not do this by default; see "Building a
distributed connector" above. Azure
user-delegation SAS is scoped to a path, so credentials are keyed by table, and
Azure always gets an explicit endpoint.

## SQL fallback

Views, materialized views, row-filtered tables, and the Databricks-only
operations (`DROP FEATURE`, `REORG`, `CLONE`, `ANALYZE`, `CLUSTER BY AUTO`,
`OPTIMIZE FULL`, `UNDROP`, row filters and masks) are served by a Databricks SQL
warehouse, since the work happens server-side. It is off by default:

```python
conn = ds.connect(
    allow_sql_fallback=True,
    warehouse_id="abc123",  # optional: a running warehouse is chosen otherwise
    staging_volume="main.default.staging",  # optional: enables writes and MERGE
)
```

Statements run through the Statement Execution API in `databricks-sdk`, with
results fetched as Arrow. Values are sent as statement parameters and
identifiers are quoted. `predicate` and `updates` strings are SQL expressions
by contract. Writes stage Parquet in the volume, load it with `read_files`, and
delete it afterward.

A warehouse changes latency and cost by orders of magnitude, so you opt in, and
a `SqlFallbackWarning` names the warehouse whenever the fallback runs.

## Known limits

[Conformance](conformance.md) has the full matrix, and the
[feature map](features.md) shows how each Databricks and open-source feature is
reached.

- DML on a kernel-only table without deletion vectors is a bounded whole-table
  rewrite, and MERGE there needs the warehouse. With deletion vectors enabled,
  UPDATE and MERGE on a change-data-feed table need the warehouse too.
- The change feed of a catalog-managed table needs the warehouse.
- Distributed planning is kernel-only; tables served by other engines are read
  on the driver.
- A `Table` captures catalog-managed commit state when it resolves. Re-open it
  to see commits made elsewhere.
- External writes to UC managed Delta are Public Preview on Databricks, and
  external access to catalog-commit tables is Beta behind a workspace preview.
  `preflight()` failing on a fresh workspace is expected.
