# Feature map

Every Delta Lake and Unity Catalog feature across Databricks and the open
ecosystem, and how deltaswamp reaches it. Where a library already does the job,
deltaswamp calls it. Its own code covers the gaps: the kernel binding,
metadata-only commits, exact filtering for kernel predicates, staging flows,
credential lifetimes, GCS bearer tokens and Azure endpoints.

## Building blocks

| Component | What it is | What deltaswamp takes from it |
|---|---|---|
| Databricks Runtime / SQL warehouses | the reference implementation, including every Databricks-only feature | the opt-in SQL fallback, via the Statement Execution API in `databricks-sdk` |
| `databricks-sdk` | auth plus every Unity Catalog REST API | all Databricks auth; tables, grants, tags, lineage, constraints, volumes, files, credential vending |
| delta-kernel-rs | the Delta protocol as a library; reads everything, writes little | the native extension: reads, catalog-managed commits, CDF, timestamp travel, file skipping, managed-table creation |
| delta-rs (`deltalake`) | the most complete open writer: MERGE, OPTIMIZE, VACUUM, RESTORE | DML and maintenance on tables it can open |
| Unity Catalog OSS | the open catalog; implements the `/delta/v1` API | a second UC backend and the test double for catalog-managed tables |
| `delta-sharing` | the Delta Sharing client | reads of shared tables, CDF, time travel |
| PyIceberg | Iceberg in Python, including REST catalogs | managed and foreign Iceberg tables through UC's Iceberg REST endpoint |
| DuckDB, Polars, Daft, Ray | query and compute engines | hand-offs over the Arrow PyCapsule interface, and `Connection.sql` |

Each covers a slice and fails differently on the rest, which is why
deltaswamp routes between them.

## How to read the matrices

**Where it exists** says who implements the feature: `DBR` (Databricks
only), `Spark` (open-source delta-spark, which Python cannot use without a
JVM), `delta-rs`, `kernel`, or `UC API`.

**deltaswamp** says how it is reached here:

| Value | Meaning |
|---|---|
| delta-rs | delegated to `deltalake` |
| kernel | the native extension over delta-kernel-rs |
| native | written by deltaswamp itself (metadata commits, predicate filtering, staging) |
| SDK | a Unity Catalog REST call through `databricks-sdk` |
| warehouse | the SQL fallback, opt-in with `allow_sql_fallback=True` |
| sharing / iceberg | the Delta Sharing or PyIceberg engine |
| — | not reachable; the row says why |

Several values in one cell are a routing chain, tried in order.

## Reading

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| Scan path tables | all | kernel, delta-rs | kernel first: it reads through writer-only features delta-rs refuses |
| Scan catalog-managed tables | DBR, kernel | kernel | needs the catalog's commit tail; no other Python library can open these |
| Column projection | all | kernel, delta-rs, warehouse | |
| Predicate filtering | all | kernel (native), delta-rs, warehouse | kernel skips files; the exact row filter is applied here, from one parsed predicate |
| Time travel by version | all | kernel, delta-rs, warehouse | |
| Time travel by timestamp | all | kernel, delta-rs, warehouse | kernel honours in-commit timestamps |
| Change data feed | DBR, Spark, delta-rs, kernel | delta-rs, kernel, sharing, warehouse | kernel covers path tables delta-rs cannot open; by version or timestamp |
| CDF on catalog-managed tables | DBR | warehouse | the kernel's TableChanges takes no catalog tail |
| History | all | delta-rs, iceberg, warehouse | |
| Detail / protocol / properties | all | kernel, delta-rs, warehouse | |
| File listing with stats | Spark, delta-rs, kernel | delta-rs, kernel | `Table.files()` |
| Deletion vectors on read | all | kernel, delta-rs | applied before any reordering |
| Views, MVs, metric views, row-filtered tables | DBR | warehouse | vending refuses them; only the warehouse can evaluate them |
| Shallow clones | DBR, Spark | warehouse | absolute paths into the source defeat credential scoping |
| Distributed scan | Spark, kernel | kernel | `plan_scan()` / `to_ray_dataset()`; per-file splits pinned to a version |
| Incremental / streaming read | DBR, Spark | native over CDF | `Table.changes()` follows the change feed version by version; reading added files without CDF needs `incremental_scan`, not yet bound |

## Writing and DML

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| Append | all | delta-rs, kernel, iceberg, warehouse | warehouse loads via a staging volume |
| Append to catalog-managed | DBR, kernel | kernel | UCCommitter; partitioned tables included |
| Overwrite | all | delta-rs, kernel, iceberg, warehouse | kernel replaces a catalog-managed table in one commit |
| replaceWhere | DBR, Spark, delta-rs | delta-rs, iceberg, warehouse, kernel | kernel: a bounded whole-table rewrite |
| Dynamic partition overwrite | DBR, Spark | native over delta-rs | predicate derived from the data |
| Schema merge on write | DBR, Spark, delta-rs | delta-rs | |
| Save modes | DBR, Spark, delta-rs | native | `Connection.write_table` |
| Idempotent writes (txnAppId) | DBR, Spark | native | checked here; delta-rs records but does not enforce |
| DELETE / UPDATE | DBR, Spark, delta-rs | delta-rs, warehouse, kernel (native rewrite) | copy-on-write; the kernel path rewrites the whole table in one commit, bounded in size |
| MERGE | DBR, Spark, delta-rs | delta-rs, warehouse | one clause API for both; the warehouse merges from a staged source |
| DML on catalog-managed tables | DBR | kernel (native rewrite), warehouse | DELETE/UPDATE/replaceWhere by rewrite; MERGE needs the warehouse; row-tracked tables need the warehouse |
| Deletion-vector authoring | DBR, Spark | — | kernel 0.28 has `update_deletion_vectors` only as an internal API; not bound yet |
| Row-level concurrency | DBR | — | a Databricks conflict-detection feature |
| Distributed write | Spark | kernel | `plan_write()`: workers write files, the driver commits them in one transaction. Catalog-managed tables included |
| COPY INTO / Auto Loader | DBR | warehouse (`Connection.sql(engine="warehouse")`) | ingestion, not table access |

## Schema and table DDL

The **native** rows are metadata-only commits written by deltaswamp: the
change is computed from the table's current protocol and metadata, then
committed with a put-if-absent of the next log file. A concurrent writer makes
the commit fail and triggers a recompute; it is never silently overwritten.

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| CREATE (path / external) | all | delta-rs, kernel | kernel for properties delta-rs rejects and for liquid clustering |
| CREATE managed (catalog-managed) | DBR, UC API | kernel + UC API | staging-table flow; see [Unity Catalog](#unity-catalog) |
| Register an existing external table | DBR, UC API | SDK | writes nothing unless the log exists |
| ADD COLUMNS | all | delta-rs, native, warehouse | native for tables delta-rs cannot write |
| RENAME / DROP COLUMN | DBR, Spark | native, warehouse | metadata-only under column mapping |
| Enable column mapping | DBR, Spark | native | none -> name only; existing names become physical names |
| ALTER COLUMN TYPE (widening) | DBR, Spark, kernel read | native, warehouse | records `delta.typeChanges` |
| SET / DROP NOT NULL | DBR, Spark, delta-rs (drop) | native, delta-rs, warehouse | SET checks the data first |
| Table and column comments | all | delta-rs, native, warehouse | |
| SET / UNSET TBLPROPERTIES | all | delta-rs, native, warehouse | native validates keys and raises the protocol when a value implies a feature |
| ADD / DROP CHECK constraint | DBR, Spark, delta-rs | delta-rs, native (drop), warehouse | adding validates existing rows, which delta-rs does |
| ADD FEATURE | DBR, Spark, delta-rs | delta-rs (features it can write), native, warehouse | native adds dependencies alongside and refuses features that need a backfill (row tracking) |
| DROP FEATURE | DBR, Spark | warehouse | needs history truncation and checkpoint protection |
| CLUSTER BY (change keys) | DBR, Spark | native, warehouse | writes the `delta.clustering` domain |
| CLUSTER BY AUTO | DBR | warehouse | predictive optimization chooses keys |
| Primary / foreign keys | UC | SDK | informational constraints |
| Row filters, column masks | UC | warehouse | |
| Identity, generated, default columns | DBR, Spark | declared at create | kernel create; writes respect them where the engine does |

## Table features

`docs/conformance.md` has the full per-engine matrix for all 36 features.
In summary: the kernel reads every standard feature except geospatial and
adaptiveMetadata-preview (both gated off in this build). For reads, delta-rs
refuses seven reader-writer features: `catalogManaged` and its preview, type
widening and variant shredding (two spellings each), and `vacuumProtocolCheck`.
Writer-only features such as `domainMetadata` (so every liquid-clustered and
row-tracked table) and in-commit timestamps block its writes but not its reads.
Collations, checkpoint protection and `icebergWriterCompatV1` have no kernel
variant; all three are writer-only, so both engines read and neither writes.

## Maintenance and the log

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| OPTIMIZE (compaction) | all | delta-rs, warehouse | |
| OPTIMIZE ZORDER BY | DBR, Spark, delta-rs | delta-rs, warehouse | |
| OPTIMIZE on clustered / managed tables, OPTIMIZE FULL | DBR | warehouse | delta-rs cannot write clustered tables |
| VACUUM (standard and LITE) | DBR, Spark, delta-rs | delta-rs, warehouse | dry run by default here |
| VACUUM on managed tables | DBR | warehouse | Databricks forbids external VACUUM on managed tables |
| RESTORE | DBR, Spark, delta-rs | delta-rs, warehouse | refused on DV tables through delta-rs (delta-rs#4613) |
| FSCK REPAIR | DBR, delta-rs | delta-rs, warehouse | |
| Checkpoint | all | delta-rs, kernel | the kernel checkpoints catalog-managed tables, publishing first |
| Log compaction | Spark, delta-rs | delta-rs | kernel's writer is a stub |
| Expired log cleanup | all | delta-rs | `Table.cleanup_metadata()` |
| Publish staged commits | kernel | kernel | required on catalog-managed tables |
| ANALYZE (DELTA) STATISTICS | DBR | warehouse | |
| REORG PURGE / UPGRADE UNIFORM | DBR, Spark | warehouse | |
| CLONE (shallow, deep) | DBR, Spark | warehouse | |
| CONVERT TO DELTA | DBR, Spark, delta-rs | delta-rs | |
| Symlink manifests | Spark, delta-rs | delta-rs | |
| Predictive optimization | DBR | — | server-side scheduling; `Table.info()` reports whether it is on |
| Auto optimize / auto compaction | DBR | — | writer-side behaviour of Databricks; stored as properties only |

## Unity Catalog

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| Name resolution, capability manifest | UC API | SDK | the manifest decides external eligibility before any read |
| Credential vending (table) | UC API | SDK | two refresh clocks; picklable providers |
| Credential vending (path) | UC API | SDK | for creating external tables |
| Catalog-managed commits | UC API | kernel | `/delta/v1`, 409 vs 429 distinguished |
| Managed-table creation | UC API | kernel + UC API | staging table, v0 commit, finalize |
| Catalogs, schemas: list/create/drop | UC API | SDK | |
| Table info (owner, comment, columns, filters, masks) | UC API | SDK | `Table.info()` |
| Grants and effective grants | UC API | SDK | |
| Tags (table and column) | UC API | SDK | |
| Ownership | UC API | SDK | |
| Lineage (table and column) | UC API | SDK | |
| Volumes and files | UC API | SDK | also the staging area for warehouse writes |
| Functions | UC API | SDK | listing |
| UNDROP | DBR | warehouse | |
| System tables, information_schema | DBR | warehouse (`Connection.sql`) | |
| Iceberg REST catalog | UC API | iceberg | |
| Hive Metastore, Glue | — | catalogs | location from the metastore, truth from the log |
| Lakehouse Federation (foreign tables) | DBR | warehouse | data lives in another system |
| Online / synced tables, vector indexes | DBR | — | not file-addressable |
| ABAC policies, Lakehouse Monitoring, clean rooms | DBR | — | out of scope for a table connector; reachable through `databricks-sdk` directly |

## Sharing and interoperability

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| Delta Sharing read | open protocol | sharing | `ds.connect("sharing:///path/profile.share")` |
| Delta Sharing CDF, time travel | open protocol | sharing | when the provider shares history |
| Managed / foreign Iceberg tables | UC | iceberg | read and write through UC's Iceberg REST endpoint |
| UniForm tables as Iceberg | DBR | iceberg (read) | the Delta path remains the default |
| UniForm metadata generation | DBR | warehouse (`sync_iceberg`) | external writes to UniForm tables are refused, since they would stale it |
| Compatibility mode copy | DBR | reported | `ResolvedTable.compatibility_mode_location` |

## Compute integrations

| Target | deltaswamp | Notes |
|---|---|---|
| pyarrow | `to_arrow`, `to_pyarrow_dataset`, `scan()` | `scan()` is a PyCapsule stream; pyarrow is optional |
| pandas | `to_pandas` | |
| Polars | `to_polars(lazy=...)` | works on tables `polars.scan_delta` cannot open |
| DuckDB | `to_duckdb`, `Connection.sql` | the same, for DuckDB's delta extension |
| Ray | `to_ray_dataset` | a Ray Data datasource; workers read byte-balanced groups of files and vend their own credentials |
| Daft | `to_daft` | |
| Cross-catalog SQL | `Connection.sql(query, tables=...)` | join a catalog-managed table with a Glue table and a path |

## Out of reach

Each item names its blocker.

- **Deletion-vector authoring**: delta-kernel-rs 0.28 has
  `update_deletion_vectors` only as an internal API, and it is not bound yet.
  DML on kernel-only tables is therefore a bounded whole-table rewrite, and
  MERGE and row-tracked tables need the warehouse.
- **CDF on catalog-managed tables** outside Databricks: the kernel's
  `TableChanges` takes no catalog commit tail.
- **Incremental reads without a change feed**: `incremental_scan` exists in
  the kernel and is not yet bound. `Table.changes()` covers tables with CDF.
- **Databricks server-side behaviour**: predictive optimization, auto
  compaction, row-level concurrency, Photon and CLUSTER BY AUTO. These are not
  table formats; they are things a Databricks cluster does. The warehouse
  fallback is the only way in.
- **UniForm metadata generation** outside Databricks.
- **Writes through the Unity Catalog Delta API on Databricks**, which covers
  managed-table creation and catalog-managed commits. Databricks allowlists
  which connectors may call those endpoints, keyed on the product User-Agent,
  and refuses anything it does not recognise:

  > The UC Delta API requires clients to identify the calling application in the
  > User-Agent header. The provided User-Agent '...' is insufficient.

  deltaswamp sends `deltaswamp/<version>` (see `deltaswamp._sdk`), which is a
  precondition, not a solution: the name has to be registered with Databricks.
  Until it is, `create_table` for a managed table and any catalog-managed write
  need `allow_sql_fallback=True`. Reads are unaffected: the commit tail for a
  catalog-managed table comes from the same API and is served normally.
- **Writing to a Unity Catalog managed table from outside Databricks at all**,
  where the metastore withholds `HAS_DIRECT_EXTERNAL_ENGINE_WRITE_SUPPORT`. This
  is a per-table decision by the catalog, not a protocol limit; the warehouse
  fallback serves those writes and the refusal says so.
