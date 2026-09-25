# Feature map

Every Delta Lake and Unity Catalog feature, on Databricks and in open source,
and how deltaswamp reaches it. Where a library already does the job,
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
| Time travel by timestamp | all | kernel, delta-rs, warehouse | kernel honors in-commit timestamps |
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
| replaceWhere | DBR, Spark, delta-rs | kernel (deletion vectors), delta-rs, iceberg, warehouse | kernel: replaced rows marked deleted, new rows appended, in one commit; a bounded whole-table rewrite on tables without deletion vectors |
| Dynamic partition overwrite | DBR, Spark | native over delta-rs | predicate derived from the data |
| Schema merge on write | DBR, Spark, delta-rs | delta-rs | |
| Save modes | DBR, Spark, delta-rs | native | `Connection.write_table` |
| Idempotent writes (txnAppId) | DBR, Spark | native | checked here; delta-rs records but does not enforce |
| DELETE / UPDATE | DBR, Spark, delta-rs | kernel (deletion vectors), delta-rs, warehouse | deletion vectors on tables that enable them; otherwise delta-rs copy-on-write, and last a bounded whole-table rewrite through the kernel |
| MERGE | DBR, Spark, delta-rs | kernel (deletion vectors), delta-rs, warehouse | one clause API for all three; the kernel evaluates clauses with DuckDB (`deltaswamp[duckdb]`), the warehouse merges from a staged source |
| DML on catalog-managed tables | DBR | kernel (deletion vectors), warehouse | DELETE/UPDATE/replaceWhere/MERGE as deletion vectors through UCCommitter; row ids kept on row-tracked tables. Without deletion vectors, DELETE/UPDATE/replaceWhere are a bounded rewrite and MERGE needs the warehouse |
| Deletion-vector authoring | DBR, Spark | kernel | bitmaps computed here, written in the protocol's file format, committed through the kernel's DV update; a second DELETE unions with the existing vector; files left empty are removed |
| Row-id preservation on UPDATE / MERGE | DBR, Spark | kernel | updated rows' ids are written to the table's materialized row-id column |
| DML + CDF | DBR, Spark | kernel (DELETE), delta-rs, warehouse | a deletion-vector DELETE needs no CDC files; UPDATE and MERGE on a CDF table need CDC files the kernel cannot write |
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
| Auto optimize / auto compaction | DBR | — | writer-side behavior of Databricks; stored as properties only |

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

| Gap | Blocker |
|---|---|
| UPDATE, MERGE and replaceWhere on a change-data-feed table the kernel alone can write | the kernel cannot write CDC files, and those commits need them; DELETE through deletion vectors does not |
| CDF on catalog-managed tables outside Databricks | the kernel's `TableChanges` takes no catalog commit tail |
| Incremental reads without a change feed | the kernel's `incremental_scan` is not bound yet; `Table.changes()` covers tables with CDF |
| Databricks server-side behavior (predictive optimization, auto compaction, row-level concurrency, Photon, CLUSTER BY AUTO) | these are things a Databricks cluster does, not table formats; the warehouse fallback is the only way in |
| UniForm metadata generation outside Databricks | Databricks-only |
| Managed-table creation and catalog-managed commits on Databricks | Databricks allowlists which connectors may write through the UC Delta API, by User-Agent |
| Writes to UC managed tables without `HAS_DIRECT_EXTERNAL_ENGINE_WRITE_SUPPORT` | a per-table decision by the catalog; the warehouse fallback serves them |

deltaswamp identifies itself as `deltaswamp/<version>` (see `deltaswamp._sdk`),
but the name still has to be registered with Databricks. Until then, managed
`create_table` and catalog-managed writes need `allow_sql_fallback=True`. Reads
are unaffected: the commit tail comes from the same API and is served normally.
