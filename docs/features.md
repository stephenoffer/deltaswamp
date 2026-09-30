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
| Time travel by timestamp | all | kernel, delta-rs, warehouse | one resolver for both direct engines: in-commit timestamps, else file times made monotonic, as Databricks resolves them |
| Change data feed | DBR, Spark, delta-rs, kernel | kernel, delta-rs, sharing, warehouse | kernel first; delta-rs reads past the last version and predicates the kernel cannot parse; by version or timestamp |
| CDF on catalog-managed tables | DBR | kernel, warehouse | read from the commits of a snapshot resolved with the catalog's tail (`engine/log_changes.py`): each commit's CDC files, else its adds and removes with deletion-vector pairs resolved, as TableChanges derives them; `plan_changes()` too |
| History | all | delta-rs, iceberg, warehouse | |
| Detail / protocol / properties | all | kernel, delta-rs, warehouse | |
| File listing with stats | Spark, delta-rs, kernel | delta-rs, kernel | `Table.files()` |
| Deletion vectors on read | all | kernel, delta-rs | applied before any reordering |
| Views, MVs, metric views, row-filtered tables | DBR | warehouse | vending refuses them; only the warehouse can evaluate them. `plan_scan()` / `to_ray_dataset()` read them in parallel: the query runs once and each result chunk is a split a worker fetches by its presigned link |
| Shallow clones | DBR, Spark | warehouse | absolute paths into the source defeat credential scoping |
| Distributed scan | Spark, kernel | kernel, warehouse | `plan_scan()` / `to_ray_dataset()`; per-file splits pinned to a version, each carrying the kernel's scan row, so workers read with no log listing or replay and no catalog call. Tables only the warehouse can read are split by result chunk instead. See [Ray Data](ray-data.md) |
| Distributed change feed | Spark | kernel | `plan_changes()`: runs of whole commits per split; catalog-managed tables included, their workers reading the unpublished commits from the staged files the plan names |
| Incremental / streaming read | DBR, Spark | native over CDF | `Table.changes()` follows the change feed version by version; `changes(..., include_snapshot=True)` first yields the table at the start version as inserts |
| Incremental read without CDF (rows added since a version) | kernel | kernel | `Table.added_since(version)` reads the files the kernel's incremental scan lists as added; refused when the range removed files, unless `only_appends=True` |

## Writing and DML

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| Append | all | delta-rs, kernel, iceberg, warehouse | warehouse loads via a staging volume |
| Append to catalog-managed | DBR, kernel | kernel | UCCommitter; partitioned tables included |
| Overwrite | all | delta-rs, kernel, iceberg, warehouse | kernel replaces a catalog-managed table in one commit |
| replaceWhere | DBR, Spark, delta-rs | kernel (deletion vectors), delta-rs, iceberg, warehouse | kernel: replaced rows marked deleted, new rows appended, in one commit; a bounded whole-table rewrite on tables without deletion vectors |
| Dynamic partition overwrite | DBR, Spark | native over delta-rs | predicate derived from the data |
| Schema merge on write | DBR, Spark, delta-rs | delta-rs, kernel, warehouse | the kernel for tables delta-rs cannot write and for column-mapped tables: new columns and nested fields (ids and physical names assigned) in the same commit as the rows; a wider type only under type widening |
| Save modes | DBR, Spark, delta-rs | native | `Connection.write_table` |
| Idempotent writes (txnAppId) | DBR, Spark | native | checked here; delta-rs records but does not enforce |
| DELETE / UPDATE | DBR, Spark, delta-rs | kernel (deletion vectors), delta-rs, warehouse | deletion vectors on tables that enable them; otherwise delta-rs copy-on-write, and last a bounded whole-table rewrite through the kernel. A row-tracked table without deletion vectors gets a kernel rewrite of only the touched files, which keeps every row id. The kernel evaluates predicates and SET values beyond its grammar (arithmetic, functions, CASE, nested fields) with DuckDB in Spark's dialect, with ANSI overflow and division-by-zero errors |
| MERGE | DBR, Spark, delta-rs | kernel (deletion vectors or copy-on-write), delta-rs, warehouse | one clause API for all three; the kernel evaluates clauses with DuckDB (`deltaswamp[duckdb]`), the warehouse merges from a staged source. On tables delta-rs cannot write (in-commit timestamps, clustering, type widening, column defaults, row tracking, legacy-calendar files) and that do not enable deletion vectors, the kernel removes each touched file and writes its other rows again beside the new ones |
| DML on catalog-managed tables | DBR | kernel (deletion vectors), warehouse | DELETE/UPDATE/replaceWhere/MERGE as deletion vectors through UCCommitter; row ids kept on row-tracked tables. Without deletion vectors, a copy-on-write of the touched files, read back a few at a time. A commit that loses a race re-reads the catalog's tail and rebases over blind appends |
| Deletion-vector authoring | DBR, Spark | kernel | bitmaps computed here, written in the protocol's file format, committed through the kernel's DV update; a second DELETE unions with the existing vector; files left empty are removed |
| Row-id preservation on DELETE / UPDATE / MERGE / replaceWhere | DBR, Spark | kernel | updated rows' ids are written to the table's materialized row-id column; a copy-on-write rewrite also writes the kept rows' ids and commit versions to the materialized columns, and inserted rows get fresh ids above the high-water mark. Removes (staged by hand, as kernel 0.28 refuses them there) and re-adds carry each file's `baseRowId` and `defaultRowCommitVersion` |
| DML + CDF | DBR, Spark | kernel, delta-rs, warehouse | UPDATE, MERGE, replaceWhere and copy-on-write DELETE write CDC files under `_change_data/` (`update_preimage`/`update_postimage`, `delete`, `insert`) and commit `cdc` actions beside the data; a deletion-vector DELETE needs none. Catalog-managed tables included |
| Row-level concurrency | DBR | — | a Databricks conflict-detection feature |
| Distributed write | Spark | kernel | `plan_write()`: workers write files, the driver commits them in one transaction, rebased and retried, idempotent after a lost response; `abort()` deletes the files of a write that cannot commit. Catalog-managed tables included. See [Ray Data](ray-data.md) |
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
| CREATE by a distributed write | DBR, Spark | kernel + UC API | `Connection.plan_write(name, schema=...)`: the table appears at commit with its data, and a failed job leaves none. Path and external tables in one commit (version 0 holds the files); managed tables stage version 0 at planning. See [Ray Data](ray-data.md) |
| ADD COLUMNS | all | delta-rs, native, warehouse | native for tables delta-rs cannot write |
| RENAME / DROP COLUMN | DBR, Spark | native, warehouse | metadata-only under column mapping |
| Enable column mapping | DBR, Spark | native | none -> name only; existing names become physical names |
| ALTER COLUMN TYPE (widening) | DBR, Spark, kernel read | native, warehouse | records `delta.typeChanges` |
| SET / DROP NOT NULL | DBR, Spark, delta-rs (drop) | native, delta-rs, warehouse | SET checks the data first |
| Table and column comments | all | delta-rs, native, warehouse | |
| SET / UNSET TBLPROPERTIES | all | delta-rs, native, warehouse | native validates keys and raises the protocol when a value implies a feature |
| ADD / DROP CHECK constraint | DBR, Spark, delta-rs | delta-rs, native, warehouse | adding validates every existing row (delta-rs, or DuckDB on the native path); kernel writes then evaluate every constraint over the rows they write |
| ADD FEATURE | DBR, Spark, delta-rs | delta-rs (features it can write), native, warehouse | native adds dependencies alongside and refuses features that need a backfill (row tracking) |
| DROP FEATURE | DBR, Spark | warehouse | needs history truncation and checkpoint protection |
| CLUSTER BY (change keys) | DBR, Spark | native, warehouse | writes the `delta.clustering` domain |
| CLUSTER BY AUTO | DBR | warehouse | predictive optimization chooses keys |
| Primary / foreign keys | UC | SDK | informational constraints |
| Row filters, column masks | UC | warehouse | |
| Identity, generated, default columns | DBR, Spark | kernel, delta-rs | identity and literal-default columns are created locally (version 0 composed with the feature; identity start/step/allowExplicitInsert typed as Spark reads them), generated columns by delta-rs. The kernel computes and checks them on every append and on workers of a distributed write (DuckDB for expressions; unevaluable ones refused at planning). A distributed write reserves identity values at planning, one slot per task; generated values follow Spark (the high-water mark rounded to start + k * step). DML on such tables stays with delta-rs or the warehouse |

## Table features

`docs/conformance.md` has the full per-engine matrix for all 36 features.
In summary: the kernel reads every standard feature except
adaptiveMetadata-preview (gated off in this build); geospatial columns read as
WKB binary. For reads, delta-rs
refuses seven reader-writer features: `catalogManaged` and its preview, type
widening and variant shredding (two spellings each), and `vacuumProtocolCheck`.
Writer-only features such as `domainMetadata` (so every liquid-clustered and
row-tracked table) and in-commit timestamps block its writes but not its reads.
Collations and `icebergWriterCompatV1` have no kernel variant; both are
writer-only, so both engines read and neither writes. Checkpoint protection has
no kernel variant either, but it binds only log cleanup, so the kernel path
writes through it and refuses only history truncation.

## Maintenance and the log

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| OPTIMIZE (compaction) | all | kernel, warehouse | committed by the kernel on the snapshot it planned from, so concurrent runs never compact a file twice (delta-rs's own commit duplicates their rows; a delta-rs connection hands it to the kernel). Legacy-calendar and INT96 files are rewritten with correct values and a Spark footer. Row-tracked tables keep every row's `_metadata.row_id` and `row_commit_version` (written into the materialized columns, as Databricks does); column-mapped tables (name and id) get physical names and field ids. `min_file_size` and `sort_by` shape a bin-packing. Refused with `writer_properties`/`min_commit_interval` |
| OPTIMIZE ZORDER BY | DBR, Spark, delta-rs | kernel, warehouse | as above; top-level columns with statistics only (not STRUCT/ARRAY/MAP). Incremental: files tagged `ZCUBE_*` by an earlier Z-order of the same columns, in cubes of at least `min_cube_size` (default the target size), are left alone; the tags are Databricks', which takes the kernel's cubes as its own |
| OPTIMIZE on liquid-clustered tables, OPTIMIZE FULL | DBR | kernel, warehouse | the kernel Z-orders over the `delta.clustering` keys, incrementally as above (`full=True` rewrites every file); the domain is left as it is. Not Databricks' clustering tree, which Databricks keeps to itself |
| OPTIMIZE on managed tables | DBR | kernel, warehouse | catalog-managed tables: committed through UCCommitter, re-planned over a blind append. Managed tables without catalog commits: warehouse, as Databricks refuses them external writes |
| VACUUM (standard and LITE) | DBR, Spark, delta-rs | delta-rs, kernel, warehouse | dry run by default here. The kernel plans it from its log replay, as Spark's VACUUM does, on deletion-vector tables and on every table delta-rs cannot commit to (clustering, row tracking, in-commit timestamps, type widening, `vacuumProtocolCheck`, ...), and commits VACUUM START/END |
| VACUUM on managed tables | DBR | kernel, warehouse | catalog-managed tables: dry run by default; a real run needs `allow_catalog_managed=True` and commits VACUUM START/END through UCCommitter, since deleting files is the catalog's policy to allow |
| RESTORE | DBR, Spark, delta-rs | delta-rs, kernel, warehouse | the kernel serves deletion-vector tables (delta-rs leaves DV changes in place, delta-rs#4613), column-mapped tables and the tables delta-rs cannot write; restored files keep their row ids, and the target's schema and properties come back. On catalog-managed tables it commits through UCCommitter, rebased over a concurrent append; a target whose metadata differs is refused, as the catalog refuses metadata changes |
| FSCK REPAIR | DBR, delta-rs | delta-rs, kernel, warehouse | the kernel serves the tables delta-rs cannot commit to: live files whose data file is gone are removed as logged (row ids and deletion vectors kept), `dataChange` true as delta-rs removes them; refused on shallow clones, and (except a dry run) on append-only and Iceberg-enabled tables |
| Checkpoint | all | delta-rs, kernel | the kernel checkpoints catalog-managed tables, publishing first, and the tables delta-rs cannot open whose protocol carries CHECK constraints, generated or identity columns or invariants (a checkpoint writes no row for them to bind); the checkpoint holds the table's own protocol and metadata. Only an unknown writer feature, `icebergCompatV1`/`V2` and the like still refuse it |
| Version checksums (`.crc`) | DBR, Spark, kernel | kernel, delta-rs | written after every commit where the previous one is at most 100 versions back (or the log is short), on tables with CHECK constraints or generated columns too, and at every kernel checkpoint of such a table; delta-rs writes none itself (delta-rs#4190) |
| Log compaction | Spark, delta-rs | delta-rs | kernel's writer is a stub |
| Expired log cleanup | all | kernel, delta-rs | `Table.cleanup_metadata()`. The kernel deletes log files older than `delta.logRetentionDuration` below the newest checkpoint committed before the boundary (so every retained version reads), v2 sidecars nothing retained references included, by in-commit timestamps where the table has them; catalog-managed tables and `checkpointProtection` are refused. delta-rs where the kernel cannot read the table |
| Publish staged commits | kernel | kernel | required on catalog-managed tables |
| ANALYZE (DELTA) STATISTICS | DBR | warehouse | |
| REORG PURGE / UPGRADE UNIFORM | DBR, Spark | warehouse | |
| CLONE (shallow, deep) | DBR, Spark | kernel, warehouse | the kernel clones a path table to a path (delta-rs#2456): shallow with absolute-path adds and vectors, deep by copying; catalog targets and catalog-scoped credentials need the warehouse. A shallow clone is for Spark and Databricks: the direct engines read only files under a table's root. Row-tracked tables are refused: the clone would have to carry every file's `baseRowId` and the high-water mark, and its rows' commit versions have no meaning in the new table's history |
| CONVERT TO DELTA | DBR, Spark, delta-rs | delta-rs | |
| Symlink manifests | Spark, delta-rs | delta-rs, kernel | `Table.generate()`. The kernel writes them as Spark does for the tables delta-rs cannot open for writing (clustering, row tracking, in-commit timestamps, type widening, defaults): a manifest per Hive-escaped partition directory listing decoded absolute paths, stale partitions' manifests deleted. Deletion vectors and column mapping are refused, as Spark refuses them |
| Predictive optimization | DBR | — | server-side scheduling; `Table.info()` reports whether it is on |
| Auto optimize / auto compaction | DBR | — | writer-side behavior of Databricks; stored as properties only |

## Unity Catalog

| Feature | Where it exists | deltaswamp | Notes |
|---|---|---|---|
| Name resolution, capability manifest | UC API | SDK | the manifest decides external eligibility before any read |
| Credential vending (table) | UC API | SDK | two refresh clocks; picklable providers shared per process; the native store reads a refreshable slot on every request; S3 region per bucket |
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
| Writes to UniForm / IcebergCompatV1-V2 tables | DBR, Spark | kernel | partition values materialized, nested field ids, `numRecords`, copy-on-write DML; schema changes stay with Databricks. A UniForm table needs `ds.connect(uniform_writes="sync")` (Iceberg metadata regenerated through the warehouse after each commit) or `"stale"` (Iceberg readers stay on the last converted version) |
| UniForm metadata generation | DBR | warehouse (`sync_iceberg`) | `MSCK REPAIR TABLE ... SYNC METADATA`, which Databricks documents for writes by clients that do not generate Iceberg metadata |
| Geometry and geography columns | DBR | kernel | read and written as WKB binary, typed GEOMETRY/GEOGRAPHY in Parquet as Databricks writes them; not inside arrays or maps, no change feed, no schema changes |
| Compatibility mode copy | DBR | reported | `ResolvedTable.compatibility_mode_location` |

## Compute integrations

| Target | deltaswamp | Notes |
|---|---|---|
| pyarrow | `to_arrow`, `to_pyarrow_dataset`, `scan()` | `scan()` is a PyCapsule stream; pyarrow is optional |
| pandas | `to_pandas` | |
| Polars | `to_polars(lazy=...)` | works on tables `polars.scan_delta` cannot open |
| DuckDB | `to_duckdb`, `Connection.sql` | the same, for DuckDB's delta extension |
| Ray | `to_ray_dataset`, `plan_scan`, `plan_changes`, `plan_write` | a Ray Data datasource; workers read byte-balanced groups of files with the plan's storage credential, refreshed from `credential_source=` or shipped catalog auth. The building blocks for `read_delta`/`write_delta` are in [Ray Data](ray-data.md) |
| Daft | `to_daft` | |
| Cross-catalog SQL | `Connection.sql(query, tables=...)` | join a catalog-managed table with a Glue table and a path |

## Out of reach

| Gap | Blocker |
|---|---|
| Databricks server-side behavior (predictive optimization, auto compaction, row-level concurrency, Photon, CLUSTER BY AUTO) | these are things a Databricks cluster does, not table formats; the warehouse fallback is the only way in |
| UniForm metadata generation outside Databricks | Databricks-only; `uniform_writes="sync"` asks the warehouse after each commit |
| Managed-table creation and catalog-managed commits on Databricks | Databricks allowlists which connectors may write through the UC Delta API, by User-Agent |
| Writes to UC managed tables without `HAS_DIRECT_EXTERNAL_ENGINE_WRITE_SUPPORT` | a per-table decision by the catalog; the warehouse fallback serves them |

deltaswamp identifies itself as `deltaswamp/<version>` (see `deltaswamp._sdk`),
but the name still has to be registered with Databricks. Until then, managed
`create_table` and catalog-managed writes need `allow_sql_fallback=True`. Reads
are unaffected: the commit tail comes from the same API and is served normally.
