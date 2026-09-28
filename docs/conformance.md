# Conformance matrix

Which engine serves which table feature, operation and property. The source
of truth is `python/deltaswamp/capability.py` (features and operations) and
`python/deltaswamp/properties.py` (properties), and
`tests/unit/test_capability.py` fails if this page drifts from them.

## The tables in code

| Table | Contents |
|---|---|
| `FEATURE_SUPPORT` | 36 table features x {kernel, delta-rs} x {read, write} |
| `OPERATION_ENGINES` | 46 operations -> engines in preference order |
| `PROPERTY_SUPPORT` | 32 table properties x {create, set} x engine |
| `FEATURE_DEPENDENCIES` | what the kernel enforces before a write |

## Read this before changing routing

`vacuumProtocolCheck` is a ReaderWriter feature, so it blocks delta-rs from
*reading* the table, not only from running VACUUM.

delta-rs has no write support for `domainMetadata`. Row tracking and liquid
clustering both depend on it, so every row-tracked and every liquid-clustered
table is delta-rs-unwritable, though delta-rs reads them: writer-only features
block its writes, not its reads. A large share of modern Databricks tables fall
in here.

The wire name for liquid clustering is `clustering`, not `clusteredTable`.

`collations`, `checkpointProtection` and `icebergWriterCompatV1` exist in
Databricks but have no kernel 0.28 variant at all. All three are writer-only, so
both engines read these tables; kernel classifies them as Unknown, which blocks
writes on both engines, so writes route to SQL or get refused. Databricks sets
`icebergWriterCompatV1` on every `USING ICEBERG` table, which is catalog-managed
Delta with UniForm underneath.

Collations are the one place a read goes quietly wrong rather than failing:
neither engine knows collation order, so `name = 'oslo'` compares bytes and
misses `'Oslo'` under `UTF8_LCASE`. Predicate scans that read a collated
column never route to a direct engine.

Several features are readable but only up to a point. Databricks adds
`variantShredding` and `delta.enableVariantShredding=true` to every VARIANT
table, and shreds each file whose values share a shape; the kernel fails on a
shredded file, and nothing in the log says which files are. So the property
routes reads that touch a VARIANT column away from the direct engines (to the
warehouse, or a refusal naming it), while `count()` and reads of other columns
stay direct. A shredded file met anyway, with the property off, is an
`EngineLimitError`. VARIANT itself reads as JSON text on every engine, since
that is all the warehouse sends, and writes take JSON text back.

Collated columns are matched by name: a predicate that reads no collated
column is still served directly; one that does, or that reads a nested field
(whose collation the Arrow schema does not carry), goes to the warehouse. `geospatial` is gated off in
this kernel build, and its `geometry(...)` schema type breaks both engines' log
parsing, so those tables read through the warehouse only.

Row filters and column masks make Unity Catalog refuse credential vending, yet
the capability manifest keeps `HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT` on such
a table. The catalog reads the policy from the table itself for that reason.

delta-rs cannot decode the CDF files Databricks writes, so the kernel serves
change data feeds.

Two features run the other way. `checkConstraints` and `generatedColumns` are
cases where delta-rs is the more capable engine, so routing must never assume
kernel wins by default. delta-kernel refuses every write to a table carrying
either, used or not (legacy writer versions 3 to 6 imply both). The kernel
paths here evaluate every CHECK constraint in DuckDB over the rows they write
(a NULL result passes, a FALSE one fails the write before anything is
committed), and commit past the kernel's refusal of `checkConstraints`, and of
`generatedColumns` where no column is generated; a constraint DuckDB cannot
evaluate as Databricks does is refused up front. The kernel's checkpoint
writer and its version checksums still refuse a table with the feature.

Unknown feature names must not raise. The kernel tolerates unknown writer-only
features when reading, and so must deltaswamp, or the first table to adopt a
newer feature becomes unreadable.

## Write modes

Which engine serves each mode. deltaswamp fills three gaps the engines leave: dynamic partition overwrite is emulated with a generated
`replaceWhere`, idempotent writes are enforced here because neither engine
deduplicates, and save modes come from `Connection.write_table`.

| Mode | API | Engine | Note |
|---|---|---|---|
| append | `t.append(data)` | delta-rs, kernel, iceberg, sql | the warehouse stages data in a volume |
| full overwrite | `t.overwrite(data)` | delta-rs, kernel | the kernel replaces a catalog-managed table in one commit |
| replaceWhere | `t.overwrite(data, predicate=...)` | kernel (deletion vectors), delta-rs, iceberg, sql, kernel (rewrite) | on a table with deletion vectors enabled the kernel marks the replaced rows deleted and appends the new ones; otherwise the kernel's last resort is a bounded whole-table rewrite |
| dynamic partition overwrite | `t.overwrite(data, partition_overwrite='dynamic')` | delta-rs | emulated; predicate built from the partition values in `data` |
| schema merge | `t.append(data, schema_mode='merge')` | delta-rs | routes as MERGE_SCHEMA, not APPEND |
| schema overwrite / RTAS | `t.replace(data)` | delta-rs | |
| create | `conn.create_table(name, schema)` | delta-rs, kernel | kernel takes over for properties delta-rs rejects, and for `cluster_by` |
| save modes | `conn.write_table(name, data, mode=...)` | both | `error`, `ignore`, `append`, `overwrite` |
| idempotent write | `t.append(data, txn=(app_id, version))` | enforced here | neither engine deduplicates; verified against delta-rs 1.6.5 |
| commit metadata | `t.append(data, commit_metadata={...})` | delta-rs | shows up in `history()` |
| distributed write | `t.plan_write()` / `plan.write()` / `plan.commit()` | kernel | workers write files, the driver commits them as one version; refused at plan time, before any file is written |
| DELETE / UPDATE / MERGE | `t.delete()`, `t.update()`, `t.merge()` | kernel (deletion vectors), delta-rs, sql | on a table with deletion vectors enabled the kernel writes them, as Databricks does, catalog-managed and row-tracked tables included; elsewhere delta-rs is copy-on-write, and the kernel's last resort is a bounded whole-table rewrite (DELETE, UPDATE) or a rewrite of the touched files (MERGE, and all DML on row-tracked tables, whose row ids it keeps) |

## Where every operation routes

Generated from `OPERATION_ENGINES`; `tests/unit/test_capability.py` fails if an
operation is missing here. `sql` is reachable only with
`allow_sql_fallback=True`. `sharing` and `iceberg` serve only tables that are
theirs, so their place in a chain matters only for tables two engines could
serve.

| Operation | Engines, in order | Why |
|---|---|---|
| `scan` | kernel, deltars, sharing, iceberg, sql | kernel reads through writer-only features delta-rs rejects |
| `time_travel` | kernel, deltars, sharing, iceberg, sql | history_manager handles the ICT-enablement boundary |
| `cdf` | kernel, deltars, sharing, sql | the kernel's TableChanges comes first: delta-rs cannot decode the CDF files Databricks writes (arrow-rs fails with 'cannot skip miniblock' on their DELTA_BINARY_PACKED pages), double-encodes a partition path containing '%' (a value 'a b' is looked up as k=a%2520b), and refuses column-mapped tables outright. Catalog-managed tables have no CDF outside Databricks |
| `incremental` | *(none)* | Table.changes() follows the change data feed, and is routed as cdf() is; Table.added_since() reads the files the kernel's incremental scan lists, and is routed as a scan |
| `history` | deltars, iceberg, sql | kernel exposes no history() API, only commit_range primitives |
| `detail` | kernel, deltars, sharing, iceberg, sql | kernel CRC path gives O(1) stats with zero I/O when a .crc exists |
| `files` | deltars, kernel | delta-rs lists add actions with stats; the kernel lists the files of tables delta-rs cannot open |
| `append` | deltars, kernel, iceberg, sql | delta-rs for path and external tables; it refuses catalog-managed tables, which then fall through to kernel and UCCommitter. The warehouse loads through a staging volume when neither can |
| `overwrite` | deltars, kernel, iceberg, sql | delta-rs first; the kernel replaces a catalog-managed table in one commit |
| `replace_where` | deltars, iceberg, sql, kernel | deletion vectors through the kernel when the table enables them; otherwise delta-rs, PyIceberg for Iceberg tables, the warehouse, and last a bounded whole-table rewrite through the kernel |
| `create` | deltars, kernel | delta-rs creates path and external tables; the kernel takes over when the properties or clustering exceed what delta-rs accepts, and for managed tables, whose storage the catalog allocates through its staging-table API |
| `merge_schema` | deltars, kernel, sql | delta-rs first; the kernel for tables it cannot write or whose column mapping it cannot extend, committing the rows and the widened schema together; the warehouse uses INSERT WITH SCHEMA EVOLUTION |
| `delete` | deltars, sql, kernel | deletion vectors through the kernel when the table enables them; otherwise delta-rs copy-on-write, then the warehouse, and last a bounded whole-table rewrite through the kernel (on a row-tracked table, a rewrite of the touched files that keeps every row id) |
| `update` | deltars, sql, kernel | deletion vectors plus new files through the kernel when the table enables them (row ids kept under row tracking); otherwise delta-rs, the warehouse, and last a bounded whole-table rewrite (on a row-tracked table, a rewrite of the touched files that keeps every row id), with literal or column assignments |
| `merge` | deltars, sql, kernel | deletion vectors through the kernel, with clauses evaluated in DuckDB, when the table enables them; otherwise delta-rs, then the warehouse, which merges from a source staged in a volume, and last the kernel copy-on-write (touched files rewritten; row ids kept under row tracking) |
| `add_column` | deltars, kernel, sql | delta-rs first; kernel for tables it cannot write |
| `drop_column` | kernel, sql | metadata-only under column mapping, which the kernel path writes; delta-rs has no DROP COLUMN |
| `rename_column` | kernel, sql | metadata-only under column mapping, which the kernel path writes; delta-rs has no RENAME COLUMN |
| `set_properties` | deltars, kernel, sql | delta-rs takes the keys it handles at create, probed against deltalake 1.6.5; it rejects column mapping, row tracking, in-commit timestamps and type widening, and enabling deletion vectors through it stamps a bogus variantType feature, so those go to the kernel |
| `add_feature` | deltars, kernel, sql | delta-rs only for features it can then write, with their dependencies present; otherwise the kernel, which adds dependencies alongside |
| `drop_feature` | sql | Databricks-only (DROP FEATURE ... TRUNCATE HISTORY) |
| `add_constraint` | deltars, kernel, sql | delta-rs first; the kernel path for tables delta-rs cannot write, after checking every existing row in DuckDB |
| `drop_constraint` | deltars, kernel, sql | a metadata-only change |
| `unset_properties` | kernel, sql | delta-rs has no way to remove a property |
| `set_comment` | deltars, kernel, sql | the table description in the Metadata action |
| `set_column_comment` | deltars, kernel, sql | a 'comment' entry in the column's field metadata |
| `alter_column_type` | kernel, sql | type widening: metadata-only under the typeWidening feature; delta-rs cannot read such a table at all |
| `set_not_null` | kernel, sql | needs every existing row checked for nulls before the commit |
| `drop_not_null` | deltars, kernel, sql | a metadata-only change |
| `cluster_by` | kernel, sql | the delta.clustering domain; delta-rs has no domain metadata support |
| `optimize` | kernel, deltars, sql | the kernel commits a compaction on the snapshot it read, so a concurrent one conflicts; delta-rs's OPTIMIZE commit rebases over it and duplicates the rows both compacted, so delta-rs hands it to the kernel. On row-tracked tables rows keep their ids and commit versions; on liquid-clustered ones it Z-orders by the clustering keys (`full=True` every file); column-mapped tables compact too. The warehouse runs it on managed tables |
| `zorder` | kernel, deltars, sql | as OPTIMIZE: delta-rs's Z-ORDER commit duplicates rows under a concurrent one |
| `vacuum` | deltars, kernel, sql | delta-rs for tables it can commit to; the kernel's log replay for the rest (clustering, row tracking, in-commit timestamps, type widening, vacuumProtocolCheck, ...) and for deletion-vector tables, whose vector files delta-rs cannot tell apart from orphans |
| `restore` | deltars, kernel, sql | delta-rs for tables it can commit to; the kernel re-adds the target version's files as logged (deletion vectors, row ids) on deletion-vector tables, which delta-rs restores wrongly (delta-rs#4613), and the tables delta-rs cannot write |
| `repair` | deltars, kernel, sql | delta-rs for tables it can commit to; the kernel removes the missing files as logged (row ids kept) on the rest |
| `checkpoint` | deltars, kernel | delta-rs for tables it can open; the kernel for the rest, including catalog-managed tables, which it publishes first |
| `log_compaction` | deltars | kernel's log_compaction_writer is a no-op stub (kernel#2337) |
| `publish` | kernel | Snapshot::publish; only kernel implements staged->published |
| `reorg` | sql | Databricks-only (REORG ... APPLY PURGE / UPGRADE UNIFORM) |
| `clone` | kernel, sql | kernel: path table to a path (a raw version 0 over the source's files); Databricks for catalog tables |
| `convert` | deltars | kernel has no CONVERT TO DELTA |
| `generate` | deltars, kernel | delta-rs for tables it can open for writing; the kernel's file listing for the rest (clustering, row tracking, in-commit timestamps, type widening, defaults). Deletion vectors and column mapping are refused, as Spark refuses them |
| `cleanup_metadata` | kernel, deltars | the kernel deletes log files older than delta.logRetentionDuration below a checkpoint every retained version reads from, by in-commit timestamps where the table has them, v2 checkpoints' sidecars included; delta-rs where the kernel cannot read the table |
| `analyze` | sql | Databricks-only (ANALYZE TABLE ... COMPUTE [DELTA] STATISTICS) |
| `sync_iceberg` | sql | Databricks-only (MSCK REPAIR TABLE ... SYNC METADATA regenerates UniForm Iceberg metadata) |
| `refresh` | sql | Databricks-only (REFRESH of a materialized view or streaming table) |

## Table properties

delta-rs rejects part of the Delta property surface with a single opaque
message, and panics on `delta.minReaderVersion`. On ALTER it takes what it takes
at create. It accepts `delta.enableDeletionVectors` but answers it, at create
or later, by stamping a spurious `variantType` reader+writer feature into the
protocol, so deltaswamp treats the key as rejected and creates DV tables with
the kernel. The kernel accepts nearly all
of it. `validate_properties` checks against this table first, so the failure
names the key and the remedy, and a create delta-rs cannot serve falls through
to the kernel automatically.

| Property | delta-rs create | delta-rs set | kernel create |
|---|---|---|---|
| `delta.appendOnly` | honored | honored | honored |
| `delta.autoOptimize.autoCompact` *(Databricks-only)* | stored, inert | stored, inert | stored (as v1) |
| `delta.autoOptimize.optimizeWrite` *(Databricks-only)* | stored, inert | stored, inert | stored (as v1) |
| `delta.checkpoint.writeStatsAsJson` | stored, inert | stored, inert | honored |
| `delta.checkpoint.writeStatsAsStruct` | honored | honored | honored |
| `delta.checkpointInterval` | honored | honored | honored |
| `delta.checkpointPolicy` | stored, inert | stored, inert | honored |
| `delta.checkpointRetentionDuration` *(Databricks-only)* | rejected | rejected | stored (as v1) |
| `delta.columnMapping.mode` | honored | rejected | honored |
| `delta.compatibility.symlinkFormatManifest.enabled` *(Databricks-only)* | rejected | rejected | n/a |
| `delta.dataSkippingNumIndexedCols` | honored | honored | honored |
| `delta.dataSkippingStatsColumns` | top-level names only | top-level names only | honored |
| `delta.deletedFileRetentionDuration` | honored | honored | honored |
| `delta.enableChangeDataFeed` | honored | honored | honored |
| `delta.enableDeletionVectors` | rejected | rejected | honored |
| `delta.enableExpiredLogCleanup` | honored | honored | honored |
| `delta.enableIcebergCompatV2` | rejected | rejected | n/a |
| `delta.enableIcebergCompatV3` | rejected | rejected | honored |
| `delta.enableInCommitTimestamps` | rejected | rejected | honored |
| `delta.enableRowTracking` | rejected | rejected | honored |
| `delta.enableTypeWidening` | rejected | rejected | honored |
| `delta.isolationLevel` | honored | honored | stored (as v1) |
| `delta.logRetentionDuration` | honored | honored | honored |
| `delta.minReaderVersion` | **crashes** | rejected | n/a |
| `delta.minWriterVersion` | honored | honored | n/a |
| `delta.parquet.compression.codec` | rejected | rejected | n/a |
| `delta.parquet.format.version` | rejected | rejected | honored |
| `delta.randomizeFilePrefixes` *(Databricks-only)* | stored, inert | stored, inert | stored (as v1) |
| `delta.setTransactionRetentionDuration` | stored, inert | stored, inert | honored |
| `delta.targetFileSize` | honored (byte strings such as `128mb` too) | honored | stored (as v1) |
| `delta.tuneFileSizesForRewrites` *(Databricks-only)* | stored, inert | stored, inert | stored (as v1) |
| `delta.universalFormat.enabledFormats` *(Databricks-only)* | rejected | rejected | n/a |

"stored (as v1)": delta-kernel refuses these keys in CREATE TABLE, so a
kernel create commits version 0 without them and applies them at once as
version 1 (a metadata commit, like set_properties); a catalog-managed create
refuses them instead. `delta.dataSkippingStatsColumns` on delta-rs covers
top-level leaf columns only (no nested fields or structs, and nothing under
column mapping), so appends and overwrites of a table that sets it go to the
kernel; delta-rs UPDATE/MERGE/OPTIMIZE still write the narrower stats. A
table setting `delta.checkpoint.writeStatsAsJson=false` gets
`writeStatsAsStruct=true` recorded with it (Spark's default for the unset
key, which this library's checkpoint writers read as false); a table that
already has JSON stats off and struct stats unset refuses `checkpoint()` and
skips automatic checkpoints, which would otherwise keep no statistics.
`delta.dataSkippingNumIndexedCols` counts differently per writer: the kernel
counts leaf fields as Spark does (`s struct<a,b,c>, x` with 2 indexes `s.a`,
`s.b`), delta-rs counts top-level columns and indexes every leaf of the ones
it takes. Extra statistics are harmless; the clustering-column check assumes
Spark's leaf counting, the stricter of the two.

On an existing table, set_properties checks values the same way whichever
engine serves it: a delta-rs ALTER is first run through the kernel path's
checks, so `delta.targetFileSize=abc`, `delta.isolationLevel=snapshot`, a
`delta.dataSkippingStatsColumns` entry naming no column, a hand-set
`delta.minWriterVersion`, or `delta.enableChangeDataFeed=true` on a table with a
`_change_type`/`_commit_version`/`_commit_timestamp` column are refused before
anything is committed. The two Databricks-only keys above are stored by the
kernel's metadata path (delta-rs rejects them) for Databricks to act on.

Keys with no row follow three rules. A `delta.feature.<name>` signal is
rejected by delta-rs and accepted by the kernel for the sixteen features in
`KERNEL_CREATE_FEATURES`; `clustering` is not one of them, because
the kernel wants clustering columns through `cluster_by`. Any other unknown
`delta.*` key is rejected by both. A custom key outside the `delta.` namespace
is rejected by delta-rs and stored verbatim by the kernel.
