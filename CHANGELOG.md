# Changelog

Notable changes, newest first. This project follows [semantic
versioning](https://semver.org/); while the major version is 0, minor releases
may break the API.

## Unreleased

First release.

### Added

- `Connection` and `Table` for Unity Catalog (Databricks and open source),
  Hive Metastore, Glue, Delta Sharing and plain object-storage paths.
- A Python binding to delta-kernel-rs, including catalog-managed tables:
  snapshots resolved against the catalog's commit tail, appends and overwrites
  through `UCCommitter`, publishing, and checkpoints.
- Per-operation routing across delta-kernel, delta-rs, PyIceberg, the
  `delta-sharing` client and an opt-in Databricks SQL warehouse.
- `capabilities()` and `can()`, which report what a table supports, which
  engine serves it, and why not when nothing does.
- Reads with projection, predicates, time travel by version or timestamp,
  change data feed, file listing and `changes()` for following a table.
- Every write mode: append, overwrite, `replaceWhere`, dynamic partition
  overwrite, schema merge, save modes, commit metadata and idempotent writes.
- DELETE, UPDATE, replaceWhere and MERGE written as deletion vectors on tables
  that enable them, including catalog-managed and row-tracked tables (row ids
  are kept). MERGE through the kernel evaluates its clauses with DuckDB. Tables
  without deletion vectors get copy-on-write, with a bounded whole-table rewrite
  for tables only the kernel can write.
- Metadata-only ALTER commits for what delta-rs cannot do: column rename and
  drop under column mapping, type widening, SET NOT NULL, clustering keys and
  unset properties.
- Managed-table creation through Unity Catalog's staging flow, and external
  table registration.
- Maintenance: OPTIMIZE, Z-ORDER, VACUUM, RESTORE, FSCK, checkpoints, log
  cleanup and manifest generation.
- Unity Catalog governance: grants, tags, ownership, lineage, key constraints,
  catalogs, schemas, volumes and files.
- Distributed reads (`plan_scan`, `to_ray_dataset`) and writes (`plan_write`),
  shipping a short-lived storage credential by default, or the picklable
  credential provider (`ship_catalog_auth=True`) so workers vend their own.
- Hand-offs to DuckDB, Polars and Daft, and cross-catalog SQL through
  `Connection.sql`.

- One storage-option merge for every engine: canonical keys, vended
  credentials over the caller's, and the caller's region and endpoint over
  vended guesses (docs/usage.md, "Storage options"). Azure sovereign clouds,
  the kernel on fully qualified `abfss://` URLs without an endpoint option, and
  the AWS China endpoint on the kernel now work.
- Writes are refused, before any side effect, on S3-compatible stores that
  ignore `If-None-Match` (a one-time probe; opt out with
  `deltaswamp_skip_put_if_absent_probe`), with `aws_conditional_put=disabled`,
  and with `AWS_S3_LOCKING_PROVIDER=dynamodb`.
- `connect(iceberg_properties=...)`; `storage_options` reach PyIceberg and
  Delta Sharing downloads.

- DELETE, UPDATE, MERGE, OPTIMIZE, Z-ORDER and RESTORE return an
  `OperationResult`: a dict that keeps the serving engine's own keys and adds
  the same normalized keys on every engine (`num_deleted_rows`,
  `num_updated_rows`, `num_inserted_rows`, `num_affected_rows`,
  `num_files_added`, `num_files_removed`, `version`; `num_removed_files` and
  `num_restored_files` for RESTORE), plus `.engine`. Code comparing a result
  for exact equality with a dict needs to compare the keys it cares about.

### Known limits

- A kernel create with `delta.targetFileSize`, `delta.isolationLevel`,
  `delta.autoOptimize.*`, `delta.tuneFileSizesForRewrites`,
  `delta.randomizeFilePrefixes` or `delta.checkpointRetentionDuration` writes
  two versions: version 0 without them (delta-kernel refuses them in CREATE),
  version 1 setting them. A catalog-managed create refuses them.
- `to_duckdb()`, `to_polars(lazy=True)`, `to_pyarrow_dataset()` and
  `Connection.sql` read lazily and push down the projection and simple
  comparisons of a column with a literal of its own kind -- integer, string,
  boolean, date or decimal columns (`=`, `<`, `IN`, `IS NULL`, `AND`/`OR` of
  those). `NOT`, float and double columns (NaN orders differently in Spark,
  pyarrow and Polars), binary and timestamp columns, functions and `LIKE` are
  not pushed: those read every row and the consumer filters afterward, with
  its own semantics (Polars' filters run in Polars).
  `to_daft()` still reads eagerly: pass `columns=`/`predicate=`.
- Snapshots are cached per process, keyed by location, version and a keyed
  digest of the storage options (so one URL on two endpoints or accounts never
  shares an entry). A reuse first re-reads the strong identity of the commit
  file the snapshot ends at (ETag or object version on object stores; device,
  inode, nanosecond mtime/ctime and size locally), so a table re-created at the
  same path is read afresh; a store that reports no ETag is never cached.
  Commits always read the log from storage, and a distributed write carries the
  table's metaData id: `plan.write()`/`plan.commit()` against a table dropped
  and re-created since planning raise `MetadataChangedError`.
- The kernel's S3 store honours the standard `AWS_*` environment variables
  (`AWS_ENDPOINT_URL`, keys, region, ...) as delta-rs does, below every option
  passed or vended; environment credential keys are used only when neither
  the caller nor the catalog gave one.
- `count()` answers from the log (numRecords less deletion-vector
  cardinality) on the kernel when every file has numRecords and the predicate
  reads only partition columns; otherwise it scans.
- `delta.dataSkippingStatsColumns` naming nested fields is honored by kernel
  writes only; delta-rs UPDATE, MERGE and OPTIMIZE (and every write to a
  column-mapped table delta-rs created) still collect top-level stats only.
- Concurrent `create_table` on one path has exactly one winner; delta-rs used
  to retry the loser at version 1 and replace the winner's schema. The loser
  gets "a Delta table already exists there".
- `wasb://` and `wasbs://` locations are refused; use `abfss://`.
- A path table with a GCS OAuth bearer token in `storage_options` is served by
  the kernel only: history, OPTIMIZE, VACUUM, RESTORE, MERGE and log cleanup,
  which only delta-rs implements, are refused on it.

- `convert_to_delta` refuses a hive-partitioned directory whose partition
  values are escaped (`region=a%20b`): delta-rs records those paths unencoded
  and the converted table cannot be read. Rewrite the data with `write_table`.
- RESTORE through delta-rs is refused across a change to column-mapping
  metadata (the mode, or `maxColumnId` after an ADD COLUMN): it would rewind
  the mode or reuse column ids. Restore such tables from Databricks.
- Adding a NOT NULL column is refused on every engine, as Databricks refuses
  it: add it nullable, backfill, then `set_not_null()`.
- On a kernel-only table without deletion vectors, DML is a whole-table
  rewrite bounded by `KernelEngine.rewrite_max_bytes` and refused on
  row-tracked tables, and MERGE needs the SQL fallback.
- The kernel cannot write CDC files, so UPDATE and MERGE on a change-data-feed
  table it alone can write need the SQL fallback. DELETE through deletion
  vectors needs none.
- The change feed fails when the range crosses an incompatible schema change
  (a dropped, renamed or retyped column), with `ChangeFeedSchemaChangeError`
  naming the version. Start the range at the change, or read it through the
  warehouse. An added column reads as null in older rows, and `changes()`
  yields each version under the schema it was written with. A range crossing
  a version with the feed off raises `UnreachableTableError` naming it
  (`.version`); `changes()` yields the versions before it first. A schema
  change the range reverses (a RESTORE to an older schema) is found only
  once the read reaches it, so a consumer of `__arrow_c_stream__`
  (`pa.table(t.cdf(...))`, polars, DuckDB) gets that error's message as an
  `ArrowInvalid`; iterate the stream, or call `read_all()`, for the typed
  error.
- Databricks managed Iceberg (`USING ICEBERG`) takes appends through the
  Iceberg REST endpoint only; overwrite by predicate, DELETE, UPDATE and
  MERGE need the SQL fallback (the endpoint takes one snapshot per commit,
  and splitting them into two commits would not be atomic).
- A MERGE with `merge_schema=True` that assigns a column only the target has
  is refused on delta-rs (1.6.5 fails it) and needs the SQL fallback.
- On a legacy writer-3-to-6 table, enabling a feature delta-rs cannot write
  (clustering, type widening) is refused locally, since no local engine could
  write the upgraded table.
- Nanosecond timestamps are written as microseconds (truncated), as Spark
  writes them.
- Commits the kernel writes, including distributed ones, record empty
  `operationParameters`: delta_kernel 0.28 overwrites whatever the engine
  supplies. `isBlindAppend` still tells an append from an overwrite.
- A MERGE on a change-data-feed table whose last NOT MATCHED clause has a
  condition is refused on delta-rs (1.6.5 inserts an all-NULL row per rejected
  source row) and needs the SQL fallback.
- DELETE/UPDATE/replaceWhere SQL beyond the kernel's predicate grammar
  (arithmetic, function calls) needs delta-rs or the warehouse, so on a
  catalog-managed table it needs the SQL fallback.
- The change feed and history of a catalog-managed table need the warehouse,
  which only a Databricks connection has: `allow_sql_fallback=True` is refused
  on OSS Unity Catalog, Hive Metastore, Glue, Delta Sharing and path
  connections, where those operations are refused without that remedy.
- Delta Sharing has no split planning: `plan_scan()` and `to_ray_dataset()`
  are refused for shared tables (presigned URLs would expire in transit to
  workers). Read with `to_arrow()` or `scan()`. Whether time travel and the
  change feed work depends on the provider sharing the table WITH HISTORY;
  `can()` says so, and refuses the change feed outright when the shared
  metadata has it off. A column-mapped or deletion-vector table needs a server
  that answers in delta format (the open-source reference server does not).
- On a Databricks USING ICEBERG table, the SQL fallback refuses CHECK
  constraints, deletion vectors, liquid clustering while deletion vectors or
  row tracking are not explicitly off, and `delta.*` properties Databricks does
  not keep; the change feed of any table with Iceberg metadata needs row
  tracking.
- Hive Metastore and Glue `drop_table` remove only the registration; the data
  files, Delta log included, stay.
- Databricks managed Iceberg (`USING ICEBERG`) takes appends through the
  Iceberg REST endpoint; overwrites need the warehouse, because the endpoint
  takes one snapshot per commit.
- Incremental reads without a change feed are not built.
- Databricks allowlists which connectors may write through the Unity Catalog
  Delta API, so writes to managed tables go through the SQL fallback until
  deltaswamp is registered. Reads are unaffected.
- Idempotent writes are checked against the last committed version before
  writing, so a concurrent writer can still commit in between.
- Identity and default columns are created only through Databricks (a catalog
  name with the SQL fallback); locally both engines refuse them, since neither
  assigns the values. Generated columns are created by delta-rs, which
  evaluates them; the kernel refuses to create or write them.
- Tables written through delta-rs keep no min/max statistics for decimal
  columns of more than 15 digits (or structs holding one): delta-rs would log
  them as rounded doubles, which Databricks trusts for data skipping. Files
  written before this change may still carry such stats.
- A binary partition column is written by the kernel only, as UTF-8 text; a
  value that is not valid UTF-8 is refused. An empty-string partition value is
  written as null, as Spark does.
- A local path containing `%xx` or a backslash is refused at create; one with
  `[`, `]`, `|` or `^` is written through the kernel. Schema evolution on a
  column-mapped table needs the warehouse.
- delta-rs does not rebase pre-1582 dates and timestamps in files Spark wrote
  in its legacy hybrid calendar; the kernel does. Timestamps before 1900 in
  such a file written in a non-UTC session zone, or in one that records no
  zone (Spark 2.x), are refused by the kernel and need the warehouse: Spark
  rebases them with per-zone tables this library does not carry. DELETE,
  UPDATE, MERGE, replaceWhere, OPTIMIZE and Z-ORDER never go to delta-rs on a
  table holding such values (or pre-1900 INT96 timestamps), since it would
  rewrite them shifted; they take the kernel's deletion-vector or whole-table
  rewrite paths, or the warehouse, or are refused.
- A time-travel timestamp after the latest commit reads the latest version on
  the direct engines; the warehouse refuses it. A RESTORE to one is refused
  everywhere, as Spark refuses it.
- A table URI (`file://`, `s3://`, ...) containing `?` or `#` is refused at
  create: a URL reads them as a query or fragment. Percent-encode them.
- delta-rs writes no statistics at all (null counts included) for decimal
  columns of more than 15 digits; Parquet offers no null-count-only level.
- VARIANT reads as JSON text on every engine and writes take JSON text,
  nested in structs too. A VARIANT inside an array or map is left in the
  engine's own form, and a table with `delta.enableVariantShredding` (every
  new Databricks VARIANT table) reads its VARIANT columns only through the
  warehouse.
- A DEFAULT that is not a plain literal is evaluated only by Databricks, so a
  write, MERGE INSERT or `SET c = DEFAULT` that needs it needs the SQL
  fallback.
- Spark SQL that the direct engines cannot be made to evaluate as Spark does
  (an array subscript, `split`, a Java date pattern, `hash`) needs the SQL
  fallback; see "Spark SQL on the direct engines" in the usage guide for what
  is translated and what still differs.
