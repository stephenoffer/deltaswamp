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
  engine serves it, and why not when nothing does. `can(op, **args)` takes the
  call's own arguments (the data as `data=`, MERGE clauses as `clauses=`, and
  method names such as `plan_write` or `z_order`) and derives the request the
  call itself routes on, so the engine it names is the engine the call uses.
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
- `append`, `overwrite` and `replace` return an `OperationResult` too, where
  they returned None (delta-rs#3952): `version` committed, `num_files`,
  `num_rows` and `num_bytes` added, `num_removed_files`, and `.engine`, read
  back from the commit on the kernel and delta-rs. A write the call skipped
  (a `txn` already committed, an empty dynamic overwrite) has `skipped=True`,
  zero counts and no `version`. Through the warehouse only `.engine` is known.

- One error-translation boundary around every engine: nothing but a
  `DeltaSwampError` reaches a caller, from a call or later from the stream or
  MERGE builder it returned. What no rule recognises is the new `EngineError`
  (`.engine`, `.operation`, `.original`), still an instance of the original's
  class where that class can be combined (a `pyarrow.ArrowInvalid`, a
  `KeyError`; not delta-rs's Rust `DeltaError` family, which maps to
  `TableNotFoundError`, `EngineLimitError`, `InvalidArgumentError` or the new
  `CommitRefusedError`). `StorageError` stays a `TimeoutError` or
  `ConnectionResetError`; a truncated or replaced data file is
  `CorruptTableError`. An exception your own data source raises reaches you
  as it was raised. The extension's exceptions carry a stable `kind` code.
  Malformed input (an unparseable timestamp, a missing column, an unknown
  table feature, contradictory `partition_overwrite`/`predicate`, a misspelt
  keyword, data that is not a table) is `InvalidArgumentError`, no longer an
  `UnreachableTableError` that read as "no engine can serve this".
- Data files the kernel writes (appends, overwrites, DML rewrites, OPTIMIZE,
  Z-ORDER, distributed workers) are snappy-compressed, as Spark's and
  delta-rs's are, or use the table's `delta.parquet.compression.codec`
  (`zstd`, `gzip`, `lz4`, `lz4_raw`, `brotli`, `uncompressed`; `lzo`, which
  arrow-rs cannot write, as snappy). They were uncompressed, and an OPTIMIZE
  grew a table 2-4x.

- Every commit writes its version checksum (`_delta_log/<version>.crc`), as
  Spark and Databricks do (delta-rs#4190, delta-kernel-rs#1781): kernel
  commits from the post-commit snapshot, delta-rs commits and raw metadata
  commits afterwards, through the kernel's `Snapshot::write_checksum`. A
  commit that changes no file (SET TBLPROPERTIES, ADD COLUMNS, VACUUM
  START/END), which the kernel gives up on, carries the previous checksum
  forward. Only when cheap -- the previous checksum at most 100 commits back,
  or a log under 100 commits with no checkpoint -- so a long history that
  never had one does not start a chain; never on a catalog-managed table (the
  catalog's writer keeps those); and never at the cost of the write.

- `Table.added_since(version, until=None, ...)` reads the rows added after a
  version without the change data feed (delta-rs#4554,
  delta-kernel-rs#1177), from the kernel's incremental scan bound as
  `Snapshot.incremental_files`. It refuses a range that removed files (DML,
  OPTIMIZE, overwrite) unless `only_appends=True`; `can("added_since")`
  says so. `changes(start, include_snapshot=True)` yields the table at
  `start` as inserts before the feed.

- A handle pinned to a version (`conn.table(..., version=n)`) can write
  (delta-rs#4417): `append` appends at the latest version (it reads nothing),
  and `delete`, `update` and `merge` read version n and commit at the latest
  through the kernel's deletion-vector conflict check against every commit
  since, raising `CommitConflictError` when one changed what they read.
  Other writes, tables without deletion vectors and catalog-managed tables
  still refuse, and `can()` agrees.

- `clone(target)` of a path table to a storage path is written by the kernel,
  with no warehouse (delta-rs#2456): version 0 with the source's protocol,
  metadata and clustering, a `CLONE` commit naming the source and its
  version, and the source's files by absolute URL (deletion vectors made
  absolute) for a shallow clone, or copies of them for `shallow=False`.
  VACUUM of a path shallow clone is refused. The direct engines do not read a
  shallow clone (their reads stay confined to the table root); Spark and
  Databricks do.

### Security

See docs/usage.md, "Security notes".

- Every SQL fragment forwarded to an engine (predicates, SET/INSERT values,
  MERGE ON and conditions, replaceWhere, CHECK constraints, generation and
  DEFAULT expressions in warehouse DDL) must be exactly one expression;
  malformed text raises `PredicateError` before routing. delta-rs acted on a
  prefix of `id = 1) AND (...` and deleted more rows.
- The kernel MERGE evaluates its clauses in a sandboxed DuckDB, one statement
  per execute: a SET value could read local files and an ON condition could
  run another statement.
- Azure endpoints are derived only under Azure Storage's domains, and only
  for the account the options name; other hosts need an explicit
  `azure_storage_endpoint`. The connection's SAS or AAD token went to any
  host a location named. Vended R2 keys go only to Cloudflare.
- Kernel reads and compaction refuse data files and deletion vectors outside
  the table root (`../`, `%2E%2E/`, another prefix or bucket).
- `create_table` / `write_table` refuse a directory holding non-Delta files,
  a managed table's staging location on local disk is refused unless the
  catalog is local, and a full VACUUM deletes only Delta-named orphans. A
  default VACUUM deleted every unrelated file in such a directory.
- The OSS UC client keeps its token on the catalog's origin across redirects,
  refuses https-to-http redirects, and warns on a plain-http remote catalog.
- Delta Sharing downloads require https to public addresses, checked on
  connect and on every redirect (`DELTASWAMP_SHARING_ALLOW_PRIVATE_URLS=1`
  for a local server).
- The Databricks SDK DEBUG log redaction covers hyphenated and nested secret
  fields, Authorization headers, URL signatures and child loggers.
- Pickled providers, catalogs, tables and connections carry no PAT or client
  secret unless `connect(ship_credentials=True)` or a plan's
  `ship_catalog_auth=True` asks for them. **Breaking:** a worker now needs
  Databricks auth of its own (for example `DATABRICKS_TOKEN`), even when the
  driver passed `token=`.
- `.` and `..` name parts are percent-encoded in Unity Catalog REST paths.

### Known limits

- `INTERVAL DAY TO SECOND` columns read as `duration[us]` and year-month
  intervals as Spark's text (`INTERVAL '1-2' YEAR TO MONTH`) on every engine,
  warehouse included; both append back unchanged. A duration nested in a
  struct, list or map cannot be staged for the warehouse. SQL naming an
  interval column (a predicate, an UPDATE or MERGE value) needs the
  warehouse; the lazy hand-offs filter such columns on the values they show.
- A predicate on a `CHAR(n)` column needs the SQL fallback: Spark compares
  CHAR values padded to `n`, the direct engines compare bytes.
- Time travel to a timestamp after the latest commit is refused
  (`InvalidArgumentError`), as Databricks refuses it.
- A `Table` opened by catalog name re-resolves the name on every read, as it
  does before every write: after DROP TABLE a read raises
  `TableNotFoundError`, and after a re-create under the same name
  `CorruptTableError`. A handle pinned to a version reads that snapshot.
- GEOMETRY and GEOGRAPHY columns are written through the warehouse only, as
  EWKT/WKT text or WKB with the column's own SRID; text naming another SRID
  fails to parse rather than being relabelled. No direct engine reads them.
- PyIceberg (0.12) cannot parse Iceberg v3 metadata holding a VARIANT, so the
  Iceberg engine refuses such a table and the Delta engines or the warehouse
  serve it. Databricks cannot SHALLOW CLONE a managed Iceberg table;
  `clone(target, shallow=False)` makes a deep one.
- `can()` refuses up front what the warehouse would: values for a GENERATED
  ALWAYS AS IDENTITY column, `when_matched_update_all()` over a source
  carrying an identity column, `drop_feature("checkConstraints")` while a
  constraint exists, and ADD COLUMNS of a name with `' ,;{}()\n\t='` on a
  table without column mapping. A missing privilege is `SqlPermissionError`
  (a `SqlStatementError` and a `PreflightError`) naming the grant.

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
- Concurrent OPTIMIZE and Z-ORDER no longer duplicate rows. delta-rs's
  conflict check ignores a concurrent compaction's `dataChange=false`
  removes, and its OPTIMIZE takes neither `max_commit_retries` nor app
  transactions, so two runs over the same files both committed (150 rows
  became 450). Every compaction is now committed by the kernel on the
  snapshot it was planned from, and a loser re-plans from the new snapshot;
  delta-rs never commits one (its after-the-fact rollback undid only the
  newest duplicate). CHECK constraints, generated and identity columns,
  invariants and the writer versions 3-6 that imply them (every table delta-rs
  gave a change data feed) no longer stand in the way: a compaction writes back
  the values it read. The kernel's OPTIMIZE is its own capability, so
  kernel-only connections and tables delta-rs cannot open (in-commit
  timestamps, type widening, `vacuumProtocolCheck`) compact too. Refused, typed
  and before any work, rather than run unsafely: `writer_properties`,
  `min_commit_interval`, app transactions and `cleanup_expired_logs=True`.
  Row-tracked tables compact with every row's id and commit version kept
  (written into the materialized columns, as Databricks' OPTIMIZE writes
  them), and overwrite with fresh ids for the new rows; liquid-clustered
  tables are Z-ordered over their clustering keys (`full=True` rewrites every
  file); name/id column-mapped tables compact too (DuckDB's delta reader reads
  the partition columns of every column-mapped table as NULL, Databricks'
  own included). Z-order is incremental through Databricks' `ZCUBE_*` tags
  (`min_cube_size`), and bin-packing takes `min_file_size` and `sort_by`. An
  OPTIMIZE that loses to a protocol change it cannot write raises
  `CommitConflictError`.
- OPTIMIZE reads each step's files with one scan, in bin order and several at
  a time, and writes several output files at once (20,000 files: 16 s before,
  3 s now, delta-rs 2.2 s; 5,000 commits without a checkpoint: 173 s, now
  10 s). Its rows stream into the commit: memory is bounded by
  `KernelEngine.compaction_max_file_bytes` (512 MiB of decoded rows per output
  file, and per Z-order sort; about four times that at peak) whatever the table
  or partition size. Where rows decode to more than that per target-sized
  file, output files come out smaller than the target and are left alone
  afterwards. A Z-order sorts a large partition in chunks of that size, and
  sizes its files by how the sorted rows encode (they were 40% over the
  target).
- `partition_filters` on the kernel's OPTIMIZE follow delta-rs: timestamp
  values in ISO 8601 with a `T` and an offset, `<`/`>` on timestamps, a NULL
  partition matching no comparison (`!=` and `not in` included), and `= ''`
  meaning NULL. A value that is not of the column's type is an
  `InvalidArgumentError`. Z-ORDER by a STRUCT, ARRAY or MAP column is refused
  (`InvalidArgumentError`) for every engine.
- Kernel commits record `operationParameters` (mode, partitionBy, predicate;
  OPTIMIZE's zOrderBy) and `isBlindAppend`. A concurrent winner counts as a
  blind append only when its `isBlindAppend` says so, or, where the writer
  records none (delta-rs), when it is a WRITE in `Append` mode: a replaceWhere
  that matched nothing (only adds) had read as one, so concurrent loads of one
  range on a deletion-vector table each committed (four writers, four copies).
- A kernel MERGE, DELETE or UPDATE that lost a commit race rebases only over
  blind appends and over commits that added nothing its read could match;
  concurrent MERGE upserts of one new key each inserted it before. The
  copy-on-write DELETE/UPDATE rebases over blind appends too (it conflicted on
  every concurrent commit), and delta-rs's "Metadata changed since last
  commit" is a `MetadataChangedError`, as the kernel's is.
- Concurrent `create_table` on one path has exactly one winner; delta-rs used
  to retry the loser at version 1 and replace the winner's schema. The loser
  gets "a Delta table already exists there".
- `wasb://` and `wasbs://` locations are refused; use `abfss://`.
- A path table with a GCS OAuth bearer token in `storage_options` is served by
  the kernel only: history, MERGE and log cleanup, which only delta-rs
  implements, are refused on it.

- `convert_to_delta` refuses a hive-partitioned directory whose partition
  values are escaped (`region=a%20b`): delta-rs records those paths unencoded
  and the converted table cannot be read. Rewrite the data with `write_table`.
- VACUUM (full and LITE, dry runs included) on the tables delta-rs cannot
  commit to -- liquid clustering, row tracking, in-commit timestamps, type
  widening, `vacuumProtocolCheck`, collations -- and on deletion-vector
  tables is planned from the kernel's log replay as Spark's VACUUM plans it:
  live files, their deletion vectors, files of removes within the retention
  (checkpoint tombstones included) and recent change data are kept; hidden
  paths are skipped; a real VACUUM commits VACUUM START/END. It was refused
  outright on the first group, and delta-rs kept every vector file on the
  second. Only Delta-named files are deleted, as before.
- RESTORE of deletion-vector tables (delta-rs left DV changes in place,
  delta-rs#4613), column-mapped tables and kernel-only tables is committed
  here as Spark's RESTORE: the target's files come back as logged (row ids
  kept), the current-only ones are removed, the target's schema and
  properties are restored (keeping `maxColumnId`, in-commit-timestamp and
  row-tracking settings), and a vacuumed target file raises
  `MissingDataFileError` unless `ignore_missing_files=True`. Refused up front
  (`can("restore", target=n)`) across a change of column-mapping mode,
  partition columns or table id, or to a version needing a dropped feature.
- Adding a NOT NULL column is refused on every engine, as Databricks refuses
  it: add it nullable, backfill, then `set_not_null()`.
- On a kernel-only table without deletion vectors, DML is a whole-table
  rewrite bounded by `KernelEngine.rewrite_max_bytes` and refused on
  row-tracked tables, and MERGE needs the SQL fallback.
- OPTIMIZE of a liquid-clustered table is a Z-order over its keys, not
  Databricks' incremental clustering tree (whose state Databricks keeps in its
  own domain); Databricks reclusters such files on its next OPTIMIZE. On a
  row-tracked table the kernel's post-commit snapshot and checksum delta do
  not count the removes the native commit stages itself; nothing reads them.
- The kernel cannot write CDC files, so UPDATE and MERGE on a change-data-feed
  table it alone can write need the SQL fallback. DELETE through deletion
  vectors needs none.
- The change feed fails when the range crosses an incompatible schema change
  (a dropped, renamed or retyped column), with `ChangeFeedSchemaChangeError`
  naming the version. Start the range at the change, or read it through the
  warehouse. An added column reads as null in older rows, also in a range
  that ends before it was added (as `table_changes()` returns it; on
  column-mapping tables the end version's schema, as Delta reads those), and
  `changes()` yields each version under the schema it was written with. A range crossing
  a version with the feed off raises `UnreachableTableError` naming it
  (`.version`); `changes()` yields the versions before it first. Every
  version's schema is compared in a range of up to 200 versions, so a change
  the range reverses (ADD COLUMN, then a RESTORE to before it) is refused up
  front too; in a longer range such a change is found only once the read
  reaches it, and a consumer of `__arrow_c_stream__` (`pa.table(t.cdf(...))`,
  polars, DuckDB) gets that error's message as an `ArrowInvalid`. With the SQL
  fallback, a range the direct engines stop at (a column-mapping RENAME or
  DROP) goes to the warehouse's `table_changes()`, which serves what it can.
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
- FLOAT and DOUBLE columns holding a NaN keep no Parquet footer min/max in
  files deltaswamp writes (Spark writes none either; arrow-rs's leave the NaN
  out, and Databricks then skipped NaN rows for `f = 'NaN'` or `f > 100`).
  The kernel checks each file; delta-rs, whose log stats come from the
  footer, checks an append of in-memory data and writes no float min/max
  (null counts included) for streamed appends and every DML rewrite. Files
  written before this change may still be skipped wrongly for NaN
  predicates until rewritten.
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
  table holding such values (or INT96 timestamps before 1900 or after
  2262-04-11, which delta-rs decodes as overflowing nanoseconds), since it
  would rewrite them shifted; they take the kernel's deletion-vector or
  whole-table rewrite paths, or the warehouse, or are refused. Reads of such a
  table never go to delta-rs: the kernel reads, and a predicate outside its
  grammar is evaluated by DuckDB afterwards (refused without DuckDB).
- Every data file the kernel writes names `org.apache.spark.version` in its
  Parquet footer. Databricks SQL warehouses read a file without it with
  Spark's legacy calendar rebase, so dates before 1582-10-15 (and
  timestamps) that deltaswamp wrote read there 2-10 days off (`0001-01-01` as
  `0001-01-03`), and OPTIMIZE turned Spark files Databricks read right into
  files it misread. delta-rs cannot write the key: writes whose rows hold
  such a value, and delta-rs rewrites of tables whose files may, go to the
  kernel or the warehouse, or are refused (a MERGE into a table without
  deletion vectors). The same holds for UPDATE and MERGE SET/INSERT values
  that are not provably after the limits (`DATE '1000-01-01'`, arithmetic,
  functions), and for streamed writes with DATE or TIMESTAMP columns, which
  cannot be inspected. The file check fails closed. Files delta-rs
  already wrote stay misread by Databricks until rewritten (`optimize()`).
- Lazy hand-offs (`to_duckdb`, `to_polars(lazy=True)`, `to_pyarrow_dataset`,
  `Connection.sql`) read the version current when they were made;
  `follow_latest=True` follows the latest and raises `MetadataChangedError`
  if the schema changed. A pinned hand-off of a version whose files were
  vacuumed fails in Polars as a `ComputeError` naming `MissingDataFileError`
  (Polars wraps every error an IO source raises).
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
