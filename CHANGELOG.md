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
  with picklable credential providers so workers vend their own credentials.
- Hand-offs to DuckDB, Polars and Daft, and cross-catalog SQL through
  `Connection.sql`.

### Known limits

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
- The change feed fails when the range crosses an incompatible schema change,
  with `ChangeFeedSchemaChangeError` naming the version. Start the range after
  the change, or read it through the warehouse.
- Commits the kernel writes, including distributed ones, record empty
  `operationParameters`: delta_kernel 0.28 overwrites whatever the engine
  supplies. `isBlindAppend` still tells an append from an overwrite.
- A MERGE on a change-data-feed table whose last NOT MATCHED clause has a
  condition is refused on delta-rs (1.6.5 inserts an all-NULL row per rejected
  source row) and needs the SQL fallback.
- DELETE/UPDATE/replaceWhere SQL beyond the kernel's predicate grammar
  (arithmetic, function calls) needs delta-rs or the warehouse, so on a
  catalog-managed table it needs the SQL fallback.
- The change feed and history of a catalog-managed table need the warehouse.
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
