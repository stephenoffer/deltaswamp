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

- On a kernel-only table without deletion vectors, DML is a whole-table
  rewrite bounded by `KernelEngine.rewrite_max_bytes` and refused on
  row-tracked tables, and MERGE needs the SQL fallback.
- The kernel cannot write CDC files, so UPDATE and MERGE on a change-data-feed
  table it alone can write need the SQL fallback. DELETE through deletion
  vectors needs none.
- The kernel's change feed fails mid-stream when the range crosses a schema
  change. Start the range after the change, or read it through the warehouse.
- The change feed and history of a catalog-managed table need the warehouse.
- Incremental reads without a change feed are not built.
- Databricks allowlists which connectors may write through the Unity Catalog
  Delta API, so writes to managed tables go through the SQL fallback until
  deltaswamp is registered. Reads are unaffected.
- Idempotent writes are checked against the last committed version before
  writing, so a concurrent writer can still commit in between.
