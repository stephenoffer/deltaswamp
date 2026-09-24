# Changelog

Notable changes, newest first. This project follows [semantic
versioning](https://semver.org/); while the major version is 0, minor releases
may break the API.

## Unreleased

First working version.

### Added

- **A Python binding to delta-kernel-rs.** There was none, which is why no other
  Python library can open a `catalogManaged` table. `Snapshot.resolve` takes the
  catalog's ratified commit tail and its maximum trustworthy version, so a table
  whose newest commits exist only as staged files reads correctly.
- One connector object per table. `Connection` and `Table` cover Unity Catalog
  managed and external tables, Hive Metastore and Glue tables, and raw object
  storage paths.
- `capabilities()`, which reports in advance what can and cannot be done with a
  table, by which engine, and why not. Fallbacks are opt-in.
- Per-operation routing across two engines. delta-kernel reads 19 table features
  delta-rs refuses to open; delta-rs has MERGE, OPTIMIZE, VACUUM and RESTORE,
  none of which the kernel implements.
- Catalog-managed writes: append, full overwrite (every file removed in the same
  commit), publish, in-commit timestamps, domain metadata.
- Creation through the kernel for the properties delta-rs rejects, including
  `delta.feature.*` signals, row tracking, type widening, custom keys and liquid
  clustering.
- Every write mode: `replaceWhere`, dynamic partition overwrite, schema merge and
  overwrite, save modes via `write_table`, commit metadata, and idempotent writes.
- Unity Catalog credential vending with two refresh clocks, and providers that
  are picklable so a distributed worker mints its own rather than receiving a
  secret.
- GCS bearer tokens. delta-rs routes the vended OAuth token into
  `google_application_credentials`, which `object_store` reads as a file path.
- Conformance matrices as tested data: 34 table features, 33 operations, 30 table
  properties. `docs/conformance.md` is asserted against them.
- A live Databricks suite authenticated with a personal access token, and
  `tests/fake_uc.py`, a Unity Catalog server speaking the real `/delta/v1`
  protocol so the catalog-managed path is testable without an account.

### Known limits

- **There is no way to author deletion vectors.** `Transaction::update_deletion_vectors`
  is absent from delta-kernel 0.28, so there is no commit hook to build on.
  `DELETE`, `UPDATE`, `MERGE` and predicate overwrites on a catalog-managed table
  route to the SQL fallback.
- **Checkpointing a catalog-managed table is impossible.** `Snapshot::checkpoint`
  deadlocks when bound through PyO3, on and off the shared runtime, against the
  default engine's background executor; delta-rs cannot open the table to do it
  instead. Their logs grow without bound. Checkpoint from Databricks periodically.
- Creating a *managed* table is refused: it needs the catalog to allocate storage
  through its staging-table API, and that flow needs a live catalog to verify.
- Registering an external table in Unity Catalog is refused rather than
  half-done. Writing the Delta log without calling `tables.create` would leave an
  orphaned log while appearing to succeed.
- Distributed scan planning is not implemented. `plan_scan` and `execute_scan`
  raise; the seams (serializable splits, picklable providers) are in place.
- Idempotent writes are enforced by this library, not by an engine. delta-rs
  records the transaction identifier and appends the same one again, so the last
  committed version is checked before writing. A concurrent writer can still
  commit in between.
