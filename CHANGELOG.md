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
- Conformance matrices as tested data: 34 table features, 46 operations, 30 table
  properties. `docs/conformance.md` is asserted against them.
- A live Databricks suite authenticated with a personal access token, and
  `tests/fake_uc.py`, a Unity Catalog server speaking the real `/delta/v1`
  protocol so the catalog-managed path is testable without an account.

### Added in the ecosystem pass

See `docs/ecosystem-audit.md` for the feature-by-feature comparison with
Databricks and the rest of the Delta ecosystem that drove this.

- **Managed-table creation and external-table registration** on Unity Catalog
  (Databricks and open source). A managed create runs the staging-table flow:
  the catalog allocates the id and storage, version 0 is written with that id,
  and the catalog finalises. `Connection.register_table` registers an existing
  log. Both used to be refused.
- **Metadata-only ALTER commits** written by this library for path tables.
  They cover what delta-rs cannot do: RENAME/DROP COLUMN under column mapping,
  enabling column mapping, type widening, SET NOT NULL (checked against the
  data), CLUSTER BY, UNSET TBLPROPERTIES, and the properties delta-rs rejects,
  with the protocol raised as values require. Committed as put-if-absent and
  recomputed on conflict.
- **Kernel predicates and timestamp travel.** One parsed SQL predicate drives
  both file skipping in the kernel and an exact row filter, so
  `scan(predicate=...)` and `scan(timestamp=...)` work on catalog-managed tables.
- **Kernel change data feed, file listing, partitioned appends and
  checkpoints.** The "deadlock" was the default engine's single-threaded
  executor waiting on itself; engines now use the multi-threaded executor, and
  catalog-managed tables can be checkpointed (after publishing).
- **DML on catalog-managed tables without Databricks**: DELETE, UPDATE and
  replaceWhere as a bounded whole-table rewrite through the kernel.
- **Delta Sharing** as a catalog and an engine (`sharing://profile`): reads,
  time travel, CDF and listing, via the `delta-sharing` client.
- **Iceberg** tables in Unity Catalog through its Iceberg REST endpoint with
  PyIceberg: reads, time travel, history, appends and overwrites.
- **The SQL fallback, rewritten** on the Statement Execution API in
  `databricks-sdk`, so the connector is no longer needed. Results come back as
  Arrow, values are parameterized, and a warehouse is chosen automatically. It
  serves every Databricks-only operation (ANALYZE, OPTIMIZE FULL, CLUSTER BY
  AUTO, REFRESH, SYNC METADATA, UNDROP, row filters and masks) and, with a
  staging volume, appends, overwrites and MERGE.
- **Governance**: `Table.info/grants/grant/revoke/tags/set_tags/set_owner/
  lineage/column_lineage/add_primary_key/add_foreign_key`, and
  `Connection.create_catalog/create_schema/volume/search_tables/list_functions`.
- **delta-rs surface completed**: drop constraint, table and column comments,
  DROP NOT NULL, log cleanup, file listing, CDF by timestamp with predicates,
  commit metadata on DML, VACUUM LITE, Z-ORDER through `optimize(zorder_by=)`.
- **Distributed reads**: `Table.plan_scan()` returns a picklable plan of
  per-file splits pinned to one version, and `to_ray_dataset()` reads it in
  parallel through a Ray Data datasource. The native file-restricted scan keeps
  deletion vectors, column mapping and partition values exact per split, on
  catalog-managed tables too.
- **Following changes**: `Table.changes(version, poll_interval=...)` yields one
  batch per committed version from the change feed.
- **Hand-offs**: `to_duckdb`, `to_polars(lazy=True)`, `to_ray_dataset`,
  `to_daft`, and `Connection.sql` for cross-catalog SQL on DuckDB or Polars.

### Added for distributed connectors

- **Distributed writes.** `Table.plan_write()` returns a picklable `WritePlan`;
  workers call `plan.write(batch)` to produce data files and an opaque
  fragment, and the driver calls `plan.commit(fragments)` to land every
  fragment in one transaction at a single version. Catalog-managed tables are
  included, so this reaches tables nothing else in Python can write. New native
  primitives `Snapshot.write_files` / `Snapshot.commit_files` carry the
  add-action metadata as Arrow IPC, which keeps kernel's own schema byte-exact
  instead of re-deriving one that would drift.
- Concurrency on a distributed commit follows the mode: an append lands on top
  of a writer that arrived while the job ran, while an overwrite against a table
  that has since moved is refused rather than silently discarding it, with
  `allow_concurrent_overwrite=True` to ask for last-writer-wins. `retries=`
  rebases a rejected commit, and a catalog-managed table says to re-open rather
  than spinning against the commit tail it captured at resolution.
- The refusal now happens on the driver, before any worker runs. The usual way
  a distributed Delta write fails is to discover at commit time that the table
  rejects it, after the compute is spent, leaving orphaned Parquet; `plan_write`
  raises there and then with the reason.
- **Legacy protocol versions are expanded to the features they imply.** Below
  reader 3 / writer 7 a protocol lists no features -- the version number is the
  feature set. Reading only the named list made such a table look featureless,
  so an engine accepted a write it could not perform and failed at commit. A
  `(2, 5)` table now correctly reports `checkConstraints` and is refused up
  front. `ResolvedTable.effective_reader_features` /
  `effective_writer_features` expose the merged view, and both engines use them.

### Found while auditing: what a distributed write can actually serve

- A distributed write is ruled in or out by the table's protocol **version**,
  not by the feature anyone is thinking about. Writer version 3 and above imply
  `checkConstraints`, which the kernel refuses whether or not a constraint
  exists, and enabling change data feed alone reaches version 4. The same
  features write fine at version 7, where only what the protocol names applies.
  `docs/usage.md` now states this, and a parametrised test pins both sides.
- A refusal now says when a blocking feature is merely implied by the version
  and the table neither names nor uses it, so the owner of a change-data-feed
  table is not sent looking for constraints they never wrote. It names only the
  features genuinely unused, so a table that does have one is not told it does
  not.
- Verified that a distributed write is portable: delta-rs reads back exactly
  what was written, for every table it can open at all. The three it refuses
  (deletion vectors, column mapping, type widening) it refused before the write
  too -- those are the tables this library exists to serve.

### Fixed: enforcement the kernel path could have skipped

Two features delta-rs evaluates itself, where letting the kernel take the write
would not raise an error -- it would write data that breaks the table's own
rules, which is worse.

- **CHECK constraints on a legacy-protocol table.** Such a table names no
  features at all: `minWriterVersion` 3 is the only evidence it has them.
  Reading the named list alone made it look featureless, so the kernel accepted
  the write and the constraint was never evaluated. The legacy expansion added
  in this release is what closes it, and a test now pins the whole chain: the
  named set is empty, `checkConstraints` is implied, writes route to delta-rs,
  and violating rows are rejected.
- **Features behind a kernel cargo flag this build does not enable.**
  `adaptiveMetadata-preview` and `geospatial` were recorded as partially
  supported, which is what kernel can do behind `adaptive-metadata-in-dev` and
  `geo-type-in-dev` -- flags `crates/native/Cargo.toml` deliberately leaves off.
  Partial is not `no`, so they passed the write check and failed at commit. The
  matrix now records what *this binary* can do, and both are refused up front.
- **Overwriting a row-tracked table.** A kernel overwrite removes every visible
  file in the same commit, and kernel 0.28 refuses a commit that stages removes
  on a row-tracked table because it cannot preserve the ids of what it removes.
  The guard existed but only covered the rewrite operations, so `overwrite`
  passed preflight and failed at commit -- after the data files were written,
  which in a distributed job means every worker had already done its work. Row
  tracking is now checked for every remove-staging operation. Appends are
  unaffected: they stage no removes and the kernel assigns fresh ids.
- **Tables that really carry a Delta invariant.** `invariants` is Supported in
  name only: the kernel refuses any write once a column has one, and it refuses
  *after* writing the data files -- so in a distributed job every worker does
  its work before anything says no. Routing on the feature name is far too
  blunt, since writer version 2 implies it for nearly every legacy table, so
  `ResolvedTable.has_invariants` now records whether the schema actually uses
  one and the kernel declines those writes up front. Reads are untouched.

### Fixed by adversarial review of the distributed-write surface

- **A fragment could be committed to the wrong table, corrupting it silently.**
  A fragment names its data files relative to the table they were written
  under, so committing one into another table wrote an add action pointing at a
  file that is not there: the commit *succeeded* and the table was unreadable
  from then on, with nothing in the log to say which write broke it. Fragments
  now carry the table root and metadata id in their Arrow schema metadata, and
  the commit refuses a mismatch. This also catches the subtler case of a table
  dropped and re-created at the same path while a job was in flight.
- **Commit failures reached callers as plain `RuntimeError`s from the
  extension**, not as the `CommitConflictError` / `BackfillRequiredError` that
  `errors.py` defines. `except DeltaSwampError` around a commit therefore
  caught nothing, which is exactly the case those types exist for -- and the
  429-means-publish distinction, carefully recovered in Rust, was lost before
  it reached anyone. Every kernel commit path now translates, and a transient
  failure raises the new `TransientCommitError`.
- **`add_column` had a different input contract per engine.** It took pyarrow
  fields on the kernel path and only `deltalake.Field`s on the delta-rs path,
  so the identical call worked or raised depending on which engine the router
  picked -- a difference a caller cannot see. Both now accept pyarrow fields and
  schemas, delta-rs fields, or a `{name: type}` mapping.

### Fixed

- Catalog-managed appends never reached the catalog: the UC commit ran outside
  the Tokio runtime its committer requires.
- `OVERWRITE` never routed to the kernel, so a catalog-managed table could not
  be overwritten despite the kernel implementing it.
- Properties read back stale after an ALTER: re-enrichment let the previous read
  override the log.
- A privilege check compared `str(Privilege.X)` against the SQL spelling and
  never matched, so 403s named the wrong missing privilege.
- The SQL fallback interpolated property values and update keys into SQL.
- Built-in catalogs no longer depend on installed entry-point metadata. Without
  it, `available_catalogs()` was empty and every `connect()` failed with "no
  catalog named 'databricks' is registered", which is what a source checkout, a
  vendored copy, a zipapp or a freezer produces. Entry points still take
  precedence, so a plugin can shadow a built-in name.
- CI could not pass as written: the lint and `python ${{ matrix.python }}` jobs
  ran the unit tests with neither the package nor its dependencies installed,
  and two test modules imported `pyarrow` and `databricks-sdk` at module scope,
  so collection failed outright. The package's own modules were already free of
  eager third-party imports; the tests now match.
- The packaging assertion in CI never learned about the `sharing` catalog added
  to `pyproject.toml`, so the native job failed on a correct build.
- `ruff` and `mypy` were installed unpinned in CI, so an upstream release turned
  the build red with no change here. Both are pinned to the versions
  `.pre-commit-config.yaml` already used.
- Type errors under `mypy --strict`: `mode` and `partition_strategy` reached
  delta-rs as `str` where it declares `Literal`, and `write_deltalake` is
  overloaded on `mode` such that only the overwrite signature accepts a
  predicate.
- `OSSUnityCatalog.drop_table` built the table name by interpolation rather
  than through the `_dotted` validator its five sibling methods use, so a path
  reference reached the server as `DELETE /tables/None.None.None` instead of
  being refused locally.

Found by running the live suite against a real AWS workspace for the first time.
Each of these made a headline claim untrue against real Databricks, and none was
reachable offline:

- **No AWS region reached object_store.** Unity Catalog vends S3 keys but no
  region, so object_store assumed `us-east-1` and every bucket outside it
  answered a redirect with no `Location` header — an opaque "Generic S3 error"
  on *every* S3-backed UC table. The catalog now passes its metastore's region,
  falling back to `AWS_REGION`/`AWS_DEFAULT_REGION`.
- **Every UC managed table raised `CorruptTableError`.** `table_uuid` is the
  Delta log's `Metadata.id`, but the Databricks catalog populated it from UC's
  `table_id`, which is the securable's own UUID (it names the storage
  directory). The identity check therefore compared two unrelated namespaces:
  0 of 6 tables in a real metastore had them equal. Databricks exposes no Delta
  metadata id, so `table_uuid` is now left unset there and the check is skipped.
- **Tables whose names need quoting could not be opened.** `tables.get` and
  `tables.delete` passed `ref.full_name`, which is SQL-quoted; a backtick in a
  REST path is a literal character, so a table with a hyphen 404d. Both now use
  `_dotted`, as the rest of the file already did.
- **`head(n)` materialised the whole table and sliced it**, so it never returned
  on a large one — it hung the live suite on a 10 TiB table. It now consumes the
  scan stream and stops once it has `n` rows: `head(3)` on that same 10 TiB
  table returns in under four seconds.
- **No product User-Agent was sent.** The UC Delta API rejects a request whose
  User-Agent does not name the calling application, which is a 400 on staging
  tables and on every `/delta/v1` commit. SDK clients are now stamped through
  `deltaswamp._sdk`. Note this is necessary but not sufficient: Databricks
  allowlists which connectors may *write* through that API, and the refusal now
  says so instead of surfacing a raw 400. Reads of catalog-managed tables go
  through the same API and are unaffected.
- Streaming tables and materialized views whose manifest advertises external
  read, but for which Unity Catalog exposes no storage location, were refused
  with each engine reporting "no storage location" and no remedy. The router now
  names the real cause once, with the SQL fallback as the remedy. In the test
  workspace that is 220 tables.
- **No catalog-managed table could be read on Databricks.** The `/delta/v1`
  response spells its fields in kebab-case (`latest-table-version`, `file-name`)
  and nests `location` under `metadata`; the parser looked for
  `latest_table_version`. `max_catalog_version` was therefore always `None` and
  the kernel refused every such table with "Max catalog version is required when
  loading a catalog-managed table" — the one thing no other Python library can
  do. `tests/fake_uc.py` answered in snake_case, which is why this passed
  offline; it now speaks the real spelling, as its own create-table response
  already did. Both Unity Catalog backends share one tolerant parser.
- **`head(n)` through the SQL warehouse was a full table scan.** The warehouse
  computes a result set before streaming any of it, and `scan` issued a bare
  `SELECT *`, so `head(3)` on a real table ran past the five-minute statement
  timeout. `limit` is now part of the engine scan protocol: the warehouse turns
  it into a real `LIMIT`, streaming engines ignore it because the caller stops
  reading anyway. The same `head(3)` now returns in 1.5s.
- **A metadata-only commit was allowed on a table carrying a writer feature the
  kernel cannot model at all.** The Delta protocol is explicit that a writer
  must not write to such a table, and "it writes no data" is the wrong test: a
  feature like `checkpointProtection` governs which checkpoints may be removed,
  so committing blind risks corrupting history rather than losing an edit. A
  feature the kernel can neither read nor write now blocks metadata commits too,
  while one it understands but cannot write data for (`identityColumns`,
  `generatedColumns`) still permits them.
- A scan that begins with a nearly expired vended credential now raises
  `CredentialExpiryWarning` naming `plan_scan()` as the remedy. The object store
  is built once per snapshot, so such a read fails partway through with a bare
  403 from storage; mid-scan refresh remains unbuilt.

- **The SQL fallback could not serve a table Unity Catalog had withdrawn from
  credential vending**, which is exactly the case it exists for. The manifest
  flags describe *direct external engine* access, and a warehouse runs inside
  Databricks, but the refusal fired before the fallback was considered while
  naming `allow_sql_fallback=True` as its own remedy. Those refusals are now
  conditional on the fallback being unavailable, and direct engines are skipped
  for a non-vendable table rather than accepting the call and failing mid-flight
  on a `CredentialError`. On a real managed table this moves DELETE, UPDATE,
  OPTIMIZE, ANALYZE, ADD COLUMN and the comment/property DDL from unreachable to
  working.

### Changed

- `vacuum()` defaults to Delta's standard (full) VACUUM on every engine;
  `lite=True` asks for the log-only variant. It was delta-rs's lite mode before.
- The `sql` extra no longer installs `databricks-sql-connector`, and the unused
  `ossuc` extra is gone. New extras: `duckdb`, `daft`, `all`.

### Known limits

- **There is no way to author deletion vectors.** `Transaction::update_deletion_vectors`
  is absent from delta-kernel 0.28. DML on kernel-only tables is a whole-table
  rewrite bounded by `KernelEngine.rewrite_max_bytes`, refused on row-tracked
  tables; MERGE on those tables needs the SQL fallback.
- The change feed of a catalog-managed table needs the SQL fallback; the
  kernel's TableChanges takes no catalog commit tail.
- Incremental reads without a change feed (the kernel's `incremental_scan`)
  are not bound; `Table.changes()` needs CDF.
- Idempotent writes are enforced by this library, not by an engine. delta-rs
  records the transaction identifier and appends the same one again, so the last
  committed version is checked before writing. A concurrent writer can still
  commit in between.
- Parts of the SQL fallback and governance have been verified against fakes,
  not a live workspace: statement parameters inside `TIMESTAMP AS OF` and
  `table_changes()`, `INSERT ... BY NAME` / `WITH SCHEMA EVOLUTION`, the lineage
  response shape, and the Databricks staging-table response.
