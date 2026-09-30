# Delta tooling for Ray Data

What a Ray Data `read_delta` / `write_delta` builds on, and where each
scenario that fails with `deltalake` alone is solved. deltaswamp does not ship
the Ray datasource or datasink itself (beyond `to_ray_dataset`); it provides
the plans they run.

## The contract

| Step | Where it runs | API |
|---|---|---|
| Plan a read | driver | `Table.plan_scan(columns=, predicate=, version=)` → `ScanPlan`; `plan.partitions(n)` |
| Read a split | worker | `plan.read(splits)` / `plan.stream(splits)` — no log listing or replay, no catalog call |
| Plan a change-feed read | driver | `Table.plan_changes(start, end)` → `ChangesPlan` (runs of whole commits) |
| Plan a write | driver | `Table.plan_write(mode=, txn=, credential_source=, identity_tasks=, domain_metadata=)` → `WritePlan`; `Connection.plan_write(name, schema=...)` for a table that may not exist yet |
| Write a block | worker | `plan.write(data, task_index=i)` → fragment bytes |
| Combine fragments | worker (optional) | `merge_fragments(fragments)` — tree-combine before the driver |
| Commit | driver | `plan.commit(fragments)` — one commit, rebased and retried, idempotent |
| Give up | driver | `plan.abort(fragments)` — deletes the job's files (a datasink's `on_write_failed`) |

The guarantees a datasink relies on:

- **Refused at plan time, never after the job ran.** Table features either
  engine cannot write, missing write credentials
  (`ExternalWriteNotAllowedError` for `EXTERNAL_WRITE_NOT_ALLOWED_FOR_TABLE`),
  defaults only Databricks evaluates, expressions DuckDB cannot evaluate,
  GENERATED ALWAYS identity columns without reserved values, system domains.
  Only a concurrent change to the table's schema can still refuse a commit,
  and then the files are aborted.
- **No orphan Parquet.** `commit()` deletes the job's files after a failure
  that certainly committed nothing (schema changed, guarded overwrite moved,
  a conflict on every attempt). After an ambiguous failure (timeout, 5xx) it
  keeps them and says so: committing the same fragments again is safe,
  because a commit that already landed is found and its version returned.
- **Driver work is O(job), not O(table).** The landed-files check reads only
  the commits since the write; overwrite removes carry no statistics;
  fragments concatenate as they arrive.
- **Workers never replay the log** and never reach the catalog. Splits carry
  the kernel's scan row for their file and one shared planned snapshot.
- **Credentials refresh inside a running task.** The native object store
  reads a credential slot on every request; workers refresh it from
  `credential_source` (a driver-side `CredentialBroker`, e.g. in a Ray actor)
  or from shipped catalog auth, one vend per process.

## Status by scenario

Tests are under `tests/integration/` unless a path says otherwise. The `live/`
ones run against a Databricks workspace (`make test-live`): reads in
`test_live_ray_tooling.py` run in a separate process from a pickled plan, as a
Ray worker would, and each interop case builds a table through distributed
plans and has a SQL warehouse read it, query it and write on top
(`DELTASWAMP_TEST_INTEROP=1`, see [Testing](testing.md#the-interop-suite)).

| # | Scenario | Status | Implementation | Tests |
|---|---|---|---|---|
| 1 | Read a catalog-managed table (UC log tail, `max_catalog_version`; delta-rs #4549) | ✅ | `catalog/databricks.py` fetches tail + max version; `KernelEngine.snapshot` resolves with them; splits pinned to the version, read on workers with no catalog call (`Snapshot.planned`, `crates/native/src/snapshot.rs`) | `test_catalog_managed.py::TestCatalogManaged`, `test_distributed.py::TestCatalogManaged`, `test_planned_scan.py::TestCatalogManaged::test_reads_after_the_staged_commits_are_removed`, `live/test_live_ray_tooling.py::TestPlannedRead::test_a_catalog_managed_table_is_read_by_a_worker` |
| 2 | Read a GCS-backed UC table with UC-vended Google OAuth tokens | ✅ kernel; delta-rs refused at plan time | `credentials/databricks.py` maps the token to `google_bearer_token`; `crates/native/src/store.rs` bearer provider, refreshed through a credential slot (`credential_slot.rs`) | `unit/test_credentials.py::test_gcs_token_becomes_the_store_layer_key`, Rust `store.rs::gcs_bearer_token_is_sent_on_requests`, `test_credential_refresh.py::TestRefresh` |
| 3 | Row filters / column masks | ✅ refused at plan time, naming the policy; `to_ray_dataset` no longer silently reads through the warehouse on the driver (`allow_driver_read=True`, size-bounded) | `router.py` access-policy refusal; `table.py` `to_ray_dataset` | `unit/test_router.py::test_access_policy_is_named_in_the_refusal`, `test_planned_scan.py::TestRayNeverReadsOnTheDriverUnasked` |
| 4 | Change data feed | ✅ distributed plan; refused at plan time on catalog-managed tables (warehouse fallback on the driver) | `Table.plan_changes` / `ChangesPlan` (`distributed.py`); kernel CDF refusal for catalog-managed | `test_planned_changes.py`, `test_bughunt_kernel.py::test_cdf_on_catalog_managed_is_refused_up_front`, `live/test_live_ray_tooling.py::TestPlannedRead::test_a_worker_reads_the_change_feed` |
| 5 | Workers refresh expired vended credentials mid-job (Azure SAS/bearer, AWS, GCS) | ✅ | `credential_slot.rs` (the store reads the slot per request), `credentials/refresh.py`, `credentials/broker.py` (`credential_source=`), per-process provider sharing; plan-time `CredentialExpiryWarning` when workers cannot refresh | `test_credential_refresh.py` (`TestRefresh`, `TestSharing`, `TestCredentialSource`), Rust `credential_slot.rs` tests |
| 6 | Managed table without catalog commits (`EXTERNAL_WRITE_NOT_ALLOWED_FOR_TABLE`) | ✅ refused by `plan_write` with the options | `plan_write` vends READ_WRITE on the driver; `ExternalWriteNotAllowedError`; router manifest refusal | `test_credential_refresh.py::TestExternalWriteNotAllowed`, `unit/test_router.py::test_manifest_write_capability_blocks_writes_only`, `live/test_live_ray_tooling.py::TestPlannedWrite::test_a_managed_table_is_written_or_refused_at_planning` |
| 7 | Append to a catalog-managed table (UC committer: stage → ratify → publish) | ✅ retried by default | `commit.rs` UCCommitter; `WritePlan.commit` re-reads the tail per attempt | `test_catalog_managed.py::TestDistributedWrite`, `test_write_lifecycle.py` |
| 8 | Overwrite a catalog-managed table (removes + UC committer + rebase) | ✅ | `overwrite_removes` (`commit.rs`, stats dropped), guarded by the planned version; `allow_concurrent_overwrite` recomputes on a fresh tail | `test_catalog_managed.py::test_overwrite_replaces_in_one_commit`, `test_write_lifecycle.py::TestLandedCommits::test_an_overwrite_that_landed_is_not_committed_again` |
| 9 | Create by name (`cat.schema.new_table`) | ✅ the table appears at commit, with its data; nothing is registered before then. A failed or aborted job deletes its files and undoes the create. Another creator mid-job: `error`/`overwrite` fail and abort, `ignore` keeps theirs, `append` joins a same-layout table | `Connection.plan_write(name, schema=, mode=)` (`_create.py`): workers resolve a template version 0 kept outside the log (`crates/native/src/pending.rs`); path/external tables commit v0 then the data and register after; managed tables stage at plan time and register at commit | `test_create_by_plan.py`, `live/interop.py` case `create_by_plan-distributed` |
| 10 | In-commit timestamps (delta-rs #3253) | ✅ `inCommitTimestamp` first in commitInfo, monotonic | kernel transaction; `PatchingCommitter` key order | `test_table_feature_writes.py::TestInCommitTimestamps` |
| 11 | Row tracking on append (baseRowId, defaultRowCommitVersion, high-water mark; delta-rs #3254/#3928) | ✅ reassigned on every rebase | kernel transaction per attempt | `test_table_feature_writes.py::TestInCommitTimestamps::test_concurrent_distributed_appends_keep_row_ids_and_times_in_order`, `test_local_table.py::test_row_tracking_takes_a_distributed_overwrite` |
| 12 | Column mapping, UniForm / IcebergCompat | ✅ column mapping (physical names, field ids); IcebergCompat V3 written; UniForm and IcebergCompat V1/V2 refused at plan time (a write would stale the Iceberg metadata; `sync_iceberg` through the warehouse) | `router.py` UniForm refusal; `capability.py` | `test_audit_distrib.py`, `test_iceberg.py::test_uniform_writes_are_refused`, `unit/test_bughunt_w3_routing.py::test_uniform_table_refuses_direct_data_writes` |
| 13 | `checkpointProtection` (Delta 4.0+; delta-rs #4462) | ✅ writes, DML, checkpoints; log cleanup still refused | `crates/native/src/restate.rs` (`HISTORY_ONLY`) | `test_table_feature_writes.py::TestCheckpointProtection`, `live/interop.py` case `checkpoint_protection-databricks_created` |
| 14 | Identity columns | ✅ blocks reserved at plan time, one slot per task; the data commit changes no metadata | `Table._reserve_identity`, `engine/values.py`, `WritePlan.write(task_index=)` | `test_table_feature_writes.py::TestIdentityColumnsLocally`, `::TestIdentityColumnsDistributed`, `live/interop.py` case `identity-databricks_created` |
| 15 | CHECK constraints, generated columns, invariants | ✅ enforced on the worker before any file is written; unevaluable expressions refused at plan time | `engine/values.py` (DuckDB), constraint checks in `write_files` | `test_table_feature_writes.py::TestGeneratedColumnsAndInvariants`, `test_audit_schema.py::test_a_distributed_write_enforces_the_constraints`, `live/interop.py` case `generated_defaults-distributed` |
| 16 | Deletion vectors emitted; appends to DV / CDF tables | ✅ DVs authored by DELETE/UPDATE/MERGE/replaceWhere; distributed append and overwrite of CDF tables | `dml.rs`, kernel DV DML | `test_deletion_vector_dml.py`, `live/test_live_ray_tooling.py::TestPlannedRead::test_a_worker_applies_databricks_deletion_vectors`, `test_table_feature_writes.py::test_a_distributed_overwrite_of_a_change_feed_table_reads_back_as_changes`, `live/interop.py` case `cdf_overwrite-distributed` |
| 17 | Domain metadata; liquid clustering, variant, type widening; geospatial | ✅ user domains via `plan_write(domain_metadata=)`; clustered/variant/widened tables written; geospatial refused at plan time | `KernelEngine.domain_metadata_refusal`, `commit.rs` | `test_table_feature_writes.py::TestDomainMetadata`, `live/interop.py` case `domain_metadata-distributed`, `test_audit_dialect.py::test_planned_scan_and_write_take_variant_as_json_text`, `unit/test_capability.py::test_geospatial_is_not_kernel_readable` |
| 18 | Multi-table jobs: per-table credentials (delta-rs #4425) | ✅ one provider per table; S3 region per bucket; environment endpoints never get vended keys | `catalog/databricks.py`, `credentials/databricks.py`, `_storage.py` | `unit/test_audit_cloud.py::test_vended_credentials_replace_every_user_credential_key`, `test_credential_refresh.py::TestBucketRegion` |
| 19 | Write a GCS-backed table | ✅ kernel write + commit through the bearer store | `store.rs`, credential slot | Rust `store.rs` tests; no GCS emulator test (see below) |
| 20 | Rebase and retry on every commit path | ✅ distributed append/overwrite, local append, DML, OPTIMIZE, RESTORE, metadata commits; catalog-managed paths re-read the catalog tail | `WritePlan.commit`, `KernelEngine._refresh_tail`, `_rebase_dv_commit` | `test_bughunt_w4_distwrite.py::test_racing_distributed_appends_all_land_by_default`, `test_catalog_managed_dml.py::TestDmlRebasesOverABlindAppend`, `::test_optimize_rebases_over_a_blind_append`, `::test_restore_rebases_over_a_concurrent_append` |
| 21 | UPDATE, DELETE, MERGE, OPTIMIZE, VACUUM, RESTORE on catalog-managed tables | ✅ all through the UC committer; real VACUUM needs `allow_catalog_managed=True` (catalog policy) | `engine/kernel.py`, `kernel_merge.py`, `Snapshot.commit_actions` (`commit.rs`) | `test_catalog_managed_dml.py`, `test_uc_lifecycle.py::TestCatalogManagedWithoutDatabricks` |

## Scale

Measured locally (see the commit messages for method):

| What | Before | After |
|---|---|---|
| Driver peak RSS committing 20k fragments (21 columns) | +622 MiB | +160 MiB |
| Worker task reading 20 files of a 100k-file table | 0.38 s, 98 MB | 0.11 s, 69 MB |
| `WritePlan.write()` resolves of a catalog-managed table, 4 calls | 4 | ≤ 1 |
| Vends for 8 tasks in one worker process (shipped auth) | 8 | 1 |

## Limits that remain

- **Databricks allowlisting.** Databricks admits catalog-managed commits and
  managed-table creation only from registered User-Agents. Until
  `deltaswamp/<version>` is registered, those need
  `allow_sql_fallback=True` on Databricks (a driver-side write). OSS Unity
  Catalog is unaffected.
- **An overwrite's commit file** is written in one put-if-absent, so the
  driver holds it whole (about 200 bytes per removed file). The removes
  themselves are produced as it is written, not held beforehand.
- **Kernel MERGE** runs in hash buckets over local spill files when its ON
  condition equates keys; one without such keys runs in memory. It needs
  local disk (`KernelEngine.dml_spill_directory`) of about the data it reads.
- **delta-rs operations** hold the credential they start with: deltalake
  (through 1.6.6) takes storage options, not a credential provider. Each
  operation vends afresh as it opens the table, and one that starts with
  less than `DeltaRsEngine.expiry_warning_seconds` (10 minutes) of
  credential life warns (`CredentialExpiryWarning`). Kernel operations
  refresh theirs as they run, and retry once when storage refuses one.
- **Create is two commits.** A new table's empty version 0 lands just before
  its data (the kernel cannot write a first version with data and one set of
  column-mapping names across workers). A driver that dies between the two
  leaves an empty table; any failure the driver sees undoes it. A managed
  table's staging allocation cannot be released (the catalog has no
  endpoint for it).
- **Emulators.** GCS and Azure SAS refresh are covered by Rust wire tests,
  not by fake-gcs-server/Azurite runs.
- **CDF on catalog-managed tables** outside Databricks: the kernel's
  `TableChanges` takes no catalog tail.
