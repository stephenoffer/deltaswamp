# Conformance matrix

`python/deltaswamp/capability.py` is the machine-readable source of truth, and
`tests/unit/test_capability.py` asserts the code agrees with it. This document
explains what the tables mean and why the rows say what they say.

## Why coverage is data, not prose

A support matrix written in a README drifts from the code within a release.
Here every claim is a row in a dict, and every load-bearing claim has a test
that names the source it came from, so anyone can re-check it.

## The three tables

| Table | Contents |
|---|---|
| `FEATURE_SUPPORT` | all 34 table features x {kernel, delta-rs} x {read, write} |
| `OPERATION_ENGINES` | all 33 operations -> engines in preference order |
| `FEATURE_DEPENDENCIES` / `FEATURE_CONFLICTS` | what kernel enforces before a write |

## Facts worth re-reading before you change routing

`vacuumProtocolCheck` is a ReaderWriter feature. That one word costs delta-rs
the whole table: it blocks *reads*, not merely VACUUM, which is easy to miss
when the name so plainly suggests otherwise.

delta-rs has no support for `domainMetadata`. Row tracking and liquid clustering
both depend on it, so every row-tracked and every liquid-clustered table is
delta-rs-unwritable. A large share of modern Databricks tables fall in here.

The wire name for liquid clustering is `clustering`, not `clusteredTable`.

`collations` and `checkpointProtection` exist in Databricks but have no kernel
0.28 variant at all. Kernel classifies them as Unknown, which blocks writes on
both engines, so they route to SQL or get refused.

Two features run the other way. `checkConstraints` and `generatedColumns` are
cases where delta-rs is the more capable engine, so routing must never assume
kernel wins by default.

Unknown feature names must not raise. Kernel tolerates unknown writer-only
features on the read path and so must we, or the first table to adopt a feature
newer than this release becomes unreadable for no good reason.

## Write modes

Every mode, and which engine serves it. `deltaswamp` fills three gaps the
engines leave: dynamic partition overwrite is emulated with a generated
`replaceWhere`, idempotent writes are enforced here because neither engine
deduplicates, and save modes come from `Connection.write_table`.

| Mode | API | Engine | Note |
|---|---|---|---|
| append | `t.append(data)` | delta-rs, kernel | kernel handles unpartitioned tables only |
| full overwrite | `t.overwrite(data)` | delta-rs | catalog-managed tables cannot be overwritten here |
| replaceWhere | `t.overwrite(data, predicate=...)` | delta-rs | |
| dynamic partition overwrite | `t.overwrite(data, partition_overwrite='dynamic')` | delta-rs | emulated; predicate built from the partition values in `data` |
| schema merge | `t.append(data, schema_mode='merge')` | delta-rs | routes as MERGE_SCHEMA, not APPEND |
| schema overwrite / RTAS | `t.replace(data)` | delta-rs | |
| create | `conn.create_table(name, schema)` | delta-rs, kernel | kernel takes over for properties delta-rs rejects, and for `cluster_by` |
| save modes | `conn.write_table(name, data, mode=...)` | both | `error`, `ignore`, `append`, `overwrite` |
| idempotent write | `t.append(data, txn=(app_id, version))` | enforced here | neither engine deduplicates; verified against delta-rs 1.6.5 |
| commit metadata | `t.append(data, commit_metadata={...})` | delta-rs | shows up in `history()` |
| DELETE / UPDATE / MERGE | `t.delete()`, `t.update()`, `t.merge()` | delta-rs | copy-on-write; catalog-managed tables need the SQL fallback |

### Where each operation routes

| Operation | Engines, in order |
|---|---|
| `append` | deltars, kernel |
| `overwrite` | deltars |
| `replace_where` | deltars |
| `merge_schema` | deltars |
| `create` | deltars, kernel |
| `delete` | deltars, sql |
| `update` | deltars, sql |
| `merge` | deltars |

## Table properties

delta-rs rejects roughly half the Delta property surface with a single opaque
message, and panics on `delta.minReaderVersion`. The kernel accepts nearly all
of it. `validate_properties` checks against this table first, so the failure
names the key and the remedy, and a create delta-rs cannot serve falls through
to the kernel automatically.

| Property | delta-rs create | delta-rs set | kernel create |
|---|---|---|---|
| `delta.appendOnly` | honored | honored | honored |
| `delta.autoOptimize.autoCompact` *(Databricks-only)* | stored, inert | rejected | n/a |
| `delta.autoOptimize.optimizeWrite` *(Databricks-only)* | stored, inert | rejected | n/a |
| `delta.checkpoint.writeStatsAsJson` | stored, inert | rejected | honored |
| `delta.checkpoint.writeStatsAsStruct` | honored | rejected | honored |
| `delta.checkpointInterval` | stored, inert | rejected | honored |
| `delta.checkpointPolicy` | stored, inert | rejected | honored |
| `delta.columnMapping.mode` | honored | rejected | honored |
| `delta.dataSkippingNumIndexedCols` | honored | rejected | honored |
| `delta.dataSkippingStatsColumns` | stored, inert | rejected | honored |
| `delta.deletedFileRetentionDuration` | honored | honored | honored |
| `delta.enableChangeDataFeed` | honored | rejected | honored |
| `delta.enableDeletionVectors` | honored | rejected | honored |
| `delta.enableExpiredLogCleanup` | stored, inert | rejected | honored |
| `delta.enableIcebergCompatV2` | rejected | rejected | n/a |
| `delta.enableIcebergCompatV3` | rejected | rejected | honored |
| `delta.enableInCommitTimestamps` | rejected | rejected | honored |
| `delta.enableRowTracking` | rejected | rejected | honored |
| `delta.enableTypeWidening` | rejected | rejected | honored |
| `delta.isolationLevel` | honored | honored | n/a |
| `delta.logRetentionDuration` | honored | honored | honored |
| `delta.minReaderVersion` | **crashes** | rejected | n/a |
| `delta.minWriterVersion` | honored | rejected | n/a |
| `delta.parquet.compression.codec` | rejected | rejected | n/a |
| `delta.parquet.format.version` | rejected | rejected | honored |
| `delta.randomizeFilePrefixes` *(Databricks-only)* | stored, inert | rejected | n/a |
| `delta.setTransactionRetentionDuration` | stored, inert | rejected | honored |
| `delta.targetFileSize` | honored | honored | n/a |
| `delta.tuneFileSizesForRewrites` *(Databricks-only)* | stored, inert | rejected | n/a |
| `delta.universalFormat.enabledFormats` *(Databricks-only)* | rejected | rejected | n/a |

Keys with no row follow three rules. A `delta.feature.<name>` signal is
rejected by delta-rs and accepted by the kernel for the sixteen features in
`KERNEL_CREATE_FEATURES`; `clustering` is deliberately not one of them, because
the kernel wants clustering columns through `cluster_by`. Any other unknown
`delta.*` key is rejected by both. A custom key outside the `delta.` namespace
is rejected by delta-rs and stored verbatim by the kernel.
