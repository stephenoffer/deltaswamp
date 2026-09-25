# Testing

Three tiers, by what they need.

| Tier | Needs | Command |
|---|---|---|
| Unit | nothing | `make test-unit` |
| Integration (local) | the built extension | `make test` |
| Live Databricks | a workspace and a PAT | `make test-live` |

`make check` runs everything CI runs.

## Unit

`tests/unit` is pure Python: no Rust toolchain, no tables on disk. It covers
reference parsing, routing and the conformance matrices.

## Integration

`tests/integration` writes real Delta tables to a temp directory and reads them
back with both engines. The cross-engine tests matter most: delta-rs writes and
the kernel reads, and the reverse. Run `make build` first.

The catalog-managed path runs here without a Databricks account.
`tests/fake_uc.py` is a Unity Catalog server that speaks the real `/delta/v1`
protocol, and `tests/integration/test_catalog_managed.py` builds a table whose
newest commit exists only as a staged file, which only a catalog-aware reader
can open.

## Live Databricks, with a personal access token

`tests/live` is opt-in. It creates and drops tables in a real metastore, which
costs money.

```bash
export DELTASWAMP_TEST_DATABRICKS=1
export DATABRICKS_HOST=https://adb-1234.5.azuredatabricks.net
export DATABRICKS_TOKEN=dapi...                      # a personal access token
export DELTASWAMP_TEST_CATALOG=main
export DELTASWAMP_TEST_SCHEMA=deltaswamp_test
export DELTASWAMP_TEST_WAREHOUSE_ID=abc123           # optional; unlocks most tests
export DELTASWAMP_TEST_EXTERNAL_LOCATION=s3://...    # optional

pytest tests/live/test_live_databricks.py -v
```

`python -m tests.live.preflight` checks the setup in a few seconds. The suite
runs the cheapest checks first: the token, the metastore prerequisites,
resolution, vending, then reads and writes.

### What it covers

| Group | Checks |
|---|---|
| Personal access token | authenticates, is used when passed explicitly, never appears in an error or a `repr` |
| Pre-flight | external data access, target catalog and schema reachable |
| Resolution | managed table, capability manifest, table UUID against the log, missing table |
| Credential vending | credentials arrive, carry an expiry, stay out of `repr`, and are never pickled |
| Reading | full read, projection, time travel, history, Arrow PyCapsule export |
| Writing | append, cross-engine agreement, idempotent replay, commit metadata |
| Catalog-managed | commit tail, kernel read, delta-rs refusal, ALTER refusal, append |
| Refusals | every refusal carries a reason; anything claimed readable is read |
| SQL fallback | warns when it serves a request |
| Lifecycle | exists, list, drop |
| Conformance sweep | walks the whole schema and checks every claim against reality |

Each test creates what it needs under a random name and drops it afterward,
even when it fails. Anything your workspace cannot verify is skipped with a
reason.

### The shape matrix

`tests/live/test_live_matrix.py` builds one managed table per shape
Databricks produces and checks deltaswamp against the warehouse on each. The
shapes: deletion vectors present, a legacy protocol, partitioning, liquid and
automatic clustering, column mapping by name and id after renames and drops,
change data feed, identity, generated columns, constraints, defaults, type
widening, variant and shredded variant, TIMESTAMP_NTZ, collations, v2
checkpoints, UniForm, in-commit timestamps, checkpoint protection, nested
types, row filters, column masks, geometry, catalog-managed and empty. It
asserts that:

- direct reads agree with the warehouse row for row, or are refused with a
  reason, exactly for the shapes expected;
- fallback reads, predicates, time travel and the change feed agree;
- every claimed log read works, and `features()` matches `DESCRIBE DETAIL`;
- writes, maintenance, DDL and RESTORE through the warehouse land as the
  warehouse sees them.

It is opt-in on top of the live suite, because it creates around thirty tables
(all dropped afterward):

```bash
DELTASWAMP_TEST_MATRIX=1 pytest tests/live/test_live_matrix.py -v
```

A PAT cannot be refreshed, so anything that outlives it stops with what looks
like an authentication error. Use OAuth M2M in production.

### Prerequisites an admin must grant

Both are off by default, the caller cannot grant either, and together they
cause most first-run failures. `connection.preflight()` checks the first one; the second
shows up as a `CredentialError` on the first vend, with the grant named in the
message.

1. **External data access on the metastore**, an account-admin setting.
2. **`EXTERNAL USE SCHEMA`**, grantable only by the catalog owner:

   ```sql
   GRANT EXTERNAL USE SCHEMA ON SCHEMA main.deltaswamp_test TO `me@example.com`;
   ```

Catalog-managed tables need a third: external access to catalog-commit tables
is a Beta feature behind a workspace preview. Without it those tests skip.

### Fixture tables

The suite discovers what exists and skips the rest, so a partial schema still
gives useful signal. To exercise everything:

```sql
CREATE SCHEMA IF NOT EXISTS main.deltaswamp_test;

-- 1. Plain managed Delta: the common case.
CREATE OR REPLACE TABLE main.deltaswamp_test.managed_plain
  (id BIGINT, city STRING) USING DELTA;
INSERT INTO main.deltaswamp_test.managed_plain VALUES (1,'oslo'), (2,'lima');

-- 2. Catalog-managed: nothing else in Python can read this.
CREATE OR REPLACE TABLE main.deltaswamp_test.catalog_managed
  (id BIGINT, city STRING) USING DELTA
  TBLPROPERTIES ('delta.feature.catalogManaged' = 'supported');
INSERT INTO main.deltaswamp_test.catalog_managed VALUES (1,'oslo');

-- 3. Deletion vectors: exercises DV-aware reads.
CREATE OR REPLACE TABLE main.deltaswamp_test.with_dvs
  (id BIGINT, city STRING) USING DELTA
  TBLPROPERTIES ('delta.enableDeletionVectors' = 'true');
INSERT INTO main.deltaswamp_test.with_dvs VALUES (1,'oslo'), (2,'lima'), (3,'cairo');
DELETE FROM main.deltaswamp_test.with_dvs WHERE id = 2;   -- writes a DV

-- 4. Liquid clustering: carries domainMetadata, so delta-rs cannot write it.
CREATE OR REPLACE TABLE main.deltaswamp_test.clustered
  (id BIGINT, city STRING) USING DELTA CLUSTER BY (id);
INSERT INTO main.deltaswamp_test.clustered VALUES (1,'oslo');

-- 5. Row filter: credential vending refuses this outright, and the capability
--    manifest is the only way to know in advance.
CREATE OR REPLACE FUNCTION main.deltaswamp_test.only_odd(id BIGINT)
  RETURN id % 2 = 1;
CREATE OR REPLACE TABLE main.deltaswamp_test.row_filtered
  (id BIGINT, city STRING) USING DELTA;
INSERT INTO main.deltaswamp_test.row_filtered VALUES (1,'oslo'), (2,'lima');
ALTER TABLE main.deltaswamp_test.row_filtered
  SET ROW FILTER main.deltaswamp_test.only_odd ON (id);

-- 6. A view: no readable file surface at all.
CREATE OR REPLACE VIEW main.deltaswamp_test.plain_view AS
  SELECT * FROM main.deltaswamp_test.managed_plain;

-- 7. UniForm / Iceberg reads: writes must be refused, because the required
--    MSCK REPAIR ... SYNC METADATA can only be run from Databricks.
CREATE OR REPLACE TABLE main.deltaswamp_test.uniform
  (id BIGINT, city STRING) USING DELTA
  TBLPROPERTIES (
    'delta.enableIcebergCompatV2' = 'true',
    'delta.universalFormat.enabledFormats' = 'iceberg'
  );
INSERT INTO main.deltaswamp_test.uniform VALUES (1,'oslo');

-- 8. Shallow clone: borrows the source's files by absolute path.
CREATE OR REPLACE TABLE main.deltaswamp_test.shallow
  SHALLOW CLONE main.deltaswamp_test.managed_plain;
```

### The conformance sweep

`TestConformanceSweep` walks the whole schema and checks two things: every
refusal carries a reason, and every read `capabilities()` claims actually
succeeds.
