# Testing

Three tiers, in increasing order of what they need.

| Tier | Needs | Command |
|---|---|---|
| Unit | nothing | `make test-unit` |
| Integration (local) | the built extension | `make test` |
| Live Databricks | a workspace and a PAT | `make test-live` |

`make check` runs everything CI runs, so a green result locally means a green
pipeline.

## Unit

Pure Python, and it runs without a Rust toolchain. The conformance matrix tests
are the ones that stop coverage claims rotting.

```bash
make test-unit
```

## Integration (local)

Writes real Delta tables to a temp directory and reads them back with both
engines. The cross-engine tests matter most: delta-rs writes, the kernel reads,
and the reverse. Divergence there is a genuine bug, not a test artifact.

```bash
make build
make test
```

The catalog-managed path is covered here without a Databricks account.
`tests/fake_uc.py` is a Unity Catalog server speaking the real `/delta/v1`
protocol, and `tests/integration/test_catalog_managed.py` builds a table whose
newest commit exists only as a staged file -- the situation that makes these
tables unreadable to anything that does not consult the catalog.

## Live Databricks, with a personal access token

**Opt-in.** These create and drop tables in a real metastore and cost money, so
they never fire by accident.

```bash
export DELTASWAMP_TEST_DATABRICKS=1
export DATABRICKS_HOST=https://adb-1234.5.azuredatabricks.net
export DATABRICKS_TOKEN=dapi...                      # a personal access token
export DELTASWAMP_TEST_CATALOG=main
export DELTASWAMP_TEST_SCHEMA=deltaswamp_test
export DELTASWAMP_TEST_WAREHOUSE_ID=abc123           # optional; unlocks most tests
export DELTASWAMP_TEST_EXTERNAL_LOCATION=s3://...    # optional

pytest tests/integration/test_live_databricks.py -v
```

Check the setup first. This takes a few seconds and tells you which of the three
usual problems you have, instead of leaving you to infer it from a failure
twenty tests in:

```bash
python -m tests.live_preflight
```

The suite is ordered so the cheapest checks fail first: the token, then the
metastore prerequisites, then resolution, then vending, then reads and writes.

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
| Honest refusals | every refusal carries a reason; anything claimed readable is read |
| SQL fallback | warns when it serves a request |
| Lifecycle | exists, list, drop |
| Conformance sweep | walks the whole schema and checks every claim against reality |

Each test creates what it needs under a random name and drops it afterwards,
including when it fails. Anything that cannot be verified on your workspace is
skipped with a reason rather than quietly passing.

### A note on PATs

A personal access token cannot be refreshed. Anything running longer than its
lifetime simply stops, and the failure looks like an authentication error rather
than an expiry. For production use prefer OAuth M2M; the token here is a
convenience for testing.

Note also that the *catalog* token and the *vended storage* credential are on
separate clocks. A healthy PAT does not imply a live storage credential, which
is the usual explanation for a job that dies about an hour in.

### Prerequisites an admin must grant

Both are off by default, and neither is grantable by the caller. They cause most
first-contact failures. `connection.preflight()` checks the first one; the second
shows up as a `CredentialError` on the first vend, with the grant named in the
message.

1. **External data access on the metastore**, an account-admin setting.
2. **`EXTERNAL USE SCHEMA`**, grantable only by the catalog owner:

   ```sql
   GRANT EXTERNAL USE SCHEMA ON SCHEMA main.deltaswamp_test TO `me@example.com`;
   ```

Catalog-managed tables need a third. External access to catalog-commit tables is
a Beta feature, gated on a workspace preview that an admin has to turn on. The
suite skips those tests with that explanation rather than failing.

### Fixture tables

The suite discovers what exists and skips the rest, so a partial schema still
gives useful signal. To exercise everything:

```sql
CREATE SCHEMA IF NOT EXISTS main.deltaswamp_test;

-- 1. Plain managed Delta: the common case.
CREATE OR REPLACE TABLE main.deltaswamp_test.managed_plain
  (id BIGINT, city STRING) USING DELTA;
INSERT INTO main.deltaswamp_test.managed_plain VALUES (1,'oslo'), (2,'lima');

-- 2. Catalog-managed: the headline. Nothing else in Python can read this.
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

### What the sweep asserts

Beyond the per-shape tests, `TestConformanceSweep` walks the schema and checks
two things that are easy to get wrong and hard to notice.

First, every refusal carries a reason. A silent `ok=False` is a bug even when
the verdict itself is right.

Second, every claimed read actually works. If `capabilities()` says yes, reading
has to succeed. That is the check that stops the matrix drifting away from what
the workspace really does.

## OSS Unity Catalog (no Databricks account)

OSS UC 0.5+ implements the same `/delta/v1` commit API the kernel targets, so
the catalog-managed path can be exercised locally:

```bash
docker run -p 8080:8080 unitycatalog/unitycatalog:latest
export DELTASWAMP_TEST_OSSUC=http://localhost:8080
pytest tests/integration -q -k ossuc
```

This is the only way to put the headline feature under CI, because the Databricks
path needs an account.
