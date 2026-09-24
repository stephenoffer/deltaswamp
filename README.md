<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo/deltaswamp-wordmark-dark.png">
  <img src="assets/logo/deltaswamp-wordmark.png" alt="deltaswamp" width="420">
</picture>

**One Python connector for every Delta table.** Unity Catalog managed and
external, Hive Metastore, Glue, plain object storage. One object, reads and
writes, whatever is behind it.

```python
import deltaswamp as ds

conn = ds.connect()  # Databricks auth, auto-detected
t = conn.table("main.sales.orders")

t.to_arrow()  # reads, whatever the table really is
t.append(df)  # writes, routed to an engine that can
t.capabilities()  # and says plainly when something can't be done
```

## Why it exists

Reading a Delta table today means picking a tool per table: `deltalake` for
object-store paths, an undocumented `uc://` scheme for some Unity Catalog
tables, a SQL warehouse for managed tables, Spark for anything catalog-managed,
Thrift for Hive Metastore, boto3 for Glue, PyIceberg for managed Iceberg. Each
brings its own auth, its own credential lifetime, and its own opaque
"unsupported table feature" error.

Nothing else reads **and** writes UC-managed, UC-external, HMS and path-based
Delta tables through a single object. `polars.Catalog` is Unity-only and
unstable. Daft and Ray Data cover the read side. DuckDB's implementation is the
most complete, and it's C++.

## What makes it work

Two open-source engines fail in opposite directions, so deltaswamp composes them
per operation instead of picking one. delta-kernel reads 19 table features
delta-rs refuses to open, including every `catalogManaged` table. delta-rs has
MERGE, OPTIMIZE, VACUUM and RESTORE, none of which the kernel implements. So the
kernel reads, delta-rs writes and maintains, and a Databricks SQL warehouse is
an opt-in fallback for what neither can do.

There was no Python binding to delta-kernel-rs. This ships one, which is why
`catalogManaged` tables open here and nowhere else in Python.

## Honesty over magic

"You shouldn't have to care what connects to what" only holds if the library
admits when it can't:

```python
>>> t.capabilities()["delete"]
Capability(ok=False, engine=None,
  reason="table has row filters; UC credential vending refuses it",
  remedy="ds.connect(..., allow_sql_fallback=True)")
```

Fallbacks are off by default. Rerouting through a SQL warehouse changes latency
and cost by orders of magnitude, so it is something you choose.

## Install

```bash
pip install deltaswamp
pip install 'deltaswamp[pyarrow,polars,sql]'   # extras as needed
```

## Docs

[Usage guide](docs/usage.md) covers the whole API.
[Architecture](docs/architecture.md) covers how it works and why.
[Conformance matrix](docs/conformance.md) covers exactly which features and
operations go where. [Ecosystem audit](docs/ecosystem-audit.md) compares it
with Databricks and the rest of the Delta ecosystem. [Testing](docs/testing.md) covers the three test tiers,
including the live Databricks suite. [Contributing](CONTRIBUTING.md) has the
development setup, and [the changelog](CHANGELOG.md) lists what works today and
what does not.

## What it covers

- **Every table shape.** UC managed (including catalog-managed), UC external,
  Iceberg in UC, Delta Sharing, Hive Metastore, Glue and plain paths. Managed
  tables can be created and external ones registered, not just opened.
- **Reads** with projection, SQL predicates, time travel by version or
  timestamp, change data feed and file listing. All of these work on tables
  only the kernel can open.
- **Writes and DML.** Append, overwrite, replaceWhere, dynamic partition
  overwrite, schema merge, idempotent writes, DELETE, UPDATE and MERGE.
- **ALTER TABLE**, including what delta-rs cannot do: rename and drop columns
  under column mapping, type widening, SET NOT NULL, clustering keys, and the
  properties delta-rs rejects. These are written as metadata-only commits.
- **Maintenance.** OPTIMIZE, Z-ORDER, VACUUM (standard and LITE), RESTORE, FSCK,
  checkpoints (catalog-managed tables included), log cleanup, and staged-commit
  publishing.
- **Governance on Unity Catalog.** Grants, tags, ownership, lineage, key
  constraints, catalogs, schemas, volumes and files. Row filters and masks go
  through the warehouse.
- **Distributed reads**: `plan_scan()` gives a picklable per-file plan, and
  `to_ray_dataset()` reads it in parallel on Ray.
- **Hand-offs** to DuckDB, Polars and Daft, and cross-catalog SQL via
  `conn.sql(...)`.

Databricks-only operations (DROP FEATURE, REORG, CLONE, ANALYZE, OPTIMIZE FULL,
CLUSTER BY AUTO, UNDROP, materialized-view refresh) run on a SQL warehouse when
you opt in. [The ecosystem audit](docs/ecosystem-audit.md) maps every Databricks
and open-source Delta/UC feature to the route that reaches it, and names the
blocker for the few that none does.

## Status

Alpha. Everything above is covered by tests against real on-disk tables, a
Unity Catalog server that speaks the real `/delta/v1` protocol, and a Delta
Sharing server; a live Databricks suite runs with a PAT.

What is not possible yet, and why:

- **Deletion vectors cannot be authored.** `Transaction::update_deletion_vectors`
  is absent from delta-kernel 0.28. DML on tables only the kernel can write is
  a bounded whole-table rewrite, and MERGE on those needs the warehouse.
- **Incremental reads without a change feed** are not built;
  `Table.changes()` follows tables that have one.

## Development

```bash
make venv && make build   # set up and compile the extension
make check                # everything CI runs
```

## License

Apache-2.0

The deltaswamp logo is a parody of the [Delta Lake](https://delta.io) logo,
built on the original mark from the
[delta-io/delta](https://github.com/delta-io/delta) repository. Delta Lake is a
trademark of LF Projects, LLC; this project is not affiliated with or endorsed
by the Delta Lake project or the Linux Foundation.
