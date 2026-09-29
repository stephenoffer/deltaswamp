# Usage guide

The whole public API, with its limits.

- [Install](#install)
- [Connecting](#connecting)
- [Opening a table](#opening-a-table)
- [Reading](#reading)
- [Writing](#writing)
- [Creating and registering tables](#creating-and-registering-tables)
- [Changing rows](#changing-rows)
- [Schema, properties and features](#schema-properties-and-features)
- [Maintenance](#maintenance)
- [Metadata](#metadata)
- [Governance](#governance)
- [Delta Sharing](#delta-sharing)
- [Iceberg](#iceberg)
- [Handing off to other engines](#handing-off-to-other-engines)
- [Distributed reads and writes](#distributed-reads-and-writes)
- [Asking what is possible](#asking-what-is-possible)
- [Errors](#errors)
- [Credentials](#credentials)
- [SQL fallback](#sql-fallback)
- [Known limits](#known-limits)

## Install

```bash
pip install deltaswamp
```

The base install pulls in `deltalake` and `databricks-sdk` and nothing else.
Data leaves through the Arrow PyCapsule interface, so pyarrow is optional.
Without it, creating a table from an Arrow-exporting schema (arro3, for
example), appends, merges on delta-rs, unfiltered scans, schema, history,
VACUUM, RESTORE, checkpoints and ALTER TABLE all work; the calls in the
`pyarrow` row below raise an `ImportError` naming the extra.

| Extra | Adds | You need it for |
|---|---|---|
| `pyarrow` | pyarrow, NumPy (Z-ORDER) | `to_arrow`, `count`, `head`, `plan_scan`, predicates on the kernel path, `delete`/`update`/predicate `overwrite`, a MERGE on the kernel (deletion-vector tables), `optimize`/`z_order`, `create_table` with a list or dict schema, the SQL fallback |
| `pandas` | pandas, pyarrow | `to_pandas` |
| `polars` | Polars, pyarrow | `to_polars`, `Connection.sql(engine="polars")` |
| `duckdb` | DuckDB, pyarrow, pytz | `to_duckdb`, `Connection.sql` |
| `daft` | Daft | `to_daft` |
| `ray` | Ray Data | `to_ray_dataset` |
| `sql` | pyarrow | the SQL warehouse fallback (the warehouse itself is reached through `databricks-sdk`) |
| `iceberg` | PyIceberg, pyarrow | Iceberg tables through Unity Catalog's Iceberg REST endpoint |
| `sharing` | `delta-sharing` | Delta Sharing |
| `hms` | `pymetastore` | Hive Metastore catalogs |
| `glue` | boto3 | AWS Glue catalogs |
| `all` | all of the above except Ray and Daft | |

No extra installs `databricks-connect`: it pins `requires-python == 3.12.*`
and conflicts with pyspark.

## Connecting

`connect()` with no arguments talks to Databricks and hands authentication to
`databricks-sdk`, with its usual precedence: explicit arguments, environment,
`~/.databrickscfg`, then cloud-native sources. Every method the SDK supports
works, from PATs and OAuth to Azure MSI, GCP service accounts and GitHub OIDC.

```python
import deltaswamp as ds

conn = ds.connect()  # environment or ~/.databrickscfg
conn = ds.connect(profile="prod")  # a named profile
conn = ds.connect(host="https://adb-123.4.azuredatabricks.net", config=my_config)
```

Prefer OAuth M2M for anything long-running. A PAT cannot be refreshed, so a job
outliving its token simply stops.

Other catalogs are selected by URI scheme:

```python
conn = ds.connect("uc://http://localhost:8080")  # open-source Unity Catalog
conn = ds.connect("hms://thrift://metastore:9083")  # Hive Metastore
conn = ds.connect("glue://")  # AWS Glue, ambient region
conn = ds.connect("glue://123456789012")  # a specific Glue catalog id
conn = ds.connect("sharing:///path/to/config.share")  # a Delta Sharing profile
conn = ds.connect("file://")  # no catalog: tables by path only
```

Useful keyword arguments:

| Argument | Effect |
|---|---|
| `allow_sql_fallback=True` | permits routing through a SQL warehouse; off by default, and refused on any connection but Databricks, since a warehouse serves only its own workspace's tables |
| `warehouse_id="..."` | which warehouse the fallback uses; chosen automatically when omitted |
| `staging_volume="cat.schema.vol"` | lets the warehouse serve writes and MERGE, staging data in that volume |
| `storage_options={...}` | object-store settings for every engine; see "Storage options" below |
| `iceberg_properties={...}` | PyIceberg catalog and FileIO properties, passed verbatim |
| `default_catalog`, `default_schema` | let you write `conn.table("orders")` |
| `catalog=MyCatalog()` | supply a catalog object directly and skip URI dispatch |

Run `conn.preflight()` once against a new workspace. It returns a list of
problems, empty when ready, and it checks the settings that block everything
else: external data access on the metastore, which is off by default and needs
an account admin, and, when the connection has `default_catalog` and
`default_schema`, EXTERNAL USE SCHEMA on that schema. With the SQL fallback on
it also checks the warehouse (a missing or malformed `warehouse_id` is named)
and notes a missing `staging_volume`. It cannot see per-table limits: Unity
Catalog accepts external writes only for some table kinds (external and
catalog-managed tables), so on most workspaces an ordinary managed table is
readable directly but writable only through the SQL fallback, even when
`preflight()` returns `[]`. `t.can("append")` reports that table by table.

### Storage options

`storage_options` are object_store settings (`aws_region`, `aws_endpoint`,
`azure_storage_account_key`, `google_service_account_key`, `proxy_url`,
`timeout`, ...). Keys are case-insensitive and every object_store alias is
accepted; before merging, each is rewritten to one canonical name per cloud
(`AWS_ENDPOINT_URL`, `endpoint_url` and `aws_endpoint` are one setting on
`s3://`), and a dict that sets one setting twice with different values is
refused. The kernel and delta-rs then get the same options, merged with a
table's vended credentials in this order:

1. Vended credentials (keys, session tokens, SAS, OAuth tokens). When the
   catalog vends any, every credential key you passed for that cloud is dropped.
2. An S3 endpoint the catalog vended (an access point, R2), together with its
   region and addressing style.
3. Everything else you passed: region, endpoint, proxy, timeouts, TLS. So
   `aws_region` fixes a UC external table whose bucket is in another region
   than the metastore, and `azure_storage_endpoint` reroutes through private
   link. A region in a different AWS partition from the vended one (us-east-1
   against a cn-north-1 credential) is ignored.
4. Other vended settings.
5. The standard `AWS_*` environment variables (`AWS_ENDPOINT_URL`,
   `AWS_REGION`, `AWS_ACCESS_KEY_ID`, ...) on `s3://`, as delta-rs reads them;
   the credential ones only when nothing above gave a credential.

The client retry policy is set the way delta-rs reads it, on both engines:
`max_retries` (an integer), `retry_timeout`, `backoff_config.init_backoff`
and `backoff_config.max_backoff` (durations as humantime spells them: `30s`,
`30 s`, `500ms`, `2 minutes`, `1h 30m`, `1d`; not bare seconds), and
`backoff_config.base`. The keys are lower-case, as delta-rs reads them.
`connect()` refuses a value either engine would refuse, and a base that is not
a finite number above 1, a zero initial backoff or a duration over 100 years,
on which object_store's backoff panics.

S3 keys with a region and no endpoint get an explicit one,
`https://s3.<region>.amazonaws.com` (`.amazonaws.com.cn` in China regions).
Azure sovereign clouds work with the URL alone:
`abfss://c@acct.dfs.core.chinacloudapi.cn/t` (or `usgovcloudapi.net`) derives
the endpoint `https://acct.blob.<suffix>`, as a public-cloud `abfss://` URL
does; only a host-less `az://container/...` needs `azure_storage_endpoint`.
`wasb://` and `wasbs://` are refused; use the `abfss://` form. A GCS OAuth
token (`google_bearer_token`, `gcp_oauth_token` or `bearer_token`) is used by
the kernel only; delta-rs cannot take one, so operations only delta-rs
implements are refused on such a table.

Every commit is a put-if-absent of the next log file. Writes are refused when
that cannot hold: with `aws_conditional_put=disabled`, with
`AWS_S3_LOCKING_PROVIDER=dynamodb` (neither engine coordinates through
DynamoDB, and a Spark `S3DynamoDBLogStore` writer ignores conditional puts),
and on an S3-compatible endpoint that ignores `If-None-Match`. For any custom
`aws_endpoint` that is not AWS or R2, the first write in a process probes the
store with two put-if-absent calls on a sentinel under `_delta_log/` (deleted
afterwards). If this process is a table's only writer, pass
`storage_options={"deltaswamp_skip_put_if_absent_probe": "true"}` to skip it.

Iceberg tables get the PyIceberg FileIO equivalents of `storage_options`
(`s3.endpoint`, `s3.region`, `s3.proxy-uri`, `s3.connect-timeout`,
`adls.account-name`, `gcs.oauth2.token`, ...), with `iceberg_properties` over
them and the catalog's vended credentials over both. Delta Sharing downloads
use `proxy_url` and `timeout`.

## Opening a table

```python
t = conn.table("main.sales.orders")  # three-level name
t = conn.table("sales.orders")  # needs default_catalog
t = conn.table("orders")  # needs default_catalog and default_schema
t = conn.table("main.sales.orders", version=12)

t = conn.open_table("s3://bucket/tables/orders")  # straight to storage
```

Names may be backtick-quoted, so a dot inside an identifier works:
``conn.table("main.`my.schema`.orders")`` is three parts, not four.

| Form | Goes to |
|---|---|
| `catalog.schema.table` | the connection's catalog |
| `uc://catalog.schema.table` | Unity Catalog |
| `hms://host:9083/db/table` | Hive Metastore |
| `glue://db.table`, `glue://<id>/db.table` | Glue |
| `share.schema.table` on a sharing connection | Delta Sharing |
| `s3://`, `gs://`, `abfss://`, `file://`, `/local/path` | storage directly |

A `glue://` or `hms://host:port/...` reference is served by that catalog
whatever the connection is bound to: the catalog is built on first use and
kept on the connection, so `conn.sql(tables={"c": "glue://crm.customers"})`
works on a Databricks connection. `uc://` and Delta Sharing names need their
own connection, and a connection bound elsewhere refuses them rather than
looking the name up in its own catalog.

`dbfs:/` and `/mnt/...` paths are refused with an explanation, since they are
unreachable from outside Databricks.

A handle opened with `version=` reads that version, and writes as a
transaction begun there would (delta-rs#4417). `append` is blind -- it reads
nothing -- so it appends at the latest version, as from any handle. `delete`,
`update` and `merge` read the pinned version and commit at the latest after
Delta's conflict check against every commit since: they go through if the
commits since were blind appends or left the files they read alone, and
raise `CommitConflictError` otherwise (open a later version and run it
again). Only the kernel's deletion-vector DML does that, so they need a table
with `delta.enableDeletionVectors`, not catalog-managed; elsewhere, and for
every other write (overwrite, schema-merging appends, OPTIMIZE, RESTORE,
ALTER), a pinned handle refuses, and `can()` says so.

## Reading

`scan()` is the primitive. It returns an Arrow stream that exports
`__arrow_c_stream__`, so Polars, DuckDB, pandas and pyarrow can all consume it
without this library depending on any of them.

```python
stream = t.scan()
stream = t.scan(columns=["id", "amount"])
stream = t.scan(predicate="amount > 100 AND region IN ('eu', 'us')")
stream = t.scan(version=3)
stream = t.scan(timestamp="2026-01-01T00:00:00Z")
```

Predicates and timestamp travel work on every engine, catalog-managed tables
included. On the kernel path one parse of the predicate both skips files by
their statistics and filters the rows exactly. The predicate language is the boolean subset of Spark SQL:
comparisons, `AND`/`OR`/`NOT`, `IN`, `BETWEEN`, `LIKE`, `IS [NOT] NULL`, `<=>`,
nested columns (`addr.zip`) and typed literals (`DATE '2026-01-01'`).
A read predicate with arithmetic or function calls goes to delta-rs or the
warehouse, which accept full SQL (DELETE, UPDATE and replaceWhere on the
kernel evaluate it with DuckDB; see below).

Literals mean what they mean in Spark, whichever engine serves the call.
Adjacent string literals concatenate, so `'it''s'` is `its` (write `'it\'s'`
for `it's`), and backslash escapes are processed. A STRING column compared
with a number (`s = 1`), or a number with a BOOLEAN (`i = true`), is refused
with a `PredicateError`: Spark would cast every value of the column, so quote
the literal to compare as text. An `IN` list holding a DOUBLE compares every
item as a DOUBLE, as Spark does.

Timestamps honor in-commit timestamps where a table has them, on delta-rs as
on the kernel, and a timestamp earlier than the oldest reconstructable version
is refused rather than silently clamped. A timestamp after the latest commit
is refused with `InvalidArgumentError` on every engine, as the warehouse
refuses it (`DELTA_TIMESTAMP_GREATER_THAN_COMMIT`): no version exists at that
time yet, and the same read would return other data once the next commit
lands. Read the latest version without a timestamp.

A commit's time is its in-commit timestamp, and before those are enabled its
commit file's modification time made monotonic: a commit whose file looks no
newer than the one before it is taken to be a millisecond after it, as Spark
and Databricks take it. File times go out of order when a log is copied or
rewritten, or written from machines whose clocks disagree; every engine
resolves a timestamp against the same times (`restore()` and the change
feed's bounds included), and `history()` and `_commit_timestamp` report them.
A feed's `starting_timestamp` names the first commit at or after it and
`ending_timestamp` the latest at or before it; either after the latest commit
is refused unless `allow_out_of_range=True`.

A predicate on a `CHAR(n)` column is evaluated by the warehouse: Spark pads a
CHAR value with spaces to `n` before comparing (`c = 'a  '` matches `'a'` in a
CHAR(3) column), which the direct engines, comparing bytes, do not. Without the
SQL fallback such a read, DELETE or UPDATE is refused; a predicate on the other
columns reads directly. VARCHAR compares as it is stored, as in Spark.

Dates and timestamps before 1582-10-15 that Databricks or Spark wrote in the
legacy hybrid calendar (Parquet files marked `org.apache.spark.legacyDateTime`)
are rebased on the kernel path exactly as Spark rebases them, so they read as
the warehouse shows them. Timestamps written in a session time zone other than
UTC, or in a file that does not record its zone (Spark 2.x, which Spark reads
in the reader's session zone), are refused before 1900-01-01T00:00:00Z rather
than guessed: until then Spark rebases them with each zone's own historical
offsets. INT96 timestamps are decoded at microsecond precision, so values
before 1677 read correctly. delta-rs does not rebase, so operations through
which it would rewrite such files -- DELETE, UPDATE, MERGE, replaceWhere,
OPTIMIZE, Z-ORDER -- are routed elsewhere (the kernel, the warehouse) or
refused when a file holds a value the rebase moves, or an INT96 timestamp
after 2262-04-11 (delta-rs decodes INT96 as nanoseconds, which overflow
there: `9999-12-31` reads as `1816-03-30`); `can()` says so. Reads of such a
table never go to delta-rs either: a predicate outside the kernel's grammar
(`abs(x) > 0`, `year(d) = 1500`) is then evaluated after the kernel's read,
by DuckDB in Spark's dialect (without DuckDB installed, the read is refused).
The check reads a file's footer only when its statistics allow such a value.

Writing, the same values need the right Parquet footer. Spark reads a file
whose footer names no Spark version (`org.apache.spark.version`) in the mode
`spark.sql.parquet.datetimeRebaseModeInRead` gives, and Databricks SQL
warehouses read such files with the legacy rebase: a proleptic `0001-01-01`
written by delta-rs, pyarrow or any other arrow-rs writer reads there as
`0001-01-03`, `1500-06-15` as `1500-06-05`, and a TIMESTAMP the same (not a
TIMESTAMP_NTZ, which Spark never rebases), while every other reader sees the
value written. A file that names Spark 3.0 or later, without
`org.apache.spark.legacyDateTime`, is read as written in any session time
zone. Every data file the kernel writes -- appends, overwrites,
replaceWhere, the rows a deletion-vector or copy-on-write DML rewrites,
OPTIMIZE and Z-ORDER through the kernel, distributed workers -- names
`org.apache.spark.version = 3.5.0` for that reason. delta-rs 1.6.5 has no way
to add a footer key, so:

- a write whose rows hold a date before 1582-10-15 or a timestamp before
  1900 (APPEND, OVERWRITE, replaceWhere, a MERGE source) goes to the kernel
  or the warehouse instead of delta-rs, or is refused when neither can serve
  it (MERGE into a table without deletion vectors, schema merge). In-memory
  data (a pyarrow Table or RecordBatch, a list of batches, a pandas or
  polars DataFrame) is inspected. A stream (a RecordBatchReader, any Arrow
  stream) cannot be without consuming it, so one with a DATE or TIMESTAMP
  column counts as holding such values and goes to the kernel;
- an UPDATE or MERGE whose SET or INSERT values may be such a value goes the
  same way. A value counts unless it is provably at or after the limits:
  NULL, a DATE or TIMESTAMP (or string) literal after them,
  `current_date()`/`current_timestamp()`, or a copy of a column whose values
  are (a target column the file check below has cleared, or a column of an
  inspected MERGE source). Arithmetic, functions and casts count;
  `can("merge", ..., clauses=[("when_matched_update", None, {"d": "..."})])`
  takes a clause's values as its third item;
- DELETE, UPDATE, MERGE, replaceWhere, and an OPTIMIZE or Z-ORDER that only
  delta-rs can run (`writer_properties=` and the other delta-rs-only
  options), skip delta-rs on a table whose statistics allow such a value in
  any file, since delta-rs would copy the rows it leaves alone into files
  Databricks misreads. `t.set_properties({'delta.enableDeletionVectors':
  'true'})` lets the kernel serve row-level DML there.

A table delta-rs (or deltaswamp before this change) already wrote with such
values reads shifted on Databricks until its files are rewritten;
`t.optimize()` through the kernel rewrites small files with the footer. The
warehouse-staging path reads its staged files with
`datetimeRebaseMode => 'CORRECTED'`.

A VARIANT column reads as JSON text (`string`), as Databricks' `to_json`
renders it, whichever engine serves the read: the warehouse can send nothing
else, so the direct engines decode the binary encoding to match. Writes take
that JSON text back (the warehouse runs it through `parse_json`; the direct
engines encode it), so a read written back stores the same values. Databricks
turns `delta.enableVariantShredding` on for every new VARIANT table, and
neither direct engine decodes a shredded file, so on such a table reads that
touch a VARIANT column go to the warehouse or are refused; `count()` and reads
of the other columns stay direct. Which columns are VARIANT is read from the
log's schema, so a real `struct<metadata: binary, value: binary>` column stays
a struct, and a VARIANT nested in a struct is converted too (one inside an
array or a map is left in the engine's binary form). The text is Databricks':
`1.0E20`, not `1e+20`; `00:00:00.5`, not `.500000`; object keys in Spark's
order. `NaN` and `Infinity` are refused, as `parse_json` refuses them.
`plan_scan()` and `plan_write()` read and take the same JSON text.

Convenience wrappers sit on top:

```python
t.to_arrow()  # pyarrow.Table
t.to_pandas()
t.to_polars()  # or to_polars(lazy=True): reads on collect, filters pushed down
t.to_pyarrow_dataset()
t.head(10)
t.count()  # exact; count(predicate="...") too
t.files()  # live data files with sizes, partition values and statistics
```

Change feeds come from `cdf()`, by version or by timestamp:

```python
for batch in t.cdf(starting_version=5, ending_version=9):
    ...
t.cdf(starting_timestamp="2026-09-01T00:00:00Z", predicate="region = 'eu'")
```

The feed has the table's current columns, as Databricks' `table_changes()`
does: a range ending before an ADD COLUMN reads that column as null. On a
column-mapping table, Delta reads a batch feed under its end version's
schema instead, and so does `cdf()`. A `columns=` projection is left as asked.

The kernel serves CDF first, and delta-rs what the kernel cannot (a read past
the table's last version, a predicate outside the kernel's grammar). A
catalog-managed table's feed needs the warehouse, since the kernel's change
feed cannot take the catalog's commit tail. Guards fire before any read.
A table without `delta.enableChangeDataFeed` is refused, because enabling it is
not retroactive. A table whose `delta.deletedFileRetentionDuration` is shorter
than its `delta.logRetentionDuration` is refused too, because files could be
vacuumed while their commits survive.

An append-only table needs no feed to be read incrementally. `added_since`
returns the rows of the data files committed after a version, from the
kernel's incremental scan of the log:

```python
new_rows = t.added_since(last_seen)  # (last_seen, latest]
t.added_since(5, until=9, columns=["id"], predicate="id > 100")
```

Only appends make those the rows added: a DELETE, UPDATE, MERGE, OPTIMIZE or
overwrite in the range removes files and adds back rows that were there
before, so such a range is refused. `only_appends=True` reads every file
added anyway, rows a rewrite carried over included, for a consumer that
dedupes on a key; `cdf()` is the exact answer. A range whose commits were
cleaned from the log is refused too.

`changes(start, include_snapshot=True)` bootstraps a consumer of the feed: the
first yield is the whole table as of `start`, every row an `insert` of that
version, and the feed follows from `start + 1`. The feed must be on from
`start + 1`, not before.

## Writing

```python
t.append(df)  # anything Arrow-shaped
t.overwrite(df)
t.overwrite(df, predicate="region = 'eu'")  # replaceWhere
t.overwrite(df, partition_overwrite="dynamic")  # replace the partitions in df
t.append(df, schema_mode="merge")  # widen the schema to fit
t.replace(df)  # new contents and schema (RTAS)
t.append(df, txn=("nightly-load", batch_id))  # idempotent
```

Each returns an `OperationResult` (a dict): the `version` it committed and the
`num_files`, `num_rows` and `num_bytes` it added (`num_removed_files` too),
read back from the commit, plus `.engine`. A write skipped because its `txn`
was already committed has `skipped=True` and no `version`. The warehouse
reports neither, so a write it served carries only `.engine`.

`txn=(app_id, version)` makes an append exactly-once, as Spark's
`txnAppId`/`txnVersion` do: an append whose version is at or below the last
one committed under `app_id` is skipped, and `t.txn_version(app_id)` says
which that was. It is read from the log (through the kernel on a
catalog-managed table, with the catalog's commit tail), so it answers even
where the warehouse does the writing; the warehouse itself cannot record a
txn, so there `txn=` is refused. `plan_write(txn=...)` refuses a committed
version instead of skipping it, before any worker runs, since the job it
plans would be work already done.

`df` can be a pyarrow Table or RecordBatchReader, a Polars DataFrame, a pandas
DataFrame, or anything else exporting the Arrow PyCapsule interface.

A column's data must fit the table's type as Delta's schema enforcement
decides it: the same type, or a widening that keeps every value (a narrower
integer, int to double, float to double, a decimal with room for all its
digits). A narrower numeric column type is taken too when every value fits it
exactly, which is how Python ints and floats (int64, double) reach an INT,
SMALLINT or FLOAT column; the warehouse's INSERT takes them the same way. A
write that would change values -- 4.7 into a BIGINT, 2**40 into an INT, 0.1
into a FLOAT, '12' into an INT, a DECIMAL(15,3) into a DECIMAL(10,2) -- raises
`InvalidArgumentError`, where delta-rs cast it silently. A stream is judged by
its types alone, since its values cannot be checked without reading it. Cast
the data first, or widen the column with `alter_column_type()`.

Databricks' ANSI interval columns read as the warehouse returns them on every
engine: `INTERVAL DAY TO SECOND` (and its narrower day-time forms) as a
microsecond `duration`, `INTERVAL YEAR TO MONTH`, `YEAR` and `MONTH` as Spark's
own text (`INTERVAL '1-2' YEAR TO MONTH`), since pyarrow, pandas and polars
cannot hold Arrow's `month_interval`. Both write back unchanged: a duration is
staged for the warehouse as microseconds and multiplied back into an interval
(Databricks casts a bare BIGINT to an interval as *seconds*), and the kernel and
delta-rs store the integers Databricks stores. A plain integer bound for a
day-time interval column is refused. A duration nested in a struct, list or map
cannot be staged for the warehouse. Year-month text is checked as Spark checks
it: the month of `'1-13'` is out of range, and a value past the INT32 of months
is refused; `YEAR TO MONTH` text written to an `INTERVAL YEAR` column keeps the
whole years, as Spark's cast does.

SQL that names an interval column -- a read or DML predicate, an UPDATE or
MERGE value -- needs the warehouse (`allow_sql_fallback=True`) and is refused
without it: the files hold the bare integers, and a direct engine would compare
those (`ym = -14` matched `INTERVAL '-1-2' YEAR TO MONTH`) where Spark compares
intervals. The lazy hand-offs (`to_duckdb`, `to_polars(lazy=True)`,
`to_pyarrow_dataset`) push no filter on such a column into the scan, so
`rel.filter("i > INTERVAL 1 DAY")` or `pl.col("ym") == "INTERVAL '-1-2' YEAR TO
MONTH"` is evaluated on the values they show.

A Delta timestamp holds microseconds. A nanosecond timestamp (pandas'
`datetime64[ns]`, `pa.timestamp("ns")`) creates a microsecond column, as Spark
does, and its values are truncated to microseconds on the way in, as every
append into a timestamp column truncates them. delta-rs would otherwise create
a `timestamp_nanos` column behind its non-standard `timestampNanos` feature,
which DuckDB, Spark and Databricks cannot read.

A column the data leaves out gets what Databricks would give it: its DEFAULT, a
generated or identity value, or a null. A literal DEFAULT (`'new'`, `42`,
`true`, `DATE'2026-01-01'`) is filled in on every engine. Any other
(`current_timestamp()`) only Databricks evaluates, so that write needs the SQL
fallback. Through the warehouse, a replaceWhere names the data's columns
(`INSERT INTO t (cols) REPLACE WHERE ...`) so the same holds there, and
`when_not_matched_insert_all()` / `when_matched_update_all()` set the source's
columns rather than `*`, as delta-rs does.

Catalog-managed tables append through the kernel, which commits via the
catalog. Partitioned tables are handled: rows are split by partition value and
written to their own partitions. `conn.write_table(name, df, mode=...)` adds
Spark's save modes (`error`, `ignore`, `append`, `overwrite`), creating the
table when it is absent.

## Creating and registering tables

```python
import pyarrow as pa

schema = pa.schema([("id", pa.int64()), ("city", pa.string())])

t = conn.create_table("s3://bucket/tables/new", schema)  # a path
t = conn.create_table("main.sales.new", schema)  # managed, catalog-managed
t = conn.create_table("main.sales.ext", schema, location="s3://bucket/ext")  # external
t = conn.create_table("main.sales.c", schema, cluster_by=["city"], comment="...")
t = conn.register_table("main.sales.old", "s3://bucket/existing-delta-table")
```

A managed table goes through the catalog's staging-table flow. The catalog
allocates the id and the storage, deltaswamp writes version 0 there with the
id and protocol the catalog requires, and the catalog then finalizes the
registration. An external table is written with path credentials the catalog
vends for creating tables, then registered, so the catalog and the log agree.
`register_table` takes the schema, partitioning and properties from the log.

Both catalog shapes need Unity Catalog (Databricks or open source). On any
other catalog they are refused, because writing a log alone would leave it
orphaned while the call appeared to succeed.

A schema is an Arrow schema, or a `{name: type}` dict or `[(name, type)]` list
whose types are Arrow types or type names -- pyarrow's (`int64`) or the SQL
and Delta names Spark uses (`bigint`, `long`, `timestamp`, `decimal(10,2)`,
`array<string>`).

`properties=` accepts nearly the whole Delta property surface, including
`delta.feature.*` signals, row tracking and in-commit timestamps. When delta-rs
would reject a property, the create falls through to the kernel.

## Changing rows

```python
t.delete("id = 42")
t.delete()  # every row
t.update({"status": "'archived'"}, predicate="age > 365")  # SQL expressions
t.update(new_values={"status": "archived"}, predicate="age > 365")  # plain values
(
    t.merge(source, "t.id = s.id", source_alias="s", target_alias="t")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute()
)
```

`DEFAULT` as a SET value (`{"city": "DEFAULT"}`, in `update` or a MERGE clause)
is the column's DEFAULT, as in Spark: a literal DEFAULT is filled in on every
engine, NULL where the column has none, and one that is an expression needs
the warehouse. A column named `default` wins over the keyword, as on
Databricks. A MERGE `when_not_matched_insert` that leaves a column out gives it
its DEFAULT too.

On a VARIANT column, `new_values` takes JSON text (`{"v": '{"q": 2}'}` stores
the object), as every write does. In SQL `updates`, Spark's meaning holds: a
string literal stores a variant *string*, and `parse_json('...')` an object.

Through the warehouse, `update` sets a struct field by its dotted path
(`{"s.a": 5}`) when no top-level column has that name, and `new_values` takes
bytes, dicts (a struct) and lists (an array) as well as scalars.

`delete`, `update`, `merge(...).execute()`, `optimize`, `z_order` and
`restore` return an `OperationResult`, a dict whose keys are named the same on
every engine, next to the serving engine's own metrics; `.engine` names the
engine. DML reports `num_deleted_rows`, `num_updated_rows`,
`num_inserted_rows`, `num_affected_rows`, `num_files_added`,
`num_files_removed` and `version`; OPTIMIZE and Z-ORDER `num_files_added` and
`num_files_removed`; RESTORE `num_removed_files` and `num_restored_files`. A key
is left out where the engine does not report it: a deletion-vector DELETE, for
one, reports rows and the version but no file counts. Read with `.get()`.

`merge` returns a builder with delta-rs's clause API, whichever engine serves
it. When the warehouse serves it, the builder generates one `MERGE INTO`
statement, with the source staged in a volume.

On a table with deletion vectors enabled (`delta.enableDeletionVectors`, the
Databricks default), all three are written as deletion vectors, as Databricks
writes them: the matching rows are marked deleted in a small
`deletion_vector_<uuid>.bin`, rewritten and inserted rows go to new files, and
nothing else is copied. This is the path for catalog-managed tables, and it
works on row-tracked tables too: surviving rows keep their `baseRowId`, and
rows an UPDATE or MERGE rewrites keep their row ids through the table's
materialized row-id column. Only the files the predicate cannot skip are read,
and the commit is staged against the snapshot that was read, so a concurrent
writer makes it conflict rather than be lost. As in Delta's default
WriteSerializable isolation, a conflict with writers that only appended (and
changed neither the schema nor the protocol, nor removed or re-vectored a file
this commit touched) is re-committed on top of them, up to
`KernelEngine.dml_commit_retries` (15) times: rows those appends added are
not deleted or updated, since the statement never read them. A winner that
was not a blind append -- a MERGE, UPDATE or DELETE -- and added a file the
statement's predicate (a MERGE: its join keys) could match makes it conflict
instead, as Spark's ConcurrentAppendException does: two concurrent upserts of
one new key used to insert it twice. Under
`delta.isolationLevel=Serializable`, and on catalog-managed tables, every
conflict is raised. The copy-on-write DELETE and UPDATE (tables without
deletion vectors) rebase over the same blind appends by running again on the
new snapshot, so they also act on the appended rows. A data file whose add carries
no `numRecords` statistic (every file in a Databricks checkpoint, which keeps
statistics only as `stats_parsed`) takes a vector too; its row count is read
from the Parquet footer.

On this path the kernel evaluates DELETE, UPDATE and replaceWhere SQL itself.
It reads comparisons, `IN`, `BETWEEN`, `LIKE`, `IS NULL` and `AND`/`OR`/`NOT`
over columns and literals natively, and a SET value that is a literal or a
column. A predicate or SET value beyond that (`id % 3 = 0`, `lower(s) = 'a'`,
`x + 1`, `CASE ...`, `st.x`) is translated as described in [Spark SQL on the
direct engines](#spark-sql-on-the-direct-engines) and evaluated by DuckDB
(`pip install 'deltaswamp[duckdb]'`): SET values over the matched rows only,
each reading the row as it was. Integer overflow, division by zero and a
malformed CAST raise `InvalidArgumentError` before anything is written, as the
same UPDATE fails on Databricks. Such a predicate gives the kernel nothing to
skip files by, so every file is read. SQL the dialect cannot translate
faithfully, or that DuckDB does not bind (`try_divide`), goes to delta-rs as
copy-on-write, or to the warehouse, and `t.can("update", updates=...,
predicate=...)` says which.

A MERGE evaluates its clauses with DuckDB (`pip install 'deltaswamp[duckdb]'`),
after translating them as described in [Spark SQL on the direct
engines](#spark-sql-on-the-direct-engines). Clause SQL DuckDB still cannot run
moves the MERGE to delta-rs or the warehouse before anything is written; so
does a DEFAULT that is an expression. It skips target
files using the source's join keys and refuses a target row that more than one
source row would modify, as Spark does. Values are stored as Spark stores them:
a fraction truncates into an integer column and rounds half-up into a narrower
decimal.

On every engine, as on Databricks: only the last clause of a kind may omit its
condition; `when_matched_update_all()` / `when_not_matched_insert_all()` need
every target column in the source, except those named in `except_cols` (and
generated or identity columns); and the aliases default to `source` and
`target`.

Without deletion vectors, delta-rs serves DML as copy-on-write, rewriting the
Parquet files that hold matching rows. On a table only the kernel can write,
`delete`, `update` and predicate overwrites do the same: the kernel rewrites
only the files that hold matching rows, keeping row ids on row-tracked tables,
with the same SQL as on the deletion-vector path, and refuses a rewrite that
would read more than `KernelEngine.dml_max_bytes` (4 GiB). On a table with the
change data feed enabled, the kernel serves only DELETE, because UPDATE and
MERGE need CDC files it cannot write. deltalake 1.6.5 inserts an all-NULL row
for each source row a conditional `when_not_matched_insert` rejects on such a
table, so with that version the MERGE is refused there (and goes to the
warehouse when the fallback is on); make the last NOT MATCHED clause
unconditional, filtering the source first, to keep it local. deltalake 1.6.6
fixed it, and the MERGE runs on delta-rs.

### Spark SQL on the direct engines

Predicates, SET values and MERGE clauses are Spark SQL, which the warehouse
runs as written. SQL the predicate grammar covers is evaluated exactly on every
engine; anything else (functions, arithmetic) is evaluated by DataFusion
(delta-rs) or DuckDB (the kernel's DELETE, UPDATE, replaceWhere and MERGE,
the Delta Sharing filter), whose
dialects read some Spark SQL differently or not at all. It is translated first
(`deltaswamp.engine.dialect`), so each engine computes what the warehouse
does:

- `"ab"` is a string, adjacent literals concatenate (`'it''s'` is `its`), and
  backslashes escape;
- `5 / 2` is 2.5 (DataFusion divided integers into an integer), `7 DIV 2`
  truncates, and dividing by zero is an error;
- a fraction CAST to an integral type truncates (DuckDB rounded 1.9 to 2), and
  `CAST('1.9' AS INT)` is an error;
- `substring(s, 0, 2)` is `ab`, a negative position counts from the end, and
  `left`/`right` of a non-positive length are empty;
- `concat` is NULL when any argument is; `log(x)` is the natural log;
  two-argument `trim`/`ltrim`/`rtrim` take the characters first;
  `regexp_replace` replaces every match; `^` is XOR;
- `RLIKE`/`REGEXP`, `<=>`, `nvl`, `nvl2`, `if`, `pmod`, `1.5D`, `7L`, `1.5BD`
  and LIKE's default `\` escape work on both;
- in DuckDB an integer literal is an INT, or a BIGINT past INT's range, as in
  Spark: a TINYINT `b` of 127 gives 127 for `b + 1 - 1` (DuckDB narrowed the
  literal to TINYINT and overflowed);
- where DuckDB filters a read (a table delta-rs misreads, a Delta Sharing
  filter) or evaluates the kernel's DELETE, UPDATE or replaceWhere, a FLOAT compares with a decimal literal as DOUBLE (`f = 0.1` is
  false for the FLOAT 0.1, as on Spark), and BIGINT arithmetic that overflows
  raises, as ANSI Spark's does, rather than being folded away. Such a filter
  is one expression over the row: a subquery (`SELECT`, `VALUES`, `PIVOT`,
  `DESCRIBE` ...), a statement or a comment is refused, with string literals
  read as DuckDB reads them.

SQL no rewrite makes agree is refused, and routes to the warehouse (the
`spark_sql` need, which `can()` reports): an array subscript (0-based in Spark,
1-based in both engines; `element_at` agrees everywhere), `split` (a regex in
Spark), a Java date or format pattern (`date_format`, `to_date(s, fmt)`),
`regexp_replace` with `$1` group references, and `hash`. SQL DataFusion
cannot parse or plan raises `EngineLimitError` (a MERGE then moves to the next
engine), not delta-rs's own `DeltaError`.

## Schema, properties and features

```python
t.add_column(pa.schema([("region", pa.string())]))  # or {"region": "string"}
t.set_properties({"delta.enableChangeDataFeed": "true"})
t.unset_properties(["owner.team"])
t.add_feature("deletionVectors")
t.add_constraint({"id_positive": "id > 0"})
t.drop_constraint("id_positive")
t.set_comment("orders, one row per line item")
t.set_column_comment("city", "shipping city")
t.set_not_null("id")  # checks existing rows first
t.drop_not_null("id")
t.alter_column_type("qty", "bigint")  # widening; needs delta.enableTypeWidening
t.cluster_by(["city"])  # liquid clustering keys; None clears them
t.rename_column("city", "town")  # needs column mapping
t.drop_column("region")  # needs column mapping
```

delta-rs serves what it can. Everything else on a path table is a
metadata-only commit that deltaswamp writes itself. That covers renames and
drops under column mapping, type widening, SET NOT NULL, clustering keys, the
properties delta-rs rejects on ALTER, and any change to a table delta-rs cannot
write, such as a liquid-clustered one. The change is computed from the table's
current protocol and metadata, and it raises the protocol when a value implies
a feature. It is committed as a put-if-absent of the next log file, so a
concurrent writer causes a recompute rather than an overwrite.

Enable column mapping first to rename or drop:
`t.set_properties({"delta.columnMapping.mode": "name"})`. The existing Parquet
column names become the physical names, so no data is rewritten.

These are refused with the reason, because they need more than a metadata
commit:

- enabling row tracking on an existing table (needs a backfill)
- UniForm / Iceberg compatibility (needs Iceberg metadata generated)
- changing column mapping other than none -> name
- a catalog-managed table's metadata, which the catalog refuses from external
  writers after version 0

Databricks-only operations need the SQL fallback:

```python
t.drop_feature("deletionVectors", truncate_history=True)
t.reorg(purge=True)
t.clone("main.sales.orders_copy", shallow=True)
t.analyze(delta_statistics=True)
t.sync_iceberg()  # regenerate UniForm metadata
t.refresh()  # a materialized view or streaming table
t.cluster_by("auto")
```

A path table cloned to a storage path needs no warehouse: the kernel writes
the clone's version 0 itself (delta-rs#2456).

```python
t = conn.table("s3://bucket/orders")
t.clone("s3://bucket/orders_dev")  # shallow: references orders' files
t.clone("s3://bucket/orders_copy", shallow=False)  # deep: copies them
t.clone("s3://bucket/orders_v12", version=12)
```

The clone carries the source's protocol, metadata (under a new table id) and
clustering, and its commit is a `CLONE` naming the source and its version, as
Databricks writes one. A shallow clone's add actions name the source's data
files and deletion vectors by absolute URL, which Spark and Databricks read;
this library's own engines read only files under a table's root (a log must
not reach other data with the connection's credentials), so they refuse to
read it -- clone deep to read it here. `vacuum(dry_run=False)` on a shallow
clone is refused, since it would treat the source's files as its own; vacuum
the source, which keeps the files its live versions reference. A deep clone
copies every live data file and deletion vector, then keeps their relative
paths. Refused (and left to Databricks' CLONE): a catalog table name as the
target, `replace=True`, a source whose storage the catalog reaches with
credentials scoped to it, and tables with row tracking or catalog-managed
commits.

## Maintenance

```python
t.optimize()  # bin-packing compaction
t.optimize(zorder_by=["customer_id"])  # or t.z_order([...])
t.optimize(full=True)  # OPTIMIZE FULL of a liquid-clustered table
t.optimize(min_file_size=64 << 20, sort_by=["day", "id"])  # bin-pack only small files, sorted
t.vacuum(retention_hours=168)  # a dry run by default
t.vacuum(retention_hours=168, dry_run=False, lite=True)
t.restore(3)  # or a datetime
t.repair()  # FSCK
t.checkpoint()
t.compact_logs()
t.cleanup_metadata()  # delete logs past delta.logRetentionDuration
t.generate()  # symlink manifests, for Presto and Athena
t.publish()  # staged catalog commits -> _delta_log
conn.convert_to_delta("s3://bucket/parquet-dir")
```

`can("vacuum")` answers for the call `vacuum()` makes with no arguments, a dry
run, which lists files and deletes nothing. A table some engine can list but
none can vacuum for real (in-commit timestamps or type widening on a
delta-rs-only path, for example) answers yes there and no to
`can("vacuum", dry_run=False)`, as the two calls behave. Ask with the
arguments you will pass.

A `_delta_log/_last_checkpoint` naming a checkpoint that is gone is refused
by both direct engines with `CorruptTableError` (listing the log from version
0 cannot tell a damaged log from a whole one); after that error the handle's
cached snapshot is dropped, so `can()` refuses too. Restore the checkpoint, or
remove the stale `_last_checkpoint` if every commit is still present.

OPTIMIZE and Z-ORDER are committed by the kernel on the snapshot they were
planned from, so concurrent runs never compact a file twice; delta-rs's own
OPTIMIZE commit does, and is never used. They take delta-rs's options except
`writer_properties`, `min_commit_interval` and app transactions, which are
refused. Rows stream from the input files into the output files, one step
(`KernelEngine.compaction_batch_bytes` of input, 1 GiB) per commit; memory is
bounded by `KernelEngine.compaction_max_file_bytes` (512 MiB of decoded rows
per output file or Z-order sort). `partition_filters` compare as the column's
type, as delta-rs does: a NULL partition matches no comparison, `= ''` means
NULL, and timestamps take ISO 8601 with or without the `T` and an offset.

A Z-order tags its files as Databricks does (`ZCUBE_ID`, `ZCUBE_ZORDER_BY`), and
the next Z-order by the same columns leaves a partition's cubes of at least
`min_cube_size` (default: the target size) alone, rewriting only new files and
small cubes; Databricks' own incremental ZORDER reads the same tags. On a
liquid-clustered table `optimize()` is that Z-order over the clustering keys
(`zorder_by` is refused, as Databricks refuses it), and `full=True` rewrites
every file. On a row-tracked table every row a compaction moves keeps its row
id and commit version, written into the table's materialized columns; an
overwrite there gives the new rows fresh ids. Kernel 0.28 refuses to stage the
removes of either commit on such a table, so the native commit writes them
itself; its post-commit snapshot and checksum delta do not count them.

`vacuum` defaults to a dry run because the real thing deletes files. `lite=True`
considers only files the log records as removed. `vacuum` returns the paths it
deleted (or would delete) relative to the table root, on every store.

On a table with deletion vectors, and on every table delta-rs cannot commit to
(clustering, row tracking, in-commit timestamps, type widening,
`vacuumProtocolCheck`, collations, ...), VACUUM is planned from the kernel's log
replay the way Spark plans it. A file is kept if a live file is it or names it
as its deletion vector, if a remove whose `deletionTimestamp` is within the
retention names it (the checkpoint's tombstones included), or if a commit
written within the retention lists it as change data. A full VACUUM lists the
table directory, skips hidden paths as Spark does (names starting with `_` or
`.`, other than `_change_data` and partition directories), and deletes the
unreferenced files last modified before the cutoff; LITE deletes only what
expired removes name. The retention defaults to
`delta.deletedFileRetentionDuration` (a week), and a shorter one needs
`enforce_retention_duration=False`. A real VACUUM commits `VACUUM START` and
`VACUUM END`, as Databricks does, and deletes nothing when nothing qualifies.
On the tables delta-rs vacuums itself, its full VACUUM keeps every
`deletion_vector_*.bin`, and a dry run after a lite vacuum still lists the
files the lite run removed, because delta-rs keeps their tombstones in the log.

`restore` of a deletion-vector table, a column-mapped table, or a table
delta-rs cannot write is committed here, as Spark's RESTORE: files live at the
target and not now are re-added exactly as the target logged them (statistics,
deletion vector, and `baseRowId`/`defaultRowCommitVersion`, so restored rows
keep their row ids); files live now and not then are removed; a file whose
deletion vector changed is removed and re-added with the target's. The
target's schema and properties come back too, except the column-mapping
`maxColumnId`, the in-commit-timestamp and the row-tracking settings, which
keep their current values; the protocol is never downgraded. No change-data
files are written: a change-feed reader derives the restore's inserts and
deletes from its adds and removes. A target file VACUUM deleted raises
`MissingDataFileError` unless `ignore_missing_files=True`.

These refuse rather than misbehave:

- `vacuum` and `restore` on a shallow clone, which borrows the source's files
  (a catalog's shallow clone, or a path table whose live files are another
  table's), and on a table carrying a feature nothing here recognizes.
- `restore` across a change of column-mapping mode, of partition columns, or
  of the table id (a replaced table), or to a version whose protocol needed a
  feature since dropped. `can("restore", target=n)` says so up front.
- OPTIMIZE, VACUUM and RESTORE on UC managed tables. Databricks forbids them
  from external clients, so they route to the warehouse.

Checkpoints work on catalog-managed tables. The kernel publishes staged commits
first, since it checkpoints only published versions. Commits made here also
checkpoint on their own at the table's `delta.checkpointInterval`.

Every commit made here also writes its version checksum,
`_delta_log/<version>.crc` (the table's size, file count, protocol and
metadata), which Databricks and Spark read to load a snapshot quickly and to
check the state they rebuilt. delta-rs writes none, so one is written after
each of its commits too. It is computed from the previous checksum, so it is
written only while the chain is intact: the previous checksum at most 100
versions back, or a log of fewer than 100 commits with no checkpoint. A table
whose history never had them is left without, and a catalog-managed table's
are left to the catalog's writer. A checksum that cannot be written never
fails the commit.

## Metadata

```python
t.schema()  # Arrow schema
t.protocol()  # (min_reader_version, min_writer_version)
t.features()  # frozenset of table feature names, verbatim
t.properties()  # delta.* table properties
t.detail()  # version, location, protocol, properties
t.history(limit=10)
t.version
t.location
t.table_type  # MANAGED, EXTERNAL, VIEW, ...; None for a table opened by path
t.is_catalog_managed
```

Metadata reflects the table as of the last call, and any write or ALTER through
this object refreshes it. `features()` returns names exactly as the log holds
them, including ones this release does not recognize, because an unknown
writer-only feature must not block a read.

## Governance

On Unity Catalog:

```python
info = t.info()  # owner, comment, columns, row filter, masks, predictive optimization
t.grants(); t.effective_grants()
t.grant("analysts", ["SELECT"]); t.revoke("analysts", "SELECT")
t.tags(); t.set_tags({"pii": "true"}, column="email"); t.unset_tags(["pii"])
t.set_owner("data-platform")
t.lineage(); t.column_lineage("amount")
t.add_primary_key("pk", ["id"]); t.add_foreign_key("fk", ["cust"], "main.sales.customers", ["id"])
t.drop_key_constraint("fk")
t.set_row_filter("main.sec.only_eu", ["region"])  # warehouse
t.set_column_mask("email", "main.sec.mask_email")  # warehouse

conn.create_catalog("sandbox"); conn.create_schema("sandbox.scratch")
conn.grant("sandbox.scratch", "analysts", ["USE_SCHEMA"])
conn.search_tables("main", table_pattern="ord%")
conn.list_volumes("main.raw"); conn.list_functions("main.udfs")
vol = conn.volume("main.raw.landing")
vol.write("drop/2026-09-24.csv", data); vol.read("drop/2026-09-24.csv"); vol.list("drop")
conn.undrop_table("main.sales.orders")  # warehouse
```

Open-source Unity Catalog lacks tags, lineage, key constraints, ownership
changes and the Files API. Those calls come back as refusals that name the
gap. Other catalogs have no governance API, and say so. OSS Unity Catalog does
not re-read an external table's log, so a schema, comment or property change
made here is also written to its catalog entry (UC 0.6's Delta API
update-table call); a server without that call gets a warning and a stale
entry. A grant the server accepts but does not record (authorization disabled)
warns.

Hive Metastore and Glue have two levels, so `db.table` names a table on those
connections (`hive_metastore.db.table`, `glue.db.table` or
`<account-id>.db.table` also work; another first part is refused). Both
list their databases with `list_schemas()` and drop registrations with
`drop_table()`, keeping the files. Hive Metastore 2, 3 and 4 are supported.

## Delta Sharing

```python
conn = ds.connect("sharing:///path/to/config.share")
conn.list_catalogs()  # shares
t = conn.table("share.schema.table")
t.to_arrow(predicate="year = 2026")
t.scan(version=12)
t.cdf(starting_version=10)  # when the provider shares history
```

Shared files are read with pyarrow, keeping Delta's types. Tables with
deletion vectors or column mapping use the client's delta-format path.
Predicates are sent as hints and then applied exactly. Shares are read-only,
and writes are refused with that reason. Every file the server names is
downloaded here over HTTP(S), including on the delta-format path, so a server
URL is never opened as a local path.

`can("cdf")` is refused when the shared metadata has no change data feed.
Time travel and the change feed otherwise depend on the provider sharing the
table WITH HISTORY, which no endpoint reports before the query; `can()` says
so in its reason. `plan_scan()` and `to_ray_dataset()` are not available for
shares: there is no split planning, and presigned URLs would expire on the way
to workers.

## Iceberg

Iceberg tables in Unity Catalog (managed or foreign) are served through the
catalog's Iceberg REST endpoint with PyIceberg: reads, time travel by snapshot
id or timestamp, history, appends, and overwrites by predicate (for Databricks
managed Iceberg, appends only; see below). UniForm Delta
tables can also be read as Iceberg, but the Delta path remains the default for
them. External writes to UniForm tables are refused, because they would leave
the Iceberg metadata stale; `t.sync_iceberg()` regenerates it on Databricks.

Databricks managed Iceberg (`CREATE TABLE ... USING ICEBERG`) is a special
case: Unity Catalog reports it as Delta, with a catalog-managed Delta log
beside the Iceberg metadata. deltaswamp recognises it (the
`delta.enableIcebergWriterCompatV1` property on a catalog-managed table), reads
it through the Delta path, and appends through the Iceberg REST endpoint.
Overwrites, deletes, updates and MERGE are refused there, because the endpoint
takes one snapshot per commit and an Iceberg overwrite commits two (running them
as two commits would not be atomic); `history()` is refused too, because the
Iceberg snapshot log does not carry Delta versions. With
`allow_sql_fallback=True` the warehouse serves both.

## Handing off to other engines

```python
t.to_duckdb(name="orders")  # a DuckDB relation, optionally a view
t.to_polars(lazy=True)
t.to_ray_dataset()  # parallel: a Ray Data datasource over a scan plan
t.to_daft()

for version, batch in t.changes(start, poll_interval=30):  # a CDF stream
    ...

conn.sql(
    "SELECT o.region, sum(o.amount) FROM o JOIN c ON o.cust = c.id GROUP BY 1",
    tables={"o": "main.sales.orders", "c": "glue://crm.customers"},
)
conn.sql("SELECT * FROM system.access.audit LIMIT 10", engine="warehouse")
```

Each hand-off reads through this library, so it works on tables the target
engine's own Delta reader cannot open. `to_polars(lazy=True)`, `to_duckdb()`,
`to_pyarrow_dataset()` and `Connection.sql` read nothing until the query runs,
and then only the columns it uses (and, for Polars, up to its row limit).
Simple filters are also pushed into the scan to skip files: a column compared
with a literal of its own kind on integer, string, boolean, date and decimal
columns, `IN`, `IS NULL`, and `AND`/`OR` of those. Nothing whose meaning could
differ is pushed -- `NOT`, float and double comparisons (Spark, pyarrow and
Polars order NaN differently), binary values -- and the consumer applies its
whole filter itself afterwards, Polars with Polars' semantics and DuckDB with
DuckDB's (DuckDB hands its filter to the scan and does not reapply it, so this
library evaluates it in DuckDB). Nothing is pushed on a VARIANT column, which
the hand-offs show as JSON text. Pass `predicate=` (SQL) as well when a filter
the engine cannot push should still skip files.

A lazy hand-off reads the version that was current when it was made, at every
scan: a DuckDB statement that scans a relation twice (a self-join) sees one
snapshot, and a frame made before a schema change keeps reading the schema it
declared. Pass `follow_latest=True` to read the latest version at each scan
instead; a scan whose schema no longer matches the declared one then raises
`MetadataChangedError`. `Connection.sql` pins each table when it is called.
`Connection.sql` runs on DuckDB by default (or `engine="polars"`), and can join
tables from different catalogs. `engine="warehouse"` sends the query to
Databricks as it stands.

## Distributed reads and writes

`plan_scan` and `plan_write` split the work between a driver and its workers.
Everything is decided on the driver, so a refusal arrives before any compute
is spent. [Ray Data](ray-data.md) maps each Ray Data `read_delta` /
`write_delta` scenario to what serves it.

```python
plan = t.plan_scan(columns=["id"], predicate="day >= '2026-09-01'")
for group in plan.partitions(8):  # byte-balanced; pickle and ship each
    part = plan.read(group)  # on a worker: same version, the plan's credential

plan = t.plan_write(mode="append")  # driver: raises now if the table refuses it
fragment = plan.write(batch)  # worker: durable, uncommitted files -> bytes
version = plan.commit(fragments)  # driver: every fragment in one commit
```

`WritePlan` and `ScanPlan` are picklable, and what crosses a process boundary
carries the table's short-lived, table-scoped *storage* credential and its
expiry -- never the catalog's credentials (a PAT, an OAuth client secret, a UC
bearer token) and never the catalog itself. The driver's own plan object keeps
full catalog access, so `commit()` runs on the driver; a worker's copy cannot
commit. `to_ray_dataset()` is built on `plan_scan`, and catalog-managed tables
work too: the commit goes through the catalog's committer.

`plan_write` vends the write credential on the driver when it plans, so a
table Unity Catalog will not vend write credentials for is refused there, not
on the first worker: `ExternalWriteNotAllowedError` for a managed table
without catalog commits (`EXTERNAL_WRITE_NOT_ALLOWED_FOR_TABLE`), naming the
ways out.

A plan's credential lasts as long as the catalog made it (about an hour on
Databricks; `plan.credential_expires_at` says when). The native object store
reads the credential on every request, so a refresh reaches a scan or a write
already under way. Workers get fresh credentials in one of two ways:

- `credential_source=`: a picklable callable ``source(table_id, operation)``
  that reaches a `deltaswamp.credentials.CredentialBroker` on the driver (with
  Ray, in an actor; the broker's docstring has the pattern). Workers ask it
  ahead of expiry, once per process, and no catalog secret leaves the driver.
- `ship_catalog_auth=True`: the plan carries the catalog's credential
  provider, token included, and each worker process vends its own (once per
  process, not per task).

With neither, a worker whose credential is within a minute of expiry raises
`CredentialError` saying to re-plan, and planning warns
(`CredentialExpiryWarning`) when the credential has less than half an hour
left.

```python
from deltaswamp.credentials import CredentialBroker

broker = CredentialBroker()
broker.add(t)  # on the driver
plan = t.plan_scan(credential_source=broker)  # in one process; see the docstring for Ray
```

A read task touches only its own files. Each split carries its file's scan row
and the plan the table's protocol and metadata, so a worker neither lists nor
replays the log: a task costs its files, not the table's history, and a
catalog-managed read keeps working after the catalog publishes and removes the
staged commits it was planned from.

Where no engine can plan a distributed read -- a table with a row filter or
column mask, which only the warehouse can read; a Delta Sharing table --
`to_ray_dataset()` raises rather than reading the whole table on the driver.
`to_ray_dataset(allow_driver_read=True)` does that read, and refuses it once
it passes `driver_read_max_bytes` (1 GiB by default).

The change feed is planned the same way. `plan_changes(start, end)` pins the
end, sizes each commit from the log and cuts the range into runs of whole
commits of about `split_bytes` of changed data (256 MiB by default); a commit
is never split, since a deletion-vector update pairs a remove with an add in
one commit. `plan.read(splits)` on a worker returns what `cdf()` returns for
those commits. A catalog-managed table, a range the feed was off for and a
range across a schema change are refused at planning.

```python
plan = t.plan_changes(120, columns=["id", "amount"])
for group in plan.partitions(8):
    part = plan.read(group)  # on a worker: _change_type, _commit_version, ...
```

Concurrency follows what the mode means. An **append** commits against the table
as it is then, so a writer that arrived while the job ran is not a conflict --
both sets of rows survive. An **overwrite** removes what it finds, so committing
against a table that has moved on would discard that writer silently; that is
refused, and `allow_concurrent_overwrite=True` is how you say last-writer-wins.
Where a catalog does arbitrate and rejects the commit, the fragments stay valid
-- they describe data files, which carry no version -- so the same fragments can
be committed again against a fresh snapshot, which `retries=` does for tables
this library commits itself. An append retries by default, as `Table.append`
does: up to 15 times, with a jittered backoff between attempts, on path tables
and on catalog-managed tables planned through their catalog, where each
attempt re-reads the catalog's commit tail.

A concurrent change to the schema, the partition columns, column mapping or a
CHECK constraint is the exception (adding a nullable column is not: the new
files read it as null, like every older file): the fragments' files were
written for the old layout, so committing them would leave the table unreadable or put values in
the wrong columns. `commit()` raises `MetadataChangedError` (a
`CommitConflictError`) and never retries it; plan the write again. A commit
that hits a catalog's backfill demand publishes the table and commits again,
as `Table.append` does.

Fragments that carry no files (every worker's data was empty) commit nothing:
an append returns the current version without adding an empty one, unless it
has a `txn=` to record. An overwrite with no files would empty the table, so it
is refused unless `commit(fragments, allow_empty_overwrite=True)` says so.

A guarded overwrite commits against the planned snapshot itself, so a writer
that lands between the check and the commit makes it conflict rather than
vanish. An idempotent `txn=` is checked again at commit, so two runs of the same
job cannot both land, and a retry refuses fragments whose files an earlier,
seemingly failed attempt already committed. Pass each fragment exactly once: a
data file named twice in one commit is refused.

Which tables accept a distributed write depends on the protocol version. A
legacy writer version implies a whole feature set, and versions 3 to 6 imply
`checkConstraints`, which the kernel refuses whether or not a constraint
exists. Enabling change data feed alone puts a table at version 4.

| Table | Distributed write |
|---|---|
| writer version 1-2 (plain legacy) | yes |
| writer version 3-6 (legacy CDF, legacy column mapping) | yes: the implied `checkConstraints` is checked here, on the workers |
| writer version 7 (feature-based) | yes, for the features the kernel writes |

Change data feed, column mapping, row tracking, in-commit timestamps, deletion
vectors, type widening, liquid clustering, `checkpointProtection`, CHECK
constraints, generated and identity columns, invariants and literal defaults
all write. UniForm (IcebergCompat V1/V2) and geospatial columns are refused
when the write is planned. DELETE, UPDATE, MERGE and maintenance run on the
driver.

### When a job fails

Every fragment must reach `commit()` or `abort()`: until then its files are
durable but belong to no version.

- `commit()` deletes the job's files itself after a failure that certainly
  committed nothing: a concurrent schema change (`MetadataChangedError`), a
  guarded overwrite whose table moved, a conflict on every attempt.
  `abort_on_failure=False` keeps them.
- After a failure whose outcome is unknown (a timeout, a 5xx from the
  catalog), nothing is deleted. Commit the same fragments again: a commit that
  already landed is found in the commits since the write, and its version is
  returned rather than committing twice. The check reads only those commits,
  never the table's whole file list.
- `plan.abort(fragments)` deletes the files of a job you give up on -- a Ray
  datasink's ``on_write_failed``. It refuses files a commit already
  references, and deletes only paths under the table root.

Fragments are Arrow IPC bytes. `merge_fragments(fragments)` concatenates
them, so workers can combine theirs before they reach the driver, and
`commit()` takes any iterable, merging as fragments arrive; driver memory stays
proportional to the job's files.

```python
from deltaswamp.distributed import merge_fragments

try:
    version = plan.commit(fragments)
except ds.errors.TransientCommitError:
    version = plan.commit(fragments)  # safe: a landed commit returns its version
```

### Computed columns and domains

A worker fills a column's literal DEFAULT when a batch leaves it out, computes
generated columns in DuckDB (`deltaswamp[duckdb]`) and checks given ones, and
evaluates CHECK constraints and invariants, all before it writes a file. A
DEFAULT only Databricks can evaluate (``current_timestamp()``) is refused at
planning unless `supplies_defaults=True` promises every batch carries it.

Identity columns need values that no two workers share. `plan_write` reserves
them when it plans, in a metadata-only commit, and each task numbers its rows
from a slot of its own:

```python
plan = t.plan_write(identity_tasks=64, identity_rows_per_task=1 << 20)
fragment = plan.write(block, task_index=ctx.task_idx)  # slot ctx.task_idx
```

Values a task does not use are a gap, which Delta allows. A GENERATED ALWAYS
column with nothing reserved is refused at planning; `identity_tasks=0` is for
data that gives every value of a GENERATED BY DEFAULT column.

`plan_write(domain_metadata={"myapp.watermark": "..."})` sets user domain
metadata in the job's commit, beside its rows; `delta.*` domains and tables
without the domainMetadata feature are refused at planning.

### Creating the table in the job

`Connection.plan_write(name, schema=..., mode=...)` plans a write whether or
not the table exists. For a new table nothing is visible until the commit:
workers write under the table's location, and `commit()` creates the table and
commits the job's files. A path or external table is created in one commit --
version 0 holds the protocol, the metadata and the job's files, with row ids,
in-commit timestamps and clustering as a write would give them -- so no reader
ever sees it empty (an external one is registered once its data is in). A
managed one is staged when planned, with its version 0, and registered by the
commit, which then commits the files through the catalog: the catalog vends the
staging location's credential once, when it allocates the table, so version 0
cannot wait for the job. A failed or aborted job deletes its files and undoes
the create. If another writer creates the table while the job runs, `error`
and `overwrite` fail, `ignore` keeps theirs, and `append` joins it when the
layout matches.

```python
plan = conn.plan_write("main.sales.orders_2026", schema=schema, mode="error")
```

It takes the save mode and `create_table`'s layout arguments (`location`,
`partition_by`, `cluster_by`, `properties`, `comment`), plus every argument of
`Table.plan_write`. For a new table, `identity_tasks` reserves the values in
version 0 itself (its high-water mark), and a `credential_source` for a new
external table is asked by the table's location: `broker.add(plan)` serves it
(`CredentialBroker.portable(plan)` for a broker in another process).

## Asking what is possible

`capabilities()` reports every operation, whether it can be served, by which
engine, and why not when it cannot.

```python
>>> t.capabilities()[ds.Operation.DELETE]
Capability(operation=<Operation.DELETE: 'delete'>, ok=False, engine=None,
  reason='table has row filters; UC credential vending refuses it',
  remedy='ds.connect(..., allow_sql_fallback=True)', blockers=())

>>> t.can("scan")
Capability(operation=<Operation.SCAN: 'scan'>, ok=True, engine=<Engine.KERNEL: 'kernel'>, ...)

>>> print(t.can("scan"))
scan: via kernel

>>> t.can("create", properties={"delta.enableRowTracking": "true"})
>>> t.can("append", data=batch, schema_mode="merge")
>>> t.can("plan_write", mode="overwrite")
>>> t.can("merge", source=updates, predicate="t.id = s.id",
...       clauses=["when_matched_update_all", "when_not_matched_insert_all"])
```

`can()` takes the arguments the call takes, including the data, and answers
for that exact request. When it says ok, the engine it names is the one the
call uses. When it refuses, the call refuses the same way before writing
anything. A method name (`z_order`, `compact_logs`, `plan_write`,
`plan_scan`, `count`) can stand in for the operation. `can("count")` differs
from `can("scan")`: a count reads no column but the ones its predicate names,
so a table whose shredded VARIANT columns the direct engines cannot read still
counts directly. `can(Operation.ZORDER, columns=[...])` takes `z_order()`'s
own spelling. A MERGE clause with a condition is given as `(name, condition)`,
e.g. `clauses=[("when_not_matched_insert_all", "s.v > 0")]`: on a change-feed
table delta-rs cannot run a MERGE whose last NOT MATCHED clause has one. A
clause that sets columns may give them as a third item,
`("when_matched_update", None, {"d": "source.d"})`: a value that may be a date
before 1582 keeps the MERGE off delta-rs. On a
handle opened with `version=`, every write is refused.

`Capability` is truthy when `ok`. Every refusal carries a reason, and a remedy
whenever one exists.

## Errors

All inherit from `DeltaSwampError`. Every engine is called through one
translation boundary, so no engine's own exception type (delta-rs's
`DeltaError`, a pyarrow or duckdb error, the extension's) reaches a caller --
not from a call, and not later from the stream or MERGE builder it returned.
What no rule recognises arrives as `EngineError`.

| Error | Means |
|---|---|
| `InvalidReferenceError` | the reference could not be parsed or resolved |
| `TableNotFoundError` | an `InvalidReferenceError`: the catalog has no such table, or hides it from this principal; `conn.drop_table(name, if_exists=True)` ignores it |
| `InvalidArgumentError` | a call's arguments are malformed or contradict each other (also a `ValueError`) |
| `UnreachableTableError` | no available engine can serve the request |
| `FallbackRequiredError` | only the SQL fallback could serve it, and it is off |
| `PropertyNotSupportedError` | a property the chosen engine cannot handle |
| `EngineLimitError` | an `UnreachableTableError`: the engine serving a read hit a limit of its own once it read the log, so the next capable engine is tried first |
| `CredentialError` | vending or refresh failed |
| `ExternalWriteNotAllowedError` | Unity Catalog vends no write credentials for the table (`EXTERNAL_WRITE_NOT_ALLOWED_FOR_TABLE`: a managed table without catalog commits); raised by `plan_write`, before any worker runs |
| `PreflightError` | a workspace prerequisite is not satisfied |
| `CommitConflictError` | another writer took that version first |
| `CommitRefusedError` | an `EngineError`: the engine refused a commit for a reason other than a lost race (a remove on an append-only table, a failure writing the commit file); retrying the same commit meets the same refusal |
| `MetadataChangedError` | a `CommitConflictError`: a concurrent commit changed the schema, partitioning or column mapping, so the write must be planned again rather than retried |
| `TransientCommitError` | a commit failed for a transient reason; the table is unchanged, so retry it as is |
| `BackfillRequiredError` | the catalog wants staged commits published |
| `StorageError` | the table's storage failed a request (unreachable, throttled, no such bucket); an `OSError` too, and an instance of the storage error's own class (`TimeoutError`, `ConnectionResetError`), so retry policies keyed on those still match |
| `CorruptTableError` | on-disk state failed a correctness check, or a data file is truncated or replaced (still an instance of the reader's error class, such as `OSError` or `pyarrow.ArrowInvalid`) |
| `MissingDataFileError` | a `CorruptTableError`: a file the snapshot references was removed (VACUUM, manual delete); `.path` names it |
| `ChangeFeedSchemaChangeError` | an `UnreachableTableError`: the change feed range crosses a schema change its rows cannot be read across; `.version` names the commit that changed it |
| `PredicateError` | a predicate uses SQL that cannot be evaluated outside a SQL engine |
| `SqlStatementError` | the SQL warehouse rejected or failed a statement; the message carries its error |
| `SqlPermissionError` | a `SqlStatementError` that is also a `PreflightError`: the warehouse refused a missing privilege, named in `.privilege` / `.securable`, with the grant to ask for in `.remedy` |
| `EnginePanicError` | an engine panicked across the FFI boundary |
| `EngineError` | an engine failed in a way no error above describes; `.engine`, `.operation` and `.original` (the engine's exception, also the `__cause__`) say where. The error stays an instance of the original's class too: an engine's `OSError` becomes an `EngineError` that is also an `OSError`, a `pyarrow.ArrowInvalid` one that is also an `ArrowInvalid`. The class combined is the original's own, or its nearest ancestor that can be: one below `Exception` that can be subclassed and built from a message, and is not a Rust extension's type (delta-rs's `DeltaError` family, this library's native errors, pyo3 panics, Polars'). delta-rs's own types map to this library's: `TableNotFoundError` to `TableNotFoundError`, `DeltaProtocolError` to `EngineLimitError` (or `InvalidArgumentError` for invariant violations), a non-conflict `CommitFailedError` to `CommitRefusedError`. An exception raised by your own data source -- a `__arrow_c_stream__` of yours, or the iterator behind a `RecordBatchReader` you pass -- reaches you as it was raised, not translated. A read that fails this way moves on to the next capable engine, with an `EngineFallbackWarning` |

Warnings all derive from `DeltaSwampWarning` (`SqlFallbackWarning`,
`EngineFallbackWarning`, `IgnoredPropertyWarning`, `CredentialExpiryWarning`),
so one filter silences them:
`warnings.filterwarnings("ignore", category=ds.DeltaSwampWarning)`.

A conflict means re-read the snapshot, recompute, then stage again at the next
version. A backfill demand is backpressure, not a rate limit: retrying it with
exponential backoff and no publish will wedge the table.

## Credentials

Unity Catalog vends short-lived, per-table storage credentials. They expire
on their own clock, separate from the catalog token the SDK refreshes. The
kernel's object store reads its credential on every request, and deltaswamp
refreshes it ahead of expiry, so a single long scan or write keeps working.
Vended S3 keys get their bucket's own region, not the metastore's, and never
go to an `AWS_ENDPOINT_URL` from the environment. Distributed plans refresh
on workers through `credential_source=` or `ship_catalog_auth=True` (see
"Distributed reads and writes"); tables served by delta-rs get a static
credential per operation. [Architecture](architecture.md#credentials) has the
details.

```python
creds = t.credentials()  # or credentials(write=True)
creds.expires_at  # epoch seconds, or None
creds.expires_within(300)
creds.redacted()  # safe to log
```

Credential *providers* are picklable; credentials are not, and pickling one
raises `TypeError`. A pickled provider, catalog, `Table` or `Connection` carries
the catalog configuration the driver resolved (host, auth type, client id) but
no literal secret: no PAT, no OAuth client secret, whether it came from
`token=` or from the environment. A worker's SDK re-derives auth from its own
environment (`DATABRICKS_TOKEN`, a profile, cloud-native auth).
`connect(ship_credentials=True)` pickles the secrets too, for workers that have
no auth of their own, and so does a plan made with `ship_catalog_auth=True`.
Distributed plans ship no catalog credentials by default; see "Distributed
reads and writes" above.

databricks-sdk logs every request and response at DEBUG. deltaswamp redacts
the records of the `databricks.sdk` logger and all its children: any field
whose name marks a secret, whatever the separators (`secret_access_key`,
`s3.secret-access-key`, `adls.sas-token.<host>`, `gcs.oauth2.token`,
`aad_token`, `access_token`, ...), `Authorization` and cookie header lines
(`debug_headers=True`), and presigned-URL signatures (`sig=`,
`X-Amz-Signature=`, ...) become `**REDACTED**`. Azure user-delegation SAS is
scoped to a path, so credentials are keyed by table, and Azure always gets an
explicit endpoint.

## Security notes

What deltaswamp refuses, and why, when the input comes from someone else --
another writer of a table's log, a catalog entry, a sharing server:

* **SQL text is one expression.** A predicate, SET or INSERT value, MERGE ON
  or clause condition, replaceWhere, CHECK constraint, or a generation or
  DEFAULT expression copied into warehouse DDL must be exactly one Spark SQL
  expression: literals closed, parentheses balanced, no `;`, no comment, no
  top-level `,`, nothing trailing. Malformed text raises `PredicateError`
  before any engine sees it; delta-rs used to act on a prefix of it
  (`id = 1) AND (...` deleted every `id = 1` row). The kernel MERGE evaluates
  its clauses in a sandboxed DuckDB (no file or network access, no
  extensions, configuration locked), one statement per execute.
  `Connection.sql(engine="duckdb")` is a plain local DuckDB with file and
  network access, by design: never build that SQL from untrusted input.
* **Credentials go only where you configured them.** An Azure location
  derives its endpoint only under Azure Storage's domains (`core.windows.net`
  and its private-link aliases, `core.chinacloudapi.cn`,
  `core.usgovcloudapi.net`, `core.cloudapi.de`, `fabric.microsoft.com`); any
  other host needs `azure_storage_endpoint` named by you, and a location in
  another account than `azure_storage_account_name` is refused. Vended R2
  keys go only to `*.r2.cloudflarestorage.com`. S3 and GCS endpoints come from
  your options or the cloud's own, never from a location. The OSS Unity Catalog
  client drops `Authorization` on a redirect to another origin, refuses a
  redirect from https to http, and warns when a token goes to a plain-http
  catalog that is not on this machine. A name part of `.` or `..` is encoded in
  REST paths.
* **A table reads only its own files.** A data file or deletion vector whose
  log path resolves outside the table root (`../x`, `%2E%2E/x`, another prefix
  or bucket) is refused by the kernel reads and by compaction.
* **Creating a table never adopts other files.** `create_table` and
  `write_table` refuse a directory that already holds files a Delta table does
  not leave behind (as Spark refuses a non-empty location), and a managed
  table's staging location on this machine's filesystem is refused unless the
  catalog runs on this machine. A full VACUUM deletes only orphans named like
  Delta files (`*.parquet`, `deletion_vector_*.bin`) and warns about the rest.
* **Delta Sharing downloads go to public https hosts.** A presigned URL must
  be https and must not resolve to a private, loopback, link-local or metadata
  address; every redirect hop is checked the same way. A local development
  server opts in with `DELTASWAMP_SHARING_ALLOW_PRIVATE_URLS=1`.

Dependency hygiene is the deployer's: the Python dependencies have lower bounds
only (`deltalake`, `duckdb`, `databricks-sdk`, ...), and the sandbox settings
and error texts this library relies on can change in a new major version.
Pinning upper bounds in your environment, a hash-pinned lock file for CI, and
`pip-audit` plus `cargo audit` / `cargo deny` in CI are recommended.

## SQL fallback

Views, materialized views, row-filtered tables, and the Databricks-only
operations (`DROP FEATURE`, `REORG`, `CLONE`, `ANALYZE`, `CLUSTER BY AUTO`,
`OPTIMIZE FULL`, `UNDROP`, row filters and masks) are served by a Databricks SQL
warehouse, since the work happens server-side. It is off by default:

```python
conn = ds.connect(
    allow_sql_fallback=True,
    warehouse_id="abc123",  # optional: a running warehouse is chosen otherwise
    staging_volume="main.default.staging",  # optional: enables writes and MERGE
)
```

Statements run through the Statement Execution API in `databricks-sdk`, with
results fetched as Arrow. Values are sent as statement parameters and
identifiers are quoted. `predicate` and `updates` strings are SQL expressions
by contract. Writes stage Parquet in the volume, load it with `read_files`, and
delete it afterward.

A warehouse changes latency and cost by orders of magnitude, so you opt in, and
a `SqlFallbackWarning` names the warehouse whenever the fallback runs.

## Known limits

[Conformance](conformance.md) has the full matrix, and the
[feature map](features.md) shows how each Databricks and open-source feature is
reached.

- Spark SQL that no direct engine can be made to evaluate as Spark does (an
  array subscript, `split`, a Java date pattern, `hash`) needs the warehouse.
  Differences that depend on the data's types stay: on delta-rs,
  `CAST(ts AS STRING)` has a `T` between date and time, `0.1 + 0.2 = 0.3` is
  false (short decimal literals are DOUBLEs), and a DOUBLE divided by zero is
  infinity rather than an error; DuckDB upper-cases `ß` to `ẞ`; neither
  raises on INT overflow where Spark's ANSI mode does.
- DML on a kernel-only table without deletion vectors rewrites the files it
  touches, and MERGE reads its source and candidate files into memory; both
  are refused past `KernelEngine.dml_max_bytes` (4 GiB) rather than running
  out of memory. UPDATE and MERGE on a change-data-feed table need the
  warehouse, since the kernel cannot write CDC files.
- The change feed of a catalog-managed table needs the warehouse.
- delta-rs reads pre-1582 dates and timestamps from Spark's legacy-calendar
  files unrebased (2-10 days off); the kernel rebases them. Ancient timestamps
  in such files written in a non-UTC session zone need the warehouse.
- delta-rs writes Parquet footers without `org.apache.spark.version`, which
  Databricks reads with the legacy calendar rebase, so writes and rewrites of
  dates before 1582-10-15 or timestamps before 1900 skip it (see above). A
  streamed write with DATE or TIMESTAMP columns, and an UPDATE or MERGE whose
  SET values cannot be bounded, count as holding such values: they go to the
  kernel, and are refused where it cannot serve them (a MERGE into a table
  without deletion vectors, `writer_properties=`).
- Distributed planning is kernel-only. For a table only another engine can
  read, `to_ray_dataset()` raises unless `allow_driver_read=True`.
- A MERGE with `merge_schema=True` whose SET or INSERT assigns a column the
  source does not have (closing an SCD2 row) is refused on delta-rs, which
  fails it, and needs the SQL fallback; the kernel MERGE does not evolve the
  schema at all.
- On a table at a legacy writer version 3 to 6 (every change-data-feed table
  created before table features), a change that turns on a feature delta-rs
  cannot write (clustering, type widening, in-commit timestamps) is refused
  locally: the upgraded protocol keeps listing checkConstraints and
  generatedColumns, as Databricks keeps them, which the kernel cannot write,
  so no local engine could write the table afterwards.
- Identity and literal-default columns are created locally (path and external
  tables: version 0 is composed here, the feature and typed identity metadata
  included; managed tables through the catalog's staging flow); generated
  columns are created by delta-rs. A catalog-managed table created by the
  kernel's own create path cannot declare identity or default columns. Once a
  table has them, the kernel writes it (appends, overwrites, distributed
  writes), computing and checking the values; DELETE, UPDATE and MERGE there
  stay with delta-rs or the warehouse.
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
- A `Table` captures catalog-managed commit state when it resolves. Re-open it
  to see commits made elsewhere.
- External writes to UC managed Delta are Public Preview on Databricks, and
  external access to catalog-commit tables is Beta behind a workspace preview.
  `preflight()` failing on a fresh workspace is expected.
