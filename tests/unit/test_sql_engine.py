"""The SQL warehouse fallback, without a warehouse.

Every operation is asserted on the exact SQL and parameters it produces, because
the two ways this engine can go wrong are both silent: a value spliced into
statement text (injection, or just a broken quote) and an identifier that is not
quoted (a column called ``a.b`` becoming a struct access).
"""

from __future__ import annotations

import datetime as dt
import io
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq
from deltaswamp.capability import (
    ENGINE_METHODS,
    OPERATION_ENGINES,
    Engine,
    Operation,
)
from deltaswamp.catalog import TableType
from deltaswamp.engine.sql import SqlEngine, SqlFallbackWarning, SqlMerger
from deltaswamp.engine.sql_backend import SqlStatementError
from deltaswamp.errors import UnreachableTableError
from deltaswamp.identity import RefKind, TableRef

from tests.helpers import resolved_table as table
from tests.unit.sql_fakes import (
    FakeClient,
    FakeStatements,
    RecordingBackend,
    engine,
    succeeded,
    warehouse,
)

NAME = "`main`.`sales`.`orders`"

# ------------------------------------------------------------ quoting


class TestIdentifiers:
    def test_hostile_names_are_one_identifier_each(self) -> None:
        eng, rec, _ = engine()
        hostile = TableRef(kind=RefKind.CATALOG, catalog="my cat", schema="sch.ema", table="t`x")
        eng.scan(table(hostile), columns=["a.b", "odd`col", "my col"])
        assert rec.last == "SELECT `a.b`, `odd``col`, `my col` FROM `my cat`.`sch.ema`.`t``x`"

    def test_nested_column_paths_are_quoted_per_part(self) -> None:
        eng, rec, _ = engine()
        eng.set_column_comment(table(), ["addr", "zip code"], "postal")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `addr`.`zip code` COMMENT 'postal'"

    def test_path_references_are_refused(self) -> None:
        eng, _, _ = engine()
        path = table("s3://bucket/t")
        got = eng.supports(Operation.SCAN, path)
        assert not got.ok and "path reference" in got.reason


# ------------------------------------------------------------ reads


class TestReads:
    def test_scan_with_version_and_predicate(self) -> None:
        eng, rec, _ = engine()
        eng.scan(table(), predicate="id > 3", version=7)
        assert rec.last == f"SELECT * FROM {NAME} VERSION AS OF 7 WHERE id > 3"

    def test_limit_is_pushed_into_the_statement(self) -> None:
        """The warehouse computes the whole result before streaming any of it.

        Without a real LIMIT, `head(3)` on a large table is a full scan billed
        to the warehouse, and slow enough to hit the statement timeout.
        """
        eng, rec, _ = engine()
        eng.scan(table(), limit=3)
        assert rec.last == f"SELECT * FROM {NAME} LIMIT 3"

    def test_limit_composes_with_predicate_and_travel(self) -> None:
        eng, rec, _ = engine()
        eng.scan(table(), predicate="id > 3", version=7, limit=5)
        assert rec.last == f"SELECT * FROM {NAME} VERSION AS OF 7 WHERE id > 3 LIMIT 5"

    def test_no_limit_means_no_clause(self) -> None:
        eng, rec, _ = engine()
        eng.scan(table())
        assert "LIMIT" not in rec.last

    def test_limit_cannot_carry_sql(self) -> None:
        """A LIMIT cannot be a bound parameter, so it is coerced, not spliced.

        Every other value in this engine goes through the binder; this one
        cannot, which is exactly the shape that becomes an injection.
        """
        eng, rec, _ = engine()
        with pytest.raises((ValueError, TypeError)):
            eng.scan(table(), limit="1; DROP TABLE x; --")  # type: ignore[arg-type]
        eng.scan(table(), limit=5)
        assert rec.last.endswith("LIMIT 5")

    def test_timestamp_travel_is_a_parameter(self) -> None:
        eng, rec, _ = engine()
        eng.scan(table(), timestamp="2024-01-01'; DROP TABLE x; --")
        assert rec.last == f"SELECT * FROM {NAME} TIMESTAMP AS OF :p0"
        assert rec.params == {"p0": ("2024-01-01'; DROP TABLE x; --", "TIMESTAMP")}

    def test_history(self) -> None:
        rows = pa.table({"version": [3, 2]})
        eng, rec, _ = engine(RecordingBackend({"DESCRIBE HISTORY": rows}))
        assert eng.history(table(), limit=2) == [{"version": 3}, {"version": 2}]
        assert rec.last == f"DESCRIBE HISTORY {NAME} LIMIT 2"

    def test_detail_adds_version_and_normalized_keys(self) -> None:
        detail = pa.table(
            {
                "location": ["s3://b/t"],
                "minReaderVersion": [3],
                "minWriterVersion": [7],
                "tableFeatures": [["deletionVectors"]],
                "partitionColumns": [["day"]],
                "properties": [[("a", "1")]],
            },
            schema=pa.schema(
                [
                    ("location", pa.string()),
                    ("minReaderVersion", pa.int32()),
                    ("minWriterVersion", pa.int32()),
                    ("tableFeatures", pa.list_(pa.string())),
                    ("partitionColumns", pa.list_(pa.string())),
                    ("properties", pa.map_(pa.string(), pa.string())),
                ]
            ),
        )
        backend = RecordingBackend(
            {"DESCRIBE DETAIL": detail, "DESCRIBE HISTORY": pa.table({"version": [12]})}
        )
        eng, rec, _ = engine(backend)
        got = eng.detail(table())
        assert got["version"] == 12
        assert got["min_reader_version"] == 3
        assert got["properties"] == {"a": "1"}
        assert got["partition_columns"] == ["day"]
        assert rec.sql == [f"DESCRIBE DETAIL {NAME}", f"DESCRIBE HISTORY {NAME} LIMIT 1"]

    def test_detail_refuses_an_old_version(self) -> None:
        eng, _, _ = engine()
        with pytest.raises(UnreachableTableError, match="current version"):
            eng.detail(table(), version=3)

    def test_cdf_versions_are_parameters(self) -> None:
        eng, rec, _ = engine()
        eng.cdf(table(), starting_version=2, ending_version=5, columns=["id"], predicate="id > 1")
        assert rec.last == (
            "SELECT `id` FROM table_changes('`main`.`sales`.`orders`', :p0, :p1) WHERE id > 1"
        )
        assert rec.params == {"p0": ("2", "BIGINT"), "p1": ("5", "BIGINT")}

    def test_cdf_timestamps(self) -> None:
        eng, rec, _ = engine()
        when = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.UTC)
        eng.cdf(table(), starting_timestamp=when, ending_timestamp="2024-06-01")
        assert rec.last == f"SELECT * FROM table_changes({_lit(NAME)}, :p0, :p1)"
        assert rec.params == {
            "p0": ("2024-05-01T12:00:00+00:00", "TIMESTAMP"),
            "p1": ("2024-06-01", "TIMESTAMP"),
        }

    def test_cdf_defaults_to_version_zero(self) -> None:
        eng, rec, _ = engine()
        eng.cdf(table())
        assert rec.params == {"p0": ("0", "BIGINT")}

    def test_describe_extended(self) -> None:
        rows = pa.table(
            {
                "col_name": ["id", "name", "", "# Detailed Table Information", "Owner", "Type"],
                "data_type": ["bigint", "string", "", "", "alice", "MANAGED"],
                "comment": ["pk", None, "", "", "", ""],
            }
        )
        eng, rec, _ = engine(RecordingBackend({"DESCRIBE TABLE EXTENDED": rows}))
        got = eng.describe_extended(table())
        assert got["columns"] == [
            {"name": "id", "type": "bigint", "comment": "pk"},
            {"name": "name", "type": "string", "comment": None},
        ]
        assert got["Owner"] == "alice" and got["Type"] == "MANAGED"
        assert rec.last == f"DESCRIBE TABLE EXTENDED {NAME}"


def _lit(text: str) -> str:
    return "'" + text + "'"


# ------------------------------------------------------------ dml


class TestDml:
    def test_delete(self) -> None:
        eng, rec, _ = engine()
        eng.delete(table(), "id = 1")
        assert rec.last == f"DELETE FROM {NAME} WHERE id = 1"

    def test_update_quotes_keys_and_binds_plain_values(self) -> None:
        eng, rec, _ = engine()
        eng.update(
            table(),
            updates={"n; DROP": "n + 1"},
            new_values={"city": "O'Hare", "ok": True},
            predicate="id = 2",
        )
        assert rec.last == (
            f"UPDATE {NAME} SET `n; DROP` = n + 1, `city` = :p0, `ok` = :p1 WHERE id = 2"
        )
        assert rec.params == {"p0": ("O'Hare", "STRING"), "p1": ("true", "BOOLEAN")}

    def test_update_needs_something(self) -> None:
        eng, _, _ = engine()
        with pytest.raises(UnreachableTableError):
            eng.update(table())


# ------------------------------------------------------------ ddl


class TestDdl:
    def test_set_properties_escapes_values(self) -> None:
        eng, rec, _ = engine()
        eng.set_properties(table(), {"k'x": "it's \\' here"})
        assert rec.last == (
            f"ALTER TABLE {NAME} SET TBLPROPERTIES ('k\\'x' = 'it\\'s \\\\\\' here')"
        )

    def test_unset_properties(self) -> None:
        eng, rec, _ = engine()
        eng.unset_properties(table(), ["a", "b"])
        assert rec.last == f"ALTER TABLE {NAME} UNSET TBLPROPERTIES IF EXISTS ('a', 'b')"
        eng.unset_properties(table(), ["a"], if_exists=False)
        assert rec.last == f"ALTER TABLE {NAME} UNSET TBLPROPERTIES ('a')"

    def test_constraints(self) -> None:
        eng, rec, _ = engine()
        eng.add_constraint(table(), {"positive id": "id > 0"})
        assert rec.last == f"ALTER TABLE {NAME} ADD CONSTRAINT `positive id` CHECK (id > 0)"
        eng.drop_constraint(table(), "positive id", if_exists=True)
        assert rec.last == f"ALTER TABLE {NAME} DROP CONSTRAINT IF EXISTS `positive id`"
        eng.drop_constraint(table(), "c")
        assert rec.last == f"ALTER TABLE {NAME} DROP CONSTRAINT `c`"

    def test_comments(self) -> None:
        eng, rec, _ = engine()
        eng.set_comment(table(), "Bob's table")
        assert rec.last == f"COMMENT ON TABLE {NAME} IS 'Bob\\'s table'"
        eng.set_comment(table(), None)
        assert rec.last == f"COMMENT ON TABLE {NAME} IS NULL"
        eng.set_column_comment(table(), "my col", "x")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `my col` COMMENT 'x'"

    def test_column_type_and_nullability(self) -> None:
        eng, rec, _ = engine()
        eng.alter_column_type(table(), "n", "BIGINT")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `n` TYPE BIGINT"
        eng.set_not_null(table(), "n")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `n` SET NOT NULL"
        eng.drop_not_null(table(), "n")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `n` DROP NOT NULL"

    def test_a_type_cannot_smuggle_a_statement(self) -> None:
        eng, _, _ = engine()
        with pytest.raises(ValueError):
            eng.alter_column_type(table(), "n", "INT; DROP TABLE x")

    @pytest.mark.parametrize(
        ("columns", "clause"),
        [
            (["a", "b c"], "CLUSTER BY (`a`, `b c`)"),
            ([], "CLUSTER BY NONE"),
            (None, "CLUSTER BY NONE"),
            ("auto", "CLUSTER BY AUTO"),
        ],
    )
    def test_cluster_by(self, columns: Any, clause: str) -> None:
        eng, rec, _ = engine()
        eng.cluster_by(table(), columns)
        assert rec.last == f"ALTER TABLE {NAME} {clause}"

    def test_columns(self) -> None:
        eng, rec, _ = engine()
        eng.add_columns(table(), {"new col": "DECIMAL(10,2)"})
        assert rec.last == f"ALTER TABLE {NAME} ADD COLUMNS (`new col` DECIMAL(10,2))"
        eng.add_columns(table(), pa.schema([("a", pa.int64()), ("t", pa.list_(pa.string()))]))
        assert rec.last == f"ALTER TABLE {NAME} ADD COLUMNS (`a` BIGINT, `t` ARRAY<STRING>)"
        eng.drop_column(table(), "a.b")
        assert rec.last == f"ALTER TABLE {NAME} DROP COLUMN `a.b`"
        eng.rename_column(table(), "old", "n`ew")
        assert rec.last == f"ALTER TABLE {NAME} RENAME COLUMN `old` TO `n``ew`"

    def test_features(self) -> None:
        eng, rec, _ = engine()
        eng.add_feature(table(), ["deletionVectors", "v2Checkpoint"])
        assert rec.last == (
            f"ALTER TABLE {NAME} SET TBLPROPERTIES ('delta.feature.deletionVectors' = "
            "'supported', 'delta.feature.v2Checkpoint' = 'supported')"
        )
        eng.drop_feature(table(), "deletionVectors", truncate_history=True)
        assert rec.last == f"ALTER TABLE {NAME} DROP FEATURE `deletionVectors` TRUNCATE HISTORY"


# ------------------------------------------------------------ maintenance


class TestMaintenance:
    def test_optimize(self) -> None:
        eng, rec, _ = engine()
        eng.optimize(table())
        assert rec.last == f"OPTIMIZE {NAME}"
        eng.optimize(table(), full=True)
        assert rec.last == f"OPTIMIZE {NAME} FULL"
        eng.optimize(table(), predicate="day > '2024'", zorder_by=["a", "b"])
        assert rec.last == f"OPTIMIZE {NAME} WHERE (day > '2024') ZORDER BY (`a`, `b`)"
        eng.zorder(table(), ["c"])
        assert rec.last == f"OPTIMIZE {NAME} ZORDER BY (`c`)"

    def test_partition_filters_are_parameters(self) -> None:
        eng, rec, _ = engine()
        eng.optimize(table(), partition_filters=[("day", "=", "x'y"), ("r", "in", [1, 2])])
        assert rec.last == f"OPTIMIZE {NAME} WHERE `day` = :p0 AND `r` IN (:p1, :p2)"
        assert rec.params == {
            "p0": ("x'y", "STRING"),
            "p1": ("1", "BIGINT"),
            "p2": ("2", "BIGINT"),
        }

    def test_unknown_arguments_are_refused_not_ignored(self) -> None:
        eng, _, _ = engine()
        with pytest.raises(UnreachableTableError, match="target_size"):
            eng.optimize(table(), target_size=1 << 20)
        eng.optimize(table(), max_concurrent_tasks=4)  # tuning only: accepted

    @pytest.mark.parametrize(
        ("kwargs", "suffix"),
        [
            ({}, " DRY RUN"),
            ({"dry_run": False}, ""),
            ({"retention_hours": 168, "dry_run": False}, " RETAIN 168 HOURS"),
            ({"lite": True, "retention_hours": 1.5}, " LITE RETAIN 1.5 HOURS DRY RUN"),
            ({"full": True, "dry_run": False}, " FULL"),
        ],
    )
    def test_vacuum(self, kwargs: dict[str, Any], suffix: str) -> None:
        eng, rec, _ = engine(RecordingBackend({"VACUUM": pa.table({"path": ["s3://b/f1"]})}))
        assert eng.vacuum(table(), **kwargs) == ["s3://b/f1"]
        assert rec.last == f"VACUUM {NAME}{suffix}"

    def test_vacuum_cannot_disable_the_retention_check(self) -> None:
        eng, _, _ = engine()
        with pytest.raises(UnreachableTableError, match="session"):
            eng.vacuum(table(), enforce_retention_duration=False)

    def test_restore(self) -> None:
        eng, rec, _ = engine()
        eng.restore(table(), 4)
        assert rec.last == f"RESTORE TABLE {NAME} TO VERSION AS OF 4"
        eng.restore(table(), "2024-01-01 00:00:00")
        assert rec.last == f"RESTORE TABLE {NAME} TO TIMESTAMP AS OF '2024-01-01 00:00:00'"
        eng.restore(table(), dt.datetime(2024, 1, 2, 3, 4, 5))
        assert rec.last == f"RESTORE TABLE {NAME} TO TIMESTAMP AS OF '2024-01-02T03:04:05'"

    def test_repair(self) -> None:
        rows = pa.table({"dataFilePath": ["f1"], "dataFileMissing": [True]})
        eng, rec, _ = engine(RecordingBackend({"FSCK": rows}))
        assert eng.repair(table(), dry_run=True) == {"dry_run": True, "files_removed": ["f1"]}
        assert rec.last == f"FSCK REPAIR TABLE {NAME} DRY RUN"
        eng.repair(table())
        assert rec.last == f"FSCK REPAIR TABLE {NAME}"

    def test_analyze(self) -> None:
        eng, rec, _ = engine()
        eng.analyze(table())
        assert rec.last == f"ANALYZE TABLE {NAME} COMPUTE STATISTICS"
        eng.analyze(table(), delta_statistics=True)
        assert rec.last == f"ANALYZE TABLE {NAME} COMPUTE DELTA STATISTICS"
        eng.analyze(table(), columns=["a", "b c"])
        assert rec.last == f"ANALYZE TABLE {NAME} COMPUTE STATISTICS FOR COLUMNS `a`, `b c`"
        eng.analyze(table(), columns="all")
        assert rec.last == f"ANALYZE TABLE {NAME} COMPUTE STATISTICS FOR ALL COLUMNS"

    def test_reorg_clone_sync(self) -> None:
        eng, rec, _ = engine()
        eng.reorg(table())
        assert rec.last == f"REORG TABLE {NAME} APPLY (PURGE)"
        eng.reorg(table(), iceberg_compat_version=2)
        assert rec.last == (f"REORG TABLE {NAME} APPLY (UPGRADE UNIFORM(ICEBERG_COMPAT_VERSION=2))")
        eng.clone(table(), "main.dev.`orders copy`", shallow=False, replace=True, version=3)
        assert rec.last == (
            f"CREATE OR REPLACE TABLE `main`.`dev`.`orders copy` DEEP CLONE {NAME} VERSION AS OF 3"
        )
        eng.sync_iceberg_metadata(table())
        assert rec.last == f"MSCK REPAIR TABLE {NAME} SYNC METADATA"
        assert rec.calls[-1][2] is False  # DDL does not fetch results

    def test_refresh_picks_the_statement_by_table_type(self) -> None:
        eng, rec, _ = engine()
        eng.refresh(table(table_type=TableType.MATERIALIZED_VIEW), full=True)
        assert rec.last == f"REFRESH MATERIALIZED VIEW {NAME} FULL"
        eng.refresh(table(table_type=TableType.STREAMING_TABLE))
        assert rec.last == f"REFRESH STREAMING TABLE {NAME}"
        with pytest.raises(UnreachableTableError):
            eng.refresh(table(table_type=TableType.MANAGED))
        got = eng.supports(Operation.REFRESH, table(table_type=TableType.MANAGED))
        assert not got.ok and "materialized views" in got.reason


# ------------------------------------------------------------ extras


class TestExtras:
    def test_query_binds_named_parameters(self) -> None:
        eng, rec, _ = engine()
        eng.query("SELECT * FROM t WHERE id = :id AND d = :d", {"id": 5, "d": dt.date(2024, 1, 1)})
        assert rec.params == {"id": ("5", "BIGINT"), "d": ("2024-01-01", "DATE")}

    def test_governance(self) -> None:
        eng, rec, _ = engine()
        t = table()
        eng.undrop("main.sales.`old orders`")
        assert rec.last == "UNDROP TABLE `main`.`sales`.`old orders`"
        eng.set_row_filter(t, "main.sec.region_filter", ["region"])
        assert rec.last == (
            f"ALTER TABLE {NAME} SET ROW FILTER `main`.`sec`.`region_filter` ON (`region`)"
        )
        eng.drop_row_filter(t)
        assert rec.last == f"ALTER TABLE {NAME} DROP ROW FILTER"
        eng.set_column_mask(t, "ssn", "main.sec.mask", using_columns=["role"])
        assert rec.last == (
            f"ALTER TABLE {NAME} ALTER COLUMN `ssn` SET MASK `main`.`sec`.`mask` "
            "USING COLUMNS (`role`)"
        )
        eng.drop_column_mask(t, "ssn")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `ssn` DROP MASK"
        eng.set_tags(t, {"pii": "yes"})
        assert rec.last == f"ALTER TABLE {NAME} SET TAGS ('pii' = 'yes')"
        eng.set_tags(t, {"pii": "it's"}, column="ssn")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `ssn` SET TAGS ('pii' = 'it\\'s')"
        eng.unset_tags(t, ["pii"], column="ssn")
        assert rec.last == f"ALTER TABLE {NAME} ALTER COLUMN `ssn` UNSET TAGS ('pii')"
        eng.set_owner(t, "data-eng@corp.com")
        assert rec.last == f"ALTER TABLE {NAME} OWNER TO `data-eng@corp.com`"
        eng.grant(t, ["select", "MODIFY"], "analysts")
        assert rec.last == f"GRANT SELECT, MODIFY ON TABLE {NAME} TO `analysts`"
        eng.revoke(t, "SELECT", "bad`actor")
        assert rec.last == f"REVOKE SELECT ON TABLE {NAME} FROM `bad``actor`"

    def test_privileges_are_validated(self) -> None:
        eng, _, _ = engine()
        with pytest.raises(ValueError):
            eng.grant(table(), "SELECT ON x TO y; --", "p")

    def test_create_table_as(self) -> None:
        eng, rec, _ = engine()
        eng.create_table_as(
            "main.dev.t2",
            "SELECT * FROM main.sales.orders WHERE id > :n",
            replace=True,
            comment="copy",
            cluster_by=["id"],
            parameters={"n": 3},
        )
        assert rec.last == (
            "CREATE OR REPLACE TABLE `main`.`dev`.`t2` CLUSTER BY (`id`) COMMENT 'copy' "
            "AS SELECT * FROM main.sales.orders WHERE id > :n"
        )
        assert rec.params == {"n": ("3", "BIGINT")}


# ------------------------------------------------------------ staged writes


def _staged_path(client: FakeClient) -> str:
    (path,) = client.files.uploaded
    return path


class TestStagedWrites:
    def test_append_uploads_parquet_runs_insert_and_deletes(self) -> None:
        eng, rec, client = engine()
        data = pa.table({"id": [1, 2], "city": ["oslo", "lima"]})
        eng.append(table(), data)
        path = _staged_path(client)
        assert path.startswith("/Volumes/cat/sch/vol/deltaswamp-staging/")
        assert path.endswith(".parquet")
        assert pq.read_table(io.BytesIO(client.files.uploaded[path])).equals(data)
        # Projected to the data's own columns: read_files adds `_rescued_data`,
        # which INSERT ... BY NAME rejects as an extra column.
        assert rec.last == (
            f"INSERT INTO {NAME} BY NAME SELECT * FROM (SELECT `id`, `city` "
            f"FROM read_files('{path}', format => 'parquet'))"
        )
        assert client.files.deleted == [path]

    def test_staged_file_is_deleted_when_the_statement_fails(self) -> None:
        eng, rec, client = engine()
        rec.fail_on = "INSERT"
        with pytest.raises(SqlStatementError, match="boom"):
            eng.append(table(), pa.table({"id": [1]}))
        path = _staged_path(client)
        assert client.files.events == [f"upload {path}", f"delete {path}"]

    def test_append_accepts_pandas_polars_and_arrow_streams(self) -> None:
        pd = pytest.importorskip("pandas")
        pl = pytest.importorskip("polars")

        for data in (
            pd.DataFrame({"id": [1]}),
            pl.DataFrame({"id": [1]}),
            pa.RecordBatchReader.from_batches(
                pa.schema([("id", pa.int64())]), [pa.record_batch({"id": [1]})]
            ),
        ):
            eng, _, client = engine()
            eng.append(table(), data)
            staged = pq.read_table(io.BytesIO(client.files.uploaded[_staged_path(client)]))
            assert staged.column("id").to_pylist() == [1]

    def test_append_with_schema_evolution(self) -> None:
        eng, rec, _ = engine()
        eng.append(table(), pa.table({"id": [1]}), schema_mode="merge")
        assert rec.last.startswith(f"INSERT WITH SCHEMA EVOLUTION INTO {NAME} BY NAME SELECT")

    def test_overwrite(self) -> None:
        eng, rec, client = engine()
        eng.overwrite(table(), pa.table({"id": [1]}))
        assert rec.last.startswith(
            f"INSERT OVERWRITE {NAME} BY NAME SELECT * FROM (SELECT `id` FROM read_files("
        )
        assert client.files.deleted

    def test_replace_where_orders_columns_like_the_table(self) -> None:
        schema = pa.table({"id": pa.array([], pa.int64()), "day": pa.array([], pa.string())})
        eng, rec, client = engine(RecordingBackend({"SELECT * FROM": schema}))
        eng.overwrite(table(), pa.table({"day": ["d1"], "id": [1]}), predicate="day = 'd1'")
        path = _staged_path(client)
        assert rec.sql == [
            f"SELECT * FROM {NAME} LIMIT 0",
            f"INSERT INTO {NAME} REPLACE WHERE day = 'd1' SELECT `id`, `day` "
            f"FROM (SELECT `day`, `id` FROM read_files('{path}', format => 'parquet'))",
        ]
        assert client.files.deleted == [path]

    def test_dynamic_overwrite_binds_partition_values(self) -> None:
        schema = pa.table({"id": pa.array([], pa.int64()), "day": pa.array([], pa.string())})
        eng, rec, _ = engine(RecordingBackend({"SELECT * FROM": schema}))
        data = pa.table({"id": [1, 2, 3], "day": ["a'1", "b", None]})
        eng.overwrite(table(partition_columns=("day",)), data, partition_overwrite="dynamic")
        assert "REPLACE WHERE (`day` = :p0) OR (`day` = :p1) OR (`day` IS NULL) SELECT" in rec.last
        assert sorted(v for v, _ in rec.params.values()) == ["a'1", "b"]  # type: ignore[type-var]

    def test_schema_overwrite_and_commit_metadata_are_refused(self) -> None:
        eng, _, client = engine()
        with pytest.raises(UnreachableTableError, match="discards"):
            eng.overwrite(table(), pa.table({"id": [1]}), schema_mode="overwrite")
        with pytest.raises(UnreachableTableError, match="commit_metadata"):
            eng.append(table(), pa.table({"id": [1]}), commit_metadata={"a": 1})
        assert not client.files.uploaded  # refused before anything was staged

    def test_writes_without_a_staging_volume_are_refused_up_front(self) -> None:
        eng, _, _ = engine(staging_volume=None)
        for op in (Operation.APPEND, Operation.OVERWRITE, Operation.REPLACE_WHERE, Operation.MERGE):
            got = eng.supports(op, table())
            assert not got.ok
            assert "staging volume" in got.reason
            assert "staging_volume=" in (got.remedy or "")
        assert eng.supports(Operation.DELETE, table()).ok

    def test_bad_staging_volume(self) -> None:
        with pytest.raises(ValueError):
            SqlEngine(staging_volume="just_a_name")


class TestMerge:
    def test_builder_generates_one_merge(self) -> None:
        eng, rec, client = engine()
        source = pa.table({"id": [1], "v": ["x"], "ts": [1]})
        merger = eng.merge(table(), source, "t.id = s.id", source_alias="s", target_alias="t")
        assert isinstance(merger, SqlMerger)
        result = (
            merger.when_matched_update({"v": "s.v"}, predicate="s.ts > t.ts")
            .when_matched_update_all(except_cols=["ts"])
            .when_matched_delete("s.v IS NULL")
            .when_not_matched_insert({"id": "s.id", "v": "s.v"}, predicate="s.id > 0")
            .when_not_matched_insert_all()
            .when_not_matched_by_source_update({"v": "'gone'"})
            .when_not_matched_by_source_delete()
            .execute()
        )
        path = _staged_path(client)
        assert rec.last == (
            f"MERGE INTO {NAME} AS `t` USING (SELECT * FROM (SELECT `id`, `v`, `ts` "
            f"FROM read_files('{path}', format => 'parquet'))) AS `s` ON t.id = s.id"
            " WHEN MATCHED AND s.ts > t.ts THEN UPDATE SET `t`.`v` = s.v"
            " WHEN MATCHED THEN UPDATE SET `t`.`id` = `s`.`id`, `t`.`v` = `s`.`v`"
            " WHEN MATCHED AND s.v IS NULL THEN DELETE"
            " WHEN NOT MATCHED AND s.id > 0 THEN INSERT (`id`, `v`) VALUES (s.id, s.v)"
            " WHEN NOT MATCHED THEN INSERT *"
            " WHEN NOT MATCHED BY SOURCE THEN UPDATE SET `t`.`v` = 'gone'"
            " WHEN NOT MATCHED BY SOURCE THEN DELETE"
        )
        assert client.files.deleted == [path]
        assert result  # metrics row, or {"status": "ok"}

    def test_default_aliases_and_schema_evolution(self) -> None:
        eng, rec, _ = engine()
        eng.merge(
            table(), pa.table({"id": [1]}), "target.id = source.id", merge_schema=True
        ).when_matched_update_all().execute()
        assert rec.last.startswith(f"MERGE WITH SCHEMA EVOLUTION INTO {NAME} AS `target` USING")
        assert "AS `source` ON target.id = source.id WHEN MATCHED THEN UPDATE SET *" in rec.last

    def test_merge_failure_still_deletes_the_staged_source(self) -> None:
        eng, rec, client = engine()
        rec.fail_on = "MERGE"
        merger = eng.merge(table(), pa.table({"id": [1]}), "target.id = source.id")
        with pytest.raises(SqlStatementError):
            merger.when_not_matched_insert_all().execute()
        assert client.files.deleted == [_staged_path(client)]

    def test_a_merge_needs_a_clause(self) -> None:
        eng, _, client = engine()
        with pytest.raises(ValueError):
            eng.merge(table(), pa.table({"id": [1]}), "p").execute()
        assert not client.files.uploaded


# ------------------------------------------------------------ supports()


class TestSupports:
    def test_every_routed_operation_is_supported_when_configured(self) -> None:
        eng, _, _ = engine()
        for op, routing in OPERATION_ENGINES.items():
            if Engine.SQL not in routing.engines:
                continue
            assert callable(getattr(eng, ENGINE_METHODS[op])), op
            shaped = table(
                table_type=TableType.MATERIALIZED_VIEW if op is Operation.REFRESH else None
            )
            got = eng.supports(op, shaped)
            assert got.ok, (op, got.reason)

    def test_refuses_without_a_warehouse(self) -> None:
        eng = SqlEngine(client=FakeClient(), auto_select_warehouse=False)
        got = eng.supports(Operation.SCAN, table())
        assert not got.ok
        assert "no SQL warehouse" in got.reason
        assert "warehouse_id" in (got.remedy or "")

    def test_refuses_when_the_workspace_has_no_warehouse(self) -> None:
        eng = SqlEngine(client=FakeClient(warehouses=[warehouse("w", "DELETED")]))
        got = eng.supports(Operation.SCAN, table())
        assert not got.ok and "has none" in got.reason

    def test_listing_failure_is_a_refusal_not_an_exception(self) -> None:
        client = FakeClient()

        def broken() -> list[Any]:
            raise RuntimeError("403 forbidden")

        client.warehouses.list = broken  # type: ignore[method-assign]
        got = SqlEngine(client=client).supports(Operation.SCAN, table())
        assert not got.ok and "403 forbidden" in got.reason

    def test_views_cannot_be_written(self) -> None:
        eng, _, _ = engine()
        view = table(table_type=TableType.VIEW)
        assert eng.supports(Operation.SCAN, view).ok
        assert not eng.supports(Operation.DELETE, view).ok
        assert not eng.supports(Operation.HISTORY, view).ok

    def test_http_path_names_the_warehouse(self) -> None:
        eng = SqlEngine(client=FakeClient(), http_path="/sql/1.0/warehouses/abc123")
        assert eng.warehouse_id == "abc123"


class TestWarehouseSelection:
    @pytest.mark.parametrize(
        ("fleet", "expected"),
        [
            (
                [
                    warehouse("stopped", "STOPPED"),
                    warehouse("classic", "RUNNING"),
                    warehouse("sls", "RUNNING", serverless=True),
                ],
                "sls",
            ),
            ([warehouse("stopped", "STOPPED"), warehouse("classic", "RUNNING")], "classic"),
            ([warehouse("gone", "DELETED"), warehouse("stopped", "STOPPED")], "stopped"),
            (
                [
                    warehouse("sls-off", "STOPPED", serverless=True),
                    warehouse("classic", "RUNNING"),
                ],
                "classic",
            ),
        ],
    )
    def test_preference_order(self, fleet: list[Any], expected: str) -> None:
        eng = SqlEngine(client=FakeClient(warehouses=fleet))
        assert eng.warehouse_id == expected

    def test_an_explicit_warehouse_skips_listing(self) -> None:
        client = FakeClient(warehouses=[warehouse("other", "RUNNING")])
        eng = SqlEngine(client=client, warehouse_id="mine")
        assert eng.warehouse_id == "mine"
        assert client.warehouses.calls == 0

    def test_selection_is_cached(self) -> None:
        client = FakeClient(warehouses=[warehouse("w", "RUNNING")])
        eng = SqlEngine(client=client)
        for _ in range(3):
            eng.supports(Operation.SCAN, table())
        assert client.warehouses.calls == 1

    def test_the_warning_names_the_chosen_warehouse(self) -> None:
        statements = FakeStatements([succeeded("s1")])
        client = FakeClient(
            warehouses=[warehouse("w1", "RUNNING", serverless=True, name="Serverless Starter")],
            statements=statements,
        )
        eng = SqlEngine(client=client)
        with pytest.warns(SqlFallbackWarning, match=r"'Serverless Starter' \(w1\).*automatically"):
            eng.drop_row_filter(table())
        assert statements.executed[0]["warehouse_id"] == "w1"


class TestTableShapeRefusals:
    """Statements the warehouse would reject for this table, refused before sending."""

    def test_zorder_on_a_clustered_table(self) -> None:
        t = table(writer_features=frozenset({"clustering", "domainMetadata"}))
        verdict = SqlEngine._table_type_refusal(Operation.ZORDER, t)
        assert verdict is not None
        assert not verdict.ok

    @pytest.mark.parametrize("kind", [TableType.VIEW, TableType.MATERIALIZED_VIEW])
    @pytest.mark.parametrize("operation", [Operation.HISTORY, Operation.DETAIL])
    def test_log_reads_on_views(self, kind: TableType, operation: Operation) -> None:
        verdict = SqlEngine._table_type_refusal(operation, table(table_type=kind))
        assert verdict is not None
        assert not verdict.ok

    def test_type_change_needs_type_widening(self) -> None:
        t = table(writer_features=frozenset({"appendOnly"}))
        verdict = SqlEngine._table_type_refusal(Operation.ALTER_COLUMN_TYPE, t)
        assert verdict is not None
        assert "enableTypeWidening" in verdict.remedy

    def test_type_change_with_type_widening(self) -> None:
        on = table(
            writer_features=frozenset({"typeWidening"}), reader_features=frozenset({"typeWidening"})
        )
        assert SqlEngine._table_type_refusal(Operation.ALTER_COLUMN_TYPE, on) is None
        by_property = table(
            writer_features=frozenset({"appendOnly"}),
            properties={"delta.enableTypeWidening": "true"},
        )
        assert SqlEngine._table_type_refusal(Operation.ALTER_COLUMN_TYPE, by_property) is None

    @pytest.mark.parametrize("operation", [Operation.DROP_COLUMN, Operation.RENAME_COLUMN])
    def test_drops_and_renames_need_column_mapping(self, operation: Operation) -> None:
        plain = table(writer_features=frozenset({"appendOnly"}))
        verdict = SqlEngine._table_type_refusal(operation, plain)
        assert verdict is not None
        assert "columnMapping" in verdict.remedy
        mapped = table(
            writer_features=frozenset({"columnMapping"}),
            reader_features=frozenset({"columnMapping"}),
            properties={"delta.columnMapping.mode": "name"},
        )
        assert SqlEngine._table_type_refusal(operation, mapped) is None


class TestCreateManaged:
    """Managed tables are created with a warehouse CREATE TABLE statement."""

    def test_ddl(self) -> None:
        eng, rec, _ = engine()
        schema = pa.schema(
            [pa.field("id", pa.int64(), nullable=False), pa.field("weird col", pa.string())]
        )
        eng.create_managed(
            "main.s.t",
            schema,
            cluster_by=["id"],
            properties={"delta.enableChangeDataFeed": "true"},
            comment="it's here",
        )
        assert rec.last == (
            "CREATE TABLE `main`.`s`.`t` (`id` BIGINT NOT NULL, `weird col` STRING) USING DELTA"
            " CLUSTER BY (`id`) COMMENT 'it\\'s here'"
            " TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
        )

    def test_partitioned_and_clustered_is_refused(self) -> None:
        eng, _, _ = engine()
        with pytest.raises(UnreachableTableError):
            eng.create_managed(
                "main.s.t", pa.schema([("id", pa.int64())]), partition_by=["id"], cluster_by=["id"]
            )
