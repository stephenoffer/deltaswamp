"""Regressions for the ALTER / maintenance audit, each proven on a real table.

Every test here failed before its fix.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any

import pytest
from deltaswamp.errors import InvalidArgumentError, UnreachableTableError

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")


def _hive_dir(tmp_path: Any, regions: list[str]) -> str:
    path = str(tmp_path / "pq")
    os.makedirs(path)
    table = pa.table({"id": pa.array(range(len(regions)), pa.int64()), "region": regions})
    pq.write_to_dataset(table, path, partition_cols=["region"])
    return path


class TestConvert:
    def test_escaped_partition_values_are_refused_before_writing(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # delta-rs wrote "region=a%20b/..." as the add path, which every reader
        # decodes to "region=a b/...": a table no engine could read.
        path = _hive_dir(tmp_path, ["a b", "eu"])
        with pytest.raises(UnreachableTableError, match="escaped value"):
            conn.convert_to_delta(path, partition_by=pa.schema([("region", pa.string())]))
        assert not os.path.exists(os.path.join(path, "_delta_log"))

    @pytest.mark.parametrize(
        "partition_by",
        [
            pa.schema([("region", pa.string())]),
            [("region", "string")],
            [("region", pa.string())],
            [pa.field("region", pa.string())],
        ],
    )
    def test_partition_by_takes_pyarrow_and_pairs(
        self, conn: Any, tmp_path: Any, partition_by: Any
    ) -> None:
        path = _hive_dir(tmp_path, ["us", "eu", "eu"])
        t = conn.convert_to_delta(path, partition_by=partition_by)
        rows = t.to_arrow().sort_by("id").to_pylist()
        assert rows == [
            {"id": 0, "region": "us"},
            {"id": 1, "region": "eu"},
            {"id": 2, "region": "eu"},
        ]

    def test_partitioned_dir_without_partition_by_names_the_columns(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = _hive_dir(tmp_path, ["us"])
        with pytest.raises(InvalidArgumentError, match=r"\['region'\].*partition_by"):
            conn.convert_to_delta(path)

    def test_untyped_partition_names_are_refused(self, conn: Any, tmp_path: Any) -> None:
        path = _hive_dir(tmp_path, ["us"])
        with pytest.raises(InvalidArgumentError, match="no type"):
            conn.convert_to_delta(path, partition_by=["region"])


def _history_table(conn: Any, tmp_path: Any, commits: int = 3) -> tuple[Any, str]:
    path = str(tmp_path / "t")
    t = conn.create_table(path, pa.schema([("id", pa.int64())]))
    for i in range(commits):
        t.append(pa.table({"id": pa.array([i], pa.int64())}))
        time.sleep(0.02)
    return t, path


def _ids(t: Any) -> list[int]:
    return sorted(t.to_arrow().column("id").to_pylist())


class TestRestoreByTimestamp:
    def test_before_the_first_commit_is_refused(self, conn: Any, tmp_path: Any) -> None:
        # delta-rs resolved it to version 0 and restored the empty table.
        t, _ = _history_table(conn, tmp_path)
        with pytest.raises(InvalidArgumentError, match="no version of the table exists"):
            t.restore(dt.datetime(2000, 1, 1, tzinfo=dt.UTC))
        with pytest.raises(InvalidArgumentError, match="no version of the table exists"):
            t.restore("2000-01-01")
        assert _ids(t) == [0, 1, 2]
        assert t.version == 3

    def test_a_future_timestamp_is_refused(self, conn: Any, tmp_path: Any) -> None:
        # delta-rs raised "Version to restore 3 should be less then last available";
        # Spark refuses it as DELTA_TIMESTAMP_GREATER_THAN_COMMIT.
        t, _ = _history_table(conn, tmp_path)
        future = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
        with pytest.raises(InvalidArgumentError, match="after the latest commit"):
            t.restore(future)
        assert t.restore(t.version).items() >= {"numRemovedFile": 0, "numRestoredFile": 0}.items()
        assert t.version == 3

    def test_restores_what_a_read_at_the_timestamp_sees(self, conn: Any, tmp_path: Any) -> None:
        t, _ = _history_table(conn, tmp_path)
        stamp = next(h["timestamp"] for h in t.history() if h["version"] == 2)
        at = dt.datetime.fromtimestamp(stamp / 1000, tz=dt.UTC)
        expected = sorted(t.to_arrow(timestamp=at).column("id").to_pylist())
        t.restore(at)
        assert _ids(t) == expected == [0, 1]

    def test_bad_timestamp_text_is_an_argument_error(self, conn: Any, tmp_path: Any) -> None:
        t, _ = _history_table(conn, tmp_path, commits=1)
        with pytest.raises(InvalidArgumentError, match="ISO-8601"):
            t.restore("yesterday-ish")


class TestAddConstraintRace:
    def test_rows_appended_during_validation_are_checked(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        # delta-rs validated the snapshot it opened, then rebased its commit
        # over a racing append of -7 and committed "id > 0" regardless.
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": pa.array([1, 2, 3], pa.int64())}))
        t = conn.open_table(path)
        original = deltalake.table.TableAlterer.add_constraint
        raced: list[bool] = []

        def racing(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not raced:
                raced.append(True)
                deltalake.write_deltalake(
                    path, pa.table({"id": pa.array([-7], pa.int64())}), mode="append"
                )
            return original(self, *args, **kwargs)

        monkeypatch.setattr(deltalake.table.TableAlterer, "add_constraint", racing)
        with pytest.raises(Exception, match="failed validation"):
            t.add_constraint({"id_pos": "id > 0"})
        assert raced
        assert "delta.constraints.id_pos" not in deltalake.DeltaTable(path).metadata().configuration

    def test_an_uncontended_constraint_still_commits(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": pa.array([1], pa.int64())}))
        conn.open_table(path).add_constraint({"id_pos": "id > 0"}, commit_metadata={"a": "b"})
        dt_ = deltalake.DeltaTable(path)
        assert dt_.metadata().configuration["delta.constraints.id_pos"] == "id > 0"


class TestChangeFeedReservedNames:
    def _cdf_table(self, conn: Any, tmp_path: Any) -> Any:
        path = str(tmp_path / "cdf")
        deltalake.write_deltalake(
            path,
            pa.table({"id": pa.array([1], pa.int64())}),
            configuration={"delta.enableChangeDataFeed": "true"},
        )
        return conn.open_table(path)

    def test_enabling_it_through_delta_rs_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": [1], "_Change_Type": ["x"]}))
        t = conn.open_table(path)
        assert t.can("set_properties", properties={"delta.enableChangeDataFeed": "true"}).engine
        with pytest.raises(UnreachableTableError, match="reserves the column names"):
            t.set_properties({"delta.enableChangeDataFeed": "true"})
        assert "delta.enableChangeDataFeed" not in t.properties()

    def test_create_and_write_table_refuse_it(self, conn: Any, tmp_path: Any) -> None:
        cdf = {"delta.enableChangeDataFeed": "true"}
        schema = pa.schema([("id", pa.int64()), ("_commit_version", pa.int64())])
        with pytest.raises(UnreachableTableError, match="reserves the column names"):
            conn.create_table(str(tmp_path / "a"), schema, properties=cdf)
        with pytest.raises(UnreachableTableError, match="reserves the column names"):
            conn.write_table(
                str(tmp_path / "b"), pa.table({"id": [1], "_change_type": ["x"]}), properties=cdf
            )
        assert not os.path.exists(tmp_path / "a" / "_delta_log")

    def test_schema_evolving_writes_refuse_it(self, conn: Any, tmp_path: Any) -> None:
        t = self._cdf_table(conn, tmp_path)
        clash = pa.table({"id": pa.array([2], pa.int64()), "_change_type": ["x"]})
        with pytest.raises(UnreachableTableError, match="reserves the column names"):
            t.append(clash, schema_mode="merge")
        with pytest.raises(UnreachableTableError, match="reserves the column names"):
            t.overwrite(clash, schema_mode="overwrite")
        assert t.schema().names == ["id"]


class TestDeltaRsAlterValidation:
    """delta-rs stored what the kernel path refuses; both now refuse alike."""

    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": pa.array([1], pa.int64())}))
        return conn.open_table(path)

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("delta.targetFileSize", "abc", "size in bytes"),
            ("delta.isolationLevel", "snapshot", "Serializable"),
            ("delta.dataSkippingStatsColumns", "nosuch", "no column 'nosuch'"),
            ("delta.minWriterVersion", "5", "not set directly"),
        ],
    )
    def test_invalid_property_values(self, table: Any, key: str, value: str, match: str) -> None:
        assert table.can("set_properties", properties={key: value}).engine is not None
        # A stats column the table lacks is the caller's mistake
        # (InvalidArgumentError); the other values are refused properties.
        with pytest.raises((UnreachableTableError, InvalidArgumentError), match=match):
            table.set_properties({key: value})
        assert key not in table.properties()

    def test_downgrading_a_writer_7_protocol(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t7")
        deltalake.write_deltalake(
            path,
            pa.table({"id": [1]}),
            configuration={"delta.enableChangeDataFeed": "true"},
        )
        t = conn.open_table(path)
        t.add_feature("appendOnly")
        before = t.protocol()
        with pytest.raises(UnreachableTableError, match="not set directly"):
            t.set_properties({"delta.minWriterVersion": "2"})
        assert t.protocol() == before

    @pytest.mark.parametrize("name", ["a b", "x=y", "p;q"])
    def test_column_names_parquet_cannot_hold(self, table: Any, name: str) -> None:
        assert str(table.can("add_column").engine) == "deltars"
        with pytest.raises(UnreachableTableError, match="Parquet does not allow"):
            table.add_column(pa.field(name, pa.int64()))
        assert table.schema().names == ["id"]

    def test_not_null_column_is_refused_before_any_engine(
        self, table: Any, monkeypatch: Any
    ) -> None:
        # The warehouse's ADD COLUMNS was sent the column without NOT NULL.
        from deltaswamp.engine.deltars import DeltaRsEngine

        calls: list[Any] = []
        monkeypatch.setattr(DeltaRsEngine, "add_columns", lambda *a, **k: calls.append(a))
        with pytest.raises(UnreachableTableError, match="must be nullable"):
            table.add_column(pa.field("n", pa.int64(), nullable=False))
        with pytest.raises(UnreachableTableError, match="must be nullable"):
            table.add_column({"n": "INT NOT NULL"})
        assert calls == []


def _cm_table(conn: Any, tmp_path: Any) -> Any:
    path = str(tmp_path / "cm")
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("s", pa.struct([("a", pa.int64()), ("b", pa.int64())])),
        ]
    )
    t = conn.create_table(path, schema, properties={"delta.columnMapping.mode": "name"})
    t.append(pa.table({"id": pa.array([1], pa.int64()), "s": [{"a": 1, "b": 2}]}, schema=schema))
    return t


class TestColumnPaths:
    def test_drop_column_takes_a_nested_path_list(self, conn: Any, tmp_path: Any) -> None:
        t = _cm_table(conn, tmp_path)
        t.drop_column(["s", "b"])
        assert t.schema().field("s").type == pa.struct([("a", pa.int64())])

    def test_rename_new_name_is_a_leaf(self, conn: Any, tmp_path: Any) -> None:
        t = _cm_table(conn, tmp_path)
        t.rename_column(["s", "a"], "s.x")
        assert [f.name for f in t.schema().field("s").type] == ["x", "b"]

    def test_case_only_rename_is_refused(self, conn: Any, tmp_path: Any) -> None:
        # Databricks refuses RENAME COLUMN id TO ID; the kernel path committed it.
        t = _cm_table(conn, tmp_path)
        with pytest.raises(UnreachableTableError, match="differs only in case"):
            t.rename_column("id", "ID")
        assert t.schema().names == ["id", "s"]

    def test_set_not_null_on_a_nested_field_says_why(self, conn: Any, tmp_path: Any) -> None:
        t = _cm_table(conn, tmp_path)
        with pytest.raises(UnreachableTableError, match="nested field"):
            t.set_not_null("s.a")
        with pytest.raises(InvalidArgumentError, match="no column 'nosuch'"):
            t.set_not_null("nosuch")


class TestWidenPartitionColumn:
    @pytest.mark.parametrize(
        ("old", "new", "values"),
        [
            (pa.int32(), "long", [1, -7]),
            (pa.int8(), "integer", [3, 4]),
            (pa.float32(), "double", [1.5, 2.25]),
        ],
    )
    def test_numeric_partition_widens(
        self, conn: Any, tmp_path: Any, old: Any, new: str, values: list[Any]
    ) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("p", old)]),
            partition_by=["p"],
            properties={"delta.enableTypeWidening": "true"},
        )
        t.append(pa.table({"id": pa.array([1, 2], pa.int64()), "p": pa.array(values, old)}))
        t.alter_column_type("p", new)
        rows = conn.open_table(path).to_arrow().sort_by("id").column("p").to_pylist()
        assert rows == values

    def test_decimal_scale_change_on_a_partition_is_refused(self, conn: Any, tmp_path: Any) -> None:
        # The kernel then could not parse the stored "1.50" as decimal(7,3).
        import decimal

        path = str(tmp_path / "t")
        t = conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("p", pa.decimal128(5, 2))]),
            partition_by=["p"],
            properties={"delta.enableTypeWidening": "true"},
        )
        t.append(
            pa.table({"id": [1], "p": pa.array([decimal.Decimal("1.50")], pa.decimal128(5, 2))})
        )
        with pytest.raises(UnreachableTableError, match="partition column"):
            t.alter_column_type("p", "decimal(7,3)")
        assert conn.open_table(path).to_arrow().num_rows == 1


class TestRestoreColumnMapping:
    def test_restore_across_column_mapping_changes_is_refused(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": pa.array([1], pa.int64()), "v": ["a"]}))
        t = conn.open_table(path)
        t.set_properties({"delta.columnMapping.mode": "name"})  # v1
        t.append(pa.table({"id": pa.array([2], pa.int64()), "v": ["b"]}))  # v2
        t.drop_column("v")  # v3
        t.add_column(pa.field("v", pa.string()))  # v4: maxColumnId 3
        for target in (0, 2):
            with pytest.raises(UnreachableTableError, match="column-mapping metadata changed"):
                t.restore(target)
        assert t.version == 4
        assert t.properties()["delta.columnMapping.maxColumnId"] == "3"


class TestLogUpkeep:
    def test_compact_logs_after_cleanup_starts_at_the_oldest_commit(
        self, conn: Any, tmp_path: Any
    ) -> None:
        t, path = _history_table(conn, tmp_path, commits=4)
        t.checkpoint()
        for i in range(3):
            t.append(pa.table({"id": pa.array([100 + i], pa.int64())}))
        old = time.time() - 40 * 86400
        for f in (tmp_path / "t" / "_delta_log").iterdir():
            os.utime(f, (old, old))
        t.cleanup_metadata()
        assert not (tmp_path / "t" / "_delta_log" / ("0" * 20 + ".json")).exists()
        t.compact_logs()
        with pytest.raises(InvalidArgumentError, match="oldest commit still there"):
            t.compact_logs(start=0)
        assert _ids(conn.open_table(path)) == [0, 1, 2, 3, 100, 101, 102]

    @pytest.mark.parametrize("call", ["checkpoint", "compact_logs", "cleanup_metadata"])
    def test_log_upkeep_forgets_cached_state(self, conn: Any, tmp_path: Any, call: str) -> None:
        t, _ = _history_table(conn, tmp_path)
        t.schema()
        assert t._enriched
        getattr(t, call)()
        assert not t._enriched

    def test_vacuum_below_retention_names_the_override(self, conn: Any, tmp_path: Any) -> None:
        t, _ = _history_table(conn, tmp_path, commits=1)
        with pytest.raises(InvalidArgumentError, match="enforce_retention_duration=False"):
            t.vacuum(retention_hours=0)
        assert t.vacuum(retention_hours=0, enforce_retention_duration=False) == []

    def test_drop_not_null_on_a_missing_column(self, conn: Any, tmp_path: Any) -> None:
        t, _ = _history_table(conn, tmp_path, commits=1)
        with pytest.raises(InvalidArgumentError, match="no column 'nosuch'"):
            t.drop_not_null("nosuch")

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("delta.compatibility.symlinkFormatManifest.enabled", "true"),
            ("delta.checkpointRetentionDuration", "interval 2 days"),
        ],
    )
    def test_real_delta_keys_are_accepted(
        self, conn: Any, tmp_path: Any, key: str, value: str
    ) -> None:
        t, _ = _history_table(conn, tmp_path, commits=1)
        t.set_properties({key: value})
        assert t.properties()[key] == value


class TestUnknownOptions:
    @pytest.fixture
    def table(self, conn: Any, tmp_path: Any) -> Any:
        return _history_table(conn, tmp_path, commits=1)[0]

    @pytest.mark.parametrize(
        ("call", "args", "kwargs"),
        [
            ("set_properties", ({"a.b": "1"},), {"foo": 1}),
            ("add_column", (pa.field("z", pa.int64()),), {"foo": 1}),
            ("add_feature", ("appendOnly",), {"foo": 1}),
            ("add_constraint", ({"c": "id > -1"},), {"foo": 1}),
            ("vacuum", (), {"retain_hours": 0}),
            ("optimize", (), {"zorder": ["id"]}),
            ("restore", (0,), {"ignore_missing_file": True}),
            ("repair", (), {"foo": 1}),
            ("replace", (pa.table({"id": pa.array([1], pa.int64())}),), {"foo": 1}),
        ],
    )
    def test_misspelt_options_are_refused(
        self, table: Any, call: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        version = table.version
        with pytest.raises(InvalidArgumentError, match="unexpected option"):
            getattr(table, call)(*args, **kwargs)
        assert table.version == version

    def test_documented_pass_through_options_still_work(self, table: Any) -> None:
        table.set_properties({"x.y": "1"}, raise_if_not_exists=False)
        table.add_feature("appendOnly", allow_protocol_versions_increase=True)
        assert table.vacuum(retention_hours=0, enforce_retention_duration=False) == []

    def test_can_refuses_misspelt_shape_keys(self, table: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="schema_mod"):
            table.can("append", schema_mod="merge")
        with pytest.raises(InvalidArgumentError, match="timestmap"):
            table.can("scan", timestmap=1)
        assert table.can("append", schema_mode="merge").operation.value == "merge_schema"
