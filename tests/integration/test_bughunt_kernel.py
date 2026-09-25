"""Regression tests for defects found auditing the kernel engine and distributed reads."""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, cast

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from deltaswamp.capability import Operation  # noqa: E402
from deltaswamp.catalog.base import ResolvedTable  # noqa: E402
from deltaswamp.credentials.base import CredentialProvider  # noqa: E402
from deltaswamp.engine.kernel import KernelEngine  # noqa: E402
from deltaswamp.errors import UnreachableTableError  # noqa: E402
from deltaswamp.identity import parse_ref  # noqa: E402


def resolved(path: str, *, bare: bool = False, **overrides: Any) -> ResolvedTable:
    """A ResolvedTable for `path`, with protocol and properties read from the log.

    `bare=True` leaves them empty, as a path resolution does before enrichment.
    """
    fields: dict[str, Any] = {}
    if not bare:
        snap = KernelEngine().snapshot(ResolvedTable(ref=parse_ref(path), location=path))
        min_reader, min_writer, readers, writers = snap.protocol()
        fields = {
            "min_reader_version": min_reader,
            "min_writer_version": min_writer,
            "reader_features": frozenset(readers),
            "writer_features": frozenset(writers),
            "properties": dict(snap.table_properties()),
            "partition_columns": tuple(snap.partition_columns),
        }
    fields.update(overrides)
    return ResolvedTable(ref=parse_ref(path), location=path, **fields)


def write(path: str, data: Any, **kwargs: Any) -> str:
    from deltalake import write_deltalake

    write_deltalake(path, data, **kwargs)
    return path


def kernel_table(path: str, schema: Any, properties: dict[str, str] | None = None) -> str:
    from deltaswamp import _native

    _native.create_table(path, schema, properties=properties)
    return path


@pytest.fixture
def mixed(tmp_path: Any) -> str:
    return write(
        str(tmp_path / "mixed"),
        pa.table(
            {
                "id": [1, 2, 3],
                "Name": ["a", "b", "c"],
                "s": [{"X": 1}, {"X": 2}, {"X": 3}],
                "part": ["x", "y", "x"],
            }
        ),
        partition_by=["part"],
    )


def read(stream: Any) -> Any:
    return pa.table(stream)


class TestProjectionAndPredicates:
    def test_projection_is_case_insensitive(self, mixed: str) -> None:
        got = read(KernelEngine().scan(resolved(mixed), columns=["ID"]))
        assert got.column_names == ["id"]
        assert sorted(got.column("id").to_pylist()) == [1, 2, 3]

    def test_predicate_column_is_case_insensitive(self, mixed: str) -> None:
        got = read(KernelEngine().scan(resolved(mixed), columns=["id"], predicate="name = 'b'"))
        assert got.to_pydict() == {"id": [2]}

    def test_nested_predicate_field_is_case_insensitive(self, mixed: str) -> None:
        got = read(KernelEngine().scan(resolved(mixed), predicate="s.x = 2"))
        assert got.column("id").to_pylist() == [2]

    def test_empty_projection_keeps_row_count(self, mixed: str) -> None:
        # Used to abort the interpreter: a non-unwinding panic in the reader.
        got = read(KernelEngine().scan(resolved(mixed), columns=[]))
        assert got.num_rows == 3
        assert got.column_names == []

    def test_empty_projection_with_predicate_has_no_columns(self, mixed: str) -> None:
        got = read(KernelEngine().scan(resolved(mixed), columns=[], predicate="id > 1"))
        assert got.num_rows == 2
        assert got.column_names == []

    def test_duplicate_projection_is_read_once(self, mixed: str) -> None:
        got = read(KernelEngine().scan(resolved(mixed), columns=["id", "ID"]))
        assert got.column_names == ["id"]

    def test_execute_scan_matches_predicate_columns_case_insensitively(self, mixed: str) -> None:
        engine = KernelEngine()
        table = resolved(mixed)
        splits = engine.plan_scan(table)
        got = read(engine.execute_scan(table, splits, columns=["ID"], predicate="id > 1"))
        assert got.column_names == ["id"]
        assert sorted(got.column("id").to_pylist()) == [2, 3]

    def test_rewrite_predicate_is_case_insensitive(self, mixed: str) -> None:
        result = KernelEngine().delete(resolved(mixed), "NAME = 'b'")
        assert result["num_deleted_rows"] == 1


@pytest.fixture
def cdf_table(tmp_path: Any) -> str:
    from deltalake import DeltaTable

    path = write(
        str(tmp_path / "cdf"),
        pa.table({"id": [1, 2, 3], "V": ["a", "b", "c"]}),
        configuration={"delta.enableChangeDataFeed": "true"},
    )
    DeltaTable(path).delete("id = 2")
    return path


class TestChangeFeed:
    def test_unenriched_table_reads_the_feed(self, cdf_table: str) -> None:
        # A path resolution carries no properties; that used to read as
        # "change data feed disabled" and refuse every such table.
        got = read(KernelEngine().cdf(resolved(cdf_table, bare=True)))
        assert got.num_rows == 4

    def test_metadata_column_is_not_duplicated(self, cdf_table: str) -> None:
        got = read(
            KernelEngine().cdf(
                resolved(cdf_table), columns=["id", "_change_type"], predicate="id > 1"
            )
        )
        assert got.column_names == ["id", "_change_type", "_commit_version", "_commit_timestamp"]

    def test_empty_projection_keeps_only_metadata(self, cdf_table: str) -> None:
        got = read(KernelEngine().cdf(resolved(cdf_table), columns=[]))
        assert got.num_rows == 4
        assert got.column_names == ["_change_type", "_commit_version", "_commit_timestamp"]

    def test_projection_and_predicate_are_case_insensitive(self, cdf_table: str) -> None:
        got = read(KernelEngine().cdf(resolved(cdf_table), columns=["v"], predicate="ID = 2"))
        assert got.column("V").to_pylist() == ["b", "b"]

    def test_zero_valued_unsupported_option_is_refused(self, cdf_table: str) -> None:
        with pytest.raises(UnreachableTableError, match="allow_out_of_range"):
            KernelEngine().cdf(resolved(cdf_table), allow_out_of_range=0)

    def test_ending_version_and_timestamp_together_are_refused(self, cdf_table: str) -> None:
        with pytest.raises(UnreachableTableError, match="not both"):
            KernelEngine().cdf(resolved(cdf_table), ending_version=1, ending_timestamp=0)


class TestSupports:
    def test_cdf_on_catalog_managed_is_refused_up_front(self, cdf_table: str) -> None:
        table = resolved(cdf_table, writer_features=frozenset({"catalogManaged"}))
        verdict = KernelEngine().supports(Operation.CDF, table)
        assert not verdict.ok
        assert "catalog" in verdict.reason

    def test_time_travel_on_a_shallow_clone_is_refused(self, mixed: str) -> None:
        table = resolved(mixed, securable_kind="TABLE_DELTA_SHALLOW_CLONE")
        assert not KernelEngine().supports(Operation.TIME_TRAVEL, table).ok

    def test_partitioned_rewrite_needs_partitioned_append(
        self, mixed: str, monkeypatch: Any
    ) -> None:
        from deltaswamp.engine import kernel

        real = kernel._native_has
        monkeypatch.setattr(
            kernel, "_native_has", lambda *f: "partitioned_append" not in f and real(*f)
        )
        verdict = KernelEngine().supports(Operation.DELETE, resolved(mixed))
        assert not verdict.ok
        assert "unpartitioned" in verdict.reason

    @pytest.mark.parametrize("writer", [3, 4, 6])
    def test_legacy_writer_protocol_refuses_data_writes(self, tmp_path: Any, writer: int) -> None:
        path = write(
            str(tmp_path / "legacy"),
            pa.table({"id": [1]}),
            configuration={"delta.minWriterVersion": str(writer)},
        )
        verdict = KernelEngine().supports(Operation.APPEND, resolved(path))
        assert not verdict.ok
        assert "checkConstraints" in verdict.reason

    @pytest.mark.parametrize(
        "op", [Operation.DELETE, Operation.UPDATE, Operation.OVERWRITE, Operation.REPLACE_WHERE]
    )
    def test_cdf_enabled_refuses_removing_writes(self, tmp_path: Any, op: Operation) -> None:
        path = kernel_table(
            str(tmp_path / "t"),
            pa.schema([("id", pa.int64())]),
            {"delta.enableChangeDataFeed": "true"},
        )
        engine = KernelEngine()
        assert engine.supports(Operation.APPEND, resolved(path)).ok
        verdict = engine.supports(op, resolved(path))
        assert not verdict.ok
        assert "change data feed" in verdict.reason

    def test_append_only_refuses_removing_writes(self, tmp_path: Any) -> None:
        path = kernel_table(
            str(tmp_path / "t"), pa.schema([("id", pa.int64())]), {"delta.appendOnly": "true"}
        )
        engine = KernelEngine()
        assert engine.supports(Operation.APPEND, resolved(path)).ok
        verdict = engine.supports(Operation.DELETE, resolved(path))
        assert not verdict.ok
        assert "append-only" in verdict.reason

    def test_schema_invariants_refuse_writes(self, tmp_path: Any) -> None:
        field = pa.field(
            "id",
            pa.int64(),
            metadata={"delta.invariants": json.dumps({"expression": {"expression": "id > 0"}})},
        )
        path = write(str(tmp_path / "inv"), pa.table({"id": [1]}, schema=pa.schema([field])))
        verdict = KernelEngine().supports(Operation.APPEND, resolved(path))
        assert not verdict.ok
        assert "invariants" in verdict.reason


class _FakeCredentials:
    expires_at = None

    def expires_within(self, _: float) -> bool:
        return False

    def as_storage_options(self) -> dict[str, str]:
        return {"vended": "yes"}


class _FakeProvider:
    def credentials(self, _: Any) -> _FakeCredentials:
        return _FakeCredentials()


class TestCreate:
    def test_create_uses_vended_credentials(self, tmp_path: Any, monkeypatch: Any) -> None:
        from deltaswamp import _native

        seen: dict[str, Any] = {}

        def fake_create(root: str, schema: Any, **kwargs: Any) -> int:
            seen.update(kwargs)
            return 0

        monkeypatch.setattr(_native, "create_table", fake_create)
        path = str(tmp_path / "new")
        table = ResolvedTable(
            ref=parse_ref(path),
            location=path,
            credential_provider=cast(CredentialProvider, _FakeProvider()),
        )
        KernelEngine(storage_options={"base": "1"}).create(table, pa.schema([("id", pa.int64())]))
        assert seen["options"] == {"base": "1", "vended": "yes"}

    def test_description_is_committed(self, tmp_path: Any) -> None:
        path = str(tmp_path / "new")
        table = ResolvedTable(ref=parse_ref(path), location=path)
        engine = KernelEngine()
        engine.create(table, pa.schema([("id", pa.int64())]), description="hello")
        meta = json.loads(engine.snapshot(table).metadata_json())
        assert meta["description"] == "hello"

    def test_unknown_create_option_is_refused(self, tmp_path: Any) -> None:
        path = str(tmp_path / "new")
        table = ResolvedTable(ref=parse_ref(path), location=path)
        with pytest.raises(UnreachableTableError, match="bogus"):
            KernelEngine().create(table, pa.schema([("id", pa.int64())]), bogus=1)
        assert not os.path.exists(os.path.join(path, "_delta_log"))


@pytest.fixture
def plain(tmp_path: Any) -> str:
    return write(str(tmp_path / "plain"), pa.table({"id": [1, 2], "city": ["oslo", "lima"]}))


def last_commit(path: str) -> list[dict[str, Any]]:
    log = os.path.join(path, "_delta_log")
    name = sorted(f for f in os.listdir(log) if f.endswith(".json"))[-1]
    with open(os.path.join(log, name)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestPredicateOverwrite:
    def test_txn_and_commit_metadata_are_committed(self, plain: str) -> None:
        KernelEngine().overwrite(
            resolved(plain),
            pa.table({"id": [1], "city": ["bergen"]}),
            predicate="id = 1",
            txn=("loader", 7),
            commit_metadata={"run": 3},
        )
        actions = last_commit(plain)
        txn = next(a["txn"] for a in actions if "txn" in a)
        assert (txn["appId"], txn["version"]) == ("loader", 7)
        info = next(a["commitInfo"] for a in actions if "commitInfo" in a)
        assert "3" in json.dumps(info)

    def test_extra_columns_are_refused_not_dropped(self, plain: str) -> None:
        with pytest.raises(UnreachableTableError, match="extra"):
            KernelEngine().overwrite(
                resolved(plain),
                pa.table({"id": [1], "city": ["x"], "extra": [5]}),
                predicate="id = 1",
            )

    def test_columns_match_case_insensitively_and_missing_nullable_is_null(
        self, plain: str
    ) -> None:
        KernelEngine().overwrite(resolved(plain), pa.table({"ID": [1]}), predicate="id = 1")
        got = sorted(read(KernelEngine().scan(resolved(plain))).to_pylist(), key=lambda r: r["id"])
        assert got == [{"id": 1, "city": None}, {"id": 2, "city": "lima"}]

    def test_unsupported_option_is_refused(self, plain: str) -> None:
        with pytest.raises(UnreachableTableError, match="schema_mode"):
            KernelEngine().overwrite(
                resolved(plain),
                pa.table({"id": [1], "city": ["x"]}),
                predicate="id = 1",
                schema_mode="overwrite",
            )


class TestUpdate:
    def test_column_assignments_read_the_old_row(self, tmp_path: Any) -> None:
        path = write(str(tmp_path / "t"), pa.table({"a": [1, 2], "b": [10, 20]}))
        KernelEngine().update(resolved(path), updates={"a": "b", "b": "a"})
        got = sorted(read(KernelEngine().scan(resolved(path))).to_pylist(), key=lambda r: r["a"])
        assert got == [{"a": 10, "b": 1}, {"a": 20, "b": 2}]

    def test_column_names_are_case_insensitive(self, plain: str) -> None:
        result = KernelEngine().update(
            resolved(plain), new_values={"CITY": "rome"}, predicate="id = 2"
        )
        assert result["num_updated_rows"] == 1
        rows = read(KernelEngine().scan(resolved(plain))).to_pylist()
        assert {r["id"]: r["city"] for r in rows} == {1: "oslo", 2: "rome"}


class TestTimeTravel:
    def test_pre_epoch_timestamp_rounds_down(self) -> None:
        from deltaswamp.engine.kernel import _timestamp_ms

        moment = dt.datetime(1969, 12, 31, 23, 59, 59, 999500, tzinfo=dt.UTC)
        assert _timestamp_ms(moment) == -1

    def test_version_past_latest_is_a_clear_error(self, plain: str) -> None:
        with pytest.raises(UnreachableTableError, match="no such version"):
            KernelEngine().snapshot(resolved(plain), version=5)

    def test_negative_version_is_a_clear_error(self, plain: str) -> None:
        with pytest.raises(UnreachableTableError, match="start at 0"):
            KernelEngine().snapshot(resolved(plain), version=-1)


class TestCheckpointInterval:
    def test_interval_comes_from_the_log_when_unenriched(self, tmp_path: Any) -> None:
        path = kernel_table(
            str(tmp_path / "t"), pa.schema([("id", pa.int64())]), {"delta.checkpointInterval": "2"}
        )
        engine = KernelEngine()
        for i in range(2):
            engine.append(resolved(path, bare=True), pa.table({"id": [i]}))
        log = os.listdir(os.path.join(path, "_delta_log"))
        assert any("checkpoint" in name for name in log)


class TestSetNotNull:
    def test_null_check_matches_the_column_case_insensitively(self, plain: str) -> None:
        engine = KernelEngine()
        engine.set_not_null(resolved(plain), "ID")
        schema = json.loads(
            json.loads(engine.snapshot(resolved(plain)).metadata_json())["schemaString"]
        )
        assert schema["fields"][0]["nullable"] is False


class TestAddColumns:
    @pytest.mark.parametrize("dtype", [pa.int64(), "bigint", "BIGINT", "long"])
    def test_mapping_types_are_delta_types(self, plain: str, dtype: Any) -> None:
        # `str(dtype)` used to commit `int64` / `bigint` verbatim, leaving the
        # table unreadable by every engine.
        engine = KernelEngine()
        engine.add_columns(resolved(plain), {"n": dtype})
        schema = json.loads(
            json.loads(engine.snapshot(resolved(plain)).metadata_json())["schemaString"]
        )
        assert schema["fields"][-1]["type"] == "long"

    def test_unknown_type_is_refused_before_commit(self, plain: str) -> None:
        engine = KernelEngine()
        with pytest.raises(UnreachableTableError, match="not a Delta type"):
            engine.add_columns(resolved(plain), {"n": "nope"})
        assert engine.snapshot(resolved(plain)).version == 0

    def test_deltalake_schema_contributes_its_fields(self, plain: str) -> None:
        from deltalake import Field, Schema
        from deltalake.schema import PrimitiveType

        engine = KernelEngine()
        engine.add_columns(resolved(plain), Schema([Field("y", PrimitiveType("long"))]))
        schema = json.loads(
            json.loads(engine.snapshot(resolved(plain)).metadata_json())["schemaString"]
        )
        assert [f["name"] for f in schema["fields"]] == ["id", "city", "y"]


class TestDatasource:
    """Read tasks run locally: no Ray cluster needed."""

    def plan(self, path: str, **kwargs: Any) -> Any:
        from deltaswamp.catalog.filesystem import FilesystemCatalog

        return ds.connect(catalog=FilesystemCatalog()).open_table(path).plan_scan(**kwargs)

    def test_empty_table_still_has_a_schema(self, tmp_path: Any) -> None:
        pytest.importorskip("ray")
        from deltaswamp.distributed import DeltaSwampDatasource

        path = write(str(tmp_path / "e"), pa.table({"id": pa.array([], pa.int64())}))
        tasks = DeltaSwampDatasource(self.plan(path)).get_read_tasks(4)
        assert len(tasks) == 1
        blocks = list(tasks[0]())
        assert blocks[0].schema.names == ["id"]

    def test_each_task_carries_only_its_splits(self, tmp_path: Any) -> None:
        pytest.importorskip("ray")
        import pickle

        from deltaswamp.distributed import DeltaSwampDatasource
        from ray import cloudpickle

        path = str(tmp_path / "t")
        for i in range(6):
            write(path, pa.table({"id": [i]}), mode="append")
        tasks = DeltaSwampDatasource(self.plan(path)).get_read_tasks(3)
        assert len(tasks) == 3
        assert all(len(t.read_fn.__defaults__[0].splits) == 2 for t in tasks)
        rows = sorted(
            r
            for t in pickle.loads(cloudpickle.dumps(tasks))  # type: ignore[no-untyped-call]
            for b in t()
            for r in b["id"].to_pylist()
        )
        assert rows == list(range(6))

    def test_task_streams_batches_and_honours_the_row_limit(self, tmp_path: Any) -> None:
        pytest.importorskip("ray")
        from deltaswamp.distributed import DeltaSwampDatasource

        path = str(tmp_path / "t")
        for i in range(4):
            write(path, pa.table({"id": [i, i + 10]}), mode="append")
        (task,) = DeltaSwampDatasource(self.plan(path)).get_read_tasks(1)
        assert len(list(task())) == 4  # one block per file, not one per task
        (limited,) = DeltaSwampDatasource(self.plan(path)).get_read_tasks(1, per_task_row_limit=3)
        assert sum(b.num_rows for b in limited()) == 3


class TestPartitionOnlyProjection:
    def test_scan_of_only_partition_columns(self, mixed: str) -> None:
        # The native reader panics when no data-file column is projected.
        got = read(KernelEngine().scan(resolved(mixed), columns=["part"]))
        assert got.column_names == ["part"]
        assert sorted(got.column("part").to_pylist()) == ["x", "x", "y"]

    def test_execute_scan_of_only_partition_columns(self, mixed: str) -> None:
        engine = KernelEngine()
        table = resolved(mixed)
        got = read(engine.execute_scan(table, engine.plan_scan(table), columns=["PART"]))
        assert got.column_names == ["part"]
        assert got.num_rows == 3


class TestExecuteScan:
    def test_mixed_versions_with_an_unversioned_split_is_a_clear_error(self, mixed: str) -> None:
        import dataclasses

        engine = KernelEngine()
        table = resolved(mixed)
        first, second = engine.plan_scan(table)
        splits = [first, dataclasses.replace(second, commit_version=None)]
        with pytest.raises(UnreachableTableError, match="different versions"):
            engine.execute_scan(table, splits)


class TestColumnMappedSplits:
    def test_split_partition_values_use_logical_names(self, tmp_path: Any) -> None:
        path = str(tmp_path / "cm")
        from deltaswamp import _native

        _native.create_table(
            path,
            pa.schema([("id", pa.int64()), ("Part", pa.string())]),
            properties={"delta.columnMapping.mode": "name"},
            partition_by=["Part"],
        )
        engine = KernelEngine()
        table = resolved(path)
        engine.append(table, pa.table({"id": [1, 2], "Part": ["a", "b"]}))
        splits = engine.plan_scan(resolved(path))
        assert sorted(s.partition_values["Part"] for s in splits) == ["a", "b"]


class TestAppendRetry:
    def test_blind_append_is_restaged_after_losing_the_race(
        self, plain: str, monkeypatch: Any
    ) -> None:
        from deltalake import write_deltalake

        engine = KernelEngine()
        real = engine.snapshot
        raced: list[bool] = []

        def racing_snapshot(table: Any, **kwargs: Any) -> Any:
            snap = real(table, **kwargs)
            if not raced:  # another writer commits between resolve and commit
                raced.append(True)
                write_deltalake(plain, pa.table({"id": [7], "city": ["x"]}), mode="append")
            return snap

        monkeypatch.setattr(engine, "snapshot", racing_snapshot)
        version = engine.append(resolved(plain), pa.table({"id": [3], "city": ["cairo"]}))
        assert version == 2
        ids = sorted(read(KernelEngine().scan(resolved(plain))).column("id").to_pylist())
        assert ids == [1, 2, 3, 7]

    def test_overwrite_conflict_is_not_retried(self, plain: str, monkeypatch: Any) -> None:
        from deltalake import write_deltalake
        from deltaswamp import _native

        engine = KernelEngine()
        real = engine.snapshot

        def racing_snapshot(table: Any, **kwargs: Any) -> Any:
            snap = real(table, **kwargs)
            write_deltalake(plain, pa.table({"id": [7], "city": ["x"]}), mode="append")
            return snap

        monkeypatch.setattr(engine, "snapshot", racing_snapshot)
        with pytest.raises(_native.CommitConflictError):
            engine.overwrite(resolved(plain), pa.table({"id": [3], "city": ["cairo"]}))


class TestReplaceWhereContract:
    def test_rows_outside_the_predicate_are_refused(self, plain: str) -> None:
        engine = KernelEngine()
        with pytest.raises(UnreachableTableError, match="do not satisfy the predicate"):
            engine.overwrite(
                resolved(plain), pa.table({"id": [9], "city": ["z"]}), predicate="id = 1"
            )
        assert engine.snapshot(resolved(plain)).version == 0

    def test_null_predicate_result_is_refused(self, plain: str) -> None:
        with pytest.raises(UnreachableTableError, match="do not satisfy"):
            KernelEngine().overwrite(
                resolved(plain), pa.table({"id": [None], "city": ["z"]}), predicate="id = 1"
            )


class TestAutomaticCheckpoint:
    def test_checkpoint_lands_on_the_committed_version(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = kernel_table(
            str(tmp_path / "t"), pa.schema([("id", pa.int64())]), {"delta.checkpointInterval": "2"}
        )
        engine = KernelEngine()
        engine.append(resolved(path), pa.table({"id": [0]}))
        engine.append(resolved(path), pa.table({"id": [1]}))
        log = os.listdir(os.path.join(path, "_delta_log"))
        assert any(n.startswith("00000000000000000002.checkpoint") for n in log)

    def test_catalog_managed_commit_is_not_auto_checkpointed(self, monkeypatch: Any) -> None:
        # The just-ratified commit is unpublished and absent from the stale
        # tail; checkpointing from here used to publish and checkpoint the
        # previous version.
        engine = KernelEngine()
        calls: list[Any] = []
        monkeypatch.setattr(engine, "checkpoint", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(engine, "snapshot", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(engine, "publish", lambda *a, **k: calls.append(a))
        table = ResolvedTable(
            ref=parse_ref("/tmp/cm"),
            location="/tmp/cm",
            writer_features=frozenset({"catalogManaged"}),
            reader_features=frozenset({"catalogManaged"}),
        )
        engine._maybe_checkpoint(table, 10)
        assert calls == []


class TestChangeFeedCase:
    def test_projected_column_in_predicate_is_not_duplicated(self, cdf_table: str) -> None:
        got = read(KernelEngine().cdf(resolved(cdf_table), columns=["id"], predicate="ID > 1"))
        assert got.column_names == ["id", "_change_type", "_commit_version", "_commit_timestamp"]
        assert sorted(got.column("id").to_pylist()) == [2, 2, 3]


class TestTxnVersion:
    def test_kernel_reads_the_committed_txn(self, plain: str) -> None:
        engine = KernelEngine()
        if engine.txn_version is None:
            pytest.skip("native build lacks app_id_version")
        assert engine.txn_version(resolved(plain), "app") is None
        engine.append(resolved(plain), pa.table({"id": [3], "city": ["x"]}), txn=("app", 7))
        assert engine.txn_version(resolved(plain), "app") == 7

    def test_repeated_txn_append_is_deduplicated_on_a_kernel_only_table(
        self, tmp_path: Any
    ) -> None:
        from deltaswamp.catalog.ossuc import OSSUnityCatalog

        from tests.fake_uc import FakeUnityCatalog

        if KernelEngine().txn_version is None:
            pytest.skip("native build lacks app_id_version")
        with FakeUnityCatalog(staging_root=tmp_path / "m") as uc:
            catalog = OSSUnityCatalog(uc.url)
            catalog.create_catalog("main")
            catalog.create_schema("main", "s")
            conn = ds.connect(catalog=catalog)
            conn.create_table("main.s.t", pa.schema([("id", pa.int64())]))
            for _ in range(2):
                conn.table("main.s.t").append(pa.table({"id": [1]}), txn=("app", 7))
            assert conn.table("main.s.t").to_arrow().num_rows == 1

    def test_missing_binding_makes_the_method_absent(self, monkeypatch: Any) -> None:
        from deltaswamp.engine import kernel

        monkeypatch.setattr(kernel, "_native_has", lambda *f: "app_id_version" not in f)
        assert not callable(getattr(KernelEngine(), "txn_version", None))


class TestReviewFollowUps:
    def test_numpy_integer_version_still_time_travels(self, plain: str) -> None:
        # [regression] an isinstance(int) check refused numpy integers (a
        # version taken from a history DataFrame), which the binding accepted.
        np = pytest.importorskip("numpy")
        snap = KernelEngine().snapshot(resolved(plain), version=np.int64(0))
        assert int(snap.version) == 0
        with pytest.raises(UnreachableTableError):
            KernelEngine().snapshot(resolved(plain), version=1.5)  # type: ignore[arg-type]
        with pytest.raises(UnreachableTableError):
            KernelEngine().snapshot(resolved(plain), version=True)

    def test_restaged_append_does_not_repeat_a_txn_the_race_winner_committed(
        self, plain: str, monkeypatch: Any
    ) -> None:
        # [incomplete] re-staging after a lost race ignored txn: if the winner
        # was a replay of the same batch (same app id and version), the retry
        # appended it a second time.
        from deltalake import CommitProperties, Transaction, write_deltalake
        from deltaswamp import _native

        if not _native_has_app_id_version():
            pytest.skip("build lacks app_id_version")
        engine = KernelEngine()
        real = engine.snapshot
        raced: list[bool] = []

        def racing_snapshot(table: Any, **kwargs: Any) -> Any:
            snap = real(table, **kwargs)
            if not raced:
                raced.append(True)
                write_deltalake(
                    plain,
                    pa.table({"id": [3], "city": ["cairo"]}),
                    mode="append",
                    commit_properties=CommitProperties(
                        app_transactions=[Transaction(app_id="job", version=5)]
                    ),
                )
            return snap

        monkeypatch.setattr(engine, "snapshot", racing_snapshot)
        with pytest.raises(_native.CommitConflictError):
            engine.append(
                resolved(plain),
                pa.table({"id": [3], "city": ["cairo"]}),
                txn=("job", 5),
                max_commit_retries=3,
            )
        ids = sorted(read(KernelEngine().scan(resolved(plain))).column("id").to_pylist())
        assert ids.count(3) == 1


def _native_has_app_id_version() -> bool:
    from deltaswamp.engine.kernel import _native_has

    return bool(_native_has("app_id_version"))


def test_kernel_does_not_claim_create_with_overwrite_mode() -> None:
    from deltaswamp.capability import Operation
    from deltaswamp.catalog.base import ResolvedTable
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.identity import parse_ref

    table = ResolvedTable(ref=parse_ref("/tmp/x"), location="file:///tmp/x")
    engine = KernelEngine()
    assert not engine.supports(Operation.CREATE, table, mode="overwrite").ok
