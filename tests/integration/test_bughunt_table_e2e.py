"""End-to-end regressions for the public API against real local Delta tables.

Each test drives Connection/Table the way a user would and checks the result
against what was written (or that a misuse is refused clearly).
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.errors import DeltaSwampError, UnreachableTableError
from deltaswamp.table import InvalidArgumentError

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

UTC = dt.UTC


@pytest.fixture
def conn() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router
    from deltaswamp.table import Connection

    return Connection(
        catalog=FilesystemCatalog(),
        router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
    )


def _data(n: int = 3) -> Any:
    return pa.table({"id": pa.array(range(1, n + 1), pa.int64()), "v": [f"r{i}" for i in range(n)]})


def _path(tmp_path: Any, name: str = "t") -> str:
    return os.path.join(str(tmp_path), name)


# ------------------------------------------------------------------ inputs


def test_write_table_accepts_a_column_dict(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), {"id": [1, 2], "v": ["a", "b"]})
    assert sorted(t.to_arrow().column("id").to_pylist()) == [1, 2]


def test_write_table_accepts_a_list_of_row_dicts(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}])
    assert t.count() == 2


def test_append_accepts_a_column_dict(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    t.append({"id": pa.array([9], pa.int64()), "v": ["z"]})
    assert t.count() == 4


def test_create_table_accepts_a_list_of_fields(conn: Any, tmp_path: Any) -> None:
    t = conn.create_table(_path(tmp_path), [("id", pa.int64()), ("v", pa.string())])
    assert t.schema().names == ["id", "v"]


def test_create_table_accepts_a_name_to_type_dict(conn: Any, tmp_path: Any) -> None:
    t = conn.create_table(_path(tmp_path), {"id": pa.int64()})
    assert t.schema().names == ["id"]


def test_create_partition_by_a_single_name_on_the_kernel(conn: Any, tmp_path: Any) -> None:
    # The kernel create (chosen for row tracking) failed with "Can't extract str to Vec".
    t = conn.create_table(
        _path(tmp_path),
        pa.schema([("y", pa.int32()), ("p", pa.string())]),
        partition_by="p",
        properties={"delta.enableRowTracking": "true"},
    )
    assert t.detail()["partition_columns"] == ["p"]


# -------------------------------------------------------------- create modes


def test_unknown_create_mode_is_refused(conn: Any, tmp_path: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="create mode"):
        conn.create_table(_path(tmp_path), pa.schema([("y", pa.int32())]), mode="bogus")


def test_create_mode_create_works_on_delta_rs(conn: Any, tmp_path: Any) -> None:
    t = conn.create_table(_path(tmp_path), pa.schema([("y", pa.int32())]), mode="create")
    assert t.schema().names == ["y"]


def test_create_ignore_leaves_the_existing_table_alone(conn: Any, tmp_path: Any) -> None:
    path = _path(tmp_path)
    conn.create_table(path, pa.schema([("y", pa.int32())]), comment="orig")
    t = conn.create_table(
        path,
        pa.schema([("y", pa.int32())]),
        mode="ignore",
        comment="new",
        properties={"delta.enableRowTracking": "true"},
    )
    assert t.version == 0
    assert "delta.enableRowTracking" not in t.properties()


def test_create_ignore_creates_when_absent_on_the_kernel(conn: Any, tmp_path: Any) -> None:
    # The kernel knows no "ignore" mode; an absent table must still be created.
    t = conn.create_table(
        _path(tmp_path),
        pa.schema([("y", pa.int32())]),
        mode="ignore",
        properties={"delta.enableRowTracking": "true"},
    )
    assert t.properties()["delta.enableRowTracking"] == "true"


def test_create_over_an_existing_table_is_a_clear_refusal(conn: Any, tmp_path: Any) -> None:
    path = _path(tmp_path)
    conn.create_table(path, pa.schema([("y", pa.int32())]))
    with pytest.raises(UnreachableTableError, match="already exists"):
        conn.create_table(path, pa.schema([("y", pa.int32())]))


def test_partition_column_missing_from_schema_is_refused(conn: Any, tmp_path: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="partition_by"):
        conn.create_table(_path(tmp_path), pa.schema([("y", pa.int32())]), partition_by=["nope"])


def test_all_partition_columns_is_refused(conn: Any, tmp_path: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="non-partition"):
        conn.create_table(_path(tmp_path), pa.schema([("p", pa.string())]), partition_by=["p"])


def test_partition_and_cluster_together_are_refused(conn: Any, tmp_path: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="not both"):
        conn.create_table(
            _path(tmp_path),
            pa.schema([("p", pa.string()), ("y", pa.int32())]),
            partition_by=["p"],
            cluster_by=["y"],
        )


# ------------------------------------------------------------------- limit


def test_to_arrow_honours_limit(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data(5))
    assert t.to_arrow(limit=2).num_rows == 2
    assert len(t.to_pandas(limit=2)) == 2
    assert len(t.to_polars(limit=2)) == 2 if _has("polars") else True


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


# ------------------------------------------------------------------ update


def test_update_of_an_unknown_column_is_refused(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    before = t.version
    with pytest.raises(InvalidArgumentError, match="no column 'nope'"):
        t.update(new_values={"nope": 1})
    assert t.version == before


def test_update_matches_column_names_case_insensitively(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    t.update(new_values={"V": "x"}, predicate="id = 1")
    rows = {r["id"]: r["v"] for r in t.to_arrow().to_pylist()}
    assert rows[1] == "x"


def test_update_of_a_struct_field_on_delta_rs_is_refused(conn: Any, tmp_path: Any) -> None:
    data = pa.table({"id": [1], "s": pa.array([{"a": 1}], pa.struct([("a", pa.int64())]))})
    t = conn.write_table(_path(tmp_path), data)
    with pytest.raises(InvalidArgumentError, match="inside a struct"):
        t.update({"s.a": "5"})


def test_update_with_nothing_to_set_is_refused_before_routing(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError):
        t.update()
    with pytest.raises(InvalidArgumentError, match="not both"):
        t.update({"v": "'a'"}, new_values={"v": "b"})


# ------------------------------------------------------------ merge/restore


def test_blank_merge_predicate_is_refused(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="blank"):
        t.merge(_data(), "  ")


def test_merge_accepts_a_column_dict_source(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    (
        t.merge(
            {"id": pa.array([1, 9], pa.int64()), "v": ["x", "y"]},
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        )
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    assert t.count() == 4


def test_restore_to_a_negative_version_is_refused(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError):
        t.restore(-1)


def test_restore_past_the_latest_version_is_refused(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="latest version is 1"):
        t.restore(99)


# ------------------------------------------------------------ schema changes


def test_drop_of_the_last_data_column_is_refused(conn: Any, tmp_path: Any) -> None:
    data = pa.table({"p": ["a", "b"], "x": pa.array([1, 2], pa.int64())})
    t = conn.write_table(_path(tmp_path), data, partition_by=["p"])
    t.set_properties({"delta.columnMapping.mode": "name"})
    with pytest.raises(InvalidArgumentError, match="last non-partition"):
        t.drop_column("x")
    assert t.count() == 2


def test_add_column_that_exists_is_refused_clearly(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="already has 'id'"):
        t.add_column([pa.field("ID", pa.int32())])


# ------------------------------------------------------------- time travel


def test_cdf_accepts_history_timestamps(conn: Any, tmp_path: Any) -> None:
    t = conn.create_table(
        _path(tmp_path), _data().schema, properties={"delta.enableChangeDataFeed": "true"}
    )
    t.append(_data())
    first = min(h["timestamp"] for h in t.history())
    assert isinstance(first, int)
    assert pa.table(t.cdf(starting_timestamp=first)).num_rows == 3


def test_changes_with_a_projection(conn: Any, tmp_path: Any) -> None:
    t = conn.create_table(
        _path(tmp_path), _data().schema, properties={"delta.enableChangeDataFeed": "true"}
    )
    t.append(_data())
    got = [(v, tb.column_names) for v, tb in t.changes(1, columns=["id"])]
    assert got == [(1, ["id"])]


def test_writes_through_a_pinned_handle_are_refused(conn: Any, tmp_path: Any) -> None:
    path = _path(tmp_path)
    conn.write_table(path, _data())
    conn.table(path).append(_data())
    pinned = conn.table(path, version=1)
    with pytest.raises(InvalidArgumentError, match="pinned"):
        pinned.delete("id = 1")
    with pytest.raises(InvalidArgumentError, match="pinned"):
        pinned.append(_data())
    assert conn.table(path).count() == 6


def test_errors_are_library_errors() -> None:
    assert issubclass(InvalidArgumentError, DeltaSwampError)


# --------------------------------------------------------------- txn dedup


class _NoTxnEngine:
    """Writes like delta-rs but, like the kernel, cannot read transaction ids."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def append(self, *args: Any, **kwargs: Any) -> None:
        self._inner.append(*args, **kwargs)


def test_txn_dedup_falls_back_when_the_writer_cannot_read_txns(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    from deltaswamp.capability import Engine

    t = conn.write_table(_path(tmp_path), _data())
    t.append(_data(), txn=("job", 1))
    stub = _NoTxnEngine(conn.router.engines[Engine.DELTARS])
    monkeypatch.setattr(type(t), "_engine", lambda self, *a, **k: stub)
    t.append(_data(), txn=("job", 1))  # a replay: must be skipped
    monkeypatch.undo()
    assert t.count() == 6


def test_txn_write_is_refused_when_no_engine_can_check(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    from deltaswamp.capability import Engine

    t = conn.write_table(_path(tmp_path), _data())
    stub = _NoTxnEngine(conn.router.engines[Engine.DELTARS])
    monkeypatch.setattr(type(t), "_engine", lambda self, *a, **k: stub)
    monkeypatch.delitem(conn.router.engines, Engine.DELTARS)
    with pytest.raises(UnreachableTableError, match="transaction"):
        t.append(_data(), txn=("job", 1))
    monkeypatch.undo()
    assert t.count() == 3


# ------------------------------------------------------------ alignment


def test_append_fills_a_left_out_nullable_column(conn: Any, tmp_path: Any) -> None:
    data = pa.table({"id": pa.array([1], pa.int64()), "v": ["a"], "n": pa.array([1.0])})
    t = conn.write_table(_path(tmp_path), data)
    t.append(pa.table({"id": pa.array([2], pa.int64()), "v": ["b"]}))
    rows = {r["id"]: r for r in t.to_arrow().to_pylist()}
    assert rows[2]["n"] is None and rows[2]["v"] == "b"


def test_append_matches_column_names_case_insensitively(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    t.append(pa.table({"ID": pa.array([9], pa.int64()), "V": ["z"]}))
    assert t.schema().names == ["id", "v"]
    assert 9 in t.to_arrow().column("id").to_pylist()


def test_overwrite_fills_a_left_out_nullable_column(conn: Any, tmp_path: Any) -> None:
    data = pa.table({"id": pa.array([1], pa.int64()), "v": ["a"], "n": pa.array([1.0])})
    t = conn.write_table(_path(tmp_path), data)
    conn.write_table(_path(tmp_path), _data(2), mode="overwrite")
    assert t.count() == 2


def test_append_pandas_with_a_map_column(conn: Any, tmp_path: Any) -> None:
    data = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64())),
        }
    )
    t = conn.write_table(_path(tmp_path), data)
    t.append(data.to_pandas())
    assert t.count() == 2


def test_write_table_with_a_mismatched_schema_creates_nothing(conn: Any, tmp_path: Any) -> None:
    path = _path(tmp_path)
    with pytest.raises(InvalidArgumentError, match="nothing was created"):
        conn.write_table(path, _data(), schema=pa.schema([("x", pa.int64())]))
    assert not conn.table_exists(path)
    assert conn.write_table(path, _data()).count() == 3


# ------------------------------------------------------- misuse, refusals


def test_set_properties_takes_a_dict(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="dict"):
        t.set_properties([("delta.appendOnly", "true")])


def test_add_constraint_needs_a_mapping_and_an_expression(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError):
        t.add_constraint("id > 0")
    with pytest.raises(InvalidArgumentError, match="blank"):
        t.add_constraint({"c": "  "})


def test_a_missing_path_is_not_sent_to_the_warehouse(conn: Any, tmp_path: Any) -> None:
    from deltaswamp.errors import FallbackRequiredError

    t = conn.table(_path(tmp_path, "nope"))
    with pytest.raises(UnreachableTableError, match="no Delta table at this path") as info:
        t.count()
    assert not isinstance(info.value, FallbackRequiredError)


def test_can_with_an_unknown_operation_names_the_valid_ones(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="append"):
        t.can("bogus")


# --------------------------------------------------------- types, cleanup


def test_uint8_values_round_trip(conn: Any, tmp_path: Any) -> None:
    data = pa.table({"id": pa.array([1], pa.int64()), "c": pa.array([200], pa.uint8())})
    t = conn.write_table(_path(tmp_path), data)
    t.append(data)
    assert t.to_arrow().column("c").to_pylist() == [200, 200]


def test_a_failed_first_write_leaves_no_empty_table(conn: Any, tmp_path: Any) -> None:
    path = _path(tmp_path)
    big = pa.table({"c": pa.array([2**63 + 5], pa.uint64())})
    with pytest.raises(Exception, match="not in range"):
        conn.write_table(path, big)
    assert not conn.table_exists(path)
    ok = pa.table({"c": pa.array([5], pa.uint64())})
    assert conn.write_table(path, ok).count() == 1


def test_restore_of_an_append_only_table_is_refused(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data(), properties={"delta.appendOnly": "true"})
    t.append(_data())
    with pytest.raises(UnreachableTableError, match="appendOnly"):
        t.restore(1)
    assert t.count() == 6


def test_scan_accepts_a_history_timestamp(conn: Any, tmp_path: Any) -> None:
    import time

    t = conn.write_table(_path(tmp_path), _data())
    time.sleep(0.01)
    t.append(_data())
    first_write = next(h for h in t.history() if h["version"] == 1)
    # Commit times resolve against the log files; one past the recorded time
    # names version 1 unambiguously.
    assert t.to_arrow(timestamp=first_write["timestamp"] + 1).num_rows == 3
    assert len(t.plan_scan(timestamp=first_write["timestamp"] + 1).splits) == 1


def test_a_named_range_like_index_is_kept(conn: Any, tmp_path: Any) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}).set_index("id")
    t = conn.write_table(_path(tmp_path), frame)
    assert set(t.schema().names) == {"id", "v"}
    t.append(frame)
    assert sorted(t.to_arrow().column("id").to_pylist()) == [1, 1, 2, 2, 3, 3]


def test_comment_must_be_text(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="comment"):
        t.set_comment(123)
    with pytest.raises(InvalidArgumentError, match="comment"):
        t.set_column_comment("id", 5)


def test_vacuum_retention_must_be_a_number(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="number"):
        t.vacuum(retention_hours="200")


def test_z_order_by_a_missing_or_partition_column_is_refused(conn: Any, tmp_path: Any) -> None:
    data = pa.table({"id": pa.array([1, 2], pa.int64()), "p": ["a", "b"]})
    t = conn.write_table(_path(tmp_path), data, partition_by=["p"])
    with pytest.raises(InvalidArgumentError, match="no such column"):
        t.z_order(["nope"])
    with pytest.raises(InvalidArgumentError, match="partition column"):
        t.optimize(zorder_by="p")
    t.z_order("id")


def test_a_left_out_partition_column_is_not_null_filled(conn: Any, tmp_path: Any) -> None:
    data = pa.table({"id": pa.array([1], pa.int64()), "p": ["a"]})
    t = conn.write_table(_path(tmp_path), data, partition_by=["p"])
    with pytest.raises(Exception):  # noqa: B017 - the engine's own refusal
        t.append(pa.table({"id": pa.array([2], pa.int64())}))
    assert t.count() == 1


def test_location_with_a_path_name_is_refused(conn: Any, tmp_path: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="location"):
        conn.write_table(_path(tmp_path), _data(), location=_path(tmp_path, "elsewhere"))
    assert not conn.table_exists(_path(tmp_path))


def test_dynamic_overwrite_with_an_empty_batch_changes_nothing(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), pa.table({"id": [1], "p": ["a"]}), partition_by=["p"])
    before = t.version
    empty = pa.table({"id": pa.array([], pa.int64()), "p": pa.array([], pa.string())})
    t.overwrite(empty, partition_overwrite="dynamic")
    assert t.version == before and t.count() == 1


def test_write_table_to_an_existing_table_does_not_drop_layout_args(
    conn: Any, tmp_path: Any
) -> None:
    path = _path(tmp_path)
    conn.write_table(path, _data())
    with pytest.raises(InvalidArgumentError, match="partition_by"):
        conn.write_table(path, _data(), mode="append", partition_by=["v"])
    with pytest.raises(InvalidArgumentError, match="set_properties"):
        conn.write_table(path, _data(), mode="overwrite", properties={"delta.appendOnly": "true"})
    assert conn.table(path).count() == 3
    conn.write_table(path, _data(), mode="append", partition_by=[])
    assert conn.table(path).count() == 6


# ---------------------------------------------------- managed create, UC


@pytest.fixture
def uc_conn(tmp_path: Any) -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.catalog.ossuc import OSSUnityCatalog
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router
    from deltaswamp.table import Connection

    from tests.fake_uc import FakeUnityCatalog

    with FakeUnityCatalog(staging_root=tmp_path / "managed") as server:
        cat = OSSUnityCatalog(server.url)
        cat.create_catalog("main")
        cat.create_schema("main", "sales")
        yield Connection(
            catalog=cat,
            router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
        )


@pytest.mark.parametrize(
    "failure",
    [
        "main.sales.t was created in Unity Catalog, but reading it back failed: boom",
        "connection reset",
    ],
)
def test_a_failed_read_back_after_a_managed_create_says_it_was_created(
    uc_conn: Any, monkeypatch: Any, failure: str
) -> None:
    from deltaswamp.errors import PreflightError

    real = uc_conn.catalog.finalize_managed_table

    def finalize_then_fail(ref: Any, body: Any) -> Any:
        real(ref, body)
        raise PreflightError(failure)

    monkeypatch.setattr(uc_conn.catalog, "finalize_managed_table", finalize_then_fail)
    with pytest.raises(UnreachableTableError, match="do not repeat the create") as info:
        uc_conn.create_table("main.sales.t", pa.schema([("id", pa.int64())]))
    assert failure in str(info.value)
    assert "did not accept" not in str(info.value)
    monkeypatch.undo()
    assert uc_conn.table("main.sales.t").count() == 0


def test_a_failed_external_registration_points_at_register_table(
    uc_conn: Any, monkeypatch: Any, tmp_path: Any
) -> None:
    from deltaswamp.errors import PreflightError

    location = str(tmp_path / "ext")

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise PreflightError("HTTP 503")

    monkeypatch.setattr(uc_conn.catalog, "register_table", refuse)
    with pytest.raises(UnreachableTableError, match="register_table") as info:
        uc_conn.create_table("main.sales.ext", pa.schema([("id", pa.int64())]), location=location)
    assert "HTTP 503" in str(info.value)
    monkeypatch.undo()
    assert uc_conn.register_table("main.sales.ext", location).count() == 0


def test_create_ignore_on_a_catalog_name(uc_conn: Any) -> None:
    schema = pa.schema([("id", pa.int64())])
    first = uc_conn.create_table("main.sales.t", schema, mode="ignore")
    again = uc_conn.create_table("main.sales.t", schema, mode="ignore")
    assert again.resolved.table_id == first.resolved.table_id


def test_changes_poll_interval_must_be_a_number(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(_path(tmp_path), _data())
    with pytest.raises(InvalidArgumentError, match="poll_interval"):
        next(t.changes(0, poll_interval="1"))


def test_a_handle_pinned_past_the_latest_version_says_so(conn: Any, tmp_path: Any) -> None:
    from deltaswamp.errors import FallbackRequiredError

    path = _path(tmp_path)
    conn.write_table(path, _data())
    with pytest.raises(UnreachableTableError, match="the latest is 1") as info:
        conn.table(path, version=99).count()
    assert not isinstance(info.value, FallbackRequiredError)
