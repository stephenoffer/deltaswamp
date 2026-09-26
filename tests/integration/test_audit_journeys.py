"""Regressions for defects found by walking whole user journeys.

Every test here failed before its fix: a follower that never advanced, a
table one ALTER left unwritable, data no other engine could read.
"""

from __future__ import annotations

from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.errors import (  # noqa: E402
    ChangeFeedSchemaChangeError,
    DeltaSwampError,
    EngineLimitError,
)


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _cdf_table(conn: Any, path: str, **properties: str) -> Any:
    props = {"delta.enableChangeDataFeed": "true", **properties}
    return conn.create_table(path, pa.schema([("id", pa.int64())]), properties=props)


class TestChangeFeedAcrossASchemaChange:
    """changes() read start..latest as one range, and a range crossing an ADD
    COLUMN failed on every poll: the follower never got past it."""

    def test_follower_advances_past_add_column(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = _cdf_table(conn, path, **{"delta.columnMapping.mode": "name"})
        t.append(pa.table({"id": [1, 2]}))
        t.add_column(pa.schema([("x", pa.string())]))
        t.append(pa.table({"id": [3], "x": ["a"]}))
        got = list(conn.open_table(path).changes(0))
        assert [(v, b.num_rows) for v, b in got] == [(1, 2), (3, 1)]
        assert "x" not in got[0][1].column_names
        assert got[1][1].column("x").to_pylist() == ["a"]

    def test_cdf_reads_older_rows_under_the_new_schema(self, conn: Any, tmp_path: Any) -> None:
        # Spark reads a batch feed across an additive change, older rows null.
        path = str(tmp_path / "t")
        t = _cdf_table(conn, path, **{"delta.columnMapping.mode": "name"})
        t.append(pa.table({"id": [1, 2]}))
        t.add_column(pa.schema([("x", pa.string())]))
        t.append(pa.table({"id": [3], "x": ["a"]}))
        feed = conn.open_table(path).cdf(starting_version=0).read_all()
        assert feed.column_names[:2] == ["id", "x"]
        rows = sorted(feed.select(["id", "x"]).to_pylist(), key=lambda r: r["id"])
        assert rows == [{"id": 1, "x": None}, {"id": 2, "x": None}, {"id": 3, "x": "a"}]

    def test_dropped_column_names_the_version(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        t = conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("x", pa.string())]),
            properties={"delta.enableChangeDataFeed": "true", "delta.columnMapping.mode": "name"},
        )
        t.append(pa.table({"id": [1], "x": ["a"]}))
        t.drop_column("x")
        t.append(pa.table({"id": [2]}))
        with pytest.raises(ChangeFeedSchemaChangeError) as caught:
            conn.open_table(path).cdf(starting_version=0)
        assert caught.value.version == 2
        assert "from version 2 on" in str(caught.value)
        # A follower yields each version under the schema it was written with.
        got = list(conn.open_table(path).changes(0))
        assert [v for v, _ in got] == [1, 3]

    def test_parquet_decode_error_is_not_a_schema_change(self, conn: Any, tmp_path: Any) -> None:
        # arrow-rs's "cannot skip miniblock of size 256" on a Photon file was
        # reported as a schema change: words in the traceback the C stream
        # embeds matched.
        path = str(tmp_path / "t")
        t = _cdf_table(conn, path, **{"delta.columnMapping.mode": "name"})
        t.append(pa.table({"id": [1]}))
        t.add_column(pa.schema([("x", pa.string())]))
        exc = OSError(
            "Invalid: C Data interface error: External error: Arrow error: Parquet argument "
            "error: Parquet error: cannot skip miniblock of size 256. Detail: Python "
            "exception: Traceback ... schema ... cast"
        )
        assert conn.open_table(path)._feed_schema_change(exc, 0, None) is None


class TestMiniblockErrorIsTyped:
    def test_miniblock_error_is_an_engine_limit(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.engine import base

        path = str(tmp_path / "t")
        t = _cdf_table(conn, path)
        t.append(pa.table({"id": [1]}))
        captured: dict[str, Any] = {}
        real = base.translating_stream

        def spy(source: Any, context: str, translate: Any = None) -> Any:
            captured["translate"] = translate
            return real(source, context, translate)

        import deltaswamp.table as table_module

        table_module.translating_stream = spy
        try:
            conn.open_table(path).cdf(starting_version=0, ending_version=1)
        finally:
            table_module.translating_stream = real
        error = captured["translate"](OSError("Parquet error: cannot skip miniblock of size 256"))
        assert isinstance(error, EngineLimitError)
        assert isinstance(error, DeltaSwampError)
        assert "allow_sql_fallback" in str(error)


def _dv_table(conn: Any, path: str, **properties: str) -> Any:
    props = {"delta.enableDeletionVectors": "true", **properties}
    t = conn.create_table(path, pa.schema([("id", pa.int64())]), properties=props)
    t.append(pa.table({"id": [1, 2, 3]}))
    return t


def _race(monkeypatch: Any, concurrent: Any) -> None:
    """Run `concurrent` once, just after the next DV DML has read its snapshot."""
    from deltaswamp.engine.kernel import KernelEngine

    real = KernelEngine._dv_dml
    state = {"done": False}

    def racing_snapshot(self: Any, table: Any, **kwargs: Any) -> Any:
        snap = real_snapshot(self, table, **kwargs)
        if kwargs.get("write") and not state["done"]:
            state["done"] = True
            concurrent()
        return snap

    real_snapshot = KernelEngine.snapshot

    def dml(self: Any, *args: Any, **kwargs: Any) -> Any:
        monkeypatch.setattr(KernelEngine, "snapshot", racing_snapshot)
        try:
            return real(self, *args, **kwargs)
        finally:
            monkeypatch.setattr(KernelEngine, "snapshot", real_snapshot)

    monkeypatch.setattr(KernelEngine, "_dv_dml", dml)


class TestDeletionVectorDmlSurvivesBlindAppends:
    """A kernel DV DELETE/UPDATE failed CommitConflictError on any concurrent
    append (23 of 24 deletes, with eight appenders); delta-rs retried."""

    def test_delete_rebases_over_a_blind_append(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = str(tmp_path / "t")
        t = _dv_table(conn, path)
        assert "kernel" in str(t.can("delete"))
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).append(pa.table({"id": [1, 9]})))
        result = conn.open_table(path).delete("id = 1")
        assert result["num_deleted_rows"] == 1
        # WriteSerializable: the appended id 1 was never read, so it stays.
        ids = sorted(conn.open_table(path).to_arrow().column("id").to_pylist())
        assert ids == [1, 2, 3, 9]

    def test_update_rebases_over_a_blind_append(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        path = str(tmp_path / "t")
        _dv_table(conn, path)
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).append(pa.table({"id": [7]})))
        conn.open_table(path).update(new_values={"id": 20}, predicate="id = 2")
        ids = sorted(conn.open_table(path).to_arrow().column("id").to_pylist())
        assert ids == [1, 3, 7, 20]

    def test_concurrent_delete_of_the_same_file_conflicts(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from deltaswamp.errors import CommitConflictError

        path = str(tmp_path / "t")
        _dv_table(conn, path)
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).delete("id = 3"))
        with pytest.raises(CommitConflictError) as caught:
            conn.open_table(path).delete("id = 1")
        assert "staged file" not in str(caught.value)
        assert "txnId" not in str(caught.value)
        ids = sorted(conn.open_table(path).to_arrow().column("id").to_pylist())
        assert ids == [1, 2]

    def test_serializable_isolation_conflicts_on_an_append(
        self, conn: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from deltaswamp.errors import CommitConflictError

        path = str(tmp_path / "t")
        _dv_table(conn, path).set_properties({"delta.isolationLevel": "Serializable"})
        other = ds.connect("file://")
        _race(monkeypatch, lambda: other.open_table(path).append(pa.table({"id": [1]})))
        with pytest.raises(CommitConflictError):
            conn.open_table(path).delete("id = 1")
