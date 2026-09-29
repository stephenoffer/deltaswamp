"""OPTIMIZE where the kernel refused it: row-tracked, liquid-clustered and
column-mapped tables; incremental Z-order, min_file_size and sort_by.

Every test here failed before its change (a refusal, or an option refused).
Databricks reads the same files in tests/live/test_live_compaction.py.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import pytest
from deltaswamp.capability import Engine
from deltaswamp.errors import EngineLimitError, UnreachableTableError

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
ds = pytest.importorskip("deltaswamp")

SCHEMA = pa.schema([("id", pa.int64()), ("v", pa.string()), ("g", pa.int32())])


def _rows(start: int, n: int) -> Any:
    ids = range(start, start + n)
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "v": [f"s{i}" for i in ids],
            "g": pa.array([i % 7 for i in ids], pa.int32()),
        }
    )


def _table(conn: Any, path: str, appends: int = 4, **kwargs: Any) -> Any:
    table = conn.create_table(path, SCHEMA, **kwargs)
    for i in range(appends):
        table.append(_rows(i * 10, 10))
    return conn.open_table(path)


def _commit(path: str, version: int) -> list[dict[str, Any]]:
    with open(os.path.join(path, "_delta_log", f"{version:020}.json")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _live_adds(path: str) -> dict[str, dict[str, Any]]:
    live: dict[str, dict[str, Any]] = {}
    for name in sorted(glob.glob(os.path.join(path, "_delta_log", "*.json"))):
        with open(name) as f:
            for line in f:
                action = json.loads(line)
                if "add" in action:
                    live[action["add"]["path"]] = action["add"]
                if "remove" in action:
                    live.pop(action["remove"]["path"], None)
    return live


def _row_tracking(path: str) -> dict[int, tuple[int, int]]:
    """Each row's (row id, row commit version), read as Databricks' _metadata does."""
    metadata = _commit(path, 0)
    config = next(a["metaData"] for a in metadata if "metaData" in a)["configuration"]
    id_col = config["delta.rowTracking.materializedRowIdColumnName"]
    cv_col = config["delta.rowTracking.materializedRowCommitVersionColumnName"]
    out: dict[int, tuple[int, int]] = {}
    for add in _live_adds(path).values():
        assert add.get("deletionVector") is None
        data = pq.read_table(os.path.join(path, add["path"]))
        names = data.column_names
        for index, row in enumerate(data.to_pylist()):
            rid = row[id_col] if id_col in names and row[id_col] is not None else None
            cv = row[cv_col] if cv_col in names and row[cv_col] is not None else None
            out[row["id"]] = (
                rid if rid is not None else add["baseRowId"] + index,
                cv if cv is not None else add["defaultRowCommitVersion"],
            )
    return out


class TestRowTracking:
    def test_optimize_keeps_every_rows_id_and_commit_version(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "rt")
        table = _table(conn, path, properties={"delta.enableRowTracking": "true"})
        before = _row_tracking(path)
        assert table.can("optimize").engine is Engine.KERNEL
        result = table.optimize()
        assert result["numFilesRemoved"] == 4 and result["numFilesAdded"] == 1
        assert _row_tracking(path) == before
        commit = _commit(path, 5)
        removes = [a["remove"] for a in commit if "remove" in a]
        assert len(removes) == 4
        assert all(r["dataChange"] is False and "baseRowId" in r for r in removes)
        adds = [a["add"] for a in commit if "add" in a]
        assert adds[0]["dataChange"] is False and adds[0]["baseRowId"] == 40
        assert conn.open_table(path).count() == 40

    def test_z_order_keeps_ids_through_the_reordering(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "rtz")
        table = _table(conn, path, properties={"delta.enableRowTracking": "true"})
        before = _row_tracking(path)
        table.z_order(["g"])
        assert _row_tracking(path) == before

    def test_optimize_of_row_tracked_deletion_vector_files(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "rtdv")
        table = _table(
            conn,
            path,
            properties={"delta.enableRowTracking": "true", "delta.enableDeletionVectors": "true"},
        )
        table.delete("id = 3")
        table.update(predicate="id = 17", updates={"v": "'u'"})
        # The DV'd files must be read with their vectors applied first.
        live = {row["id"] for row in conn.open_table(path).to_arrow().to_pylist()}
        conn.open_table(path).optimize()
        after = _row_tracking(path)
        assert set(after) == live and 3 not in after
        ids = [rid for rid, _ in after.values()]
        assert len(ids) == len(set(ids))
        assert after[17][1] == 6  # the UPDATE's commit, kept
        assert after[18] == (18, 2)

    def test_overwrite_of_a_row_tracked_table_gives_new_rows_fresh_ids(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "rtow")
        table = _table(
            conn,
            path,
            properties={"delta.enableRowTracking": "true", "delta.enableDeletionVectors": "true"},
        )
        table.delete("id = 1")
        verdict = table.can("overwrite")
        assert verdict.ok and verdict.engine is Engine.KERNEL
        conn.open_table(path).overwrite(_rows(100, 3))
        assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == [100, 101, 102]
        assert [rid for rid, _ in _row_tracking(path).values()] == [40, 41, 42]
        assert len(_live_adds(path)) == 1

    def test_replace_where_on_a_row_tracked_table_keeps_the_other_rows_ids(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "rtrw")
        table = _table(conn, path, properties={"delta.enableRowTracking": "true"})
        before = _row_tracking(path)
        verdict = table.can("overwrite", predicate="id < 5")
        assert verdict.ok and verdict.engine is Engine.KERNEL
        table.overwrite(_rows(0, 5), predicate="id < 5")
        after = _row_tracking(path)
        # Only the first file is rewritten: its other rows keep id and version;
        # the replaced rows are new rows, with fresh ids.
        assert {i: after[i] for i in range(5, 40)} == {i: before[i] for i in range(5, 40)}
        assert min(after[i][0] for i in range(5)) >= 40


class TestLiquidClustering:
    def test_optimize_z_orders_by_the_clustering_keys(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "cl")
        table = _table(conn, path, cluster_by=["g"])
        verdict = table.can("optimize")
        assert verdict.ok and verdict.engine is Engine.KERNEL
        assert "clustering keys" in verdict.reason
        result = table.optimize(target_size=1500)
        assert result["numFilesAdded"] > 1
        latest = max(int(os.path.basename(f)[:20]) for f in glob.glob(f"{path}/_delta_log/*.json"))
        commit = _commit(path, latest)
        info = next(a["commitInfo"] for a in commit if "commitInfo" in a)
        assert info["operationParameters"]["clusterBy"] == '["g"]'
        assert not any("domainMetadata" in a for a in commit)
        adds = [a["add"] for a in commit if "add" in a]
        assert all(a["tags"]["ZCUBE_ZORDER_BY"] == '["g"]' for a in adds)
        # Clustered: each output file holds a narrow range of the key.
        spans = [
            json.loads(a["stats"])["maxValues"]["g"] - json.loads(a["stats"])["minValues"]["g"]
            for a in adds
        ]
        assert max(spans) < 6
        assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == list(range(40))
        assert conn.open_table(path).cluster_by is not None

    def test_a_second_optimize_is_incremental_and_full_rewrites(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "cl2")
        table = _table(conn, path, cluster_by=["g"])
        table.optimize(min_cube_size=1)
        assert conn.open_table(path).optimize(min_cube_size=1)["numFilesAdded"] == 0
        conn.open_table(path).append(_rows(100, 5))
        again = conn.open_table(path).optimize(min_cube_size=1)
        assert again["numFilesRemoved"] == 1 and again["totalFilesSkipped"] == 1
        assert conn.open_table(path).can("optimize", full=True).ok
        full = conn.open_table(path).optimize(full=True, min_cube_size=1)
        assert full["numFilesRemoved"] == 2

    def test_z_order_by_and_full_are_refused_where_they_do_not_fit(
        self, conn: Any, tmp_path: Any
    ) -> None:
        clustered = _table(conn, str(tmp_path / "cl3"), cluster_by=["g"])
        assert not clustered.can("zorder", columns=["id"]).ok
        with pytest.raises((UnreachableTableError, EngineLimitError), match="Z-ORDER"):
            clustered.z_order(["id"])
        plain = _table(conn, str(tmp_path / "plain"))
        assert not plain.can("optimize", full=True).ok

    def test_a_row_tracked_clustered_table_keeps_its_ids(self, conn: Any, tmp_path: Any) -> None:
        # Databricks' CLUSTER BY enables row tracking by default.
        path = str(tmp_path / "clrt")
        table = _table(conn, path, cluster_by=["g"], properties={"delta.enableRowTracking": "true"})
        before = _row_tracking(path)
        table.optimize()
        assert _row_tracking(path) == before


class TestColumnMapping:
    @pytest.mark.parametrize("mode", ["name", "id"])
    def test_optimize_writes_physical_names_and_field_ids(
        self, conn: Any, tmp_path: Any, mode: str
    ) -> None:
        path = str(tmp_path / mode)
        table = conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("v", pa.string()), ("part", pa.string())]),
            partition_by=["part"],
            properties={"delta.columnMapping.mode": mode},
        )
        for i in range(3):
            table.append(
                pa.table(
                    {"id": pa.array([i, i + 10], pa.int64()), "v": ["a", "b"], "part": ["x", "y"]}
                )
            )
        conn.open_table(path).rename_column("v", "value")
        table = conn.open_table(path)
        before = sorted(table.to_arrow().to_pylist(), key=lambda r: r["id"])
        assert table.can("optimize").engine is Engine.KERNEL
        assert table.optimize()["numFilesAdded"] == 2
        after = sorted(conn.open_table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
        assert after == before
        schema = json.loads(
            next(a["metaData"] for a in _commit(path, 0) if "metaData" in a)["schemaString"]
        )
        physical = {
            f["name"]: f["metadata"]["delta.columnMapping.physicalName"] for f in schema["fields"]
        }
        for add in _live_adds(path).values():
            # Partition values keyed by physical name, as the protocol asks.
            assert list(add["partitionValues"]) == [physical["part"]]
            written = pq.ParquetFile(os.path.join(path, add["path"])).schema_arrow
            assert written.names == [physical["id"], physical["v"]]
            assert written.field(0).metadata[b"PARQUET:field_id"] == b"1"


class TestOptions:
    def test_z_order_skips_files_already_z_ordered_by_the_same_columns(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "z")
        table = _table(conn, path)
        table.z_order(["g"])
        adds = list(_live_adds(path).values())
        assert adds[0]["tags"]["ZCUBE_ZORDER_BY"] == '["g"]'
        assert "ZCUBE_ID" in adds[0]["tags"]
        # Before, the second Z-order rewrote the one file it had just written.
        second = conn.open_table(path).z_order(["g"])
        assert second["numFilesAdded"] == 0 and second["totalFilesSkipped"] == 1
        # Other columns are another order: rewritten.
        assert conn.open_table(path).z_order(["id"])["numFilesRemoved"] == 1

    def test_min_file_size_leaves_larger_files_alone(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "m")
        table = _table(conn, path)
        table.append(_rows(1000, 400))
        big = max(a["size"] for a in _live_adds(path).values())
        result = conn.open_table(path).optimize(min_file_size=big)
        assert result["numFilesRemoved"] == 4 and result["totalFilesSkipped"] == 1

    def test_sort_by_orders_each_bins_rows(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "s")
        table = _table(conn, path)
        result = table.optimize(sort_by=["g", "id"])
        assert result["preserveInsertionOrder"] is False
        (add,) = _live_adds(path).values()
        rows = pq.read_table(os.path.join(path, add["path"])).to_pylist()
        assert [(r["g"], r["id"]) for r in rows] == sorted((r["g"], r["id"]) for r in rows)
        with pytest.raises(Exception, match="one or the other"):
            conn.open_table(path).optimize(sort_by="g", zorder_by="id")
