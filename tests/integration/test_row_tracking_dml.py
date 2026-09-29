"""Kernel DML where delta-rs cannot write: copy-on-write MERGE, and row tracking.

Tables whose features only the kernel writes (in-commit timestamps, liquid
clustering, type widening, column defaults) take MERGE copy-on-write when they
do not enable deletion vectors: every file holding a touched row is removed,
and its other rows are written again beside the new ones.

On a row-tracked table every DML keeps the row ids the Delta protocol promises
stable: a row the DML keeps has its id and commit version afterwards, an
updated row its id (with this commit's version), an inserted row a fresh id
above the high-water mark. Ids are read two ways: through the kernel, and
straight from the log and the Parquet files (the materialized column, else the
file's baseRowId plus the row's position), as Databricks' `_metadata` does.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest
from deltaswamp.capability import Engine

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
deltalake = pytest.importorskip("deltalake")
native = pytest.importorskip("deltaswamp._native")

pytestmark = pytest.mark.skipif(
    "row_tracking_dml" not in native.FEATURES, reason="native build predates row-tracking DML"
)

from tests.contract.tables import BY_NAME  # noqa: E402

_RT = {"delta.enableRowTracking": "true"}
_RT_DV = {**_RT, "delta.enableDeletionVectors": "true"}


def _commits(path: str) -> list[list[dict[str, Any]]]:
    logs = sorted(pathlib.Path(path, "_delta_log").glob("*.json"))
    return [[json.loads(line) for line in log.read_text().splitlines()] for log in logs]


def _config(path: str) -> dict[str, str]:
    config: dict[str, str] = {}
    for commit in _commits(path):
        for action in commit:
            if "metaData" in action:
                config = action["metaData"]["configuration"]
    return config


def _live_adds(path: str) -> dict[str, dict[str, Any]]:
    live: dict[str, dict[str, Any]] = {}
    for commit in _commits(path):
        for action in commit:
            if "add" in action:
                live[action["add"]["path"]] = action["add"]
            if "remove" in action:
                live.pop(action["remove"]["path"], None)
    return live


def _row_tracking_from_files(path: str) -> dict[int, tuple[int, int]]:
    """Each row's (row id, row commit version), from the log and the Parquet files alone."""
    config = _config(path)
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
            assert row["id"] not in out
            out[row["id"]] = (
                rid if rid is not None else add["baseRowId"] + index,
                cv if cv is not None else add["defaultRowCommitVersion"],
            )
    return out


def _row_tracking_from_kernel(conn: Any, path: str) -> dict[int, tuple[int, int]]:
    """The same, read through the kernel's positional scan."""
    from deltaswamp.engine.kernel import KernelEngine

    snapshot = KernelEngine().snapshot(conn.open_table(path).resolved)
    rows = pa.table(snapshot.scan(row_positions=True, row_tracking=True))
    return {
        i: (rid, cv)
        for i, rid, cv in zip(
            rows.column("id").to_pylist(),
            rows.column("__deltaswamp_row_id").to_pylist(),
            rows.column("__deltaswamp_row_commit_version").to_pylist(),
            strict=True,
        )
    }


def _check_row_tracking_log(path: str) -> None:
    """What the protocol asks of a row-tracked table's log, commit by commit.

    Every add has a baseRowId and a defaultRowCommitVersion (its commit's
    version for a new file); every remove and every re-add of a file whose
    vector changed carries the values of the add it replaces; and the
    high-water mark covers every id assigned, never moving back.
    """
    live: dict[str, dict[str, Any]] = {}
    mark = -1
    for version, commit in enumerate(_commits(path)):
        new_adds, removed = [], set()
        for action in commit:
            if "remove" in action:
                remove = action["remove"]
                old = live.pop(remove["path"])
                assert remove.get("baseRowId") == old["baseRowId"], (version, remove)
                assert remove.get("defaultRowCommitVersion") == old["defaultRowCommitVersion"]
                removed.add(remove["path"])
        for action in commit:
            if "add" in action:
                add = action["add"]
                assert add.get("baseRowId") is not None, (version, add)
                assert add.get("defaultRowCommitVersion") is not None, (version, add)
                if add["path"] in removed:  # a new deletion vector on the same file
                    assert add.get("deletionVector") is not None
                else:
                    assert add["defaultRowCommitVersion"] == version
                    new_adds.append(add)
                live[add["path"]] = add
            if "domainMetadata" in action and action["domainMetadata"]["domain"] == (
                "delta.rowTracking"
            ):
                new_mark = json.loads(action["domainMetadata"]["configuration"])[
                    "rowIdHighWaterMark"
                ]
                assert new_mark >= mark
                mark = new_mark
        for add in new_adds:
            records = json.loads(add["stats"])["numRecords"]
            assert add["baseRowId"] + records - 1 <= mark, (version, add, mark)


def _deltars_ids(path: str) -> list[int]:
    return sorted(deltalake.DeltaTable(path).to_pyarrow_table().column("id").to_pylist())


def _row_tracked(conn: Any, path: str, properties: dict[str, str], **kwargs: Any) -> Any:
    schema = pa.schema([("id", pa.int64()), ("v", pa.string()), ("p", pa.string())])
    conn.create_table(path, schema, properties=properties, **kwargs)
    table = conn.open_table(path)
    table.append(
        pa.table({"id": pa.array([1, 2, 3], pa.int64()), "v": ["a", "b", "c"], "p": ["x"] * 3})
    )
    table.append(
        pa.table({"id": pa.array([4, 5, 6], pa.int64()), "v": ["d", "e", "f"], "p": ["y"] * 3})
    )
    return conn.open_table(path)


# ----------------------------------------------------------- copy-on-write MERGE


@pytest.mark.parametrize("fixture", ["ict", "clustered", "type_widening", "defaults"])
def test_kernel_merges_copy_on_write_where_only_it_writes(
    conn: Any, tmp_path: Any, fixture: str
) -> None:
    path = str(tmp_path / fixture)
    os.makedirs(path)
    BY_NAME[fixture].build(conn, path)
    table = conn.open_table(path)
    source = pa.table(
        {
            "id": pa.array([2, 5, 9], pa.int64()),
            "v": ["B", "gone", "i"],
            "grp": ["x", "z", "q"],
            "w": pa.array([20, 50, 90], pa.int32()),
        }
    )
    verdict = table.can(
        "merge",
        source=source,
        predicate="target.id = source.id",
        source_alias="source",
        target_alias="target",
        clauses=["when_matched_delete", "when_matched_update", "when_not_matched_insert"],
    )
    assert verdict.ok and verdict.engine is Engine.KERNEL, verdict.reason
    before = set(_live_adds(path))
    result = (
        table.merge(
            source=source,
            predicate="target.id = source.id",
            source_alias="source",
            target_alias="target",
        )
        .when_matched_delete("source.v = 'gone'")
        .when_matched_update({"v": "source.v", "w": "source.w"})
        .when_not_matched_insert({"id": "source.id", "v": "source.v", "grp": "source.grp"})
        .execute()
    )
    assert result.engine == "kernel"
    assert result["num_target_rows_updated"] == 1
    assert result["num_target_rows_deleted"] == 1
    assert result["num_target_rows_inserted"] == 1

    rows = conn.open_table(path).to_arrow().sort_by("id")
    assert rows.column("id").to_pylist() == [1, 2, 3, 4, 6, 9]
    assert rows.column("v").to_pylist() == ["a", "B", "c", "d", "f", "i"]
    assert rows.column("w").to_pylist()[1] == 20
    if fixture == "defaults":
        # The INSERT left `n` out: it takes the column's DEFAULT, 42.
        assert rows.column("n").to_pylist()[-1] == 42
        # ... and no other column of the rows it kept or updated changed.
        assert rows.column("n").to_pylist()[:-1] == [1, 2, 3, 4, 6]
    # delta-rs, an independent reader, sees the same rows (typeWidening is a
    # reader feature it may refuse).
    if fixture != "type_widening":
        assert _deltars_ids(path) == [1, 2, 3, 4, 6, 9]

    # The commit: every touched file removed, the rows it kept written again,
    # no deletion vector anywhere, and not a blind append.
    commit = _commits(path)[-1]
    info = next(a["commitInfo"] for a in commit if "commitInfo" in a)
    assert info["operation"] == "MERGE"
    assert info.get("isBlindAppend") is not True
    removed = {a["remove"]["path"] for a in commit if "remove" in a}
    assert removed and removed <= before
    assert all(a["remove"]["dataChange"] for a in commit if "remove" in a)
    adds = [a["add"] for a in commit if "add" in a]
    assert adds and all(a.get("deletionVector") is None for a in adds)
    assert set(_live_adds(path)) == (before - removed) | {a["path"] for a in adds}


def test_copy_on_write_merge_leaves_untouched_files_alone(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "ict")
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("v", pa.string())]),
        properties={"delta.enableInCommitTimestamps": "true"},
    )
    table = conn.open_table(path)
    for chunk in ([1, 2], [3, 4], [5, 6]):
        table.append(pa.table({"id": pa.array(chunk, pa.int64()), "v": ["o"] * 2}))
    before = set(_live_adds(path))
    source = pa.table({"id": pa.array([3, 7], pa.int64()), "v": ["n", "n"]})
    table.merge(
        source=source, predicate="t.id = s.id", source_alias="s", target_alias="t"
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    commit = _commits(path)[-1]
    removed = {a["remove"]["path"] for a in commit if "remove" in a}
    # Only the file holding id 3 is rewritten; the skipping predicate the ON
    # clause implies kept the others from being read at all.
    assert len(removed) == 1 and len(before - removed) == 2
    rows = conn.open_table(path).to_arrow().sort_by("id").to_pylist()
    assert [(r["id"], r["v"]) for r in rows] == [
        (1, "o"),
        (2, "o"),
        (3, "n"),
        (4, "o"),
        (5, "o"),
        (6, "o"),
        (7, "n"),
    ]
    assert "inCommitTimestamp" in next(a["commitInfo"] for a in commit if "commitInfo" in a)


# ------------------------------------------------------------------ row tracking


@pytest.mark.parametrize("properties", [_RT, _RT_DV], ids=["copy_on_write", "deletion_vectors"])
@pytest.mark.parametrize("partitioned", [False, True], ids=["flat", "partitioned"])
def test_dml_keeps_row_ids_stable(
    conn: Any, tmp_path: Any, properties: dict[str, str], partitioned: bool
) -> None:
    path = str(tmp_path / "rt")
    table = _row_tracked(conn, path, properties, partition_by=["p"] if partitioned else None)
    dv = "delta.enableDeletionVectors" in properties
    start = _row_tracking_from_kernel(conn, path)
    assert sorted(rid for rid, _ in start.values()) == [0, 1, 2, 3, 4, 5]
    for operation in ("delete", "update", "merge", "replace_where"):
        assert table.can(operation).engine is Engine.KERNEL, operation

    def version() -> int:
        return int(conn.open_table(path).version)

    expected = dict(start)
    table.update({"v": "'B'"}, predicate="id = 2")
    expected[2] = (start[2][0], version())  # its id, this commit's version
    assert _row_tracking_from_kernel(conn, path) == expected

    table.delete("id = 5")
    del expected[5]
    assert _row_tracking_from_kernel(conn, path) == expected

    source = pa.table(
        {"id": pa.array([3, 9, 6], pa.int64()), "v": ["C", "i", "drop"], "p": ["x", "x", "y"]}
    )
    conn.open_table(path).merge(
        source=source, predicate="t.id = s.id", source_alias="s", target_alias="t"
    ).when_matched_delete(
        "s.v = 'drop'"
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    merged = version()
    del expected[6]
    expected[3] = (start[3][0], merged)
    after = _row_tracking_from_kernel(conn, path)
    fresh = after[9]
    # An inserted row: a fresh id no row had, at this commit's version.
    assert fresh[1] == merged and fresh[0] > max(rid for rid, _ in start.values())
    expected[9] = fresh
    assert after == expected

    table = conn.open_table(path)
    table.overwrite(
        pa.table({"id": pa.array([4], pa.int64()), "v": ["D"], "p": ["y"]}), predicate="id = 4"
    )
    after = _row_tracking_from_kernel(conn, path)
    # replaceWhere deletes the rows it replaces and inserts new ones: fresh.
    assert after[4][0] not in {rid for rid, _ in expected.values()}
    assert after[4][1] == version()
    expected[4] = after[4]
    assert after == expected

    rows = conn.open_table(path).to_arrow().sort_by("id")
    assert rows.column("id").to_pylist() == [1, 2, 3, 4, 9]
    assert rows.column("v").to_pylist() == ["a", "B", "C", "D", "i"]
    ids = [rid for rid, _ in after.values()]
    assert len(set(ids)) == len(ids)
    _check_row_tracking_log(path)
    if not dv:
        # Read without the kernel: the materialized columns, else the file's
        # baseRowId and defaultRowCommitVersion, as Databricks reads them.
        assert _row_tracking_from_files(path) == expected
        assert _deltars_ids(path) == [1, 2, 3, 4, 9]
        # Copy-on-write wrote no vector, and removed the files it rewrote.
        assert not list(pathlib.Path(path).rglob("deletion_vector_*.bin"))


def test_row_ids_survive_a_second_rewrite_and_a_compaction(conn: Any, tmp_path: Any) -> None:
    """Materialized ids written by one DML are read back and kept by the next."""
    path = str(tmp_path / "rt")
    table = _row_tracked(conn, path, _RT)
    start = _row_tracking_from_files(path)
    table.update({"v": "'x1'"}, predicate="id = 1")
    table.update({"v": "'x2'"}, predicate="id = 2")
    table.delete("id = 3")
    table.optimize()
    after = _row_tracking_from_files(path)
    assert {i: rid for i, (rid, _) in after.items()} == {
        i: rid for i, (rid, _) in start.items() if i != 3
    }
    # Untouched rows kept their commit versions through every rewrite.
    assert {i: after[i][1] for i in (4, 5, 6)} == {i: start[i][1] for i in (4, 5, 6)}
    assert after == _row_tracking_from_kernel(conn, path)
    _check_row_tracking_log(path)


def test_dv_dml_and_compaction_carry_row_tracking_fields_on_removes(
    conn: Any, tmp_path: Any
) -> None:
    """Every remove, and every re-add under a new vector, keeps the file's baseRowId.

    delta-kernel-rs#3418 validates this; Spark relies on it.
    """
    path = str(tmp_path / "rt")
    table = _row_tracked(conn, path, _RT_DV)
    table.delete("id = 1")
    table.update({"v": "'e2'"}, predicate="id = 5")
    table.delete("id = 2 OR id = 3")  # the rest of the first file
    table.optimize()
    removes = [a["remove"] for c in _commits(path) for a in c if "remove" in a]
    assert removes
    assert all(r.get("baseRowId") is not None for r in removes)
    assert all(r.get("defaultRowCommitVersion") is not None for r in removes)
    _check_row_tracking_log(path)
    assert sorted(_row_tracking_from_kernel(conn, path)) == [4, 5, 6]


def test_row_tracking_supported_but_not_enabled_takes_dml(conn: Any, tmp_path: Any) -> None:
    """Ids are assigned but not promised stable: nothing is carried, removes still staged."""
    path = str(tmp_path / "rt")
    table = _row_tracked(conn, path, {"delta.feature.rowTracking": "supported"})
    assert table.can("merge").engine is Engine.KERNEL
    table.delete("id = 1")
    source = pa.table({"id": pa.array([2, 8], pa.int64()), "v": ["B", "h"], "p": ["x", "x"]})
    table.merge(
        source=source, predicate="t.id = s.id", source_alias="s", target_alias="t"
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    assert _deltars_ids(path) == [2, 3, 4, 5, 6, 8]
    _check_row_tracking_log(path)


def test_clone_of_a_row_tracked_table_stays_refused_with_its_reason(
    conn: Any, tmp_path: Any
) -> None:
    table = _row_tracked(conn, str(tmp_path / "rt"), _RT)
    verdict = table.can("clone", target=str(tmp_path / "clone"))
    assert not verdict.ok
    assert "row" in verdict.reason and "commit version" in verdict.reason
