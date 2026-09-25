"""DELETE, UPDATE and replaceWhere written as deletion vectors, as Databricks does.

Every result is checked twice: through the kernel, and through delta-rs, whose
deletion-vector reader is an independent implementation. The golden table was
written by Databricks, so its existing vector exercises the union with a
vector this library did not write.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

import pytest
from deltaswamp.capability import Engine, Operation
from deltaswamp.errors import UnreachableTableError

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")
native = pytest.importorskip("deltaswamp._native")

pytestmark = pytest.mark.skipif(
    "deletion_vector_dml" not in native.FEATURES, reason="native build predates DV DML"
)

GOLDEN = pathlib.Path(__file__).parents[1] / "data" / "golden" / "table-with-dv-small"


def _values(conn: Any, path: str, column: str = "value") -> list[Any]:
    return sorted(conn.open_table(path).to_arrow().column(column).to_pylist())


def _deltars_rows(path: str, column: str = "value") -> list[Any]:
    # to_pyarrow_dataset() ignores deletion vectors; QueryBuilder applies them.
    from deltalake import QueryBuilder

    table = deltalake.DeltaTable(path)
    result = QueryBuilder().register("t", table).execute(f"SELECT {column} FROM t").read_all()
    return sorted(pa.table(result).column(column).to_pylist())


def _last_commit(path: str) -> list[dict[str, Any]]:
    logs = sorted(pathlib.Path(path, "_delta_log").glob("*.json"))
    return [json.loads(line) for line in logs[-1].read_text().splitlines()]


def _dv_files(path: str) -> list[pathlib.Path]:
    return sorted(pathlib.Path(path).rglob("deletion_vector_*.bin"))


@pytest.fixture
def golden(tmp_path: Any) -> str:
    """A copy of a Databricks-written table whose one file has a vector (rows 0, 9)."""
    target = tmp_path / "golden"
    shutil.copytree(GOLDEN, target)
    return str(target)


def _dv_table(
    conn: Any, tmp_path: Any, *, properties: dict[str, str] | None = None, **kw: Any
) -> str:
    path = str(tmp_path / "t")
    props = {"delta.enableDeletionVectors": "true", **(properties or {})}
    conn.create_table(
        path, pa.schema([("id", pa.int64()), ("city", pa.string())]), properties=props, **kw
    )
    table = conn.open_table(path)
    table.append(pa.table({"id": [1, 2, 3, 4], "city": ["oslo", "lima", "cairo", "rome"]}))
    table = conn.open_table(path)
    table.append(pa.table({"id": [5, 6], "city": ["paris", "tokyo"]}))
    return path


class TestDelete:
    def test_routes_to_the_kernel_ahead_of_delta_rs(self, conn: Any, golden: str) -> None:
        for op in (Operation.DELETE, Operation.UPDATE, Operation.REPLACE_WHERE):
            assert conn.open_table(golden).can(op).engine is Engine.KERNEL

    def test_unions_with_a_vector_databricks_wrote(self, conn: Any, golden: str) -> None:
        result = conn.open_table(golden).delete("value = 3 OR value = 4")
        assert result["num_deleted_rows"] == 2
        assert _values(conn, golden) == [1, 2, 5, 6, 7, 8]
        assert _deltars_rows(golden) == [1, 2, 5, 6, 7, 8]

        actions = _last_commit(golden)
        add = next(a["add"] for a in actions if "add" in a)
        remove = next(a["remove"] for a in actions if "remove" in a)
        assert add["path"] == remove["path"]
        assert add["deletionVector"]["cardinality"] == 4  # 0, 9 and the two new rows
        assert add["deletionVector"]["storageType"] == "u"
        assert remove["deletionVector"]["cardinality"] == 2
        assert json.loads(add["stats"])["tightBounds"] is False
        # The data file is untouched; only a vector was written.
        assert len(list(pathlib.Path(golden).glob("*.parquet"))) == 1
        assert len(_dv_files(golden)) == 2

    def test_deleting_already_deleted_rows_commits_nothing(self, conn: Any, golden: str) -> None:
        before = conn.open_table(golden).version
        assert conn.open_table(golden).delete("value = 0")["num_deleted_rows"] == 0
        assert conn.open_table(golden).version == before

    def test_a_file_left_empty_is_removed(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        assert conn.open_table(path).delete("id >= 5")["num_deleted_rows"] == 2
        actions = _last_commit(path)
        assert any("remove" in a for a in actions)
        assert not any("add" in a for a in actions), "no vector for a file with no rows left"
        assert _values(conn, path, "id") == [1, 2, 3, 4]
        assert _deltars_rows(path, "id") == [1, 2, 3, 4]

    def test_delete_without_a_predicate_empties_the_table(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        assert conn.open_table(path).delete()["num_deleted_rows"] == 6
        assert conn.open_table(path).to_arrow().num_rows == 0

    def test_null_predicate_results_keep_the_row(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        table = conn.open_table(path)
        table.append(pa.table({"id": [7], "city": pa.array([None], pa.string())}))
        assert conn.open_table(path).delete("city = 'oslo'")["num_deleted_rows"] == 1
        assert _values(conn, path, "id") == [2, 3, 4, 5, 6, 7]

    def test_repeated_deletes_accumulate(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        conn.open_table(path).delete("id = 1")
        conn.open_table(path).delete("id = 3")
        conn.open_table(path).delete("id = 6")
        assert _values(conn, path, "id") == [2, 4, 5]
        assert _deltars_rows(path, "id") == [2, 4, 5]

    def test_partitioned(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "p")
        conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("grp", pa.string())]),
            partition_by=["grp"],
            properties={"delta.enableDeletionVectors": "true"},
        )
        conn.open_table(path).append(pa.table({"id": [1, 2, 3, 4], "grp": ["a", "a", "b", "b"]}))
        assert conn.open_table(path).delete("id IN (1, 3)")["num_deleted_rows"] == 2
        assert _values(conn, path, "id") == [2, 4]
        assert _deltars_rows(path, "id") == [2, 4]

    def test_change_feed_reports_the_deleted_rows(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path, properties={"delta.enableChangeDataFeed": "true"})
        table = conn.open_table(path)
        assert table.can(Operation.DELETE).ok
        version = table.delete("id = 2")["version"]
        changes = pa.table(conn.open_table(path).cdf(starting_version=version))
        rows = changes.to_pylist()
        assert [(r["id"], r["_change_type"]) for r in rows] == [(2, "delete")]


class TestUpdate:
    def test_updates_through_a_vector_and_a_new_file(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        result = conn.open_table(path).update(new_values={"city": "LIMA"}, predicate="id = 2")
        assert result["num_updated_rows"] == 1
        rows = sorted(conn.open_table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
        assert rows[1] == {"id": 2, "city": "LIMA"}
        assert len(rows) == 6
        assert _deltars_rows(path, "city").count("LIMA") == 1
        actions = _last_commit(path)
        adds = [a["add"] for a in actions if "add" in a]
        assert sum(1 for a in adds if a.get("deletionVector")) == 1
        assert sum(1 for a in adds if not a.get("deletionVector")) == 1

    def test_update_from_another_column(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        conn.open_table(path).update(updates={"city": "'x'"}, predicate="id > 4")
        cities = _values(conn, path, "city")
        assert cities.count("x") == 2

    def test_replace_where(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        conn.open_table(path).overwrite(
            pa.table({"id": [3, 30], "city": ["c", "cc"]}), predicate="id >= 3 AND id <= 30"
        )
        assert _values(conn, path, "id") == [1, 2, 3, 30]
        with pytest.raises(UnreachableTableError, match="do not satisfy the predicate"):
            conn.open_table(path).overwrite(
                pa.table({"id": [99], "city": ["z"]}), predicate="id < 10"
            )


class TestRowTracking:
    @pytest.fixture
    def tracked(self, conn: Any, tmp_path: Any) -> str:
        return _dv_table(conn, tmp_path, properties={"delta.enableRowTracking": "true"})

    @staticmethod
    def _row_ids(conn: Any, path: str) -> dict[int, int]:
        snapshot = conn.router.engines[Engine.KERNEL].snapshot(conn.open_table(path).resolved)
        table = pa.table(snapshot.scan(row_positions=True, row_ids=True))
        return dict(
            zip(
                table.column("id").to_pylist(),
                table.column("__deltaswamp_row_id").to_pylist(),
                strict=True,
            )
        )

    def test_delete_keeps_every_surviving_row_id(self, conn: Any, tracked: str) -> None:
        before = self._row_ids(conn, tracked)
        conn.open_table(tracked).delete("id IN (2, 5, 6)")
        after = self._row_ids(conn, tracked)
        assert after == {k: v for k, v in before.items() if k not in (2, 5, 6)}
        # Row tracking forbids removes here, so an emptied file keeps a full vector.
        actions = _last_commit(tracked)
        removed = {a["remove"]["path"] for a in actions if "remove" in a}
        added = {a["add"]["path"] for a in actions if "add" in a}
        assert removed and removed <= added

    def test_update_keeps_the_updated_row_id(self, conn: Any, tracked: str) -> None:
        properties = conn.open_table(tracked).properties()
        if "delta.rowTracking.materializedRowIdColumnName" not in properties:
            assert not conn.open_table(tracked).can(Operation.UPDATE).ok
            pytest.skip("the kernel create names no materialized row-id column")
        before = self._row_ids(conn, tracked)
        conn.open_table(tracked).update(new_values={"city": "LIMA"}, predicate="id = 2")
        assert self._row_ids(conn, tracked) == before


class TestSafety:
    def test_a_stale_snapshot_conflicts(self, conn: Any, tmp_path: Any) -> None:
        path = _dv_table(conn, tmp_path)
        kernel = conn.router.engines[Engine.KERNEL]
        stale = kernel.snapshot(conn.open_table(path).resolved, write=True)
        positions = pa.table(stale.scan(row_positions=True))
        conn.open_table(path).delete("id = 1")  # a concurrent writer
        deletions = pa.table(
            {
                "path": positions.column("__deltaswamp_file").slice(0, 1),
                "row_index": positions.column("__deltaswamp_row_index").slice(0, 1),
            }
        )
        with pytest.raises(Exception, match=r"(?i)conflict|already exists|committed version"):
            stale.commit_dml(deletions.to_reader(), operation="DELETE")
        assert _values(conn, path, "id") == [2, 3, 4, 5, 6]

    def test_unknown_files_and_out_of_range_rows_are_refused(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = _dv_table(conn, tmp_path)
        kernel = conn.router.engines[Engine.KERNEL]
        snapshot = kernel.snapshot(conn.open_table(path).resolved, write=True)
        live = pa.table(snapshot.files()).column("path")[0].as_py()
        for bad_path, row, match in [
            ("nope.parquet", 0, "not live"),
            (live, 10_000, "out of range"),
        ]:
            deletions = pa.table({"path": [bad_path], "row_index": pa.array([row], pa.int64())})
            with pytest.raises(Exception, match=match):
                snapshot.commit_dml(deletions.to_reader(), operation="DELETE")
        assert _dv_files(path) == [], "nothing written for refused input"

    def test_tables_without_deletion_vectors_keep_copy_on_write(
        self, conn: Any, tmp_path: Any
    ) -> None:
        from deltalake import write_deltalake

        path = str(tmp_path / "plain")
        write_deltalake(path, pa.table({"id": [1, 2, 3]}))
        assert conn.open_table(path).can(Operation.DELETE).engine is Engine.DELTARS
        conn.open_table(path).delete("id = 2")
        assert _dv_files(path) == []


class TestMerge:
    @pytest.fixture(autouse=True)
    def _duckdb(self) -> None:
        pytest.importorskip("duckdb")

    @pytest.fixture
    def target(self, conn: Any, tmp_path: Any) -> str:
        path = str(tmp_path / "m")
        conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("qty", pa.int64()), ("grp", pa.string())]),
            partition_by=["grp"],
            properties={
                "delta.enableDeletionVectors": "true",
                "delta.enableRowTracking": "true",
            },
        )
        conn.open_table(path).append(
            pa.table({"id": [1, 2, 3, 4], "qty": [10, 20, 30, 40], "grp": ["a", "a", "b", "b"]})
        )
        return path

    def _rows(self, conn: Any, path: str) -> list[tuple[Any, ...]]:
        return sorted(tuple(r.values()) for r in conn.open_table(path).to_arrow().to_pylist())

    def test_every_clause_kind(self, conn: Any, target: str) -> None:
        table = conn.open_table(target)
        assert table.can(Operation.MERGE).engine is Engine.KERNEL
        source = pa.table(
            {
                "id": [2, 3, 5, 6],
                "qty": [1, 2, 3, 4],
                "grp": ["a", "b", "a", "b"],
                "op": ["upd", "del", "ins", "skip"],
            }
        )
        before = TestRowTracking._row_ids(conn, target)
        result = (
            table.merge(source, "t.id = s.id", source_alias="s", target_alias="t")
            .when_matched_delete(predicate="s.op = 'del'")
            .when_matched_update({"qty": "t.qty + s.qty"})
            .when_not_matched_insert_all(predicate="s.op = 'ins'")
            .when_not_matched_by_source_update({"qty": "t.qty * 100"}, predicate="t.id = 4")
            .execute()
        )
        assert result["num_target_rows_updated"] == 2
        assert result["num_target_rows_deleted"] == 1
        assert result["num_target_rows_inserted"] == 1
        assert self._rows(conn, target) == [(1, 10, "a"), (2, 21, "a"), (4, 4000, "b"), (5, 3, "a")]
        assert _deltars_rows(target, "qty") == [3, 10, 21, 4000]
        after = TestRowTracking._row_ids(conn, target)
        for key in (1, 2, 4):
            assert after[key] == before[key], "updated and untouched rows keep their ids"
        assert after[5] not in before.values(), "an inserted row gets a fresh id"

    def test_a_target_row_matched_twice_is_refused(self, conn: Any, target: str) -> None:
        source = pa.table({"id": [1, 1], "qty": [5, 6], "grp": ["a", "a"]})
        with pytest.raises(Exception, match="more than one source row"):
            (
                conn.open_table(target)
                .merge(source, "t.id = s.id", source_alias="s", target_alias="t")
                .when_matched_update_all()
                .execute()
            )
        assert self._rows(conn, target)[0] == (1, 10, "a")

    def test_a_merge_that_changes_nothing_commits_nothing(self, conn: Any, target: str) -> None:
        version = conn.open_table(target).version
        source = pa.table({"id": [99], "qty": [1], "grp": ["a"]})
        result = (
            conn.open_table(target)
            .merge(source, "t.id = s.id", source_alias="s", target_alias="t")
            .when_matched_delete()
            .execute()
        )
        assert result["num_target_rows_deleted"] == 0
        assert conn.open_table(target).version == version

    def test_skipping_by_join_keys_reads_only_matching_files(self, conn: Any, target: str) -> None:
        from deltaswamp.engine.kernel_merge import KernelMerger

        kernel = conn.router.engines[Engine.KERNEL]
        resolved = conn.open_table(target).resolved
        merger = KernelMerger(
            kernel,
            resolved,
            pa.table({"id": [3]}),
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        )
        schema = pa.schema(kernel.snapshot(resolved).schema())
        skipping = merger._skipping(schema)
        assert skipping is not None and '"id"' in skipping

    def test_tables_without_deletion_vectors_are_not_merged_here(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "plain")
        conn.create_table(path, pa.schema([("id", pa.int64())]))
        verdict = conn.router.engines[Engine.KERNEL].supports(
            Operation.MERGE, conn.open_table(path).resolved
        )
        assert not verdict.ok and "deletion vectors" in verdict.reason


def _z85(data: bytes) -> str:
    alphabet = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
    )
    out = []
    for i in range(0, len(data), 4):
        value = int.from_bytes(data[i : i + 4], "big")
        chunk = []
        for _ in range(5):
            value, rem = divmod(value, 85)
            chunk.append(alphabet[rem])
        out.append("".join(reversed(chunk)))
    return "".join(out)


class TestExistingVectorsAndFiles:
    def test_unions_with_an_inline_vector(self, conn: Any, golden: str) -> None:
        """Rewrite the golden table's persisted vector as an inline one ('i')."""
        root = pathlib.Path(golden)
        blob = (root / "deletion_vector_61d16c75-6994-46b7-a15b-8b538852e50e.bin").read_bytes()
        size = int.from_bytes(blob[1:5], "big")
        payload = blob[5 : 5 + size]  # magic + bitmap
        padded = payload + b"\0" * (-len(payload) % 4)
        log = root / "_delta_log" / "00000000000000000001.json"
        actions = [json.loads(line) for line in log.read_text().splitlines()]
        for action in actions:
            if "add" in action:
                action["add"]["deletionVector"] = {
                    "storageType": "i",
                    "pathOrInlineDv": _z85(padded),
                    "sizeInBytes": size,
                    "cardinality": 2,
                }
        log.write_text("\n".join(json.dumps(a) for a in actions) + "\n")
        (root / "deletion_vector_61d16c75-6994-46b7-a15b-8b538852e50e.bin").unlink()
        assert _values(conn, golden) == list(range(1, 9)), "the inline vector reads"

        conn.open_table(golden).delete("value = 5")
        assert _values(conn, golden) == [1, 2, 3, 4, 6, 7, 8]
        assert _deltars_rows(golden) == [1, 2, 3, 4, 6, 7, 8]
        add = next(a["add"] for a in _last_commit(golden) if "add" in a)
        assert add["deletionVector"]["storageType"] == "u"
        assert add["deletionVector"]["cardinality"] == 3

    def test_files_without_statistics_are_rewritten(self, conn: Any, tmp_path: Any) -> None:
        from deltalake import write_deltalake

        path = str(tmp_path / "nostats")
        write_deltalake(path, pa.table({"id": pa.array([1, 2, 3, 4], pa.int64())}))
        conn.open_table(path).set_properties({"delta.enableDeletionVectors": "true"})
        log = pathlib.Path(path, "_delta_log", "00000000000000000000.json")
        actions = [json.loads(line) for line in log.read_text().splitlines()]
        for action in actions:
            if "add" in action:
                action["add"].pop("stats", None)
        log.write_text("\n".join(json.dumps(a) for a in actions) + "\n")

        assert conn.open_table(path).delete("id = 2")["num_deleted_rows"] == 1
        assert _values(conn, path, "id") == [1, 3, 4]
        assert _deltars_rows(path, "id") == [1, 3, 4]
        actions = _last_commit(path)
        assert any("remove" in a for a in actions)
        assert all(not a["add"].get("deletionVector") for a in actions if "add" in a)

    def test_randomized_prefixes_put_the_vector_in_a_subdirectory(
        self, conn: Any, tmp_path: Any
    ) -> None:
        # The kernel refuses this key at CREATE, and delta-rs cannot create a
        # DV table cleanly, so it is set afterwards, as a metadata commit.
        path = _dv_table(conn, tmp_path)
        conn.open_table(path).set_properties({"delta.randomizeFilePrefixes": "true"})
        conn.open_table(path).delete("id = 1")
        (dv,) = _dv_files(path)
        assert dv.parent != pathlib.Path(path), "the vector sits under its prefix"
        add = next(a["add"] for a in _last_commit(path) if "add" in a)
        assert add["deletionVector"]["pathOrInlineDv"].startswith(dv.parent.name)
        assert _values(conn, path, "id") == [2, 3, 4, 5, 6]
        assert _deltars_rows(path, "id") == [2, 3, 4, 5, 6]

    def test_delete_without_a_predicate_removes_files_unread(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = _dv_table(conn, tmp_path)
        conn.open_table(path).delete("id = 1")  # one file already carries a vector
        assert conn.open_table(path).delete()["num_deleted_rows"] == 5
        actions = _last_commit(path)
        assert sum(1 for a in actions if "remove" in a) == 2
        assert not any("add" in a for a in actions)

    def test_delete_without_a_predicate_on_a_row_tracked_table(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = _dv_table(conn, tmp_path, properties={"delta.enableRowTracking": "true"})
        assert conn.open_table(path).delete()["num_deleted_rows"] == 6
        assert conn.open_table(path).to_arrow().num_rows == 0
        assert _deltars_rows(path, "id") == []
        adds = [a["add"] for a in _last_commit(path) if "add" in a]
        assert adds and all(
            a["deletionVector"]["cardinality"] == json.loads(a["stats"])["numRecords"] for a in adds
        )
