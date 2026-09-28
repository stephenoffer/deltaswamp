"""CHECK constraints and schema evolution on the tables only the kernel writes.

delta-rs cannot write a table with in-commit timestamps, clustering, row
tracking, type widening or column defaults, and cannot change a column-mapped
table's schema on write; delta-kernel refuses every write to a table with the
checkConstraints feature and has no schema evolution. So ADD CONSTRAINT and
``schema_mode="merge"`` had no local engine on those tables, and a constraint
added elsewhere left them unwritable. The kernel paths now add a constraint
after checking every existing row, enforce every constraint on every write
(a NULL result passes, a FALSE one fails the write and commits nothing), and
commit a widened schema together with the rows that need it.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import deltaswamp as ds  # noqa: E402
from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.errors import InvalidArgumentError, UnreachableTableError  # noqa: E402

from tests.contract.tables import BY_NAME, base_data  # noqa: E402

if not ds.has_native():  # pragma: no cover
    pytest.skip("native extension not built", allow_module_level=True)

#: Contract fixtures delta-rs cannot write at all.
KERNEL_ONLY = ["ict", "clustered", "type_widening", "defaults", "row_tracking"]
#: Column-mapped fixtures, whose schema delta-rs cannot change on write.
COLUMN_MAPPED = ["cm_name", "cm_id", "legacy_2_5"]


def _build(conn: Any, tmp_path: Any, name: str) -> tuple[str, Any]:
    path = str(tmp_path / name)
    os.makedirs(path)
    BY_NAME[name].build(conn, path)
    return path, conn.open_table(path)


def _commit(path: str, version: int) -> list[dict[str, Any]]:
    with open(os.path.join(path, "_delta_log", f"{version:020}.json")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ids(t: Any) -> list[Any]:
    return sorted(t.to_arrow().column("id").to_pylist(), key=lambda v: (v is None, v))


def _kernel_only(conn: Any) -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.KERNEL: KernelEngine()})
    )


def _deltalake_ids(path: str, column: str = "id") -> list[Any] | None:
    """`column` as delta-rs reads it, or None where it cannot read the table.

    Through its query engine: its pyarrow dataset reads every column of a
    column-mapped table as NULL, whoever wrote it.
    """
    from deltalake import DeltaTable, QueryBuilder

    try:
        table = pa.table(
            QueryBuilder().register("t", DeltaTable(path)).execute(f"SELECT {column} FROM t")
        )
    except Exception as exc:
        if "not yet supported" in str(exc) or "TypeWidening" in str(exc):
            return None  # typeWidening: delta-rs reads none of it
        raise
    return sorted(table.column(0).to_pylist(), key=lambda v: (v is None, v))


# ------------------------------------------------------------ ADD CONSTRAINT


@pytest.mark.parametrize("fixture", KERNEL_ONLY)
def test_add_constraint_on_a_table_only_the_kernel_writes(
    conn: Any, tmp_path: Any, fixture: str
) -> None:
    path, t = _build(conn, tmp_path, fixture)
    verdict = t.can("add_constraint", constraints={"Positive": "id > 0"})
    assert verdict.ok and str(verdict.engine) == "kernel", verdict
    before = t.version
    t.add_constraint({"Positive": "id > 0"})
    t = conn.open_table(path)
    assert t.version == before + 1
    actions = _commit(path, t.version)
    (protocol,) = [a["protocol"] for a in actions if "protocol" in a]
    assert "checkConstraints" in protocol["writerFeatures"]
    (metadata,) = [a["metaData"] for a in actions if "metaData" in a]
    # Stored lower-cased, as Spark stores constraint names.
    assert metadata["configuration"]["delta.constraints.positive"] == "id > 0"
    assert not any("add" in a for a in actions)
    # Both engines still read the table.
    assert _ids(t) == [1, 2, 3, 4, 5, 6]
    theirs = _deltalake_ids(path)
    assert theirs is None or theirs == [1, 2, 3, 4, 5, 6]
    # And the kernel keeps writing it, enforcing the constraint.
    assert t.can("append", data=base_data()).ok
    with pytest.raises(InvalidArgumentError, match="CHECK constraint positive"):
        t.append(pa.table({"id": pa.array([-1], pa.int64())}))
    t.append(pa.table({"id": pa.array([7], pa.int64())}))
    assert _ids(conn.open_table(path)) == [1, 2, 3, 4, 5, 6, 7]


def test_add_constraint_checks_every_existing_row(conn: Any, tmp_path: Any) -> None:
    path, t = _build(conn, tmp_path, "ict")
    before = t.version
    with pytest.raises(InvalidArgumentError, match=r"violated by 3 row\(s\)"):
        t.add_constraint({"big": "id > 3"})
    with pytest.raises(InvalidArgumentError, match="not a boolean expression"):
        t.add_constraint({"sum": "id + 1"})
    with pytest.raises(InvalidArgumentError, match="nope"):
        t.add_constraint({"missing": "nope > 1"})
    assert conn.open_table(path).version == before
    # A NULL result passes, as in Spark: w is NULL in no row, v < 'c' is
    # FALSE in four, but `v < 'c' OR NULL` is NULL there.
    t.add_constraint({"loose": "v < 'c' OR CAST(NULL AS BOOLEAN)"})
    with pytest.raises(InvalidArgumentError, match="already has a constraint"):
        conn.open_table(path).add_constraint({"LOOSE": "id > 0"})


def test_add_constraint_rechecks_rows_a_concurrent_writer_added(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """The check and the commit read one snapshot; a row written between them
    makes the commit lose its put-if-absent, and the retry finds the row."""
    from deltaswamp.engine.kernel import KernelEngine

    path, t = _build(conn, tmp_path, "ict")
    real = KernelEngine._state
    raced = []

    def state(self: Any, table: Any) -> Any:
        out = real(self, table)
        if not raced:
            raced.append(True)
            conn.open_table(path).append(pa.table({"id": pa.array([-5], pa.int64())}))
        return out

    monkeypatch.setattr(KernelEngine, "_state", state)
    with pytest.raises(InvalidArgumentError, match="CHECK constraint positive"):
        t.add_constraint({"positive": "id > 0"})
    t = conn.open_table(path)
    assert "delta.constraints.positive" not in (t.detail().get("properties") or {})
    assert -5 in _ids(t)


def test_a_legacy_protocol_moves_to_writer_version_3(conn: Any, tmp_path: Any) -> None:
    """Below writer 3 the protocol moves to 3, which implies checkConstraints."""
    path, _t = _build(conn, tmp_path, "legacy_1_2")
    kernel = _kernel_only(conn).open_table(path)
    kernel.add_constraint({"positive": "id > 0"})
    (protocol,) = [a["protocol"] for a in _commit(path, 2) if "protocol" in a]
    assert protocol == {"minReaderVersion": 1, "minWriterVersion": 3}
    # delta-rs enforces what the kernel added.
    with pytest.raises(InvalidArgumentError):
        conn.open_table(path).append(pa.table({"id": pa.array([-1], pa.int64())}))


def test_drop_constraint_on_a_table_only_the_kernel_writes(conn: Any, tmp_path: Any) -> None:
    path, t = _build(conn, tmp_path, "clustered")
    t.add_constraint({"positive": "id > 0"})
    t = conn.open_table(path)
    verdict = t.can("drop_constraint", name="positive")
    assert verdict.ok and str(verdict.engine) == "kernel", verdict
    t.drop_constraint("positive")
    t = conn.open_table(path)
    t.append(pa.table({"id": pa.array([-1], pa.int64())}))
    assert -1 in _ids(t)


# ------------------------------------------------------------- enforcement


def _constrained(conn: Any, tmp_path: Any, properties: dict[str, str] | None = None) -> Any:
    path = str(tmp_path / "t")
    t = conn.create_table(
        path,
        base_data().schema,
        properties={"delta.enableInCommitTimestamps": "true", **(properties or {})},
    )
    t.append(base_data())
    conn.open_table(path).add_constraint({"positive": "id > 0 AND w < 100"})
    return path, conn.open_table(path)


def _refused_and_unchanged(conn: Any, path: str, call: Any) -> None:
    before = conn.open_table(path)
    with pytest.raises(InvalidArgumentError, match="CHECK constraint positive"):
        call(before)
    after = conn.open_table(path)
    assert after.version == before.version
    assert _ids(after) == _ids(before)


def test_every_kernel_write_enforces_the_constraints(conn: Any, tmp_path: Any) -> None:
    path, t = _constrained(conn, tmp_path)

    def one(i: int, w: int | None = 1) -> Any:
        return pa.table({"id": pa.array([i], pa.int64()), "w": pa.array([w], pa.int32())})

    _refused_and_unchanged(conn, path, lambda t: t.append(one(-1)))
    _refused_and_unchanged(conn, path, lambda t: t.append(one(1, 500)))
    _refused_and_unchanged(conn, path, lambda t: t.overwrite(one(-3)))
    _refused_and_unchanged(conn, path, lambda t: t.overwrite(one(-2), predicate="id = -2"))
    _refused_and_unchanged(conn, path, lambda t: t.update({"id": "-5"}, predicate="id = 1"))
    widened = one(-1).append_column("z", pa.array([1]))
    _refused_and_unchanged(conn, path, lambda t: t.append(widened, schema_mode="merge"))
    # NULL passes: `NULL > 0 AND ...` is NULL.
    t.append(one(8, None))
    t.append(pa.table({"id": pa.array([None], pa.int64())}))
    t = conn.open_table(path)
    t.update({"id": "50"}, predicate="id = 1")
    t = conn.open_table(path)
    t.overwrite(one(9), predicate="id = 9")
    assert _ids(conn.open_table(path)) == [2, 3, 4, 5, 6, 8, 9, 50, None]
    assert _deltalake_ids(path) == [2, 3, 4, 5, 6, 8, 9, 50, None]
    t = conn.open_table(path)
    t.overwrite(one(4))
    assert _ids(conn.open_table(path)) == [4]


def test_deletion_vector_dml_enforces_the_constraints(conn: Any, tmp_path: Any) -> None:
    path, t = _constrained(conn, tmp_path, {"delta.enableDeletionVectors": "true"})
    _refused_and_unchanged(conn, path, lambda t: t.update({"id": "-5"}, predicate="id = 1"))
    source = pa.table(
        {
            "id": pa.array([2, -7], pa.int64()),
            "v": ["b", "q"],
            "grp": ["x", "q"],
            "w": pa.array([1, 1], pa.int32()),
        }
    )

    def merge(t: Any, source: Any = source) -> Any:
        return (
            t.merge(source, "target.id = source.id")
            .when_matched_update({"w": "source.w + 200"})
            .when_not_matched_insert_all()
            .execute()
        )

    _refused_and_unchanged(conn, path, merge)
    t = conn.open_table(path)
    assert t.can("merge").engine == Engine.KERNEL
    t.merge(source.slice(0, 1), "target.id = source.id").when_matched_update(
        {"w": "source.w + 5"}
    ).execute()
    t = conn.open_table(path)
    t.delete("id = 3")
    rows = {r["id"]: r["w"] for r in conn.open_table(path).to_arrow().to_pylist()}
    assert rows == {1: 1, 2: 6, 4: 4, 5: 5, 6: 6}


def test_a_distributed_write_enforces_the_constraints(conn: Any, tmp_path: Any) -> None:
    path, t = _constrained(conn, tmp_path)
    plan = t.plan_write()
    with pytest.raises(InvalidArgumentError, match="CHECK constraint positive"):
        plan.write(pa.table({"id": pa.array([-1], pa.int64())}))
    fragment = plan.write(pa.table({"id": pa.array([10], pa.int64())}))
    plan.commit([fragment])
    assert 10 in _ids(conn.open_table(path))


def test_a_constraint_added_elsewhere_is_enforced(conn: Any, tmp_path: Any) -> None:
    """A writer-version-3 table with a constraint (as Spark leaves one) takes
    kernel writes now, checked, where the kernel refused the table outright."""
    from tests.contract.tables import BASE_FIELDS, write_log

    path = str(tmp_path / "t")
    write_log(
        path,
        BASE_FIELDS,
        {"minReaderVersion": 1, "minWriterVersion": 3},
        base_data(),
        {"delta.constraints.id_positive": "id > 0"},
    )
    kernel = _kernel_only(conn).open_table(path)
    assert kernel.can("append").ok
    with pytest.raises(InvalidArgumentError, match="CHECK constraint id_positive"):
        kernel.append(pa.table({"id": pa.array([0], pa.int64())}))
    kernel.append(pa.table({"id": pa.array([7], pa.int64())}))
    (protocol,) = [a for a in _commit(path, 0) if "protocol" in a]
    assert protocol["protocol"]["minWriterVersion"] == 3  # the table's own, untouched
    assert not any("protocol" in a for a in _commit(path, 2))
    assert _deltalake_ids(path) == [1, 2, 3, 4, 5, 6, 7]


def test_a_constraint_duckdb_cannot_evaluate_is_refused_before_writing(
    conn: Any, tmp_path: Any
) -> None:
    from tests.contract.tables import BASE_FIELDS, write_log

    path = str(tmp_path / "t")
    write_log(
        path,
        BASE_FIELDS,
        {"minReaderVersion": 1, "minWriterVersion": 7, "writerFeatures": ["checkConstraints"]},
        base_data(),
        {"delta.constraints.h": "hash(id) != 0"},
    )
    kernel = _kernel_only(conn).open_table(path)
    verdict = kernel.can("append")
    assert not verdict.ok and "hash" in verdict.reason, verdict
    with pytest.raises(UnreachableTableError):
        kernel.append(pa.table({"id": pa.array([7], pa.int64())}))
    assert _ids(kernel) == [1, 2, 3, 4, 5, 6]


# ------------------------------------------------------------ schema evolution


def _field(schema_string: str, name: str) -> dict[str, Any]:
    fields = json.loads(schema_string)["fields"]
    found: dict[str, Any] = next(f for f in fields if f["name"] == name)
    return found


@pytest.mark.parametrize("fixture", KERNEL_ONLY + COLUMN_MAPPED)
def test_merge_schema_commits_the_new_column_with_its_rows(
    conn: Any, tmp_path: Any, fixture: str
) -> None:
    path, t = _build(conn, tmp_path, fixture)
    data = t.to_arrow().append_column("extra", pa.array(range(6), pa.int64()))
    verdict = t.can("merge_schema", data=data, schema_mode="merge")
    assert verdict.ok and str(verdict.engine) == "kernel", verdict
    before = t.version
    t.append(data, schema_mode="merge")
    t = conn.open_table(path)
    # One commit: the new metaData and the rows it describes, not blind.
    assert t.version == before + 1
    actions = _commit(path, t.version)
    (info,) = [a["commitInfo"] for a in actions if "commitInfo" in a]
    assert info.get("isBlindAppend") is False
    (metadata,) = [a["metaData"] for a in actions if "metaData" in a]
    assert any("add" in a for a in actions)
    extra = _field(metadata["schemaString"], "extra")
    assert extra["nullable"] is True
    if fixture in COLUMN_MAPPED:
        ids = [
            f["metadata"]["delta.columnMapping.id"]
            for f in json.loads(metadata["schemaString"])["fields"]
        ]
        assert extra["metadata"]["delta.columnMapping.id"] == max(ids) == len(ids)
        assert extra["metadata"]["delta.columnMapping.physicalName"].startswith("col-")
        assert metadata["configuration"]["delta.columnMapping.maxColumnId"] == str(len(ids))
    rows = t.to_arrow()
    assert rows.num_rows == 12
    assert sorted(v for v in rows.column("extra").to_pylist() if v is not None) == list(range(6))
    assert rows.column("extra").null_count == 6
    theirs = _deltalake_ids(path, "extra")
    assert theirs is None or theirs == [*range(6), *[None] * 6]


def test_merge_schema_adds_nested_fields(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    s = pa.struct([("a", pa.int64())])
    t = conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("s", s), ("l", pa.list_(s))]),
        # A (2, 5) protocol, which delta-rs reads (it reads no column mapping
        # listed as a table feature) but cannot change the schema of.
        properties={"delta.columnMapping.mode": "name"},
    )
    t.append(
        pa.table(
            {
                "id": [1],
                "s": pa.array([{"a": 1}], s),
                "l": pa.array([[{"a": 1}]], pa.list_(s)),
            }
        )
    )
    wide = pa.struct([("a", pa.int64()), ("b", pa.string())])
    conn.open_table(path).append(
        pa.table(
            {
                "id": [2],
                "s": pa.array([{"a": 2, "b": "x"}], wide),
                "l": pa.array([[{"a": 2, "b": "y"}]], pa.list_(wide)),
            }
        ),
        schema_mode="merge",
    )
    t = conn.open_table(path)
    rows = sorted(t.to_arrow().to_pylist(), key=lambda r: r["id"])
    assert rows == [
        {"id": 1, "s": {"a": 1, "b": None}, "l": [{"a": 1, "b": None}]},
        {"id": 2, "s": {"a": 2, "b": "x"}, "l": [{"a": 2, "b": "y"}]},
    ]
    (metadata,) = [a["metaData"] for a in _commit(path, t.version) if "metaData" in a]
    s_field = _field(metadata["schemaString"], "s")
    b = next(f for f in s_field["type"]["fields"] if f["name"] == "b")
    assert b["metadata"]["delta.columnMapping.physicalName"].startswith("col-")
    element = _field(metadata["schemaString"], "l")["type"]["elementType"]
    ids = {f["metadata"]["delta.columnMapping.id"] for f in element["fields"]}
    assert len(ids) == 2
    assert _deltalake_ids(path) == [1, 2]
    assert _deltalake_ids(path, "s['b']") == ["x", None]


def test_overwrite_with_merge_schema(conn: Any, tmp_path: Any) -> None:
    path, t = _build(conn, tmp_path, "ict")
    data = pa.table({"id": pa.array([9], pa.int64()), "extra": ["x"]})
    verdict = t.can("overwrite", data=data, schema_mode="merge")
    assert verdict.ok and str(verdict.engine) == "kernel", verdict
    t.overwrite(data, schema_mode="merge")
    t = conn.open_table(path)
    assert t.to_arrow().to_pylist() == [{"id": 9, "v": None, "grp": None, "w": None, "extra": "x"}]
    assert "extra" in t.schema().names


def test_merge_schema_widens_a_type_only_under_type_widening(conn: Any, tmp_path: Any) -> None:
    wider = pa.table({"id": pa.array([7], pa.int64()), "w": pa.array([2**40], pa.int64())})
    path, t = _build(conn, tmp_path, "ict")
    with pytest.raises(UnreachableTableError, match="type widening"):
        t.append(wider, schema_mode="merge")
    path, t = _build(conn, tmp_path, "type_widening")
    t.append(wider, schema_mode="merge")
    t = conn.open_table(path)
    assert t.schema().field("w").type == pa.int64()
    assert 2**40 in t.to_arrow().column("w").to_pylist()
    (metadata,) = [a["metaData"] for a in _commit(path, t.version) if "metaData" in a]
    changes = _field(metadata["schemaString"], "w")["metadata"]["delta.typeChanges"]
    assert changes == [{"fromType": "integer", "toType": "long"}]
    # A narrower type is the table's own, cast as it is written.
    t.append(pa.table({"id": pa.array([8], pa.int64()), "w": pa.array([1], pa.int8())}))


def test_merge_schema_that_changes_nothing_is_a_plain_append(conn: Any, tmp_path: Any) -> None:
    path, t = _build(conn, tmp_path, "ict")
    t.append(base_data(), schema_mode="merge")
    actions = _commit(path, conn.open_table(path).version)
    assert not any("metaData" in a for a in actions)


def test_merge_schema_keeps_a_new_column_of_a_constrained_table_checked(
    conn: Any, tmp_path: Any
) -> None:
    path, t = _constrained(conn, tmp_path)
    t.append(
        pa.table({"id": pa.array([10], pa.int64()), "note": ["n"]}),
        schema_mode="merge",
    )
    t = conn.open_table(path)
    assert "note" in t.schema().names
    with pytest.raises(InvalidArgumentError, match="CHECK constraint positive"):
        t.append(pa.table({"id": pa.array([-1], pa.int64()), "note": ["m"]}))
