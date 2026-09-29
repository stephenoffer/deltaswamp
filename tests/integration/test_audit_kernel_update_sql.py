"""UPDATE, DELETE and replaceWhere SQL beyond the kernel's grammar, on tables only it writes.

The kernel used to take only literal or column SET values and grammar
predicates, so ``t.update({"n": "n + 1"})`` on an in-commit-timestamp table
was refused without a SQL warehouse. It now evaluates such SQL with DuckDB in
Spark's dialect (as the kernel MERGE does): over the matched rows only, with
Spark's NULL, overflow and division-by-zero semantics (checked against a SQL
warehouse in ANSI mode, Databricks' default). SQL the dialect cannot translate
faithfully, or DuckDB cannot bind, still routes elsewhere, and can() says so.
"""

from __future__ import annotations

from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import deltaswamp as ds  # noqa: E402
from deltaswamp.engine.kernel import KernelEngine  # noqa: E402
from deltaswamp.errors import InvalidArgumentError, UnreachableTableError  # noqa: E402

ICT = {"delta.enableInCommitTimestamps": "true"}
#: Tables delta-rs cannot write, each served by a different kernel commit path.
KERNEL_ONLY = {
    "rewrite": ICT,
    "deletion_vectors": {**ICT, "delta.enableDeletionVectors": "true"},
    "row_tracking": {"delta.enableRowTracking": "true"},
    "column_mapping": {**ICT, "delta.columnMapping.mode": "name"},
}

SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("n", pa.int32()),
        ("b", pa.int8()),
        ("f", pa.float32()),
        ("d", pa.decimal128(10, 2)),
        ("s", pa.string()),
        ("st", pa.struct([("x", pa.int64()), ("y", pa.string())])),
    ]
)
ROWS = {
    "id": [1, 2, 3, 4],
    "n": [10, None, 2147483647, -3],
    "b": [127, 1, None, -128],
    "f": [1.5, None, 0.25, 2.0],
    "d": [None, None, None, None],
    "s": ["a", "Bee", None, "x,y"],
    "st": [{"x": 5, "y": "p"}, None, {"x": None, "y": "q"}, {"x": 7, "y": None}],
}


@pytest.fixture
def conn() -> Any:
    return ds.connect()


def _make(conn: Any, path: str, properties: dict[str, str] | None, **kw: Any) -> Any:
    conn.create_table(path, SCHEMA, properties=properties, **kw)
    t = conn.open_table(path)
    t.append(pa.table({**ROWS, "d": pa.array([None] * 4, pa.decimal128(10, 2))}, schema=SCHEMA))
    return conn.open_table(path)


def _rows(conn: Any, path: str) -> list[dict[str, Any]]:
    return sorted(conn.open_table(path).to_arrow().to_pylist(), key=lambda r: r["id"])


EXPRESSIONS = [
    # (SET values, predicate): each evaluated as Spark does.
    ({"n": "n + 1"}, "id <= 2"),  # NULL + 1 is NULL
    ({"n": "n * 2 - id", "s": "upper(s) || '!'"}, "id IN (1, 2, 4)"),
    ({"s": "CASE WHEN n > 0 THEN 'pos' WHEN n < 0 THEN 'neg' ELSE s END"}, None),
    ({"n": "CAST(f * 3 AS INT)"}, "f IS NOT NULL"),  # truncates: 1.5 * 3 -> 4
    ({"b": "b + 1 - 1"}, "id = 1"),  # INT arithmetic, not TINYINT: 127 stays
    ({"n": "id", "s": "concat(s, CAST(id AS STRING))"}, "id > 1"),
    ({"n": "st.x", "s": "st.y"}, None),  # nested fields
    ({"d": "n / 4"}, "id = 1"),  # a DOUBLE quotient stored into DECIMAL(10,2)
    ({"f": "f + 0.1"}, None),  # FLOAT with a decimal literal is DOUBLE, stored as FLOAT
    ({"s": "coalesce(s, 'none')", "n": "abs(n)"}, "id >= 3"),
]


@pytest.mark.parametrize("kind", sorted(KERNEL_ONLY))
@pytest.mark.parametrize("case", range(len(EXPRESSIONS)))
def test_kernel_update_matches_delta_rs(conn: Any, tmp_path: Any, kind: str, case: int) -> None:
    """The kernel's answer is delta-rs's on a plain table, for SQL both evaluate."""
    updates, predicate = EXPRESSIONS[case]
    kernel_path, plain_path = str(tmp_path / "k"), str(tmp_path / "p")
    kernel = _make(conn, kernel_path, KERNEL_ONLY[kind])
    plain = _make(conn, plain_path, None)
    assert kernel.can("update", updates=updates, predicate=predicate).engine.value == "kernel"
    got = kernel.update(updates, predicate=predicate)
    want = plain.update(updates, predicate=predicate)
    assert got.engine == "kernel" and want.engine == "deltars"
    assert got["num_updated_rows"] == want["num_updated_rows"]
    assert _rows(conn, kernel_path) == _rows(conn, plain_path)


def test_spark_values(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    t = _make(conn, path, KERNEL_ONLY["deletion_vectors"])
    t.update({"n": "n + 1", "b": "b + 1 - 1", "s": "st.y"}, predicate="id <= 2")
    rows = _rows(conn, path)
    assert [(r["n"], r["b"], r["s"]) for r in rows[:2]] == [(11, 127, "p"), (None, 1, None)]
    # Every SET value reads the row as it was: a swap.
    conn.open_table(path).update({"n": "id", "id": "n"}, predicate="id = 4")
    assert (-3, 4) in {(r["id"], r["n"]) for r in _rows(conn, path)}


@pytest.mark.parametrize("kind", sorted(KERNEL_ONLY))
@pytest.mark.parametrize(
    ("updates", "error"),
    [
        # As a SQL warehouse (ANSI mode) raises ARITHMETIC_OVERFLOW,
        # DIVIDE_BY_ZERO and REMAINDER_BY_ZERO.
        ({"n": "n + 1"}, "Overflow"),
        ({"n": "n / 0"}, "DIVIDE_BY_ZERO"),
        ({"n": "n % 0"}, "REMAINDER_BY_ZERO"),
        ({"n": "CAST(s AS INT)"}, "CAST_INVALID_INPUT"),
    ],
)
def test_errors_as_on_databricks(
    conn: Any, tmp_path: Any, kind: str, updates: dict[str, str], error: str
) -> None:
    path = str(tmp_path / "t")
    t = _make(conn, path, KERNEL_ONLY[kind])
    before = _rows(conn, path)
    predicate = "id = 3" if error == "Overflow" else "id = 2" if "CAST" in updates["n"] else None
    with pytest.raises(InvalidArgumentError, match=error):
        t.update(updates, predicate=predicate)
    assert _rows(conn, path) == before


def test_only_matched_rows_are_evaluated(conn: Any, tmp_path: Any) -> None:
    """`10 / n` fails on n = 0 only if that row is updated, as in Spark."""
    for kind in KERNEL_ONLY:
        path = str(tmp_path / kind)
        t = _make(conn, path, KERNEL_ONLY[kind])
        t.update({"n": "0"}, predicate="id = 2")
        conn.open_table(path).update({"f": "10 / n"}, predicate="id = 1")
        assert _rows(conn, path)[0]["f"] == 1.0


def test_storing_out_of_range_is_refused(conn: Any, tmp_path: Any) -> None:
    """INT 128 stored into a TINYINT fails, as ANSI store assignment does."""
    path = str(tmp_path / "t")
    t = _make(conn, path, ICT)
    with pytest.raises(InvalidArgumentError):
        t.update({"b": "b + 1"}, predicate="id = 1")


def test_row_tracking_keeps_row_ids(conn: Any, tmp_path: Any) -> None:
    for kind in ("row_tracking", "row_tracking_dv"):
        props = {"delta.enableRowTracking": "true"}
        if kind == "row_tracking_dv":
            props["delta.enableDeletionVectors"] = "true"
        path = str(tmp_path / kind)
        t = _make(conn, path, props)

        def tracked(path: str = path) -> dict[int, tuple[Any, int, int]]:
            snapshot = KernelEngine().snapshot(conn.open_table(path).resolved)
            rows = pa.table(snapshot.scan(row_positions=True, row_tracking=True))
            return {
                i: (n, rid, cv)
                for i, n, rid, cv in zip(
                    rows.column("id").to_pylist(),
                    rows.column("n").to_pylist(),
                    rows.column("__deltaswamp_row_id").to_pylist(),
                    rows.column("__deltaswamp_row_commit_version").to_pylist(),
                    strict=True,
                )
            }

        before = tracked()
        result = t.update({"n": "n - id"}, predicate="id % 2 = 1")
        assert result.engine == "kernel"
        after = tracked()
        assert (after[1][0], after[3][0]) == (9, 2147483644)
        for i in (1, 3):
            assert after[i][1] == before[i][1] and after[i][2] == result["version"]
        for i in (2, 4):
            assert after[i] == before[i]


def test_partition_column_moves_rows(conn: Any, tmp_path: Any) -> None:
    schema = pa.schema([("id", pa.int64()), ("p", pa.int64())])
    for kind in ("rewrite", "deletion_vectors"):
        path = str(tmp_path / kind)
        conn.create_table(path, schema, partition_by=["p"], properties=KERNEL_ONLY[kind])
        conn.open_table(path).append(pa.table({"id": [1, 2, 3], "p": [0, 0, 1]}, schema=schema))
        t = conn.open_table(path)
        assert t.update({"p": "p + id * 10"}, predicate="p = 0").engine == "kernel"
        rows = conn.open_table(path).to_arrow(predicate="p = 20").to_pylist()
        assert rows == [{"id": 2, "p": 20}]
        assert sorted(r["p"] for r in _rows(conn, path)) == [1, 10, 20]


def test_check_constraints_hold(conn: Any, tmp_path: Any) -> None:
    for kind in ("rewrite", "deletion_vectors"):
        path = str(tmp_path / kind)
        t = _make(conn, path, KERNEL_ONLY[kind])
        t.add_constraint({"small": "id < 10"})
        t = conn.open_table(path)
        before = _rows(conn, path)
        with pytest.raises(Exception, match="small"):
            t.update({"id": "id * 20"}, predicate="id = 1")
        assert _rows(conn, path) == before
        assert conn.open_table(path).update({"id": "id * 5"}, predicate="id = 1").engine == "kernel"


def test_type_widening_table(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    t = _make(conn, path, {**ICT, "delta.enableTypeWidening": "true"})
    assert t.update({"n": "n + id"}, predicate="id = 1").engine == "kernel"
    assert _rows(conn, path)[0]["n"] == 11


@pytest.mark.parametrize("kind", sorted(KERNEL_ONLY))
def test_delete_with_functions(conn: Any, tmp_path: Any, kind: str) -> None:
    kernel_path, plain_path = str(tmp_path / "k"), str(tmp_path / "p")
    kernel = _make(conn, kernel_path, KERNEL_ONLY[kind])
    plain = _make(conn, plain_path, None)
    predicate = "lower(s) = 'bee' OR id % 4 = 0 OR n - 1 > 10"
    assert kernel.can("delete", predicate=predicate).engine.value == "kernel"
    got, want = kernel.delete(predicate), plain.delete(predicate)
    assert got.engine == "kernel" and got["num_deleted_rows"] == want["num_deleted_rows"] == 3
    assert _rows(conn, kernel_path) == _rows(conn, plain_path)


def test_delete_predicate_error_writes_nothing(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    t = _make(conn, path, KERNEL_ONLY["deletion_vectors"])
    version = t.version
    with pytest.raises(InvalidArgumentError, match="DIVIDE_BY_ZERO"):
        t.delete("id / 0 > 1")
    assert conn.open_table(path).version == version


@pytest.mark.parametrize("kind", sorted(KERNEL_ONLY))
def test_replace_where_with_functions(conn: Any, tmp_path: Any, kind: str) -> None:
    path = str(tmp_path / "t")
    t = _make(conn, path, KERNEL_ONLY[kind])
    data = t.to_arrow(predicate="id = 2").to_pylist()[0]
    new = pa.Table.from_pylist([{**data, "id": 6, "s": "six"}], schema=SCHEMA)
    predicate = "id % 2 = 0"
    assert t.can("overwrite", predicate=predicate).engine.value == "kernel"
    t.overwrite(new, predicate=predicate)
    assert [r["id"] for r in _rows(conn, path)] == [1, 3, 6]
    bad = pa.Table.from_pylist([{**data, "id": 7}], schema=SCHEMA)
    with pytest.raises(InvalidArgumentError, match="do not satisfy"):
        conn.open_table(path).overwrite(bad, predicate=predicate)


@pytest.mark.parametrize(
    ("updates", "predicate"),
    [
        # Spark meanings DuckDB cannot be made to compute: the warehouse's.
        ({"s": "split(s, ',')[0]"}, None),
        ({"n": "hash(s)"}, None),
        # Functions DuckDB does not have: not claimed by the kernel.
        ({"n": "try_divide(n, 0)"}, None),
        ({"n": "1"}, "try_divide(n, 0) IS NULL"),
    ],
)
def test_what_the_kernel_leaves_to_the_warehouse(
    conn: Any, tmp_path: Any, updates: dict[str, str], predicate: str | None
) -> None:
    path = str(tmp_path / "t")
    t = _make(conn, path, ICT)
    verdict = t.can("update", updates=updates, predicate=predicate)
    assert not verdict.ok
    with pytest.raises(UnreachableTableError):
        t.update(updates, predicate=predicate)


def test_malformed_sql_is_still_refused(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    t = _make(conn, path, ICT)
    for text in ("n + 1; DROP TABLE x", "n + 1) OR (1 = 1", "n -- comment"):
        with pytest.raises(Exception, match=r"single SQL expression|one expression"):
            t.update({"n": text})
    # No file access from the sandbox, and no subquery: never the kernel's.
    assert not t.can("update", updates={"s": "read_text('/etc/passwd')"}).ok
    with pytest.raises(Exception, match="SELECT"):
        t.update({"s": "(SELECT 1)"})
