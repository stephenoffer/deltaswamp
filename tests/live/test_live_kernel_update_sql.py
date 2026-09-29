"""Kernel UPDATE/DELETE with Spark SQL expressions, checked against Databricks.

Each test builds a table only the kernel writes (in-commit timestamps, row
tracking, deletion vectors), copies it, runs DML with computed SET values and
function predicates through deltaswamp (the kernel serves it, evaluating the
SQL with DuckDB in Spark's dialect), and runs the same statements on the copy
through a SQL warehouse. Both are uploaded to a scratch UC volume and read by
path: the rows, and on a row-tracked table `_metadata.row_id` and
`_metadata.row_commit_version`, must agree. So must the failures: an UPDATE
that overflows INT fails on both and changes nothing.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_kernel_update_sql.py -v

The volume (named a8up_...) is created in the target schema and dropped
afterwards.
"""

from __future__ import annotations

import shutil
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import deltaswamp as ds  # noqa: E402
from deltaswamp.errors import InvalidArgumentError  # noqa: E402

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = pytest.mark.databricks


class _UpdateVolume(_Volume):
    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"a8up_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _UpdateVolume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("n", pa.int32()),
        ("b", pa.int8()),
        ("s", pa.string()),
        ("st", pa.struct([("x", pa.int64()), ("y", pa.string())])),
    ]
)
ROWS = pa.table(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "n": [10, None, 2147483647, -3, 0, 7],
        "b": [127, 1, None, -128, 5, 0],
        "s": ["a", "Bee", None, "x,y", "e", "f"],
        "st": [
            {"x": 5, "y": "p"},
            None,
            {"x": None, "y": "q"},
            {"x": 7, "y": None},
            {"x": 1, "y": "r"},
            {"x": 2, "y": "s"},
        ],
    },
    schema=SCHEMA,
)

#: (kind, SET values or None for a DELETE, predicate): run by both sides.
STATEMENTS: list[tuple[dict[str, str] | None, str | None]] = [
    ({"n": "n + 1", "s": "upper(s) || '!'"}, "id % 2 = 1 AND id <> 3"),
    ({"s": "CASE WHEN n > 0 THEN 'pos' WHEN n < 0 THEN 'neg' ELSE s END", "b": "b + 1 - 1"}, None),
    ({"n": "st.x * 10 + id", "s": "concat(st.y, CAST(id AS STRING))"}, "st.y IS NOT NULL"),
    ({"n": "CAST(n / 4 AS INT)"}, "n IS NOT NULL AND abs(n) < 1000"),
    (None, "lower(s) = 'neg' OR id % 6 = 0"),
]


def _sql(ref: str, updates: dict[str, str] | None, predicate: str | None) -> str:
    where = f" WHERE {predicate}" if predicate else ""
    if updates is None:
        return f"DELETE FROM {ref}{where}"
    sets = ", ".join(f"{k} = {v}" for k, v in updates.items())
    return f"UPDATE {ref} SET {sets}{where}"


def _read(volume: Any, ref: str, tracked: bool) -> list[tuple[Any, ...]]:
    extra = ", _metadata.row_id, _metadata.row_commit_version" if tracked else ""
    rows = volume.sql(f"SELECT id, n, b, s, to_json(st){extra} FROM {ref} ORDER BY id")
    return [tuple(r) for r in rows]


@pytest.mark.parametrize(
    "properties",
    [
        {"delta.enableInCommitTimestamps": "true"},
        {"delta.enableInCommitTimestamps": "true", "delta.enableRowTracking": "true"},
        {
            "delta.enableInCommitTimestamps": "true",
            "delta.enableRowTracking": "true",
            "delta.enableDeletionVectors": "true",
        },
    ],
    ids=["rewrite", "row_tracking", "row_tracking_dv"],
)
def test_kernel_sql_dml_matches_databricks(
    volume: Any, tmp_path: Any, properties: dict[str, str]
) -> None:
    tracked = properties.get("delta.enableRowTracking") == "true"
    conn = ds.connect()
    path, copy = str(tmp_path / "ours"), str(tmp_path / "theirs")
    conn.create_table(path, SCHEMA, properties=properties)
    conn.open_table(path).append(ROWS.slice(0, 3))
    conn.open_table(path).append(ROWS.slice(3))
    shutil.copytree(path, copy)

    for updates, predicate in STATEMENTS:
        t = conn.open_table(path)
        if updates is None:
            assert t.delete(predicate).engine == "kernel"
        else:
            assert t.update(updates, predicate=predicate).engine == "kernel"
    # ARITHMETIC_OVERFLOW on Databricks (ANSI mode); nothing written here.
    version = conn.open_table(path).version
    with pytest.raises(InvalidArgumentError, match="Overflow"):
        conn.open_table(path).update({"n": "n + 2147483647"}, predicate="id = 1")
    assert conn.open_table(path).version == version

    tag = "_".join(sorted(k.split(".")[-1] for k in properties))
    ours = volume.upload(path, f"ours_{tag}")
    theirs = volume.upload(copy, f"theirs_{tag}")
    for updates, predicate in STATEMENTS:
        volume.sql(_sql(theirs, updates, predicate))
    with pytest.raises(AssertionError, match="ARITHMETIC_OVERFLOW"):
        volume.sql(_sql(theirs, {"n": "n + 2147483647"}, "id = 1"))

    got, want = _read(volume, ours, tracked), _read(volume, theirs, tracked)
    assert got == want
    assert [r[0] for r in got] == ["1", "2", "3", "5"]
