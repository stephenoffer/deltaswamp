"""Files deltaswamp writes, queried by path on Databricks: data skipping and INT96.

Each test writes a table locally, uploads it to a scratch UC volume, and asks
the warehouse questions whose answer depends on the files' statistics: a
wrong bound makes Databricks skip a file holding matching rows. The INT96
test goes the other way: Databricks writes nested timestamps (INT96, as its
warehouses do) and deltaswamp reads them.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_interop_stats.py -v

The volume is created in the target schema and dropped afterwards.
"""

from __future__ import annotations

import contextlib
import decimal
import io
import pathlib
import time
import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402
from deltaswamp.capability import Engine  # noqa: E402

pytestmark = pytest.mark.databricks


def _only(kind: Engine) -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    engine = KernelEngine() if kind is Engine.KERNEL else DeltaRsEngine()
    return Connection(catalog=FilesystemCatalog(), router=Router(engines={kind: engine}))


class _Volume:
    """A scratch UC volume, files uploaded to it, SQL run on the warehouse."""

    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"dsw_interop_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")

    def sql(self, statement: str) -> list[list[Any]]:
        api = self.w.statement_execution
        r = api.execute_statement(
            warehouse_id=self.config.warehouse_id, statement=statement, wait_timeout="50s"
        )
        while r.status.state.value in ("PENDING", "RUNNING"):
            time.sleep(1)
            r = api.get_statement(r.statement_id)
        if r.status.error:
            raise AssertionError(f"{statement[:160]}: {r.status.error.message}")
        return list(r.result.data_array or []) if r.result else []

    def upload(self, local: str, rel: str) -> str:
        for f in sorted(pathlib.Path(local).rglob("*")):
            if f.is_file():
                self.w.files.upload(
                    f"{self.root}/{rel}/{f.relative_to(local)}",
                    io.BytesIO(f.read_bytes()),
                    overwrite=True,
                )
        return f"delta.`{self.root}/{rel}`"

    def download(self, rel: str, local: str) -> None:
        def walk(d: str) -> None:
            for e in self.w.files.list_directory_contents(d):
                if e.is_directory:
                    walk(e.path)
                    continue
                dest = pathlib.Path(local, e.path[len(f"{self.root}/{rel}/") :])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(self.w.files.download(e.path).contents.read())

        walk(f"{self.root}/{rel}")

    def count(self, ref: str, predicate: str) -> int:
        return int(self.sql(f"SELECT count(*) FROM {ref} WHERE {predicate}")[0][0])

    def drop(self) -> None:
        with contextlib.suppress(Exception):
            self.sql(f"DROP VOLUME IF EXISTS {self.config.prefix}.{self.name}")


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _Volume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


D = decimal.Decimal
BIG = "12345678901234567890.123456789012345678"
D38 = "9" * 38


@pytest.mark.parametrize("op", ["delete", "update", "merge"])
def test_deletion_vector_dml_keeps_decimal_bounds_databricks_trusts(
    volume: Any, tmp_path: Any, op: str
) -> None:
    path = str(tmp_path / op)
    conn = _only(Engine.KERNEL)
    data = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "dec": pa.array([D(BIG), D("0.1"), D("-" + BIG)], pa.decimal128(38, 18)),
            "d0": pa.array([D(D38), D(1), D("-" + D38)], pa.decimal128(38, 0)),
        }
    )
    conn.create_table(path, data.schema, properties={"delta.enableDeletionVectors": "true"})
    conn.open_table(path).append(data)
    t = ds.connect("file://").open_table(path)
    if op == "delete":
        t.delete("id = 2")
    elif op == "update":
        t.update({"id": "id"}, predicate="id = 2")
    else:
        (
            t.merge(pa.table({"id": pa.array([2], pa.int64())}), "target.id = source.id")
            .when_matched_delete()
            .execute()
        )
    ref = volume.upload(path, f"dv_{op}")
    for predicate in (f"dec = {BIG}", f"dec = -{BIG}", f"d0 = {D38}", f"d0 = -{D38}"):
        assert volume.count(ref, predicate) == 1, predicate
    assert volume.count(ref, "dec > 12345678901234567890") == 1


NAN = float("nan")
FLOATS = [1.5, NAN, 3.0, None, -7.0, NAN, 2.0, float("-inf")]
NAN_PREDICATES = [
    "f = double('NaN')",
    "f > 100",
    "NOT (f < 100)",
    "f > double('Infinity')",
    "fl = float('NaN')",
    "st.x > 100",
    "f >= 3",
]


def test_nan_rows_are_not_skipped(volume: Any, tmp_path: Any) -> None:
    data = pa.table(
        {
            "id": pa.array(range(1, 9), pa.int64()),
            "f": pa.array(FLOATS, pa.float64()),
            "fl": pa.array(FLOATS, pa.float32()),
            "st": pa.array([{"x": v} for v in FLOATS], pa.struct([("x", pa.float64())])),
        }
    )
    refs = {}
    for name, kind in (("kernel", Engine.KERNEL), ("deltars", Engine.DELTARS)):
        path = str(tmp_path / name)
        conn = _only(kind)
        conn.create_table(path, data.schema)
        conn.open_table(path).append(data)
        if kind is Engine.DELTARS:
            conn.open_table(path).update({"id": "id"}, predicate="id = 1")
        refs[name] = volume.upload(path, f"nan_{name}")

    def lit(v: Any) -> str:
        if v is None:
            return "NULL"
        if v != v:
            return "double('NaN')"
        return "double('-Infinity')" if v == float("-inf") else repr(v)

    reference = f"delta.`{volume.root}/nan_databricks`"
    volume.sql(f"CREATE TABLE {reference} (id BIGINT, f DOUBLE, fl FLOAT, st STRUCT<x: DOUBLE>)")
    values = ", ".join(
        f"({i}, {lit(v)}, CAST({lit(v)} AS FLOAT), named_struct('x', {lit(v)}))"
        for i, v in enumerate(FLOATS, 1)
    )
    volume.sql(f"INSERT INTO {reference} VALUES {values}")
    for predicate in NAN_PREDICATES:
        want = volume.count(reference, predicate)
        got = {name: volume.count(ref, predicate) for name, ref in refs.items()}
        assert got == {name: want for name in refs}, (predicate, want, got)


def test_databricks_nested_int96_timestamps_read(volume: Any, tmp_path: Any) -> None:
    import datetime as dt

    ts = "TIMESTAMP'2020-01-01 12:34:56.789012Z'"
    old = "TIMESTAMP'1850-03-01 01:02:03.456789Z'"
    want_ts = dt.datetime(2020, 1, 1, 12, 34, 56, 789012, tzinfo=dt.UTC)
    want_old = dt.datetime(1850, 3, 1, 1, 2, 3, 456789, tzinfo=dt.UTC)
    shapes = {
        "a": ("ARRAY<TIMESTAMP>", f"array({ts}, NULL, {old})", [want_ts, None, want_old]),
        "s": ("STRUCT<y: ARRAY<TIMESTAMP>>", f"named_struct('y', array({ts}))", {"y": [want_ts]}),
        "aos": (
            "ARRAY<STRUCT<x: BIGINT, y: ARRAY<TIMESTAMP>>>",
            f"array(named_struct('x', 1L, 'y', array({old})))",
            [{"x": 1, "y": [want_old]}],
        ),
    }
    for name, (kind, value, want) in shapes.items():
        ref = f"delta.`{volume.root}/int96_{name}`"
        volume.sql(f"CREATE TABLE {ref} (id BIGINT, c {kind})")
        volume.sql(f"INSERT INTO {ref} VALUES (1, {value})")
        local = str(tmp_path / f"int96_{name}")
        volume.download(f"int96_{name}", local)
        rows = pa.table(ds.connect("file://").open_table(local).to_arrow()).to_pylist()
        assert rows == [{"id": 1, "c": want}], name
