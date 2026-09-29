"""IcebergCompat (UniForm) and geospatial writes, checked against Databricks.

Databricks creates each table as a path table on a scratch volume (UniForm
itself is refused on path tables, DELTA_UNIFORM_NOT_SUPPORTED, so the compat
feature alone is), deltaswamp writes it through every path, and Databricks
reads the result. For IcebergCompatV2, Databricks then converts a copy of the
table to Iceberg (UniForm on a Unity Catalog table, MSCK REPAIR TABLE ... SYNC
METADATA), and an Iceberg client reads it through Unity Catalog's Iceberg REST
endpoint: the Parquet files deltaswamp wrote must carry everything Iceberg
resolves columns and partitions by.
"""

from __future__ import annotations

import pickle
import struct
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.databricks


class _Warehouse:
    def __init__(self, workspace: Any, warehouse_id: str) -> None:
        self.workspace = workspace
        self.warehouse_id = warehouse_id

    def rows(self, statement: str) -> list[list[Any]]:
        from databricks.sdk.service.sql import StatementState

        api = self.workspace.statement_execution
        r = api.execute_statement(
            warehouse_id=self.warehouse_id, statement=statement, wait_timeout="50s"
        )
        while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(1)
            r = api.get_statement(r.statement_id)
        if r.status.state != StatementState.SUCCEEDED:
            raise AssertionError(f"{statement[:200]}: {r.status.error}")
        return r.result.data_array or [] if r.result else []


@pytest.fixture
def warehouse(live_config: Any, live_workspace: Any) -> _Warehouse:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required: the warehouse is the oracle")
    return _Warehouse(live_workspace, live_config.warehouse_id)


@pytest.fixture
def volume(live_config: Any, warehouse: _Warehouse) -> Any:
    name = f"dsw_ug_{uuid.uuid4().hex[:10]}"
    full = f"{live_config.catalog}.{live_config.schema}.{name}"
    warehouse.rows(f"CREATE VOLUME {full}")
    try:
        yield f"/Volumes/{live_config.catalog}/{live_config.schema}/{name}"
    finally:
        warehouse.rows(f"DROP VOLUME IF EXISTS {full}")


def _download(workspace: Any, remote: str, local: Path) -> None:
    for entry in workspace.files.list_directory_contents(remote):
        target = local / entry.name
        if entry.is_directory:
            target.mkdir(parents=True, exist_ok=True)
            _download(workspace, entry.path, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(workspace.files.download(entry.path).contents.read())


def _upload(workspace: Any, local: Path, remote: str) -> None:
    for f in sorted(local.rglob("*")):
        if f.is_file():
            with open(f, "rb") as fh:
                workspace.files.upload(
                    f"{remote}/{f.relative_to(local).as_posix()}", fh, overwrite=True
                )


def _worker(plan: Any) -> Any:
    return pickle.loads(pickle.dumps(plan))


def _point(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, x, y)


def test_geospatial_round_trip(
    live_workspace: Any, warehouse: _Warehouse, volume: str, tmp_path: Path
) -> None:
    import deltaswamp as ds

    remote = f"{volume}/geo"
    ref = f"delta.`{remote}`"
    warehouse.rows(f"CREATE TABLE {ref} (id BIGINT, g GEOMETRY(4326), h GEOGRAPHY(4326))")
    warehouse.rows(
        f"INSERT INTO {ref} VALUES (1, ST_GeomFromText('POINT(1 2)', 4326), "
        "ST_GeogFromText('POINT(3 4)'))"
    )
    local = tmp_path / "geo"
    _download(live_workspace, remote, local)
    conn = ds.connect()

    def rows(ids: list[int]) -> Any:
        return pa.table(
            {
                "id": pa.array(ids, pa.int64()),
                "g": pa.array([_point(i, i + 1) for i in ids], pa.binary()),
                "h": pa.array([_point(i, -i) for i in ids], pa.binary()),
            }
        )

    conn.open_table(str(local)).append(rows([2, 3]))
    plan = conn.open_table(str(local)).plan_write()
    plan.commit([_worker(plan).write(rows([4]))])
    conn.open_table(str(local)).delete("id = 3")
    conn.open_table(str(local)).update({"id": "id + 10"}, predicate="id = 4")
    _upload(live_workspace, local, remote)

    seen = warehouse.rows(f"SELECT id, ST_AsText(g), ST_AsText(h) FROM {ref} ORDER BY id")
    assert [(int(i), g, h) for i, g, h in seen] == [
        (1, "POINT(1 2)", "POINT(3 4)"),
        (2, "POINT(2 3)", "POINT(2 -2)"),
        (14, "POINT(4 5)", "POINT(4 -4)"),
    ]
    # No file is skipped wrongly by a geo predicate.
    assert int(warehouse.rows(f"SELECT count(*) FROM {ref} WHERE ST_X(g) > 1.5")[0][0]) == 2

    warehouse.rows(
        f"INSERT INTO {ref} VALUES (9, ST_GeomFromText('POINT(7 8)', 4326), "
        "ST_GeogFromText('POINT(9 10)'))"
    )
    warehouse.rows(f"DELETE FROM {ref} WHERE id = 1")
    back = tmp_path / "geo_back"
    _download(live_workspace, remote, back)
    got = {r["id"]: r["g"] for r in ds.connect().open_table(str(back)).to_arrow().to_pylist()}
    assert sorted(got) == [2, 9, 14]
    assert got[2] == _point(2, 3) and got[9] == _point(7, 8)


_V2_SQL = (
    "(id BIGINT, s STRING, a ARRAY<INT>, m MAP<STRING, INT>, ts TIMESTAMP, p STRING) "
    "USING DELTA PARTITIONED BY (p) TBLPROPERTIES ("
    "'delta.enableIcebergCompatV2' = 'true', 'delta.columnMapping.mode' = 'name')"
)


def _v2_rows(ids: list[int], part: str) -> Any:
    import datetime as dt

    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "s": pa.array([f"s{i}" for i in ids]),
            "a": pa.array([[i, i + 1] for i in ids], pa.list_(pa.int32())),
            "m": pa.array([[("k", i)] for i in ids], pa.map_(pa.string(), pa.int32())),
            "ts": pa.array(
                [dt.datetime(2024, 5, 1, tzinfo=dt.UTC)] * len(ids), pa.timestamp("us", tz="UTC")
            ),
            "p": pa.array([part] * len(ids)),
        }
    )


def test_iceberg_compat_v2_round_trip_and_iceberg_read(
    live_config: Any, live_workspace: Any, warehouse: _Warehouse, volume: str, tmp_path: Path
) -> None:
    import deltaswamp as ds

    remote = f"{volume}/v2"
    ref = f"delta.`{remote}`"
    warehouse.rows(f"CREATE TABLE {ref} {_V2_SQL}")
    warehouse.rows(
        f"INSERT INTO {ref} VALUES (1, 'a', array(1, 2), map('k', 1), "
        "TIMESTAMP '2024-01-01 00:00:00', 'x')"
    )
    local = tmp_path / "v2"
    _download(live_workspace, remote, local)
    conn = ds.connect()
    conn.open_table(str(local)).append(_v2_rows([2, 3], "x"))
    plan = conn.open_table(str(local)).plan_write()
    plan.commit([_worker(plan).write(_v2_rows([4, 5], "y"))])
    conn.open_table(str(local)).delete("id = 3")
    conn.open_table(str(local)).update({"s": "'upd'"}, predicate="id = 4")
    conn.open_table(str(local)).optimize()
    _upload(live_workspace, local, remote)

    expected = [(1, "a", "x"), (2, "s2", "x"), (4, "upd", "y"), (5, "s5", "y")]
    seen = warehouse.rows(f"SELECT id, s, p, a[0], m['k'] FROM {ref} ORDER BY id")
    assert [(int(i), s, p) for i, s, p, _, _ in seen] == expected
    assert [int(a0) for *_, a0, _ in seen] == [1, 2, 4, 5]
    # Databricks writes on top, then deltaswamp reads it.
    warehouse.rows(f"UPDATE {ref} SET s = 'dbx' WHERE id = 5")
    back = tmp_path / "v2_back"
    _download(live_workspace, remote, back)
    assert {r["id"]: r["s"] for r in ds.connect().open_table(str(back)).to_arrow().to_pylist()}[
        5
    ] == "dbx"

    # Iceberg: Databricks converts a copy of the table (UniForm needs a Unity
    # Catalog table) over the same Parquet files, and an Iceberg client reads it.
    pyiceberg = pytest.importorskip("pyiceberg.catalog")
    name = f"dsw_ice_{uuid.uuid4().hex[:10]}"
    uc = f"{live_config.catalog}.{live_config.schema}.{name}"
    warehouse.rows(f"CREATE TABLE {uc} DEEP CLONE {ref}")
    try:
        warehouse.rows(
            f"ALTER TABLE {uc} SET TBLPROPERTIES ("
            "'delta.enableIcebergCompatV2' = 'true', "
            "'delta.universalFormat.enabledFormats' = 'iceberg')"
        )
        warehouse.rows(f"MSCK REPAIR TABLE {uc} SYNC METADATA")
        catalog = pyiceberg.load_catalog(
            "uc",
            type="rest",
            uri=f"{live_config.host.rstrip('/')}/api/2.1/unity-catalog/iceberg-rest",
            warehouse=live_config.catalog,
            token=live_config.token,
        )
        table = catalog.load_table(f"{live_config.schema}.{name}")
        got = sorted(
            (r["id"], r["s"], r["p"])
            for r in table.scan(selected_fields=("id", "s", "p")).to_arrow().to_pylist()
        )
        assert got == [(1, "a", "x"), (2, "s2", "x"), (4, "upd", "y"), (5, "dbx", "y")]
        files = {t.file.file_path.rsplit("/", 1)[-1] for t in table.scan().plan_files()}
        # Iceberg reads files deltaswamp wrote, not only Databricks' own.
        assert any(not f.startswith("part-") for f in files), files
    finally:
        warehouse.rows(f"DROP TABLE IF EXISTS {uc}")


def test_iceberg_compat_v1_round_trip(
    live_workspace: Any, warehouse: _Warehouse, volume: str, tmp_path: Path
) -> None:
    import deltaswamp as ds

    remote = f"{volume}/v1"
    ref = f"delta.`{remote}`"
    warehouse.rows(
        f"CREATE TABLE {ref} (id BIGINT, s STRING, p STRING) USING DELTA PARTITIONED BY (p) "
        "TBLPROPERTIES ('delta.enableIcebergCompatV1' = 'true', "
        "'delta.columnMapping.mode' = 'name')"
    )
    warehouse.rows(f"INSERT INTO {ref} VALUES (1, 'a', 'x')")
    local = tmp_path / "v1"
    _download(live_workspace, remote, local)
    conn = ds.connect()
    conn.open_table(str(local)).append(
        pa.table({"id": pa.array([2], pa.int64()), "s": ["b"], "p": ["y"]})
    )
    conn.open_table(str(local)).delete("id = 1")
    _upload(live_workspace, local, remote)
    assert [(int(i), s, p) for i, s, p in warehouse.rows(f"SELECT id, s, p FROM {ref}")] == [
        (2, "b", "y")
    ]
