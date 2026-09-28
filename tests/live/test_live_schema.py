"""Kernel CHECK constraints and schema evolution, checked against Databricks.

Each test builds a table locally on the kernel path (one delta-rs cannot
write, or a column-mapped one whose schema it cannot change), uploads it to a
scratch UC volume, and has the warehouse read it by path: Databricks must
honour a constraint the kernel added -- an INSERT violating it fails there --
and read a schema the kernel evolved on write, column mapping included.

Opt-in on top of the live suite (it needs DELTASWAMP_TEST_WAREHOUSE_ID):

    DELTASWAMP_TEST_DATABRICKS=1 pytest tests/live/test_live_schema.py -v

The volume (named a8sc_...) is created in the target schema and dropped
afterwards.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import deltaswamp as ds  # noqa: E402

from .test_live_interop_stats import _Volume  # noqa: E402

pytestmark = pytest.mark.databricks


class _SchemaVolume(_Volume):
    def __init__(self, workspace: Any, config: Any) -> None:
        self.w, self.config = workspace, config
        self.name = f"a8sc_{uuid.uuid4().hex[:10]}"
        self.root = f"/Volumes/{config.catalog}/{config.schema}/{self.name}"
        self.sql(f"CREATE VOLUME {config.prefix}.{self.name}")


@pytest.fixture(scope="module")
def volume(live_config: Any, live_workspace: Any) -> Any:
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required to query files by path")
    vol = _SchemaVolume(live_workspace, live_config)
    try:
        yield vol
    finally:
        vol.drop()


def _rows(volume: Any, ref: str, columns: str) -> list[tuple[Any, ...]]:
    return sorted(
        (tuple(r) for r in volume.sql(f"SELECT {columns} FROM {ref}")),
        key=lambda r: tuple((v is None, str(v)) for v in r),
    )


def _served_by_kernel(t: Any, operation: str, **shape: Any) -> None:
    verdict = t.can(operation, **shape)
    assert verdict.ok and str(verdict.engine) == "kernel", verdict


def test_databricks_honours_a_constraint_the_kernel_added(volume: Any, tmp_path: Any) -> None:
    """In-commit timestamps: a table delta-rs cannot write at all."""
    conn = ds.connect()
    path = str(tmp_path / "ict")
    t = conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("v", pa.string())]),
        properties={
            "delta.enableInCommitTimestamps": "true",
            "delta.enableDeletionVectors": "true",
        },
    )
    t.append(pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}))
    t = conn.table(path)
    _served_by_kernel(t, "add_constraint", constraints={"positive": "id > 0"})
    t.add_constraint({"positive": "id > 0"})
    # The kernel enforces it on its own writes, and they stay readable there.
    with pytest.raises(ds.InvalidArgumentError, match="CHECK constraint positive"):
        conn.table(path).append(pa.table({"id": [-1], "v": ["x"]}))
    conn.table(path).append(pa.table({"id": [4], "v": ["d"]}))
    conn.table(path).update({"v": "'B'"}, predicate="id = 2")

    ref = volume.upload(path, "ict")
    assert _rows(volume, ref, "id, v") == [("1", "a"), ("2", "B"), ("3", "c"), ("4", "d")]
    properties = dict(volume.sql(f"SHOW TBLPROPERTIES {ref}"))
    assert properties.get("delta.constraints.positive") == "id > 0"
    with pytest.raises(AssertionError, match=r"(?i)constraint|violat"):
        volume.sql(f"INSERT INTO {ref} VALUES (-5, 'bad')")
    volume.sql(f"INSERT INTO {ref} VALUES (5, 'e')")
    assert [r[0] for r in _rows(volume, ref, "id")] == ["1", "2", "3", "4", "5"]


def test_databricks_reads_a_schema_the_kernel_evolved_under_column_mapping(
    volume: Any, tmp_path: Any
) -> None:
    """A (2, 5) column-mapped table: delta-rs cannot change its schema on write."""
    conn = ds.connect()
    path = str(tmp_path / "cm")
    s = pa.struct([("a", pa.int64())])
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("s", s)]),
        properties={"delta.columnMapping.mode": "name"},
    )
    conn.table(path).append(pa.table({"id": [1], "s": pa.array([{"a": 1}], s)}))
    wide = pa.struct([("a", pa.int64()), ("b", pa.string())])
    data = pa.table({"id": [2], "s": pa.array([{"a": 2, "b": "x"}], wide), "extra": ["new"]})
    t = conn.table(path)
    _served_by_kernel(t, "append", data=data, schema_mode="merge")
    t.append(data, schema_mode="merge")

    ref = volume.upload(path, "cm")
    assert _rows(volume, ref, "id, s.a, s.b, extra") == [
        ("1", "1", None, None),
        ("2", "2", "x", "new"),
    ]
    # Databricks writes the evolved columns under the physical names assigned.
    volume.sql(f"INSERT INTO {ref} VALUES (3, named_struct('a', 3, 'b', 'y'), 'db')")
    assert _rows(volume, ref, "id, s.b, extra")[-1] == ("3", "y", "db")


def test_databricks_reads_a_schema_evolved_on_a_table_only_the_kernel_writes(
    volume: Any, tmp_path: Any
) -> None:
    """Liquid clustering (domainMetadata) and type widening, merged on write."""
    conn = ds.connect()
    path = str(tmp_path / "tw")
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("w", pa.int32())]),
        cluster_by=["id"],
        properties={"delta.enableTypeWidening": "true"},
    )
    conn.table(path).append(pa.table({"id": [1], "w": pa.array([1], pa.int32())}))
    # A long into the INTEGER column (widened, as Databricks widens), and a new column.
    data = pa.table({"id": [2], "w": pa.array([2**40], pa.int64()), "note": ["n"]})
    t = conn.table(path)
    _served_by_kernel(t, "append", data=data, schema_mode="merge")
    t.append(data, schema_mode="merge")

    ref = volume.upload(path, "tw")
    assert _rows(volume, ref, "id, w, note") == [("1", "1", None), ("2", str(2**40), "n")]
    kinds = {r[0]: r[1] for r in volume.sql(f"DESCRIBE {ref}") if r[0] in ("w", "note")}
    assert kinds == {"w": "bigint", "note": "string"}
