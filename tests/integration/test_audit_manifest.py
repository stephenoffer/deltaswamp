"""Symlink manifests (`generate()`) of the tables only the kernel can write.

Round-8 maintenance gaps: GENERATE was refused on every table delta-rs cannot
open for writing (clustering, row tracking, in-commit timestamps, type
widening, column defaults), though a manifest is only the snapshot's file
list. The kernel writes it as Spark does (crates/native/src/manifest.rs).
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.errors import UnreachableTableError  # noqa: E402

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

ICT = {"delta.enableInCommitTimestamps": "true"}


def _manifests(path: str) -> dict[str, list[str]]:
    root = pathlib.Path(path, "_symlink_format_manifest")
    return {
        str(f.parent.relative_to(root)): f.read_text().splitlines() for f in root.rglob("manifest")
    }


def _live(conn: Any, path: str) -> list[str]:
    files = conn.open_table(path).files()
    column = files.column("path") if hasattr(files, "column") else files["path"]
    return sorted(str(pathlib.Path(path, p)) for p in _decoded(list(column)))


def _decoded(paths: list[Any]) -> list[str]:
    from urllib.parse import unquote

    return [unquote(str(p)) for p in paths]


def test_an_unpartitioned_kernel_only_table(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t t%")
    conn.write_table(path, pa.table({"id": [1, 2]}), properties=ICT)
    conn.open_table(path).append(pa.table({"id": [3]}))
    t = conn.open_table(path)
    assert t.can("generate").engine is Engine.KERNEL
    t.generate()
    manifests = _manifests(path)
    assert list(manifests) == ["."]
    lines = manifests["."]
    assert len(lines) == 2
    assert all(os.path.isabs(line) and os.path.exists(line) for line in lines)
    assert sorted(lines) == _live(conn, path)
    # Regenerated after an overwrite: the old files are gone from it.
    conn.open_table(path).overwrite(pa.table({"id": [9]}))
    conn.open_table(path).generate()
    (line,) = _manifests(path)["."]
    assert os.path.exists(line) and [line] == _live(conn, path)


def test_partition_manifests_are_hive_escaped_and_stale_ones_go(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    data = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "g": ["a b", None, "x/y", "é=%", ""],
            "h": pa.array([1, 1, 2, None, 3], pa.int32()),
        }
    )
    conn.write_table(path, data, partition_by=["g", "h"], properties=ICT)
    t = conn.open_table(path)
    assert t.can("generate").engine is Engine.KERNEL
    t.generate()
    manifests = _manifests(path)
    assert sorted(manifests) == sorted(
        [
            "g=a b/h=1",
            "g=__HIVE_DEFAULT_PARTITION__/h=1",
            "g=x%2Fy/h=2",
            "g=é%3D%25/h=__HIVE_DEFAULT_PARTITION__",
            "g=__HIVE_DEFAULT_PARTITION__/h=3",
        ]
    )
    lines = sorted(line for ls in manifests.values() for line in ls)
    assert lines == _live(conn, path)
    assert all(os.path.exists(line) for line in lines)
    # A partition emptied by an overwrite loses its manifest.
    conn.open_table(path).overwrite(
        pa.table({"id": pa.array([6], pa.int64()), "g": ["a b"], "h": pa.array([1], pa.int32())})
    )
    conn.open_table(path).generate()
    manifests = _manifests(path)
    assert list(manifests) == ["g=a b/h=1"]
    assert manifests["g=a b/h=1"] == _live(conn, path)


def test_an_empty_table_gets_an_empty_manifest(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    conn.create_table(path, pa.schema([("id", pa.int64())]), properties=ICT)
    conn.open_table(path).generate()
    assert _manifests(path) == {".": []}


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"cluster_by": ["id"]}, id="clustered"),
        pytest.param({"properties": {"delta.enableRowTracking": "true"}}, id="row_tracking"),
        pytest.param({"properties": {"delta.enableTypeWidening": "true"}}, id="type_widening"),
    ],
)
def test_other_kernel_only_tables(conn: Any, tmp_path: Any, kw: dict[str, Any]) -> None:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": pa.array([1, 2, 3], pa.int64())}), **kw)
    t = conn.open_table(path)
    assert t.can("generate").engine is Engine.KERNEL
    t.generate()
    assert _manifests(path)["."] == _live(conn, path)


def test_the_kernel_writes_what_delta_rs_writes(conn: Any, tmp_path: Any) -> None:
    from deltaswamp.engine.kernel import KernelEngine

    ours, theirs = str(tmp_path / "a"), str(tmp_path / "b")
    data = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "g": ["x", "y", "y"]})
    deltalake.write_deltalake(ours, data, partition_by=["g"])
    deltalake.write_deltalake(theirs, data, partition_by=["g"])
    KernelEngine().generate(conn.open_table(ours)._enrich())
    deltalake.DeltaTable(theirs).generate()

    def shape(path: str) -> dict[str, list[str]]:
        return {
            k: sorted(line.removeprefix(path) for line in v) for k, v in _manifests(path).items()
        }

    # Same directories, and the same files relative to each table's root
    # (delta-rs's file names differ only by their uuids, so compare counts).
    a, b = shape(ours), shape(theirs)
    assert sorted(a) == sorted(b) == ["g=x", "g=y"]
    assert {k: len(v) for k, v in a.items()} == {k: len(v) for k, v in b.items()}
    assert all(line.startswith("/g=") for v in a.values() for line in v)


@pytest.mark.parametrize(
    ("props", "why"),
    [
        ({"delta.columnMapping.mode": "name", **ICT}, "columnMapping"),
        ({"delta.enableDeletionVectors": "true", **ICT}, "deletionVectors"),
    ],
)
def test_tables_a_manifest_cannot_describe_are_refused(
    conn: Any, tmp_path: Any, props: dict[str, str], why: str
) -> None:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1]}), properties=props)
    t = conn.open_table(path)
    verdict = t.can("generate")
    assert not verdict.ok and why in verdict.reason
    with pytest.raises(UnreachableTableError, match=why):
        t.generate()
    assert not pathlib.Path(path, "_symlink_format_manifest").exists()
