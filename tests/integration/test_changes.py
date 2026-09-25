"""Following the change feed version by version, as a streaming read would."""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


@pytest.fixture
def table(tmp_path: Any) -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    conn = ds.connect(catalog=FilesystemCatalog())
    t = conn.create_table(
        str(tmp_path / "c"),
        pa.schema([("id", pa.int64())]),
        properties={"delta.enableChangeDataFeed": "true"},
    )
    t.append(pa.table({"id": [1, 2]}))
    t.append(pa.table({"id": [3]}))
    t.delete("id = 1")
    return t


def summary(batches: Any) -> list[tuple[int, list[tuple[int, str]]]]:
    return [
        (
            v,
            sorted(
                zip(b.column("id").to_pylist(), b.column("_change_type").to_pylist(), strict=True)
            ),
        )
        for v, b in batches
    ]


def test_one_batch_per_version_in_order(table: Any) -> None:
    assert summary(table.changes(1)) == [
        (1, [(1, "insert"), (2, "insert")]),
        (2, [(3, "insert")]),
        (3, [(1, "delete")]),
    ]


def test_resuming_after_the_last_version_yields_nothing(table: Any) -> None:
    assert list(table.changes(4)) == []


def test_resume_from_a_recorded_offset(table: Any) -> None:
    assert [v for v, _ in table.changes(3)] == [3]


def test_polling_sees_later_commits(table: Any) -> None:
    stream = table.changes(3, poll_interval=0.01)
    assert next(stream)[0] == 3
    table.append(pa.table({"id": [9]}))
    version, batch = next(stream)
    assert version == 4
    assert batch.column("id").to_pylist() == [9]
    stream.close()


def test_cdf_with_an_escaped_partition_path_reads_through_the_kernel(tmp_path: Any) -> None:
    """delta-rs double-encodes the path: 'a b' is stored as k=a%20b and its CDF
    reader looks for k=a%2520b. Twelve CDF tables in the delta-kernel golden
    corpus fail that way; the kernel reads them all."""
    from deltalake import write_deltalake

    path = str(tmp_path / "t")
    write_deltalake(
        path,
        pa.table({"k": ["a b", "c"], "v": [1, 2]}),
        partition_by=["k"],
        configuration={"delta.enableChangeDataFeed": "true"},
    )
    table = ds.connect().table(path)
    assert table.can(ds.Operation.CDF).engine is ds.Engine.KERNEL
    changes = pa.table(table.cdf(starting_version=0))
    assert sorted(changes.column("k").to_pylist()) == ["a b", "c"]


def test_legacy_protocol_features_are_reported(tmp_path: Any) -> None:
    """A (1, 2) table names no features; DESCRIBE DETAIL lists the implied ones."""
    from deltalake import write_deltalake

    path = str(tmp_path / "t")
    write_deltalake(path, pa.table({"id": [1]}))
    assert {"appendOnly", "invariants"} <= ds.connect().table(path).features()
