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
