"""A DML commit rebases over a concurrent one whatever the checksum does to the JSON.

With a `.crc` per version, a snapshot of the newer version loads its protocol
and metaData from the checksum, whose JSON orders the fields differently from
the log replay the older snapshot used. The rebase check compared the text,
took every concurrent commit for a metadata change, and failed the second of
two concurrent MERGEs on a row-tracked table with MetadataChangedError.
"""

from __future__ import annotations

from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")


def _upsert(table: Any, key: int, value: int) -> None:
    source = pa.table({"id": [key], "n": [value]})
    (
        table.merge(source, "t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )


@pytest.mark.parametrize(
    "properties",
    [
        {"delta.enableRowTracking": "true"},
        {"delta.enableInCommitTimestamps": "true"},
        {"delta.enableDeletionVectors": "true"},
    ],
    ids=["row_tracking", "ict", "dv"],
)
def test_a_merge_rebases_over_a_concurrent_merge(
    tmp_path: Any, monkeypatch: Any, properties: dict[str, str]
) -> None:
    from deltaswamp.engine.kernel import KernelEngine

    path = str(tmp_path / "t")
    conn = ds.connect()
    schema = pa.schema([("id", pa.int64()), ("n", pa.int64())])
    conn.create_table(path, schema, properties=properties).append(
        pa.table({"id": [1, 2], "n": [0, 0]})
    )
    commit = KernelEngine._commit_dv_changes
    raced: list[bool] = []

    def racing(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Another writer commits between this MERGE's read and its commit.
        if not raced:
            raced.append(True)
            _upsert(ds.connect().table(path), 1, 10)
        return commit(self, *args, **kwargs)

    monkeypatch.setattr(KernelEngine, "_commit_dv_changes", racing)
    _upsert(conn.table(path), 3, 30)
    assert raced
    rows = sorted(conn.table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
    assert rows == [{"id": 1, "n": 10}, {"id": 2, "n": 0}, {"id": 3, "n": 30}]
