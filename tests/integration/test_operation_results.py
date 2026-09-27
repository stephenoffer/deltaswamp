"""DML and maintenance results carry the same keys whichever engine serves them."""

from __future__ import annotations

from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")

_DATA = pa.table({"id": pa.array([1, 2, 3, 4], pa.int64()), "s": ["a", "b", "c", "d"]})


@pytest.mark.parametrize(
    "properties",
    [{}, {"delta.enableDeletionVectors": "true"}],
    ids=["copy_on_write", "deletion_vectors"],
)
def test_dml_results_share_keys(tmp_path: Any, properties: dict[str, str]) -> None:
    conn = ds.connect("file://")
    t = conn.create_table(str(tmp_path / "t"), _DATA.schema, properties=properties)
    t.append(_DATA)
    deleted = t.delete("id = 1")
    assert isinstance(deleted, ds.OperationResult)
    assert deleted["num_deleted_rows"] == 1 and deleted.engine is not None
    updated = t.update(new_values={"s": "z"}, predicate="id = 2")
    assert updated["num_updated_rows"] == 1
    merged = (
        t.merge(
            pa.table({"id": pa.array([3, 9], pa.int64()), "s": ["m", "n"]}), "target.id = source.id"
        )
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    assert merged["num_updated_rows"] == 1 and merged["num_inserted_rows"] == 1
    assert merged["num_affected_rows"] == 2
    optimized = t.optimize()
    assert "num_files_added" in optimized and "num_files_removed" in optimized
