"""Regressions for defects found by the deep audit, each proven on a real table.

Every test here failed before its fix: most of them silently corrupted a table
or lost data while the call reported success.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _age(path: str, days: float) -> None:
    old = time.time() - days * 86400
    for f in pathlib.Path(path).rglob("*"):
        if "_delta_log" not in f.parts:
            os.utime(f, (old, old))


class TestVacuumKeepsDeletionVectors:
    """delta-rs's full VACUUM deleted live deletion-vector files."""

    def _table(self, conn: Any, tmp_path: Any) -> tuple[Any, str]:
        path = str(tmp_path / "t")
        t = conn.create_table(
            path,
            pa.schema([("id", pa.int64())]),
            properties={"delta.enableDeletionVectors": "true"},
        )
        t.append(pa.table({"id": pa.array(range(10), pa.int64())}))
        t.delete("id < 3")
        assert list(pathlib.Path(path).rglob("deletion_vector_*.bin"))
        _age(path, 8)
        return t, path

    def test_full_vacuum_keeps_every_vector_file(self, conn: Any, tmp_path: Any) -> None:
        # Refused outright before, dry runs included; now the orphans go and
        # every deletion-vector file stays.
        t, path = self._table(conn, tmp_path)
        orphan = pathlib.Path(path) / "part-00000-orphan.parquet"
        orphan.write_bytes(b"left by a crashed write")
        _age(path, 8)
        vectors = sorted(pathlib.Path(path).rglob("deletion_vector_*.bin"))
        assert t.can("vacuum")
        planned = t.vacuum(retention_hours=0, enforce_retention_duration=False)
        assert planned == ["part-00000-orphan.parquet"]
        assert orphan.exists()
        t.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
        assert not orphan.exists()
        assert sorted(pathlib.Path(path).rglob("deletion_vector_*.bin")) == vectors
        assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == list(
            range(3, 10)
        )

    def test_lite_vacuum_keeps_the_table_readable(self, conn: Any, tmp_path: Any) -> None:
        t, path = self._table(conn, tmp_path)
        assert t.can("vacuum", lite=True)
        t.vacuum(lite=True, dry_run=False)
        assert list(pathlib.Path(path).rglob("deletion_vector_*.bin"))
        assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == list(
            range(3, 10)
        )


_TWO_COLS = pa.schema([("id", pa.int64()), ("v", pa.int64())])


class TestAddFeatureKeepsLegacyFeatures:
    """delta-rs replaced a legacy protocol's implied features with the new one."""

    @pytest.mark.parametrize(
        "properties",
        [{}, {"delta.columnMapping.mode": "name"}, {"delta.enableChangeDataFeed": "true"}],
    )
    def test_implied_features_survive(
        self, conn: Any, tmp_path: Any, properties: dict[str, str]
    ) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, _TWO_COLS, properties=properties)
        t = conn.open_table(path)
        t.append(pa.table({"id": [1], "v": [1]}))
        if not properties:
            t.add_constraint({"v_pos": "v > 0"})
        before = conn.open_table(path).resolved.effective_writer_features
        conn.open_table(path).add_feature("v2Checkpoint")
        after = conn.open_table(path)
        assert before <= after.resolved.effective_writer_features
        assert after.to_arrow().to_pylist() == [{"id": 1, "v": 1}]
        if not properties:
            with pytest.raises(Exception, match=r"(?i)constraint|invalid|violat"):
                after.append(pa.table({"id": [2], "v": [-5]}))


class TestPropertiesOnTableFeatureProtocols:
    def test_set_properties_on_a_dv_table_adds_no_variant_feature(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, _TWO_COLS, properties={"delta.enableDeletionVectors": "true"})
        for props in (
            {"delta.logRetentionDuration": "interval 60 days"},
            {"delta.enableChangeDataFeed": "true"},
            {"delta.checkpointPolicy": "v2"},
        ):
            conn.open_table(path).set_properties(props)
            assert "variantType" not in conn.open_table(path).features()
        assert "v2Checkpoint" in conn.open_table(path).features()


class TestMergeReadsStringLiteralsAsSpark:
    """Kernel MERGE clauses run on DuckDB, which reads `''` as an escaped quote;
    Spark (and so the warehouse) reads `'it''s'` as adjacent literals, `its`."""

    def test_adjacent_literals_concatenate(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(
            path,
            pa.schema([("id", pa.int64()), ("v", pa.string())]),
            properties={"delta.enableDeletionVectors": "true"},
        )
        conn.open_table(path).append(pa.table({"id": [1, 2], "v": ["it's", "its"]}))
        (
            conn.open_table(path)
            .merge(pa.table({"id": [1, 2]}), "t.id = s.id", source_alias="s", target_alias="t")
            .when_matched_delete("t.v = 'it''s'")
            .execute()
        )
        assert conn.open_table(path).to_arrow().column("v").to_pylist() == ["it's"]
