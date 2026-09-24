"""Probe tests for the property matrix.

`PROPERTY_SUPPORT` records what each engine does with each Delta property. Those
rows came from probing the installed engines, so they are only true for the
versions pinned today. These tests re-run the probes, which means an engine
upgrade that changes behavior fails here instead of drifting silently -- the
same job the KERNEL_VERSION pin does for the Rust side.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import PROPERTY_SUPPORT, Engine, PropertyEffect, property_support
from deltaswamp.properties import effect_for

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

SCHEMA_FIELDS = [("id", pa.int64()), ("city", pa.string())]


def _probe_deltars_create(path: str, properties: dict[str, str]) -> PropertyEffect:
    """What does the installed delta-rs actually do with this property?"""
    from deltalake import DeltaTable, write_deltalake

    try:
        write_deltalake(path, pa.table({"id": [1], "city": ["oslo"]}), configuration=properties)
    except Exception:
        return PropertyEffect.REJECTED
    except BaseException as exc:
        if type(exc).__name__ == "PanicException":
            return PropertyEffect.CRASH
        raise
    stored = dict(DeltaTable(path).metadata().configuration)
    key = next(iter(properties))
    return PropertyEffect.HONORED if key in stored else PropertyEffect.STORED


class TestMatrixIsInternallyConsistent:
    def test_every_row_keys_itself(self) -> None:
        for key, row in PROPERTY_SUPPORT.items():
            assert row.key == key

    def test_lookup_falls_back_to_prefix_rules(self) -> None:
        assert property_support("delta.feature.rowTracking").kernel_create is PropertyEffect.HONORED
        assert property_support("delta.unknownThing").kernel_create is PropertyEffect.REJECTED
        assert property_support("custom.anything").kernel_create is PropertyEffect.STORED

    def test_clustering_is_not_settable_as_a_feature_signal(self) -> None:
        """The kernel deliberately excludes it: clustering columns go through
        the data layout, not a property."""
        row = property_support("delta.feature.clustering")
        assert row.kernel_create is PropertyEffect.REJECTED
        assert "cluster_by" in row.note

    def test_delta_rs_rejects_every_feature_signal(self) -> None:
        for name in ("deletionVectors", "rowTracking", "catalogManaged", "v2Checkpoint"):
            assert (
                property_support(f"delta.feature.{name}").deltars_create is PropertyEffect.REJECTED
            )


class TestProbesAgainstInstalledDeltaRs:
    """If one of these fails, delta-rs changed and the matrix needs updating."""

    @pytest.mark.parametrize(
        "key",
        [k for k, r in PROPERTY_SUPPORT.items() if r.deltars_create is PropertyEffect.REJECTED],
    )
    def test_rejected_properties_really_are_rejected(self, key: str, tmp_path: Any) -> None:
        value = "true" if key.startswith("delta.enable") else "1"
        got = _probe_deltars_create(str(tmp_path / "t"), {key: value})
        assert got is PropertyEffect.REJECTED, f"{key}: matrix says rejected, engine says {got}"

    @pytest.mark.parametrize(
        "key",
        ["delta.appendOnly", "delta.enableChangeDataFeed", "delta.columnMapping.mode"],
    )
    def test_honored_properties_really_are_stored(self, key: str, tmp_path: Any) -> None:
        value = "name" if key.endswith("mode") else "true"
        got = _probe_deltars_create(str(tmp_path / "t"), {key: value})
        assert got is PropertyEffect.HONORED, f"{key}: matrix says honored, engine says {got}"

    def test_min_reader_version_still_panics(self, tmp_path: Any) -> None:
        """Documented as a crash. If this ever stops panicking, the matrix row
        and the routing that avoids it can both be relaxed."""
        got = _probe_deltars_create(str(tmp_path / "t"), {"delta.minReaderVersion": "3"})
        assert got is PropertyEffect.CRASH

    def test_configuration_is_ignored_on_append(self, tmp_path: Any) -> None:
        """Documented delta-rs behavior: properties passed to an append are
        silently discarded, which is why we refuse them up front."""
        from deltalake import DeltaTable, write_deltalake

        path = str(tmp_path / "t")
        write_deltalake(path, pa.table({"id": [1], "city": ["oslo"]}))
        write_deltalake(
            path,
            pa.table({"id": [2], "city": ["lima"]}),
            mode="append",
            configuration={"delta.appendOnly": "true"},
        )
        assert dict(DeltaTable(path).metadata().configuration) == {}


@pytest.mark.skipif(not ds.has_native(), reason="native extension not built")
class TestProbesAgainstKernel:
    @pytest.mark.parametrize(
        "properties",
        [
            {"delta.enableRowTracking": "true"},
            {"delta.enableInCommitTimestamps": "true"},
            {"delta.enableTypeWidening": "true"},
            {"delta.feature.deletionVectors": "supported"},
            {"delta.feature.v2Checkpoint": "supported"},
            {"custom.owner": "analytics"},
        ],
    )
    def test_kernel_accepts_what_delta_rs_rejects(
        self, properties: dict[str, str], tmp_path: Any
    ) -> None:
        from deltaswamp._native import create_table

        key = next(iter(properties))
        assert effect_for(key, Engine.DELTARS, ds.Operation.CREATE) is PropertyEffect.REJECTED
        target = str(tmp_path / key.replace(".", "_"))
        assert create_table(target, pa.schema(SCHEMA_FIELDS), properties=properties) == 0

    def test_kernel_refuses_both_layouts_at_once(self, tmp_path: Any) -> None:
        from deltaswamp._native import create_table

        with pytest.raises(ValueError, match="either partitioned or clustered"):
            create_table(
                str(tmp_path / "t"),
                pa.schema(SCHEMA_FIELDS),
                partition_by=["city"],
                cluster_by=["id"],
            )
