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
from deltaswamp.capability import Engine
from deltaswamp.properties import PROPERTY_SUPPORT, PropertyEffect, effect_for, property_support

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


#: Keys delta-rs accepts at create but mishandles, so the matrix treats them
#: as rejected to keep those creates on the kernel. Each is probed for the
#: mishandling instead; if that stops, the row can go back to honored.
_AVOIDED_ON_DELTARS = frozenset({"delta.enableDeletionVectors"})


class TestProbesAgainstInstalledDeltaRs:
    def test_deltars_still_stamps_variant_type_on_deletion_vector_tables(
        self, tmp_path: Any
    ) -> None:
        import json

        from deltalake import write_deltalake

        path = tmp_path / "t"
        write_deltalake(
            str(path),
            pa.table({"id": [1]}),
            configuration={"delta.enableDeletionVectors": "true"},
        )
        log = (path / "_delta_log" / "00000000000000000000.json").read_text()
        protocol = next(json.loads(x)["protocol"] for x in log.splitlines() if '"protocol"' in x)
        assert "variantType" in protocol.get("readerFeatures", []), (
            "delta-rs no longer stamps variantType; delta.enableDeletionVectors can be "
            "honored on its create path again"
        )

    """If one of these fails, delta-rs changed and the matrix needs updating."""

    @pytest.mark.parametrize(
        "key",
        [
            k
            for k, r in PROPERTY_SUPPORT.items()
            if r.deltars_create is PropertyEffect.REJECTED and k not in _AVOIDED_ON_DELTARS
        ],
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


_SET_VALUES = {
    "delta.columnMapping.mode": "name",
    "delta.checkpointPolicy": "v2",
    "delta.minWriterVersion": "4",
    "delta.minReaderVersion": "2",
    "delta.isolationLevel": "WriteSerializable",
    "delta.parquet.compression.codec": "zstd",
    "delta.parquet.format.version": "2.12.0",
    "delta.universalFormat.enabledFormats": "iceberg",
    "delta.dataSkippingStatsColumns": "id",
    "delta.targetFileSize": "134217728",
}


def _set_value(key: str) -> str:
    if key in _SET_VALUES:
        return _SET_VALUES[key]
    if key.endswith("Duration"):
        return "interval 7 days"
    if key.startswith(("delta.enable", "delta.autoOptimize", "delta.appendOnly")) or key in (
        "delta.checkpoint.writeStatsAsJson",
        "delta.checkpoint.writeStatsAsStruct",
        "delta.randomizeFilePrefixes",
        "delta.tuneFileSizesForRewrites",
    ):
        return "true"
    return "5"


class TestSetProbesAgainstInstalledDeltaRs:
    """The ALTER column drifted once already: every SET key was recorded as
    rejected, while deltalake 1.6.5 took most of them."""

    @pytest.mark.parametrize("key", sorted(PROPERTY_SUPPORT))
    def test_set_claim_matches_the_engine(self, key: str, tmp_path: Any) -> None:
        from deltalake import DeltaTable, write_deltalake

        row = PROPERTY_SUPPORT[key]
        if row.deltars_set is PropertyEffect.UNSUPPORTED:
            pytest.skip("not probed")
        path = str(tmp_path / "t")
        write_deltalake(path, pa.table({"id": [1], "city": ["oslo"]}))
        try:
            DeltaTable(path).alter.set_table_properties({key: _set_value(key)})
            accepted = True
        except Exception:
            accepted = False
        except BaseException as exc:
            if type(exc).__name__ != "PanicException":
                raise
            accepted = False
        if key == "delta.enableDeletionVectors":
            # Accepted, but it stamps a spurious variantType feature into the
            # protocol, so the matrix records it as rejected on purpose.
            protocol = DeltaTable(path).protocol()
            assert accepted and "variantType" in (protocol.writer_features or [])
            assert row.deltars_set is PropertyEffect.REJECTED
            return
        if key == "delta.minReaderVersion":
            # Accepted, and it leaves protocol (2, 2): reader version 2 means
            # column mapping, which needs writer 5. Rejected on purpose.
            assert accepted and DeltaTable(path).protocol().min_writer_version < 5
            assert row.deltars_set is PropertyEffect.REJECTED
            return
        claimed = row.deltars_set is not PropertyEffect.REJECTED
        assert accepted is claimed, (
            f"{key}: matrix says {row.deltars_set}, engine accepted={accepted}"
        )
