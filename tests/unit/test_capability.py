"""The conformance matrix.

These tests exist so the coverage claims cannot rot. Each one encodes a fact
established from kernel/delta-rs source, with the source of truth named in the
docstring so a future reader can re-verify it rather than trust it.
"""

from __future__ import annotations

import pytest
from deltaswamp.capability import (
    DATABRICKS_ONLY_OPERATIONS,
    FEATURE_CONFLICTS,
    FEATURE_DEPENDENCIES,
    FEATURE_SUPPORT,
    OPERATION_ENGINES,
    PROPERTY_SUPPORT,
    READ_OPERATIONS,
    Engine,
    Operation,
    Support,
    TableFeature,
    feature_from_wire,
)


class TestMatrixCompleteness:
    def test_every_feature_has_a_row(self) -> None:
        for feature in TableFeature:
            assert feature in FEATURE_SUPPORT, f"{feature.value} missing from FEATURE_SUPPORT"

    def test_every_operation_is_routed(self) -> None:
        for op in Operation:
            assert op in OPERATION_ENGINES, f"{op.value} has no routing entry"

    def test_every_row_is_self_consistent(self) -> None:
        for feature, row in FEATURE_SUPPORT.items():
            assert row.feature is feature

    def test_dependencies_reference_known_features(self) -> None:
        for feature, deps in FEATURE_DEPENDENCIES.items():
            assert feature in FEATURE_SUPPORT
            for dep in deps:
                assert dep in FEATURE_SUPPORT

    def test_conflicts_are_symmetric(self) -> None:
        """If A forbids B, B must forbid A -- an asymmetry would let us build
        a table one engine accepts and another rejects."""
        for feature, conflicts in FEATURE_CONFLICTS.items():
            for other in conflicts:
                if other in FEATURE_CONFLICTS:
                    assert feature in FEATURE_CONFLICTS[other], (
                        f"{feature.value} forbids {other.value} but not vice versa"
                    )


class TestEnginesFailInOppositeDirections:
    """The central architectural finding, asserted rather than asserted-in-prose."""

    def test_kernel_reads_everything_deltars_reads(self) -> None:
        """Kernel's read coverage is a strict superset of delta-rs's.

        Kernel evaluates support per operation and ignores writer-only features
        on the read path; delta-rs does a flat set-difference. The one exception
        is timestampNanos, a non-standard delta-rs-only extension.
        """
        for feature, row in FEATURE_SUPPORT.items():
            if feature is TableFeature.TIMESTAMP_NANOS:
                continue
            if row.deltars_read is not Support.NO:
                assert row.kernel_read is not Support.NO, (
                    f"{feature.value}: delta-rs reads it but kernel does not -- "
                    "that would break the 'kernel is the default read engine' premise"
                )

    def test_there_are_features_only_kernel_can_read(self) -> None:
        """If this ever reached zero, the Rust extension would not earn its keep."""
        kernel_only = [
            f.value
            for f, r in FEATURE_SUPPORT.items()
            if r.kernel_read is not Support.NO and r.deltars_read is Support.NO
        ]
        assert len(kernel_only) >= 15, kernel_only

    def test_scan_prefers_kernel(self) -> None:
        assert OPERATION_ENGINES[Operation.SCAN].primary is Engine.KERNEL

    def test_merge_never_routes_to_kernel(self) -> None:
        """delta_kernel 0.28 has no MERGE implementation at all."""
        assert Engine.KERNEL not in OPERATION_ENGINES[Operation.MERGE].engines

    @pytest.mark.parametrize(
        "op", [Operation.OPTIMIZE, Operation.VACUUM, Operation.RESTORE, Operation.REPAIR]
    )
    def test_maintenance_never_routes_to_kernel(self, op: Operation) -> None:
        """Kernel implements none of these."""
        assert Engine.KERNEL not in OPERATION_ENGINES[op].engines

    def test_publish_is_kernel_only(self) -> None:
        """Only kernel implements the staged->published transition."""
        assert OPERATION_ENGINES[Operation.PUBLISH].engines == (Engine.KERNEL,)

    def test_log_compaction_avoids_kernel_stub(self) -> None:
        """kernel's log_compaction_writer exists in the API but is a no-op (kernel#2337)."""
        assert Engine.KERNEL not in OPERATION_ENGINES[Operation.LOG_COMPACTION].engines


class TestEasilyMissedFacts:
    def test_vacuum_protocol_check_blocks_deltars_reads(self) -> None:
        """It is a ReaderWriter feature, so it blocks reads -- not merely VACUUM."""
        row = FEATURE_SUPPORT[TableFeature.VACUUM_PROTOCOL_CHECK]
        assert row.kind.value == "readerWriter"
        assert row.deltars_read is Support.NO

    def test_domain_metadata_makes_tables_deltars_unwritable(self) -> None:
        """This silently excludes every liquid-clustered and row-tracked table."""
        assert FEATURE_SUPPORT[TableFeature.DOMAIN_METADATA].deltars_write is Support.NO
        for dependent in (TableFeature.ROW_TRACKING, TableFeature.CLUSTERING):
            assert TableFeature.DOMAIN_METADATA in FEATURE_DEPENDENCIES[dependent]
            assert FEATURE_SUPPORT[dependent].deltars_write is Support.NO

    def test_clustering_wire_name(self) -> None:
        """It serialises as 'clustering', not 'clusteredTable'."""
        assert TableFeature.CLUSTERING.value == "clustering"

    def test_catalog_managed_requires_in_commit_timestamp(self) -> None:
        assert FEATURE_DEPENDENCIES[TableFeature.CATALOG_MANAGED] == frozenset(
            {TableFeature.IN_COMMIT_TIMESTAMP}
        )

    def test_catalog_managed_is_unreadable_by_deltars(self) -> None:
        """delta-rs cannot even open one (delta-rs#4549)."""
        row = FEATURE_SUPPORT[TableFeature.CATALOG_MANAGED]
        assert row.deltars_read is Support.NO
        assert row.kernel_read is Support.YES

    def test_deletion_vectors_write_is_partial_on_both_engines(self) -> None:
        """Kernel installs descriptors but computes no bitmaps; delta-rs never
        emits DVs at all. Authoring them is work this project has to do."""
        row = FEATURE_SUPPORT[TableFeature.DELETION_VECTORS]
        assert row.kernel_write is Support.PARTIAL
        assert row.deltars_write is Support.PARTIAL

    @pytest.mark.parametrize(
        "feature",
        [TableFeature.COLLATIONS, TableFeature.CHECKPOINT_PROTECTION],
    )
    def test_features_with_no_kernel_variant_block_both_engines(
        self, feature: TableFeature
    ) -> None:
        """No kernel 0.28 variant -> classified Unknown -> writes blocked."""
        row = FEATURE_SUPPORT[feature]
        assert row.kernel_write is Support.NO
        assert row.deltars_write is Support.NO

    def test_checkconstraints_deltars_ahead_of_kernel(self) -> None:
        """A case where delta-rs is the more capable engine, so routing must not
        assume kernel is always better."""
        row = FEATURE_SUPPORT[TableFeature.CHECK_CONSTRAINTS]
        assert row.kernel_write is Support.NO
        assert row.deltars_write is Support.YES


class TestForwardCompatibility:
    def test_unknown_feature_returns_none_rather_than_raising(self) -> None:
        """Unknown features must be tolerable on the read path; choking here
        would break every table that adopts a feature newer than this release."""
        assert feature_from_wire("someFutureFeature2027") is None

    def test_known_feature_round_trips(self) -> None:
        assert feature_from_wire("catalogManaged") is TableFeature.CATALOG_MANAGED
        assert feature_from_wire("clustering") is TableFeature.CLUSTERING


class TestDatabricksOnlyOperations:
    def test_they_route_to_sql_alone(self) -> None:
        for op in DATABRICKS_ONLY_OPERATIONS:
            assert OPERATION_ENGINES[op].engines == (Engine.SQL,)

    @pytest.mark.parametrize(
        "op",
        [
            Operation.DROP_COLUMN,
            Operation.RENAME_COLUMN,
            Operation.DROP_FEATURE,
            Operation.REORG,
            Operation.CLONE,
        ],
    )
    def test_expected_members(self, op: Operation) -> None:
        assert op in DATABRICKS_ONLY_OPERATIONS

    def test_no_read_operation_is_databricks_only(self) -> None:
        """Every read must have at least one direct-to-storage path."""
        assert not (DATABRICKS_ONLY_OPERATIONS & READ_OPERATIONS)


class TestDocsMatchTheMatrices:
    """The published tables must not drift from the code they describe."""

    @staticmethod
    def _conformance() -> str:
        from pathlib import Path

        return (Path(__file__).parents[2] / "docs" / "conformance.md").read_text()

    def test_every_property_row_is_documented(self) -> None:
        text = self._conformance()
        missing = [key for key in PROPERTY_SUPPORT if f"`{key}`" not in text]
        assert not missing, f"undocumented properties: {sorted(missing)}"

    def test_every_write_operation_is_documented(self) -> None:
        from deltaswamp.capability import Operation as Op

        text = self._conformance()
        write_ops = [
            Op.APPEND,
            Op.OVERWRITE,
            Op.REPLACE_WHERE,
            Op.MERGE_SCHEMA,
            Op.CREATE,
            Op.DELETE,
            Op.UPDATE,
            Op.MERGE,
        ]
        missing = [op.value for op in write_ops if f"`{op.value}`" not in text]
        assert not missing, f"undocumented write operations: {sorted(missing)}"

    def test_the_crash_property_is_called_out(self) -> None:
        """delta.minReaderVersion panics, and a reader must not miss that."""
        assert "crashes" in self._conformance()
