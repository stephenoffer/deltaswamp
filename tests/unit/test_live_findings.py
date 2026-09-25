"""Defects found by running every Databricks table shape through deltaswamp.

Each test here pins a behaviour that was wrong against a real Unity Catalog
workspace and right nowhere else -- the offline suite passed throughout. The
docstrings say what was observed, so the next person to touch the routing can
tell a deliberate refusal from an accident.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from deltaswamp.capability import (
    FEATURE_SUPPORT,
    OPERATION_ENGINES,
    Capability,
    Engine,
    FeatureKind,
    Operation,
    Support,
    TableFeature,
)
from deltaswamp.catalog import ResolvedTable, TableType
from deltaswamp.catalog.databricks import DatabricksUnityCatalog
from deltaswamp.credentials.base import Cloud, Credentials, StaticCredentialProvider
from deltaswamp.engine.deltars import DeltaRsEngine, _pin_s3_endpoint
from deltaswamp.engine.kernel import KernelEngine
from deltaswamp.engine.sql import SqlEngine
from deltaswamp.identity import RefKind, TableRef, parse_ref
from deltaswamp.router import Router


def resolved(**kwargs: Any) -> ResolvedTable:
    kwargs.setdefault("location", "s3://bucket/t")
    kwargs.setdefault("data_source_format", "DELTA")
    ref = TableRef(kind=RefKind.CATALOG, catalog="main", schema="s", table="t")
    return ResolvedTable(ref=ref, **kwargs)


class Yes:
    """An engine that accepts everything, so the router's own gates are visible."""

    supports_distributed_scan = False
    supports_predicates = True

    def __init__(self, kind: Engine) -> None:
        self.kind = kind

    def supports(self, operation: Operation, table: ResolvedTable, **_: Any) -> Capability:
        return Capability(operation, ok=True, engine=self.kind)

    def available(self) -> bool:
        return True


def router(*, fallback: bool) -> Router:
    kinds = (Engine.KERNEL, Engine.DELTARS, Engine.SQL)
    return Router(engines={k: Yes(k) for k in kinds}, allow_sql_fallback=fallback)


# ------------------------------------------------------------- S3 endpoint


class TestS3Endpoint:
    """delta-rs spent ~3s per process probing EC2 metadata for a region it was
    given: with no endpoint it builds an AWS SDK config whose region chain
    ignores `aws_region`. An explicit endpoint skips the SDK entirely."""

    def test_vended_keys_get_a_regional_endpoint(self) -> None:
        options = {"aws_access_key_id": "k", "aws_region": "us-west-2"}
        _pin_s3_endpoint(options)
        assert options["aws_endpoint"] == "https://s3.us-west-2.amazonaws.com"

    def test_china_regions_use_their_own_domain(self) -> None:
        options = {"aws_access_key_id": "k", "aws_region": "cn-north-1"}
        _pin_s3_endpoint(options)
        assert options["aws_endpoint"] == "https://s3.cn-north-1.amazonaws.com.cn"

    @pytest.mark.parametrize("key", ["aws_endpoint", "AWS_ENDPOINT_URL", "endpoint"])
    def test_an_explicit_endpoint_is_never_overridden(self, key: str) -> None:
        options = {"aws_access_key_id": "k", "aws_region": "us-west-2", key: "http://minio"}
        _pin_s3_endpoint(options)
        assert options[key] == "http://minio"
        assert len(options) == 3

    def test_no_region_or_no_keys_means_no_change(self) -> None:
        for options in ({"aws_access_key_id": "k"}, {"aws_region": "us-west-2"}):
            before = dict(options)
            _pin_s3_endpoint(options)
            assert options == before


# ------------------------------------------------------------------- CDF


class TestChangeDataFeed:
    def test_kernel_serves_cdf_first(self) -> None:
        """delta-rs cannot decode the CDF files Databricks writes ("cannot skip
        miniblock of size 256") while claiming the operation."""
        assert OPERATION_ENGINES[Operation.CDF].primary is Engine.KERNEL

    def test_delta_rs_refuses_column_mapped_cdf_up_front(self) -> None:
        """Refused in supports(), so the router can try the next engine."""
        table = resolved(
            properties={"delta.enableChangeDataFeed": "true", "delta.columnMapping.mode": "name"}
        )
        verdict = DeltaRsEngine().supports(Operation.CDF, table)
        assert not verdict.ok
        assert "column mapping" in verdict.reason

    def test_kernel_refuses_catalog_managed_cdf_up_front(self) -> None:
        table = resolved(
            reader_features=frozenset({"catalogManaged"}),
            writer_features=frozenset({"catalogManaged", "inCommitTimestamp"}),
            properties={"delta.enableChangeDataFeed": "true"},
        )
        verdict = KernelEngine().supports(Operation.CDF, table)
        assert not verdict.ok
        assert "commit tail" in verdict.reason


def test_restore_on_deletion_vectors_is_refused_in_supports() -> None:
    table = resolved(
        reader_features=frozenset({"deletionVectors"}),
        writer_features=frozenset({"deletionVectors"}),
    )
    verdict = DeltaRsEngine().supports(Operation.RESTORE, table)
    assert not verdict.ok
    assert "delta-rs#4613" in verdict.reason


# ------------------------------------------------------------ add_feature


class TestAddFeature:
    """delta-rs was first for every feature, took only its own enum (TypeError
    on a string), and added rowTracking without domainMetadata -- leaving a
    table neither engine would write."""

    def test_delta_rs_takes_features_it_can_then_write(self) -> None:
        verdict = DeltaRsEngine().supports(
            Operation.ADD_FEATURE, resolved(), features=["changeDataFeed", "v2Checkpoint"]
        )
        assert verdict.ok

    @pytest.mark.parametrize("feature", ["rowTracking", "domainMetadata", "typeWidening"])
    def test_delta_rs_declines_features_it_cannot_write(self, feature: str) -> None:
        verdict = DeltaRsEngine().supports(Operation.ADD_FEATURE, resolved(), features=[feature])
        assert not verdict.ok

    def test_local_add_feature_round_trip(self, tmp_path: Any) -> None:
        pa = pytest.importorskip("pyarrow")
        deltalake = pytest.importorskip("deltalake")
        import deltaswamp as ds

        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": [1, 2]}))
        table = ds.connect().table(path)
        table.add_feature(["changeDataFeed", "v2Checkpoint"])
        protocol = deltalake.DeltaTable(path).protocol()
        assert {"changeDataFeed", "v2Checkpoint"} <= set(protocol.writer_features or [])
        # Still writable afterwards, which is the point.
        ds.connect().table(path).append(pa.table({"id": [3]}))
        assert ds.connect().table(path).count() == 3


# ----------------------------------------------------- vending and policies


class TestFineGrainedAccess:
    """A row filter or mask makes vending refuse ("not supported on assigned
    clusters"), but the manifest still says HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT.
    Trusting the manifest meant a confident yes, then a CredentialError, with
    the warehouse never tried."""

    @staticmethod
    def info(**kwargs: Any) -> Any:
        from types import SimpleNamespace

        kwargs.setdefault("columns", [])
        return SimpleNamespace(**kwargs)

    def test_row_filter_is_detected(self) -> None:
        from types import SimpleNamespace

        info = self.info(row_filter=SimpleNamespace(function_name="main.s.rf"))
        assert DatabricksUnityCatalog._access_policy(info) == "row filter main.s.rf"

    def test_column_masks_are_detected(self) -> None:
        from types import SimpleNamespace

        column = SimpleNamespace(name="city", mask=SimpleNamespace(function_name="main.s.m"))
        info = self.info(row_filter=None, columns=[column, SimpleNamespace(name="id", mask=None)])
        assert DatabricksUnityCatalog._access_policy(info) == "column mask on city (main.s.m)"

    def test_plain_table_has_no_policy(self) -> None:
        assert DatabricksUnityCatalog._access_policy(self.info(row_filter=None)) is None

    def test_policy_routes_to_the_warehouse(self) -> None:
        table = resolved(external_read_supported=False, access_policy="row filter main.s.rf")
        assert router(fallback=True).capability(Operation.SCAN, table).engine is Engine.SQL
        refused = router(fallback=False).capability(Operation.SCAN, table)
        assert not refused.ok
        assert "row filter main.s.rf" in refused.reason


def test_vended_gcs_bearer_tokens_are_refused_by_delta_rs() -> None:
    """deltalake ignores the token and falls back to the GCE metadata server."""
    credentials = Credentials(
        cloud=Cloud.GCP, url="gs://b/t", expires_at=None, secrets={"google_bearer_token": "x"}
    )
    table = ResolvedTable(
        ref=parse_ref("gs://b/t"),
        location="gs://b/t",
        data_source_format="DELTA",
        credential_provider=StaticCredentialProvider(credentials),
    )
    verdict = DeltaRsEngine().supports(Operation.MERGE, table)
    assert not verdict.ok
    assert "bearer token" in verdict.reason


# ---------------------------------------------------- unreadable metadata


class TestUnreadableLog:
    """A geometry column makes both engines fail to parse the schema. With the
    fallback on, the open error was ignored and the kernel chosen anyway."""

    def test_direct_engines_are_skipped_even_with_the_fallback(self) -> None:
        table = resolved(open_error="Unsupported Delta table type: 'geometry(OGC:CRS84)'")
        assert router(fallback=True).capability(Operation.SCAN, table).engine is Engine.SQL

    def test_protocol_is_recovered_from_catalog_properties(self) -> None:
        from deltaswamp.table import _protocol_from_properties

        table = resolved(
            properties={
                "delta.minReaderVersion": "3",
                "delta.minWriterVersion": "7",
                "delta.feature.geospatial": "supported",
                "delta.feature.rowTracking": "supported",
                "delta.feature.deletionVectors": "supported",
                "delta.enableRowTracking": "true",
            }
        )
        got = _protocol_from_properties(table)
        assert got["min_reader_version"] == 3
        assert got["writer_features"] == {"geospatial", "rowTracking", "deletionVectors"}
        # rowTracking is writer-only, so it must not be reported as a reader feature.
        assert got["reader_features"] == {"geospatial", "deletionVectors"}

    def test_geospatial_is_not_claimed_readable(self) -> None:
        assert FEATURE_SUPPORT[TableFeature.GEOSPATIAL].kernel_read is Support.NO


def test_shredded_variants_fail_inside_the_call(monkeypatch: Any) -> None:
    """Databricks enables shredding on every VARIANT table but shreds a file only
    when its values share a shape, so the kernel is tried -- and must fail where
    the read can still be retried, not halfway through the caller's iteration."""
    pa = pytest.importorskip("pyarrow")
    features = frozenset({"variantType", "variantShredding"})
    table = resolved(reader_features=features, writer_features=features)
    assert KernelEngine().supports(Operation.SCAN, table).ok

    def shredded(*_: Any, **__: Any) -> Any:
        schema = pa.schema([("id", pa.int64())])

        def batches() -> Any:
            if schema is not None:
                raise RuntimeError("The field v presumed to be of Variant type might be shredded")
            yield pa.record_batch([], schema=schema)

        return pa.RecordBatchReader.from_batches(schema, batches())

    monkeypatch.setattr(KernelEngine, "_scan", shredded)
    with pytest.raises(Exception, match="shredded"):
        KernelEngine().scan(table)


class TestCollations:
    """`name = 'oslo'` on a UTF8_LCASE column found one row through the kernel
    and two through Databricks: a wrong answer, not an error."""

    def test_collations_are_writer_only(self) -> None:
        assert FEATURE_SUPPORT[TableFeature.COLLATIONS].kind is FeatureKind.WRITER

    def test_predicates_on_collated_tables_do_not_go_direct(self) -> None:
        table = resolved(writer_features=frozenset({"collations"}))
        predicated = router(fallback=True).capability(
            Operation.SCAN, table, needs=frozenset({"predicates"})
        )
        assert predicated.engine is Engine.SQL
        assert router(fallback=True).capability(Operation.SCAN, table).engine is Engine.KERNEL


# ------------------------------------------------------------ warehouse


class TestWarehouseRefusals:
    def test_zorder_on_a_clustered_table(self) -> None:
        """Databricks answers DELTA_CLUSTERING_WITH_ZORDER_BY."""
        table = resolved(writer_features=frozenset({"clustering", "domainMetadata"}))
        verdict = SqlEngine._table_type_refusal(Operation.ZORDER, table)
        assert verdict is not None
        assert not verdict.ok

    @pytest.mark.parametrize("kind", [TableType.VIEW, TableType.MATERIALIZED_VIEW])
    @pytest.mark.parametrize("operation", [Operation.HISTORY, Operation.DETAIL])
    def test_log_reads_on_views(self, kind: TableType, operation: Operation) -> None:
        """Both fail on the warehouse with EXPECT_TABLE_NOT_VIEW."""
        verdict = SqlEngine._table_type_refusal(operation, resolved(table_type=kind))
        assert verdict is not None
        assert not verdict.ok


# ------------------------------------------------------ read-only retries


class TestReadRetries:
    """delta-rs cannot parse the stats Databricks writes for a CLONE (-1 where it
    wants a u64), so history on a deep clone failed despite being claimed."""

    def test_a_failing_read_moves_to_the_next_engine(self, tmp_path: Any) -> None:
        import deltaswamp as ds
        from deltaswamp.errors import EngineFallbackWarning

        pa = pytest.importorskip("pyarrow")
        deltalake = pytest.importorskip("deltalake")
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": [1]}))
        conn = ds.connect()
        table = conn.table(path)

        class Broken(DeltaRsEngine):
            def history(self, *_: Any, **__: Any) -> Any:
                raise RuntimeError("Invalid JSON in file stats")

        class Backup(Yes):
            def history(self, *_: Any, **__: Any) -> Any:
                return [{"version": 0}]

        conn.router.engines[Engine.DELTARS] = Broken()
        conn.router.engines[Engine.ICEBERG] = Backup(Engine.ICEBERG)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert table.history() == [{"version": 0}]
        assert any(issubclass(w.category, EngineFallbackWarning) for w in caught)

    def test_the_original_error_surfaces_when_nothing_else_can(self, tmp_path: Any) -> None:
        import deltaswamp as ds

        pa = pytest.importorskip("pyarrow")
        deltalake = pytest.importorskip("deltalake")
        path = str(tmp_path / "t")
        deltalake.write_deltalake(path, pa.table({"id": [1]}))
        conn = ds.connect()

        class Broken(DeltaRsEngine):
            def history(self, *_: Any, **__: Any) -> Any:
                raise RuntimeError("Invalid JSON in file stats")

        conn.router.engines[Engine.DELTARS] = Broken()
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            conn.table(path).history()


class TestTypeChanges:
    """Databricks refuses even INT -> BIGINT until type widening is on."""

    def test_refused_without_type_widening(self) -> None:
        table = resolved(writer_features=frozenset({"appendOnly"}))
        verdict = SqlEngine._table_type_refusal(Operation.ALTER_COLUMN_TYPE, table)
        assert verdict is not None
        assert "enableTypeWidening" in verdict.remedy

    def test_allowed_with_it(self) -> None:
        on = resolved(
            writer_features=frozenset({"typeWidening"}), reader_features=frozenset({"typeWidening"})
        )
        assert SqlEngine._table_type_refusal(Operation.ALTER_COLUMN_TYPE, on) is None
        by_property = resolved(
            writer_features=frozenset({"appendOnly"}),
            properties={"delta.enableTypeWidening": "true"},
        )
        assert SqlEngine._table_type_refusal(Operation.ALTER_COLUMN_TYPE, by_property) is None


class TestManagedCreateThroughTheWarehouse:
    """Databricks refuses its staging-table API to connectors it has not
    allowlisted, so managed creates needed a warehouse route."""

    def test_ddl(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from tests.unit.test_sql_engine import engine

        eng, rec, _ = engine()
        schema = pa.schema(
            [pa.field("id", pa.int64(), nullable=False), pa.field("weird col", pa.string())]
        )
        eng.create_managed(
            "main.s.t",
            schema,
            cluster_by=["id"],
            properties={"delta.enableChangeDataFeed": "true"},
            comment="it's here",
        )
        assert rec.last == (
            "CREATE TABLE `main`.`s`.`t` (`id` BIGINT NOT NULL, `weird col` STRING) USING DELTA"
            " CLUSTER BY (`id`) COMMENT 'it\\'s here'"
            " TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
        )

    def test_partitioned_and_clustered_is_refused(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from deltaswamp.errors import UnreachableTableError

        from tests.unit.test_sql_engine import engine

        eng, _, _ = engine()
        with pytest.raises(UnreachableTableError):
            eng.create_managed(
                "main.s.t", pa.schema([("id", pa.int64())]), partition_by=["id"], cluster_by=["id"]
            )


def test_history_timestamps_do_not_depend_on_the_engine(tmp_path: Any) -> None:
    """delta-rs returned epoch milliseconds and the warehouse a datetime."""
    import datetime as dt

    import deltaswamp as ds

    pa = pytest.importorskip("pyarrow")
    deltalake = pytest.importorskip("deltalake")
    path = str(tmp_path / "t")
    deltalake.write_deltalake(path, pa.table({"id": [1]}))
    conn = ds.connect()

    class Warehouse(DeltaRsEngine):
        def history(self, *_: Any, **__: Any) -> Any:
            return [{"version": 0, "timestamp": dt.datetime(2026, 1, 1, tzinfo=dt.UTC)}]

    conn.router.engines[Engine.DELTARS] = Warehouse()
    assert conn.table(path).history()[0]["timestamp"] == 1767225600000


class TestTableLevelProhibitions:
    """Operations the table itself forbids, so no engine may claim them."""

    @pytest.mark.parametrize(
        "operation",
        [Operation.DELETE, Operation.UPDATE, Operation.OVERWRITE, Operation.REPLACE_WHERE],
    )
    def test_append_only(self, operation: Operation) -> None:
        table = resolved(properties={"delta.appendOnly": "true"})
        verdict = router(fallback=True).capability(operation, table)
        assert not verdict.ok
        assert "append-only" in verdict.reason
        assert router(fallback=True).capability(Operation.APPEND, table).ok

    @pytest.mark.parametrize("operation", [Operation.TIME_TRAVEL, Operation.CDF, Operation.RESTORE])
    def test_history_escapes_an_access_policy(self, operation: Operation) -> None:
        table = resolved(external_read_supported=False, access_policy="row filter main.s.rf")
        assert not router(fallback=True).capability(operation, table).ok
        assert router(fallback=True).capability(Operation.HISTORY, table).ok


class TestWarehouseOnlyShapes:
    """Refused outright while naming the fallback as the remedy, so enabling
    it changed nothing."""

    def test_catalog_managed_alter_goes_to_the_warehouse(self) -> None:
        table = resolved(
            reader_features=frozenset({"catalogManaged"}),
            writer_features=frozenset({"catalogManaged", "inCommitTimestamp"}),
        )
        assert router(fallback=True).capability(Operation.ADD_COLUMN, table).engine is Engine.SQL
        assert not router(fallback=False).capability(Operation.ADD_COLUMN, table).ok

    def test_foreign_and_hive_metastore_go_to_the_warehouse(self) -> None:
        foreign = resolved(table_type=TableType.FOREIGN)
        assert router(fallback=True).capability(Operation.SCAN, foreign).engine is Engine.SQL
        hive = ResolvedTable(
            ref=TableRef(kind=RefKind.CATALOG, catalog="hive_metastore", schema="s", table="t"),
            location="s3://b/t",
            data_source_format="DELTA",
        )
        assert router(fallback=True).capability(Operation.SCAN, hive).engine is Engine.SQL
        assert not router(fallback=False).capability(Operation.SCAN, hive).ok


@pytest.mark.parametrize("operation", [Operation.DROP_COLUMN, Operation.RENAME_COLUMN])
def test_column_drops_and_renames_need_column_mapping(operation: Operation) -> None:
    """Databricks answers DELTA_UNSUPPORTED_DROP_COLUMN without it."""
    plain = resolved(writer_features=frozenset({"appendOnly"}))
    verdict = SqlEngine._table_type_refusal(operation, plain)
    assert verdict is not None
    assert "columnMapping" in verdict.remedy
    mapped = resolved(
        writer_features=frozenset({"columnMapping"}),
        reader_features=frozenset({"columnMapping"}),
        properties={"delta.columnMapping.mode": "name"},
    )
    assert SqlEngine._table_type_refusal(operation, mapped) is None
