"""Regression tests for defects found auditing the public surface (table.py)."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Capability, Engine, Operation
from deltaswamp.catalog import ResolvedTable, TableType
from deltaswamp.errors import (
    CorruptTableError,
    DeltaSwampError,
    InvalidReferenceError,
    UnreachableTableError,
)
from deltaswamp.identity import parse_ref
from deltaswamp.table import InvalidArgumentError

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


@pytest.fixture
def conn() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router
    from deltaswamp.table import Connection

    return Connection(
        catalog=FilesystemCatalog(),
        router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
    )


@pytest.fixture
def path(tmp_path: Any) -> str:
    from deltalake import write_deltalake

    p = str(tmp_path / "tbl")
    write_deltalake(p, pa.table({"id": [1, 2, 3], "city": ["oslo", "lima", "cairo"]}))
    write_deltalake(p, pa.table({"id": [4], "city": ["rome"]}), mode="append")
    return p


def _rows(table: Any) -> int:
    return int(table.to_arrow().num_rows)


# ------------------------------------------------------------------ scan args


class TestScanArguments:
    def test_empty_column_list_is_refused_instead_of_aborting(self, conn: Any, path: str) -> None:
        # Used to abort the interpreter with a non-unwinding Rust panic.
        with pytest.raises(InvalidArgumentError, match="columns"):
            conn.open_table(path).to_arrow(columns=[])

    def test_a_single_column_name_is_accepted(self, conn: Any, path: str) -> None:
        assert conn.open_table(path).to_arrow(columns="city").column_names == ["city"]

    def test_version_and_timestamp_together_are_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError, match="not both"):
            conn.open_table(path).to_arrow(version=0, timestamp="2020-01-01T00:00:00Z")

    def test_invalid_argument_is_also_a_value_error(self) -> None:
        assert issubclass(InvalidArgumentError, ValueError)
        assert issubclass(InvalidArgumentError, DeltaSwampError)

    def test_negative_version_is_refused_clearly(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError, match=">= 0"):
            conn.open_table(path).to_arrow(version=-1)

    def test_negative_version_on_open_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError):
            conn.open_table(path, version=-1)

    def test_bool_version_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError):
            conn.open_table(path, version=True)

    def test_negative_limit_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError, match="limit"):
            conn.open_table(path).scan(limit=-1)

    def test_blank_scan_predicate_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError, match="blank"):
            conn.open_table(path).count(predicate="  ")

    def test_negative_head_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError):
            conn.open_table(path).head(-1)


class TestPinnedVersion:
    def test_timestamp_overrides_the_pinned_version(self, conn: Any, path: str) -> None:
        # Used to send version=0 and the timestamp together: "not both".
        t = conn.open_table(path, version=0)
        assert t.to_arrow(timestamp="2999-01-01T00:00:00Z").num_rows == 4

    def test_pinned_read_routes_as_time_travel(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        seen: list[Operation] = []
        original = conn.router.engine_for

        def spy(operation: Operation, *args: Any, **kwargs: Any) -> Any:
            seen.append(operation)
            return original(operation, *args, **kwargs)

        monkeypatch.setattr(conn.router, "engine_for", spy)
        assert conn.open_table(path, version=0).to_arrow().num_rows == 3
        assert seen[-1] is Operation.TIME_TRAVEL

    def test_plan_scan_routes_timestamp_as_time_travel(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        seen: list[tuple[Operation, frozenset[str]]] = []
        original = conn.router.engine_for

        def spy(operation: Operation, table: Any, *, needs: Any = frozenset(), **kw: Any) -> Any:
            seen.append((operation, needs))
            return original(operation, table, needs=needs, **kw)

        monkeypatch.setattr(conn.router, "engine_for", spy)
        # Only the routing is under test.
        with contextlib.suppress(Exception):
            conn.open_table(path, version=0).plan_scan(timestamp="2999-01-01T00:00:00Z")
        op, needs = seen[-1]
        assert op is Operation.TIME_TRAVEL
        assert "timestamp_travel" in needs


# ------------------------------------------------------------------ metadata


class TestMetadata:
    def test_schema_is_a_pyarrow_schema(self, conn: Any, path: str) -> None:
        # The kernel returned an arro3 Schema, which compares unequal to pyarrow's.
        schema = conn.open_table(path).schema()
        assert isinstance(schema, pa.Schema)
        assert schema.names == ["id", "city"]

    def test_schema_without_a_snapshot_does_not_read_the_table(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        t = conn.open_table(path)
        consumed: list[int] = []

        class Stream:
            def __init__(self) -> None:
                self._table = pa.table({"id": [1, 2], "city": ["a", "b"]})

            def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
                def batches() -> Any:
                    for b in self._table.to_batches():
                        consumed.append(b.num_rows)
                        yield b

                reader = pa.RecordBatchReader.from_batches(self._table.schema, batches())
                return reader.__arrow_c_stream__(requested_schema)

        class NoSnapshot:
            def scan(self, *_: Any, **__: Any) -> Any:
                return Stream()

        monkeypatch.setattr(t, "_engine", lambda *a, **k: NoSnapshot())
        assert t.schema().names == ["id", "city"]
        assert consumed == []

    def test_history_zero_is_empty_on_every_engine(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)

        class Warehouse:
            def history(self, table: Any, *, limit: Any = None) -> list[dict[str, Any]]:
                # The SQL engine only appends LIMIT for a truthy limit.
                return [{"version": 1}, {"version": 0}]

        t._engine = lambda *a, **k: Warehouse()
        assert t.history(0) == []

    def test_negative_history_limit_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError):
            conn.open_table(path).history(-1)


# ---------------------------------------------------------------- enrichment


def _detail_returning(metadata_id: str, calls: list[int]) -> Any:
    def detail(table: Any, *, version: Any = None) -> dict[str, Any]:
        calls.append(1)
        return {"min_reader_version": 1, "min_writer_version": 2, "metadata_id": metadata_id}

    return detail


class TestEnrichment:
    def test_identity_mismatch_is_refused_on_every_call(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        from deltaswamp.table import Table

        resolved = ResolvedTable(
            ref=parse_ref(path),
            location=path,
            table_type=TableType.MANAGED,
            table_uuid="catalog-id",
        )
        t = Table(conn, resolved)
        kernel = conn.router.engines[Engine.KERNEL]
        monkeypatch.setattr(kernel, "detail", _detail_returning("log-id", []))
        with pytest.raises(CorruptTableError):
            t.capabilities()
        # The second call used to find the cache filled and carry on.
        with pytest.raises(CorruptTableError):
            t.to_arrow()

    def test_a_transient_open_failure_is_retried(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        t = conn.open_table(path)
        kernel = conn.router.engines[Engine.KERNEL]
        deltars = conn.router.engines[Engine.DELTARS]
        real_kernel, real_deltars = kernel.detail, deltars.detail

        def boom(*_: Any, **__: Any) -> Any:
            raise OSError("503 from storage")

        monkeypatch.setattr(kernel, "detail", boom)
        monkeypatch.setattr(deltars, "detail", boom)
        assert t.can(Operation.SCAN).ok is False
        monkeypatch.setattr(kernel, "detail", real_kernel)
        monkeypatch.setattr(deltars, "detail", real_deltars)
        # Used to stay refused for the life of the handle.
        assert t.can(Operation.SCAN).ok is True
        assert t.resolved.open_error is None
        assert t.count() == 4

    def test_open_error_is_cleared_after_invalidate(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        t = conn.open_table(path)
        t._resolved = dataclasses.replace(t._resolved, open_error="stale")
        t._invalidate()
        assert t.can(Operation.SCAN).ok is True


# --------------------------------------------------------------------- writes


class TestWriteArguments:
    def test_blank_delete_predicate_is_refused(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        with pytest.raises(InvalidArgumentError, match="blank"):
            t.delete("")
        assert t.count() == 4

    def test_blank_delete_predicate_never_reaches_the_warehouse(self, conn: Any, path: str) -> None:
        # The SQL engine builds WHERE only for a truthy predicate: "" deleted all.
        t = conn.open_table(path)
        reached: list[Any] = []

        class Warehouse:
            def delete(self, table: Any, predicate: Any = None, **_: Any) -> dict[str, Any]:
                reached.append(predicate)
                return {}

        t._engine = lambda *a, **k: Warehouse()
        with pytest.raises(InvalidArgumentError):
            t.delete("   ")
        assert reached == []

    def test_blank_overwrite_predicate_is_not_a_full_overwrite(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        with pytest.raises(InvalidArgumentError):
            t.overwrite(pa.table({"id": [9], "city": ["x"]}), predicate="")
        assert t.count() == 4

    def test_blank_update_predicate_is_refused(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        with pytest.raises(InvalidArgumentError):
            t.update(new_values={"city": "x"}, predicate="")
        assert sorted(t.to_arrow().column("city").to_pylist()) == ["cairo", "lima", "oslo", "rome"]

    def test_unknown_append_schema_mode_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError, match="schema_mode"):
            conn.open_table(path).append(pa.table({"id": [5], "city": ["x"]}), schema_mode="mrege")

    def test_append_cannot_overwrite_the_schema(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        with pytest.raises(InvalidArgumentError):
            t.append(pa.table({"id": [5], "city": ["x"]}), schema_mode="overwrite")
        assert t.count() == 4

    def test_malformed_txn_is_refused(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        with pytest.raises(InvalidArgumentError, match="txn"):
            t.append(pa.table({"id": [5], "city": ["x"]}), txn="abc")
        with pytest.raises(InvalidArgumentError, match="txn"):
            t.append(pa.table({"id": [5], "city": ["x"]}), txn=("", 1))
        assert t.count() == 4

    def test_empty_set_properties_makes_no_commit(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        before = t.version
        t.set_properties({})
        assert t.version == before

    def test_empty_unset_properties_makes_no_commit(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        before = t.version
        t.unset_properties([])
        assert t.version == before

    def test_empty_add_constraint_is_refused_clearly(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError):
            conn.open_table(path).add_constraint({})

    def test_empty_z_order_is_refused_clearly(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError):
            conn.open_table(path).z_order([])

    def test_optimize_accepts_a_single_zorder_column(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.optimize(zorder_by="id")
        assert t.count() == 4


class TestChangeFeedArguments:
    def test_reversed_cdf_range_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError, match="before"):
            conn.open_table(path).cdf(starting_version=3, ending_version=1)

    def test_negative_changes_start_is_refused(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidArgumentError):
            next(iter(conn.open_table(path).changes(-1)))


# ----------------------------------------------------------------- connection


class TestConnection:
    def test_unknown_sql_engine_reads_nothing(self, conn: Any, path: str, monkeypatch: Any) -> None:
        from deltaswamp.table import Table

        def fail(*_: Any, **__: Any) -> Any:
            raise AssertionError("scanned before the engine was checked")

        monkeypatch.setattr(Table, "scan", fail)
        with pytest.raises(UnreachableTableError, match="engine"):
            conn.sql("select 1", tables={"t": path}, engine="spark")

    def test_sql_statement_without_a_result_returns_none(self, conn: Any) -> None:
        pytest.importorskip("duckdb")
        assert conn.sql("create table z(a int)") is None

    def test_sql_closes_its_duckdb_connection(self, conn: Any, path: str, monkeypatch: Any) -> None:
        duckdb = pytest.importorskip("duckdb")
        opened: list[Any] = []
        real = duckdb.connect

        def tracking(*a: Any, **k: Any) -> Any:
            con = real(*a, **k)
            opened.append(con)
            return con

        monkeypatch.setattr(duckdb, "connect", tracking)
        assert conn.sql("select count(*) n from t", tables={"t": path}).num_rows == 1
        with pytest.raises(duckdb.ConnectionException):
            opened[0].execute("select 1")

    def test_table_exists_does_not_use_static_storage_options(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        from deltalake import DeltaTable

        # A catalog table's storage is reachable only with its vended
        # credentials; the static-options probe said it was absent.
        monkeypatch.setattr(DeltaTable, "is_deltatable", staticmethod(lambda *a, **k: False))
        assert conn.table_exists(path) is True

    def test_table_exists_is_false_for_a_missing_path(self, conn: Any, tmp_path: Any) -> None:
        assert conn.table_exists(str(tmp_path / "missing")) is False

    def test_drop_table_refuses_a_path(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidReferenceError, match="storage path"):
            conn.drop_table(path)

    def test_drop_table_unsupported_is_a_refusal(self, conn: Any) -> None:
        with pytest.raises(UnreachableTableError):
            conn.drop_table("a.b.c")

    def test_register_table_refuses_a_path_name(
        self, conn: Any, path: str, monkeypatch: Any
    ) -> None:
        registered: list[Any] = []

        class Lifecycle:
            def path_credentials(self, *_: Any) -> Any:
                return None

            def register_table(self, ref: Any, *_: Any, **__: Any) -> Any:
                registered.append(ref)
                return ResolvedTable(ref=ref, location=path)

        monkeypatch.setattr(conn, "_lifecycle_catalog", lambda what: Lifecycle())
        with pytest.raises(InvalidReferenceError, match="storage path"):
            conn.register_table(path, path)
        assert registered == []

    def test_volume_refuses_a_path(self, conn: Any, path: str) -> None:
        with pytest.raises(InvalidReferenceError, match="storage path"):
            conn.volume(path)

    def test_create_at_a_path_keeps_the_comment_on_delta_rs(self, conn: Any, tmp_path: Any) -> None:
        from deltalake import DeltaTable

        p = str(tmp_path / "c")
        conn.create_table(p, pa.schema([("a", pa.int64())]), comment="hello")
        assert DeltaTable(p).metadata().description == "hello"

    def test_create_at_a_path_keeps_the_comment_on_the_kernel(
        self, conn: Any, tmp_path: Any
    ) -> None:
        from deltalake import DeltaTable

        p = str(tmp_path / "k")
        conn.create_table(p, pa.schema([("a", pa.int64())]), comment="hello", cluster_by=["a"])
        assert DeltaTable(p).metadata().description == "hello"


# ----------------------------------------------------------------- conversions


class TestConversions:
    def test_ray_dataset_accepts_a_limit(self, conn: Any, path: str) -> None:
        pytest.importorskip("ray.data")
        assert conn.open_table(path).to_ray_dataset(limit=2).count() == 2

    def test_ray_dataset_of_an_empty_table_keeps_its_schema(self, conn: Any, tmp_path: Any) -> None:
        pytest.importorskip("ray.data")
        p = str(tmp_path / "empty")
        t = conn.create_table(p, pa.schema([("a", pa.int64())]))
        schema = t.to_ray_dataset().schema()
        assert schema is not None and list(schema.names) == ["a"]


# ------------------------------------------------------------------- merger


class TestMerger:
    def test_copying_a_merger_does_not_recurse(self) -> None:
        from deltaswamp.table import _InvalidatingMerger

        merger = _InvalidatingMerger(object(), lambda: None)
        copied = copy.copy(merger)
        assert copied._builder is merger._builder


# ------------------------------------------------------------- governance


class TestGovernanceRefresh:
    def test_row_filter_refreshes_the_capability_manifest(self, conn: Any, path: str) -> None:
        from deltaswamp.table import Table

        class Warehouse:
            def available(self) -> bool:
                return True

            def set_row_filter(self, *_: Any) -> None:
                pass

            def supports(self, op: Any, table: Any, **_: Any) -> Capability:
                return Capability(op, ok=True, engine=Engine.SQL)

        class Catalog:
            name = "fake"

            def resolve(self, ref: Any) -> ResolvedTable:
                # After the filter, the catalog withdraws direct reads.
                return ResolvedTable(ref=ref, location=path, external_read_supported=False)

        conn.catalog = Catalog()
        conn.router.engines[Engine.SQL] = Warehouse()
        conn.router.allow_sql_fallback = True
        ref = parse_ref("main.s.t")
        t = Table(conn, ResolvedTable(ref=ref, location=path, external_read_supported=True))
        assert t.can(Operation.SCAN).engine is Engine.KERNEL
        t.set_row_filter("main.s.f", ["city"])
        assert t.resolved.external_read_supported is False
        assert t.can(Operation.SCAN).engine is Engine.SQL
