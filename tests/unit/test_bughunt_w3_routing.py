"""Regression tests for the third-pass routing audit: does each routing claim
match what the engine really does?

Most of these build a real local table (delta-rs and the native kernel) and
check both the verdict and the call, because every defect here was a verdict
that said yes and a call that then failed (or a no that the engine could serve).
"""

from __future__ import annotations

import glob
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Capability, Engine, Operation
from deltaswamp.catalog import ResolvedTable
from deltaswamp.errors import FallbackRequiredError, UnreachableTableError
from deltaswamp.identity import parse_ref
from deltaswamp.router import Router

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")

needs_native = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

SCHEMA = pa.schema([("id", pa.int64()), ("v", pa.string())])


def _data(n: int = 0) -> Any:
    return pa.table({"id": pa.array([1 + n, 2 + n, 3 + n], pa.int64()), "v": ["a", "b", "c"]})


@pytest.fixture
def conn() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine

    return Connection(
        catalog=FilesystemCatalog(),
        router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
    )


@pytest.fixture
def kernel_only() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.kernel import KernelEngine

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.KERNEL: KernelEngine()})
    )


def _make(conn: Any, tmp_path: Path, props: dict[str, str] | None = None, **kw: Any) -> str:
    path = str(tmp_path / "t")
    conn.create_table(path, SCHEMA, properties=props, **kw)
    conn.table(path).append(_data())
    conn.table(path).append(_data(3))
    return path


class _Yes:
    supports_predicates = True

    def __init__(self, kind: Engine) -> None:
        self.kind = kind

    def supports(self, operation: Operation, table: ResolvedTable, **_: object) -> Capability:
        return Capability(operation, ok=True, engine=self.kind)


# ------------------------------------------------------------ column mapping


@needs_native
def test_optimize_on_a_column_mapped_table_is_refused_up_front(conn: Any, tmp_path: Path) -> None:
    """delta-rs raised "Column mapping is not supported for write operation
    'OPTIMIZE'" after being chosen; the mapping here is on a legacy protocol
    (writer 5), so no feature list names it."""
    path = _make(conn, tmp_path, {"delta.columnMapping.mode": "name"})
    t = conn.table(path)
    assert not t.can("optimize").ok
    assert "columnMapping" in t.can("optimize").reason
    assert not t.can("zorder").ok
    with pytest.raises(UnreachableTableError, match="columnMapping"):
        t.optimize()


@needs_native
def test_optimize_with_column_mapping_mode_none_still_routes(conn: Any, tmp_path: Path) -> None:
    """The feature listed but mode none: delta-rs optimizes it fine."""
    path = _make(conn, tmp_path, {"delta.feature.columnMapping": "supported"})
    assert conn.table(path).can("optimize").engine is Engine.DELTARS
    conn.table(path).optimize()


@needs_native
def test_add_column_on_a_column_mapped_table_routes_to_the_kernel(
    conn: Any, tmp_path: Path
) -> None:
    """delta-rs failed "not supported for write operation 'ADD COLUMN'"."""
    path = _make(conn, tmp_path, {"delta.columnMapping.mode": "name"})
    assert conn.table(path).can("add_column").engine is Engine.KERNEL
    conn.table(path).add_column([pa.field("x", pa.int32())])
    schema = json.loads(deltalake.DeltaTable(path).schema().to_json())
    added = next(f for f in schema["fields"] if f["name"] == "x")
    assert "delta.columnMapping.physicalName" in added["metadata"]
    assert conn.table(path).to_arrow().num_rows == 6


@needs_native
def test_generate_manifest_refused_on_legacy_column_mapping(conn: Any, tmp_path: Path) -> None:
    """deltars.generate looked only at the feature lists, so a writer-5 mapped
    table got a manifest pointing at physically-named Parquet columns."""
    path = _make(conn, tmp_path, {"delta.columnMapping.mode": "name"})
    with pytest.raises(UnreachableTableError, match="columnMapping"):
        conn.table(path).generate()
    assert not os.path.exists(os.path.join(path, "_symlink_format_manifest"))


@needs_native
def test_rename_column_without_column_mapping_is_refused_at_routing(
    conn: Any, tmp_path: Path
) -> None:
    path = _make(conn, tmp_path)
    for op in ("rename_column", "drop_column"):
        verdict = conn.table(path).can(op)
        assert not verdict.ok
        assert "column mapping" in verdict.reason


# --------------------------------------------------------------- row tracking


@needs_native
def test_overwrite_of_a_row_tracking_table_is_refused_up_front(conn: Any, tmp_path: Path) -> None:
    """The kernel's transaction refused at commit: "Remove actions are not yet
    supported on tables with rowTracking"."""
    path = _make(conn, tmp_path, {"delta.enableRowTracking": "true"})
    verdict = conn.table(path).can("overwrite")
    assert not verdict.ok
    assert "overwrite is not supported on a table with rowTracking" in verdict.reason
    with pytest.raises(UnreachableTableError, match="rowTracking"):
        conn.table(path).overwrite(_data(10))
    assert conn.table(path).to_arrow().num_rows == 6


@needs_native
def test_append_to_a_row_tracking_table_still_routes_to_the_kernel(
    conn: Any, tmp_path: Path
) -> None:
    path = _make(conn, tmp_path, {"delta.enableRowTracking": "true"})
    assert conn.table(path).can("append").engine is Engine.KERNEL


def test_suspended_row_tracking_does_not_block_a_kernel_overwrite() -> None:
    router = Router(engines={Engine.KERNEL: _Yes(Engine.KERNEL)})
    table = ResolvedTable(
        ref=parse_ref("/tmp/t"),
        location="/tmp/t",
        writer_features=frozenset({"rowTracking", "domainMetadata"}),
        properties={"delta.rowTrackingSuspended": "true"},
    )
    assert router.capability(Operation.OVERWRITE, table).ok


# ---------------------------------------------- log-level ops through delta-rs


@needs_native
def test_history_of_a_type_widening_table(conn: Any, tmp_path: Path) -> None:
    """typeWidening blocks delta-rs scans, not its log reads; history had no
    other engine and was refused outright."""
    path = _make(conn, tmp_path, {"delta.enableTypeWidening": "true"})
    assert conn.table(path).can("history").engine is Engine.DELTARS
    assert len(conn.table(path).history()) == 3


@needs_native
def test_history_of_a_vacuum_protocol_check_table(conn: Any, tmp_path: Path) -> None:
    path = _make(conn, tmp_path, {"delta.feature.vacuumProtocolCheck": "supported"})
    assert len(conn.table(path).history()) == 3


@needs_native
@pytest.mark.parametrize(
    "props",
    [
        {"delta.enableInCommitTimestamps": "true"},
        {"delta.enableTypeWidening": "true"},
        {"delta.enableRowTracking": "true"},
    ],
)
def test_vacuum_dry_run_is_served_but_a_real_vacuum_is_not(
    conn: Any, tmp_path: Path, props: dict[str, str]
) -> None:
    """A dry run deletes nothing and commits nothing; a real VACUUM commits
    VACUUM START/END, which delta-rs refuses on these tables."""
    path = _make(conn, tmp_path, props)
    assert conn.table(path).vacuum() == []  # the default is a dry run
    with pytest.raises(UnreachableTableError):
        conn.table(path).vacuum(dry_run=False, retention_hours=0)


@needs_native
def test_cleanup_and_log_compaction_of_a_clustered_table(conn: Any, tmp_path: Path) -> None:
    """Neither commits, and delta-rs keeps the clustering domain in both."""
    path = _make(conn, tmp_path, cluster_by=["id"])
    conn.table(path).compact_logs()
    assert glob.glob(os.path.join(path, "_delta_log", "*.compacted.json"))
    conn.table(path).cleanup_metadata()
    kernel = conn.router.engines[Engine.KERNEL]
    domain = kernel.snapshot(conn.table(path)._enrich()).domain_metadata("delta.clustering")
    assert json.loads(domain) == {"clusteringColumns": [["id"]]}
    assert conn.table(path).to_arrow().num_rows == 6


@needs_native
def test_cleanup_is_still_refused_on_in_commit_timestamps(conn: Any, tmp_path: Path) -> None:
    path = _make(conn, tmp_path, {"delta.enableInCommitTimestamps": "true"})
    assert not conn.table(path).can("cleanup_metadata").ok


def test_history_exemptions_do_not_leak_into_writes() -> None:
    from deltaswamp.engine.deltars import DeltaRsEngine

    router = Router(engines={Engine.DELTARS: DeltaRsEngine()})
    table = ResolvedTable(
        ref=parse_ref("/tmp/t"),
        location="/tmp/t",
        reader_features=frozenset({"typeWidening"}),
        writer_features=frozenset({"typeWidening"}),
    )
    assert router.capability(Operation.HISTORY, table).ok
    assert not router.capability(Operation.SCAN, table).ok
    assert not router.capability(Operation.OPTIMIZE, table).ok


# ------------------------------------------------------------ kernel checkpoint


def _legacy(path: Path, writer: int) -> str:
    from deltalake import write_deltalake

    p = str(path / f"w{writer}")
    config = {
        3: None,
        4: {"delta.enableChangeDataFeed": "true"},
    }[writer]
    write_deltalake(p, _data(), configuration=config)
    if writer == 3:
        deltalake.DeltaTable(p).alter.add_constraint({"pos": "id > 0"})
    return p


@needs_native
@pytest.mark.parametrize("writer", [3, 4])
def test_kernel_checkpoint_refuses_a_legacy_writer_protocol(
    kernel_only: Any, tmp_path: Path, writer: int
) -> None:
    """The kernel's checkpoint writer fails "Feature 'checkConstraints' is not
    supported" on writer 3-6; supports() never checked, so it was claimed."""
    p = _legacy(tmp_path, writer)
    verdict = kernel_only.table(p).can("checkpoint")
    assert not verdict.ok
    assert "legacy writer protocol" in verdict.reason


@needs_native
def test_kernel_checkpoint_refuses_invariants(kernel_only: Any, tmp_path: Path) -> None:
    from deltalake import DeltaTable, Field, Schema, write_deltalake
    from deltalake.schema import PrimitiveType

    p = str(tmp_path / "inv")
    DeltaTable.create(
        p,
        Schema(
            [
                Field(
                    "id",
                    PrimitiveType("long"),
                    metadata={"delta.invariants": '{"expression": {"expression": "id > 0"}}'},
                ),
                Field("v", PrimitiveType("string")),
            ]
        ),
    )
    write_deltalake(p, _data(), mode="append")
    verdict = kernel_only.table(p).can("checkpoint")
    assert not verdict.ok and "invariants" in verdict.reason


def test_kernel_checkpoint_refuses_unwritable_writer_features() -> None:
    from deltaswamp.engine.kernel import KernelEngine

    if not KernelEngine.available():
        pytest.skip("native extension not built")
    router = Router(engines={Engine.KERNEL: KernelEngine()})
    table = ResolvedTable(
        ref=parse_ref("/tmp/t"),
        location="/tmp/t",
        min_reader_version=3,
        min_writer_version=7,
        writer_features=frozenset({"identityColumns"}),
    )
    verdict = router.capability(Operation.CHECKPOINT, table)
    assert not verdict.ok and "identityColumns" in verdict.reason


@needs_native
def test_kernel_checkpoint_of_a_modern_table_still_routes(kernel_only: Any, tmp_path: Path) -> None:
    from deltalake import write_deltalake

    p = str(tmp_path / "ntz")
    write_deltalake(
        p,
        pa.table({"id": [1], "ts": pa.array([0], pa.timestamp("us"))}),
    )
    kernel_only.table(p).checkpoint()


# ----------------------------------------------------------------- appendOnly


@needs_native
@pytest.mark.parametrize("op", ["overwrite", "replace_where", "delete", "update", "restore"])
def test_append_only_table_refuses_removals_at_routing(conn: Any, tmp_path: Path, op: str) -> None:
    """delta-rs was chosen and then failed at commit ("includes Remove action
    with data change but Delta table is append-only")."""
    path = _make(conn, tmp_path, {"delta.appendOnly": "true"})
    verdict = conn.table(path).can(op)
    assert not verdict.ok
    assert "append-only" in verdict.reason
    assert conn.table(path).can("append").ok


# ------------------------------------------------------------------- UniForm


def test_uniform_table_refuses_direct_data_writes() -> None:
    """The kernel appended to a UniForm table, leaving its Iceberg view stale;
    delta-rs and the kernel's metadata path both already refused."""
    router = Router(engines={Engine.KERNEL: _Yes(Engine.KERNEL)})
    table = ResolvedTable(
        ref=parse_ref("/tmp/t"),
        location="/tmp/t",
        properties={"delta.universalFormat.enabledFormats": "iceberg"},
    )
    for op in (Operation.APPEND, Operation.OVERWRITE, Operation.DELETE):
        verdict = router.capability(op, table)
        assert not verdict.ok and "UniForm" in verdict.reason
    assert router.capability(Operation.SCAN, table).ok


# ------------------------------------------------------------------ remedies


def test_open_error_that_says_the_log_is_missing_does_not_suggest_the_warehouse() -> None:
    router = Router(engines={Engine.KERNEL: _Yes(Engine.KERNEL)}, allow_sql_fallback=False)
    table = ResolvedTable(
        ref=parse_ref("main.s.t"),
        location="s3://b/t",
        open_error="DeltaError: Not a Delta table: No files in log segment",
    )
    verdict = router.capability(Operation.SCAN, table)
    assert not verdict.ok
    assert "allow_sql_fallback" not in verdict.remedy
    assert "s3://b/t" in verdict.remedy
    with pytest.raises(UnreachableTableError) as info:
        router.engine_for(Operation.SCAN, table)
    assert not isinstance(info.value, FallbackRequiredError)


def test_open_error_from_vending_still_suggests_the_warehouse() -> None:
    router = Router(engines={Engine.KERNEL: _Yes(Engine.KERNEL)}, allow_sql_fallback=False)
    table = ResolvedTable(
        ref=parse_ref("main.s.t"),
        location="s3://b/t",
        open_error="CredentialError: vending refused",
    )
    verdict = router.capability(Operation.SCAN, table)
    assert "allow_sql_fallback" in verdict.remedy


def test_open_error_on_a_glue_table_does_not_suggest_the_warehouse() -> None:
    router = Router(engines={Engine.KERNEL: _Yes(Engine.KERNEL)}, allow_sql_fallback=False)
    table = ResolvedTable(
        ref=parse_ref("glue://db.t"),
        location="s3://b/t",
        open_error="CredentialError: AccessDenied",
    )
    verdict = router.capability(Operation.SCAN, table)
    assert "allow_sql_fallback" not in verdict.remedy


def test_path_table_reason_does_not_blame_the_disabled_fallback() -> None:
    """A path has no Unity Catalog name; the warehouse would refuse it with
    "a SQL warehouse addresses tables by name" even with the fallback on."""

    class _No(_Yes):
        def supports(self, operation: Operation, table: ResolvedTable, **_: object) -> Capability:
            return Capability(operation, ok=False, reason="nope")

    router = Router(engines={Engine.DELTARS: _No(Engine.DELTARS)}, allow_sql_fallback=False)
    table = ResolvedTable(ref=parse_ref("/tmp/t"), location="/tmp/t")
    verdict = router.capability(Operation.OPTIMIZE, table)
    assert "fallback is disabled" not in verdict.reason
    assert "by name" in verdict.reason
    assert "register the table in Unity Catalog" in verdict.remedy
    with pytest.raises(UnreachableTableError) as info:
        router.engine_for(Operation.OPTIMIZE, table)
    assert not isinstance(info.value, FallbackRequiredError)


def test_databricks_only_operation_on_a_path_is_not_fallback_required() -> None:
    router = Router(engines={}, allow_sql_fallback=False)
    table = ResolvedTable(ref=parse_ref("/tmp/t"), location="/tmp/t")
    with pytest.raises(UnreachableTableError) as info:
        router.engine_for(Operation.CLONE, table)
    assert not isinstance(info.value, FallbackRequiredError)
    named = ResolvedTable(ref=parse_ref("main.s.t"), location="/tmp/t")
    with pytest.raises(FallbackRequiredError):
        router.engine_for(Operation.CLONE, named)


def test_non_delta_hms_table_does_not_suggest_the_warehouse() -> None:
    router = Router(engines={}, allow_sql_fallback=False)
    table = ResolvedTable(
        ref=parse_ref("hms://host:9083/db/t"), location="s3://b/t", data_source_format="PARQUET"
    )
    verdict = router.capability(Operation.SCAN, table)
    assert not verdict.ok and "allow_sql_fallback" not in verdict.remedy


# ---------------------------------------------------------------- table.py


@needs_native
def test_polars_frame_with_map_columns_appends(conn: Any, tmp_path: Path) -> None:
    """Polars reads a map as LargeList<Struct<key, value>>, which delta-rs
    refused to cast back ("Cannot cast field m ... to Map")."""
    pl = pytest.importorskip("polars")
    m = pa.map_(pa.string(), pa.int64())
    schema = pa.schema(
        [("id", pa.int64()), ("m", m), ("s", pa.struct([("mm", m)])), ("l", pa.list_(m))]
    )
    path = str(tmp_path / "maps")
    conn.create_table(path, schema)
    conn.table(path).append(
        pa.table(
            {
                "id": [1, 2],
                "m": pa.array([[("a", 1)], None], m),
                "s": pa.array([{"mm": [("b", 2)]}, None], schema.field("s").type),
                "l": pa.array([[[("c", 3)]], []], schema.field("l").type),
            }
        )
    )
    frame = conn.table(path).to_polars()
    assert isinstance(frame, pl.DataFrame)
    conn.table(path).append(frame)
    rows = sorted(conn.table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
    assert rows[0] == rows[1]
    assert rows[0]["m"] == [("a", 1)] and rows[2]["m"] is None
    assert rows[0]["s"] == {"mm": [("b", 2)]} and rows[2]["s"] is None
    assert rows[0]["l"] == [[("c", 3)]] and rows[2]["l"] == []


@needs_native
def test_append_leaves_generated_columns_of_a_legacy_table_to_the_engine(
    conn: Any, tmp_path: Path
) -> None:
    """On writer 4 no feature list names generatedColumns, so the missing
    generated column was filled with nulls and delta-rs's check failed."""
    from deltalake import DeltaTable, Field, Schema
    from deltalake.schema import PrimitiveType

    path = str(tmp_path / "gen")
    DeltaTable.create(
        path,
        Schema(
            [
                Field("id", PrimitiveType("long")),
                Field(
                    "g", PrimitiveType("long"), metadata={"delta.generationExpression": "id * 2"}
                ),
            ]
        ),
    )
    conn.table(path).append(pa.table({"id": pa.array([1, 2], pa.int64())}))
    rows = sorted(conn.table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
    assert rows == [{"id": 1, "g": 2}, {"id": 2, "g": 4}]


def test_duckdb_extra_declares_pytz() -> None:
    """DuckDB imports pytz to fetch TIMESTAMP WITH TIME ZONE values; without it
    fetchall() on any tz-aware column failed "Required module 'pytz'"."""
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    assert any(dep.startswith("pytz") for dep in extras["duckdb"])
    assert any(dep.startswith("pytz") for dep in extras["all"])


# -------------------------------------------------------------- catalog/ossuc


def test_ossuc_azure_http_endpoint_allows_http() -> None:
    from deltaswamp.catalog.ossuc import _parse_vended

    creds = _parse_vended(
        {
            "url": "http://127.0.0.1:10000/devstoreaccount1/cont/t",
            "azure_user_delegation_sas": {"sas_token": "sv=1"},
        }
    )
    assert creds.secrets["azure_allow_http"] == "true"
    tls = _parse_vended(
        {
            "url": "abfss://cont@acct.dfs.core.windows.net/t",
            "azure_user_delegation_sas": {"sas_token": "sv=1"},
        }
    )
    assert "azure_allow_http" not in tls.secrets


def test_staging_credentials_for_an_http_azure_endpoint_allow_http() -> None:
    from deltaswamp.governance import staging_storage_options

    options, _ = staging_storage_options(
        "http://127.0.0.1:10000/devstoreaccount1/cont/t",
        [
            {
                "prefix": "http://127.0.0.1:10000/devstoreaccount1/cont/t",
                "operation": "READ_WRITE",
                "config": {"azure.sas-token": "sv=1"},
            }
        ],
    )
    assert options["azure_allow_http"] == "true"


def test_struct_field_names_are_quoted_in_type_text() -> None:
    """``struct<a b:int>`` / ``struct<a:b:int>`` do not parse back as the type."""
    from deltaswamp.governance import delta_schema_to_columns

    schema = {
        "type": "struct",
        "fields": [
            {
                "name": "s",
                "type": {
                    "type": "struct",
                    "fields": [
                        {"name": "a b", "type": "integer", "nullable": True, "metadata": {}},
                        {"name": "x:y", "type": "long", "nullable": True, "metadata": {}},
                        {"name": "ok_1", "type": "string", "nullable": True, "metadata": {}},
                        {"name": "q`t", "type": "string", "nullable": True, "metadata": {}},
                    ],
                },
                "nullable": True,
                "metadata": {},
            }
        ],
    }
    (column,) = delta_schema_to_columns(schema)
    assert column["type_text"] == "struct<`a b`:int,`x:y`:bigint,ok_1:string,`q``t`:string>"


@needs_native
def test_repair_dry_run_on_an_in_commit_timestamp_table(conn: Any, tmp_path: Path) -> None:
    path = _make(conn, tmp_path, {"delta.enableInCommitTimestamps": "true"})
    os.remove(glob.glob(os.path.join(path, "*.parquet"))[0])
    result = conn.table(path).repair(dry_run=True)
    assert result["dry_run"] is True and len(result["files_removed"]) == 1
    with pytest.raises(UnreachableTableError):
        conn.table(path).repair()
