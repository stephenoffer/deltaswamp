"""Wave-4 docs/API audit: documented behaviour vs actual, and engine parity.

Each test pins one defect found by running the documented examples against
local tables, or by comparing what the same public call returns (or raises)
across engines.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, cast

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402
from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.catalog.base import Catalog  # noqa: E402
from deltaswamp.engine.deltars import DeltaRsEngine  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    DeltaSwampError,
    InvalidArgumentError,
    UnreachableTableError,
)


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


@pytest.fixture
def table(conn: Any, tmp_path: Any) -> Any:
    t = conn.create_table(
        str(tmp_path / "t"), pa.schema([("id", pa.int64()), ("city", pa.string())])
    )
    t.append(pa.table({"id": [1, 2, 3], "city": ["a", "b", "c"]}))
    return t


# ------------------------------------------------------------- identity


def test_connection_repr_hides_storage_secrets() -> None:
    conn = ds.connect("file://", storage_options={"aws_secret_access_key": "SEKRET"})
    assert "SEKRET" not in repr(conn)
    assert conn.storage_options["aws_secret_access_key"] == "SEKRET"


# ------------------------------------------------------ catalog listing


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.list_catalogs(),
        lambda c: c.list_schemas("x"),
        lambda c: c.list_tables("a", "b"),
    ],
)
def test_catalog_listing_refuses_as_a_deltaswamp_error(conn: Any, call: Any) -> None:
    with pytest.raises(UnreachableTableError):
        call(conn)


def test_a_minimal_plugin_catalog_refuses_missing_methods(tmp_path: Any) -> None:
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    class Minimal:
        """The documented plugin contract: resolve and list_tables only."""

        name = "mine"

        def resolve(self, ref: Any) -> Any:
            return FilesystemCatalog().resolve(ref)

        def list_tables(self, catalog: str, schema: str) -> list[Any]:
            return []

    conn = ds.connect(catalog=cast(Catalog, Minimal()))
    with pytest.raises(UnreachableTableError, match="list_catalogs"):
        conn.list_catalogs()
    with pytest.raises(UnreachableTableError, match="drop_table"):
        conn.drop_table("a.b.c")
    assert conn.list_tables("a", "b") == []


# ------------------------------------------------------------ hand-offs


def test_to_duckdb_view_is_queryable_by_name(table: Any) -> None:
    duckdb = pytest.importorskip("duckdb")
    table.to_duckdb(name="w4_docs_orders")
    assert duckdb.sql("SELECT count(*) FROM w4_docs_orders").fetchall() == [(3,)]


# ---------------------------------------------------- return-type parity


def test_rename_and_drop_column_return_a_dict(table: Any) -> None:
    table.set_properties({"delta.columnMapping.mode": "name"})
    renamed = table.rename_column("city", "town")
    assert isinstance(renamed, dict) and isinstance(renamed["version"], int)
    table.add_column(pa.schema([("extra", pa.string())]))
    dropped = table.drop_column("extra")
    assert isinstance(dropped, dict) and dropped["version"] == table.version


def test_files_has_one_layout_whichever_engine_lists_them(conn: Any, tmp_path: Any) -> None:
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("ts", pa.timestamp("us", "UTC")),
            ("s", pa.struct([("x", pa.int32())])),
            ("r", pa.string()),
            ("d", pa.date32()),
        ]
    )
    t = conn.create_table(str(tmp_path / "p"), schema, partition_by=["r", "d"])
    stamp = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    t.append(
        pa.table(
            {
                "id": [1, 2],
                "ts": [stamp, stamp],
                "s": [{"x": 1}, {"x": None}],
                "r": ["a", None],
                "d": [dt.date(2026, 1, 2)] * 2,
            }
        )
    )
    via_deltars = t.files()
    kernel = conn.router.engines[Engine.KERNEL]
    conn.router.engines.pop(Engine.DELTARS)
    try:
        t._invalidate()
        via_kernel = t.files()
    finally:
        conn.router.engines[Engine.DELTARS] = DeltaRsEngine()
    assert isinstance(via_deltars, pa.Table) and isinstance(via_kernel, pa.Table)
    assert kernel is conn.router.engines[Engine.KERNEL]
    shared = [c for c in via_kernel.column_names if c != "deletion_vector"]
    assert shared == via_deltars.column_names
    key = "path"
    assert sorted(via_kernel.select(shared).to_pylist(), key=lambda r: r[key]) == sorted(
        via_deltars.to_pylist(), key=lambda r: r[key]
    )


# ------------------------------------------------------ misuse errors


def test_write_table_unknown_mode_is_an_argument_error(conn: Any, tmp_path: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="mode"):
        conn.write_table(str(tmp_path / "x"), {"id": [1]}, mode="sideways")
    assert not os.path.exists(tmp_path / "x")


def test_plan_write_unknown_mode_is_an_argument_error(table: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="mode"):
        table.plan_write(mode="upsert")


def test_sql_unknown_engine_is_an_argument_error(conn: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="engine"):
        conn.sql("select 1", engine="spark")


def test_errors_are_all_deltaswamp_errors() -> None:
    assert issubclass(InvalidArgumentError, DeltaSwampError)


# ---------------------------------------------------------------- can()


@pytest.fixture
def kernel_only(conn: Any, tmp_path: Any) -> Any:
    """A table delta-rs cannot write (in-commit timestamps), so the kernel serves."""
    t = conn.create_table(
        str(tmp_path / "ict"),
        pa.schema([("id", pa.int64()), ("r", pa.string())]),
        partition_by=["r"],
        properties={"delta.enableInCommitTimestamps": "true"},
    )
    t.append({"id": [1], "r": ["a"]})
    return t


@pytest.mark.parametrize(
    ("operation", "shape", "call"),
    [
        (
            "append",
            {"schema_mode": "merge"},
            lambda t: t.append(pa.table({"id": [2], "r": ["b"], "x": [1]}), schema_mode="merge"),
        ),
        (
            "overwrite",
            {"partition_overwrite": "dynamic"},
            lambda t: t.overwrite(pa.table({"id": [2], "r": ["a"]}), partition_overwrite="dynamic"),
        ),
        (
            "append",
            {"writer_properties": {"compression": "ZSTD"}},
            lambda t: t.append(
                pa.table({"id": [2], "r": ["a"]}), writer_properties={"compression": "ZSTD"}
            ),
        ),
        (
            "overwrite",
            {"schema_mode": "overwrite"},
            lambda t: t.replace(pa.table({"q": [1]})),
        ),
    ],
)
def test_can_answers_for_the_call_it_describes(
    kernel_only: Any, operation: str, shape: dict[str, Any], call: Any
) -> None:
    verdict = kernel_only.can(operation, **shape)
    assert not verdict, verdict
    with pytest.raises(UnreachableTableError):
        call(kernel_only)


def test_can_still_says_yes_where_the_call_succeeds(kernel_only: Any) -> None:
    assert kernel_only.can("append", txn=("job", 1))
    kernel_only.append({"id": [5], "r": ["a"]}, txn=("job", 1))
    assert kernel_only.can("scan", predicate="id > 1")


# ----------------------------------------------------------------- cdf()


@pytest.fixture
def cdf_table(conn: Any, tmp_path: Any) -> Any:
    t = conn.create_table(
        str(tmp_path / "cdf"),
        pa.schema([("id", pa.int64()), ("r", pa.string())]),
        properties={"delta.enableChangeDataFeed": "true"},
    )
    t.append({"id": [1, 2], "r": ["a", "b"]})
    t.delete("id = 1")
    return t


def test_cdf_refuses_a_misspelt_option(cdf_table: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="start_version"):
        cdf_table.cdf(start_version=0)


def test_cdf_projection_keeps_the_change_metadata(cdf_table: Any) -> None:
    feed = pa.table(cdf_table.cdf(starting_version=0, columns=["id"]))
    assert feed.column_names == ["id", "_change_type", "_commit_version", "_commit_timestamp"]


def test_cdf_metadata_types_agree_across_engines(conn: Any, cdf_table: Any) -> None:
    via_deltars = pa.table(cdf_table.cdf(starting_version=0))
    saved = conn.router.engines.pop(Engine.DELTARS)
    try:
        via_kernel = pa.table(cdf_table.cdf(starting_version=0))
    finally:
        conn.router.engines[Engine.DELTARS] = saved
    for name, wanted in (
        ("_change_type", pa.string()),
        ("_commit_version", pa.int64()),
        ("_commit_timestamp", pa.timestamp("us", tz="UTC")),
    ):
        assert via_deltars.schema.field(name).type == wanted
        assert via_kernel.schema.field(name).type == wanted
    key = [("_commit_version", "ascending"), ("id", "ascending")]
    cols = ["id", "_change_type", "_commit_version", "_commit_timestamp"]
    assert (
        via_deltars.select(cols).sort_by(key).to_pylist()
        == via_kernel.select(cols).sort_by(key).to_pylist()
    )


def test_a_pinned_handle_says_so_in_its_repr(conn: Any, table: Any) -> None:
    assert "version=0" in repr(conn.open_table(table.location, version=0))
    assert "version=" not in repr(conn.open_table(table.location))


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.delete("id = 1", dryrun=True),
        lambda t: t.update(new_values={"city": "x"}, where="id = 1"),
    ],
)
def test_dml_typo_options_are_argument_errors(table: Any, call: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="unexpected option"):
        call(table)
    assert table.count() == 3
