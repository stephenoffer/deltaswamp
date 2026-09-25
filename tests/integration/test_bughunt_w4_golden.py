"""Protocol conformance against tables written by other writers (w4_golden).

Each test pins a defect found by running every table of the delta-kernel-rs
golden/acceptance corpus through deltaswamp. Small corpus tables are vendored
under tests/data/golden (Apache-2.0); the rest are rebuilt here in miniature.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

import deltaswamp as ds  # noqa: E402
from deltaswamp import errors  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden"


def _copy(name: str, tmp_path: Path) -> str:
    dest = tmp_path / name
    shutil.copytree(GOLDEN / name, dest)
    return str(dest)


def _partition_values(path: str) -> list[dict[str, str]]:
    out = []
    for f in sorted(glob.glob(os.path.join(path, "_delta_log", "*.json"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            action = json.loads(line)
            if "add" in action:
                out.append(action["add"]["partitionValues"])
    return out


def _dec(*values: str) -> pa.Array:
    return pa.array([Decimal(v) for v in values], pa.decimal128(5, 2))


# --------------------------------------------------- negative decimal partitions


def _need_native() -> None:
    if not ds.has_native():
        pytest.skip("needs the native kernel engine")


def test_append_negative_fractional_decimal_partition_is_not_corrupted(tmp_path: Path) -> None:
    _need_native()
    path = str(tmp_path / "t")
    conn = ds.connect()
    t = conn.write_table(path, pa.table({"p": _dec("2.25"), "v": [1]}), partition_by=["p"])
    t.append(pa.table({"p": _dec("-1.50", "-0.05"), "v": [2, 3]}))
    values = {pv["p"] for pv in _partition_values(path)}
    assert values == {"2.25", "-1.50", "-0.05"}, values
    got = sorted(conn.table(path).to_arrow().column("p").to_pylist())
    assert got == [Decimal("-1.50"), Decimal("-0.05"), Decimal("2.25")]


def test_write_table_creating_with_negative_decimal_partition(tmp_path: Path) -> None:
    _need_native()
    path = str(tmp_path / "t")
    t = ds.connect().write_table(path, pa.table({"p": _dec("-1.50"), "v": [1]}), partition_by=["p"])
    assert [pv["p"] for pv in _partition_values(path)] == ["-1.50"]
    assert t.to_arrow().column("p").to_pylist() == [Decimal("-1.50")]


def test_overwrite_negative_fractional_decimal_partition(tmp_path: Path) -> None:
    _need_native()
    path = str(tmp_path / "t")
    conn = ds.connect()
    t = conn.write_table(path, pa.table({"p": _dec("2.25"), "v": [1]}), partition_by=["p"])
    t.overwrite(pa.table({"p": _dec("-0.50"), "v": [9]}))
    assert conn.table(path).to_arrow().column("p").to_pylist() == [Decimal("-0.50")]


def test_positive_decimal_partitions_still_route_to_deltars(tmp_path: Path) -> None:
    path = str(tmp_path / "t")
    t = ds.connect().write_table(path, pa.table({"p": _dec("2.25"), "v": [1]}), partition_by=["p"])
    assert t._data_needs(pa.table({"p": _dec("1.50"), "v": [2]})) == frozenset()
    assert t._data_needs(pa.table({"p": _dec("-1.50"), "v": [2]})) == frozenset(
        {"negative_decimal_partition_values"}
    )


def test_merge_negative_decimal_partition_refused_before_commit(tmp_path: Path) -> None:
    path = str(tmp_path / "t")
    conn = ds.connect()
    t = conn.write_table(path, pa.table({"p": _dec("2.25"), "v": [1]}), partition_by=["p"])
    source = pa.table({"p": _dec("-1.50"), "v": [2]})
    with pytest.raises(errors.DeltaSwampError, match="negative_decimal_partition_values"):
        t.merge(source, "s.v = t.v", source_alias="s", target_alias="t")
    # Nothing was committed, and the table still reads.
    assert conn.table(path).version == 1
    assert conn.table(path).count() == 1


def test_update_to_negative_decimal_partition_is_not_corrupted(tmp_path: Path) -> None:
    _need_native()
    path = str(tmp_path / "t")
    conn = ds.connect()
    t = conn.write_table(path, pa.table({"p": _dec("2.25"), "v": [1]}), partition_by=["p"])
    t.update(new_values={"p": Decimal("-3.75")}, predicate="v = 1")
    assert "-3.-75" not in {pv["p"] for pv in _partition_values(path)}
    assert conn.table(path).to_arrow().column("p").to_pylist() == [Decimal("-3.75")]


# ----------------------------------------------------------- deletion vectors


def test_files_of_dv_table_carries_deletion_vectors(tmp_path: Path) -> None:
    _need_native()
    t = ds.connect().table(_copy("table-with-dv-small", tmp_path))
    files = pa.table(t.files())
    assert "deletion_vector" in files.column_names
    dv = json.loads(files.column("deletion_vector")[0].as_py())
    assert files.column("num_records")[0].as_py() - dv["cardinality"] == t.count() == 8


# --------------------------------------------------------- change data feed


def test_cdf_disabled_mid_range_is_a_clear_error(tmp_path: Path) -> None:
    t = ds.connect().table(_copy("table-with-cdf", tmp_path))
    with pytest.raises(errors.UnreachableTableError, match="not enabled at version 2"):
        t.cdf(starting_version=0)
    # With no start the gap is refused too, rather than silently skipped.
    with pytest.raises(errors.UnreachableTableError, match="not enabled at version 2"):
        t.cdf()
    assert pa.table(t.cdf(starting_version=3)).num_rows == 0


def _cdf_enabled_later(tmp_path: Path) -> str:
    import deltalake as dl

    path = str(tmp_path / "later")
    dl.write_deltalake(path, pa.table({"id": [1]}))
    dl.DeltaTable(path).alter.set_table_properties({"delta.enableChangeDataFeed": "true"})
    dl.write_deltalake(path, pa.table({"id": [2]}), mode="append")
    return path


def test_kernel_cdf_without_start_begins_where_feed_is_enabled(tmp_path: Path) -> None:
    _need_native()
    from deltaswamp.engine.kernel import KernelEngine

    t = ds.connect().table(_cdf_enabled_later(tmp_path))
    feed = pa.table(KernelEngine().cdf(t._enrich()))
    assert feed.column("id").to_pylist() == [2]
    assert set(feed.column("_commit_version").to_pylist()) == {2}


def test_kernel_cdf_over_disabled_version_is_a_clear_error(tmp_path: Path) -> None:
    _need_native()
    from deltaswamp.engine.kernel import KernelEngine

    t = ds.connect().table(_cdf_enabled_later(tmp_path))
    with pytest.raises(errors.UnreachableTableError, match="not enabled at version 0"):
        KernelEngine().cdf(t._enrich(), starting_version=0)


def test_kernel_cdf_across_schema_change_is_a_clear_error(tmp_path: Path) -> None:
    _need_native()
    from deltaswamp.engine.kernel import KernelEngine

    t = ds.connect().table(_copy("table-with-cdf", tmp_path))
    with pytest.raises(errors.UnreachableTableError, match="schema changed within"):
        KernelEngine().cdf(t._enrich(), starting_version=3)


# ------------------------------------------------- log cleaned by retention


def _cleaned_log(tmp_path: Path) -> str:
    """Commits 0-4 removed after a checkpoint at 5, as log retention leaves it."""
    import deltalake as dl

    path = str(tmp_path / "cleaned")
    cdf = {"delta.enableChangeDataFeed": "true"}
    for i in range(6):
        dl.write_deltalake(
            path, pa.table({"id": [i]}), mode="append", configuration=cdf if i == 0 else None
        )
    dl.DeltaTable(path).create_checkpoint()
    for i in range(6, 9):
        dl.write_deltalake(path, pa.table({"id": [i]}), mode="append")
    for i in range(6):
        os.remove(os.path.join(path, "_delta_log", f"{i:020d}.json"))
    return path


def test_pinned_handle_on_cleaned_version_says_so(tmp_path: Path) -> None:
    path = _cleaned_log(tmp_path)
    conn = ds.connect()
    with pytest.raises(errors.UnreachableTableError, match="no longer in its log") as info:
        conn.table(path, version=2).count()
    assert "no Delta table at this path" not in str(info.value)
    assert conn.table(path, version=5).count() == 6  # the checkpoint is still readable


def test_scan_at_cleaned_version_is_a_library_error(tmp_path: Path) -> None:
    _need_native()
    t = ds.connect().table(_cleaned_log(tmp_path))
    with pytest.raises(errors.UnreachableTableError, match="log retention"):
        t.to_arrow(version=2)


def test_cdf_without_start_on_cleaned_log_starts_at_oldest_commit(tmp_path: Path) -> None:
    t = ds.connect().table(_cleaned_log(tmp_path))
    feed = pa.table(t.cdf())
    assert sorted(feed.column("_commit_version").to_pylist()) == [6, 7, 8]


def test_cdf_from_cleaned_version_is_a_clear_error(tmp_path: Path) -> None:
    t = ds.connect().table(_cleaned_log(tmp_path))
    with pytest.raises(errors.UnreachableTableError, match="oldest commit still there is 6"):
        t.cdf(starting_version=0)


def test_kernel_cdf_on_cleaned_log(tmp_path: Path) -> None:
    _need_native()
    from deltaswamp.engine.kernel import KernelEngine

    t = ds.connect().table(_cleaned_log(tmp_path))
    feed = pa.table(KernelEngine().cdf(t._enrich()))
    assert sorted(feed.column("_commit_version").to_pylist()) == [6, 7, 8]
    with pytest.raises(errors.UnreachableTableError, match="oldest commit still there is 6"):
        KernelEngine().cdf(t._enrich(), starting_version=0)


# ------------------------------------------- mixed-case columns (Spark tables)


def _mixed_case(tmp_path: Path) -> str:
    """Like the golden hive/deltatbl-column-names-case-insensitive table."""
    import deltalake as dl

    path = str(tmp_path / "mixed")
    data = pa.table({"FooBar": pa.array(range(6), pa.int32()), "BarFoo": ["foo0", "foo1"] * 3})
    dl.write_deltalake(path, data, partition_by=["BarFoo"])
    return path


def _rows(path: str) -> list[tuple[int, str]]:
    table = ds.connect().table(path).to_arrow()
    return sorted(
        zip(table.column("FooBar").to_pylist(), table.column("BarFoo").to_pylist(), strict=True)
    )


def test_update_expression_resolves_columns_case_insensitively(tmp_path: Path) -> None:
    path = _mixed_case(tmp_path)
    ds.connect().table(path).update({"FOOBAR": "foobar + 100"}, predicate="FooBar = 0")
    assert (100, "foo0") in _rows(path)


def test_merge_predicate_resolves_columns_case_insensitively(tmp_path: Path) -> None:
    path = _mixed_case(tmp_path)
    t = ds.connect().table(path)
    source = pa.table({"FooBar": pa.array([2], pa.int32()), "BarFoo": ["x"]})
    t.merge(
        source, "s.FOOBAR = t.foobar", source_alias="s", target_alias="t"
    ).when_matched_update_all().execute()
    assert (2, "x") in _rows(path)


def test_merge_update_all_with_differently_cased_source_columns(tmp_path: Path) -> None:
    path = _mixed_case(tmp_path)
    t = ds.connect().table(path)
    source = pa.table({"foobar": pa.array([4], pa.int32()), "barfoo": ["NEW"]})
    result = (
        t.merge(source, "s.foobar = t.FooBar", source_alias="s", target_alias="t")
        .when_matched_update_all()
        .execute()
    )
    assert result["num_target_rows_updated"] == 1
    # It used to report the row as updated and leave BarFoo unchanged.
    assert (4, "NEW") in _rows(path)


def test_merge_clause_sql_resolves_columns_case_insensitively(tmp_path: Path) -> None:
    path = _mixed_case(tmp_path)
    t = ds.connect().table(path)
    source = pa.table({"FooBar": pa.array([4, 44], pa.int32()), "BarFoo": ["N2", "N3"]})
    t.merge(source, "s.FooBar = t.FooBar", source_alias="s", target_alias="t").when_matched_update(
        updates={"barfoo": "s.BARFOO"}, predicate="t.FOOBAR > 0"
    ).when_not_matched_insert(updates={"foobar": "s.FOOBAR", "BARFOO": "s.barfoo"}).execute()
    rows = _rows(path)
    assert (4, "N2") in rows and (44, "N3") in rows


def test_delete_with_function_predicate_resolves_columns(tmp_path: Path) -> None:
    path = _mixed_case(tmp_path)
    t = ds.connect().table(path)
    t.delete("abs(FOOBAR) = 3")
    t.delete("FOOBAR + 1 = 5")
    assert [r[0] for r in _rows(path)] == [0, 1, 2, 5]


def test_fold_case_leaves_literals_functions_and_keywords_alone() -> None:
    from deltaswamp.engine.deltars import _fold_case

    cols: dict[str | None, list[str]] = {None: ["FooBar", "End", "a", "A"]}
    assert _fold_case("upper('foobar') || FOOBAR", cols) == "upper('foobar') || `FooBar`"
    assert _fold_case("CASE WHEN foobar > 1 THEN 1 ELSE 0 END", cols) == (
        "CASE WHEN `FooBar` > 1 THEN 1 ELSE 0 END"
    )
    assert _fold_case("a + A + `foobar`", cols) == "a + A + `foobar`"  # exact or quoted


# ------------------------------------------------------ generated columns


def _generated(tmp_path: Path) -> str:
    import deltalake as dl
    from deltalake import Field, Schema

    path = str(tmp_path / "gen")
    schema = Schema(
        [
            Field("id", "long", nullable=True),
            Field("g", "long", nullable=True, metadata={"delta.generationExpression": "id * 2"}),
        ]
    )
    dl.DeltaTable.create(path, schema=schema)
    ds.connect().table(path).append(pa.table({"id": [1, 2]}))
    return path


def test_update_recomputes_generated_columns(tmp_path: Path) -> None:
    path = _generated(tmp_path)
    t = ds.connect().table(path)
    t.update({"id": "id + 10"})
    t.update(new_values={"id": 7}, predicate="id = 11")
    rows = sorted(ds.connect().table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
    assert rows == [{"id": 7, "g": 14}, {"id": 12, "g": 24}]


# ------------------------------------------------------ corrupt-log remedies


@pytest.mark.parametrize(
    "error",
    [
        # Verbatim from the golden versions-not-contiguous, deltalog-state-
        # reconstruction-*, and deltalog-invalid-protocol-version tables.
        "DeltaError: Kernel error: Generic delta kernel error: Expected contiguous commit "
        "files, but found gap",
        "DeltaError: Kernel error: No table metadata found in delta log.",
        "DeltaError: Kernel error: No protocol found in delta log.",
        "DeltaError: Kernel error: Invalid protocol action in the delta log: Reader features "
        "must not be present when minimum reader version != 3",
        "DeltaError: Kernel error: Invalid argument error: Found unmasked nulls for "
        'non-nullable StructArray field "schemaString"',
    ],
)
def test_corrupt_log_does_not_suggest_the_warehouse(error: str) -> None:
    from deltaswamp.catalog import ResolvedTable
    from deltaswamp.identity import parse_ref
    from deltaswamp.router import _open_error_remedy

    resolved = ResolvedTable(
        ref=parse_ref("main.sales.orders"), location="s3://b/t", open_error=error
    )
    remedy = _open_error_remedy(resolved)
    assert "allow_sql_fallback" not in remedy
    assert "readable Delta log" in remedy


def test_partition_delete_without_stats_reports_deleted_rows(tmp_path: Path) -> None:
    # Spark-written files with no statistics: delta-rs drops whole files and
    # reported no num_deleted_rows at all.
    t = ds.connect().table(_copy("hive-case-insensitive", tmp_path))
    result = t.delete("barfoo = 'foo0'")
    assert result["num_deleted_rows"] == 5
    assert t.count() == 5
    assert t.delete("barfoo = 'nothing'")["num_deleted_rows"] == 0


# -------------------------------------------- MERGE with generated columns


def _generated_merge(tmp_path: Path) -> ds.Table:
    import deltalake as dl
    from deltalake import Field, Schema

    path = str(tmp_path / "genm")
    schema = Schema(
        [
            Field("id", "long", nullable=True),
            Field("v", "long", nullable=True),
            Field("g", "long", nullable=True, metadata={"delta.generationExpression": "id * 2"}),
            Field("h", "long", nullable=True, metadata={"delta.generationExpression": "id + v"}),
        ]
    )
    dl.DeltaTable.create(path, schema=schema)
    ds.connect().table(path).append(pa.table({"id": [1, 2], "v": [10, 20]}))
    return ds.connect().table(path)


def _by_id(t: ds.Table) -> list[dict[str, int]]:
    rows = ds.connect().table(t.location).to_arrow().to_pylist()
    return sorted(rows, key=lambda r: r["id"])


def test_merge_matched_update_recomputes_generated_columns(tmp_path: Path) -> None:
    t = _generated_merge(tmp_path)
    source = pa.table({"id": [2], "v": [5]})
    t.merge(source, "s.id = t.id", source_alias="s", target_alias="t").when_matched_update(
        updates={"id": "s.id + 100"}
    ).execute()
    # g = id * 2 from the new id; h = id + v from the new id and the kept v.
    assert _by_id(t)[-1] == {"id": 102, "v": 20, "g": 204, "h": 122}


def test_merge_update_all_recomputes_generated_columns(tmp_path: Path) -> None:
    t = _generated_merge(tmp_path)
    source = pa.table({"id": [7], "v": [20]})
    t.merge(source, "s.v = t.v", source_alias="s", target_alias="t").when_matched_update_all(
        except_cols=["v"]
    ).execute()
    assert _by_id(t)[-1] == {"id": 7, "v": 20, "g": 14, "h": 27}


def test_merge_insert_computes_generated_columns(tmp_path: Path) -> None:
    t = _generated_merge(tmp_path)
    source = pa.table({"id": [3], "v": [1]})
    t.merge(
        source, "s.id = t.id", source_alias="s", target_alias="t"
    ).when_not_matched_insert_all().execute()
    assert _by_id(t)[-1] == {"id": 3, "v": 1, "g": 6, "h": 4}


def test_merge_generated_recompute_without_target_alias_is_refused(tmp_path: Path) -> None:
    t = _generated_merge(tmp_path)
    merger = t.merge(pa.table({"id": [2], "v": [5]}), "source.id = id", source_alias="source")
    with pytest.raises(errors.InvalidArgumentError, match="target_alias"):
        merger.when_matched_update(updates={"id": "source.id + 1"})
    assert _by_id(t) == [
        {"id": 1, "v": 10, "g": 2, "h": 11},
        {"id": 2, "v": 20, "g": 4, "h": 22},
    ]


# ------------------------------------------------ vacuumed / deleted files


def _vacuumed(tmp_path: Path) -> str:
    import deltalake as dl

    path = str(tmp_path / "vac")
    dl.write_deltalake(path, pa.table({"id": [1, 2]}))
    dl.write_deltalake(path, pa.table({"id": [3]}), mode="overwrite")
    log = Path(path, "_delta_log", f"{0:020d}.json").read_text(encoding="utf-8")
    add = next(json.loads(line)["add"] for line in log.splitlines() if '"add"' in line)
    os.remove(os.path.join(path, add["path"]))
    return path


@pytest.mark.parametrize(
    "read",
    [
        lambda t: t.to_arrow(),
        lambda t: t.to_pandas(),
        lambda t: t.count(),
        lambda t: t.head(1),
        lambda t: list(t.scan()),
        lambda t: t.scan().read_all(),
        lambda t: t.plan_scan().read(),
    ],
    ids=["to_arrow", "to_pandas", "count", "head", "iterate", "read_all", "plan_scan"],
)
def test_missing_data_file_is_a_library_error(tmp_path: Path, read) -> None:
    _need_native()
    t = ds.connect().table(_vacuumed(tmp_path), version=0)
    with pytest.raises(errors.MissingDataFileError, match="VACUUM") as info:
        read(t)
    assert isinstance(info.value, errors.CorruptTableError)
    assert info.value.path.endswith(".parquet")


def test_missing_data_file_message_crosses_the_c_stream(tmp_path: Path) -> None:
    _need_native()
    import polars as pl

    t = ds.connect().table(_vacuumed(tmp_path), version=0)
    with pytest.raises(Exception, match="missing from storage"):
        pa.table(t.scan())
    with pytest.raises(Exception, match="missing from storage"):
        pl.DataFrame(t.scan())


def test_scan_stream_still_exports_through_the_c_interface(tmp_path: Path) -> None:
    import duckdb
    import polars as pl

    t = ds.connect().table(_vacuumed(tmp_path))
    assert pa.table(t.scan()).num_rows == 1
    assert pl.DataFrame(t.scan()).height == 1
    assert pa.RecordBatchReader.from_stream(t.scan()).read_all().num_rows == 1
    stream = t.scan()  # noqa: F841 -- duckdb resolves it by name
    assert duckdb.sql("select count(*) from stream").fetchone()[0] == 1
    assert t.to_polars(lazy=True).collect().height == 1


def test_missing_data_file_error_pickles() -> None:
    import pickle

    err = pickle.loads(pickle.dumps(errors.MissingDataFileError("s3://b/f.parquet", "gone")))
    assert err.path == "s3://b/f.parquet" and str(err) == "gone"


def test_missing_data_file_is_typed_through_ray(tmp_path: Path) -> None:
    _need_native()
    pytest.importorskip("ray")
    t = ds.connect().table(_vacuumed(tmp_path), version=0)
    with pytest.raises(errors.MissingDataFileError):
        t.to_ray_dataset().count()
