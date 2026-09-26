"""Every public entry point answers wrong input with a DeltaSwampError.

A battery of wrong inputs -- wrong types, missing columns, bad predicates, bad
timestamps, negative sizes, unknown keyword arguments, conflicting arguments --
is thrown at the Connection, the Table, the distributed plans and (through the
fake Unity Catalog) volumes. Each must raise a `DeltaSwampError`; input that is
simply wrong must raise `InvalidArgumentError` (or `PredicateError`, the
predicate parser's own error, or `InvalidReferenceError` for a malformed name).
Input that is well formed but names something absent or out of range (version
999, a timestamp before the first commit) may be refused with any
DeltaSwampError, and so may input a table-level refusal reaches first (renaming
a missing column on a table without column mapping). A raw ValueError, TypeError, DeltaError,
ArrowInvalid or OSError escaping is a finding, and so is wrong input accepted
silently. (A misspelt keyword on a method with a fixed signature raises
Python's own TypeError, which is idiomatic and not tested; one passed through
``**kwargs`` is the library's to check.)

Known findings are strict xfails in `KNOWN`, so each flips when it is fixed.
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from deltaswamp.errors import DeltaSwampError, InvalidArgumentError, InvalidReferenceError
from deltaswamp.predicate import PredicateError

from tests.contract.conftest import Tables, connect_path, connect_uc
from tests.contract.tables import base_data

pa = pytest.importorskip("pyarrow")

BAD_INPUT = (InvalidArgumentError, InvalidReferenceError, PredicateError)


@dataclass
class Env:
    conn: Any
    table: Any
    path: str
    tables: Tables


@dataclass(frozen=True)
class Bad:
    id: str
    call: Callable[[Env], Any]
    #: True: the input is plainly wrong, so InvalidArgumentError (or PredicateError).
    #: False: any DeltaSwampError will do (the input is valid but unservable).
    bad_input: bool = True


def _data() -> Any:
    return base_data()


def _plan(env: Env) -> Any:
    return env.table.plan_write()


def _scan_plan(env: Env) -> Any:
    return env.table.plan_scan()


T = "table"
TABLE_CASES: list[Bad] = [
    # --- reads
    Bad("scan_columns_missing", lambda e: e.table.to_arrow(columns=["nope"])),
    Bad("scan_columns_int", lambda e: e.table.to_arrow(columns=[1])),
    Bad("scan_predicate_int", lambda e: e.table.to_arrow(predicate=1)),
    Bad("scan_predicate_garbage", lambda e: e.table.to_arrow(predicate="id ===")),
    Bad("scan_predicate_missing_column", lambda e: e.table.to_arrow(predicate="nope = 1")),
    Bad("scan_version_negative", lambda e: e.table.to_arrow(version=-1)),
    Bad("scan_version_str", lambda e: e.table.to_arrow(version="1")),
    Bad("scan_version_future", lambda e: e.table.to_arrow(version=999), False),
    Bad("scan_version_bool", lambda e: e.table.to_arrow(version=True)),
    Bad("scan_timestamp_garbage", lambda e: e.table.to_arrow(timestamp="yesterday-ish")),
    Bad("scan_timestamp_int", lambda e: e.table.to_arrow(timestamp=[1])),
    Bad("scan_timestamp_before_history", lambda e: e.table.to_arrow(timestamp="1990-01-01"), False),
    Bad(
        "scan_version_and_timestamp", lambda e: e.table.to_arrow(version=0, timestamp="2024-01-01")
    ),
    Bad("scan_limit_negative", lambda e: e.table.scan(limit=-1)),
    Bad("scan_limit_str", lambda e: e.table.scan(limit="1")),
    Bad("to_arrow_unknown_kwarg", lambda e: e.table.to_arrow(colums=["id"])),
    Bad("to_pandas_unknown_kwarg", lambda e: e.table.to_pandas(colums=["id"])),
    Bad("to_polars_unknown_kwarg", lambda e: e.table.to_polars(colums=["id"])),
    Bad("head_negative", lambda e: e.table.head(-1)),
    Bad("head_str", lambda e: e.table.head("2")),
    Bad("count_predicate_garbage", lambda e: e.table.count(predicate="id ===")),
    Bad("history_negative", lambda e: e.table.history(limit=-1)),
    Bad("history_str", lambda e: e.table.history(limit="1")),
    Bad("cdf_unknown_kwarg", lambda e: e.table.cdf(start_version=0)),
    Bad("cdf_negative", lambda e: e.table.cdf(starting_version=-1)),
    Bad("cdf_reversed", lambda e: e.table.cdf(starting_version=2, ending_version=1)),
    Bad("cdf_timestamp_garbage", lambda e: e.table.cdf(starting_timestamp="not a time"), False),
    Bad("changes_negative", lambda e: list(e.table.changes(-1))),
    Bad("changes_str", lambda e: list(e.table.changes("0"))),
    Bad("changes_poll_str", lambda e: list(e.table.changes(0, poll_interval="1"))),
    Bad("txn_version_int", lambda e: e.table.txn_version(1)),
    Bad("can_unknown_operation", lambda e: e.table.can("teleport")),
    Bad("can_unknown_shape", lambda e: e.table.can("append", schema_mod="merge")),
    # --- writes
    Bad("append_none", lambda e: e.table.append(None)),
    Bad("append_str", lambda e: e.table.append("rows")),
    Bad("append_dict_of_scalars", lambda e: e.table.append({"id": 1})),
    Bad("append_wrong_type", lambda e: e.table.append(pa.table({"id": ["x"]}))),
    Bad("append_missing_columns", lambda e: e.table.append(pa.table({"zzz": [1]}))),
    Bad("append_schema_mode_bad", lambda e: e.table.append(_data(), schema_mode="sideways")),
    Bad("append_txn_str", lambda e: e.table.append(_data(), txn="app")),
    Bad("append_txn_negative", lambda e: e.table.append(_data(), txn=("app", -1))),
    Bad("append_commit_metadata_list", lambda e: e.table.append(_data(), commit_metadata=[1])),
    Bad("append_retries_negative", lambda e: e.table.append(_data(), max_commit_retries=-1)),
    Bad("append_retries_str", lambda e: e.table.append(_data(), max_commit_retries="2")),
    Bad("append_target_file_size_negative", lambda e: e.table.append(_data(), target_file_size=-1)),
    Bad("append_partition_by_mismatch", lambda e: e.table.append(_data(), partition_by=["grp"])),
    Bad("overwrite_predicate_garbage", lambda e: e.table.overwrite(_data(), predicate="id ===")),
    Bad(
        "overwrite_partition_mode_bad",
        lambda e: e.table.overwrite(_data(), partition_overwrite="sometimes"),
    ),
    Bad(
        "overwrite_predicate_and_dynamic",
        lambda e: e.table.overwrite(_data(), predicate="id = 1", partition_overwrite="dynamic"),
    ),
    Bad("replace_unknown_kwarg", lambda e: e.table.replace(_data(), colour="red")),
    # --- dml
    Bad("delete_predicate_int", lambda e: e.table.delete(1)),
    Bad("delete_predicate_garbage", lambda e: e.table.delete("id ===")),
    Bad("delete_missing_column", lambda e: e.table.delete("nope = 1")),
    Bad("delete_unknown_kwarg", lambda e: e.table.delete("id = 1", dry_run=True)),
    Bad("update_nothing", lambda e: e.table.update({})),
    Bad("update_both", lambda e: e.table.update({"v": "'a'"}, new_values={"v": "a"})),
    Bad("update_missing_column", lambda e: e.table.update({"nope": "1"})),
    Bad("update_not_a_dict", lambda e: e.table.update(["v"])),
    Bad("update_bad_expression", lambda e: e.table.update({"v": "'a"}, predicate="id = 1")),
    Bad("update_predicate_garbage", lambda e: e.table.update({"v": "'a'"}, predicate="id ===")),
    Bad("update_wrong_type_value", lambda e: e.table.update(new_values={"id": "x"})),
    Bad("update_unknown_kwarg", lambda e: e.table.update({"v": "'a'"}, where="id = 1")),
    Bad("merge_no_predicate", lambda e: e.table.merge(_data(), None)),
    Bad("merge_predicate_int", lambda e: e.table.merge(_data(), 1)),
    Bad("merge_source_str", lambda e: e.table.merge("rows", "t.id = s.id")),
    Bad(
        "merge_unknown_kwarg",
        lambda e: e.table.merge(_data(), "t.id = s.id", source_aliass="s"),
    ),
    Bad(
        "merge_bad_predicate_execute",
        lambda e: e.table.merge(_data(), "t.nope = s.id", source_alias="s", target_alias="t")
        .when_matched_update_all()
        .execute(),
    ),
    Bad(
        "merge_bad_clause_expression",
        lambda e: e.table.merge(_data(), "t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_update({"v": "s.nope"})
        .execute(),
    ),
    # --- ddl
    Bad("add_column_empty", lambda e: e.table.add_column([])),
    Bad("add_column_duplicate", lambda e: e.table.add_column([pa.field("id", pa.int64())])),
    Bad("add_column_str", lambda e: e.table.add_column("newcol")),
    Bad("add_column_bad_sql_type", lambda e: e.table.add_column({"c": "notatype"})),
    Bad("add_column_unknown_kwarg", lambda e: e.table.add_column([pa.field("c", pa.int8())], x=1)),
    Bad("drop_column_missing", lambda e: e.table.drop_column("nope"), False),
    Bad("drop_column_int", lambda e: e.table.drop_column(1)),
    Bad("rename_column_missing", lambda e: e.table.rename_column("nope", "x"), False),
    Bad("rename_column_to_existing", lambda e: e.table.rename_column("v", "id"), False),
    Bad("rename_column_empty", lambda e: e.table.rename_column("v", ""), False),
    Bad("set_properties_list", lambda e: e.table.set_properties(["a"])),
    Bad(
        "set_properties_bad_value",
        lambda e: e.table.set_properties({"delta.appendOnly": "maybe"}),
        False,
    ),
    Bad("unset_properties_int", lambda e: e.table.unset_properties(1)),
    Bad(
        "unset_properties_missing_strict", lambda e: e.table.unset_properties("x", if_exists=False)
    ),
    Bad("add_feature_unknown", lambda e: e.table.add_feature("teleportation")),
    Bad("add_feature_int", lambda e: e.table.add_feature(1)),
    Bad("drop_feature_unknown_kwarg", lambda e: e.table.drop_feature("x", force=True)),
    Bad("add_constraint_empty", lambda e: e.table.add_constraint({})),
    Bad("add_constraint_garbage", lambda e: e.table.add_constraint({"c": "id >"})),
    Bad("add_constraint_missing_column", lambda e: e.table.add_constraint({"c": "nope > 0"})),
    Bad("add_constraint_violated", lambda e: e.table.add_constraint({"c": "id > 100"})),
    Bad("drop_constraint_missing", lambda e: e.table.drop_constraint("nope"), False),
    Bad("set_comment_int", lambda e: e.table.set_comment(1)),
    Bad("set_column_comment_missing", lambda e: e.table.set_column_comment("nope", "c"), False),
    Bad("alter_column_type_missing", lambda e: e.table.alter_column_type("nope", "long"), False),
    Bad("alter_column_type_garbage", lambda e: e.table.alter_column_type("w", "notatype"), False),
    Bad("alter_column_type_narrowing", lambda e: e.table.alter_column_type("id", "int"), False),
    Bad("set_not_null_missing", lambda e: e.table.set_not_null("nope")),
    Bad("drop_not_null_missing", lambda e: e.table.drop_not_null("nope")),
    Bad("cluster_by_missing", lambda e: e.table.cluster_by(["nope"])),
    Bad("cluster_by_int", lambda e: e.table.cluster_by(1)),
    # --- maintenance
    Bad("optimize_zorder_missing", lambda e: e.table.optimize(zorder_by=["nope"])),
    Bad("optimize_target_size_negative", lambda e: e.table.optimize(target_size=-1)),
    Bad("optimize_unknown_kwarg", lambda e: e.table.optimize(zorder=["id"])),
    Bad("z_order_empty", lambda e: e.table.z_order([])),
    Bad("vacuum_retention_negative", lambda e: e.table.vacuum(retention_hours=-1)),
    Bad("vacuum_retention_str", lambda e: e.table.vacuum(retention_hours="1")),
    Bad("vacuum_unknown_kwarg", lambda e: e.table.vacuum(dryrun=True)),
    Bad("restore_none", lambda e: e.table.restore(None)),
    Bad("restore_negative", lambda e: e.table.restore(-1)),
    Bad("restore_future", lambda e: e.table.restore(999), False),
    Bad("restore_timestamp_garbage", lambda e: e.table.restore("not a time")),
    Bad("restore_timestamp_before_history", lambda e: e.table.restore("1990-01-01"), False),
    Bad(
        "restore_datetime_before_history", lambda e: e.table.restore(dt.datetime(1990, 1, 1)), False
    ),
    Bad("scan_date_object", lambda e: e.table.to_arrow(timestamp=dt.date(1990, 1, 1)), False),
    Bad("restore_unknown_kwarg", lambda e: e.table.restore(0, force=True)),
    Bad("repair_unknown_kwarg", lambda e: e.table.repair(dryrun=True)),
    Bad("compact_logs_negative", lambda e: e.table.compact_logs(-1, 0)),
    Bad("compact_logs_reversed", lambda e: e.table.compact_logs(1, 0)),
    Bad("clone_unknown_kwarg", lambda e: e.table.clone("x", deep=True)),
    Bad("reorg_unknown_kwarg", lambda e: e.table.reorg(purgee=True)),
    # --- distributed plans
    Bad("plan_write_mode_bad", lambda e: e.table.plan_write(mode="merge")),
    Bad("plan_write_txn_str", lambda e: e.table.plan_write(txn="job")),
    Bad("plan_write_metadata_list", lambda e: e.table.plan_write(commit_metadata=[1])),
    Bad("plan_write_write_str", lambda e: _plan(e).write("rows")),
    Bad("plan_write_write_wrong_schema", lambda e: _plan(e).write(pa.table({"zzz": [1]}))),
    Bad("plan_write_commit_garbage", lambda e: _plan(e).commit([b"not a fragment"])),
    Bad("plan_write_commit_not_bytes", lambda e: _plan(e).commit([1])),
    Bad("plan_write_commit_retries_negative", lambda e: _plan(e).commit([], retries=-1)),
    Bad("plan_scan_columns_missing", lambda e: e.table.plan_scan(columns=["nope"])),
    Bad("plan_scan_predicate_garbage", lambda e: e.table.plan_scan(predicate="id ===")),
    Bad("plan_scan_version_negative", lambda e: e.table.plan_scan(version=-1)),
    Bad("plan_scan_timestamp_garbage", lambda e: e.table.plan_scan(timestamp="whenever")),
    Bad("plan_scan_partitions_zero", lambda e: _scan_plan(e).partitions(0)),
    Bad("plan_scan_partitions_str", lambda e: _scan_plan(e).partitions("2")),
    Bad("plan_scan_read_garbage", lambda e: _scan_plan(e).read(["not a split"])),
]

CONNECTION_CASES: list[Bad] = [
    Bad("open_table_int", lambda e: e.conn.open_table(1)),
    Bad(
        "open_table_missing",
        lambda e: e.conn.open_table(e.tables.fresh_dir("missing")).to_arrow(),
        False,
    ),
    Bad("open_table_version_negative", lambda e: e.conn.open_table(e.path, version=-1).to_arrow()),
    Bad("open_table_version_str", lambda e: e.conn.open_table(e.path, version="0").to_arrow()),
    Bad("table_empty_name", lambda e: e.conn.table("")),
    Bad("table_bad_name", lambda e: e.conn.table("a.b.c.d.e")),
    Bad("create_table_no_schema", lambda e: e.conn.create_table(e.tables.fresh_dir("c"), None)),
    Bad(
        "create_table_schema_str", lambda e: e.conn.create_table(e.tables.fresh_dir("c"), "id int")
    ),
    Bad(
        "create_table_mode_bad",
        lambda e: e.conn.create_table(e.tables.fresh_dir("c"), _data().schema, mode="sideways"),
    ),
    Bad(
        "create_table_partition_missing",
        lambda e: e.conn.create_table(e.tables.fresh_dir("c"), _data().schema, partition_by=["x"]),
    ),
    Bad(
        "create_table_cluster_missing",
        lambda e: e.conn.create_table(e.tables.fresh_dir("c"), _data().schema, cluster_by=["x"]),
    ),
    Bad(
        "create_table_partition_and_cluster",
        lambda e: e.conn.create_table(
            e.tables.fresh_dir("c"), _data().schema, partition_by=["grp"], cluster_by=["id"]
        ),
    ),
    Bad(
        "create_table_properties_list",
        lambda e: e.conn.create_table(e.tables.fresh_dir("c"), _data().schema, properties=["a"]),
    ),
    Bad(
        "create_table_bad_property_value",
        lambda e: e.conn.create_table(
            e.tables.fresh_dir("c"),
            _data().schema,
            properties={"delta.columnMapping.mode": "sideways"},
        ),
        False,
    ),
    Bad(
        "create_table_exists",
        lambda e: e.conn.create_table(e.path, _data().schema),
        False,
    ),
    Bad("write_table_none", lambda e: e.conn.write_table(e.tables.fresh_dir("w"), None)),
    Bad("write_table_str", lambda e: e.conn.write_table(e.tables.fresh_dir("w"), "rows")),
    Bad(
        "write_table_mode_bad",
        lambda e: e.conn.write_table(e.tables.fresh_dir("w"), _data(), mode="sideways"),
    ),
    Bad(
        "write_table_unknown_kwarg",
        lambda e: e.conn.write_table(e.tables.fresh_dir("w"), _data(), colour="red"),
    ),
    Bad(
        "write_table_exists_error",
        lambda e: e.conn.write_table(e.path, _data(), mode="error"),
        False,
    ),
    Bad("convert_missing_dir", lambda e: e.conn.convert_to_delta(e.tables.fresh_dir("cv")), False),
    Bad("convert_unknown_kwarg", lambda e: e.conn.convert_to_delta(e.path, colour="red")),
    Bad("sql_engine_bad", lambda e: e.conn.sql("select 1", engine="oracle")),
    Bad("sql_tables_list", lambda e: e.conn.sql("select 1", tables=[e.path])),
    Bad("sql_garbage", lambda e: e.conn.sql("selec from", tables={"t": e.table})),
    Bad("sql_missing_table", lambda e: e.conn.sql("select * from nope", tables={"t": e.table})),
    Bad("sql_int", lambda e: e.conn.sql(1)),
    Bad("list_tables_int", lambda e: e.conn.list_tables(1, 2), False),
    Bad("drop_table_path_missing", lambda e: e.conn.drop_table(e.tables.fresh_dir("d")), False),
    Bad("connect_bad_fallback", lambda e: _connect(allow_sql_fallback="false")),
    Bad("connect_uri_and_catalog", lambda e: _connect("uc://http://x", catalog=e.conn.catalog)),
    Bad("connect_bad_uri", lambda e: _connect("teleport://x")),
]


def _connect(*args: Any, **kwargs: Any) -> Any:
    import deltaswamp as ds

    return ds.connect(*args, **kwargs)


#: OSS Unity Catalog has no Files API, so locally only opening a volume is
#: reachable; reading and writing its files is covered by the live suite.
VOLUME_CASES: list[Bad] = [
    Bad(
        "volume_open_missing", lambda e: connect_uc(e.tables.uc).volume("main.contract.nope"), False
    ),
    Bad("volume_open_bad_name", lambda e: connect_uc(e.tables.uc).volume("nope")),
    Bad("volume_open_int", lambda e: connect_uc(e.tables.uc).volume(1)),
    Bad(
        "volume_create_bad_type",
        lambda e: connect_uc(e.tables.uc).create_volume("main.contract.vx", volume_type="SIDEWAYS"),
    ),
    Bad(
        "volume_create_external_no_location",
        lambda e: connect_uc(e.tables.uc).create_volume("main.contract.vy", volume_type="EXTERNAL"),
    ),
]

#: case id -> finding. Strict xfails: each flips to XPASS (a failure) once fixed.
_FINDINGS: dict[str, tuple[str, ...]] = {
    "E1: an unknown keyword passed through **kwargs escapes as a raw TypeError": (
        "to_arrow_unknown_kwarg",
        "to_pandas_unknown_kwarg",
        "to_polars_unknown_kwarg",
        "write_table_unknown_kwarg",
        "convert_unknown_kwarg",
    ),
    "E2: data that is not a table (str, dict of scalars) escapes as a raw Python error": (
        "append_str",
        "append_dict_of_scalars",
        "merge_source_str",
        "plan_write_write_str",
        "write_table_str",
    ),
    "E3: data not matching the schema or partitioning escapes as a raw delta-rs error": (
        "append_partition_by_mismatch",
    ),
    "E4: a malformed SQL predicate/expression on the delta-rs path escapes as a raw DeltaError": (
        "overwrite_predicate_garbage",
        "delete_predicate_garbage",
        "update_bad_expression",
        "update_predicate_garbage",
        "add_constraint_garbage",
    ),
    "E5: a list/int where a dict/str is expected escapes as a raw Python error": (
        "update_not_a_dict",
        "unset_properties_int",
        "cluster_by_int",
        "create_table_properties_list",
    ),
    "E6: a missing constraint or column on the delta-rs ALTER path escapes as a raw DeltaError": (
        "drop_constraint_missing",
        "set_column_comment_missing",
    ),
    "E7: optimize(target_size=-1) escapes as a raw OverflowError": (),
    "E8: ScanPlan.read() with objects that are not splits escapes as a raw AttributeError": (
        "plan_scan_read_garbage",
    ),
    "E9: convert_to_delta on an empty directory escapes as a raw DeltaError": (
        "convert_missing_dir",
    ),
    "E10: Connection.sql() lets duckdb's raw exceptions escape": (
        "sql_garbage",
        "sql_missing_table",
        "sql_int",
    ),
    "E11: malformed input is refused as UnreachableTableError, not InvalidArgumentError": (
        "scan_timestamp_garbage",
        "scan_timestamp_int",
        "plan_scan_timestamp_garbage",
        "overwrite_predicate_and_dynamic",
        "add_column_str",
        "add_column_bad_sql_type",
        "unset_properties_missing_strict",
        "add_feature_unknown",
        "add_feature_int",
        "set_not_null_missing",
        "drop_not_null_missing",
        "cluster_by_missing",
    ),
}
KNOWN: dict[str, str] = {c: reason for reason, ids in _FINDINGS.items() for c in ids}

ALL_CASES = [("table", c) for c in TABLE_CASES] + [("connection", c) for c in CONNECTION_CASES]
ALL_CASES += [("volume", c) for c in VOLUME_CASES]


def _params() -> list[Any]:
    out = []
    for group, case in ALL_CASES:
        marks = []
        if case.id in KNOWN:
            marks.append(pytest.mark.xfail(strict=True, reason=KNOWN[case.id]))
        out.append(pytest.param(case, id=f"{group}::{case.id}", marks=marks))
    return out


def test_case_ids_are_unique() -> None:
    ids = [c.id for _, c in ALL_CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", _params())
def test_wrong_input_raises_a_deltaswamp_error(contract_tables: Tables, case: Bad) -> None:
    conn = connect_path()
    path = contract_tables.path("plain", writable=True)
    env = Env(conn, conn.open_table(path), path, contract_tables)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            case.call(env)
        except DeltaSwampError as exc:
            if case.bad_input:
                assert isinstance(exc, BAD_INPUT), (
                    f"wrong input raised {type(exc).__name__}, not InvalidArgumentError: {exc}"
                )
            return
        except Exception as exc:
            pytest.fail(
                f"raw {type(exc).__module__}.{type(exc).__name__} escaped: {str(exc)[:400]}"
            )
    pytest.fail("wrong input was accepted without an error")
