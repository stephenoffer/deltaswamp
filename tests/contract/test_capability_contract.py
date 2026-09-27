"""`t.can(op, **args)` must agree with calling the operation with those arguments.

For every `Operation`, a representative valid call is made on every table
permutation in `tables.FIXTURES` (plus a catalog-managed table behind a fake
Unity Catalog), and three things are checked:

* ``can()`` says ok exactly when the call does not raise a routing refusal
  (`UnreachableTableError` and its subclasses). A refusal from ``can()`` must
  come with a refusal from the call, before anything was written; an ok must
  come with a call that succeeds.
* When ok, the engine ``can()`` names is the engine that served the call. The
  engines of each connection are instrumented (see `conftest.instrument`) so
  this is observed, not inferred.
* The result is right: the keys the table holds afterwards are the ones the
  operation implies, read both through this library and through delta-rs's own
  reader where it can open the table.

Known disagreements are listed in `KNOWN` as strict xfails: each one is a
finding, and fixing it flips the test to XPASS, which fails until the entry is
removed.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from deltaswamp.capability import ENGINE_METHODS, Operation
from deltaswamp.errors import UnreachableTableError

from tests.contract.conftest import Tables, connect_path, connect_uc, instrument
from tests.contract.tables import FIXTURES, Fixture

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

UC = "catalog_managed"
TABLES = [f.name for f in FIXTURES] + [UC]


# ----------------------------------------------------------------- context


@dataclass
class Ctx:
    """One table under test, and what the cases need to know about it."""

    conn: Any
    table: Any
    name: str
    fixture: str
    tables: Tables
    key: str = "id"
    #: A string column to SET (None on tables without one).
    text: str | None = "v"
    #: An int column that can be widened.
    narrow: str = "w"
    generated: bool = False
    before: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def reopen(self) -> Any:
        if self.fixture == UC:
            return self.conn.table(self.name)
        return self.conn.open_table(self.name)

    def rows(self, version: int | None = None) -> Any:
        t = self.reopen()
        return t.to_arrow() if version is None else t.to_arrow(version=version)

    def keys(self, table: Any = None) -> list[int]:
        table = self.rows() if table is None else table
        return sorted(table.column(self.key).to_pylist())

    def shifted(self, delta: int, where: Any = None) -> Any:
        """The table's own rows with the key moved by `delta` (a valid append)."""
        rows = self.before if where is None else where
        key = pc.add(rows.column(self.key), pa.scalar(delta, rows.schema.field(self.key).type))
        rows = rows.set_column(rows.schema.get_field_index(self.key), self.key, key)
        if self.generated:
            doubled = pc.multiply(key, pa.scalar(2, key.type))
            rows = rows.set_column(rows.schema.get_field_index("id2"), "id2", doubled)
        return rows

    def only(self, key: int) -> Any:
        return self.before.filter(pc.equal(self.before.column(self.key), key))


def _latest_timestamp(path: str) -> str:
    """When the latest commit was made, the way a reader resolves timestamps."""
    log = sorted(pathlib.Path(path, "_delta_log").glob("[0-9]*.json"))[-1]
    millis = int(log.stat().st_mtime * 1000)
    for line in log.read_text().splitlines():
        info = json.loads(line).get("commitInfo") if line.strip() else None
        if info and "inCommitTimestamp" in info:
            millis = int(info["inCommitTimestamp"])
    return dt.datetime.fromtimestamp(millis / 1000, dt.UTC).isoformat()


# ------------------------------------------------------------------- cases


@dataclass(frozen=True)
class Case:
    """One call of one operation: what to ask can(), what to call, what to expect."""

    op: Operation
    id: str
    call: Callable[[Any, Ctx], Any]
    shape: Callable[[Ctx], dict[str, Any]] = lambda cx: {}
    writes: bool = True
    #: The keys the table should hold afterwards, from the keys before.
    expect: Callable[[Ctx, list[int]], list[int]] | None = None
    #: What a read call's result should hold, when it returns rows.
    rows: Callable[[Ctx, Any], None] | None = None
    #: Tables this case does not apply to (with the reason).
    skip: dict[str, str] = field(default_factory=dict)
    #: May finish without any engine call (a no-op the Table answers itself).
    no_engine_ok: bool = False
    #: The engine methods that do this call's work; others (the Table reading
    #: the log to route, a commit listing files) are not service.
    serves: tuple[str, ...] = ()

    def serving_methods(self) -> frozenset[str]:
        if self.serves:
            return frozenset(self.serves)
        method = ENGINE_METHODS.get(self.op)
        extra = _READ_METHODS if self.op in (P.SCAN, P.TIME_TRAVEL) else ()
        return frozenset({method, *extra} if method else extra)


_READ_METHODS = ("scan", "execute_scan", "plan_scan", "metadata_count")
_WRITE_PLAN = ("write_files", "commit_files", "append", "overwrite")


def _same(cx: Ctx, before: list[int]) -> list[int]:
    return before


def _keys_are(cx: Ctx, result: Any, wanted: list[int] | None = None) -> None:
    table = pa.table(result) if not isinstance(result, pa.Table) else result
    got = sorted(table.column(cx.key).to_pylist())
    assert got == (wanted if wanted is not None else cx.keys(cx.before))


def _merge_args(cx: Ctx) -> dict[str, Any]:
    return {
        "source": pa.concat_tables([cx.only(1), cx.shifted(1000, cx.only(1))]),
        "predicate": f"target.{cx.key} = source.{cx.key}",
        "source_alias": "source",
        "target_alias": "target",
    }


def _merge(cx: Ctx, t: Any) -> Any:
    return (
        t.merge(**_merge_args(cx)).when_matched_update_all().when_not_matched_insert_all().execute()
    )


def _plan_write(cx: Ctx, t: Any, **kwargs: Any) -> Any:
    plan = t.plan_write(**kwargs)
    return plan.commit([plan.write(cx.shifted(100))])


def _plan_scan(t: Any, cx: Ctx) -> Any:
    return t.plan_scan().read()


def _changes(t: Any, cx: Ctx) -> Any:
    return list(t.changes(0))


def _set_text(cx: Ctx) -> dict[str, str]:
    return {cx.text: "'zz'"} if cx.text else {cx.key: f"{cx.key}"}


def _widen(cx: Ctx) -> tuple[str, str]:
    return (cx.narrow, "long") if cx.narrow == "w" else (cx.narrow, "timestamp_ntz")


def _clone_target(cx: Ctx) -> str:
    return str(cx.extra.setdefault("clone", cx.tables.fresh_dir("clone")))


def _beyond(cx: Ctx) -> str:
    """A predicate outside the kernel's grammar (arithmetic), true for key 1 only."""
    return f"{cx.key} % 7 = 1"


P = Operation
CASES: list[Case] = [
    # --- reads
    Case(P.SCAN, "to_arrow", lambda t, cx: t.to_arrow(), writes=False, rows=_keys_are),
    Case(
        P.SCAN,
        "scan_predicate",
        lambda t, cx: pa.table(t.scan(predicate=f"{cx.key} = 1")),
        lambda cx: {"predicate": f"{cx.key} = 1"},
        writes=False,
        rows=lambda cx, r: _keys_are(cx, r, [1]),
    ),
    Case(
        P.SCAN,
        "scan_beyond_grammar",
        lambda t, cx: pa.table(t.scan(predicate=_beyond(cx))),
        lambda cx: {"predicate": _beyond(cx)},
        writes=False,
        rows=lambda cx, r: _keys_are(cx, r, [1]),
    ),
    Case(
        P.SCAN,
        "scan_columns",
        lambda t, cx: pa.table(t.scan(columns=[cx.key])),
        lambda cx: {"columns": [cx.key]},
        writes=False,
        rows=_keys_are,
    ),
    Case(P.SCAN, "to_pandas", lambda t, cx: t.to_pandas(), writes=False),
    Case(P.SCAN, "to_polars", lambda t, cx: t.to_polars(), writes=False),
    Case(P.SCAN, "to_duckdb", lambda t, cx: t.to_duckdb().fetchall(), writes=False),
    Case(
        P.SCAN, "to_pyarrow_dataset", lambda t, cx: t.to_pyarrow_dataset().to_table(), writes=False
    ),
    Case(P.SCAN, "head", lambda t, cx: t.head(2), writes=False),
    Case(P.SCAN, "count", lambda t, cx: t.count(), writes=False),
    Case(P.SCAN, "plan_scan", _plan_scan, writes=False, rows=_keys_are),
    Case(
        P.TIME_TRAVEL,
        "version_0",
        lambda t, cx: t.to_arrow(version=0),
        lambda cx: {"version": 0},
        writes=False,
    ),
    Case(
        P.TIME_TRAVEL,
        "timestamp_latest",
        lambda t, cx: t.to_arrow(timestamp=cx.extra["ts"]),
        lambda cx: {"timestamp": cx.extra["ts"]},
        writes=False,
        rows=_keys_are,
        skip={UC: "staged commits carry the catalog's timestamps, not file times"},
    ),
    Case(
        P.CDF,
        "cdf",
        lambda t, cx: pa.table(t.cdf(starting_version=0)),
        lambda cx: {"starting_version": 0},
        writes=False,
    ),
    Case(P.CDF, "changes", _changes, lambda cx: {"starting_version": 0}, writes=False),
    Case(P.HISTORY, "history", lambda t, cx: t.history(), writes=False),
    Case(P.DETAIL, "detail", lambda t, cx: t.detail(), writes=False),
    Case(P.FILES, "files", lambda t, cx: t.files(), writes=False),
    # --- writes
    Case(
        P.APPEND,
        "append",
        lambda t, cx: t.append(cx.shifted(100)),
        expect=lambda cx, b: sorted(b + [k + 100 for k in b]),
    ),
    Case(
        P.APPEND,
        "append_commit_metadata",
        lambda t, cx: t.append(cx.shifted(100), commit_metadata={"contract": "yes"}),
        lambda cx: {"commit_metadata": {"contract": "yes"}},
        expect=lambda cx, b: sorted(b + [k + 100 for k in b]),
    ),
    Case(
        P.APPEND,
        "append_txn",
        lambda t, cx: t.append(cx.shifted(100), txn=("contract", 1)),
        lambda cx: {"txn": ("contract", 1)},
        expect=lambda cx, b: sorted(b + [k + 100 for k in b]),
    ),
    Case(
        P.APPEND,
        "append_max_commit_retries",
        lambda t, cx: t.append(cx.shifted(100), max_commit_retries=2),
        lambda cx: {"max_commit_retries": 2},
        expect=lambda cx, b: sorted(b + [k + 100 for k in b]),
    ),
    Case(
        P.APPEND,
        "plan_write",
        lambda t, cx: _plan_write(cx, t),
        serves=_WRITE_PLAN,
        expect=lambda cx, b: sorted(b + [k + 100 for k in b]),
    ),
    Case(
        P.APPEND,
        "plan_write_txn_metadata",
        lambda t, cx: _plan_write(cx, t, txn=("contract", 1), commit_metadata={"contract": "yes"}),
        lambda cx: {"txn": ("contract", 1), "commit_metadata": {"contract": "yes"}},
        serves=_WRITE_PLAN,
        expect=lambda cx, b: sorted(b + [k + 100 for k in b]),
    ),
    Case(
        P.MERGE_SCHEMA,
        "append_schema_merge",
        lambda t, cx: t.append(
            cx.shifted(100).append_column("extra", pa.array([1] * cx.before.num_rows, pa.int64())),
            schema_mode="merge",
        ),
        lambda cx: {"schema_mode": "merge"},
        expect=lambda cx, b: sorted(b + [k + 100 for k in b]),
    ),
    Case(
        P.OVERWRITE,
        "overwrite",
        lambda t, cx: t.overwrite(cx.shifted(100)),
        expect=lambda cx, b: [k + 100 for k in b],
    ),
    Case(
        P.OVERWRITE,
        "overwrite_commit_metadata",
        lambda t, cx: t.overwrite(cx.shifted(100), commit_metadata={"contract": "yes"}),
        lambda cx: {"commit_metadata": {"contract": "yes"}},
        expect=lambda cx, b: [k + 100 for k in b],
    ),
    Case(
        P.OVERWRITE,
        "overwrite_txn",
        lambda t, cx: t.overwrite(cx.shifted(100), txn=("contract", 1)),
        lambda cx: {"txn": ("contract", 1)},
        expect=lambda cx, b: [k + 100 for k in b],
    ),
    Case(
        P.REPLACE_WHERE,
        "overwrite_dynamic_partitions",
        lambda t, cx: t.overwrite(cx.only(1), partition_overwrite="dynamic"),
        lambda cx: {"partition_overwrite": "dynamic"},
        # Partition grp=x (ids 1, 2) is replaced by the one row; an unpartitioned
        # table is one partition, replaced whole (as Spark does).
        expect=lambda cx, b: [k for k in b if k != 2] if cx.fixture == "partitioned" else [1],
    ),
    Case(
        P.OVERWRITE,
        "plan_write_overwrite",
        lambda t, cx: _plan_write(cx, t, mode="overwrite"),
        serves=_WRITE_PLAN,
        expect=lambda cx, b: [k + 100 for k in b],
    ),
    Case(
        P.REPLACE_WHERE,
        "replace_where",
        lambda t, cx: t.overwrite(cx.only(1), predicate=f"{cx.key} = 1"),
        lambda cx: {"predicate": f"{cx.key} = 1"},
        expect=_same,
    ),
    Case(
        P.REPLACE_WHERE,
        "replace_where_beyond_grammar",
        lambda t, cx: t.overwrite(cx.only(1), predicate=_beyond(cx)),
        lambda cx: {"predicate": _beyond(cx)},
        expect=_same,
    ),
    # --- dml
    Case(
        P.DELETE,
        "delete",
        lambda t, cx: t.delete(f"{cx.key} = 1"),
        lambda cx: {"predicate": f"{cx.key} = 1"},
        expect=lambda cx, b: [k for k in b if k != 1],
    ),
    Case(
        P.DELETE,
        "delete_beyond_grammar",
        lambda t, cx: t.delete(_beyond(cx)),
        lambda cx: {"predicate": _beyond(cx)},
        expect=lambda cx, b: [k for k in b if k != 1],
    ),
    Case(
        P.DELETE,
        "delete_commit_metadata",
        lambda t, cx: t.delete(f"{cx.key} = 1", commit_metadata={"contract": "yes"}),
        lambda cx: {"predicate": f"{cx.key} = 1", "commit_metadata": {"contract": "yes"}},
        expect=lambda cx, b: [k for k in b if k != 1],
    ),
    Case(
        P.UPDATE,
        "update",
        lambda t, cx: t.update(_set_text(cx), predicate=f"{cx.key} = 1"),
        lambda cx: {"updates": _set_text(cx), "predicate": f"{cx.key} = 1"},
        expect=_same,
    ),
    Case(
        P.UPDATE,
        "update_new_values",
        lambda t, cx: t.update(new_values={cx.narrow: None}, predicate=f"{cx.key} = 1"),
        lambda cx: {"new_values": {cx.narrow: None}, "predicate": f"{cx.key} = 1"},
        expect=_same,
    ),
    Case(
        P.UPDATE,
        "update_beyond_grammar",
        lambda t, cx: t.update({cx.key: f"{cx.key} + 0"}, predicate=_beyond(cx)),
        lambda cx: {"updates": {cx.key: f"{cx.key} + 0"}, "predicate": _beyond(cx)},
        expect=_same,
        skip={"generated": "SET id would violate the generated id2 = id * 2 only if changed"},
    ),
    Case(
        P.MERGE,
        "merge",
        lambda t, cx: _merge(cx, t),
        _merge_args,
        expect=lambda cx, b: sorted([*b, 1001]),
    ),
    Case(
        P.UPDATE,
        "update_set_beyond_grammar",
        lambda t, cx: t.update({cx.narrow: f"{cx.narrow} * 2"}, predicate=f"{cx.key} = 1"),
        lambda cx: {"updates": {cx.narrow: f"{cx.narrow} * 2"}, "predicate": f"{cx.key} = 1"},
        expect=_same,
        skip={"legacy_calendar": "its date column has no arithmetic"},
    ),
    Case(
        P.UPDATE,
        "update_commit_metadata",
        lambda t, cx: t.update(
            _set_text(cx), predicate=f"{cx.key} = 1", commit_metadata={"contract": "yes"}
        ),
        lambda cx: {
            "updates": _set_text(cx),
            "predicate": f"{cx.key} = 1",
            "commit_metadata": {"contract": "yes"},
        },
        expect=_same,
    ),
    Case(
        P.MERGE,
        "merge_commit_metadata",
        lambda t, cx: t.merge(**_merge_args(cx), commit_metadata={"contract": "yes"})
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(),
        lambda cx: {**_merge_args(cx), "commit_metadata": {"contract": "yes"}},
        expect=lambda cx, b: sorted([*b, 1001]),
    ),
    # --- ddl
    Case(
        P.ADD_COLUMN,
        "add_column",
        lambda t, cx: t.add_column([pa.field("added", pa.int64())]),
        lambda cx: {"fields": [pa.field("added", pa.int64())]},
        expect=_same,
    ),
    Case(
        P.DROP_COLUMN,
        "drop_column",
        lambda t, cx: t.drop_column(cx.narrow),
        lambda cx: {"column": cx.narrow},
        expect=_same,
    ),
    Case(
        P.RENAME_COLUMN,
        "rename_column",
        lambda t, cx: t.rename_column(cx.narrow, "renamed"),
        lambda cx: {"old": cx.narrow, "new": "renamed"},
        expect=_same,
    ),
    Case(
        P.SET_PROPERTIES,
        "set_properties",
        lambda t, cx: t.set_properties({"contract.key": "value"}),
        lambda cx: {"properties": {"contract.key": "value"}},
        expect=_same,
    ),
    Case(
        P.SET_PROPERTIES,
        "set_properties_delta",
        lambda t, cx: t.set_properties({"delta.logRetentionDuration": "interval 60 days"}),
        lambda cx: {"properties": {"delta.logRetentionDuration": "interval 60 days"}},
        expect=_same,
    ),
    Case(
        P.UNSET_PROPERTIES,
        "unset_properties",
        lambda t, cx: t.unset_properties(["contract.absent"]),
        lambda cx: {"keys": ["contract.absent"]},
        expect=_same,
    ),
    Case(
        P.ADD_FEATURE,
        "add_feature",
        lambda t, cx: t.add_feature("deletionVectors", allow_protocol_versions_increase=True),
        lambda cx: {"feature": "deletionVectors", "allow_protocol_versions_increase": True},
        expect=_same,
    ),
    Case(
        P.DROP_FEATURE,
        "drop_feature",
        lambda t, cx: t.drop_feature("deletionVectors"),
        lambda cx: {"feature": "deletionVectors"},
        expect=_same,
    ),
    Case(
        P.ADD_CONSTRAINT,
        "add_constraint",
        lambda t, cx: t.add_constraint({"key_positive": f"{cx.key} > 0"}),
        lambda cx: {"constraints": {"key_positive": f"{cx.key} > 0"}},
        expect=_same,
    ),
    Case(
        P.DROP_CONSTRAINT,
        "drop_constraint",
        lambda t, cx: t.drop_constraint("id_positive", if_exists=True),
        lambda cx: {"name": "id_positive", "if_exists": True},
        expect=_same,
    ),
    Case(
        P.SET_COMMENT,
        "set_comment",
        lambda t, cx: t.set_comment("contract"),
        lambda cx: {"comment": "contract"},
        expect=_same,
    ),
    Case(
        P.SET_COLUMN_COMMENT,
        "set_column_comment",
        lambda t, cx: t.set_column_comment(cx.key, "the key"),
        lambda cx: {"column": cx.key, "comment": "the key"},
        expect=_same,
    ),
    Case(
        P.ALTER_COLUMN_TYPE,
        "alter_column_type",
        lambda t, cx: t.alter_column_type(*_widen(cx)),
        lambda cx: {"column": _widen(cx)[0], "new_type": _widen(cx)[1]},
        expect=_same,
    ),
    Case(
        P.SET_NOT_NULL,
        "set_not_null",
        lambda t, cx: t.set_not_null(cx.key),
        lambda cx: {"column": cx.key},
        expect=_same,
    ),
    Case(
        P.DROP_NOT_NULL,
        "drop_not_null",
        lambda t, cx: t.drop_not_null(cx.key),
        lambda cx: {"column": cx.key},
        expect=_same,
    ),
    Case(
        P.CLUSTER_BY,
        "cluster_by",
        lambda t, cx: t.cluster_by([cx.key]),
        lambda cx: {"columns": [cx.key]},
        expect=_same,
    ),
    Case(
        P.CLUSTER_BY,
        "cluster_by_auto",
        lambda t, cx: t.cluster_by("auto"),
        lambda cx: {"columns": "auto"},
        expect=_same,
    ),
    Case(
        P.SET_PROPERTIES,
        "set_properties_commit_metadata",
        lambda t, cx: t.set_properties({"contract.key": "v"}, commit_metadata={"contract": "yes"}),
        lambda cx: {"properties": {"contract.key": "v"}, "commit_metadata": {"contract": "yes"}},
        expect=_same,
    ),
    # --- maintenance
    Case(P.OPTIMIZE, "optimize", lambda t, cx: t.optimize(), expect=_same),
    Case(
        P.OPTIMIZE,
        "optimize_commit_metadata",
        lambda t, cx: t.optimize(commit_metadata={"contract": "yes"}),
        lambda cx: {"commit_metadata": {"contract": "yes"}},
        expect=_same,
    ),
    Case(
        P.OPTIMIZE,
        "optimize_full",
        lambda t, cx: t.optimize(full=True),
        lambda cx: {"full": True},
        expect=_same,
    ),
    Case(
        P.OPTIMIZE,
        "optimize_predicate",
        lambda t, cx: t.optimize(predicate=f"{cx.key} = 1"),
        lambda cx: {"predicate": f"{cx.key} = 1"},
        expect=_same,
    ),
    Case(
        P.ZORDER,
        "optimize_zorder",
        lambda t, cx: t.optimize(zorder_by=[cx.key]),
        lambda cx: {"zorder_by": [cx.key]},
        serves=("optimize", "zorder"),
        expect=_same,
    ),
    Case(
        P.ZORDER,
        "z_order",
        lambda t, cx: t.z_order(cx.key),
        lambda cx: {"zorder_by": [cx.key]},
        expect=_same,
    ),
    Case(
        P.VACUUM,
        "vacuum_dry_run",
        lambda t, cx: t.vacuum(dry_run=True),
        lambda cx: {"dry_run": True},
        expect=_same,
    ),
    Case(
        P.VACUUM,
        "vacuum_lite",
        lambda t, cx: t.vacuum(lite=True, dry_run=False),
        lambda cx: {"lite": True, "dry_run": False},
        expect=_same,
    ),
    Case(
        P.VACUUM,
        "vacuum_full",
        lambda t, cx: t.vacuum(dry_run=False),
        lambda cx: {"dry_run": False},
        expect=_same,
    ),
    Case(
        P.RESTORE,
        "restore_0",
        lambda t, cx: t.restore(0),
        lambda cx: {"target": 0},
        expect=lambda cx, b: cx.extra["v0"],
        no_engine_ok=True,
    ),
    Case(
        P.REPAIR,
        "repair_dry_run",
        lambda t, cx: t.repair(dry_run=True),
        lambda cx: {"dry_run": True},
        expect=_same,
    ),
    Case(P.CHECKPOINT, "checkpoint", lambda t, cx: t.checkpoint(), expect=_same),
    Case(P.LOG_COMPACTION, "compact_logs", lambda t, cx: t.compact_logs(), expect=_same),
    Case(P.CLEANUP_METADATA, "cleanup_metadata", lambda t, cx: t.cleanup_metadata(), expect=_same),
    Case(P.PUBLISH, "publish", lambda t, cx: t.publish(), expect=_same),
    Case(P.GENERATE, "generate", lambda t, cx: t.generate(), expect=_same),
    Case(P.REORG, "reorg", lambda t, cx: t.reorg(purge=True), expect=_same),
    Case(
        P.CLONE,
        "clone",
        lambda t, cx: t.clone(_clone_target(cx)),
        lambda cx: {"target": _clone_target(cx)},
        expect=_same,
    ),
    Case(P.ANALYZE, "analyze", lambda t, cx: t.analyze(), expect=_same),
    Case(P.SYNC_ICEBERG, "sync_iceberg", lambda t, cx: t.sync_iceberg(), expect=_same),
    Case(P.REFRESH, "refresh", lambda t, cx: t.refresh(), expect=_same),
]

#: Operations with no Table method: asserted refused (INCREMENTAL), or covered
#: at the connection level (CREATE, CONVERT) by the tests below the matrix.
NO_TABLE_METHOD = {P.INCREMENTAL, P.CREATE, P.CONVERT}


def test_every_operation_has_a_case() -> None:
    covered = {c.op for c in CASES} | NO_TABLE_METHOD
    missing = sorted(op.value for op in Operation if op not in covered)
    assert not missing, f"add a contract case for {missing}"


# ------------------------------------------------------------ known findings

#: Tables whose writes delta-rs serves (the rest go to the kernel).
_KERNEL_WRITES = {"clustered", "defaults", "ict", "row_tracking", "type_widening", UC}
_ALL = set(TABLES)

try:
    import pytz  # type: ignore[import-untyped]  # noqa: F401

    _HAS_PYTZ = True
except ImportError:
    _HAS_PYTZ = False

#: case id -> (finding, tables it fails on). Each entry is a strict xfail: a
#: finding written down, which fails the suite (XPASS) once it is fixed.
#: The findings are described with repros in the contract findings report.
KNOWN: dict[str, tuple[str, set[str]]] = {
    "scan_beyond_grammar": (
        "C1: can(scan, predicate=<arithmetic>) says ok via kernel; scan raises PredicateError",
        _ALL,
    ),
    "plan_write": (
        "C3: plan_write routes on distributed_write, which can() cannot express: "
        "can(append) names delta-rs, the plan uses the kernel or refuses",
        _ALL - _KERNEL_WRITES,
    ),
    "plan_write_txn_metadata": (
        "C3: plan_write routes on distributed_write, which can() cannot express",
        _ALL - _KERNEL_WRITES,
    ),
    "plan_write_overwrite": (
        "C3: plan_write routes on distributed_write, which can() cannot express",
        _ALL - _KERNEL_WRITES - {"append_only"},
    ),
    "add_feature": (
        "C4: can(add_feature, feature=...) names delta-rs; the call routes on features= "
        "and the kernel serves",
        _ALL - _KERNEL_WRITES,
    ),
    "delete_commit_metadata": (
        "C5: can(delete, commit_metadata=...) says ok via kernel; the kernel refuses the option",
        _KERNEL_WRITES - {"row_tracking"} | {"dv", "dv_cdf", "legacy_calendar"},
    ),
    "update_commit_metadata": (
        "C5: can(update, commit_metadata=...) says ok via kernel; the kernel refuses the option",
        _KERNEL_WRITES - {"row_tracking"} | {"dv", "legacy_calendar"},
    ),
    "merge": (
        "C6: MERGE on an appendOnly table: can() says ok; delta-rs's raw CommitFailedError escapes",
        {"append_only"},
    ),
    "merge_commit_metadata": (
        "C6: MERGE on an appendOnly table: can() says ok; delta-rs's raw CommitFailedError escapes",
        {"append_only"},
    ),
    "overwrite_dynamic_partitions": (
        "C9: can(overwrite, partition_overwrite='dynamic') says ok on an unpartitioned table "
        "delta-rs writes; the call refuses",
        _ALL - _KERNEL_WRITES - {"partitioned", "append_only", "legacy_calendar"},
    ),
}
if not _HAS_PYTZ:
    KNOWN["to_duckdb"] = (
        "C7: to_duckdb on a zoned timestamp column without pytz: duckdb's raw "
        "InvalidInputException escapes",
        {"legacy_calendar"},
    )


def _known(table: str, case: str) -> str | None:
    reason, tables = KNOWN.get(case, ("", set()))
    return reason if table in tables else None


def _params() -> list[Any]:
    out = []
    for table in TABLES:
        for case in CASES:
            marks = []
            reason = _known(table, case.id)
            if reason is not None:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            out.append(pytest.param(table, case, id=f"{table}::{case.id}", marks=marks))
    return out


# ------------------------------------------------------------------ runner


def _context(tables: Tables, fixture: str, writes: bool) -> Ctx:
    if fixture == UC:
        conn = connect_uc(tables.uc)
        name = tables.catalog_managed(writable=writes)
        table = conn.table(name)
        cx = Ctx(conn, table, name, fixture, tables)
    else:
        spec: Fixture = next(f for f in FIXTURES if f.name == fixture)
        conn = connect_path()
        name = tables.path(fixture, writable=writes)
        table = conn.open_table(name)
        cx = Ctx(conn, table, name, fixture, tables, generated="id2" in spec.extra)
        if spec.calendar:
            cx.key, cx.text, cx.narrow = "rid", None, "dt"
    return cx


def _deltars_keys(cx: Ctx) -> list[int] | None:
    """The keys as delta-rs's own reader sees them, or None if it cannot open the table."""
    if cx.fixture == UC:
        return None
    try:
        from deltalake import DeltaTable, QueryBuilder

        dt_ = DeltaTable(cx.name)
        result = QueryBuilder().register("t", dt_).execute(f"SELECT {cx.key} FROM t").read_all()
    except Exception:
        return None
    return sorted(pa.table(result).column(0).to_pylist())


def _is_refusal(exc: BaseException | None) -> bool:
    return isinstance(exc, UnreachableTableError)


@pytest.mark.parametrize(("fixture", "case"), _params())
def test_can_agrees_with_the_call(contract_tables: Tables, fixture: str, case: Case) -> None:
    if fixture in case.skip:
        pytest.skip(case.skip[fixture])
    cx = _context(contract_tables, fixture, case.writes)
    served = instrument(cx.conn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx.before = cx.table.to_arrow()
        if case.op is P.RESTORE:
            cx.extra["v0"] = cx.keys(cx.rows(version=0))
        if case.op is P.TIME_TRAVEL and fixture != UC:
            cx.extra["ts"] = _latest_timestamp(cx.name)
        before_keys = cx.keys(cx.before)

        verdict = cx.table.can(case.op, **case.shape(cx))
        served.active = True
        error: BaseException | None = None
        result: Any = None
        try:
            result = case.call(cx.table, cx)
        except Exception as exc:
            error = exc
        finally:
            served.active = False

    if not verdict.ok:
        assert _is_refusal(error), f"can() refused ({verdict.reason[:300]}) but the call " + (
            "succeeded" if error is None else f"raised {type(error).__name__}: {error}"
        )
        if case.writes:
            # A refusal comes before any side effect.
            assert cx.keys() == before_keys
        return

    assert error is None, (
        f"can() said ok via {verdict.engine}, but the call raised "
        f"{type(error).__name__}: {str(error)[:500]}"
    )
    methods = case.serving_methods()
    engines = served.kinds_of(methods)
    if engines or not case.no_engine_ok:
        assert engines == [str(verdict.engine)], (
            f"can() named {verdict.engine}, but the call was served by {engines} ({served.calls})"
        )
    if case.rows is not None and result is not None:
        case.rows(cx, result)
    if case.expect is not None:
        wanted = case.expect(cx, before_keys)
        assert cx.keys() == wanted, "the table does not hold the rows the operation implies"
        theirs = _deltars_keys(cx)
        if theirs is not None:
            assert theirs == wanted, "delta-rs reads different rows than this library"


# --------------------------------------------------- connection-level operations

CREATE_SHAPES: dict[str, dict[str, Any]] = {
    "plain": {},
    "partitioned": {"partition_by": ["grp"]},
    "clustered": {"cluster_by": ["id"]},
    "cm_name": {"properties": {"delta.columnMapping.mode": "name"}},
    "dv_cdf": {
        "properties": {"delta.enableDeletionVectors": "true", "delta.enableChangeDataFeed": "true"}
    },
    "row_tracking": {"properties": {"delta.enableRowTracking": "true"}},
    "ict": {"properties": {"delta.enableInCommitTimestamps": "true"}},
    "type_widening": {"properties": {"delta.enableTypeWidening": "true"}},
    "v2_checkpoint": {"properties": {"delta.checkpointPolicy": "v2"}},
    "append_only": {"properties": {"delta.appendOnly": "true"}},
    "feature_signal": {"properties": {"delta.feature.variantType": "supported"}},
}


@pytest.mark.parametrize("shape", list(CREATE_SHAPES), ids=list(CREATE_SHAPES))
def test_can_create_agrees_with_create_table(contract_tables: Tables, shape: str) -> None:
    from tests.contract.tables import base_data

    kwargs = CREATE_SHAPES[shape]
    conn = connect_path()
    served = instrument(conn)
    path = contract_tables.fresh_dir(f"create-{shape}")
    verdict = conn.open_table(path).can(P.CREATE, **kwargs)
    served.active = True
    try:
        conn.create_table(path, base_data().schema, **kwargs)
        error = None
    except Exception as exc:
        error = exc
    served.active = False
    if not verdict.ok:
        assert _is_refusal(error), f"can() refused ({verdict.reason}) but create gave {error!r}"
        return
    assert error is None, f"can() said ok via {verdict.engine}, create raised {error!r}"
    assert served.kinds_of(frozenset({"create"})) == [str(verdict.engine)], served.calls
    assert conn.open_table(path).to_arrow().num_rows == 0


@pytest.mark.xfail(
    strict=True,
    reason="C8: can(convert) on a Parquet directory refuses (no Delta log to read); "
    "convert_to_delta converts it",
)
def test_can_convert_agrees_with_convert_to_delta(contract_tables: Tables) -> None:
    import pyarrow.parquet as pq

    from tests.contract.tables import base_data

    conn = connect_path()
    served = instrument(conn)
    path = pathlib.Path(contract_tables.fresh_dir("convert"))
    path.mkdir()
    pq.write_table(base_data(), path / "part-0.parquet")
    verdict = conn.open_table(str(path)).can(P.CONVERT)
    served.active = True
    try:
        table = conn.convert_to_delta(str(path))
        error = None
    except Exception as exc:
        error = exc
    served.active = False
    if not verdict.ok:
        assert _is_refusal(error), f"can() refused ({verdict.reason}) but convert gave {error!r}"
        return
    assert error is None, f"can() said ok via {verdict.engine}, convert raised {error!r}"
    assert served.kinds_of(frozenset({"convert"})) == [str(verdict.engine)], served.calls
    assert sorted(table.to_arrow().column("id").to_pylist()) == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("fixture", ["plain", UC])
def test_incremental_is_refused_everywhere(contract_tables: Tables, fixture: str) -> None:
    cx = _context(contract_tables, fixture, writes=False)
    assert not cx.table.can(P.INCREMENTAL).ok


@pytest.mark.xfail(
    strict=True,
    reason="C10: a catalog-managed table without vacuumProtocolCheck: can(append) says ok via "
    "kernel; the kernel's committer refuses with a generic InvalidArgumentError",
)
def test_catalog_managed_without_vacuum_protocol_check(contract_tables: Tables) -> None:
    conn = connect_uc(contract_tables.uc)
    name = contract_tables.catalog_managed(writable=True, vacuum_protocol_check=False)
    table = conn.table(name)
    data = table.to_arrow()
    verdict = table.can(P.APPEND)
    try:
        table.append(data)
        error = None
    except Exception as exc:
        error = exc
    if verdict.ok:
        assert error is None, f"can() said ok via {verdict.engine}, append raised {error!r}"
    else:
        assert _is_refusal(error), f"can() refused ({verdict.reason}), append gave {error!r}"
