"""One request, derived once, routed the same way by `Table.can()` and the call.

`t.can(op, **args)` and the method it describes used to work out what the
request needs separately: `can()` in one translation of the arguments, each
method in its own code at the call site. The two drifted -- can() said yes and
the call was refused, or can() named one engine and another served -- every
time a method learned a need that `can()` did not.

Now both build a `Request` here, from the same arguments, and route on it: the
operation the call really is (an append with ``schema_mode="merge"`` is
MERGE_SCHEMA), the needs its arguments and data imply, and the shape each
engine's ``supports()`` judges. `can()` is the router's verdict on that
request, and the call asks the router for an engine for the very same request.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .capability import ENGINE_METHODS, READ_OPERATIONS, Capability, Operation
from .errors import ChangeFeedSchemaChangeError, EngineLimitError, UnreachableTableError

if TYPE_CHECKING:
    from .table import Table

#: The call's data was not given (can() without data=).
NO_DATA: Any = object()


@dataclasses.dataclass(frozen=True)
class Request:
    """What one call routes on."""

    #: The operation the call routes as.
    operation: Operation
    #: Requirements engines declare as ``supports_<need>`` (and router hints).
    needs: frozenset[str]
    #: The call's arguments (its data left out), which each engine's
    #: ``supports()`` judges.
    shape: Mapping[str, Any]
    #: The operation that was asked, when the call routes as another one.
    asked: Operation


#: Table method names `can()` accepts in place of an operation, and the
#: arguments that name them. `plan_write` is an append or an overwrite written
#: by workers and committed together, so it needs ``distributed_write``.
METHOD_OPERATIONS: dict[str, tuple[Operation, dict[str, Any]]] = {
    "z_order": (Operation.ZORDER, {}),
    "compact_logs": (Operation.LOG_COMPACTION, {}),
    "changes": (Operation.INCREMENTAL, {}),
    "plan_write": (Operation.APPEND, {"distributed": True}),
    "plan_scan": (Operation.SCAN, {"distributed": True}),
    "to_arrow": (Operation.SCAN, {}),
    "create_table": (Operation.CREATE, {}),
    "convert_to_delta": (Operation.CONVERT, {}),
}


def derive(table: Table, operation: Operation, args: Mapping[str, Any], data: Any) -> Request:
    """The request a call of `operation` with `args` (and `data`) makes on `table`."""
    shape = {k: v for k, v in args.items() if k not in ("source", "data")}
    rule = _RULES.get(operation)
    op, needs = operation, set[str]()
    if rule is not None:
        op, needs = rule(table, operation, shape, data)
    return Request(op, frozenset(needs), shape, operation)


#: What the calls refuse on a handle opened at a past version (see
#: `Table._check_writable`): every write, since every engine writes to the
#: latest version.
_PINNED_REFUSED: frozenset[Operation] = frozenset(
    {
        Operation.APPEND,
        Operation.OVERWRITE,
        Operation.REPLACE_WHERE,
        Operation.MERGE_SCHEMA,
        Operation.DELETE,
        Operation.UPDATE,
        Operation.MERGE,
        Operation.OPTIMIZE,
        Operation.ZORDER,
        Operation.RESTORE,
        Operation.ADD_COLUMN,
        Operation.DROP_COLUMN,
        Operation.RENAME_COLUMN,
        Operation.SET_PROPERTIES,
        Operation.ADD_FEATURE,
        Operation.DROP_FEATURE,
        Operation.ADD_CONSTRAINT,
        Operation.DROP_CONSTRAINT,
        Operation.UNSET_PROPERTIES,
        Operation.SET_COMMENT,
        Operation.SET_COLUMN_COMMENT,
        Operation.ALTER_COLUMN_TYPE,
        Operation.SET_NOT_NULL,
        Operation.DROP_NOT_NULL,
        Operation.CLUSTER_BY,
        Operation.REORG,
    }
)


def refusal(table: Table, request: Request) -> Capability | None:
    """The refusal the call makes of `request` before any engine is asked, if any.

    A handle pinned to a version refuses every write in the method; can()
    asked only the router, which judged the latest table and said yes.
    """
    version = table._version
    if request.operation is Operation.OVERWRITE and request.shape.get("schema_mode") == "overwrite":
        kept = _replace_keeps(table)
        if kept:
            return Capability(
                request.asked,
                ok=False,
                reason=f"replacing the schema would keep the table's {kept}: delta-rs carries "
                "them over into the new table, where REPLACE TABLE drops them, and fails or "
                "enforces them on data that may not have their columns",
                remedy="drop the constraints first (drop_constraint()), or write the data as "
                "a new table (write_table() at another location)",
            )
    if version is None or request.operation not in _PINNED_REFUSED:
        return None
    return Capability(
        request.asked,
        ok=False,
        reason=f"this handle is pinned to version {version}, and every engine writes to the "
        "latest version",
        remedy="open the table without version= to write to it",
    )


def _replace_keeps(table: Table) -> str:
    """What of the old table a schema-replacing overwrite would wrongly keep ("" if nothing)."""
    from .errors import DeltaSwampError

    kept = []
    properties = table._enrich().properties
    if any(str(k).lower().startswith("delta.constraints.") for k in properties):
        kept.append("CHECK constraints")
    try:
        fields = list(table.schema())
    except (DeltaSwampError, ImportError):
        fields = []
    computed = (b"delta.generationExpression", b"delta.identity.")
    if any(f.metadata and any(k.startswith(computed) for k in f.metadata) for f in fields):
        kept.append("generated or identity columns")
    return " and ".join(kept)


# ------------------------------------------------------------------- reads


def _read(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    get = shape.get
    predicate = get("predicate")
    needs: set[str] = set()
    if get("distributed"):
        needs.add("distributed_scan")
    if predicate is not None:
        needs |= table._predicate_needs(predicate)
        # A predicate the kernel's grammar does not read (`id % 7 = 1`) goes
        # to an engine that evaluates SQL, as for DELETE: the kernel scan
        # raised PredicateError after can() had named it.
        needs |= table._expression_needs(predicate)
    if not get("distributed"):
        needs |= table._variant_needs(get("columns"), predicate)
    if get("timestamp") is not None:
        needs.add("timestamp_travel")
    if get("version") is not None or get("timestamp") is not None or table._version is not None:
        op = Operation.TIME_TRAVEL
    return op, needs


def _feed(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    """cdf() and changes(), which read rows as a scan does and route the same way.

    changes() is the incremental read, and it follows the change data feed:
    can(INCREMENTAL) refused on every table while changes() served it. The
    feed was routed with no needs at all, so can("cdf", predicate="id % 2 = 1")
    named the kernel, whose grammar then refused the call that delta-rs serves.
    """
    get = shape.get
    predicate = get("predicate")
    needs: set[str] = set()
    if predicate is not None:
        needs |= table._predicate_needs(predicate)
        needs |= table._expression_needs(predicate)
    needs |= table._variant_needs(get("columns"), predicate)
    if get("allow_out_of_range"):
        # Only delta-rs reads past the table's last version; the kernel
        # refused it at read time, so can() named it and delta-rs served.
        needs.add("out_of_range_feed")
    else:
        shape.pop("allow_out_of_range", None)
    return Operation.CDF, needs


# ------------------------------------------------------------------ writes


def _write(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    get = shape.get
    schema_mode = get("schema_mode")
    partition_overwrite = get("partition_overwrite") or "static"
    if get("mode") == "overwrite" and get("distributed"):
        op = Operation.OVERWRITE
    needs = set(
        table._write_needs(
            schema_mode,
            get("commit_metadata"),
            get("txn"),
            get("writer_properties"),
            partition_overwrite,
        )
    )
    if get("distributed"):
        needs.add("distributed_write")
    if op is Operation.APPEND and schema_mode == "merge":
        op = Operation.MERGE_SCHEMA
    elif op is Operation.OVERWRITE and (
        get("predicate") is not None or partition_overwrite == "dynamic"
    ):
        op = Operation.REPLACE_WHERE
    if op is Operation.REPLACE_WHERE:
        needs |= table._expression_needs(get("predicate"))
    if data is not NO_DATA:
        needs |= table._data_needs(data, get("partition_by"))
        if schema_mode != "overwrite":
            needs |= table._default_needs(data)
        needs |= table._identity_needs(data)
    return op, needs


def _commit_options(table: Table, shape: dict[str, Any]) -> set[str]:
    """The needs of DML's commit options, judged as a write's are."""
    return set(
        table._write_needs(
            None, shape.get("commit_metadata"), None, shape.get("writer_properties"), "static"
        )
    )


def _delete(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    return Operation.DELETE, set(table._expression_needs(shape.get("predicate"))) | (
        _commit_options(table, shape) | table._char_needs(shape.get("predicate"))
    )


def _update(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    updates, new_values = shape.get("updates"), shape.get("new_values")
    needs = set(table._update_needs(updates, False) | table._update_needs(new_values, True))
    defaults, default_needs = table._update_defaults(updates)
    needs |= default_needs
    spelled = (
        {k: defaults.get(k, v) for k, v in updates.items()} if isinstance(updates, dict) else None
    )
    needs |= table._expression_needs(shape.get("predicate"), spelled)
    needs |= table._char_needs(shape.get("predicate"))
    return Operation.UPDATE, needs | _commit_options(table, shape)


#: MERGE builder clauses whose commit removes (rewrites or deletes) target rows.
_REMOVING_CLAUSES = ("when_matched_update", "when_matched_delete", "when_not_matched_by_source")


def _merge(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    needs: set[str] = set()
    if data is not NO_DATA:
        needs |= table._data_needs(data)
    if shape.get("merge_schema"):
        # The kernel MERGE cannot evolve the schema; routed without the need,
        # can() said "via kernel" and the call then refused.
        needs.add("schema_merge")
    clauses = shape.get("clauses")
    if isinstance(clauses, str):
        # One clause named on its own; a set or a single string was ignored,
        # and can() said yes on an append-only table the call then refused.
        clauses = (clauses,)
    if not isinstance(clauses, (list, tuple, set, frozenset)):
        return Operation.MERGE, needs
    # Each clause is its method name, or (name, condition) for one with a
    # condition, as the builder reports them at execute().
    named = [
        (str(c[0]), c[1] if len(c) > 1 else None)
        if isinstance(c, (list, tuple)) and c
        else (str(c), None)
        for c in clauses
    ]
    if any(name.startswith(_REMOVING_CLAUSES) for name, _ in named):
        needs.add("removes_rows")
    inserts = [
        condition
        for name, condition in named
        if name.startswith("when_not_matched") and not name.startswith("when_not_matched_by")
    ]
    if inserts and inserts[-1] is not None and _change_feed_on(table):
        # delta-rs inserts an all-NULL row for each source row the last
        # NOT MATCHED condition rejects on a change-feed table, and refuses
        # at execute(); can() named it all the same.
        needs.add("conditional_insert_with_feed")
    if data is not NO_DATA and named:
        needs |= table._identity_needs(data, [name for name, _ in named])
    return Operation.MERGE, needs


def _change_feed_on(table: Table) -> bool:
    properties = table._enrich().properties
    return str(properties.get("delta.enableChangeDataFeed", "false")).lower() == "true"


# ------------------------------------------------------------- maintenance


def _optimize(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    # z_order(columns) and optimize(zorder_by=) are the same Z-ORDER.
    if shape.get("zorder_by") or (op is Operation.ZORDER and shape.get("columns")):
        op = Operation.ZORDER
    needs: set[str] = set()
    if shape.get("full"):
        needs.add("optimize_full")
    if shape.get("predicate") is not None:
        needs.add("optimize_predicate")
    return op, needs


def _vacuum(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    # Judged as the call runs by default: t.vacuum() is a dry run, and
    # can("vacuum") refused it by judging a real one. The router accepts a dry
    # run on tables a real VACUUM, which commits, cannot touch, and refuses a
    # full one where delta-rs would delete live deletion vectors.
    shape["dry_run"] = bool(shape.get("dry_run", True))
    shape["lite"] = bool(shape.get("lite", False))
    return Operation.VACUUM, set()


def _repair(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    # A dry run commits nothing, so the router may accept it on tables a real
    # REPAIR (which commits removes) cannot touch.
    shape["dry_run"] = shape.get("dry_run") is True
    return Operation.REPAIR, set()


# --------------------------------------------------------------------- ddl


def _add_feature(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    # The engines judge the list of names (`features=`); the call takes one
    # name or several as `feature`. can(add_feature, feature=...) passed the
    # call's spelling through, no engine read it, and it named delta-rs while
    # the call went to the kernel.
    feature = shape.pop("feature", None)
    if feature is not None and shape.get("features") is None:
        shape["features"] = (
            list(feature) if isinstance(feature, (list, tuple, set, frozenset)) else [feature]
        )
    return Operation.ADD_FEATURE, set()


def _set_properties(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    from .properties import with_checkpoint_stats

    properties = shape.get("properties")
    if isinstance(properties, dict) and properties:
        properties = _spelled(properties)
        shape["properties"] = with_checkpoint_stats(properties, table.properties()) or properties
    return Operation.SET_PROPERTIES, set()


def _spelled(properties: dict[Any, Any]) -> dict[Any, Any]:
    """Property values in the spelling every engine parses (True -> 'true', 10 -> '10').

    The kernel's path normalised a value delta-rs's refused (a bool, ' true ',
    '7 days'), so can() named delta-rs, which then refused the call. A value
    neither reads is left as it is, for the engine to refuse.
    """
    from .engine.metadata import _canonical_key, _normalise_value

    spelled = {}
    for key, raw in properties.items():
        canonical = _canonical_key(key) if isinstance(key, str) else key
        if raw is None or canonical == "delta.dataSkippingStatsColumns":
            spelled[key] = raw  # this one is checked against the schema
            continue
        try:
            spelled[key] = _normalise_value(canonical, raw, None, "set properties")  # type: ignore[arg-type]
        except Exception:
            spelled[key] = raw
    return spelled


def _cluster_by(
    table: Table, op: Operation, shape: dict[str, Any], data: Any
) -> tuple[Operation, set[str]]:
    columns = shape.get("columns")
    auto = isinstance(columns, str) and columns.lower() == "auto"
    return Operation.CLUSTER_BY, {"auto_clustering"} if auto else set()


_Rule = Callable[["Table", Operation, dict[str, Any], Any], tuple[Operation, set[str]]]

_RULES: dict[Operation, _Rule] = {
    Operation.SCAN: _read,
    Operation.TIME_TRAVEL: _read,
    Operation.CDF: _feed,
    Operation.INCREMENTAL: _feed,
    Operation.APPEND: _write,
    Operation.MERGE_SCHEMA: _write,
    Operation.OVERWRITE: _write,
    Operation.REPLACE_WHERE: _write,
    Operation.DELETE: _delete,
    Operation.UPDATE: _update,
    Operation.MERGE: _merge,
    Operation.OPTIMIZE: _optimize,
    Operation.ZORDER: _optimize,
    Operation.VACUUM: _vacuum,
    Operation.REPAIR: _repair,
    Operation.ADD_FEATURE: _add_feature,
    Operation.SET_PROPERTIES: _set_properties,
    Operation.CLUSTER_BY: _cluster_by,
}


# ------------------------------------------------------------ strict routing

#: Set to 1 to check the contract as calls run (the test suite does): an engine
#: the router chose must not then refuse the call itself while another engine
#: would have served it.
STRICT_ROUTING_ENV = "DELTASWAMP_STRICT_ROUTING"

#: Methods that serve an operation besides its `ENGINE_METHODS` one.
_ALSO_SERVING: dict[Operation, frozenset[str]] = {
    Operation.SCAN: frozenset({"plan_scan", "metadata_count"}),
    Operation.TIME_TRAVEL: frozenset({"plan_scan", "metadata_count"}),
}


class RoutingContractViolation(BaseException):
    """Strict routing: an engine refused a call that another engine would have served.

    Raised only with ``DELTASWAMP_STRICT_ROUTING=1``. Such a refusal is a
    condition of the table or the arguments that ``supports()`` or the
    request's needs should have stated, so that routing moved on to the engine
    that can serve the call and ``can()`` named it. A refusal no engine would
    avoid (a version that does not exist, rows that break a constraint) is
    about the request, not routing, and is left as it is. A BaseException, so
    the handlers that try the next engine on a failure do not swallow it.
    """


def strict_engine(
    engine: Any,
    operation: Operation,
    elsewhere: Callable[[Any], Any] | None = None,
    engines: Mapping[Any, Any] | None = None,
) -> Any:
    """`engine`, checked for refusals after routing when strict routing is on.

    `elsewhere(kind)` is the router's verdict on the same request with engine
    `kind` excluded. With `engines` (the router's), a read's refusal is put to
    the engine named there as well: a read changes nothing, and one that both
    refuse (a version the log does not hold) is about the request.
    """
    if os.environ.get(STRICT_ROUTING_ENV) != "1" or engine is None:
        return engine
    return _StrictEngine(engine, operation, elsewhere, engines)


class _StrictEngine:
    """Delegates to an engine, turning a routing refusal from it into a violation.

    It passes for the engine (`isinstance`, pickling, attributes), so the code
    that routed it runs as it would without the check.
    """

    __slots__ = ("_elsewhere", "_engines", "_inner", "_operation")

    def __init__(
        self,
        inner: Any,
        operation: Operation,
        elsewhere: Callable[[Any], Any] | None,
        engines: Mapping[Any, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_operation", operation)
        object.__setattr__(self, "_elsewhere", elsewhere)
        object.__setattr__(self, "_engines", engines)

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        # The inner object's own answer, not type(): it may be the error
        # boundary (engine/boundary.py), which reports the engine's class.
        cls: type = self._inner.__class__
        return cls

    def __reduce_ex__(self, protocol: Any) -> Any:
        return self._inner.__reduce_ex__(protocol)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._inner, name)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        operation = self._operation
        serving = name == ENGINE_METHODS.get(operation) or name in _ALSO_SERVING.get(operation, ())
        if not serving or not callable(attr):
            # A probe (the log, a snapshot) is not the call the router routed.
            return attr
        kind = getattr(self._inner, "kind", None)
        elsewhere = self._elsewhere
        engines = self._engines if operation in READ_OPERATIONS else None

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return attr(*args, **kwargs)
            except (EngineLimitError, ChangeFeedSchemaChangeError):
                # Declared read-time limits, which the caller hands to the
                # next engine (see `Table._read`).
                raise
            except UnreachableTableError as exc:
                other = elsewhere(kind) if elsewhere is not None else None
                if other is None or not getattr(other, "ok", False):
                    raise
                if engines is not None and _refused_there_too(
                    engines.get(other.engine), name, args, kwargs
                ):
                    raise
                raise RoutingContractViolation(
                    f"strict routing: {operation.value} was routed to "
                    f"{getattr(kind, 'value', kind)}, whose {name}() then refused it, while "
                    f"{getattr(other.engine, 'value', other.engine)} would have served it: {exc}"
                ) from exc

        return call


def _refused_there_too(engine: Any, name: str, args: Any, kwargs: Any) -> bool:
    """Whether `engine` refuses the same read as a request error (not by a limit of its own)."""
    from .errors import DeltaSwampError

    if engine is None:
        return False
    try:
        result = getattr(engine, name)(*args, **kwargs)
    except EngineLimitError:
        return False
    except DeltaSwampError:
        return True
    close = getattr(result, "close", None)
    if callable(close):
        close()
    return False
