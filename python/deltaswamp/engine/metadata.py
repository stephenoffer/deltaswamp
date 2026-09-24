"""Metadata-only commits, written by this library itself.

Most of `ALTER TABLE` never touches a data file. Renaming or dropping a column
under column mapping, widening a type, flipping nullability, changing the
clustering keys, setting a property, writing a comment: each is one commit that
replaces the `metaData` action (and sometimes the `protocol`, or a
`domainMetadata` action) and nothing else. Yet delta-rs implements a handful of
these, rejects most of the property surface on ALTER, and cannot touch a table
that carries `domainMetadata` at all -- which is every liquid-clustered table.
Databricks treats the rest as its own.

This module computes those commits as pure functions over the table's current
protocol and metadata, which is what makes each rule testable on its own. The
kernel engine reads the state through the native snapshot and writes the result
with a put-if-absent of the next log file, so a concurrent writer makes the
commit fail rather than be overwritten, and the change is recomputed against
the new state.

Every refusal here is a rule from the Delta protocol or from what Spark
enforces, stated as the reason. Where a change needs the data checked first
(SET NOT NULL), the caller supplies the check; where it needs data rewritten
(enabling row tracking, which needs a backfill), it is refused.
"""

from __future__ import annotations

import copy
import json
import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..capability import (
    FEATURE_DEPENDENCIES,
    FEATURE_SUPPORT,
    FeatureKind,
    TableFeature,
    feature_from_wire,
)
from ..errors import UnreachableTableError

__all__ = [
    "Change",
    "TableState",
    "add_columns",
    "add_feature",
    "alter_column_type",
    "arrow_to_delta_schema",
    "build_actions",
    "cluster_by",
    "drop_column",
    "drop_constraint",
    "rename_column",
    "set_column_comment",
    "set_comment",
    "set_nullability",
    "set_properties",
    "unset_properties",
]

CLUSTERING_DOMAIN = "delta.clustering"
_CM_ID = "delta.columnMapping.id"
_CM_PHYSICAL = "delta.columnMapping.physicalName"
_CM_MAX = "delta.columnMapping.maxColumnId"
_CM_MODE = "delta.columnMapping.mode"
_TYPE_CHANGES = "delta.typeChanges"
_GENERATION = "delta.generationExpression"


@dataclass(frozen=True)
class TableState:
    """What a metadata change is computed against: one snapshot's P&M."""

    version: int
    protocol: dict[str, Any]
    metadata: dict[str, Any]
    #: The commit timestamp of `version` (its in-commit timestamp when enabled).
    timestamp: int | None = None
    #: `delta.clustering` domain configuration, when the table has one.
    clustering: dict[str, Any] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        parsed: dict[str, Any] = json.loads(self.metadata["schemaString"])
        return parsed

    @property
    def configuration(self) -> dict[str, str]:
        return dict(self.metadata.get("configuration") or {})

    @property
    def column_mapping_mode(self) -> str:
        return self.configuration.get(_CM_MODE, "none").lower()


@dataclass
class Change:
    """The result of a metadata operation, ready to render as log actions."""

    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    domains: list[dict[str, Any]] = field(default_factory=list)


def _refuse(operation: str, reason: str, remedy: str | None = None) -> UnreachableTableError:
    return UnreachableTableError(operation, reason, remedy)


# ------------------------------------------------------------------ protocol

#: Features implied by a legacy (pre-table-features) writer version.
_LEGACY_WRITER: dict[int, tuple[str, ...]] = {
    1: (),
    2: ("appendOnly", "invariants"),
    3: ("appendOnly", "invariants", "checkConstraints"),
    4: ("appendOnly", "invariants", "checkConstraints", "changeDataFeed", "generatedColumns"),
    5: (
        "appendOnly",
        "invariants",
        "checkConstraints",
        "changeDataFeed",
        "generatedColumns",
        "columnMapping",
    ),
    6: (
        "appendOnly",
        "invariants",
        "checkConstraints",
        "changeDataFeed",
        "generatedColumns",
        "columnMapping",
        "identityColumns",
    ),
}
_LEGACY_READER: dict[int, tuple[str, ...]] = {1: (), 2: ("columnMapping",)}


def supported_features(protocol: Mapping[str, Any]) -> set[str]:
    """Every feature the protocol supports, explicit or implied by version."""
    writer = int(protocol.get("minWriterVersion", 1))
    if writer >= 7:
        return set(protocol.get("writerFeatures") or [])
    return set(_LEGACY_WRITER.get(writer, ()))


def with_features(protocol: Mapping[str, Any], names: Iterable[str]) -> dict[str, Any] | None:
    """The protocol extended to support `names`, or None if it already does.

    Upgrades a legacy protocol to table features when needed, carrying over
    every feature its old version implied -- dropping one would silently make
    an existing table invalid. Dependencies are added too.
    """
    wanted: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in wanted:
            continue
        wanted.add(name)
        feature = feature_from_wire(name)
        if feature is not None:
            pending.extend(dep.value for dep in FEATURE_DEPENDENCIES.get(feature, ()))

    have = supported_features(protocol)
    if wanted <= have:
        return None

    reader = int(protocol.get("minReaderVersion", 1))
    writer = int(protocol.get("minWriterVersion", 1))
    writer_features = set(have) | wanted
    needs_reader = {
        n
        for n in writer_features
        if (f := feature_from_wire(n)) is not None
        and FEATURE_SUPPORT[f].kind is FeatureKind.READER_WRITER
    }

    out: dict[str, Any] = {
        "minReaderVersion": reader,
        "minWriterVersion": 7,
        "writerFeatures": sorted(writer_features),
    }
    new_reader_features = needs_reader & wanted
    if reader >= 3:
        out["readerFeatures"] = sorted(set(protocol.get("readerFeatures") or []) | needs_reader)
    elif new_reader_features - set(_LEGACY_READER.get(reader, ())):
        out["minReaderVersion"] = 3
        out["readerFeatures"] = sorted(needs_reader)
    del writer  # legacy writer version is fully described by writer_features now
    return out


# -------------------------------------------------------------- schema walking


def _fields(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = schema["fields"]
    return fields


def _find(schema: dict[str, Any], path: list[str]) -> tuple[list[dict[str, Any]], int]:
    """(containing field list, index) for a column path; case-insensitive like Spark."""
    container = _fields(schema)
    for depth, name in enumerate(path):
        index = next(
            (i for i, f in enumerate(container) if f["name"].lower() == name.lower()), None
        )
        if index is None:
            raise _refuse(
                f"find column {'.'.join(path)}",
                f"the table has no column {'.'.join(path[: depth + 1])!r}",
            )
        if depth == len(path) - 1:
            return container, index
        child = container[index]["type"]
        if not (isinstance(child, dict) and child.get("type") == "struct"):
            raise _refuse(
                f"find column {'.'.join(path)}",
                f"{'.'.join(path[: depth + 1])!r} is not a struct, so it has no nested fields",
            )
        container = _fields(child)
    raise _refuse("find column", "an empty column path")  # pragma: no cover


def _split(column: str) -> list[str]:
    from ..identity import split_identifier

    return split_identifier(column)


def _walk(datatype: Any) -> Iterable[dict[str, Any]]:
    """Every struct field nested anywhere inside a type, depth-first."""
    if not isinstance(datatype, dict):
        return
    kind = datatype.get("type")
    if kind == "struct":
        for f in datatype["fields"]:
            yield f
            yield from _walk(f["type"])
    elif kind == "array":
        yield from _walk(datatype["elementType"])
    elif kind == "map":
        yield from _walk(datatype["keyType"])
        yield from _walk(datatype["valueType"])


def _physical_name(f: Mapping[str, Any]) -> str:
    return str((f.get("metadata") or {}).get(_CM_PHYSICAL, f["name"]))


def _references(expression: str, name: str) -> bool:
    """Whether a SQL expression mentions `name` as an identifier. Conservative."""
    pattern = r"(?<![A-Za-z0-9_`])`?" + re.escape(name) + r"`?(?![A-Za-z0-9_])"
    return re.search(pattern, expression, flags=re.IGNORECASE) is not None


def _dependents(state: TableState, name: str) -> list[str]:
    """Constraints, generated columns and stats settings that mention `name`."""
    found: list[str] = []
    for key, value in state.configuration.items():
        if key.startswith("delta.constraints.") and _references(value, name):
            found.append(f"CHECK constraint {key.removeprefix('delta.constraints.')}")
        if key == "delta.dataSkippingStatsColumns" and _references(value, name):
            found.append("delta.dataSkippingStatsColumns")
    for f in _walk(state.schema):
        expression = (f.get("metadata") or {}).get(_GENERATION)
        if isinstance(expression, str) and _references(expression, name):
            found.append(f"generated column {f['name']}")
    return found


def _clustering_columns(state: TableState) -> list[list[str]]:
    if not state.clustering:
        return []
    columns: list[list[str]] = state.clustering.get("clusteringColumns") or []
    return columns


def _new_metadata(state: TableState, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = copy.deepcopy(state.metadata)
    if schema is not None:
        metadata["schemaString"] = json.dumps(schema, separators=(",", ":"))
    return metadata


# ------------------------------------------------------------ arrow -> delta


def arrow_to_delta_type(arrow_type: Any) -> Any:
    """Map an Arrow type to its Delta schema JSON form."""
    import pyarrow as pa
    import pyarrow.types as t

    if t.is_boolean(arrow_type):
        return "boolean"
    if t.is_int8(arrow_type):
        return "byte"
    if t.is_int16(arrow_type):
        return "short"
    if t.is_int32(arrow_type):
        return "integer"
    if t.is_int64(arrow_type):
        return "long"
    if t.is_float32(arrow_type):
        return "float"
    if t.is_float64(arrow_type):
        return "double"
    if t.is_string(arrow_type) or t.is_large_string(arrow_type) or t.is_string_view(arrow_type):
        return "string"
    if t.is_binary(arrow_type) or t.is_large_binary(arrow_type) or t.is_binary_view(arrow_type):
        return "binary"
    if t.is_date(arrow_type):
        return "date"
    if t.is_timestamp(arrow_type):
        return "timestamp" if arrow_type.tz is not None else "timestamp_ntz"
    if t.is_decimal(arrow_type):
        return f"decimal({arrow_type.precision},{arrow_type.scale})"
    if t.is_struct(arrow_type):
        return {
            "type": "struct",
            "fields": [
                arrow_to_delta_field(arrow_type.field(i)) for i in range(arrow_type.num_fields)
            ],
        }
    if t.is_list(arrow_type) or t.is_large_list(arrow_type):
        return {
            "type": "array",
            "elementType": arrow_to_delta_type(arrow_type.value_type),
            "containsNull": arrow_type.value_field.nullable,
        }
    if t.is_map(arrow_type):
        return {
            "type": "map",
            "keyType": arrow_to_delta_type(arrow_type.key_type),
            "valueType": arrow_to_delta_type(arrow_type.item_type),
            "valueContainsNull": arrow_type.item_field.nullable,
        }
    raise _refuse(
        "convert an Arrow type to a Delta type",
        f"{arrow_type} has no Delta equivalent",
    ) from pa.ArrowNotImplementedError(str(arrow_type))


def arrow_to_delta_field(arrow_field: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in (arrow_field.metadata or {}).items():
        k = key.decode() if isinstance(key, bytes) else key
        v = value.decode() if isinstance(value, bytes) else value
        metadata[k] = v
    return {
        "name": arrow_field.name,
        "type": arrow_to_delta_type(arrow_field.type),
        "nullable": arrow_field.nullable,
        "metadata": metadata,
    }


def arrow_to_delta_schema(schema: Any) -> dict[str, Any]:
    return {"type": "struct", "fields": [arrow_to_delta_field(f) for f in schema]}


# ------------------------------------------------------------- column mapping


def _assign_ids(
    fields: list[dict[str, Any]], next_id: int, *, physical: Callable[[dict[str, Any]], str]
) -> int:
    """Give every field (recursively) a column-mapping id and physical name."""
    for f in fields:
        meta = f.setdefault("metadata", {})
        meta[_CM_ID] = next_id
        meta[_CM_PHYSICAL] = physical(f)
        next_id += 1
        for nested in _walk(f["type"]):
            nmeta = nested.setdefault("metadata", {})
            nmeta[_CM_ID] = next_id
            nmeta[_CM_PHYSICAL] = physical(nested)
            next_id += 1
    return next_id


def _require_column_mapping(state: TableState, operation: str) -> None:
    if state.column_mapping_mode not in ("name", "id"):
        raise _refuse(
            operation,
            "renaming or dropping a column without rewriting data needs column mapping; "
            "without it the column's name is also its name in every Parquet file",
            "set_properties({'delta.columnMapping.mode': 'name'}) first",
        )


# ------------------------------------------------------------------ operations


def set_comment(state: TableState, comment: str | None) -> Change:
    metadata = _new_metadata(state)
    if comment:
        metadata["description"] = comment
    else:
        metadata.pop("description", None)
    return Change("SET TBLPROPERTIES", {"comment": comment or ""}, metadata=metadata)


def set_column_comment(state: TableState, column: str, comment: str | None) -> Change:
    schema = state.schema
    container, index = _find(schema, _split(column))
    meta = container[index].setdefault("metadata", {})
    if comment:
        meta["comment"] = comment
    else:
        meta.pop("comment", None)
    return Change(
        "CHANGE COLUMN",
        {"column": column, "comment": comment or ""},
        metadata=_new_metadata(state, schema),
    )


def set_nullability(state: TableState, column: str, nullable: bool) -> Change:
    """SET/DROP NOT NULL. The caller must have checked for nulls before SET."""
    schema = state.schema
    path = _split(column)
    container, index = _find(schema, path)
    if not nullable and len(path) > 1:
        raise _refuse(
            f"SET NOT NULL on nested column {column}",
            "NOT NULL on a nested field is only enforceable when every enclosing struct is "
            "non-null too, which this path does not verify",
            "use the SQL fallback",
        )
    container[index]["nullable"] = nullable
    operation = "CHANGE COLUMN"
    return Change(
        operation,
        {"column": column, "nullable": str(nullable).lower()},
        metadata=_new_metadata(state, schema),
    )


def add_columns(state: TableState, new_fields: list[dict[str, Any]]) -> Change:
    """ADD COLUMNS. `new_fields` are Delta schema field dicts."""
    schema = state.schema
    existing = {f["name"].lower() for f in _fields(schema)}
    configuration = state.configuration
    for f in new_fields:
        if f["name"].lower() in existing:
            raise _refuse(f"add column {f['name']}", "a column with that name already exists")
        if not f.get("nullable", True):
            raise _refuse(
                f"add NOT NULL column {f['name']}",
                "existing rows have no value for a new column, so it must be nullable",
            )
        existing.add(f["name"].lower())

    added = copy.deepcopy(new_fields)
    if state.column_mapping_mode in ("name", "id"):
        next_id = int(configuration.get(_CM_MAX, "0")) + 1
        next_id = _assign_ids(added, next_id, physical=lambda _f: f"col-{uuid.uuid4()}")
        configuration[_CM_MAX] = str(next_id - 1)

    schema["fields"] = _fields(schema) + added
    metadata = _new_metadata(state, schema)
    metadata["configuration"] = configuration
    return Change(
        "ADD COLUMNS",
        {"columns": json.dumps([f["name"] for f in added])},
        metadata=metadata,
    )


def drop_column(state: TableState, column: str) -> Change:
    _require_column_mapping(state, f"drop column {column}")
    schema = state.schema
    path = _split(column)
    container, index = _find(schema, path)
    target = container[index]
    name = target["name"]

    if len(path) == 1 and name.lower() in {
        p.lower() for p in state.metadata.get("partitionColumns", [])
    }:
        raise _refuse(f"drop column {column}", "it is a partition column")
    if len(path) == 1 and len(container) == 1:
        raise _refuse(f"drop column {column}", "a table must keep at least one column")
    physical = [_physical_name(target)]
    if physical in _clustering_columns(state) or any(
        c[: len(physical)] == physical for c in _clustering_columns(state)
    ):
        raise _refuse(
            f"drop column {column}",
            "it is a clustering column",
            "change the clustering keys with cluster_by() first",
        )
    dependents = _dependents(state, name)
    if dependents:
        raise _refuse(
            f"drop column {column}",
            f"it is referenced by {', '.join(dependents)}",
            "drop those first",
        )
    del container[index]
    return Change(
        "DROP COLUMNS",
        {"columns": json.dumps([column])},
        metadata=_new_metadata(state, schema),
    )


def rename_column(state: TableState, old: str, new: str) -> Change:
    _require_column_mapping(state, f"rename column {old}")
    schema = state.schema
    path = _split(old)
    container, index = _find(schema, path)
    current = container[index]["name"]
    if any(f["name"].lower() == new.lower() for f in container if f is not container[index]):
        raise _refuse(f"rename column {old} to {new}", f"a column named {new!r} already exists")
    dependents = _dependents(state, current)
    if dependents:
        raise _refuse(
            f"rename column {old}",
            f"it is referenced by {', '.join(dependents)}, whose expressions name it",
            "drop and re-create those after the rename",
        )
    container[index]["name"] = new
    metadata = _new_metadata(state, schema)
    if len(path) == 1:
        metadata["partitionColumns"] = [
            new if p.lower() == current.lower() else p
            for p in state.metadata.get("partitionColumns", [])
        ]
    return Change(
        "RENAME COLUMN",
        {"oldColumnPath": old, "newColumnPath": new},
        metadata=metadata,
    )


# Widenings the typeWidening feature allows (Delta protocol, "Type Widening").
_INTEGRAL = ("byte", "short", "integer", "long")
_INTEGRAL_DIGITS = {"byte": 3, "short": 5, "integer": 10, "long": 20}
_DECIMAL = re.compile(r"decimal\((\d+),\s*(\d+)\)")


def _widening_allowed(old: str, new: str) -> bool:
    if old == new:
        return False
    if old in _INTEGRAL and new in _INTEGRAL:
        return _INTEGRAL.index(new) > _INTEGRAL.index(old)
    if old == "float" and new == "double":
        return True
    if old in ("byte", "short", "integer") and new == "double":
        return True
    if old == "date" and new == "timestamp_ntz":
        return True
    new_dec = _DECIMAL.fullmatch(new)
    if new_dec is None:
        return False
    precision, scale = int(new_dec.group(1)), int(new_dec.group(2))
    old_dec = _DECIMAL.fullmatch(old)
    if old_dec is not None:
        p0, s0 = int(old_dec.group(1)), int(old_dec.group(2))
        return precision >= p0 and scale >= s0 and (precision - scale) >= (p0 - s0)
    if old in _INTEGRAL:
        return precision - scale >= _INTEGRAL_DIGITS[old]
    return False


def alter_column_type(state: TableState, column: str, new_type: str) -> Change:
    """Widen a column's type without rewriting data (the typeWidening feature)."""
    config = state.configuration
    if config.get("delta.enableTypeWidening", "false").lower() != "true":
        raise _refuse(
            f"change the type of {column}",
            "changing a type without rewriting data needs the typeWidening feature enabled",
            "set_properties({'delta.enableTypeWidening': 'true'}) first",
        )
    schema = state.schema
    path = _split(column)
    container, index = _find(schema, path)
    target = container[index]
    old_type = target["type"]
    new_type = new_type.strip().lower().replace(" ", "")
    new_type = {"int": "integer", "bigint": "long", "smallint": "short", "tinyint": "byte"}.get(
        new_type, new_type
    )
    if not isinstance(old_type, str) or not _widening_allowed(old_type, new_type):
        raise _refuse(
            f"change the type of {column} from {old_type} to {new_type}",
            "only widening changes are allowed without a rewrite: byte->short->int->long, "
            "float->double, byte/short/int->double, date->timestamp_ntz, and decimals "
            "whose precision and scale do not shrink",
        )
    if len(path) == 1 and target["name"].lower() in {
        p.lower() for p in state.metadata.get("partitionColumns", [])
    }:
        raise _refuse(f"change the type of {column}", "it is a partition column")
    target["type"] = new_type
    meta = target.setdefault("metadata", {})
    history = list(meta.get(_TYPE_CHANGES) or [])
    history.append({"fromType": old_type, "toType": new_type})
    meta[_TYPE_CHANGES] = history
    return Change(
        "CHANGE COLUMN",
        {"column": column, "type": new_type},
        metadata=_new_metadata(state, schema),
    )


def drop_constraint(state: TableState, name: str, *, if_exists: bool = False) -> Change:
    key = f"delta.constraints.{name.lower()}"
    configuration = state.configuration
    match = next((k for k in configuration if k.lower() == key), None)
    if match is None:
        if if_exists:
            return Change("DROP CONSTRAINT", {"name": name})
        raise _refuse(f"drop constraint {name}", "the table has no such constraint")
    del configuration[match]
    metadata = _new_metadata(state)
    metadata["configuration"] = configuration
    return Change("DROP CONSTRAINT", {"name": name}, metadata=metadata)


def cluster_by(state: TableState, columns: list[str] | None) -> Change:
    """Set liquid-clustering keys. Takes effect for data written afterwards."""
    if state.metadata.get("partitionColumns"):
        raise _refuse(
            "cluster a partitioned table",
            "a table is either partitioned or clustered, not both",
        )
    schema = state.schema
    physical: list[list[str]] = []
    for column in columns or []:
        path = _split(column)
        names: list[str] = []
        container = _fields(schema)
        for depth, part in enumerate(path):
            found = next((f for f in container if f["name"].lower() == part.lower()), None)
            if found is None:
                raise _refuse(f"cluster by {column}", f"the table has no column {column!r}")
            names.append(_physical_name(found))
            if depth < len(path) - 1:
                if not (isinstance(found["type"], dict) and found["type"].get("type") == "struct"):
                    raise _refuse(f"cluster by {column}", f"{part!r} is not a struct")
                container = _fields(found["type"])
            elif isinstance(found["type"], dict):
                raise _refuse(
                    f"cluster by {column}",
                    "clustering columns must be of a primitive type, since clustering uses "
                    "their min/max statistics",
                )
        physical.append(names)

    protocol = with_features(state.protocol, ["clustering", "domainMetadata"])
    domain = {
        "domainMetadata": {
            "domain": CLUSTERING_DOMAIN,
            "configuration": json.dumps({"clusteringColumns": physical}),
            "removed": False,
        }
    }
    return Change(
        "CLUSTER BY",
        {"newClusteringColumns": ",".join(columns or [])},
        protocol=protocol,
        domains=[domain],
    )


#: Features a bare `delta.feature.<name> = supported` may add to an existing
#: table here. Everything else needs work beyond a metadata commit.
_ADDABLE_FEATURES = frozenset(
    {
        "appendOnly",
        "invariants",
        "checkConstraints",
        "changeDataFeed",
        "deletionVectors",
        "domainMetadata",
        "timestampNtz",
        "typeWidening",
        "v2Checkpoint",
        "vacuumProtocolCheck",
        "variantType",
        "inCommitTimestamp",
        "allowColumnDefaults",
    }
)

#: Features that need more than a metadata commit to turn on for existing data.
_NOT_ADDABLE: dict[str, str] = {
    "rowTracking": "row tracking needs every existing file backfilled with row ids",
    "catalogManaged": "a table becomes catalog-managed only when a catalog creates it",
    "clustering": "use cluster_by(), which writes the clustering domain too",
    "icebergCompatV1": "UniForm needs Iceberg metadata generated alongside, which only "
    "Databricks does",
    "icebergCompatV2": "UniForm needs Iceberg metadata generated alongside, which only "
    "Databricks does",
    "icebergCompatV3": "UniForm needs Iceberg metadata generated alongside, which only "
    "Databricks does",
    "identityColumns": "identity columns are declared at CREATE",
    "generatedColumns": "generated columns are declared at CREATE",
}

#: Properties whose true value turns on a feature.
_ENABLING: dict[str, str] = {
    "delta.appendOnly": "appendOnly",
    "delta.enableChangeDataFeed": "changeDataFeed",
    "delta.enableDeletionVectors": "deletionVectors",
    "delta.enableTypeWidening": "typeWidening",
    "delta.enableInCommitTimestamps": "inCommitTimestamp",
}

#: Keys the kernel recognises; anything else under `delta.` is refused.
_KNOWN_KEYS = frozenset(
    {
        "delta.appendOnly",
        "delta.autoOptimize.autoCompact",
        "delta.autoOptimize.optimizeWrite",
        "delta.checkpoint.writeStatsAsJson",
        "delta.checkpoint.writeStatsAsStruct",
        "delta.checkpointInterval",
        "delta.checkpointPolicy",
        "delta.columnMapping.mode",
        "delta.dataSkippingNumIndexedCols",
        "delta.dataSkippingStatsColumns",
        "delta.deletedFileRetentionDuration",
        "delta.enableChangeDataFeed",
        "delta.enableDeletionVectors",
        "delta.enableExpiredLogCleanup",
        "delta.enableInCommitTimestamps",
        "delta.enableTypeWidening",
        "delta.isolationLevel",
        "delta.logRetentionDuration",
        "delta.parquet.compression.codec",
        "delta.randomPrefixLength",
        "delta.randomizeFilePrefixes",
        "delta.setTransactionRetentionDuration",
        "delta.targetFileSize",
        "delta.tuneFileSizesForRewrites",
    }
)

_BOOLEAN_KEYS = frozenset(
    {
        "delta.appendOnly",
        "delta.autoOptimize.autoCompact",
        "delta.autoOptimize.optimizeWrite",
        "delta.checkpoint.writeStatsAsJson",
        "delta.checkpoint.writeStatsAsStruct",
        "delta.enableChangeDataFeed",
        "delta.enableDeletionVectors",
        "delta.enableExpiredLogCleanup",
        "delta.enableInCommitTimestamps",
        "delta.enableTypeWidening",
        "delta.randomizeFilePrefixes",
        "delta.tuneFileSizesForRewrites",
    }
)


def set_properties(state: TableState, properties: Mapping[str, str]) -> Change:
    """SET TBLPROPERTIES, including the protocol changes the values imply."""
    configuration = state.configuration
    features: set[str] = set()
    schema: dict[str, Any] | None = None
    operation = f"set table properties {sorted(properties)}"

    for key, raw in properties.items():
        value = str(raw)
        lowered = value.lower()
        if key in ("delta.minReaderVersion", "delta.minWriterVersion"):
            raise _refuse(
                operation,
                f"{key} is not set directly; enabling a feature raises the protocol as needed",
                "set the feature instead, e.g. delta.feature.deletionVectors = supported",
            )
        if key.startswith("delta.constraints."):
            raise _refuse(operation, "constraints are added through add_constraint()")
        if key.startswith("delta.feature."):
            name = key.removeprefix("delta.feature.")
            if lowered not in ("supported", "enabled"):
                raise _refuse(operation, f"{key} only accepts 'supported'")
            if name in _NOT_ADDABLE:
                raise _refuse(
                    operation, f"cannot add {name} to an existing table: {_NOT_ADDABLE[name]}"
                )
            if name not in _ADDABLE_FEATURES:
                raise _refuse(operation, f"{name} is not a feature this path can add")
            features.add(name)
            continue
        if key in ("delta.enableRowTracking",) and lowered == "true":
            raise _refuse(
                operation,
                "enabling row tracking on an existing table needs every file backfilled "
                "with row ids, which is a data operation",
                "use the SQL fallback, or create the table with row tracking",
            )
        if key.startswith(("delta.enableIcebergCompat", "delta.universalFormat")):
            raise _refuse(operation, _NOT_ADDABLE["icebergCompatV3"], "use the SQL fallback")
        if key.startswith("delta.") and key not in _KNOWN_KEYS:
            raise _refuse(operation, f"{key} is not a Delta table property this path knows")
        if key in _BOOLEAN_KEYS and lowered not in ("true", "false"):
            raise _refuse(operation, f"{key} must be 'true' or 'false', not {value!r}")

        if key in _ENABLING and lowered == "true":
            features.add(_ENABLING[key])
        if key == "delta.checkpointPolicy":
            if lowered not in ("classic", "v2"):
                raise _refuse(operation, "delta.checkpointPolicy is 'classic' or 'v2'")
            if lowered == "v2":
                features.add("v2Checkpoint")
        if key == _CM_MODE:
            schema = _enable_column_mapping(state, lowered, configuration, operation)
            if lowered != "none":
                features.add("columnMapping")
        if (
            key == "delta.enableInCommitTimestamps"
            and lowered == "true"
            and (state.configuration.get(key, "false").lower() != "true")
        ):
            # Readers use these to know from which version timestamps are
            # in-commit; the timestamp is filled in when the commit is built.
            configuration["delta.inCommitTimestampEnablementVersion"] = str(state.version + 1)
            configuration["delta.inCommitTimestampEnablementTimestamp"] = "__ICT__"
        configuration[key] = value

    metadata = _new_metadata(state, schema)
    metadata["configuration"] = configuration
    return Change(
        "SET TBLPROPERTIES",
        {"properties": json.dumps(dict(properties), sort_keys=True)},
        protocol=with_features(state.protocol, features) if features else None,
        metadata=metadata,
    )


def _enable_column_mapping(
    state: TableState, mode: str, configuration: dict[str, str], operation: str
) -> dict[str, Any] | None:
    current = state.column_mapping_mode
    if mode == current:
        return None
    if mode not in ("none", "name", "id"):
        raise _refuse(operation, f"delta.columnMapping.mode is none, name or id, not {mode!r}")
    if current != "none" or mode != "name":
        raise _refuse(
            operation,
            f"changing column mapping from {current} to {mode} is not a metadata-only change; "
            "only none -> name is (the existing Parquet column names become the physical names)",
        )
    schema = state.schema
    last = _assign_ids(_fields(schema), 1, physical=lambda f: str(f["name"]))
    configuration[_CM_MAX] = str(last - 1)
    return schema


def unset_properties(state: TableState, keys: Iterable[str], *, if_exists: bool = True) -> Change:
    configuration = state.configuration
    for key in keys:
        if key.startswith("delta.feature.") or key in (_CM_MODE, _CM_MAX):
            raise _refuse(
                f"unset {key}",
                "removing it would change the protocol or orphan column-mapped data",
                "drop the feature through the SQL fallback",
            )
        if key.startswith("delta.inCommitTimestampEnablement"):
            raise _refuse(f"unset {key}", "readers need it to interpret commit timestamps")
        if key not in configuration:
            if if_exists:
                continue
            raise _refuse(f"unset {key}", "the table has no such property")
        del configuration[key]
    metadata = _new_metadata(state)
    metadata["configuration"] = configuration
    return Change(
        "UNSET TBLPROPERTIES", {"properties": json.dumps(sorted(keys))}, metadata=metadata
    )


def add_feature(state: TableState, feature: str | TableFeature) -> Change:
    name = feature.value if isinstance(feature, TableFeature) else str(feature)
    return set_properties(state, {f"delta.feature.{name}": "supported"})


# ------------------------------------------------------------------ rendering


def build_actions(
    state: TableState,
    change: Change,
    *,
    engine_info: str,
    now_ms: int | None = None,
) -> list[str]:
    """Render a change as the newline-delimited actions of one commit.

    `commitInfo` comes first, which the in-commit-timestamp feature requires,
    and carries a timestamp strictly after the previous commit's when that
    feature is on -- in-commit timestamps must be monotonic.
    """
    now = int(time.time() * 1000) if now_ms is None else now_ms
    new_config = (change.metadata or state.metadata).get("configuration") or {}
    ict_on = str(new_config.get("delta.enableInCommitTimestamps", "false")).lower() == "true"
    commit_info: dict[str, Any] = {
        "timestamp": now,
        "operation": change.operation,
        "operationParameters": {k: str(v) for k, v in change.parameters.items()},
        "engineInfo": engine_info,
        "isBlindAppend": False,
        "txnId": str(uuid.uuid4()),
    }
    ict = max(now, (state.timestamp or 0) + 1) if ict_on else None
    if ict is not None:
        commit_info = {"inCommitTimestamp": ict, **commit_info}
        commit_info["timestamp"] = ict

    metadata = change.metadata
    if metadata is not None:
        configuration = metadata.get("configuration") or {}
        if configuration.get("delta.inCommitTimestampEnablementTimestamp") == "__ICT__":
            configuration["delta.inCommitTimestampEnablementTimestamp"] = str(ict or now)

    actions: list[dict[str, Any]] = [{"commitInfo": commit_info}]
    if change.protocol is not None:
        actions.append({"protocol": change.protocol})
    if metadata is not None:
        actions.append({"metaData": metadata})
    actions.extend(change.domains)
    return [json.dumps(a, separators=(",", ":")) for a in actions]


# ------------------------------------------------------------- version zero


def _wire(protocol: Mapping[str, Any], *names: str) -> Any:
    """Read a protocol field in either kebab-case (UC wire) or camelCase."""
    for name in names:
        if name in protocol:
            return protocol[name]
    return None


def initial_actions(
    *,
    table_id: str,
    schema: dict[str, Any],
    required_protocol: Mapping[str, Any],
    configuration: Mapping[str, str],
    partition_columns: list[str] | None = None,
    cluster_by: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    engine_info: str,
    now_ms: int | None = None,
) -> list[str]:
    """Version 0 of a catalog-managed table, as the catalog requires it.

    The catalog allocates the table id, and version 0's ``metaData.id`` must
    equal it -- the kernel's own create path draws a random id, which is why
    this commit is composed here. Features come from three places: the
    catalog's required protocol, the ``delta.feature.*`` keys it lists among
    required properties, and whatever the caller's properties imply.
    """
    config = {k: v for k, v in configuration.items() if not k.startswith("delta.feature.")}
    features: set[str] = set(_wire(required_protocol, "writer-features", "writerFeatures") or [])
    features |= set(_wire(required_protocol, "reader-features", "readerFeatures") or [])
    features |= {
        k.removeprefix("delta.feature.") for k in configuration if k.startswith("delta.feature.")
    }
    for key, feature in _ENABLING.items():
        if str(config.get(key, "")).lower() == "true":
            features.add(feature)
    if str(config.get("delta.checkpointPolicy", "")).lower() == "v2":
        features.add("v2Checkpoint")
    if str(config.get("delta.enableRowTracking", "")).lower() == "true":
        features.add("rowTracking")

    schema = copy.deepcopy(schema)
    if config.get(_CM_MODE, "none").lower() in ("name", "id"):
        features.add("columnMapping")
        last = _assign_ids(_fields(schema), 1, physical=lambda _f: f"col-{uuid.uuid4()}")
        config[_CM_MAX] = str(last - 1)

    domains: list[dict[str, Any]] = []
    if cluster_by:
        if partition_columns:
            raise _refuse("create the table", "a table is either partitioned or clustered")
        features |= {"clustering", "domainMetadata"}
        state = TableState(
            version=-1,
            protocol={"minReaderVersion": 1, "minWriterVersion": 1},
            metadata={"schemaString": json.dumps(schema)},
        )
        domains = cluster_by_domains(state, cluster_by)

    upgraded = with_features({"minReaderVersion": 1, "minWriterVersion": 1}, features) or {}
    reader_features = sorted(
        n
        for n in upgraded.get("writerFeatures", [])
        if (f := feature_from_wire(n)) is not None
        and FEATURE_SUPPORT[f].kind is FeatureKind.READER_WRITER
    )
    protocol = {
        "minReaderVersion": max(
            3, int(_wire(required_protocol, "min-reader-version", "minReaderVersion") or 3)
        ),
        "minWriterVersion": 7,
        "readerFeatures": reader_features,
        "writerFeatures": sorted(upgraded.get("writerFeatures", [])),
    }

    now = int(time.time() * 1000) if now_ms is None else now_ms
    commit_info: dict[str, Any] = {
        "timestamp": now,
        "operation": "CREATE TABLE",
        "operationParameters": {
            "partitionBy": json.dumps(partition_columns or []),
            "clusterBy": json.dumps(cluster_by or []),
        },
        "engineInfo": engine_info,
        "isBlindAppend": True,
        "txnId": str(uuid.uuid4()),
    }
    if "inCommitTimestamp" in features:
        commit_info = {"inCommitTimestamp": now, **commit_info}
    metadata: dict[str, Any] = {
        "id": table_id,
        "name": name,
        "description": description,
        "format": {"provider": "parquet", "options": {}},
        "schemaString": json.dumps(schema, separators=(",", ":")),
        "partitionColumns": list(partition_columns or []),
        "configuration": dict(config),
        "createdTime": now,
    }
    actions: list[dict[str, Any]] = [
        {"commitInfo": commit_info},
        {"protocol": protocol},
        {"metaData": metadata},
        *domains,
    ]
    return [json.dumps(a, separators=(",", ":")) for a in actions]


def cluster_by_domains(state: TableState, columns: list[str]) -> list[dict[str, Any]]:
    """Just the domain actions of `cluster_by`, for a table being created."""
    change = cluster_by(state, columns)
    return change.domains
