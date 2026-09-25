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

#: Stable feature name -> the preview name that provides the same capability.
_PREVIEW_OF: dict[str, str] = {
    "typeWidening": "typeWidening-preview",
    "variantType": "variantType-preview",
    "variantShredding": "variantShredding-preview",
    "collations": "collations-preview",
}

#: Features a table must support to hold a column of the given Delta type.
_TYPE_FEATURES: dict[str, str] = {
    "timestamp_ntz": "timestampNtz",
    "variant": "variantType",
}

#: Characters Parquet column names cannot carry. Without column mapping the
#: logical name is the Parquet name, so Spark refuses them there.
_INVALID_NAME_CHARS = frozenset(" ,;{}()\n\t=")

#: Columns the change data feed adds to every change row.
_CDF_RESERVED = frozenset({"_change_type", "_commit_version", "_commit_timestamp"})

#: Features whose presence means the catalog, not the log, arbitrates commits.
_CATALOG_FEATURES = frozenset({"catalogManaged", "catalogOwned-preview"})


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
    # A table already carrying the preview variant of a feature supports it:
    # adding the stable name too would lock out writers that know only the
    # preview, for no change in behaviour.
    wanted = {n for n in wanted if _PREVIEW_OF.get(n) not in have}
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


def _if_changed(state: TableState, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """`metadata`, or None when it says what the current one does (nothing to commit)."""

    def canonical(m: Mapping[str, Any]) -> dict[str, Any]:
        # The kernel serialises an absent name/description as null.
        out = {k: v for k, v in m.items() if k != "schemaString" and v is not None}
        out["configuration"] = dict(m.get("configuration") or {})
        raw = m.get("schemaString")
        out["schemaString"] = json.loads(raw) if isinstance(raw, str) else raw
        return out

    return None if canonical(metadata) == canonical(state.metadata) else metadata


def _type_features(datatype: Any) -> set[str]:
    """Table features a column of this Delta type needs, anywhere inside it."""
    if isinstance(datatype, str):
        feature = _TYPE_FEATURES.get(datatype.lower())
        return {feature} if feature else set()
    if not isinstance(datatype, dict):
        return set()
    kind = datatype.get("type")
    found: set[str] = set()
    if kind == "struct":
        for f in datatype.get("fields") or []:
            found |= _type_features(f.get("type"))
    elif kind == "array":
        found |= _type_features(datatype.get("elementType"))
    elif kind == "map":
        found |= _type_features(datatype.get("keyType"))
        found |= _type_features(datatype.get("valueType"))
    return found


#: Primitive type names a Delta schema may carry (the protocol's spelling).
_PRIMITIVE_TYPES = frozenset(
    {
        "string",
        "long",
        "integer",
        "short",
        "byte",
        "float",
        "double",
        "boolean",
        "binary",
        "date",
        "timestamp",
        "timestamp_ntz",
        "variant",
    }
)
#: SQL / Arrow spellings users reach for, and the Delta name each means.
_SPELLING_HINTS = {
    "int": "integer",
    "int32": "integer",
    "bigint": "long",
    "int64": "long",
    "smallint": "short",
    "int16": "short",
    "tinyint": "byte",
    "int8": "byte",
    "bool": "boolean",
    "real": "float",
    "float32": "float",
    "float64": "double",
    "varchar": "string",
    "char": "string",
    "text": "string",
    "str": "string",
    "timestampntz": "timestamp_ntz",
}


def _valid_type(datatype: Any, operation: str, where: str) -> Any:
    """`datatype` checked as a Delta schema type, with optional flags defaulted.

    A type the protocol does not define -- ``int``, ``bigint``, ``int64`` --
    lands in `schemaString` verbatim, and from then on every reader fails to
    parse the table's schema. Array and map types need their nullability
    flags, which readers treat as required.
    """
    if isinstance(datatype, str):
        if datatype in _PRIMITIVE_TYPES:
            return datatype
        match = _DECIMAL.fullmatch(datatype)
        if match is not None:
            precision, scale = int(match.group(1)), int(match.group(2))
            if 1 <= precision <= 38 and 0 <= scale <= precision:
                return f"decimal({precision},{scale})"
            raise _refuse(
                operation,
                f"{where} has type {datatype}, outside Delta's decimal range "
                "(precision 1-38, 0 <= scale <= precision)",
            )
        hint = _SPELLING_HINTS.get(datatype.strip().lower())
        if hint is None and datatype.strip().lower() in _PRIMITIVE_TYPES:
            hint = datatype.strip().lower()
        raise _refuse(
            operation,
            f"{where} has type {datatype!r}, which is not a Delta type"
            + (f"; Delta spells it {hint!r}" if hint else ""),
        )
    if isinstance(datatype, Mapping):
        kind = datatype.get("type")
        out = dict(datatype)
        if kind == "struct":
            fields = datatype.get("fields")
            if not isinstance(fields, list) or not fields:
                raise _refuse(operation, f"{where} is a struct with no fields")
            out["fields"] = [_valid_field(f, operation, where) for f in fields]
            return out
        if kind == "array":
            if "elementType" not in datatype:
                raise _refuse(operation, f"{where} is an array with no elementType")
            out["elementType"] = _valid_type(datatype["elementType"], operation, f"{where}[]")
            out["containsNull"] = bool(datatype.get("containsNull", True))
            return out
        if kind == "map":
            if "keyType" not in datatype or "valueType" not in datatype:
                raise _refuse(operation, f"{where} is a map without keyType and valueType")
            out["keyType"] = _valid_type(datatype["keyType"], operation, f"{where} key")
            out["valueType"] = _valid_type(datatype["valueType"], operation, f"{where} value")
            out["valueContainsNull"] = bool(datatype.get("valueContainsNull", True))
            return out
    raise _refuse(operation, f"{where} has type {datatype!r}, which is not a Delta type")


def _valid_field(f: Any, operation: str, where: str) -> dict[str, Any]:
    if not isinstance(f, Mapping) or not isinstance(f.get("name"), str) or "type" not in f:
        raise _refuse(operation, f"{f!r} in {where} is not a Delta schema field (name, type)")
    out = dict(f)
    out["type"] = _valid_type(f["type"], operation, f"{where}.{f['name']}")
    out["nullable"] = f.get("nullable", True)
    if not isinstance(out["nullable"], bool):
        raise _refuse(operation, f"{where}.{f['name']}: nullable must be True or False")
    out["metadata"] = dict(f.get("metadata") or {})
    return out


def _check_names(fields: list[dict[str, Any]], operation: str, *, column_mapping: bool) -> None:
    """Refuse empty names, sibling duplicates, and Parquet-invalid names without CM."""
    seen: set[str] = set()
    for f in fields:
        name = f.get("name")
        if not isinstance(name, str) or not name:
            raise _refuse(operation, "a column needs a non-empty name")
        if name.lower() in seen:
            raise _refuse(operation, f"the column name {name!r} appears twice in one struct")
        seen.add(name.lower())
        if not column_mapping and _INVALID_NAME_CHARS & set(name):
            raise _refuse(
                operation,
                f"the column name {name!r} contains a character Parquet does not allow "
                "(one of ' ,;{}()\\n\\t='), and without column mapping the name is also the "
                "Parquet column name",
                "set_properties({'delta.columnMapping.mode': 'name'}) first",
            )
        for nested in _structs_in(f.get("type")):
            _check_names(nested["fields"], operation, column_mapping=column_mapping)


def _structs_in(datatype: Any) -> Iterable[dict[str, Any]]:
    """The struct types directly reachable from a type through arrays and maps."""
    if not isinstance(datatype, dict):
        return
    kind = datatype.get("type")
    if kind == "struct":
        yield datatype
    elif kind == "array":
        yield from _structs_in(datatype.get("elementType"))
    elif kind == "map":
        yield from _structs_in(datatype.get("keyType"))
        yield from _structs_in(datatype.get("valueType"))


def _max_column_id(state: TableState) -> int:
    """The highest column-mapping id in use: the recorded maximum or any field's."""
    recorded = state.configuration.get(_CM_MAX)
    highest = int(recorded) if recorded not in (None, "") else 0
    for f in _walk(state.schema):
        value = (f.get("metadata") or {}).get(_CM_ID)
        if value is not None:
            highest = max(highest, int(value))
    return highest


def _physical_path(schema: dict[str, Any], path: list[str]) -> list[str]:
    """The physical names along a logical column path (which must exist)."""
    names: list[str] = []
    container = _fields(schema)
    for depth, part in enumerate(path):
        found = next(f for f in container if f["name"].lower() == part.lower())
        names.append(_physical_name(found))
        if depth < len(path) - 1:
            container = _fields(found["type"])
    return names


# ------------------------------------------------------------ arrow -> delta


def arrow_to_delta_type(arrow_type: Any) -> Any:
    """Map an Arrow type to its Delta schema JSON form."""
    import pyarrow as pa
    import pyarrow.types as t

    if t.is_dictionary(arrow_type):
        # Dictionary encoding is an in-memory detail; the column holds values.
        return arrow_to_delta_type(arrow_type.value_type)
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
        if not 0 < arrow_type.precision <= 38 or not 0 <= arrow_type.scale <= arrow_type.precision:
            raise _refuse(
                "convert an Arrow type to a Delta type",
                f"{arrow_type} is outside Delta's decimal range (precision 1-38, "
                "0 <= scale <= precision)",
            )
        return f"decimal({arrow_type.precision},{arrow_type.scale})"
    if t.is_struct(arrow_type):
        return {
            "type": "struct",
            "fields": [
                arrow_to_delta_field(arrow_type.field(i)) for i in range(arrow_type.num_fields)
            ],
        }
    if (
        t.is_list(arrow_type)
        or t.is_large_list(arrow_type)
        or t.is_fixed_size_list(arrow_type)
        or getattr(t, "is_list_view", lambda _t: False)(arrow_type)
        or getattr(t, "is_large_list_view", lambda _t: False)(arrow_type)
    ):
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
        if k.startswith("delta.columnMapping.") or k.startswith("PARQUET:"):
            # Ids and physical names belong to the table the schema was read
            # from; carried into another table they would collide or point
            # at columns that do not exist there. They are assigned afresh.
            continue
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
        metadata["description"] = str(comment)
    else:
        metadata.pop("description", None)
    return Change(
        "SET TBLPROPERTIES",
        {"comment": comment or ""},
        metadata=_if_changed(state, metadata),
    )


def set_column_comment(state: TableState, column: str, comment: str | None) -> Change:
    schema = state.schema
    container, index = _find(schema, _split(column))
    meta = container[index].setdefault("metadata", {})
    if comment:
        meta["comment"] = str(comment)
    else:
        meta.pop("comment", None)
    return Change(
        "CHANGE COLUMN",
        {"column": column, "comment": comment or ""},
        metadata=_if_changed(state, _new_metadata(state, schema)),
    )


def set_nullability(state: TableState, column: str, nullable: bool) -> Change:
    """SET/DROP NOT NULL. The caller must have checked for nulls before SET."""
    if not isinstance(nullable, bool):
        raise _refuse(f"change the nullability of {column}", "nullable must be True or False")
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
        metadata=_if_changed(state, _new_metadata(state, schema)),
    )


def _cdf_enabled(configuration: Mapping[str, str]) -> bool:
    return str(configuration.get("delta.enableChangeDataFeed", "false")).lower() == "true"


def add_columns(state: TableState, new_fields: list[dict[str, Any]]) -> Change:
    """ADD COLUMNS. `new_fields` are Delta schema field dicts."""
    schema = state.schema
    existing = {f["name"].lower() for f in _fields(schema)}
    configuration = state.configuration
    column_mapping = state.column_mapping_mode in ("name", "id")
    for f in new_fields:
        if not isinstance(f, Mapping) or not isinstance(f.get("name"), str) or "type" not in f:
            raise _refuse("add columns", f"{f!r} is not a Delta schema field (name, type)")
        if f["name"].lower() in existing:
            raise _refuse(f"add column {f['name']}", "a column with that name already exists")
        if not f.get("nullable", True):
            raise _refuse(
                f"add NOT NULL column {f['name']}",
                "existing rows have no value for a new column, so it must be nullable",
            )
        if _cdf_enabled(configuration) and f["name"].lower() in _CDF_RESERVED:
            raise _refuse(
                f"add column {f['name']}",
                "the change data feed is enabled, and it reserves that name for its own "
                "change-row columns",
            )
        existing.add(f["name"].lower())
    if not new_fields:
        return Change("ADD COLUMNS", {"columns": "[]"})

    added = copy.deepcopy(new_fields)
    for f in added:
        f.setdefault("nullable", True)
        f.setdefault("metadata", {})
        f["type"] = _valid_type(f["type"], f"add column {f['name']}", repr(f["name"]))
        meta = f["metadata"] = dict(f["metadata"] or {})
        if _GENERATION in meta or any(str(k).startswith("delta.identity.") for k in meta):
            # Spark refuses these on ADD COLUMNS: a generated or identity
            # column is declared at CREATE, and needs its table feature.
            raise _refuse(
                f"add column {f['name']}",
                "generated and identity columns can only be declared when the table is created",
            )
        if "CURRENT_DEFAULT" in meta and "allowColumnDefaults" not in supported_features(
            state.protocol
        ):
            raise _refuse(
                f"add column {f['name']} with a default",
                "a column default needs the allowColumnDefaults table feature, and writers "
                "that do not know it would ignore the default",
                "add_feature('allowColumnDefaults') first",
            )
        # Ids and physical names are this table's to assign; copied from another
        # table's schema they would collide with existing columns.
        for key in [k for k in meta if str(k).startswith("delta.columnMapping.")]:
            del meta[key]
    _check_names(added, "add columns", column_mapping=column_mapping)
    if column_mapping:
        next_id = _max_column_id(state) + 1
        next_id = _assign_ids(added, next_id, physical=lambda _f: f"col-{uuid.uuid4()}")
        configuration[_CM_MAX] = str(next_id - 1)

    features: set[str] = set()
    for f in added:
        features |= _type_features(f["type"])
    schema["fields"] = _fields(schema) + added
    metadata = _new_metadata(state, schema)
    metadata["configuration"] = configuration
    return Change(
        "ADD COLUMNS",
        {"columns": json.dumps([f["name"] for f in added])},
        protocol=with_features(state.protocol, features) if features else None,
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
        p.lower() for p in state.metadata.get("partitionColumns") or []
    }:
        raise _refuse(f"drop column {column}", "it is a partition column")
    if len(container) == 1:
        raise _refuse(
            f"drop column {column}",
            "a table must keep at least one column"
            if len(path) == 1
            else "a struct must keep at least one field",
        )
    partitions = {p.lower() for p in state.metadata.get("partitionColumns") or []}
    if (
        len(path) == 1
        and partitions
        and all(f["name"].lower() in partitions for f in container if f is not target)
    ):
        # Data files hold only the non-partition columns; with none left there
        # is nothing to write, and Spark refuses the schema.
        raise _refuse(
            f"drop column {column}",
            "it is the last non-partition column, and a table needs at least one",
        )
    # Clustering keys are physical paths from the root, so compare the whole
    # physical path: the leaf's physical name alone never matches a nested key.
    physical = _physical_path(schema, path)
    if any(c[: len(physical)] == physical for c in _clustering_columns(state)):
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
    # A nested rename may name the new column by its full path, as Spark's
    # RENAME COLUMN a.b TO a.c does; the new leaf is its last part.
    new_path = _split(new) if "`" in new or "." in new else [new]
    if len(new_path) > 1 and (
        len(new_path) != len(path)
        or [p.lower() for p in new_path[:-1]] != [p.lower() for p in path[:-1]]
    ):
        raise _refuse(
            f"rename column {old} to {new}",
            "a rename keeps the column in its struct; only the last part of the path may change",
            "quote a name containing a dot with backticks",
        )
    leaf = new_path[-1]
    if not leaf:
        raise _refuse(f"rename column {old}", "the new name is empty")
    if any(f["name"].lower() == leaf.lower() for f in container if f is not container[index]):
        raise _refuse(f"rename column {old} to {new}", f"a column named {leaf!r} already exists")
    if len(path) == 1 and _cdf_enabled(state.configuration) and leaf.lower() in _CDF_RESERVED:
        raise _refuse(
            f"rename column {old} to {new}",
            "the change data feed is enabled, and it reserves that name for its own "
            "change-row columns",
        )
    if leaf == current:
        return Change("RENAME COLUMN", {"oldColumnPath": old, "newColumnPath": new})
    dependents = _dependents(state, current)
    if dependents:
        raise _refuse(
            f"rename column {old}",
            f"it is referenced by {', '.join(dependents)}, whose expressions name it",
            "drop and re-create those after the rename",
        )
    container[index]["name"] = leaf
    metadata = _new_metadata(state, schema)
    if len(path) == 1:
        metadata["partitionColumns"] = [
            leaf if p.lower() == current.lower() else p
            for p in state.metadata.get("partitionColumns") or []
        ]
    return Change(
        "RENAME COLUMN",
        {"oldColumnPath": old, "newColumnPath": new},
        metadata=metadata,
    )


# Widenings the typeWidening feature allows (Delta protocol, "Type Widening").
_INTEGRAL = ("byte", "short", "integer", "long")
#: Digits a decimal must keep left of the point to take an integral type
#: (Delta protocol, Type Widening: byte/short/int -> decimal(10 + k1, k2),
#: long -> decimal(20 + k1, k2), k1 >= k2 >= 0). Byte and short share int's
#: bound: decimal(5,0) from a byte is not a widening readers accept.
_INTEGRAL_DIGITS = {"byte": 10, "short": 10, "integer": 10, "long": 20}
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
    if not 0 < precision <= 38 or scale > precision:
        return False
    old_dec = _DECIMAL.fullmatch(old.replace(" ", ""))
    if old_dec is not None:
        p0, s0 = int(old_dec.group(1)), int(old_dec.group(2))
        return precision >= p0 and scale >= s0 and (precision - scale) >= (p0 - s0)
    if old in _INTEGRAL:
        return precision - scale >= _INTEGRAL_DIGITS[old]
    return False


_TYPE_ALIASES = {
    "int": "integer",
    "bigint": "long",
    "smallint": "short",
    "tinyint": "byte",
    "real": "float",
    "numeric": "decimal",
    "dec": "decimal",
}


def _normalise_type(name: str) -> str:
    text = name.strip().lower().replace(" ", "")
    head, paren, tail = text.partition("(")
    head = _TYPE_ALIASES.get(head, head)
    if head == "decimal" and not paren:
        return "decimal(10,0)"  # SQL's DECIMAL without arguments
    if head == "decimal" and re.fullmatch(r"\d+\)", tail):
        return f"decimal({tail[:-1]},0)"  # SQL's DECIMAL(p): scale 0
    return head + paren + tail


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
    new_type = _normalise_type(new_type)
    if not isinstance(old_type, str) or not _widening_allowed(old_type, new_type):
        raise _refuse(
            f"change the type of {column} from {old_type} to {new_type}",
            "only widening changes are allowed without a rewrite: byte->short->int->long, "
            "float->double, byte/short/int->double, date->timestamp_ntz, and decimals "
            "whose precision and scale do not shrink",
        )
    if len(path) == 1 and target["name"].lower() in {
        p.lower() for p in state.metadata.get("partitionColumns") or []
    }:
        raise _refuse(f"change the type of {column}", "it is a partition column")
    # Spark refuses a type change under a CHECK constraint or a generated
    # column: the expression was validated against the old type, and a
    # generated column's stored type no longer matches what it computes.
    dependents = [
        d for d in _dependents(state, target["name"]) if d != "delta.dataSkippingStatsColumns"
    ]
    if dependents:
        raise _refuse(
            f"change the type of {column}",
            f"it is referenced by {', '.join(dependents)}",
            "drop those first, and re-create them after the change",
        )
    target["type"] = new_type
    meta = target.setdefault("metadata", {})
    history = list(meta.get(_TYPE_CHANGES) or [])
    history.append({"fromType": old_type, "toType": new_type})
    meta[_TYPE_CHANGES] = history
    # date -> timestamp_ntz puts a timestamp_ntz column in the schema, which
    # the table may only hold with the timestampNtz feature.
    features = {"typeWidening"} | _type_features(new_type)
    return Change(
        "CHANGE COLUMN",
        {"column": column, "type": new_type},
        protocol=with_features(state.protocol, features),
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


#: Types whose min/max statistics clustering can use (Spark's
#: SkippingEligibleDataType); boolean and binary collect none.
_CLUSTERABLE = frozenset(
    {
        "byte",
        "short",
        "integer",
        "long",
        "float",
        "double",
        "date",
        "timestamp",
        "timestamp_ntz",
        "string",
    }
)
#: Spark's limit on liquid clustering keys.
_MAX_CLUSTERING_COLUMNS = 4


def _leaf_paths(
    fields: list[dict[str, Any]], prefix: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    """Logical leaf column paths in schema order, as stats indexing counts them."""
    out: list[tuple[str, ...]] = []
    for f in fields:
        path = (*prefix, f["name"].lower())
        datatype = f["type"]
        if isinstance(datatype, dict) and datatype.get("type") == "struct":
            out.extend(_leaf_paths(datatype["fields"], path))
        else:
            out.append(path)
    return out


def _has_stats(
    state: TableState,
    schema: dict[str, Any],
    path: list[str],
    config: Mapping[str, str] | None = None,
) -> bool:
    """Whether writers collect min/max statistics for this column."""
    config = state.configuration if config is None else config
    wanted = tuple(p.lower() for p in path)
    listed = config.get("delta.dataSkippingStatsColumns")
    if listed:
        for entry in listed.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = tuple(p.lower() for p in _split(entry))
            if wanted[: len(parts)] == parts:
                return True
        return False
    try:
        limit = int(config.get("delta.dataSkippingNumIndexedCols", "32"))
    except ValueError:
        return True  # an unreadable setting is not a reason to refuse here
    if limit < 0:
        return True
    return wanted in _leaf_paths(_fields(schema))[:limit]


def _logical_path(schema: dict[str, Any], physical: list[str]) -> list[str] | None:
    """The logical names along a physical column path, or None if it is gone."""
    names: list[str] = []
    container: list[dict[str, Any]] | None = _fields(schema)
    for depth, part in enumerate(physical):
        found = next((f for f in container or [] if _physical_name(f) == part), None)
        if found is None:
            return None
        names.append(found["name"])
        if depth < len(physical) - 1:
            datatype = found["type"]
            if not (isinstance(datatype, dict) and datatype.get("type") == "struct"):
                return None
            container = _fields(datatype)
    return names


def _check_clustering_stats(
    state: TableState, configuration: Mapping[str, str], operation: str
) -> None:
    """Refuse a stats setting that leaves a clustering key without statistics.

    Spark validates this on ALTER TBLPROPERTIES too: clustering is driven by
    the keys' min/max statistics, and a key no writer collects stats for
    silently stops clustering anything.
    """
    columns = _clustering_columns(state)
    stats_keys = ("delta.dataSkippingNumIndexedCols", "delta.dataSkippingStatsColumns")
    before = state.configuration
    if not columns or all(before.get(k) == configuration.get(k) for k in stats_keys):
        return
    schema = state.schema
    for physical in columns:
        logical = _logical_path(schema, physical)
        if logical is not None and not _has_stats(state, schema, logical, configuration):
            raise _refuse(
                operation,
                f"the clustering column {'.'.join(logical)} would get no statistics, and "
                "clustering needs them",
                "keep it inside delta.dataSkippingNumIndexedCols / "
                "delta.dataSkippingStatsColumns, or change the clustering keys first",
            )


def cluster_by(state: TableState, columns: list[str] | str | None) -> Change:
    """Set liquid-clustering keys. Takes effect for data written afterwards."""
    if state.metadata.get("partitionColumns"):
        raise _refuse(
            "cluster a partitioned table",
            "a table is either partitioned or clustered, not both",
        )
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns or [])
    if len(columns) > _MAX_CLUSTERING_COLUMNS:
        raise _refuse(
            f"cluster by {columns}",
            f"a table can be clustered by at most {_MAX_CLUSTERING_COLUMNS} columns",
        )
    schema = state.schema
    physical: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for column in columns:
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
            elif not (
                found["type"] in _CLUSTERABLE or _DECIMAL.fullmatch(found["type"]) is not None
            ):
                raise _refuse(
                    f"cluster by {column}",
                    f"its type {found['type']} has no min/max statistics for clustering to use",
                )
        key = tuple(n.lower() for n in names)
        if key in seen:
            raise _refuse(f"cluster by {columns}", f"{column!r} is listed twice")
        seen.add(key)
        if not _has_stats(state, schema, path):
            raise _refuse(
                f"cluster by {column}",
                "writers collect no statistics for it (it is outside "
                "delta.dataSkippingNumIndexedCols / delta.dataSkippingStatsColumns), and "
                "clustering needs them",
                "add it to delta.dataSkippingStatsColumns first",
            )
        physical.append(names)

    current = _clustering_columns(state)
    # Clearing keys on a table that was never clustered: adding the
    # clustering and domainMetadata features for an empty key list would lock
    # delta-rs out of writing the table for no effect.
    if state.clustering is None and not physical:
        return Change("CLUSTER BY", {"newClusteringColumns": ""})
    # Keys already exactly these: rewriting the domain is a commit that changes
    # nothing (and bumps the version other writers conflict on).
    if state.clustering is not None and physical == current:
        return Change("CLUSTER BY", {"newClusteringColumns": ",".join(columns)})

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
        {"newClusteringColumns": ",".join(columns)},
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
        "delta.enableRowTracking",
    }
)


#: Integer-valued keys and the smallest value each accepts.
_INTEGER_KEYS: dict[str, int] = {
    "delta.checkpointInterval": 1,
    "delta.dataSkippingNumIndexedCols": -1,
    "delta.randomPrefixLength": 1,
}
_DURATION_KEYS = frozenset(
    {
        "delta.deletedFileRetentionDuration",
        "delta.logRetentionDuration",
        "delta.setTransactionRetentionDuration",
    }
)
_INTERVAL = re.compile(
    r"(?:interval\s+)?(\d+)\s*"
    r"((?:nanosecond|microsecond|millisecond|second|minute|hour|day|week)s?)",
    re.IGNORECASE,
)
_FILE_SIZE = re.compile(r"\d+\s*(?:b|k|kb|m|mb|g|gb)?", re.IGNORECASE)
_ISOLATION_LEVELS = {"serializable": "Serializable", "writeserializable": "WriteSerializable"}
_CODECS = frozenset(
    {"uncompressed", "none", "snappy", "gzip", "lzo", "brotli", "lz4", "lz4_raw", "zstd"}
)
#: Canonical spelling of every `delta.` key this path handles. Spark matches
#: `delta.` keys case-insensitively and stores the canonical spelling; readers
#: look keys up by that exact spelling, so `delta.appendonly` would be ignored.
_CANONICAL_KEYS: dict[str, str] = {
    k.lower(): k
    for k in (
        *_KNOWN_KEYS,
        "delta.enableRowTracking",
        "delta.minReaderVersion",
        "delta.minWriterVersion",
        "delta.columnMapping.maxColumnId",
        "delta.inCommitTimestampEnablementVersion",
        "delta.inCommitTimestampEnablementTimestamp",
    )
}


def _canonical_key(key: str) -> str:
    key = str(key)
    lowered = key.lower()
    if lowered in _CANONICAL_KEYS:
        return _CANONICAL_KEYS[lowered]
    for prefix in ("delta.feature.", "delta.constraints."):
        if lowered.startswith(prefix):
            return prefix + key[len(prefix) :]
    return key


def _canonical_feature(name: str) -> str:
    """A feature name in its wire spelling; Spark matches them case-insensitively."""
    for feature in TableFeature:
        if feature.value.lower() == name.lower():
            return feature.value
    return name


def _normalise_value(key: str, raw: Any, state: TableState, operation: str) -> str:
    """Validate a property value and return the spelling readers parse."""
    if raw is None:
        raise _refuse(operation, f"{key} has no value", "unset_properties() removes a property")
    value = str(raw).lower() if isinstance(raw, bool) else str(raw)
    stripped = value.strip()
    lowered = stripped.lower()
    if key in _BOOLEAN_KEYS:
        if lowered not in ("true", "false"):
            raise _refuse(operation, f"{key} must be 'true' or 'false', not {value!r}")
        # Readers parse these with a case-sensitive boolean parser.
        return lowered
    if key in _INTEGER_KEYS:
        try:
            number = int(stripped)
        except ValueError:
            number = None
        if number is None or number < _INTEGER_KEYS[key]:
            raise _refuse(
                operation,
                f"{key} must be an integer of at least {_INTEGER_KEYS[key]}, not {value!r}",
            )
        return str(number)
    if key in _DURATION_KEYS:
        match = _INTERVAL.fullmatch(stripped)
        if match is None:
            raise _refuse(
                operation,
                f"{key} must be an interval such as 'interval 7 days', not {value!r}",
            )
        return f"interval {int(match.group(1))} {match.group(2).lower()}"
    if key == "delta.targetFileSize":
        if _FILE_SIZE.fullmatch(stripped) is None:
            raise _refuse(operation, f"{key} must be a size in bytes, such as 134217728 or 128mb")
        return stripped
    if key == "delta.isolationLevel":
        if lowered not in _ISOLATION_LEVELS:
            raise _refuse(operation, f"{key} is Serializable or WriteSerializable, not {value!r}")
        return _ISOLATION_LEVELS[lowered]
    if key == "delta.parquet.compression.codec":
        if lowered not in _CODECS:
            raise _refuse(operation, f"{key} is not a Parquet codec: {value!r}")
        return lowered
    if key == "delta.checkpointPolicy":
        if lowered not in ("classic", "v2"):
            raise _refuse(operation, "delta.checkpointPolicy is 'classic' or 'v2'")
        return lowered
    if key == _CM_MODE:
        return lowered
    if key == "delta.dataSkippingStatsColumns":
        schema = state.schema
        for entry in stripped.split(","):
            if entry.strip():
                _find(schema, _split(entry.strip()))
        return stripped
    return value


def set_properties(state: TableState, properties: Mapping[str, Any]) -> Change:
    """SET TBLPROPERTIES, including the protocol changes the values imply."""
    configuration = state.configuration
    features: set[str] = set()
    schema: dict[str, Any] | None = None
    operation = f"set table properties {sorted(map(str, properties))}"

    for raw_key, raw in properties.items():
        key = _canonical_key(raw_key)
        if key in ("delta.minReaderVersion", "delta.minWriterVersion"):
            raise _refuse(
                operation,
                f"{key} is not set directly; enabling a feature raises the protocol as needed",
                "set the feature instead, e.g. delta.feature.deletionVectors = supported",
            )
        if key.startswith("delta.constraints."):
            raise _refuse(operation, "constraints are added through add_constraint()")
        if key.startswith("delta.feature."):
            name = _canonical_feature(key.removeprefix("delta.feature."))
            if raw is None or str(raw).lower() not in ("supported", "enabled"):
                raise _refuse(operation, f"{key} only accepts 'supported'")
            if name in _NOT_ADDABLE:
                raise _refuse(
                    operation, f"cannot add {name} to an existing table: {_NOT_ADDABLE[name]}"
                )
            if name not in _ADDABLE_FEATURES:
                raise _refuse(operation, f"{name} is not a feature this path can add")
            features.add(name)
            continue
        if key == "delta.enableRowTracking" and str(raw).lower() == "true":
            raise _refuse(
                operation,
                "enabling row tracking on an existing table needs every file backfilled "
                "with row ids, which is a data operation",
                "use the SQL fallback, or create the table with row tracking",
            )
        if key.lower().startswith(("delta.enableicebergcompat", "delta.universalformat")):
            raise _refuse(operation, _NOT_ADDABLE["icebergCompatV3"], "use the SQL fallback")
        if key.startswith("delta.") and key not in _KNOWN_KEYS and key != "delta.enableRowTracking":
            raise _refuse(operation, f"{key} is not a Delta table property this path knows")
        value = _normalise_value(key, raw, state, operation)
        lowered = value.lower()

        if key in _ENABLING and lowered == "true":
            features.add(_ENABLING[key])
        if key == "delta.enableChangeDataFeed" and lowered == "true":
            clash = [f["name"] for f in _fields(state.schema) if f["name"].lower() in _CDF_RESERVED]
            if clash:
                raise _refuse(
                    operation,
                    f"the change data feed reserves the column names {sorted(_CDF_RESERVED)}, "
                    f"and the table has {clash}",
                    "rename those columns first",
                )
        if key == "delta.checkpointPolicy" and lowered == "v2":
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

    _check_clustering_stats(state, configuration, operation)
    metadata = _new_metadata(state, schema)
    metadata["configuration"] = configuration
    protocol = with_features(state.protocol, features) if features else None
    return Change(
        "SET TBLPROPERTIES",
        {"properties": json.dumps({str(k): str(v) for k, v in properties.items()}, sort_keys=True)},
        protocol=protocol,
        metadata=_if_changed(state, metadata),
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
    # Materialised once: the keys are iterated here and again for commitInfo,
    # and a generator would come up empty the second time.
    keys = [keys] if isinstance(keys, str) else [str(k) for k in keys]
    configuration = state.configuration
    removed: list[str] = []
    for raw_key in keys:
        key = _canonical_key(raw_key)
        if key.startswith("delta.feature.") or key in (_CM_MODE, _CM_MAX):
            raise _refuse(
                f"unset {key}",
                "removing it would change the protocol or orphan column-mapped data",
                "drop the feature through the SQL fallback",
            )
        if key.startswith("delta.inCommitTimestampEnablement"):
            raise _refuse(f"unset {key}", "readers need it to interpret commit timestamps")
        if key.lower().startswith("delta.rowtracking."):
            # The materialised row-id / row-commit-version column names: writers
            # that preserve row ids across a rewrite look them up, and existing
            # files already carry columns under these names.
            raise _refuse(f"unset {key}", "row tracking needs it to find materialised row ids")
        if key not in configuration:
            if if_exists:
                continue
            raise _refuse(f"unset {key}", "the table has no such property")
        del configuration[key]
        removed.append(key)
    if not removed:
        return Change("UNSET TBLPROPERTIES", {"properties": json.dumps(sorted(keys))})
    _check_clustering_stats(state, configuration, f"unset table properties {sorted(removed)}")
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

    Refuses a catalog-managed table: its commits are ratified by the catalog,
    and a log file written here directly would fork the table's history.
    """
    managed = _CATALOG_FEATURES & (
        set(state.protocol.get("readerFeatures") or [])
        | set(state.protocol.get("writerFeatures") or [])
    )
    if managed:
        raise _refuse(
            f"commit {change.operation}",
            f"the table is catalog-managed ({', '.join(sorted(managed))}); its commits go "
            "through the catalog, and a log file written directly would not be ratified",
            "make the change through the catalog (e.g. Databricks SQL)",
        )
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
    configuration: Mapping[str, Any],
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
    operation = "create the table"
    for key, value in configuration.items():
        if value is None:
            raise _refuse(operation, f"the property {key} has no value")
    schema = copy.deepcopy(schema)
    schema = _valid_type(schema, operation, "the table schema")
    # A state over the new schema, for the checks that consult one.
    draft = TableState(
        version=-1,
        protocol={"minReaderVersion": 1, "minWriterVersion": 1},
        metadata={"schemaString": json.dumps(schema), "partitionColumns": [], "configuration": {}},
    )
    # Readers parse boolean properties case-sensitively, and the log's
    # configuration is a string map: True must land as "true", not a JSON bool.
    # Known `delta.` keys get the canonical spelling and a validated value,
    # exactly as set_properties gives them: `delta.enablechangedatafeed` or
    # mode "Name" would otherwise be stored and then ignored by every reader.
    config: dict[str, str] = {}
    signals: set[str] = set()
    for raw_key, raw in configuration.items():
        key = _canonical_key(raw_key)
        if key.startswith("delta.feature."):
            if str(raw).lower() not in ("supported", "enabled"):
                raise _refuse(operation, f"{key} only accepts 'supported', not {raw!r}")
            signals.add(_canonical_feature(key.removeprefix("delta.feature.")))
            continue
        if key in _KNOWN_KEYS or key in _BOOLEAN_KEYS:
            config[key] = _normalise_value(key, raw, draft, operation)
        else:
            config[key] = str(raw).lower() if isinstance(raw, bool) else str(raw)
    mode = config.get(_CM_MODE)
    if mode is not None and mode not in ("none", "name", "id"):
        raise _refuse(operation, f"delta.columnMapping.mode is none, name or id, not {mode!r}")
    if _cdf_enabled(config):
        clash = [f["name"] for f in _fields(schema) if f["name"].lower() in _CDF_RESERVED]
        if clash:
            raise _refuse(
                operation,
                f"the change data feed reserves the column names {sorted(_CDF_RESERVED)}, "
                f"and the schema has {clash}",
            )
    features: set[str] = set(_wire(required_protocol, "writer-features", "writerFeatures") or [])
    required_readers = set(_wire(required_protocol, "reader-features", "readerFeatures") or [])
    features |= required_readers
    features |= signals
    for key, feature in _ENABLING.items():
        if str(config.get(key, "")).lower() == "true":
            features.add(feature)
    if str(config.get("delta.checkpointPolicy", "")).lower() == "v2":
        features.add("v2Checkpoint")
    if str(config.get("delta.enableRowTracking", "")).lower() == "true":
        features.add("rowTracking")
        # Writers that materialise row ids look these names up; without them
        # a row-tracking table cannot be compacted or updated by Spark.
        config.setdefault(
            "delta.rowTracking.materializedRowIdColumnName", f"_row-id-col-{uuid.uuid4()}"
        )
        config.setdefault(
            "delta.rowTracking.materializedRowCommitVersionColumnName",
            f"_row-commit-version-col-{uuid.uuid4()}",
        )
    if features & _CATALOG_FEATURES:
        # A catalog-managed table orders commits by in-commit timestamp, so
        # the feature it depends on must be enabled, not merely supported.
        # The kernel's create path and UCCommitter both refuse a catalog-managed
        # table with it off, so an explicit "false" would leave a table that
        # accepts no commit after version 0.
        if config.get("delta.enableInCommitTimestamps", "true") != "true":
            raise _refuse(
                operation,
                "a catalog-managed table requires in-commit timestamps, so "
                "delta.enableInCommitTimestamps cannot be false",
                "drop the delta.enableInCommitTimestamps property",
            )
        config["delta.enableInCommitTimestamps"] = "true"
        features.add("inCommitTimestamp")

    column_mapping = config.get(_CM_MODE, "none").lower() in ("name", "id")
    _check_names(_fields(schema), "create the table", column_mapping=column_mapping)
    features |= _type_features(schema)
    if column_mapping:
        features.add("columnMapping")
        last = _assign_ids(_fields(schema), 1, physical=lambda _f: f"col-{uuid.uuid4()}")
        config[_CM_MAX] = str(last - 1)

    # Partition columns are top-level primitive columns, recorded with the
    # schema's own spelling (Spark resolves them case-insensitively).
    by_name = {f["name"].lower(): f for f in _fields(schema)}
    partitions: list[str] = []
    for column in partition_columns or []:
        found = by_name.get(str(column).lower())
        if found is None:
            raise _refuse("create the table", f"partition column {column!r} is not in the schema")
        if isinstance(found["type"], dict):
            raise _refuse(
                "create the table", f"partition column {column!r} is not of a primitive type"
            )
        if found["name"] in partitions:
            raise _refuse("create the table", f"partition column {column!r} is listed twice")
        partitions.append(found["name"])
    if partitions and len(partitions) == len(_fields(schema)):
        raise _refuse(
            "create the table", "every column is a partition column, leaving no data columns"
        )

    domains: list[dict[str, Any]] = []
    if cluster_by:
        if partitions:
            raise _refuse("create the table", "a table is either partitioned or clustered")
        features |= {"clustering", "domainMetadata"}
        state = TableState(
            version=-1,
            protocol={"minReaderVersion": 1, "minWriterVersion": 1},
            metadata={
                "schemaString": json.dumps(schema),
                "partitionColumns": [],
                "configuration": dict(config),
            },
        )
        domains = cluster_by_domains(state, cluster_by)

    upgraded = with_features({"minReaderVersion": 1, "minWriterVersion": 1}, features) or {}
    writer_features = set(upgraded.get("writerFeatures", []))
    # Reader-writer features the catalog requires but this library does not
    # know still belong in readerFeatures: dropping them would let readers
    # that do not understand them open the table.
    reader_features = sorted(
        {
            n
            for n in writer_features
            if (f := feature_from_wire(n)) is not None
            and FEATURE_SUPPORT[f].kind is FeatureKind.READER_WRITER
        }
        | required_readers
    )
    protocol = {
        "minReaderVersion": max(
            3, int(_wire(required_protocol, "min-reader-version", "minReaderVersion") or 3)
        ),
        "minWriterVersion": 7,
        "readerFeatures": reader_features,
        "writerFeatures": sorted(writer_features | required_readers),
    }

    now = int(time.time() * 1000) if now_ms is None else now_ms
    commit_info: dict[str, Any] = {
        "timestamp": now,
        "operation": "CREATE TABLE",
        "operationParameters": {
            "partitionBy": json.dumps(partitions),
            "clusterBy": json.dumps(list(cluster_by or [])),
        },
        "engineInfo": engine_info,
        "isBlindAppend": True,
        "txnId": str(uuid.uuid4()),
    }
    # Dependencies count: catalogManaged brings inCommitTimestamp along.
    if "inCommitTimestamp" in writer_features:
        commit_info = {"inCommitTimestamp": now, **commit_info}
    metadata: dict[str, Any] = {"id": table_id}
    if name is not None:
        metadata["name"] = name
    if description is not None:
        metadata["description"] = description
    metadata.update(
        {
            "format": {"provider": "parquet", "options": {}},
            "schemaString": json.dumps(schema, separators=(",", ":")),
            "partitionColumns": partitions,
            "configuration": dict(config),
            "createdTime": now,
        }
    )
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
