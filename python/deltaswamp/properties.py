"""Validating table properties before an engine sees them.

delta-rs reports every property problem as "Error parsing property" and panics
on `delta.minReaderVersion`. Checking against `PROPERTY_SUPPORT` first produces
an error that names the key, the engine and the fix.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .capability import Engine, Operation, TableFeature, feature_from_wire
from .errors import IgnoredPropertyWarning, PropertyNotSupportedError

__all__ = [
    "CHECKPOINT_STATS_REMEDY",
    "FEATURE_SIGNAL_PREFIX",
    "KERNEL_CREATE_DEFERRED",
    "KERNEL_CREATE_FEATURES",
    "PROPERTY_SUPPORT",
    "PropertyEffect",
    "PropertySupport",
    "checkpoint_drops_stats",
    "effect_for",
    "engine_can_set",
    "parse_byte_size",
    "property_support",
    "validate_properties",
    "with_checkpoint_stats",
]


# ---------------------------------------------------------------------------
# Table properties
# ---------------------------------------------------------------------------
#
# Delta configuration is where the two engines diverge most sharply and most
# quietly. delta-rs rejects about half of the spec with one opaque message --
# "Kernel: Generic delta kernel error: Error parsing property" -- and panics
# outright on `delta.minReaderVersion`. The kernel accepts nearly all of it.
#
# The rows below come from probing the installed delta-rs and from reading
# kernel's ALLOWED_DELTA_FEATURES / ALLOWED_DELTA_PROPERTIES. Probe tests in
# `tests/integration/test_properties.py` re-run them, so an engine upgrade that
# changes behavior fails loudly instead of drifting.


class PropertyEffect(StrEnum):
    """What an engine does with a table property."""

    #: Stored, and it changes the protocol or the writer's behavior.
    HONORED = "honored"
    #: Stored verbatim, but nothing here acts on it. Databricks may.
    STORED = "stored"
    #: The engine raises. We refuse first, with a message that names the key.
    REJECTED = "rejected"
    #: The engine panics through the FFI boundary. Never let one reach a user.
    CRASH = "crash"
    #: The engine has no path for this operation at all.
    UNSUPPORTED = "unsupported"

    def usable(self) -> bool:
        return self in (PropertyEffect.HONORED, PropertyEffect.STORED)


@dataclass(frozen=True, slots=True)
class PropertySupport:
    """One row of the property matrix."""

    key: str
    deltars_create: PropertyEffect
    deltars_set: PropertyEffect
    kernel_create: PropertyEffect
    databricks_only: bool = False
    note: str = ""


_H, _ST, _RJ, _CR, _UN = (
    PropertyEffect.HONORED,
    PropertyEffect.STORED,
    PropertyEffect.REJECTED,
    PropertyEffect.CRASH,
    PropertyEffect.UNSUPPORTED,
)


def _prop(
    key: str,
    dc: PropertyEffect,
    dset: PropertyEffect,
    kc: PropertyEffect,
    databricks_only: bool = False,
    note: str = "",
) -> tuple[str, PropertySupport]:
    return key, PropertySupport(key, dc, dset, kc, databricks_only, note)


PROPERTY_SUPPORT: dict[str, PropertySupport] = dict(
    [
        # --- both engines handle these
        _prop("delta.appendOnly", _H, _H, _H),
        _prop("delta.columnMapping.mode", _H, _RJ, _H),
        _prop("delta.enableChangeDataFeed", _H, _H, _H),
        _prop(
            "delta.enableDeletionVectors",
            # Treated as rejected at create too, so such a create goes to the
            # kernel: delta-rs 1.6.5 answers it by stamping a variantType
            # reader+writer feature nobody asked for (and with a v2 checkpoint
            # policy writes a protocol that violates the spec, then panics).
            _RJ,
            _RJ,
            _H,
            False,
            "delta-rs writes duplicate feature entries and an unexpected "
            "variantType reader+writer feature into the protocol when this is "
            "enabled, at create or later, so the kernel creates DV tables",
        ),
        _prop("delta.dataSkippingNumIndexedCols", _H, _H, _H),
        _prop("delta.checkpoint.writeStatsAsStruct", _H, _H, _H),
        # --- delta-rs stores but does not act on
        _prop(
            "delta.checkpointPolicy",
            # Rejected rather than stored: delta-rs keeps the value but adds no
            # v2Checkpoint feature, a protocol Databricks treats as invalid, and
            # on a table-features protocol it panics. The kernel honors it.
            _RJ,
            _RJ,
            _H,
            False,
            "delta-rs stores it but adds no v2Checkpoint feature, so a v2 policy "
            "is inert there; the kernel honors it",
        ),
        _prop("delta.checkpointInterval", _H, _H, _H),
        _prop("delta.logRetentionDuration", _H, _H, _H),
        _prop("delta.deletedFileRetentionDuration", _H, _H, _H),
        # delta-rs acts on it: a write that lands on a checkpoint interval
        # deletes expired log files (as Spark does) unless this is false.
        _prop("delta.enableExpiredLogCleanup", _H, _H, _H),
        _prop("delta.setTransactionRetentionDuration", _ST, _ST, _H),
        _prop(
            "delta.dataSkippingStatsColumns",
            _H,
            _H,
            _H,
            False,
            "delta-rs honors top-level leaf names only (not nested fields, structs or "
            "column-mapped names), so appends and overwrites of a table that sets it go "
            "to the kernel, which honors every form",
        ),
        _prop("delta.checkpoint.writeStatsAsJson", _ST, _ST, _H),
        # The kernel refuses these at CREATE but stores them through the same
        # metadata commit set_properties() writes, so a kernel create applies
        # them as version 1 (KERNEL_CREATE_DEFERRED).
        _prop("delta.targetFileSize", _H, _H, _ST, False, "delta-rs-only writer hint"),
        _prop("delta.isolationLevel", _H, _H, _ST),
        _prop("delta.tuneFileSizesForRewrites", _ST, _ST, _ST, True),
        _prop("delta.autoOptimize.optimizeWrite", _ST, _ST, _ST, True),
        _prop("delta.autoOptimize.autoCompact", _ST, _ST, _ST, True),
        _prop("delta.randomizeFilePrefixes", _ST, _ST, _ST, True),
        # --- kernel only: delta-rs rejects these outright
        _prop("delta.enableRowTracking", _RJ, _RJ, _H),
        _prop("delta.enableInCommitTimestamps", _RJ, _RJ, _H),
        _prop("delta.enableTypeWidening", _RJ, _RJ, _H),
        _prop("delta.enableIcebergCompatV3", _RJ, _RJ, _H),
        _prop("delta.parquet.format.version", _RJ, _RJ, _H),
        # --- neither engine
        _prop(
            "delta.enableIcebergCompatV2",
            _RJ,
            _RJ,
            _UN,
            False,
            "V2 is superseded by V3, which the kernel supports",
        ),
        _prop(
            "delta.universalFormat.enabledFormats",
            _RJ,
            _RJ,
            _UN,
            True,
            "UniForm metadata generation is a Databricks-side job",
        ),
        _prop("delta.parquet.compression.codec", _RJ, _RJ, _UN),
        # Real Delta keys that delta-rs rejects ("Error parsing property");
        # the kernel's metadata path stores them for Databricks to act on.
        _prop(
            "delta.compatibility.symlinkFormatManifest.enabled",
            _RJ,
            _RJ,
            _UN,
            True,
            "Databricks regenerates the manifest on every write; writes from here do "
            "not, so call generate() after them",
        ),
        _prop("delta.checkpointRetentionDuration", _RJ, _RJ, _ST, True),
        # --- protocol versions: never set these by hand
        _prop(
            "delta.minReaderVersion",
            _CR,
            _RJ,
            _UN,
            False,
            "delta-rs PANICS ('Reader features should be present in writer "
            "features'), which is not a catchable Python exception. Enable the "
            "feature you want and let the writer raise the version",
        ),
        _prop(
            "delta.minWriterVersion",
            _H,
            _H,
            _UN,
            False,
            "raises the protocol without adding the matching features; prefer "
            "enabling features by name",
        ),
    ]
)

#: Feature signals (`delta.feature.<name> = supported`). delta-rs rejects every
#: one; the kernel accepts this subset at create.
KERNEL_CREATE_FEATURES: frozenset[TableFeature] = frozenset(
    {
        TableFeature.DOMAIN_METADATA,
        TableFeature.COLUMN_MAPPING,
        TableFeature.IN_COMMIT_TIMESTAMP,
        TableFeature.VACUUM_PROTOCOL_CHECK,
        TableFeature.CATALOG_MANAGED,
        TableFeature.DELETION_VECTORS,
        TableFeature.V2_CHECKPOINT,
        TableFeature.APPEND_ONLY,
        TableFeature.CHANGE_DATA_FEED,
        TableFeature.TYPE_WIDENING,
        TableFeature.ROW_TRACKING,
        TableFeature.VARIANT_TYPE,
        TableFeature.VARIANT_SHREDDING,
        TableFeature.INVARIANTS,
        TableFeature.MATERIALIZE_PARTITION_COLUMNS,
        TableFeature.ICEBERG_COMPAT_V3,
    }
)

FEATURE_SIGNAL_PREFIX = "delta.feature."

#: Plain metadata keys delta-kernel refuses in CREATE TABLE ("not supported
#: during CREATE TABLE") although the metadata commit behind set_properties()
#: stores them on the same table a moment later. Refusing them made a DV,
#: row-tracked or clustered table impossible to create with common Databricks
#: settings, so the kernel create commits them as version 1 instead.
KERNEL_CREATE_DEFERRED: frozenset[str] = frozenset(
    {
        "delta.targetFileSize",
        "delta.isolationLevel",
        "delta.tuneFileSizesForRewrites",
        "delta.autoOptimize.optimizeWrite",
        "delta.autoOptimize.autoCompact",
        "delta.randomizeFilePrefixes",
        "delta.checkpointRetentionDuration",
    }
)

_STATS_JSON = "delta.checkpoint.writeStatsAsJson"
_STATS_STRUCT = "delta.checkpoint.writeStatsAsStruct"


def checkpoint_drops_stats(properties: Mapping[str, str] | None) -> bool:
    """Whether a checkpoint written here would keep no file statistics at all.

    With writeStatsAsJson=false and writeStatsAsStruct unset, Spark writes
    struct stats (its default for the struct is true), but delta-rs and the
    kernel read unset as false and write neither -- every file then loses
    data skipping, on Databricks too, until the next full rewrite.
    """
    props = properties or {}
    return (
        str(props.get(_STATS_JSON, "true")).lower() == "false" and props.get(_STATS_STRUCT) is None
    )


CHECKPOINT_STATS_REMEDY = (
    "set delta.checkpoint.writeStatsAsStruct=true (Spark's default, which Databricks "
    "applies when it is unset) with set_properties(), then checkpoint"
)


def with_checkpoint_stats(
    properties: Mapping[str, str] | None, existing: Mapping[str, str] | None = None
) -> dict[str, str] | None:
    """`properties`, recording writeStatsAsStruct=true where they turn JSON stats off.

    Spark treats an unset writeStatsAsStruct as true; this library's writers
    treat it as false. Written down with the property that makes it matter,
    both read the table the same way and checkpoints keep their statistics.
    """
    if properties is None:
        return None
    out = dict(properties)
    if (
        str(out.get(_STATS_JSON, "")).lower() == "false"
        and _STATS_STRUCT not in out
        and (existing or {}).get(_STATS_STRUCT) is None
    ):
        out[_STATS_STRUCT] = "true"
    return out


_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1 << 10,
    "kb": 1 << 10,
    "m": 1 << 20,
    "mb": 1 << 20,
    "g": 1 << 30,
    "gb": 1 << 30,
    "t": 1 << 40,
    "tb": 1 << 40,
    "p": 1 << 50,
    "pb": 1 << 50,
}


def parse_byte_size(value: object) -> int | None:
    """A Spark byte string ("134217728", "128mb", "1g") in bytes; None if it is not one.

    Databricks reads delta.targetFileSize this way (binary units, case-blind).
    delta-rs parses only a bare integer and silently falls back to its own
    default for anything else, so "2mb" was accepted and then ignored.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    digits = len(text) - len(text.lstrip("0123456789"))
    number, unit = text[:digits], text[digits:].strip()
    if not number or not number.isascii() or unit not in _BYTE_UNITS:
        return None
    size = int(number) * _BYTE_UNITS[unit]
    return size if size > 0 else None


def property_support(key: str) -> PropertySupport:
    """Look up a property, applying the prefix rules for keys with no row.

    Three rules cover everything not listed explicitly:

    * ``delta.feature.<name>`` -- delta-rs rejects all of them. The kernel
      accepts the names in `KERNEL_CREATE_FEATURES`. Note `clustering` is
      excluded: the kernel wants clustering columns through its
      data layout, not through a feature signal.
    * any other unknown ``delta.*`` key -- delta-rs rejects it; the kernel
      rejects it too, since its allow-list is closed.
    * a custom key outside the ``delta.`` namespace -- delta-rs rejects it,
      which surprises people; the kernel stores it verbatim.
    """
    known = PROPERTY_SUPPORT.get(key)
    if known is not None:
        return known

    if key.startswith(FEATURE_SIGNAL_PREFIX):
        name = key[len(FEATURE_SIGNAL_PREFIX) :]
        feature = feature_from_wire(name)
        kernel = (
            PropertyEffect.HONORED
            if feature is not None and feature in KERNEL_CREATE_FEATURES
            else PropertyEffect.REJECTED
        )
        note = "delta-rs rejects every delta.feature.* signal"
        if name == "clustering":
            note += (
                "; the kernel rejects this one too -- pass cluster_by= instead, "
                "which sets clustering through its data layout"
            )
        return PropertySupport(key, _RJ, _RJ, kernel, note=note)

    if key.startswith("delta."):
        return PropertySupport(
            key, _RJ, _RJ, _RJ, note="unrecognized delta.* key; both allow-lists are closed"
        )

    return PropertySupport(
        key,
        _RJ,
        _RJ,
        _ST,
        note="a custom key outside the delta. namespace: delta-rs rejects it, the kernel stores it",
    )


#: Operations that create a table, as opposed to altering one.
_CREATE_OPS = frozenset({Operation.CREATE})


_BOOL_KEYS = frozenset(
    {
        "delta.appendOnly",
        "delta.enableChangeDataFeed",
        "delta.enableDeletionVectors",
        "delta.checkpoint.writeStatsAsStruct",
        "delta.checkpoint.writeStatsAsJson",
        "delta.enableExpiredLogCleanup",
        "delta.tuneFileSizesForRewrites",
        "delta.autoOptimize.optimizeWrite",
        "delta.autoOptimize.autoCompact",
        "delta.randomizeFilePrefixes",
        "delta.enableRowTracking",
        "delta.enableInCommitTimestamps",
        "delta.enableTypeWidening",
        "delta.enableIcebergCompatV2",
        "delta.enableIcebergCompatV3",
    }
)
# Not delta.targetFileSize: Databricks also accepts byte strings such as "128mb".
_POSITIVE_INT_KEYS = frozenset({"delta.checkpointInterval"})
_VERSION_KEYS = frozenset({"delta.minReaderVersion", "delta.minWriterVersion"})
_INTERVAL_KEYS = frozenset(
    {
        "delta.logRetentionDuration",
        "delta.deletedFileRetentionDuration",
        "delta.setTransactionRetentionDuration",
    }
)
_INTERVAL_UNITS = frozenset(
    {
        "nanosecond",
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
    }
)
_ENUM_VALUES: dict[str, frozenset[str]] = {
    "delta.columnMapping.mode": frozenset({"none", "name", "id"}),
    "delta.checkpointPolicy": frozenset({"classic", "v2"}),
}


def _value_problem(key: str, value: object) -> str | None:
    """Why `value` is not a valid value for `key`, as the engines parse it; else None.

    The engines' parsers are strict -- `"True"` is not a boolean to either --
    and they fail in the worst ways: delta-rs with an opaque "Error parsing
    property", the kernel by silently filing the value under unknown
    properties, so `delta.appendOnly = "True"` is stored and never enforced.
    """
    if value is None:
        # delta-rs's configuration mapping allows None; leave it to the engine.
        return None
    if not isinstance(value, str):
        return f"the value must be a string, not {type(value).__name__} ({value!r})"
    if key in _BOOL_KEYS:
        if value not in ("true", "false"):
            return f"{value!r} is not 'true' or 'false' (lowercase)"
    elif key in _POSITIVE_INT_KEYS:
        if not (value.isascii() and value.isdigit() and int(value) > 0):
            return f"{value!r} is not a positive integer"
    elif key == "delta.targetFileSize":
        if parse_byte_size(value) is None:
            return f"{value!r} is not a positive size in bytes, such as '134217728' or '128mb'"
    elif key in _VERSION_KEYS:
        if not (value.isascii() and value.isdigit()):
            return f"{value!r} is not a protocol version number"
    elif key == "delta.dataSkippingNumIndexedCols":
        # Python's int() also takes " 5" and "5_0"; Rust's i64 parser does not.
        body = value[1:] if value[:1] in ("+", "-") else value
        number = int(value) if body.isascii() and body.isdigit() else None
        if number is None or number < -1:
            return f"{value!r} is not an integer >= -1"
    elif key in _INTERVAL_KEYS:
        words = value.split()
        if (
            len(words) != 3
            or words[0] != "interval"
            or not (words[1].isascii() and words[1].isdigit())
            or words[2].removesuffix("s") not in _INTERVAL_UNITS
        ):
            return (
                f"{value!r} is not an interval such as 'interval 30 days' "
                "(units up to weeks; months and years are not supported)"
            )
    elif key in _ENUM_VALUES and value not in _ENUM_VALUES[key]:
        return f"{value!r} is not one of {', '.join(sorted(_ENUM_VALUES[key]))}"
    return None


def _spelling_hint(key: str) -> str:
    """For an unknown key that differs from a known one only by case, the right spelling."""
    if key in PROPERTY_SUPPORT:
        return ""
    lowered = key.lower()
    for known in PROPERTY_SUPPORT:
        if known.lower() == lowered:
            return f"; property keys are case-sensitive here -- did you mean {known}?"
    return ""


def effect_for(key: str, engine: Engine, operation: Operation) -> PropertyEffect:
    """What `engine` will do with `key` for this operation."""
    row = property_support(key)
    if engine is Engine.KERNEL:
        # The kernel only validates properties at create; there is no ALTER path.
        return row.kernel_create if operation in _CREATE_OPS else PropertyEffect.UNSUPPORTED
    if engine is Engine.DELTARS:
        return row.deltars_create if operation in _CREATE_OPS else row.deltars_set
    # A SQL warehouse is Databricks itself, so it accepts the whole spec.
    return PropertyEffect.HONORED


def engine_can_set(properties: Mapping[str, str], engine: Engine, operation: Operation) -> bool:
    """True if `engine` can handle every property without raising."""
    return all(effect_for(key, engine, operation).usable() for key in properties)


def validate_properties(
    properties: Mapping[str, str] | None,
    engine: Engine,
    operation: Operation,
    *,
    warn: bool = True,
) -> None:
    """Raise if `engine` cannot handle these properties; warn about inert ones.

    Groups every problem into one message rather than failing on the first key,
    because a caller passing six properties wants to know about all six.
    """
    if not properties:
        return

    rejected: list[str] = []
    crashing: list[str] = []
    unsupported: list[str] = []
    inert: list[str] = []

    invalid: list[str] = []

    for key in properties:
        row = property_support(key)
        effect = effect_for(key, engine, operation)
        detail = f"{key}" + (f" ({row.note})" if row.note else "") + _spelling_hint(key)
        if engine in (Engine.DELTARS, Engine.KERNEL) and effect.usable():
            problem = _value_problem(key, properties[key])
            if problem is not None:
                invalid.append(f"{key}: {problem}")
        if effect is PropertyEffect.REJECTED:
            rejected.append(detail)
        elif effect is PropertyEffect.CRASH:
            crashing.append(detail)
        elif effect is PropertyEffect.UNSUPPORTED:
            unsupported.append(detail)
        elif effect is PropertyEffect.STORED:
            inert.append(detail)

    if crashing or rejected or unsupported or invalid:
        problems: list[str] = []
        if invalid:
            problems.append(
                "invalid values (the engine would store them but never act on them): "
                + "; ".join(invalid)
            )
        if crashing:
            problems.append(
                "would crash the engine (a Rust panic, not a catchable error): "
                + "; ".join(crashing)
            )
        if rejected:
            problems.append(f"{engine.value} rejects: " + "; ".join(rejected))
        if unsupported:
            problems.append(f"{engine.value} has no path for: " + "; ".join(unsupported))

        other = Engine.KERNEL if engine is Engine.DELTARS else Engine.DELTARS
        remedy = ""
        if not (crashing or rejected or unsupported):
            remedy = (
                "write each value exactly as Delta parses it: lowercase 'true'/'false', "
                "a plain integer, or 'interval <n> <unit>'"
            )
        elif operation in _CREATE_OPS and all(
            effect_for(k, other, operation).usable() for k in properties
        ):
            remedy = (
                f"the {other.value} engine accepts all of these; creating without the "
                f"properties {engine.value} refuses"
                + (", or passing cluster_by=, routes there" if other is Engine.KERNEL else "")
            )
        else:
            remedy = "drop the property, or apply it from Databricks via the SQL fallback"

        raise PropertyNotSupportedError(
            f"set these table properties with {engine.value}",
            "; ".join(problems),
            remedy,
        )

    listed = properties.get("delta.dataSkippingStatsColumns")
    mapped = str(properties.get("delta.columnMapping.mode", "none")).lower() not in ("", "none")
    if warn and engine is Engine.DELTARS and listed and (mapped or "." in str(listed)):
        # Appends and overwrites of such a table go to the kernel when it can
        # write the table; delta-rs's own writes (UPDATE, MERGE, OPTIMIZE, and
        # every write to a column-mapped table it created) still drop these.
        warnings.warn(
            "delta-rs writes no statistics for the nested fields in, or (under column "
            "mapping) any of, delta.dataSkippingStatsColumns; only kernel writes honor them",
            IgnoredPropertyWarning,
            stacklevel=3,
        )
    if warn and inert:
        warnings.warn(
            f"{engine.value} stores but does not act on: " + "; ".join(inert),
            IgnoredPropertyWarning,
            stacklevel=3,
        )
