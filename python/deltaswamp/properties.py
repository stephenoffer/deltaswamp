"""Validating table properties before an engine sees them.

delta-rs reports every property problem with one message -- "Kernel: Generic
delta kernel error: Error parsing property" -- and panics on
`delta.minReaderVersion`. Neither tells you which key was wrong or what to do.
Checking against `PROPERTY_SUPPORT` first turns both into an error that names
the key, the engine, and the way forward.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

from .capability import (
    PROPERTY_SUPPORT,
    Engine,
    Operation,
    PropertyEffect,
    property_support,
)
from .errors import IgnoredPropertyWarning, PropertyNotSupportedError

__all__ = ["effect_for", "engine_can_set", "validate_properties"]

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

    if warn and inert:
        warnings.warn(
            f"{engine.value} stores but does not act on: " + "; ".join(inert),
            IgnoredPropertyWarning,
            stacklevel=3,
        )
