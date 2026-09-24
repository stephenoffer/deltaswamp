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
    Engine,
    Operation,
    PropertyEffect,
    property_support,
)
from .errors import IgnoredPropertyWarning, PropertyNotSupportedError

__all__ = ["effect_for", "engine_can_set", "validate_properties"]

#: Operations that create a table, as opposed to altering one.
_CREATE_OPS = frozenset({Operation.CREATE})


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

    for key in properties:
        row = property_support(key)
        effect = effect_for(key, engine, operation)
        detail = f"{key}" + (f" ({row.note})" if row.note else "")
        if effect is PropertyEffect.REJECTED:
            rejected.append(detail)
        elif effect is PropertyEffect.CRASH:
            crashing.append(detail)
        elif effect is PropertyEffect.UNSUPPORTED:
            unsupported.append(detail)
        elif effect is PropertyEffect.STORED:
            inert.append(detail)

    if crashing or rejected or unsupported:
        problems: list[str] = []
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
        if operation in _CREATE_OPS and all(
            effect_for(k, other, operation).usable() for k in properties
        ):
            remedy = (
                f"the {other.value} engine accepts all of these; creating without the "
                "properties delta-rs rejects, or passing cluster_by=, routes there"
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
