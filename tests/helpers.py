"""Builders shared across the suite.

Nothing here imports the native extension, pyarrow or deltalake at module
level, so the pure-Python unit tests can use it with only pytest installed.
"""

from __future__ import annotations

from typing import Any

from deltaswamp.capability import Capability, Engine, Operation
from deltaswamp.catalog import ResolvedTable
from deltaswamp.identity import TableRef, parse_ref
from deltaswamp.router import Router

#: A Delta schema string with a partition-friendly column, for register calls.
DELTA_SCHEMA: dict[str, Any] = {
    "type": "struct",
    "fields": [
        {"name": "id", "type": "long", "nullable": False, "metadata": {"comment": "key"}},
        {"name": "amount", "type": "decimal(10,2)", "nullable": True, "metadata": {}},
        {
            "name": "tags",
            "type": {"type": "array", "elementType": "string", "containsNull": True},
            "nullable": True,
            "metadata": {},
        },
        {"name": "region", "type": "string", "nullable": True, "metadata": {}},
    ],
}


def resolved_table(ref: str | TableRef = "main.sales.orders", **fields: Any) -> ResolvedTable:
    """A `ResolvedTable` with a storage location, overridable field by field."""
    fields.setdefault("location", "s3://bucket/t")
    return ResolvedTable(ref=parse_ref(ref) if isinstance(ref, str) else ref, **fields)


class FakeEngine:
    """An engine that says yes to a fixed set of operations (all of them by default)."""

    supports_distributed_scan = False
    supports_predicates = True  # the router reads `supports_<need>` for each request need

    def __init__(self, kind: Engine, yes: set[Operation] | None = None) -> None:
        self.kind = kind
        self._yes = yes if yes is not None else set(Operation)

    def supports(self, operation: Operation, table: ResolvedTable, **_: Any) -> Capability:
        if operation in self._yes:
            return Capability(operation, ok=True, engine=self.kind)
        return Capability(operation, ok=False, reason=f"{self.kind.value} declines")

    def available(self) -> bool:
        return True


def accepting_router(*, fallback: bool) -> Router:
    """Kernel, delta-rs and SQL all accepting, so only the router's own gates decide."""
    kinds = (Engine.KERNEL, Engine.DELTARS, Engine.SQL)
    return Router(engines={k: FakeEngine(k) for k in kinds}, allow_sql_fallback=fallback)


def direct_router() -> Router:
    """The real kernel and delta-rs engines, with no warehouse."""
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine

    return Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()})


def native() -> Any:
    """The compiled extension module. Callers skip when `has_native()` is false."""
    from deltaswamp import _native

    return _native


def snapshot(path: str, **kwargs: Any) -> Any:
    """A kernel snapshot of the table at `path`."""
    return native().Snapshot.resolve(path, **kwargs)
