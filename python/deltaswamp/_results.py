"""One result shape per operation, whichever engine served it.

Each engine reported its own metrics: delta-rs ``num_target_rows_updated``
for a MERGE, the kernel ``num_updated_rows``, the warehouse a nested
``metrics`` struct for OPTIMIZE; ``r["num_updated_rows"]`` was a KeyError on
some engines only. A result keeps every key its engine gave (code written
against one engine keeps working) and adds the same normalized keys on all
of them, plus the engine that served the call.
"""

from __future__ import annotations

import json
from typing import Any


class OperationResult(dict[str, Any]):
    """The engine's metrics, plus normalized keys, as a dict.

    `engine` names the engine that served the call; `raw` is what it
    returned, untouched.
    """

    engine: str | None
    raw: Any

    def __init__(self, raw: Any, engine: Any, normalized: dict[str, Any]) -> None:
        super().__init__(raw if isinstance(raw, dict) else {})
        for key, value in normalized.items():
            if value is not None:
                self.setdefault(key, value)
        self.engine = None if engine is None else str(getattr(engine, "value", engine))
        self.raw = raw

    def __repr__(self) -> str:
        return f"OperationResult(engine={self.engine!r}, {dict.__repr__(self)})"


def _first(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return value
    return None


def _either(outer: dict[str, Any], inner: dict[str, Any], key: str) -> Any:
    """`key` from the flat result, else from a nested metrics struct (0 counts)."""
    value = outer.get(key)
    return inner.get(key) if value is None else value


def _int(value: Any) -> int | None:
    try:
        return None if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return None


def dml(raw: Any, engine: Any) -> OperationResult:
    """DELETE, UPDATE and MERGE: rows deleted, updated, inserted; files; version."""
    r = raw if isinstance(raw, dict) else {}
    deleted = _int(_first(r, "num_deleted_rows", "num_target_rows_deleted"))
    updated = _int(_first(r, "num_updated_rows", "num_target_rows_updated"))
    inserted = _int(_first(r, "num_inserted_rows", "num_target_rows_inserted"))
    affected = _int(r.get("num_affected_rows"))
    if affected is None and any(v is not None for v in (deleted, updated, inserted)):
        affected = (deleted or 0) + (updated or 0) + (inserted or 0)
    return OperationResult(
        raw,
        engine,
        {
            "num_deleted_rows": deleted,
            "num_updated_rows": updated,
            "num_inserted_rows": inserted,
            "num_affected_rows": affected,
            "num_files_added": _int(_first(r, "num_added_files", "num_target_files_added")),
            "num_files_removed": _int(_first(r, "num_removed_files", "num_target_files_removed")),
            "version": _int(r.get("version")),
        },
    )


def optimize(raw: Any, engine: Any) -> OperationResult:
    """OPTIMIZE and Z-ORDER: files added and removed."""
    # delta-rs hands its compactions to the kernel, and the result says so.
    engine = getattr(raw, "served_by", engine)
    r = raw if isinstance(raw, dict) else {}
    metrics = r.get("metrics")
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except ValueError:
            metrics = None
    m = metrics if isinstance(metrics, dict) else {}
    return OperationResult(
        raw,
        engine,
        {
            "num_files_added": _int(_either(r, m, "numFilesAdded")),
            "num_files_removed": _int(_either(r, m, "numFilesRemoved")),
        },
    )


def restore(raw: Any, engine: Any) -> OperationResult:
    """RESTORE: files removed and restored."""
    r = raw if isinstance(raw, dict) else {}
    return OperationResult(
        raw,
        engine,
        {
            "num_removed_files": _int(_first(r, "num_removed_files", "numRemovedFile")),
            "num_restored_files": _int(_first(r, "num_restored_files", "numRestoredFile")),
        },
    )
