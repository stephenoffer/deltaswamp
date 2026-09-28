"""Regressions for the round-7 live interop findings, reproduced on local tables."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


# ----------------------------------------------------------------- helpers


def _kernel_only() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.KERNEL: KernelEngine()})
    )


def _deltars_only() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(), router=Router(engines={Engine.DELTARS: DeltaRsEngine()})
    )


def _adds(path: str) -> list[dict[str, Any]]:
    """The add actions of the newest commit."""
    log = os.path.join(path, "_delta_log")
    last = sorted(n for n in os.listdir(log) if n.endswith(".json"))[-1]
    with open(os.path.join(log, last)) as f:
        return [json.loads(line)["add"] for line in f if '"add"' in line]


# --------------------------------------------------- 1: DV stats precision


class TestDeletionVectorStatsKeepDecimals:
    """The kernel's DV update re-serialised each add's stats through f64."""

    MAX = "12345678901234567890.123456789012345678"
    D0 = "9" * 38

    def _table(self, tmp_path: Any) -> str:
        import decimal

        d = decimal.Decimal
        path = str(tmp_path / "dv")
        conn = _kernel_only()
        conn.create_table(
            path,
            pa.schema(
                [("id", pa.int64()), ("dec", pa.decimal128(38, 18)), ("d0", pa.decimal128(38, 0))]
            ),
            properties={"delta.enableDeletionVectors": "true"},
        )
        conn.open_table(path).append(
            pa.table(
                {
                    "id": pa.array([1, 2, 3], pa.int64()),
                    "dec": pa.array(
                        [d(self.MAX), d("0.1"), d("-" + self.MAX)], pa.decimal128(38, 18)
                    ),
                    "d0": pa.array([d(self.D0), d(1), d("-" + self.D0)], pa.decimal128(38, 0)),
                }
            )
        )
        return path

    @pytest.mark.parametrize("op", ["delete", "update", "merge"])
    def test_the_rewritten_add_keeps_exact_bounds(self, tmp_path: Any, op: str) -> None:
        path = self._table(tmp_path)
        t = ds.connect("file://").open_table(path)
        if op == "delete":
            t.delete("id = 2")
        elif op == "update":
            t.update({"id": "id"}, predicate="id = 2")
        else:
            (
                t.merge(pa.table({"id": pa.array([2], pa.int64())}), "target.id = source.id")
                .when_matched_delete()
                .execute()
            )
        dv_adds = [a for a in _adds(path) if a.get("deletionVector")]
        assert dv_adds, _adds(path)
        stats = dv_adds[0]["stats"]
        assert f'"dec":{self.MAX}' in stats and f'"dec":-{self.MAX}' in stats, stats
        assert f'"d0":{self.D0}' in stats and f'"d0":-{self.D0}' in stats, stats
        assert '"tightBounds":false' in stats
        assert "e+" not in stats
