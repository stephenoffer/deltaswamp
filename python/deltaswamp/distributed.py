"""Distributed reads: plan once on the driver, read splits on workers.

A plan is a list of `ScanSplit`s pinned to one snapshot version. What travels
to a worker is the engine, the resolved table (with its *credential provider*,
never a credential) and the splits. The worker re-resolves that exact version
-- including a catalog-managed table's commit tail, which was captured at
resolution -- and reads only its files, vending its own storage credentials.
So a long job does not die on a token frozen at submission time, and no secret
appears in a task payload.

The Ray Data datasource is a thin layer over that: one read task per group of
splits, balanced by bytes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

__all__ = ["DeltaSwampDatasource", "ScanPlan", "balance"]


@dataclass(frozen=True)
class ScanPlan:
    """Everything a worker needs to read part of one snapshot."""

    engine: Any
    table: Any
    splits: tuple[Any, ...]
    columns: tuple[str, ...] | None = None
    predicate: str | None = None

    @property
    def version(self) -> int | None:
        return self.splits[0].commit_version if self.splits else None

    @property
    def total_bytes(self) -> int:
        return sum(s.size for s in self.splits)

    def read(self, splits: Iterable[Any] | None = None) -> Any:
        """Read some (default: all) of the planned splits as a pyarrow Table."""
        import pyarrow as pa

        chosen = list(self.splits if splits is None else splits)
        stream = self.engine.execute_scan(
            self.table,
            chosen,
            columns=list(self.columns) if self.columns is not None else None,
            predicate=self.predicate,
        )
        return pa.table(stream)

    def partitions(self, n: int) -> list[tuple[Any, ...]]:
        return balance(self.splits, n)


def balance(splits: Iterable[Any], n: int) -> list[tuple[Any, ...]]:
    """Group splits into at most `n` bins of roughly equal bytes.

    Largest-first greedy assignment: good enough to keep one huge file from
    serialising the job, and deterministic, so a retried task reads the same
    files.
    """
    items = sorted(splits, key=lambda s: (-s.size, s.path))
    if not items:
        return []
    count = max(1, min(n, len(items)))
    bins: list[list[Any]] = [[] for _ in range(count)]
    loads = [0] * count
    for split in items:
        target = loads.index(min(loads))
        bins[target].append(split)
        loads[target] += split.size
    return [tuple(b) for b in bins if b]


def _datasource_base() -> type:
    from ray.data import Datasource

    # cast, not an ignore: Ray is absent in the lint environment (where this is
    # Any) and present in the test one (where it is a class), and only a cast is
    # correct under both.
    return cast(type, Datasource)


def DeltaSwampDatasource(plan: ScanPlan) -> Any:
    """A Ray Data `Datasource` over a scan plan.

    Built lazily so importing this module does not import Ray.
    """
    from ray.data import ReadTask
    from ray.data.block import BlockMetadata

    base = _datasource_base()

    class _Datasource(base):  # type: ignore[misc,valid-type]
        def __init__(self, plan: ScanPlan) -> None:
            self._plan = plan

        def get_name(self) -> str:
            return "deltaswamp"

        def estimate_inmemory_data_size(self) -> int | None:
            return self._plan.total_bytes or None

        def get_read_tasks(self, parallelism: int, **_: Any) -> list[Any]:
            tasks = []
            for group in self._plan.partitions(parallelism):
                plan = self._plan

                def read(group: tuple[Any, ...] = group, plan: ScanPlan = plan) -> Iterator[Any]:
                    yield plan.read(group)

                metadata = BlockMetadata(
                    num_rows=None,
                    size_bytes=sum(s.size for s in group),
                    exec_stats=None,
                    input_files=tuple(s.path for s in group),
                )
                tasks.append(ReadTask(read, metadata))
            return tasks

    return _Datasource(plan)
