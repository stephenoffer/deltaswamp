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
from dataclasses import dataclass, replace
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

        return pa.table(self.stream(splits))

    def stream(self, splits: Iterable[Any] | None = None) -> Any:
        """Read some (default: all) of the planned splits as an Arrow stream."""
        chosen = list(self.splits if splits is None else splits)
        return self.engine.execute_scan(
            self.table,
            chosen,
            columns=list(self.columns) if self.columns is not None else None,
            predicate=self.predicate,
        )

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

        def get_read_tasks(
            self, parallelism: int, per_task_row_limit: int | None = None, **_: Any
        ) -> list[Any]:
            # An empty plan still gets one task, so the dataset has a schema
            # rather than none at all.
            groups = self._plan.partitions(parallelism) or [()]
            tasks = []
            for group in groups:
                # Each task carries only its own splits: closing over the whole
                # plan shipped every split to every task.
                plan = replace(self._plan, splits=tuple(group)) if group else self._plan

                def read(plan: ScanPlan = plan, empty: bool = not group) -> Iterator[Any]:
                    import pyarrow as pa

                    # Batch by batch: materialising a whole group first held
                    # every file of the task in memory at once.
                    # Through the engine, not a newer ScanPlan method: the plan
                    # is unpickled against whatever release the worker has.
                    stream = plan.engine.execute_scan(
                        plan.table,
                        [] if empty else list(plan.splits),
                        columns=list(plan.columns) if plan.columns is not None else None,
                        predicate=plan.predicate,
                    )
                    reader = pa.RecordBatchReader.from_stream(stream)
                    produced = False
                    for batch in reader:
                        if batch.num_rows:
                            produced = True
                            yield pa.Table.from_batches([batch], schema=reader.schema)
                    if not produced:
                        yield reader.schema.empty_table()

                metadata = BlockMetadata(
                    num_rows=None,
                    size_bytes=sum(s.size for s in group),
                    exec_stats=None,
                    input_files=tuple(s.path for s in group),
                )
                # Ray slices each task's output to the limit a downstream
                # `limit()` pushed into the read; older Ray has no such argument.
                limit = (
                    {} if per_task_row_limit is None else {"per_task_row_limit": per_task_row_limit}
                )
                tasks.append(ReadTask(read, metadata, **limit))
            return tasks

    return _Datasource(plan)
