"""Distributed reads and writes: plan on the driver, do the work on workers.

A scan plan is a list of `ScanSplit`s pinned to one snapshot version. A worker
receives the engine, the resolved table (with its credential provider, never a
credential) and its splits, re-resolves that exact version and vends its own
storage credentials. The Ray Data datasource makes one read task per
byte-balanced group of splits.

Writes run in reverse. `WritePlan` checks on the driver that the commit can
succeed before any worker runs, workers write data files and return opaque
fragments, and the driver commits every fragment in one transaction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, cast

__all__ = ["DeltaSwampDatasource", "ScanPlan", "WritePlan", "balance"]


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


@dataclass(frozen=True)
class WritePlan:
    """A validated distributed write: workers produce files, the driver commits.

    Built by `Table.plan_write()`, which refuses up front if the table cannot
    accept the write, so a job never runs against a table that will reject it.

    Picklable by the same rule as `ScanPlan`: it carries the engine, the
    resolved table and its *credential provider*, never a credential. A worker
    vends its own, so nothing secret enters a task payload and a long job does
    not die on a token frozen at submission time.
    """

    engine: Any
    table: Any
    mode: str = "append"
    version: int | None = None
    txn: tuple[str, int] | None = None
    commit_metadata: dict[str, Any] | None = None

    #: Retries an ordinary append gets when `retries` is not given. Concurrent
    #: jobs really do collide -- four committing at once leaves one winner and
    #: three `CommitConflictError`s -- and rebasing an append is always correct,
    #: so the default matches `KernelEngine.metadata_commit_attempts` rather
    #: than leaving every connector to write the same loop. An overwrite gets
    #: none, and so does a catalog-managed table, which cannot rebase here.
    default_append_retries: ClassVar[int] = 5

    @property
    def overwrite(self) -> bool:
        return self.mode == "overwrite"

    def write(self, data: Any) -> bytes:
        """Worker side: write `data` as files, returning a fragment to send back.

        The files are durable when this returns but belong to no version yet.
        Every fragment must reach `commit()` or the files are orphaned.
        """
        result: bytes = self.engine.write_files(self.table, data)
        return result

    def commit(
        self,
        fragments: Iterable[bytes],
        *,
        operation: str = "WRITE",
        retries: int | None = None,
        allow_concurrent_overwrite: bool = False,
    ) -> int:
        """Driver side: commit every fragment as one transaction.

        Returns the committed version. The commit resolves the table again, so
        an append lands on top of whatever else has been written since the plan
        was made rather than failing on it, which is what an append means.

        It still has to win the race for its version, and concurrent jobs
        do collide: four committing at once leaves one winner and three
        conflicts. Rebasing an append is always correct, so `retries` defaults
        to `default_append_retries` and the losers simply commit at the next
        version. Pass `retries=0` to see the conflict instead.

        An overwrite is the opposite: it removes what it finds, so committing
        against a table that has moved on would discard a writer that arrived
        after planning. That is refused unless `allow_concurrent_overwrite` says
        the last writer should win.

        On a catalog-managed table the catalog arbitrates and can still reject
        the commit outright. The table is untouched when it does, and the
        fragments stay valid: they describe data files, which carry no version,
        so the same fragments can be committed again against a fresh snapshot.
        `retries` re-attempts that here for tables this library commits itself;
        a catalog-managed table has to be re-opened through its catalog first,
        and says so rather than spinning against a stale commit tail. An
        overwrite defaults to no retries, because retrying one means overwriting
        the writer that just won.
        """
        from .errors import CommitConflictError, UnreachableTableError

        collected = list(fragments)
        if retries is None:
            # Only an ordinary append gets them. Retrying an overwrite means
            # overwriting the writer that just won, and a catalog-managed table
            # cannot rebase here at all -- it would spin against the commit tail
            # captured when it was resolved, so defaulting to a retry that
            # cannot work would only change which error the caller sees.
            retries = (
                0
                if (self.overwrite or self.table.is_catalog_managed)
                else self.default_append_retries
            )
        attempts = max(0, retries) + 1
        last: Exception | None = None
        conflict_version = -1
        for _ in range(attempts):
            # Re-checked every attempt, not once: losing a race means the table
            # moved by definition, so a retry is exactly when an overwrite is
            # most likely to be discarding someone.
            if self.overwrite and not allow_concurrent_overwrite:
                self._refuse_if_the_table_moved()
            try:
                version: int = self.engine.commit_files(
                    self.table,
                    collected,
                    overwrite=self.overwrite,
                    operation=operation,
                    txn=self.txn,
                    commit_metadata=self.commit_metadata,
                )
            except CommitConflictError as exc:
                last = exc
                conflict_version = exc.version
                if attempts == 1:
                    # No retry was asked for, so the conflict is the answer.
                    # Diverting to a different error type here would hide it
                    # from a caller catching CommitConflictError, which is what
                    # every other write path raises.
                    raise
                if self.table.is_catalog_managed:
                    # The ratified tail and the version ceiling were captured
                    # when the table was resolved, so re-committing here would
                    # keep racing against the same stale view.
                    raise UnreachableTableError(
                        "retry the commit",
                        "this table is catalog-managed, and its commit tail was captured "
                        f"when it was resolved, so a retry here would reuse it ({exc})",
                        "re-open the table through the catalog and commit the same "
                        "fragments against the fresh snapshot -- they stay valid",
                    ) from exc
                continue
            return version

        raise CommitConflictError(
            conflict_version,
            f"another writer committed first on each of {attempts} attempts ({last}). "
            "The fragments are still valid: re-open the table and commit them again.",
        )

    def _refuse_if_the_table_moved(self) -> None:
        """Refuse an overwrite whose planned snapshot is no longer current.

        The commit removes every file it can see, so if the table advanced
        between planning and committing, those rows would go too -- silently,
        and with nothing in the log to say a writer was lost.
        """
        from .errors import UnreachableTableError

        if self.version is None:
            return
        current = self.engine.detail(self.table).get("version")
        if current is not None and current != self.version:
            raise UnreachableTableError(
                "commit this overwrite",
                f"the table was at version {self.version} when the write was planned and "
                f"is at {current} now. An overwrite removes what it finds, so committing "
                "would discard whatever was written in between",
                "re-plan against the current version, or pass "
                "allow_concurrent_overwrite=True to let the last writer win",
            )


def balance(splits: Iterable[Any], n: int) -> list[tuple[Any, ...]]:
    """Group splits into at most `n` bins of roughly equal bytes.

    Largest-first greedy assignment: good enough to keep one huge file from
    serializing the job, and deterministic, so a retried task reads the same
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
