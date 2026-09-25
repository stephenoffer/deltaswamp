"""Distributed reads and writes: plan on the driver, do the work on workers.

A plan is a list of `ScanSplit`s pinned to one snapshot version. What travels
to a worker is the engine, the resolved table (with its *credential provider*,
never a credential) and the splits. The worker re-resolves that exact version
-- including a catalog-managed table's commit tail, which was captured at
resolution -- and reads only its files, vending its own storage credentials.
So a long job does not die on a token frozen at submission time, and no secret
appears in a task payload.

The Ray Data datasource is a thin layer over that: one read task per group of
splits, balanced by bytes.

Writes run the same shape in reverse. `WritePlan` settles on the driver whether
the commit can succeed *before* any worker runs, each worker writes data files
and returns an opaque fragment, and the driver commits every fragment as one
transaction. That ordering is the point: the common failure in distributed Delta
writers is discovering at commit time that the table refuses the write, after an
hour of compute, leaving orphaned Parquet behind. Here the refusal arrives
before the first byte is written, carrying the reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, cast

__all__ = ["DeltaSwampDatasource", "ScanPlan", "WritePlan", "balance"]


@dataclass(frozen=True)
class ScanPlan:
    """Everything a worker needs to read part of one snapshot."""

    engine: Any
    table: Any
    splits: tuple[Any, ...]
    columns: tuple[str, ...] | None = None
    predicate: str | None = None
    # The planned snapshot's version. The splits carry it too, but a plan of
    # an empty snapshot has no splits, and reading one then resolved the
    # *latest* version -- another schema, if the table evolved since.
    snapshot_version: int | None = None
    #: Ship the catalog's own credential provider (and with it the catalog
    #: token) to workers, so they can re-vend. Off by default: see _for_workers.
    ship_catalog_auth: bool = False

    @property
    def version(self) -> int | None:
        return self.splits[0].commit_version if self.splits else self.snapshot_version

    def __reduce__(self) -> tuple[Any, ...]:
        # What crosses a process boundary: see `_for_workers`.
        fields = {f: getattr(self, f) for f in self.__dataclass_fields__}
        if not self.ship_catalog_auth:
            fields["table"] = _for_workers(self.table, write=False)
        return (_rebuild, (type(self), fields))

    @property
    def total_bytes(self) -> int:
        return sum(s.size for s in self.splits)

    def read(self, splits: Iterable[Any] | None = None) -> Any:
        """Read some (default: all) of the planned splits as a pyarrow Table."""
        import pyarrow as pa

        from .engine.base import TranslatingStream

        stream = self.stream(splits)
        # read_all() keeps the typed missing-file error pa.table() would flatten.
        return stream.read_all() if isinstance(stream, TranslatingStream) else pa.table(stream)

    def stream(self, splits: Iterable[Any] | None = None) -> Any:
        """Read some (default: all) of the planned splits as an Arrow stream."""
        chosen = list(self.splits if splits is None else splits)
        if splits is not None:
            # A split names its file relative to its own table, so one from
            # another plan read *this* table's root: a missing file, or rows
            # from whatever file happened to share the name -- and an empty
            # result with no error when none did.
            planned = {s.path for s in self.splits}
            stray = [s.path for s in chosen if getattr(s, "path", None) not in planned]
            if stray:
                from .errors import InvalidArgumentError

                raise InvalidArgumentError(
                    f"{len(stray)} split(s) are not part of this plan (e.g. {stray[0]!r}); "
                    "read splits only through the plan that produced them"
                )
        pinned = self.snapshot_version
        extra = {"version": pinned} if not chosen and pinned is not None else {}
        stream = self.engine.execute_scan(
            self.table,
            chosen,
            columns=list(self.columns) if self.columns is not None else None,
            predicate=self.predicate,
            **extra,
        )
        from .engine.base import translating_stream

        # A split whose file was vacuumed since planning failed as a bare
        # OSError; name the file instead.
        where = getattr(self.table, "location", None) or "the table"
        at = f" at version {self.version}" if self.version is not None else ""
        return translating_stream(stream, f"{where}{at}")

    def partitions(self, n: int) -> list[tuple[Any, ...]]:
        return balance(self.splits, n)


@dataclass(frozen=True)
class WritePlan:
    """A validated distributed write: workers produce files, the driver commits.

    Built by `Table.plan_write()`, which refuses up front if the table cannot
    accept the write, so a job never runs against a table that will reject it.

    Picklable like `ScanPlan`: what crosses a process boundary carries the
    table's short-lived storage credential (vended on the driver when the plan
    is pickled) and never the catalog's credentials or the catalog itself --
    unless planned with ``ship_catalog_auth=True``. The driver's own copy keeps
    full catalog access, so `commit()` belongs on the driver.
    """

    engine: Any
    table: Any
    mode: str = "append"
    version: int | None = None
    txn: tuple[str, int] | None = None
    commit_metadata: dict[str, Any] | None = None
    #: The catalog the table was resolved through. A catalog-managed table's
    #: commit tail is re-read from it at commit time: the tail captured at
    #: planning made every commit after anyone else's a guaranteed 409.
    catalog: Any = None
    #: Ship the catalog's credential provider and the catalog itself (and with
    #: them the catalog token) to workers. Off by default: see _for_workers.
    ship_catalog_auth: bool = False

    @property
    def overwrite(self) -> bool:
        return self.mode == "overwrite"

    def __reduce__(self) -> tuple[Any, ...]:
        # What crosses a process boundary: see `_for_workers`. The driver keeps
        # its own copy, with full catalog access, for the commit.
        fields = {f: getattr(self, f) for f in self.__dataclass_fields__}
        if not self.ship_catalog_auth:
            fields["table"] = _for_workers(self.table, write=True)
            fields["catalog"] = None
        return (_rebuild, (type(self), fields))

    def write(self, data: Any) -> bytes:
        """Worker side: write `data` as files, returning a fragment to send back.

        The files are durable when this returns but belong to no version yet.
        Every fragment must reach `commit()` or the files are orphaned.
        """
        from .errors import InvalidArgumentError

        if data is None:
            raise InvalidArgumentError("plan.write() needs data; got None")
        if isinstance(data, list) and data:
            import pyarrow as pa

            # The shapes `Table.append` accepts: row dicts, or record batches
            # (what a Ray block iterator yields). Both failed in pa.table().
            if all(isinstance(row, dict) for row in data):
                data = pa.Table.from_pylist(data)
            elif all(isinstance(b, pa.RecordBatch) for b in data):
                data = pa.Table.from_batches(data)
        result: bytes = self.engine.write_files(self.table, data)
        return result

    def commit(
        self,
        fragments: Iterable[bytes],
        *,
        operation: str | None = "WRITE",
        retries: int | None = None,
        allow_concurrent_overwrite: bool = False,
    ) -> int:
        """Driver side: commit every fragment as one transaction.

        Returns the committed version. The commit resolves the table again, so
        an append lands on top of whatever else has been written since the plan
        was made rather than failing on it -- which is what an append means, and
        why one rarely conflicts here at all.

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
        and says so rather than spinning against a stale commit tail. Left as
        None, an append on a path table retries as `Table.append` does (the
        engine's `append_commit_retries`); anything else does not.
        """
        from .errors import (
            CommitConflictError,
            InvalidArgumentError,
            TransientCommitError,
            UnreachableTableError,
        )

        collected = _fragments_arg(fragments)
        # None was recorded in the log as "UNKNOWN", and "" as a blank
        # operation, in every reader's history.
        if operation is None:
            operation = "WRITE"
        if not isinstance(operation, str) or not operation.strip():
            raise InvalidArgumentError(
                f"operation must be a non-empty string such as 'WRITE', not {operation!r}"
            )
        if retries is None:
            # A blind append commutes with any concurrent commit and its
            # fragments are reusable, so losing a race is no reason to fail.
            # With no retries, five of six racing distributed appends raised
            # CommitConflictError where Table.append would have retried.
            simple = not self.overwrite and not self.table.is_catalog_managed
            default = getattr(self.engine, "append_commit_retries", 0)
            retries = int(default) if simple and isinstance(default, int) else 0
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise InvalidArgumentError(f"retries must be a non-negative int, not {retries!r}")
        attempts = retries + 1
        last: Exception | None = None
        conflict_version = -1
        # A guarded overwrite commits against the planned snapshot itself, not
        # a fresh one: then a writer that lands between the check below and the
        # commit makes the commit conflict, instead of having its files removed.
        pinned = self.version if self.overwrite and not allow_concurrent_overwrite else None
        table = self.table
        for attempt in range(attempts):
            refreshed = self._with_fresh_tail(table)
            table = refreshed if refreshed is not None else table
            if (attempt or not self.overwrite) and self._already_landed(table, collected):
                # A commit reported as failed can still have landed (a put
                # that timed out after it was written), and a driver that
                # restarts may commit its saved fragments a second time.
                # Re-committing added every file again in a new version: the
                # change feed reported the rows twice.
                raise UnreachableTableError(
                    "commit these fragments",
                    "their files are already in the table, so an earlier commit landed "
                    + (f"after all ({last})" if last is not None else "them before"),
                    "check the table's history; do not commit these fragments again",
                )
            # Re-checked every attempt, not once: losing a race means the table
            # moved by definition, so a retry is exactly when an overwrite is
            # most likely to be discarding someone.
            if self.overwrite and not allow_concurrent_overwrite:
                self._refuse_if_the_table_moved(table)
            try:
                version: int = self.engine.commit_files(
                    table,
                    collected,
                    overwrite=self.overwrite,
                    operation=operation,
                    txn=self.txn,
                    commit_metadata=self.commit_metadata,
                    **({"version": pinned} if pinned is not None else {}),
                )
            except TransientCommitError as exc:
                # Nobody won the version and the table is unchanged, so the
                # very same commit can go again -- which `retries` promises.
                # Not on a catalog-managed table: a failed ratification call
                # may have landed, and only the catalog can say.
                last = exc
                if attempts == 1 or self.table.is_catalog_managed:
                    raise
                continue
            except CommitConflictError as exc:
                last = exc
                conflict_version = exc.version
                if attempts == 1:
                    # No retry was asked for, so the conflict is the answer.
                    # Diverting to a different error type here would hide it
                    # from a caller catching CommitConflictError, which is what
                    # every other write path raises.
                    raise
                if self.table.is_catalog_managed and self.catalog is None:
                    # The ratified tail and the version ceiling were captured
                    # when the table was resolved, and with no catalog to
                    # re-read them from, a retry would race the same stale view.
                    raise UnreachableTableError(
                        "retry the commit",
                        "this table is catalog-managed, and its commit tail was captured "
                        f"when it was resolved, so a retry here would reuse it ({exc})",
                        "re-open the table through the catalog and commit the same "
                        "fragments against the fresh snapshot -- they stay valid",
                    ) from exc
                continue
            return version

        if isinstance(last, TransientCommitError):
            raise last
        raise CommitConflictError(
            conflict_version,
            f"another writer committed first on each of {attempts} attempts ({last}). "
            "The fragments are still valid: re-open the table and commit them again.",
        )

    def _already_landed(self, table: Any, fragments: list[bytes]) -> bool:
        """Whether any fragment's data file is already live in the table."""
        from urllib.parse import unquote

        try:
            import pyarrow as pa

            ours: set[str] = set()
            for fragment in fragments:
                if fragment:
                    paths = pa.ipc.open_stream(fragment).read_all().column("path")
                    ours.update(unquote(p) for p in paths.to_pylist() if p)
            if not ours:
                return False
            live = pa.table(self.engine.files(table)).column("path").to_pylist()
        except Exception:
            return False  # cannot tell; commit as before
        return any(p is not None and unquote(p) in ours for p in live)

    def _with_fresh_tail(self, table: Any) -> Any:
        """`table` with its catalog commit tail re-read, or None when not applicable."""
        from .errors import CorruptTableError

        if not getattr(table, "is_catalog_managed", False) or self.catalog is None:
            return None
        fresh = self.catalog.resolve(table.ref)
        if table.table_id and fresh.table_id and table.table_id != fresh.table_id:
            raise CorruptTableError(
                f"{table.ref} was dropped and re-created since this write was planned "
                f"(its table id is now {fresh.table_id!r}); the fragments belong to the old "
                "table's storage, so plan the write again against the new one"
            )
        return replace(
            table, log_tail=fresh.log_tail, max_catalog_version=fresh.max_catalog_version
        )

    def _refuse_if_the_table_moved(self, table: Any = None) -> None:
        """Refuse an overwrite whose planned snapshot is no longer current.

        The commit removes every file it can see, so if the table advanced
        between planning and committing, those rows would go too -- silently,
        and with nothing in the log to say a writer was lost.
        """
        from .errors import UnreachableTableError

        if self.version is None:
            return
        try:
            current = self.engine.detail(table or self.table).get("version")
        except Exception:
            return  # cannot tell; the commit itself will still be atomic
        if current is not None and current != self.version:
            raise UnreachableTableError(
                "commit this overwrite",
                f"the table was at version {self.version} when the write was planned and "
                f"is at {current} now. An overwrite removes what it finds, so committing "
                "would discard whatever was written in between",
                "re-plan against the current version, or pass "
                "allow_concurrent_overwrite=True to let the last writer win",
            )


#: commitInfo keys the kernel writes itself; a caller's value for one was
#: refused by the extension only at commit, after the job had run.
_RESERVED_COMMIT_KEYS = frozenset(
    k.lower()
    for k in (
        "timestamp",
        "inCommitTimestamp",
        "operation",
        "operationParameters",
        "operationMetrics",
        "kernelVersion",
        "isBlindAppend",
        "engineInfo",
        "txnId",
    )
)


def _commit_metadata_arg(metadata: Any) -> dict[str, str] | None:
    """`commit_metadata` for a planned write, validated before any worker runs.

    Values are stringified as the kernel stores them; None became the string
    "None" in the log, so it is refused rather than recorded as data.
    """
    from .errors import InvalidArgumentError

    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise InvalidArgumentError(f"commit_metadata must be a dict, not {type(metadata).__name__}")
    out: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise InvalidArgumentError(f"commit_metadata keys must be non-empty strings: {key!r}")
        if key.lower() in _RESERVED_COMMIT_KEYS:
            raise InvalidArgumentError(
                f"commit_metadata key {key!r} is reserved by the Delta commitInfo action; "
                "use a different key, e.g. userMetadata"
            )
        if value is None:
            raise InvalidArgumentError(
                f"commit_metadata[{key!r}] is None; it would be recorded as the string 'None'"
            )
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, (bool, dict, list, tuple)):
            # str() wrote Python's spelling ("True", "{'a': 1}") into a JSON
            # log, where delta-rs records the same value as true / {"a": 1}.
            import json

            try:
                out[key] = json.dumps(value)
            except (TypeError, ValueError):
                out[key] = str(value)
        else:
            out[key] = str(value)
    return out


def _fragments_arg(fragments: Any) -> list[Any]:
    """The fragments to commit, checked before anything is sent to the log.

    `commit(fragment)` with one bare fragment iterated its bytes as ints and
    failed deep in the binding, and a worker that returned None (or a str)
    surfaced as "Can't extract `str` to `Vec`" with no hint which one.
    """
    from .errors import InvalidArgumentError

    if isinstance(fragments, (bytes, bytearray, memoryview)):
        raise InvalidArgumentError(
            "commit() takes a list of fragments; wrap a single one: commit([fragment])"
        )
    if fragments is None or isinstance(fragments, (str, dict)):
        raise InvalidArgumentError(
            f"commit() takes a list of fragments from plan.write(), not {type(fragments).__name__}"
        )
    collected = list(fragments)
    for index, fragment in enumerate(collected):
        if not isinstance(fragment, (bytes, bytearray, memoryview)):
            raise InvalidArgumentError(
                f"fragment {index} is a {type(fragment).__name__}, not the bytes "
                "plan.write() returns; did that worker fail?"
            )
    return [bytes(f) for f in collected]


def _rebuild(cls: type, fields: dict[str, Any]) -> Any:
    return cls(**fields)


#: A shipped storage credential this close to expiry is refused on the worker.
SHIPPED_CREDENTIAL_MARGIN_SECONDS = 60.0


class ShippedCredentials:
    """The storage credential a plan carries to workers, and nothing else.

    A catalog credential provider pickles its catalog configuration -- for
    Databricks, the SDK Config's attributes, which can include a PAT or a
    client secret; for OSS UC, the bearer token -- and plans travel through
    the Ray object store and task payloads. So a plan ships the short-lived,
    table-scoped *storage* credential vended on the driver instead, with its
    expiry, and no way to reach the catalog. A worker that outlives it gets a
    clear error, not a storage 403.

    For a catalog-managed table the worker still needs the workspace URL to
    build the (never used) committer its write context requires; the token is
    not shipped, so nothing on a worker can commit.
    """

    def __init__(
        self,
        credentials: Any,
        table_id: str | None,
        workspace_url: str | None = None,
    ) -> None:
        self._credentials = credentials
        self._table_id = table_id
        self._workspace_url = workspace_url

    @property
    def table_id(self) -> str | None:
        return self._table_id

    @property
    def expires_at(self) -> float | None:
        return getattr(self._credentials, "expires_at", None)

    def credentials(self, operation: Any = None) -> Any:
        from .credentials import Operation
        from .errors import CredentialError

        wanted = Operation.READ if operation is None else operation
        shipped = getattr(self._credentials, "operation", None)
        if str(wanted).upper() == Operation.READ_WRITE.value and shipped is Operation.READ:
            raise CredentialError(
                "this plan carries a read-only storage credential and cannot write; "
                "plan the write with plan_write() on the driver"
            )
        if self._credentials.expires_within(SHIPPED_CREDENTIAL_MARGIN_SECONDS):
            import time

            left = (self.expires_at or 0) - time.time()
            raise CredentialError(
                "the storage credential shipped with this plan "
                + ("has expired" if left <= 0 else f"expires in {left:.0f}s")
                + ", and a worker cannot re-vend it: plans carry no catalog credentials. "
                "Re-plan on the driver (plan_scan()/plan_write() again), or plan with "
                "ship_catalog_auth=True so workers can refresh it themselves"
            )
        return self._credentials

    def invalidate(self) -> None:
        """Nothing to refresh from here; the next call re-checks expiry."""

    def staging_auth(self) -> tuple[str, str]:
        """`(workspace_url, "")`: enough to build a write context, not to commit."""
        return self._workspace_url or "", ""

    def __repr__(self) -> str:
        return f"ShippedCredentials({self._credentials!r})"


def _for_workers(table: Any, *, write: bool) -> Any:
    """`table` as a worker should receive it: no catalog credentials.

    The provider is replaced by the storage credential it vends now, on the
    driver. A table with no provider (a local path, static options) has
    nothing to strip.
    """
    from .credentials import Operation

    provider = getattr(table, "credential_provider", None)
    if provider is None or isinstance(provider, ShippedCredentials):
        return table
    credentials = provider.credentials(Operation.READ_WRITE if write else Operation.READ)
    workspace_url = None
    if getattr(table, "is_catalog_managed", False) and write:
        auth = getattr(provider, "workspace_auth", None)
        if callable(auth):
            workspace_url = auth()[0]
    shipped = ShippedCredentials(credentials, getattr(provider, "table_id", None), workspace_url)
    return replace(table, credential_provider=shipped)


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


def _absolute(root: str | None, path: str) -> str:
    """A split's file as a full URI: the log names it relative to the table.

    Ray reports these as the dataset's `input_files()`; bare relative names
    identified no file anyone could open.
    """
    from urllib.parse import unquote

    if not root or "://" in path or path.startswith("/"):
        return path
    return root.rstrip("/") + "/" + unquote(path)


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
                    pinned = getattr(plan, "snapshot_version", None)
                    stream = plan.engine.execute_scan(
                        plan.table,
                        [] if empty else list(plan.splits),
                        columns=list(plan.columns) if plan.columns is not None else None,
                        predicate=plan.predicate,
                        **({"version": pinned} if empty and pinned is not None else {}),
                    )
                    try:
                        from deltaswamp.engine.base import TranslatingStream
                    except ImportError:  # an older release on the worker
                        reader: Any = pa.RecordBatchReader.from_stream(stream)
                    else:
                        # A file vacuumed since planning: name it, not OSError.
                        where = getattr(plan.table, "location", None) or "the table"
                        reader = TranslatingStream(stream, f"{where} (a Ray read task)")
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
                    input_files=tuple(
                        _absolute(getattr(self._plan.table, "location", None), s.path)
                        for s in group
                    ),
                )
                # Ray slices each task's output to the limit a downstream
                # `limit()` pushed into the read; older Ray has no such argument.
                limit = (
                    {} if per_task_row_limit is None else {"per_task_row_limit": per_task_row_limit}
                )
                tasks.append(ReadTask(read, metadata, **limit))
            return tasks

    return _Datasource(plan)
