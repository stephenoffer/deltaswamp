"""Distributed reads and writes: plan on the driver, do the work on workers.

A scan plan is a list of `ScanSplit`s pinned to one snapshot version. A worker
receives the engine, the resolved table and its splits, and re-resolves that
exact version. By default the table carries one short-lived storage credential
vended on the driver (`ShippedCredentials`), never the catalog's credentials;
with ``ship_catalog_auth=True`` it carries the credential provider, and each
worker vends its own. The Ray Data datasource makes one read task per
byte-balanced group of splits.

Writes run in reverse. `WritePlan` checks on the driver that the commit can
succeed before any worker runs, workers write data files and return opaque
fragments, and the driver commits every fragment in one transaction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import Any, ClassVar, cast

from ._util import commit_backoff

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
    #: Where the table's VARIANT columns are (paths of names), read as JSON
    #: text as `Table` reads them; the engines give the binary encoding.
    variant_paths: tuple[tuple[str, ...], ...] = ()
    #: Where its interval columns are, by kind (`engine.intervals`): read as
    #: `Table` reads them, a duration or text, where the engines give integers.
    interval_paths: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = ()

    @property
    def version(self) -> int | None:
        return self.splits[0].commit_version if self.splits else self.snapshot_version

    def __reduce__(self) -> tuple[Any, ...]:
        # What crosses a process boundary: see `_for_workers`.
        fields = {f.name: getattr(self, f.name) for f in dataclass_fields(self)}
        if not self.ship_catalog_auth:
            fields["table"] = _for_workers(self.table, write=False)
        else:
            fields["table"] = _shipping_table(self.table)
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
            from .engine.base import ScanSplit
            from .errors import InvalidArgumentError

            foreign = [s for s in chosen if not isinstance(s, ScanSplit)]
            if foreign:
                # A string raised a bare AttributeError ('str' has no 'path').
                raise InvalidArgumentError(
                    f"read() takes this plan's splits (ScanSplit objects), not "
                    f"{type(foreign[0]).__name__} {foreign[0]!r:.80}"
                )
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

        if self.variant_paths:
            from ._variant import json_text_stream

            # to_arrow() gives VARIANT as JSON text; so does a planned read.
            stream = json_text_stream(stream, frozenset(self.variant_paths))
        if self.interval_paths:
            from .engine.intervals import interval_stream

            stream = interval_stream(stream, {g: frozenset(p) for g, p in self.interval_paths})
        # A split whose file was vacuumed since planning failed as a bare
        # OSError; name the file instead.
        where = getattr(self.table, "location", None) or "the table"
        at = f" at version {self.version}" if self.version is not None else ""
        return translating_stream(stream, f"{where}{at}")

    def partitions(self, n: int) -> list[tuple[Any, ...]]:
        from .errors import InvalidArgumentError

        # 0, -5 and True quietly came back as one group, and None as a bare
        # TypeError from the comparison inside `balance`.
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise InvalidArgumentError(f"partitions(n) needs a positive int, not {n!r}")
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
    #: Where the table's VARIANT columns are: JSON text given for one is
    #: encoded as `Table.append` encodes it.
    variant_paths: tuple[tuple[str, ...], ...] = ()
    #: Where its interval columns are (see `ScanPlan`): a duration or text
    #: given for one is written as the integers Delta stores.
    interval_paths: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = ()
    #: The table's metaData id at the planned version. Workers reuse a cached
    #: snapshot, and the commit reads a fresh one; both must be this table,
    #: not one dropped and re-created at the same path since planning.
    table_identity: str | None = None
    #: Identity values reserved for this write when it was planned, as
    #: (column, first, step, count) blocks, and the number of equal slots
    #: each is cut into: `write(data, task_index=i)` draws from slot i.
    identity_blocks: tuple[tuple[str, int, int, int], ...] = ()
    identity_slots: int = 0
    #: Tells this plan's identity slots apart from another's in one process.
    plan_id: str = ""
    #: User domain metadata the commit sets beside the rows (domain ->
    #: configuration), checked when the write was planned.
    domain_metadata: dict[str, str] | None = None

    #: Retries an ordinary append gets when `retries` is not given. Concurrent
    #: jobs really do collide -- four committing at once leaves one winner and
    #: three `CommitConflictError`s -- and rebasing an append is always correct,
    #: so the default matches `KernelEngine.append_commit_retries` (and
    #: delta-rs), with a jittered backoff between attempts, rather than leaving
    #: every connector to write the same loop. An overwrite gets none, and so
    #: does a catalog-managed table, which cannot rebase here.
    default_append_retries: ClassVar[int] = 15

    @property
    def overwrite(self) -> bool:
        return self.mode == "overwrite"

    def __reduce__(self) -> tuple[Any, ...]:
        # What crosses a process boundary: see `_for_workers`. The driver keeps
        # its own copy, with full catalog access, for the commit.
        fields = {f.name: getattr(self, f.name) for f in dataclass_fields(self)}
        if not self.ship_catalog_auth:
            fields["table"] = _for_workers(self.table, write=True)
            fields["catalog"] = None
        else:
            from .credentials.databricks import shipping

            fields["table"] = _shipping_table(self.table)
            fields["catalog"] = shipping(fields["catalog"])
        return (_rebuild, (type(self), fields))

    def write(self, data: Any, *, task_index: int | None = None) -> bytes:
        """Worker side: write `data` as files, returning a fragment to send back.

        The files are durable when this returns but belong to no version yet.
        Every fragment must reach `commit()` or the files are orphaned.

        `task_index` is this worker task's number, from 0 to the plan's
        ``identity_tasks - 1`` (a Ray datasink's ``ctx.task_idx``): on a table
        with identity columns it picks the slot of reserved values the task's
        rows are numbered from. Calls with one index must come from one
        process, which hands them successive values of the slot.
        """
        from .errors import InvalidArgumentError

        if data is None:
            raise InvalidArgumentError("plan.write() needs data; got None")
        identity = self._identity_slot(task_index)
        from ._util import not_table_data

        refusal = not_table_data(data)
        if refusal is not None:
            raise InvalidArgumentError(refusal)
        if isinstance(data, list) and data:
            import pyarrow as pa

            # The shapes `Table.append` accepts: row dicts, or record batches
            # (what a Ray block iterator yields). Both failed in pa.table().
            if all(isinstance(row, dict) for row in data):
                data = pa.Table.from_pylist(data)
            elif all(isinstance(b, pa.RecordBatch) for b in data):
                data = pa.Table.from_batches(data)
        if self.variant_paths:
            import pyarrow as pa

            from ._variant import binary_columns

            if isinstance(data, pa.RecordBatch):
                data = pa.Table.from_batches([data])
            if isinstance(data, pa.Table):
                # The engines take the binary encoding; JSON text failed with a
                # raw "Expected Struct, got Utf8".
                data = binary_columns(pa, data, frozenset(self.variant_paths))
        if self.interval_paths:
            import pyarrow as pa

            from .engine.intervals import storage_columns

            if isinstance(data, pa.RecordBatch):
                data = pa.Table.from_batches([data])
            if isinstance(data, pa.Table):
                data = storage_columns(pa, data, {g: frozenset(p) for g, p in self.interval_paths})

        # At the planned version: resolved once per process and reused by
        # every later write(), where the latest snapshot cost a log replay
        # per call (0.6 s each, 5000 commits past a checkpoint).
        kwargs: dict[str, Any] = {}
        if self.table_identity:
            kwargs["table_identity"] = self.table_identity
        if identity:
            kwargs["identity"] = identity
        result: bytes = self.engine.write_files(self.table, data, version=self.version, **kwargs)
        return result

    def _identity_slot(self, task_index: int | None) -> dict[str, Any] | None:
        """The cursor over task `task_index`'s slot of each reserved identity block."""
        if not self.identity_blocks:
            return None
        from .engine.values import IdentityBlock, slot_cursor
        from .errors import InvalidArgumentError

        if task_index is None or isinstance(task_index, bool) or int(task_index) != task_index:
            raise InvalidArgumentError(
                "the table has identity columns, so plan.write() needs task_index= (0 to "
                f"{self.identity_slots - 1}): each task numbers its rows from its own slot "
                "of the values reserved when the write was planned"
            )
        return {
            name: slot_cursor(
                self.plan_id,
                int(task_index),
                name,
                IdentityBlock(first, step, count).slot(int(task_index), self.identity_slots),
            )
            for name, first, step, count in self.identity_blocks
        }

    def commit(
        self,
        fragments: Iterable[bytes],
        *,
        operation: str | None = "WRITE",
        retries: int | None = None,
        allow_concurrent_overwrite: bool = False,
        allow_empty_overwrite: bool = False,
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
        and says so rather than spinning against a stale commit tail. Left as
        None, an ordinary append on a path table gets `default_append_retries`;
        an overwrite gets none, because retrying one means overwriting the
        writer that just won.

        A concurrent change to the schema (other than adding a nullable
        column), partitioning or column mapping is never retried: the
        fragments' files were written for the old layout, so the commit raises
        `MetadataChangedError` and the write must be planned again.

        No fragments with any files (every worker's data was empty) commits
        nothing: an append returns the current version without adding an
        empty one, and an overwrite -- which would empty the table -- is
        refused unless `allow_empty_overwrite=True` says that is intended.
        """
        from .errors import (
            CommitConflictError,
            InvalidArgumentError,
            MetadataChangedError,
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
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise InvalidArgumentError(f"retries must be a non-negative int, not {retries!r}")
        if not any(collected):
            # A job whose workers all produced nothing added an empty version
            # on every call (and every retry of the job), and an overwrite
            # silently truncated the table. With a txn the commit still
            # matters -- it records that the batch is done -- so it goes ahead.
            if self.overwrite and not allow_empty_overwrite:
                raise UnreachableTableError(
                    "commit this overwrite",
                    "no fragment carries any files, so it would remove every row in the "
                    "table and add none",
                    "pass allow_empty_overwrite=True if emptying the table is intended",
                )
            if not self.overwrite and self.txn is None:
                current = self._with_fresh_tail(self.table) or self.table
                version_now = self.engine.detail(current).get("version")
                if version_now is not None:
                    return int(version_now)
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
                version: int = self._backfilled(
                    lambda target: self.engine.commit_files(
                        target,
                        collected,
                        overwrite=self.overwrite,
                        operation=operation,
                        txn=self.txn,
                        commit_metadata=self.commit_metadata,
                        **({"version": pinned} if pinned is not None else {}),
                        **({"table_identity": self.table_identity} if self.table_identity else {}),
                        **(
                            {"domain_metadata": self.domain_metadata}
                            if self.domain_metadata
                            else {}
                        ),
                    ),
                    table,
                )
            except MetadataChangedError:
                # The files do not fit the table any more: no retry can help.
                raise
            except TransientCommitError as exc:
                # Nobody won the version and the table is unchanged, so the
                # very same commit can go again -- which `retries` promises.
                # Not on a catalog-managed table: a failed ratification call
                # may have landed, and only the catalog can say.
                last = exc
                if attempts == 1 or self.table.is_catalog_managed:
                    raise
                commit_backoff(attempt)
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
                # Losers retrying at once collide again; spread them out.
                commit_backoff(attempt)
                continue
            return version

        if isinstance(last, TransientCommitError):
            raise last
        raise CommitConflictError(
            conflict_version,
            f"another writer committed first on each of {attempts} attempts ({last}). "
            "The fragments are still valid: re-open the table and commit them again.",
        )

    def _backfilled(self, commit: Any, table: Any) -> int:
        """`commit(table)`; on a catalog's backfill demand, publish and commit once more.

        What `Table._backfilled` does for `Table.append`. Nothing on this path
        ever published, so once the catalog's cap of unpublished commits was
        reached every distributed commit failed with the 429 -- after its job
        had run -- while appends through the same table kept working. The
        refused commit changed nothing and the fragments stay valid, so the
        same commit can go again once the tail is published.
        """
        from .errors import BackfillRequiredError, DeltaSwampError

        try:
            result: int = commit(table)
            return result
        except BackfillRequiredError as exc:
            if not getattr(table, "is_catalog_managed", False) or self.catalog is None:
                # Without the catalog the tail cannot be re-read after
                # publishing, so a second commit would race a stale view.
                raise
            try:
                self.engine.publish(table)
            except DeltaSwampError:
                raise exc from None
            refreshed = self._with_fresh_tail(table)
            result = commit(refreshed if refreshed is not None else table)
            return result

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

    def __getstate__(self) -> dict[str, Any]:
        # Shipping the storage credential is this class's whole purpose, so it
        # opts in explicitly: `Credentials` itself refuses to pickle.
        state = self.__dict__.copy()
        credentials = state.pop("_credentials")
        state["_credential_state"] = (
            credentials._state() if hasattr(credentials, "_state") else None
        )
        if state["_credential_state"] is None:
            state["_credentials"] = credentials
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        from .credentials import Credentials

        carried = state.pop("_credential_state", None)
        self.__dict__.update(state)
        if carried is not None:
            self._credentials = Credentials._from_state(carried)


def _shipping_table(table: Any) -> Any:
    """`table` with a provider that pickles its catalog secrets: ship_catalog_auth=True.

    A provider pickles no literal secret by default (`credentials.databricks.shipping`);
    a plan that ships catalog auth is the caller asking for exactly that.
    """
    from .credentials.databricks import shipping

    provider = getattr(table, "credential_provider", None)
    if provider is None or not hasattr(provider, "_ship_secrets"):
        return table
    return replace(table, credential_provider=shipping(provider))


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
    serializing the job, and deterministic, so a retried task reads the same
    files.
    """
    items = sorted(splits, key=lambda s: (-s.size, s.path))
    if not items:
        return []
    import heapq

    count = max(1, min(n, len(items)))
    bins: list[list[Any]] = [[] for _ in range(count)]
    # (load, bin index): the lightest bin, lowest index on a tie -- the same
    # choice as scanning for min(loads), in O(log n) rather than O(n) per split.
    loads = [(0, i) for i in range(count)]
    for split in items:
        load, target = heapq.heappop(loads)
        bins[target].append(split)
        heapq.heappush(loads, (load + split.size, target))
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
            groups = self._plan.partitions(max(1, int(parallelism))) or [()]
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
                    paths = getattr(plan, "variant_paths", ())
                    if paths:
                        from deltaswamp._variant import json_text_stream

                        reader = json_text_stream(reader, frozenset(paths))
                    intervals = getattr(plan, "interval_paths", ())
                    if intervals:
                        from deltaswamp.engine.intervals import interval_stream

                        reader = interval_stream(reader, {g: frozenset(p) for g, p in intervals})
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
