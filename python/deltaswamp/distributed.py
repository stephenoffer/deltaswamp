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

import os
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import Any, ClassVar, cast

from ._util import commit_backoff

__all__ = ["DeltaSwampDatasource", "ScanPlan", "WritePlan", "balance", "merge_fragments"]


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
    #: Where workers get a fresh storage credential when the shipped one
    #: nears expiry: a picklable ``source(table_id, operation)`` reaching a
    #: driver-side `credentials.CredentialBroker`. None: they cannot.
    credential_source: Any = None

    @property
    def credential_expires_at(self) -> float | None:
        """When the storage credential workers get expires (epoch seconds), if known.

        Workers refresh it when the plan has a `credential_source` or ships
        catalog auth; otherwise a job running past this fails on its workers.
        """
        return _credential_expiry(self.table, write=False)

    @property
    def version(self) -> int | None:
        return self.splits[0].commit_version if self.splits else self.snapshot_version

    def __reduce__(self) -> tuple[Any, ...]:
        # What crosses a process boundary: see `_for_workers`.
        fields = {f.name: getattr(self, f.name) for f in dataclass_fields(self)}
        if not self.ship_catalog_auth:
            fields["table"] = _for_workers(self.table, write=False, source=self.credential_source)
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
    #: The top-level columns with a literal DEFAULT, as Arrow fields: a batch
    #: that leaves one out gets the default, as `Table.append` fills it.
    default_fields: tuple[Any, ...] = ()
    #: See `ScanPlan.credential_source`.
    credential_source: Any = None

    @property
    def credential_expires_at(self) -> float | None:
        """When the storage credential workers write with expires, if known."""
        return _credential_expiry(self.table, write=True)

    #: Retries an ordinary append gets when `retries` is not given. Concurrent
    #: jobs really do collide -- four committing at once leaves one winner and
    #: three `CommitConflictError`s -- and rebasing an append is always correct,
    #: so the default matches `KernelEngine.append_commit_retries` (and
    #: delta-rs), with a jittered backoff between attempts, rather than leaving
    #: every connector to write the same loop. An overwrite gets none, and so
    #: does a catalog-managed table resolved without its catalog, which cannot
    #: re-read its commit tail.
    default_append_retries: ClassVar[int] = 15

    @property
    def overwrite(self) -> bool:
        return self.mode == "overwrite"

    def __reduce__(self) -> tuple[Any, ...]:
        # What crosses a process boundary: see `_for_workers`. The driver keeps
        # its own copy, with full catalog access, for the commit.
        fields = {f.name: getattr(self, f.name) for f in dataclass_fields(self)}
        if not self.ship_catalog_auth:
            fields["table"] = _for_workers(self.table, write=True, source=self.credential_source)
            fields["catalog"] = None
        else:
            from .credentials.databricks import shipping

            fields["table"] = _shipping_table(self.table)
            fields["catalog"] = shipping(fields["catalog"])
        return (_rebuild, (type(self), fields))

    def write(self, data: Any) -> bytes:
        """Worker side: write `data` as files, returning a fragment to send back.

        The files are durable when this returns but belong to no version yet.
        Every fragment must reach `commit()` or the files are orphaned.
        """
        from .errors import InvalidArgumentError

        if data is None:
            raise InvalidArgumentError("plan.write() needs data; got None")
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
        if self.default_fields:
            # Left to the kernel, a batch without a defaulted column failed
            # on every worker, after planning had accepted the write.
            data = _with_defaults(data, self.default_fields)
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
        identity = {"table_identity": self.table_identity} if self.table_identity else {}
        result: bytes = self.engine.write_files(self.table, data, version=self.version, **identity)
        return result

    def commit(
        self,
        fragments: Iterable[bytes],
        *,
        operation: str | None = "WRITE",
        retries: int | None = None,
        allow_concurrent_overwrite: bool = False,
        allow_empty_overwrite: bool = False,
        abort_on_failure: bool = True,
    ) -> int:
        """Driver side: commit every fragment as one transaction.

        Returns the committed version. The commit resolves the table again, so
        an append lands on top of whatever else has been written since the plan
        was made rather than failing on it, which is what an append means.

        It still has to win the race for its version, and concurrent jobs
        do collide: four committing at once leaves one winner and three
        conflicts. Rebasing an append is always correct, so `retries` defaults
        to `default_append_retries` and the losers simply commit at the next
        version. Pass `retries=0` to see the conflict instead. A
        catalog-managed table planned through its catalog rebases the same
        way: every attempt re-reads the catalog's commit tail.

        An overwrite is the opposite: it removes what it finds, so committing
        against a table that has moved on would discard a writer that arrived
        after planning. That is refused unless `allow_concurrent_overwrite` says
        the last writer should win. Left as None, `retries` is 0 for one.

        `fragments` may be any iterable, a generator included; they are merged
        as they arrive (see `merge_fragments`), so the driver never holds one
        schema per worker.

        Every attempt first checks whether these files already landed -- a
        commit reported as failed can have taken effect (a put or a catalog
        call that timed out after it did), and a restarted driver may commit
        its saved fragments again. It reads only the commits made since the
        write was planned. When they did land, the version they landed at is
        returned and nothing is committed again.

        A concurrent change to the schema (other than adding a nullable
        column), partitioning or column mapping is never retried: the
        fragments' files were written for the old layout, so the commit raises
        `MetadataChangedError` and the write must be planned again.

        When the commit fails for certain -- refused before anything was
        written, `MetadataChangedError`, an overwrite whose table moved, or a
        conflict on every attempt -- the fragments' files are deleted
        (`abort`), since nothing will ever reference them; pass
        ``abort_on_failure=False`` to keep them for another commit. A failure
        whose outcome is unknown (`TransientCommitError`: a timeout or a 5xx)
        never deletes anything: commit the same fragments again, which returns
        the version if the first attempt landed, and abort only after that.

        No fragments with any files (every worker's data was empty) commits
        nothing: an append returns the current version without adding an
        empty one, and an overwrite -- which would empty the table -- is
        refused unless `allow_empty_overwrite=True` says that is intended.
        """
        from .errors import InvalidArgumentError

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
            # Retrying an overwrite means overwriting the writer that just
            # won. A catalog-managed table rebases only through its catalog:
            # without it, the commit tail captured at resolution would make
            # every retry lose again.
            rebases = not self.table.is_catalog_managed or self.catalog is not None
            retries = self.default_append_retries if rebases and not self.overwrite else 0
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise InvalidArgumentError(f"retries must be a non-negative int, not {retries!r}")
        try:
            return self._commit(
                collected,
                operation=operation,
                retries=retries,
                allow_concurrent_overwrite=allow_concurrent_overwrite,
                allow_empty_overwrite=allow_empty_overwrite,
            )
        except _CertainFailure as failure:
            if abort_on_failure:
                self._abort_after(collected, failure.error)
            raise failure.error from failure.error.__cause__

    def _commit(
        self,
        collected: list[bytes],
        *,
        operation: str,
        retries: int,
        allow_concurrent_overwrite: bool,
        allow_empty_overwrite: bool,
    ) -> int:
        """`commit`'s attempts. A failure that surely committed nothing is a `_CertainFailure`."""
        from .errors import (
            CommitConflictError,
            MetadataChangedError,
            TransientCommitError,
            UnreachableTableError,
        )

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
        paths, written_at = _fragment_files(collected)
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
            # Every attempt, the first and an overwrite's included: a driver
            # that restarts commits its saved fragments with nothing to say an
            # earlier run landed them, and an overwrite committed twice added
            # and removed the same files in one version.
            landed = self._landed_version(table, paths, written_at)
            if landed is not None:
                return landed
            # Re-checked every attempt, not once: losing a race means the table
            # moved by definition, so a retry is exactly when an overwrite is
            # most likely to be discarding someone.
            if self.overwrite and not allow_concurrent_overwrite:
                try:
                    self._refuse_if_the_table_moved(table)
                except UnreachableTableError as exc:
                    raise _CertainFailure(exc) from None
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
                    ),
                    table,
                )
            except MetadataChangedError as exc:
                # The files do not fit the table any more: no retry can help,
                # and nothing was committed.
                raise _CertainFailure(exc) from None
            except TransientCommitError as exc:
                # Nobody is known to have won the version, so the same commit
                # can go again -- which `retries` promises. The landed check
                # at the top of the next attempt settles whether this one took
                # effect after all, a catalog's ratification included, so a
                # catalog-managed table planned through its catalog retries too.
                last = exc
                if attempts == 1 or (self.table.is_catalog_managed and self.catalog is None):
                    _unknown_outcome(exc)
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
                    raise _CertainFailure(exc) from None
                if self.table.is_catalog_managed and self.catalog is None:
                    # The ratified tail and the version ceiling were captured
                    # when the table was resolved, and with no catalog to
                    # re-read them from, a retry would race the same stale view.
                    refusal = UnreachableTableError(
                        "retry the commit",
                        "this table is catalog-managed, and its commit tail was captured "
                        f"when it was resolved, so a retry here would reuse it ({exc})",
                        "re-open the table through the catalog and commit the same "
                        "fragments against the fresh snapshot (with abort_on_failure=False "
                        "here, so they are kept)",
                    )
                    refusal.__cause__ = exc
                    raise _CertainFailure(refusal) from None
                # Losers retrying at once collide again; spread them out.
                commit_backoff(attempt)
                continue
            return version

        if isinstance(last, TransientCommitError):
            raise _unknown_outcome(last)
        raise _CertainFailure(
            CommitConflictError(
                conflict_version,
                f"another writer committed first on each of {attempts} attempts ({last}). "
                "Nothing was committed.",
            )
        )

    def abort(self, fragments: Iterable[bytes]) -> int:
        """Delete the data files `fragments` describe; the number deleted.

        For a write that will not be committed: a failed or cancelled job
        (a Ray datasink's ``on_write_failed``), or a `commit` that raised
        `TransientCommitError` and, committed again, still did not land.
        `commit` already aborts on a failure that certainly committed nothing.

        Refused when any of the files is in a commit made since the write was
        planned -- deleting a committed file would corrupt the table -- or
        when that cannot be told. Files that cannot be deleted are logged,
        not raised: an abort is cleanup, and the job's own error matters more.
        """
        from .errors import UnreachableTableError

        collected = _fragments_arg(fragments)
        paths, written_at = _fragment_files(collected)
        if not paths:
            return 0
        if self.table.is_catalog_managed and self.catalog is None:
            # Only the catalog knows its newest commits: the tail captured at
            # resolution cannot show that these files were committed since.
            raise UnreachableTableError(
                "abort these fragments",
                "the table is catalog-managed and this plan has no catalog to read its "
                "newest commits from, so whether the files were committed cannot be told",
                "abort through the plan returned by plan_write() on the driver",
            )
        table = self._with_fresh_tail(self.table) or self.table
        if self._landed_version(table, paths, written_at) is not None:
            raise UnreachableTableError(
                "abort these fragments",
                "their files were committed, so deleting them would corrupt the table",
                "leave them; the write succeeded",
            )
        failed = self.engine.delete_uncommitted(table, sorted(paths))
        if failed:
            import logging

            logging.getLogger("deltaswamp").warning(
                "abort could not delete %d of %d uncommitted data file(s) (VACUUM will): %s",
                len(failed),
                len(paths),
                "; ".join(failed[:10]),
            )
        return len(paths) - len(failed)

    def _abort_after(self, collected: list[bytes], error: Exception) -> None:
        """`abort` after `error`, which stays the one raised; a failed abort is logged."""
        import logging

        try:
            deleted = self.abort(collected)
        except Exception as exc:
            logging.getLogger("deltaswamp").warning(
                "the commit failed (%s) and its data files could not be deleted: %s", error, exc
            )
            return
        if deleted:
            logging.getLogger("deltaswamp").info(
                "the commit failed, so its %d data file(s) were deleted", deleted
            )

    def _landed_version(self, table: Any, paths: set[str], written_at: int | None) -> int | None:
        """The version these files landed at, or None if they did not.

        Reads only the commits after the files were written (`written_at`,
        from the fragments; else the planned version), so fragments committed
        again through a later plan are still found. Raises when that cannot
        be told (a commit since was cleaned up), or when only some of the
        files landed -- neither is a reason to commit them again.
        """
        from .errors import InvalidArgumentError, UnreachableTableError

        if not paths:
            return None
        check = getattr(self.engine, "commits_adding", None)
        if not callable(check):
            return None  # an engine without distributed writes has nothing to find
        after = min((v for v in (written_at, self.version) if v is not None), default=0)
        try:
            found = check(table, after, sorted(paths))
        except Exception as exc:
            raise UnreachableTableError(
                "commit these fragments",
                f"whether an earlier commit already landed them cannot be told ({exc})",
                "check the table's history for these files before committing again",
            ) from exc
        if not found:
            return None
        if len(found) == 1 and found[0][1] == len(paths):
            return int(found[0][0])
        raise InvalidArgumentError(
            f"only some of these fragments' files are in the table (commits "
            f"{sorted(v for v, _ in found)} add {sum(n for _, n in found)} of {len(paths)}); "
            "they were committed in part elsewhere, so they cannot be committed again"
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


def _checked_fragments(fragments: Any, what: str = "commit") -> Iterator[bytes]:
    """Each fragment as bytes, checked as it arrives.

    `commit(fragment)` with one bare fragment iterated its bytes as ints and
    failed deep in the binding, and a worker that returned None (or a str)
    surfaced as "Can't extract `str` to `Vec`" with no hint which one.
    """
    from .errors import InvalidArgumentError

    if isinstance(fragments, (bytes, bytearray, memoryview)):
        raise InvalidArgumentError(
            f"{what}() takes a list of fragments; wrap a single one: {what}([fragment])"
        )
    if fragments is None or isinstance(fragments, (str, dict)):
        raise InvalidArgumentError(
            f"{what}() takes a list of fragments from plan.write(), not {type(fragments).__name__}"
        )
    for index, fragment in enumerate(fragments):
        if not isinstance(fragment, (bytes, bytearray, memoryview)):
            raise InvalidArgumentError(
                f"fragment {index} is a {type(fragment).__name__}, not the bytes "
                "plan.write() returns; did that worker fail?"
            )
        yield bytes(fragment)


def _fragments_arg(fragments: Any) -> list[bytes]:
    """The fragments to commit, checked and merged as they arrive.

    A fragment is an Arrow IPC stream: a schema (the add-metadata columns,
    per-column stats included, and the layout the files were written under)
    and a batch of a few rows, each batch message describing every nested
    buffer. At a row or two per task that was ten kilobytes per fragment, and
    20k tasks held 600 MiB on the driver. Consecutive fragments of one plan
    are concatenated into a few large ones as they are read; one that does not
    decode, or carries a different schema, is passed on as it is for the
    commit to name.
    """
    out: list[bytes] = []
    merger = _Merger()
    for fragment in _checked_fragments(fragments):
        if not fragment:
            continue  # no files
        try:
            merged = merger.add(fragment)
        except _NotMergeable:
            out.extend(merger.flush())
            out.append(fragment)
            continue
        out.extend(merged)
    out.extend(merger.flush())
    return out


class _NotMergeable(Exception):
    """A fragment that does not decode, left for the commit to name."""


class _Merger:
    """Concatenates fragments of one schema into few, large ones."""

    #: Pending batches are concatenated once there are this many.
    COMBINE_EVERY = 512
    #: A merged fragment is cut once it holds this many rows (files).
    ROWS = 250_000

    def __init__(self) -> None:
        self.schema: Any = None
        self.batches: list[Any] = []
        self.rows = 0

    def add(self, fragment: bytes) -> list[bytes]:
        """Take `fragment`; the merged fragments that are complete because of it."""
        import pyarrow as pa

        try:
            reader = pa.ipc.open_stream(fragment)
            schema = reader.schema
            batches = list(reader)
        except Exception as exc:
            raise _NotMergeable() from exc
        done: list[bytes] = []
        if self.schema is not None and not schema.equals(self.schema, check_metadata=True):
            done = self.flush()
        self.schema = schema
        self.batches.extend(b for b in batches if b.num_rows)
        self.rows += sum(b.num_rows for b in batches)
        if len(self.batches) >= self.COMBINE_EVERY:
            self.batches = self._combined()
        if self.rows >= self.ROWS:
            done.extend(self.flush())
        return done

    def _combined(self) -> list[Any]:
        import pyarrow as pa

        if len(self.batches) <= 1:
            return self.batches
        table = pa.Table.from_batches(self.batches, schema=self.schema).combine_chunks()
        return list(table.to_batches())

    def flush(self) -> list[bytes]:
        """The pending fragments as one, if any."""
        import pyarrow as pa

        if self.schema is None:
            return []
        batches = self._combined()
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, self.schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
        self.schema, self.batches, self.rows = None, [], 0
        return [sink.getvalue().to_pybytes()]


def merge_fragments(fragments: Iterable[bytes]) -> bytes:
    """Merge fragments from one plan's workers into one fragment.

    For combining results before they reach the driver -- a tree reduction
    over a large job -- so the driver receives a few fragments rather than
    one per task. `commit` accepts the result like any other fragment, and
    merges what it is given too. Fragments from different plans (another
    table, or one written under a different layout) are refused.
    """
    import pyarrow as pa

    from .errors import InvalidArgumentError

    merger = _Merger()
    merger.ROWS = 2**62  # one fragment, however large
    for index, fragment in enumerate(_checked_fragments(fragments, "merge_fragments")):
        if not fragment:
            continue
        try:
            schema = pa.ipc.open_stream(fragment).schema
        except Exception as exc:
            raise InvalidArgumentError(
                f"fragment {index} is not a fragment plan.write() produced ({exc})"
            ) from exc
        if merger.schema is not None and not schema.equals(merger.schema, check_metadata=True):
            raise InvalidArgumentError(
                f"fragment {index} was written for a different table or layout than the "
                "first; merge only the fragments of one plan"
            )
        merger.add(fragment)
    merged = merger.flush()
    return merged[0] if merged else b""


def _fragment_files(fragments: list[bytes]) -> tuple[set[str], int | None]:
    """The data file paths `fragments` describe, and the earliest version they were written at.

    The version is None when a fragment does not record it (one written by
    an older release).
    """
    import pyarrow as pa

    from .engine.kernel import FRAGMENT_WRITTEN_AT

    paths: set[str] = set()
    earliest: int | None = None
    unrecorded = False
    for fragment in fragments:
        if not fragment:
            continue
        try:
            table = pa.ipc.open_stream(fragment).read_all()
        except Exception:
            continue  # the commit names a malformed fragment itself
        if "path" in table.column_names:
            paths.update(p for p in table.column("path").to_pylist() if p)
        written = (table.schema.metadata or {}).get(FRAGMENT_WRITTEN_AT.encode())
        try:
            at = int(written) if written is not None else None
        except ValueError:
            at = None
        if at is None:
            unrecorded = True
        elif earliest is None or at < earliest:
            earliest = at
    return paths, None if unrecorded else earliest


def _with_defaults(data: Any, fields: tuple[Any, ...]) -> Any:
    """`data` with each of `fields` it leaves out filled with the column's literal DEFAULT."""
    import pyarrow as pa

    from .table import _default_column

    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])
    elif not isinstance(data, pa.Table) and type(data).__name__ == "DataFrame":
        try:
            data = pa.Table.from_pandas(data, preserve_index=False)
        except Exception:
            return data
    if not isinstance(data, pa.Table):
        return data
    present = {name.lower() for name in data.column_names}
    for field in fields:
        if field.name.lower() in present:
            continue
        column = _default_column(pa, field, data.num_rows)
        if column is not None:
            data = data.append_column(field, column)
    return data


class _CertainFailure(Exception):
    """A commit failure that certainly committed nothing; carries the error to raise."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


def _unknown_outcome(exc: Exception) -> Exception:
    """`exc` (a TransientCommitError), saying what to do with the fragments."""
    from .errors import TransientCommitError

    if not isinstance(exc, TransientCommitError):
        return exc
    note = (
        " Whether the commit took effect is unknown, so its data files were kept: commit "
        "the same fragments again (a landed commit is recognized and its version "
        "returned), and abort them only if that says they did not land."
    )
    if note not in str(exc):
        exc.args = (str(exc) + note, *exc.args[1:])
    return exc


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
        source: Any = None,
        identity: str | None = None,
    ) -> None:
        self._credentials = credentials
        self._table_id = table_id
        self._workspace_url = workspace_url
        #: `credential_source` of the plan: asked for a fresh credential.
        self._source = source
        #: The driver provider's identity: every task of a plan (and every
        #: plan of one principal and table) shares a slot and a cache entry.
        self._identity = identity or f"shipped-{uuid.uuid4().hex}"

    @property
    def table_id(self) -> str | None:
        return self._table_id

    @property
    def refreshable(self) -> bool:
        """Whether a fresh credential can be had here (a credential source)."""
        return self._source is not None

    def credential_identity(self) -> str:
        return f"shipped-{self._identity}"

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
        if self._source is not None:
            self._credentials = _sourced(self, Operation(str(wanted).upper()))
        if self._credentials.expires_within(SHIPPED_CREDENTIAL_MARGIN_SECONDS):
            import time

            left = (self.expires_at or 0) - time.time()
            raise CredentialError(
                "the storage credential shipped with this plan "
                + ("has expired" if left <= 0 else f"expires in {left:.0f}s")
                + ", and a worker cannot re-vend it: plans carry no catalog credentials. "
                "Re-plan on the driver (plan_scan()/plan_write() again), or plan with "
                "credential_source= (a driver-side CredentialBroker) or "
                "ship_catalog_auth=True so workers can refresh it"
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


#: Credentials a credential source served in this process, and when it was
#: last asked, by (identity, operation): every task of a plan in one worker
#: shares one answer, so a job asks once per refresh per process.
_SOURCED: dict[tuple[str, str], tuple[Any, float]] = {}
_SOURCED_LOCK = threading.Lock()

#: A credential source is not asked again sooner than this, unless the
#: credential has expired: one it answers with a short-lived credential
#: would otherwise be asked on every request.
SOURCE_MIN_INTERVAL_SECONDS = 30.0


def _sourced(shipped: ShippedCredentials, operation: Any) -> Any:
    """The freshest credential for `shipped`: its own, this process's, or the source's."""
    from .credentials import Credentials
    from .credentials.base import DEFAULT_REFRESH_MARGIN_SECONDS
    from .errors import CredentialError

    key = (shipped.credential_identity(), str(operation.value))
    current = shipped._credentials
    with _SOURCED_LOCK:
        known, asked = _SOURCED.get(key, (None, 0.0))
        if known is not None and (known.expires_at or 0) > (current.expires_at or 0):
            current = known
        now = time.time()
        if not current.expires_within(DEFAULT_REFRESH_MARGIN_SECONDS) or (
            not current.is_expired and now - asked < SOURCE_MIN_INTERVAL_SECONDS
        ):
            return current
        # Asked under the lock: the other tasks in this process wait for the
        # one answer rather than each asking the driver.
        _SOURCED[key] = (current, now)
        try:
            answer = shipped._source(shipped.table_id, operation.value)
        except Exception as exc:
            if not current.is_expired:
                return current  # asked again after the interval
            raise CredentialError(
                f"the plan's credential source could not vend a fresh credential: {exc}"
            ) from exc
        served = answer if isinstance(answer, Credentials) else Credentials._from_state(answer)
        _SOURCED[key] = (served, now)
        return served


def _credential_expiry(table: Any, *, write: bool) -> float | None:
    """When the credential a plan ships for `table` expires (vends it on the driver)."""
    from .credentials import Operation

    provider = getattr(table, "credential_provider", None)
    if provider is None:
        return None
    return getattr(
        provider.credentials(Operation.READ_WRITE if write else Operation.READ), "expires_at", None
    )


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


def _for_workers(table: Any, *, write: bool, source: Any = None) -> Any:
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
    # The same for every pickle of this provider, so a plan's tasks share one
    # slot, cache entry and refresh in each worker process.
    identity = f"object-{id(provider):x}-{os.getpid()}"
    get_identity = getattr(provider, "credential_identity", None)
    if callable(get_identity):
        identity = get_identity()
    shipped = ShippedCredentials(
        credentials,
        getattr(provider, "table_id", None),
        workspace_url,
        source=source,
        identity=identity,
    )
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
