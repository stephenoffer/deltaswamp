"""The change data feed read from the log, commit by commit.

The kernel's TableChanges lists `_delta_log/` itself and builds its snapshots
without a catalog's commit tail, so it cannot open a catalog-managed table:
the commits the catalog ratified but has not published are not in that
listing, and the kernel refuses a catalog-managed snapshot built without the
catalog's say. This reader takes the commits from a snapshot resolved with
the tail (`Snapshot.commit_log`, which reads a ratified commit from its staged
file) and derives each commit's changes as the protocol, and TableChanges,
do:

* a commit with `cdc` actions: its CDC files are its changes, each row's kind
  in `_change_type`, and its adds and removes are ignored;
* otherwise each `add` and `remove` with `dataChange` true: an add's live rows
  are inserts, a remove's live rows deletes. A path both removed and added in
  one commit is a deletion vector replaced: the rows the new vector deletes
  that the old one did not are deletes, and any the old one deleted that the
  new one does not are inserts.

Files are read through the kernel's own scan (`scan(scan_rows=...)` at the end
snapshot), so deletion vectors, column mapping, partition values and legacy
calendars are applied as for any read. Rows stream a file group at a time and
commits a bounded run at a time: the range is never held on the driver.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..errors import EngineLimitError, UnreachableTableError

#: Commits read from the log per step.
COMMITS_PER_STEP = 100

META = ("_change_type", "_commit_version", "_commit_timestamp")
_ROW_INDEX = "__deltaswamp_row_index"
_FILE = "__deltaswamp_file"


def _scan_row(action: dict[str, Any]) -> str:
    """An add, remove or cdc action as the kernel's scan row for its file."""
    return json.dumps(
        {
            "path": action["path"],
            "size": int(action.get("size") or 0),
            "modificationTime": action.get("modificationTime")
            or action.get("deletionTimestamp")
            or 0,
            "deletionVector": action.get("deletionVector"),
            "fileConstantValues": {
                "partitionValues": action.get("partitionValues") or {},
                "baseRowId": action.get("baseRowId"),
                "defaultRowCommitVersion": action.get("defaultRowCommitVersion"),
            },
        }
    )


def _feed_enabled(metadata: dict[str, Any]) -> bool:
    configuration = metadata.get("configuration") or {}
    return str(configuration.get("delta.enableChangeDataFeed", "false")).lower() == "true"


class LogChangeFeed:
    """One change-feed read: versions `start..end` of `snapshot`'s table.

    `snapshot` is the table at `end`, resolved with the catalog's tail where
    the table has one; `start_snapshot` the table at `start`. `columns` are
    the table columns to read (None: all); the change metadata always comes.
    `commit_times` maps a version to its commit time (ms) where the commit
    carries no in-commit timestamp.
    """

    def __init__(
        self,
        snapshot: Any,
        start_snapshot: Any,
        start: int,
        columns: list[str] | None,
        commit_times: dict[int, int] | None = None,
    ) -> None:
        import pyarrow as pa

        self.snapshot = snapshot
        self.start = start
        self.end = int(snapshot.version)
        self.columns = columns
        self.commit_times = commit_times or {}
        end_metadata = json.loads(snapshot.metadata_json())
        self.schema_string = end_metadata.get("schemaString")
        start_metadata = json.loads(start_snapshot.metadata_json())
        if not _feed_enabled(start_metadata):
            raise self._off(start)
        if start_metadata.get("schemaString") != self.schema_string:
            raise self._schema_changed()
        data_schema = pa.RecordBatchReader.from_stream(
            snapshot.scan(columns=columns, scan_rows=[])
        ).schema
        self.data_schema = data_schema
        self.schema = pa.schema(
            [
                *data_schema,
                pa.field("_change_type", pa.string(), nullable=False),
                pa.field("_commit_version", pa.int64(), nullable=False),
                pa.field("_commit_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            ],
            metadata=data_schema.metadata,
        )

    @staticmethod
    def _off(version: int) -> UnreachableTableError:
        error = UnreachableTableError(
            "read the change data feed",
            f"the change data feed was not enabled at version {version}, "
            "which the requested range includes",
            "start the range after the feed was enabled",
        )
        # Named, as the kernel path names it, so a follower can read up to it.
        error.version = version  # type: ignore[attr-defined]
        return error

    @staticmethod
    def _schema_changed() -> EngineLimitError:
        return EngineLimitError(
            "read the change data feed",
            "the table's schema changed within the requested range, and the "
            "change feed is read across one schema only",
            "read the ranges before and after the schema change separately",
        )

    def reader(self) -> Any:
        import pyarrow as pa

        return pa.RecordBatchReader.from_batches(self.schema, self._batches())

    # -------------------------------------------------------------- commits

    def _commits(self) -> Iterator[tuple[int, list[dict[str, Any]]]]:
        after = self.start - 1
        while after < self.end:
            until = min(after + COMMITS_PER_STEP, self.end)
            for version, text in self.snapshot.commit_log(after, until):
                actions = [json.loads(line) for line in text.splitlines() if line.strip()]
                yield int(version), actions
            after = until

    def _batches(self) -> Iterator[Any]:
        for version, actions in self._commits():
            yield from self._commit_changes(version, actions)

    def _commit_changes(self, version: int, actions: list[dict[str, Any]]) -> Iterator[Any]:
        info: dict[str, Any] = {}
        cdc: list[dict[str, Any]] = []
        adds: dict[str, dict[str, Any]] = {}
        removes: dict[str, dict[str, Any]] = {}
        for action in actions:
            if "commitInfo" in action:
                info = action["commitInfo"] or {}
            elif "metaData" in action:
                metadata = action["metaData"] or {}
                if not _feed_enabled(metadata):
                    raise self._off(version)
                if metadata.get("schemaString") != self.schema_string:
                    raise self._schema_changed()
            elif "cdc" in action:
                cdc.append(action["cdc"])
            elif "add" in action and action["add"].get("dataChange", True):
                adds[action["add"]["path"]] = action["add"]
            elif "remove" in action and action["remove"].get("dataChange", True):
                removes[action["remove"]["path"]] = action["remove"]
        timestamp = info.get("inCommitTimestamp")
        if timestamp is None:
            timestamp = self.commit_times.get(version, info.get("timestamp"))
        if timestamp is None:
            raise EngineLimitError(
                "read the change data feed",
                f"commit {version} records no time, in-commit or otherwise",
            )
        stamp = (version, int(timestamp))
        if cdc:
            for action in cdc:
                yield from self._change_file(action, stamp)
            return
        paired = adds.keys() & removes.keys()
        inserts = [a for p, a in adds.items() if p not in paired]
        deletes = [r for p, r in removes.items() if p not in paired]
        if inserts:
            yield from self._whole(inserts, "insert", stamp)
        if deletes:
            yield from self._whole(deletes, "delete", stamp)
        for path in sorted(paired):
            yield from self._replaced_vector(removes[path], adds[path], stamp)

    # ---------------------------------------------------------------- reads

    def _read(self, actions: list[dict[str, Any]], positions: bool = False) -> Any:
        import pyarrow as pa

        return pa.RecordBatchReader.from_stream(
            self.snapshot.scan(
                columns=self.columns,
                scan_rows=[_scan_row(a) for a in actions],
                row_positions=positions,
            )
        )

    def _stamped(self, batch: Any, kind: Any, stamp: tuple[int, int]) -> Any:
        import pyarrow as pa

        n = batch.num_rows
        kinds = kind if isinstance(kind, pa.Array) else pa.array([kind] * n, pa.string())
        columns = [batch.column(name) for name in self.data_schema.names]
        return pa.RecordBatch.from_arrays(
            [
                *columns,
                kinds,
                pa.array([stamp[0]] * n, pa.int64()),
                pa.array([stamp[1] * 1000] * n, pa.timestamp("us", tz="UTC")),
            ],
            schema=self.schema,
        )

    def _whole(self, actions: list[dict[str, Any]], kind: str, stamp: tuple[int, int]) -> Any:
        """Every live row of `actions`' files, as `kind` changes."""
        for batch in self._read(actions):
            if batch.num_rows:
                yield self._stamped(batch, kind, stamp)

    def _replaced_vector(
        self, removed: dict[str, Any], added: dict[str, Any], stamp: tuple[int, int]
    ) -> Iterator[Any]:
        """The rows a file's replaced deletion vector deleted (or restored)."""
        import pyarrow as pa
        import pyarrow.compute as pc

        old = self._vector_rows(removed.get("deletionVector"))
        new = self._vector_rows(added.get("deletionVector"))
        newly_deleted = new - old
        restored = old - new
        for rows, action, kind in (
            (newly_deleted, removed, "delete"),
            (restored, added, "insert"),
        ):
            if not rows:
                continue
            wanted = pa.array(sorted(rows), pa.int64())
            for batch in self._read([action], positions=True):
                index = batch.column(_ROW_INDEX).cast(pa.int64())
                keep = pc.is_in(index, value_set=wanted)
                picked = batch.filter(keep)
                if picked.num_rows:
                    yield self._stamped(picked, kind, stamp)

    def _vector_rows(self, descriptor: dict[str, Any] | None) -> set[int]:
        if not descriptor:
            return set()
        return set(self.snapshot.deletion_vector_rows(json.dumps(descriptor)))

    def _change_file(self, action: dict[str, Any], stamp: tuple[int, int]) -> Iterator[Any]:
        """A CDC file's rows, each with the kind of change it records."""
        import pyarrow as pa

        rows = self._read([action]).read_all()
        kinds = self.snapshot.file_column(action["path"], "_change_type", int(action["size"]))
        kinds = pa.table(kinds).column(0).combine_chunks()
        if len(kinds) != rows.num_rows:
            raise EngineLimitError(
                "read the change data feed",
                f"change file {action['path']} read {rows.num_rows} rows but "
                f"{len(kinds)} change types",
            )
        offset = 0
        for batch in rows.to_batches():
            n = batch.num_rows
            if n:
                yield self._stamped(batch, kinds.slice(offset, n).cast(pa.string()), stamp)
            offset += n
