"""Rows a kernel DML holds, spilled to local Parquet instead of memory.

A MERGE's target and source, its updated and inserted rows, and the positions
of the rows it deletes can each be far larger than memory. They are written
here as they are produced and read back as often as needed: a commit that
loses a race and rebases writes the same rows again, from the same files.

Everything lives under one temporary directory, removed when the spill is
closed (or collected): the rows are the table's data, so they never outlive
the operation that wrote them.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import weakref
from collections.abc import Callable, Iterable, Iterator
from typing import Any

__all__ = ["Rows", "Spill", "spill_directory"]

#: Rows per Parquet row group and per batch read back: small enough that a
#: batch of wide rows stays a few MB, large enough to write efficiently.
_BATCH_ROWS = 65_536


def spill_directory(root: str | None, prefix: str) -> str:
    """A fresh private directory under `root` (the system temp dir when None)."""
    if root is not None:
        os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=root)


class Spill:
    """Rows of one schema, written to Parquet as they arrive and replayable.

    `track` names columns whose distinct values are kept as the rows go by
    (the data-file paths of deletion positions: one per file, not per row).
    """

    def __init__(
        self,
        schema: Any,
        directory: str,
        name: str = "rows",
        *,
        track: Iterable[str] = (),
    ) -> None:
        # Every field nullable: the spill holds rows before the commit checks
        # them, and a NULL bound for a NOT NULL column is refused there, with
        # the error that names it, not by the Parquet writer here.
        self.schema = _nullable(schema)
        self._path = os.path.join(directory, f"{name}.parquet")
        self._writer: Any = None
        self.num_rows = 0
        #: The rows' Arrow size, as they were written (not the file's).
        self.nbytes = 0
        self._track: dict[str, set[Any]] = {column: set() for column in track}

    # ------------------------------------------------------------- writing

    def write(self, rows: Any) -> None:
        """Append a record batch, a table, or every batch of a reader."""
        import pyarrow as pa

        if isinstance(rows, pa.RecordBatch):
            self._write_batch(rows)
        elif isinstance(rows, pa.Table):
            for batch in rows.to_batches(max_chunksize=_BATCH_ROWS):
                self._write_batch(batch)
        else:
            for batch in rows:
                self._write_batch(batch)

    def _write_batch(self, batch: Any) -> None:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        if batch.num_rows == 0:
            return
        if batch.schema != self.schema:
            batch = batch.cast(self.schema)
        if self._writer is None:
            # zstd at its fastest level: the spill is read back once or twice,
            # and disk bandwidth, not CPU, bounds a large MERGE.
            self._writer = pq.ParquetWriter(
                self._path, self.schema, compression="zstd", compression_level=1
            )
        self._writer.write_batch(batch, row_group_size=_BATCH_ROWS)
        self.num_rows += batch.num_rows
        self.nbytes += batch.nbytes
        for column, seen in self._track.items():
            seen.update(pc.unique(batch.column(column)).to_pylist())

    def finish(self) -> Spill:
        """Close the file for writing; reading needs it finished."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self

    # ------------------------------------------------------------- reading

    def distinct(self, column: str) -> set[Any]:
        """The distinct values of a tracked column."""
        return set(self._track[column])

    def batches(self) -> Iterator[Any]:
        import pyarrow.parquet as pq

        self.finish()
        if self.num_rows == 0:
            return
        with pq.ParquetFile(self._path) as file:
            yield from file.iter_batches(batch_size=_BATCH_ROWS)

    def to_reader(self) -> Any:
        """A new reader over every row written, in the order written."""
        import pyarrow as pa

        return pa.RecordBatchReader.from_batches(self.schema, self.batches())

    def dataset(self) -> Any:
        """The rows as a pyarrow dataset, which DuckDB scans as often as a
        query needs without holding them."""
        import pyarrow.dataset as pads

        self.finish()
        if self.num_rows == 0:
            return pads.dataset(self.schema.empty_table())
        return pads.dataset(self._path, schema=self.schema, format="parquet")

    def close(self) -> None:
        """Remove the file: its rows are not read again."""
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
            self._writer = None
        with contextlib.suppress(OSError):
            os.remove(self._path)


class Rows:
    """Rows produced on demand: each `to_reader()` runs `produce` again.

    For rows derived from others -- a copy-on-write rewrite's kept rows --
    that a retried commit writes again without holding them in between.
    """

    num_rows: int | None = None

    def __init__(self, schema: Any, produce: Callable[[], Iterator[Any]]) -> None:
        self.schema = schema
        self._produce = produce

    def to_reader(self) -> Any:
        import pyarrow as pa

        return pa.RecordBatchReader.from_batches(self.schema, self._produce())


class SpillDirectory:
    """One operation's spill directory, removed on close or collection."""

    def __init__(self, root: str | None, prefix: str) -> None:
        self.path = spill_directory(root, prefix)
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.path, True)

    def spill(self, schema: Any, name: str, **kwargs: Any) -> Spill:
        return Spill(schema, self.path, name, **kwargs)

    def close(self) -> None:
        self._finalizer()

    def __enter__(self) -> SpillDirectory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _nullable(schema: Any) -> Any:
    """`schema` with every field, nested ones included, nullable."""
    import pyarrow as pa

    def relax(datatype: Any) -> Any:
        if pa.types.is_struct(datatype):
            return pa.struct([field_of(datatype.field(i)) for i in range(datatype.num_fields)])
        if pa.types.is_large_list(datatype):
            return pa.large_list(field_of(datatype.value_field))
        if pa.types.is_fixed_size_list(datatype):
            return pa.list_(field_of(datatype.value_field), datatype.list_size)
        if pa.types.is_list(datatype):
            return pa.list_(field_of(datatype.value_field))
        if pa.types.is_map(datatype):
            return pa.map_(
                datatype.key_field.type,
                field_of(datatype.item_field),
                keys_sorted=datatype.keys_sorted,
            )
        return datatype

    def field_of(field: Any) -> Any:
        return pa.field(field.name, relax(field.type), True, field.metadata)

    return pa.schema([field_of(f) for f in schema], metadata=schema.metadata)
