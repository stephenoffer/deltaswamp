"""CHECK constraints, evaluated over the rows a kernel write commits.

delta-kernel 0.28 marks `checkConstraints` NotSupported for writes and refuses
any transaction on a table whose protocol carries it, constraint or none. The
kernel write paths here evaluate every constraint over every row they write
instead, as Spark does -- a row passes when the expression is TRUE or NULL and
fails the write when it is FALSE -- and tell the native commit so
(``constraints_checked=True``), which then commits past the kernel's refusal
(see `crate::restate`). The same evaluation validates the existing rows before
ADD CONSTRAINT commits.

Expressions are Spark SQL, respelled for DuckDB by `engine.dialect` and
evaluated in the sandboxed in-memory DuckDB `engine.duckfilter` sets up (no
file or network access, no extensions, the configuration locked), with Spark's
arithmetic and comparison semantics. One DuckDB cannot be made to evaluate as
Databricks does is refused before anything is written.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping
from typing import Any

from ..errors import EngineLimitError, InvalidArgumentError

__all__ = [
    "CONSTRAINT_PREFIX",
    "ConstraintCheck",
    "enforcement_refusal",
    "table_constraints",
]

CONSTRAINT_PREFIX = "delta.constraints."


def table_constraints(properties: Mapping[str, Any] | None) -> dict[str, str]:
    """The table's CHECK constraints, name -> Spark SQL expression."""
    return {
        str(key)[len(CONSTRAINT_PREFIX) :]: str(value)
        for key, value in (properties or {}).items()
        if str(key).lower().startswith(CONSTRAINT_PREFIX)
    }


def duckdb_text(expression: str) -> str:
    """`expression` as DuckDB evaluates it the way Databricks does.

    Raises `EngineLimitError` for SQL DuckDB cannot be made to compute as
    Spark does, and `PredicateError` for text that is not one expression over
    the row.
    """
    from . import sharing
    from .dialect import to_duckdb

    text = to_duckdb(expression)
    sharing._screen_expression(text)
    return text


def enforcement_refusal(constraints: Mapping[str, str]) -> str | None:
    """Why the kernel paths cannot evaluate `constraints` here, if they cannot."""
    if not constraints:
        return None
    if importlib.util.find_spec("duckdb") is None or importlib.util.find_spec("pyarrow") is None:
        return (
            "the table has CHECK constraints, which the kernel write paths evaluate with "
            "DuckDB, and DuckDB is not installed (pip install 'deltaswamp[duckdb]')"
        )
    for name, expression in sorted(constraints.items()):
        try:
            duckdb_text(expression)
        except EngineLimitError as exc:
            return (
                f"the table's CHECK constraint {name} ({expression}) uses SQL DuckDB cannot "
                f"evaluate as Databricks does ({exc.reason})"
            )
        except Exception as exc:  # malformed: another engine's parser may read it
            return f"the table's CHECK constraint {name} ({expression}) cannot be read: {exc}"
    return None


def _describe(batch: Any, index: int, limit: int = 8) -> str:
    """One row of `batch`, as Spark's DELTA_VIOLATE_CONSTRAINT_WITH_VALUES lists it."""
    values = []
    for name in batch.schema.names[:limit]:
        values.append(f"{name} : {batch.column(name)[index].as_py()!r}")
    more = ", ..." if batch.num_columns > limit else ""
    return ", ".join(values) + more


class ConstraintCheck:
    """`constraints` evaluated over batches of rows under `schema` (the table's).

    A batch may leave out a nullable column (it is NULL there, as the write
    fills it) and name columns in any case; DuckDB binds names without regard
    to case. Each batch shape gets its own sandboxed connection, bound once.
    `close()` releases them.
    """

    def __init__(
        self,
        constraints: Mapping[str, str],
        schema: Any,
        *,
        what: str,
        violation: str = "the data violates the table's constraints, so nothing was written",
    ) -> None:
        self.constraints = {name: (expr, duckdb_text(expr)) for name, expr in constraints.items()}
        self.schema = schema
        self.what = what
        self.violation = violation
        self._filters: dict[Any, list[tuple[str, str, Any]]] = {}

    def _complete(self, batch: Any) -> Any:
        """`batch` with a NULL column for every table column it leaves out."""
        import pyarrow as pa

        have = {name.lower() for name in batch.schema.names}
        missing = [f for f in self.schema if f.name.lower() not in have]
        if not missing:
            return batch
        arrays = list(batch.columns) + [pa.nulls(batch.num_rows, f.type) for f in missing]
        names = list(batch.schema.names) + [f.name for f in missing]
        return pa.RecordBatch.from_arrays(arrays, names=names)

    def _bound(self, schema: Any) -> list[tuple[str, str, Any]]:
        bound = self._filters.get(schema)
        if bound is not None:
            return bound
        bound = []
        try:
            for name, (expression, text) in sorted(self.constraints.items()):
                bound.append((name, expression, self._filter(name, expression, text, schema)))
        except BaseException:
            for *_, check in bound:
                check.close()
            raise
        self._filters[schema] = bound
        return bound

    @staticmethod
    def _filter(name: str, expression: str, text: str, schema: Any) -> Any:
        """The rows of a batch under `schema` where the constraint is FALSE.

        NOT of NULL is NULL, and a NULL result passes, as Spark has it.
        """
        from ..predicate import PredicateError
        from .duckfilter import RowFilter

        try:
            check = RowFilter(f"NOT ({text})", schema, spark=True)
        except PredicateError as exc:
            raise InvalidArgumentError(
                f"CHECK constraint {name} ({expression}) cannot be evaluated over the "
                f"table's columns: {exc}"
            ) from None
        kind = check.con.from_arrow(schema.empty_table()).project(f"({text})").types[0]
        if str(kind).upper() != "BOOLEAN":
            # CAST(... AS BOOLEAN) takes a number too; Spark refuses a
            # constraint that is not a condition.
            check.close()
            raise InvalidArgumentError(
                f"CHECK constraint {name} ({expression}) is not a boolean expression: "
                f"it is {str(kind).lower()}"
            )
        return check

    def bind(self) -> None:
        """Bind every constraint against the table schema: an unknown column or a
        non-boolean expression is refused before any row is read."""
        import pyarrow as pa

        self._bound(pa.schema(list(self.schema)))

    def check(self, batch: Any) -> None:
        """Raise `InvalidArgumentError` when a row of `batch` violates a constraint."""
        if batch.num_rows == 0 or not self.constraints:
            return
        complete = self._complete(batch)
        for name, expression, check in self._bound(complete.schema):
            violating = check(complete)
            if violating.num_rows:
                raise InvalidArgumentError(
                    f"{self.violation}: CHECK constraint {name} ({expression}) violated by "
                    f"{violating.num_rows} row(s) of {self.what}, e.g. "
                    f"{_describe(violating, 0)}"
                )

    def checked(self, reader: Any) -> Any:
        """`reader`, each batch checked as it is read; the check closes at its end."""
        import pyarrow as pa

        def batches() -> Iterator[Any]:
            try:
                for batch in reader:
                    self.check(batch)
                    yield batch
            finally:
                self.close()

        return pa.RecordBatchReader.from_batches(reader.schema, batches())

    def check_table(self, table: Any) -> None:
        """`check` over every batch of an Arrow table, then close."""
        try:
            for batch in table.to_batches():
                self.check(batch)
        finally:
            self.close()

    def close(self) -> None:
        for bound in self._filters.values():
            for _name, _expression, check in bound:
                check.close()
        self._filters.clear()
