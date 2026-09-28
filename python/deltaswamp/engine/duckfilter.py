"""A boolean expression evaluated by DuckDB over Arrow batches, one batch at a time.

Two reads filter rows by an expression no pyarrow kernel means the same by:
the kernel read of a table delta-rs misreads, under a Spark SQL predicate
outside the kernel's grammar (``abs(id) > 0``, respelled for DuckDB by
`engine.dialect`), and the lazy DuckDB hand-offs, whose filters DuckDB pushes
into the dataset as pyarrow Expressions and then no longer applies itself --
so pyarrow's NaN and -0.0 semantics answered a DuckDB query. Both evaluate
here, in a sandboxed in-memory DuckDB: no file or network access, no
extension loading, the configuration locked, and the relational API, which
parses one expression list and never a statement.
"""

from __future__ import annotations

from typing import Any

from .. import predicate as sqlpred

__all__ = ["SPARK_CONFIG", "RowFilter", "spark_widened"]

#: DuckDB settings for evaluating a Spark SQL predicate. DuckDB's expression
#: rewriter moves constants across a comparison, so ``rid + 9223372036854775807
#: > 0`` became ``rid > -9223372036854775807`` and kept every row where Spark
#: (ANSI) raises ARITHMETIC_OVERFLOW; without it the addition is evaluated and
#: overflows as it does on Databricks.
SPARK_CONFIG: dict[str, Any] = {"disabled_optimizers": "expression_rewriter"}


def _widened_type(pa: Any, kind: Any) -> Any:
    if pa.types.is_float32(kind) or pa.types.is_float16(kind):
        return pa.float64()
    if pa.types.is_struct(kind):
        fields = [kind.field(i) for i in range(kind.num_fields)]
        return pa.struct([f.with_type(_widened_type(pa, f.type)) for f in fields])
    if pa.types.is_large_list(kind):
        return pa.large_list(kind.value_field.with_type(_widened_type(pa, kind.value_type)))
    if pa.types.is_list(kind):
        return pa.list_(kind.value_field.with_type(_widened_type(pa, kind.value_type)))
    if pa.types.is_map(kind):
        return pa.map_(
            kind.key_field.with_type(_widened_type(pa, kind.key_type)),
            kind.item_field.with_type(_widened_type(pa, kind.item_type)),
        )
    return kind


def spark_widened(schema: Any) -> Any:
    """`schema` with every FLOAT as DOUBLE, or None when it has no FLOAT.

    Spark compares a FLOAT with a decimal literal (``f = 0.1``) as DOUBLE,
    both sides widened; DuckDB narrowed the literal to FLOAT instead, so
    ``abs(f) = 0.1`` matched the FLOAT 0.1 that Spark does not. A FLOAT
    widens to DOUBLE exactly, so FLOAT-to-FLOAT comparisons are unchanged.
    """
    import pyarrow as pa

    widened = pa.schema(
        [f.with_type(_widened_type(pa, f.type)) for f in schema], metadata=schema.metadata
    )
    return None if widened == schema else widened


class RowFilter:
    """`text` evaluated per row of each batch; the rows where it is TRUE are kept.

    Binds the expression against `schema` when made, so an unknown column or
    a function DuckDB lacks fails inside the call rather than in the middle of
    a stream. Failures are PredicateError. `close()` releases the connection.
    """

    def __init__(self, text: str, schema: Any, *, spark: bool = False) -> None:
        import duckdb
        import pyarrow as pa

        from .dialect import install_duckdb_macros

        self.text = text
        self.expression = f"CAST(({text}) AS BOOLEAN) AS __deltaswamp_keep"
        # `spark`: the text is a Spark SQL predicate respelled for DuckDB, to
        # be evaluated as Spark evaluates it (`SPARK_CONFIG`, `spark_widened`).
        # Without it, a lazy hand-off's DuckDB filter keeps DuckDB's meaning.
        self.widened = spark_widened(schema) if spark else None
        config: dict[str, Any] = {
            "enable_external_access": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            **(SPARK_CONFIG if spark else {}),
        }
        self.con = duckdb.connect(":memory:", config=config)
        schema = self.widened or schema
        try:
            install_duckdb_macros(self.con)
            # Databricks evaluates a zoned literal in the session zone, UTC by
            # default -- not the zone of the machine this runs on.
            self.con.execute("SET TimeZone = 'UTC'")
            self.con.execute("SET lock_configuration = true")
            self.con.from_arrow(pa.table(schema.empty_table())).project(self.expression)
        except duckdb.Error as exc:
            self.close()
            raise sqlpred.PredicateError(f"cannot evaluate {text!r}: {exc}") from None

    def __call__(self, batch: Any) -> Any:
        import duckdb
        import pyarrow as pa
        import pyarrow.compute as pc

        if batch.num_rows == 0:
            return batch
        try:
            rows = pa.Table.from_batches([batch])
            if self.widened is not None:
                rows = rows.cast(self.widened)
            result = self.con.from_arrow(rows).project(self.expression)
            keep = result.arrow()
            if isinstance(keep, pa.RecordBatchReader):
                keep = keep.read_all()
        except duckdb.Error as exc:
            # The expression failed on the data (a cast of a string that is not
            # a number), as it fails on Databricks.
            raise sqlpred.PredicateError(f"cannot evaluate {self.text!r}: {exc}") from None
        if keep.num_rows != batch.num_rows:
            raise sqlpred.PredicateError(f"{self.text!r} is not one boolean expression per row")
        mask = pc.fill_null(keep.column(0).combine_chunks().cast(pa.bool_()), False)
        return batch.filter(mask)

    def filtered(self, reader: Any, keep: list[str] | None = None) -> Any:
        """`reader` filtered (and narrowed to `keep`), as a RecordBatchReader.

        The connection is closed once the stream ends. Errors raised while
        reading keep their Python type for a Python consumer.
        """
        import pyarrow as pa

        schema = reader.schema
        if keep is not None:
            schema = pa.schema([schema.field(n) for n in keep])

        def batches() -> Any:
            try:
                for batch in reader:
                    kept = self(batch)
                    yield kept if keep is None else kept.select(keep)
            finally:
                self.close()

        return pa.RecordBatchReader.from_batches(schema, batches())

    def close(self) -> None:
        self.con.close()
