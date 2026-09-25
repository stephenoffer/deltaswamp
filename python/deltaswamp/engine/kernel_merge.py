"""MERGE through the kernel, written as deletion vectors.

The kernel has no MERGE, so it is assembled here from the pieces DELETE and
UPDATE already use: a positional read of the target, the clauses evaluated in
DuckDB (so clause conditions and SET expressions are real SQL, with the source
and target aliases), and one `commit_dml` that marks every touched target row
deleted and appends the rewritten and inserted rows. That is how Databricks
writes a MERGE on a table with deletion vectors enabled.

Semantics follow Spark's MERGE:

* clauses of a kind are tried in order, and a row takes the first whose
  condition is TRUE (NULL counts as false);
* a target row matched by more than one source row is an error when a matched
  clause would modify it, since which source row wins would be arbitrary --
  except when every matched clause is an unconditional DELETE;
* a source row that matches no target row is offered to the NOT MATCHED
  clauses, a target row that matches no source row to the NOT MATCHED BY
  SOURCE clauses;
* on a table with row tracking enabled, updated rows keep their row ids.

Only the target files that can hold a match are read. The ON condition is
parsed for `target.col = source.col` conjuncts, and the target is skipped
with `col IN (<the source's values>)`, which the ON condition implies, so no
file holding a match can be skipped. A NOT MATCHED BY SOURCE clause needs
every target row, and turns skipping off.
"""

from __future__ import annotations

from typing import Any

from ..errors import InvalidArgumentError, UnreachableTableError

__all__ = ["KernelMerger"]

#: Positional columns a DV scan adds (see `_native.Snapshot.scan`).
_FILE = "__deltaswamp_file"
_INDEX = "__deltaswamp_row_index"
_ROW_ID = "__deltaswamp_row_id"
#: Our own row number for source rows.
_SRC_ROW = "__deltaswamp_source_row"
#: The most distinct join-key values turned into a skipping IN-list.
_SKIP_VALUES_LIMIT = 10_000


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


class KernelMerger:
    """The delta-rs `TableMerger` surface, executed through the kernel."""

    def __init__(
        self,
        engine: Any,
        table: Any,
        source: Any,
        predicate: str,
        *,
        source_alias: str | None = None,
        target_alias: str | None = None,
        **options: Any,
    ) -> None:
        import pyarrow as pa

        passthrough = {k: options.pop(k, None) for k in ("commit_metadata", "engine_info")}
        merge_schema = options.pop("merge_schema", False)
        if merge_schema:
            raise UnreachableTableError(
                "merge with merge_schema on the kernel",
                "a kernel MERGE writes rows in the table's schema and cannot evolve it",
            )
        given = sorted(k for k, v in options.items() if v is not None)
        if given:
            raise UnreachableTableError(
                f"merge with {', '.join(given)} on the kernel",
                "the kernel MERGE does not implement these options",
            )
        if not isinstance(predicate, str) or not predicate.strip():
            raise InvalidArgumentError(
                f"a MERGE needs a join predicate (its ON condition), got {predicate!r}"
            )
        self._source_alias = source_alias or "source"
        self._target_alias = target_alias or "target"
        if self._source_alias.lower() == self._target_alias.lower():
            raise InvalidArgumentError(
                f"the source and target aliases are both {self._source_alias!r}"
            )
        self._engine = engine
        self._table = table
        self._source = pa.table(source) if not isinstance(source, pa.Table) else source
        self._predicate = predicate
        self._passthrough = passthrough
        # (kind, condition, verb, argument)
        self._clauses: list[tuple[str, str | None, str, Any]] = []

    # ---------------------------------------------------------------- clauses

    def _add(self, kind: str, predicate: str | None, verb: str, arg: Any) -> KernelMerger:
        if verb in ("UPDATE", "INSERT") and not arg:
            raise InvalidArgumentError(f"a MERGE {verb} clause needs at least one column to set")
        self._clauses.append((kind, predicate, verb, arg))
        return self

    def when_matched_update(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> KernelMerger:
        return self._add("matched", predicate, "UPDATE", dict(updates))

    def when_matched_update_all(
        self, predicate: str | None = None, except_cols: list[str] | None = None
    ) -> KernelMerger:
        return self._add("matched", predicate, "UPDATE_ALL", list(except_cols or []))

    def when_matched_delete(self, predicate: str | None = None) -> KernelMerger:
        return self._add("matched", predicate, "DELETE", None)

    def when_not_matched_insert(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> KernelMerger:
        return self._add("not_matched", predicate, "INSERT", dict(updates))

    def when_not_matched_insert_all(
        self, predicate: str | None = None, except_cols: list[str] | None = None
    ) -> KernelMerger:
        return self._add("not_matched", predicate, "INSERT_ALL", list(except_cols or []))

    def when_not_matched_by_source_update(
        self, updates: dict[str, str], predicate: str | None = None
    ) -> KernelMerger:
        return self._add("not_matched_by_source", predicate, "UPDATE", dict(updates))

    def when_not_matched_by_source_delete(self, predicate: str | None = None) -> KernelMerger:
        return self._add("not_matched_by_source", predicate, "DELETE", None)

    # -------------------------------------------------------------- execution

    def execute(self) -> dict[str, Any]:
        import duckdb
        import pyarrow as pa

        if not self._clauses:
            raise InvalidArgumentError("a MERGE needs at least one WHEN clause")
        engine, table = self._engine, self._table
        snapshot = engine.snapshot(table, write=True)
        schema = pa.schema(snapshot.schema())
        by_source = any(k == "not_matched_by_source" for k, *_ in self._clauses)
        row_ids = _row_tracking_enabled(table) and any(
            verb == "UPDATE" or verb == "UPDATE_ALL"
            for kind, _, verb, _ in self._clauses
            if kind != "not_matched"
        )

        skipping = None if by_source else self._skipping(schema)
        extra = {"row_ids": True} if row_ids else {}
        target = pa.table(snapshot.scan(predicate=skipping, row_positions=True, **extra))
        source = self._source.append_column(
            _SRC_ROW, pa.array(range(self._source.num_rows), pa.int64())
        )

        con = duckdb.connect()
        try:
            con.register("__target", target)
            con.register("__source", source)
            result = self._evaluate(con, schema, target, row_ids)
        finally:
            con.close()
        deletions, data, metrics = result

        if deletions.num_rows == 0 and (data is None or data.num_rows == 0):
            return {**metrics, "version": int(snapshot.version)}
        result_version = engine._commit_dv_changes(
            table,
            snapshot,
            deletions,
            data,
            operation="MERGE",
            **self._passthrough,
        )
        return {**metrics, "version": int(result_version)}

    def _skipping(self, schema: Any) -> str | None:
        """A kernel skipping predicate the ON condition implies, or None."""
        import pyarrow as pa
        import pyarrow.compute as pc

        from .. import predicate as sqlpred

        try:
            node = sqlpred.parse(self._predicate)
        except Exception:
            return None
        conjuncts = list(node.args) if node.op == "and" else [node]
        target, source = self._target_alias.lower(), self._source_alias.lower()
        targets = {f.name.lower(): f.name for f in schema}
        sources = {name.lower(): name for name in self._source.column_names}
        parts: list[str] = []
        for part in conjuncts:
            if part.op != "eq" or len(part.args) != 2:
                continue
            sides = {}
            for arg in part.args:
                if isinstance(arg, sqlpred.Column) and len(arg.path) == 2:
                    sides[arg.path[0].lower()] = arg.path[1].lower()
            if set(sides) != {target, source}:
                continue
            column, key = targets.get(sides[target]), sources.get(sides[source])
            if column is None or key is None:
                continue
            values = pc.unique(self._source.column(key).drop_null())
            if len(values) == 0 or len(values) > _SKIP_VALUES_LIMIT:
                continue
            try:
                values = values.cast(schema.field(column).type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                continue
            literals = [_sql_literal(v) for v in values.to_pylist()]
            if any(lit is None for lit in literals):
                continue
            text = f"{_quote_sql_ident(column)} IN ({', '.join(literals)})"  # type: ignore[arg-type]
            try:
                sqlpred.parse(text)
            except Exception:
                continue
            parts.append(text)
        if not parts:
            return None
        try:
            return sqlpred.to_kernel_json(sqlpred.parse(" AND ".join(parts)), schema)
        except Exception:
            return None

    def _evaluate(
        self, con: Any, schema: Any, target: Any, row_ids: bool
    ) -> tuple[Any, Any, dict[str, Any]]:
        import pyarrow as pa

        t, s = _quote(self._target_alias), _quote(self._source_alias)
        on = f"({self._predicate})"
        positions = f"{t}.{_quote(_FILE)} AS path, {t}.{_quote(_INDEX)} AS row_index"
        carried = f", {t}.{_quote(_ROW_ID)} AS {_quote(_ROW_ID)}" if row_ids else ""
        columns = [f.name for f in schema]

        clauses = {
            kind: [c for c in self._clauses if c[0] == kind]
            for kind in ("matched", "not_matched", "not_matched_by_source")
        }

        # A target row matched by several source rows cannot be updated
        # deterministically.
        matched = clauses["matched"]
        only_deletes = all(verb == "DELETE" and cond is None for _, cond, verb, _ in matched)
        if matched and not only_deletes:
            dup = con.execute(
                f"SELECT count(*) FROM (SELECT {t}.{_quote(_FILE)}, {t}.{_quote(_INDEX)} "
                f"FROM __target AS {t} JOIN __source AS {s} ON {on} "
                f"GROUP BY 1, 2 HAVING count(*) > 1)"
            ).fetchone()[0]
            if dup:
                raise InvalidArgumentError(
                    f"the MERGE matched {dup} target row(s) with more than one source row; "
                    "which source row updates it would be arbitrary. Deduplicate the source "
                    "on the join key"
                )

        deletions: list[Any] = []
        outputs: list[Any] = []
        counts = {"updated": 0, "deleted": 0, "inserted": 0}

        def applies(kind_clauses: list[Any], index: int) -> str:
            own = kind_clauses[index][1]
            parts = [f"COALESCE(({own}), FALSE)"] if own else []
            parts += [f"NOT COALESCE(({c[1]}), FALSE)" for c in kind_clauses[:index] if c[1]]
            # An earlier unconditional clause takes every row.
            if any(c[1] is None for c in kind_clauses[:index]):
                return "FALSE"
            return " AND ".join(parts) or "TRUE"

        for kind, relation in (
            ("matched", f"__target AS {t} JOIN __source AS {s} ON {on}"),
            (
                "not_matched_by_source",
                f"__target AS {t} WHERE NOT EXISTS (SELECT 1 FROM __source AS {s} WHERE {on})",
            ),
        ):
            for index, (_, _, verb, arg) in enumerate(clauses[kind]):
                condition = applies(clauses[kind], index)
                where = "WHERE" if kind == "matched" else "AND"
                sql_from = f"FROM {relation} {where} {condition}"
                if verb == "DELETE":
                    part = _fetch(con, f"SELECT {positions} {sql_from}")
                    deletions.append(part)
                    counts["deleted"] += part.num_rows
                    continue
                exprs = self._assignments(verb, arg, columns, kind)
                select = ", ".join(f"{e} AS {_quote(c)}" for c, e in exprs.items())
                part = _fetch(con, f"SELECT {positions}{carried}, {select} {sql_from}")
                deletions.append(part.select(["path", "row_index"]))
                outputs.append(part.drop_columns(["path", "row_index"]))
                counts["updated"] += part.num_rows

        not_matched = clauses["not_matched"]
        for index, (_, _, verb, arg) in enumerate(not_matched):
            condition = applies(not_matched, index)
            exprs = self._insert_values(verb, arg, columns)
            select = ", ".join(f"{e} AS {_quote(c)}" for c, e in exprs.items())
            part = _fetch(
                con,
                f"SELECT {select} FROM __source AS {s} WHERE NOT EXISTS "
                f"(SELECT 1 FROM __target AS {t} WHERE {on}) AND {condition}",
            )
            if row_ids:
                part = part.append_column(_ROW_ID, pa.nulls(part.num_rows, pa.int64()))
            outputs.append(part)
            counts["inserted"] += part.num_rows

        deletion_schema = pa.schema([("path", pa.string()), ("row_index", pa.int64())])
        deletion_table = (
            pa.concat_tables([d.cast(deletion_schema) for d in deletions])
            if deletions
            else deletion_schema.empty_table()
        )
        data = None
        if outputs:
            target_schema = pa.schema(
                list(schema) + ([pa.field(_ROW_ID, pa.int64())] if row_ids else [])
            )
            data = pa.concat_tables([_cast_to(o, target_schema) for o in outputs])
        metrics = {
            "num_source_rows": self._source.num_rows,
            "num_target_rows_updated": counts["updated"],
            "num_target_rows_deleted": counts["deleted"],
            "num_target_rows_inserted": counts["inserted"],
            "num_updated_rows": counts["updated"],
            "num_deleted_rows": counts["deleted"],
            "num_inserted_rows": counts["inserted"],
        }
        return deletion_table, data, metrics

    def _target_column(self, key: Any, columns: list[str]) -> str:
        text = str(key)
        prefix = self._target_alias + "."
        if text.lower().startswith(prefix.lower()) and len(text) > len(prefix):
            text = text[len(prefix) :]
        text = text.strip('`"')
        for column in columns:
            if column.lower() == text.lower():
                return column
        raise InvalidArgumentError(f"the MERGE sets {key!r}, which is not a target column")

    def _assignments(self, verb: str, arg: Any, columns: list[str], kind: str) -> dict[str, str]:
        """The new value of every target column, for an UPDATE clause."""
        t, s = _quote(self._target_alias), _quote(self._source_alias)
        exprs = {c: f"{t}.{_quote(c)}" for c in columns}
        if verb == "UPDATE":
            for key, value in arg.items():
                exprs[self._target_column(key, columns)] = f"({value})"
            return exprs
        # UPDATE_ALL: every column the source has, except `except_cols`.
        if kind != "matched":
            raise InvalidArgumentError("update_all is only valid for matched rows")
        excluded = {str(c).lower() for c in arg}
        sources = {n.lower(): n for n in self._source.column_names if n != _SRC_ROW}
        for column in columns:
            name = sources.get(column.lower())
            if name is not None and column.lower() not in excluded:
                exprs[column] = f"{s}.{_quote(name)}"
        return exprs

    def _insert_values(self, verb: str, arg: Any, columns: list[str]) -> dict[str, str]:
        s = _quote(self._source_alias)
        exprs = {c: "NULL" for c in columns}
        if verb == "INSERT":
            for key, value in arg.items():
                exprs[self._target_column(key, columns)] = f"({value})"
            return exprs
        excluded = {str(c).lower() for c in arg}
        sources = {n.lower(): n for n in self._source.column_names if n != _SRC_ROW}
        for column in columns:
            name = sources.get(column.lower())
            if name is not None and column.lower() not in excluded:
                exprs[column] = f"{s}.{_quote(name)}"
        return exprs


def _fetch(con: Any, sql: str) -> Any:
    """A query's result as a `pyarrow.Table`, on every supported DuckDB.

    `.arrow()` returned a Table before DuckDB 1.4 and a RecordBatchReader
    since; `pa.table` accepts both.
    """
    import pyarrow as pa

    return pa.table(con.execute(sql).arrow())


def _row_tracking_enabled(table: Any) -> bool:
    return str(table.properties.get("delta.enableRowTracking", "false")).lower() == "true"


def _cast_to(data: Any, schema: Any) -> Any:
    """`data` with `schema`'s columns and types, by name."""
    import pyarrow as pa

    columns = []
    for field in schema:
        if field.name in data.column_names:
            columns.append(data.column(field.name).cast(field.type))
        else:
            columns.append(pa.nulls(data.num_rows, field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def _quote_sql_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _sql_literal(value: Any) -> str | None:
    """`value` as a SQL literal the predicate parser reads, or None if it has none.

    Only exact types: a float or timestamp spelled back as text could round,
    and a skipping predicate that is off by one ulp skips a file with a match.
    """
    import datetime as dt

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return f"DATE '{value.isoformat()}'"
    return None
