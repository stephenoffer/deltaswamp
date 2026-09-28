"""MERGE through the kernel, written as deletion vectors or copy-on-write.

The kernel has no MERGE, so it is assembled here from the pieces DELETE and
UPDATE already use: a positional read of the target, the clauses evaluated in
DuckDB (so clause conditions and SET expressions are real SQL, with the source
and target aliases), and one `commit_dml` that marks every touched target row
deleted and appends the rewritten and inserted rows. That is how Databricks
writes a MERGE on a table with deletion vectors enabled. On a table without
them, the same commit removes every touched file instead and writes its
untouched rows again beside the new ones (`KernelEngine._as_file_rewrites`),
as Spark's copy-on-write MERGE does.

Semantics follow Spark's MERGE:

* clauses of a kind are tried in order, and a row takes the first whose
  condition is TRUE (NULL counts as false);
* a target row matched by more than one source row is an error when a matched
  clause would modify it, since which source row wins would be arbitrary --
  except when every matched clause is an unconditional DELETE;
* a source row that matches no target row is offered to the NOT MATCHED
  clauses, a target row that matches no source row to the NOT MATCHED BY
  SOURCE clauses;
* on a table with row tracking enabled, updated rows keep their row ids (and
  a copy-on-write rewrite's untouched rows their commit versions too), and
  inserted rows get fresh ones.

Clause text is Spark SQL, as everywhere else in the API (and on the
warehouse, which runs the same text). DuckDB evaluates it, so `_spark_sql`
first rewrites it (see `dialect`) into DuckDB SQL that means the same: string
literals, `DIV`, `RLIKE`, CAST truncation, `substring`, `concat` and the other
places where the two dialects differ. `DEFAULT` as a SET or INSERT value, and
a column an INSERT leaves out, take the column's DEFAULT. Values are stored
into the target column as Spark's store assignment does: fractions truncate
into an integer column and round half-up into a narrower decimal.

Only the target files that can hold a match are read. The ON condition is
parsed for `target.col = source.col` conjuncts, and the target is skipped
with `col IN (<the source's values>)`, which the ON condition implies, so no
file holding a match can be skipped. A NOT MATCHED BY SOURCE clause needs
every target row, and turns skipping off.
"""

from __future__ import annotations

from typing import Any

from ..errors import EngineLimitError, InvalidArgumentError, UnreachableTableError

__all__ = ["KernelMerger"]

#: Positional columns a DV scan adds (see `_native.Snapshot.scan`).
_FILE = "__deltaswamp_file"
_INDEX = "__deltaswamp_row_index"
_ROW_ID = "__deltaswamp_row_id"
#: Our own row number for source rows.
_SRC_ROW = "__deltaswamp_source_row"
#: The touched rows' positions in the evaluated SELECTs. Reserved names, so a
#: target column called `path` or `row_index` cannot collide with them.
_POS_FILE = "__deltaswamp_pos_file"
_POS_INDEX = "__deltaswamp_pos_row_index"
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
        #: A pinned handle's version: read there, conflict-checked at commit.
        self._read_version = options.pop("read_version", None)
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
        #: SQL DuckDB cannot be made to evaluate as Spark does, found while the
        #: clauses were added; raised at execute, before anything is read or
        #: written, so `Table` can hand the MERGE to another engine.
        self._refusal: EngineLimitError | None = None
        self._on = self._translate(predicate)
        self._passthrough = passthrough
        #: Lower-cased VARIANT columns of the target, from the log.
        self._variants: frozenset[str] = frozenset()
        # (kind, condition, verb, argument)
        self._clauses: list[tuple[str, str | None, str, Any]] = []
        self._by_source_conditions: list[str | None] = []

    def _translate(self, text: str) -> str:
        from .dialect import check_expression

        # Malformed text is the caller's mistake for every engine: a
        # PredicateError, never an EngineLimitError another engine retries.
        check_expression(text, "the MERGE clause text")
        try:
            return _spark_sql(text)
        except EngineLimitError as exc:
            self._refusal = self._refusal or exc
            return "FALSE"

    # ---------------------------------------------------------------- clauses

    def _add(self, kind: str, predicate: str | None, verb: str, arg: Any) -> KernelMerger:
        if verb in ("UPDATE", "INSERT") and not arg:
            raise InvalidArgumentError(f"a MERGE {verb} clause needs at least one column to set")
        if any(k == kind and c is None for k, c, _, _ in self._clauses):
            # Spark refuses this at parse time; taking the first silently
            # dropped every later clause of the kind.
            raise InvalidArgumentError(
                f"a {_KIND_NAMES[kind]} clause follows an unconditional one, which takes "
                "every row; only the last clause of a kind may omit its condition"
            )
        if kind == "not_matched_by_source":
            # As written: the skipping predicate is built from it (`_skipping`).
            self._by_source_conditions.append(predicate)
        if predicate is not None:
            predicate = self._translate(predicate)
        if verb in ("UPDATE", "INSERT"):
            # `SET c = DEFAULT` is the column's DEFAULT, as in Spark; resolved
            # against the schema at execute.
            arg = {
                key: _DEFAULT if _is_default_keyword(value) else self._translate(_expression(value))
                for key, value in arg.items()
            }
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
        if self._refusal is not None:
            raise self._refusal
        engine, table = self._engine, self._table
        snapshot = engine.snapshot(table, version=self._read_version, write=True)
        schema = pa.schema(snapshot.schema())
        if table.features & {"variantType", "variantType-preview"}:
            from .._variant import log_variant_columns

            self._variants = log_variant_columns(getattr(snapshot, "metadata_json", dict)())
        row_ids = _row_tracking_enabled(table) and any(
            verb == "UPDATE" or verb == "UPDATE_ALL"
            for kind, _, verb, _ in self._clauses
            if kind != "not_matched"
        )

        skipping = self._skipping(schema)
        extra = {"row_ids": True} if row_ids else {}
        target = pa.table(snapshot.scan(predicate=skipping, row_positions=True, **extra))
        source = self._source.append_column(
            _SRC_ROW, pa.array(range(self._source.num_rows), pa.int64())
        )

        from .dialect import install_duckdb_macros

        # The clause text is the caller's SQL, spliced into the statements
        # below. It evaluates in the same sandbox as the library's other DuckDB
        # expressions (`duckfilter`): no file or network access, no extension
        # loading, the configuration locked -- read_text() or COPY in a SET
        # value otherwise read or wrote local files.
        con = duckdb.connect(
            ":memory:",
            config={
                "enable_external_access": False,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            },
        )
        try:
            install_duckdb_macros(con)
            con.register("__target", target)
            con.register("__source", source)
            con.execute("SET lock_configuration = true")
            result = self._evaluate(con, schema, target, row_ids)
        except (
            duckdb.ParserException,
            duckdb.BinderException,
            duckdb.CatalogException,
            duckdb.NotImplementedException,
        ) as exc:
            # SQL this dialect bridge does not cover (a Spark function DuckDB
            # lacks, say). Nothing is written yet, so another engine may try.
            raise EngineLimitError(
                "merge on the kernel",
                f"DuckDB, which evaluates the kernel MERGE's clauses, cannot run them: {exc}",
                "delta-rs or the SQL warehouse (allow_sql_fallback=True) evaluate other "
                "SQL; or rewrite the clause",
            ) from exc
        except duckdb.Error as exc:
            # The clauses ran and failed on the data -- a division by zero, a
            # string that is not a number -- as they fail on Databricks.
            # Nothing is written.
            raise InvalidArgumentError(f"the MERGE failed evaluating its clauses: {exc}") from exc
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
            # The rows read are the ones this bounds, so only files it keeps
            # can hold a row a concurrent MERGE added that this one must see.
            read_predicate=skipping,
            predicate=self._predicate,
            **self._passthrough,
        )
        return {**metrics, "version": int(result_version)}

    def _skipping(self, schema: Any) -> str | None:
        """A kernel skipping predicate the ON condition implies, or None.

        Only target rows some source row can match are read: each
        ``target.k = source.s`` conjunct bounds ``k`` to the source's values,
        as an IN-list, or past `_SKIP_VALUES_LIMIT` distinct values as their
        range (a 20,000-key source used to read -- and hold -- every row of
        the table). A NOT MATCHED BY SOURCE clause acts on the other rows
        too, so with one the rows its condition may select are read as well,
        and with an unconditional one every row is.
        """
        from .. import predicate as sqlpred

        bounds = self._key_bounds(schema)
        if not self._by_source_conditions:
            if not bounds:
                return None
            try:
                return sqlpred.to_kernel_json(sqlpred.parse(" AND ".join(bounds)), schema)
            except Exception:
                return None
        if not bounds or any(c is None for c in self._by_source_conditions):
            return None
        try:
            alternatives = [sqlpred.parse(" AND ".join(bounds))]
            for condition in self._by_source_conditions:
                alternatives.append(self._unqualified(sqlpred.parse(str(condition)), schema))
        except Exception:
            return None
        # An OR renders only when every alternative does, so an unreadable
        # condition skips nothing rather than too much.
        return sqlpred.to_kernel_json(sqlpred.Node("or", tuple(alternatives)), schema)

    def _unqualified(self, node: Any, schema: Any) -> Any:
        """`node` with ``target.col`` spelled ``col``; raises on any other column."""
        from .. import predicate as sqlpred

        names = {f.name.lower() for f in schema}
        target = self._target_alias.lower()

        def walk(value: Any) -> Any:
            if isinstance(value, sqlpred.Column):
                path = value.path
                if len(path) >= 2 and path[0].lower() == target:
                    path = path[1:]
                elif path[0].lower() not in names or path[0].lower() == target:
                    raise ValueError("not a target column")
                return sqlpred.Column(tuple(path))
            if isinstance(value, sqlpred.Node):
                return sqlpred.Node(value.op, tuple(walk(a) for a in value.args), value.negated)
            return value

        return walk(node)

    def _key_bounds(self, schema: Any) -> list[str]:
        """``col IN (...)`` / ``col BETWEEN lo AND hi`` for each key the ON clause equates."""
        import pyarrow as pa
        import pyarrow.compute as pc

        from .. import predicate as sqlpred

        try:
            node = sqlpred.parse(self._predicate)
        except Exception:
            return []
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
            if len(values) == 0:
                continue
            if not _same_comparison_type(values.type, schema.field(column).type):
                # The ON clause compares with type coercion ('01' = 1 is
                # true), while a bound cast to the target's type is 1 -> '1':
                # the file holding '01' was skipped, its row read as
                # unmatched, and the source row inserted a second time.
                continue
            try:
                values = values.cast(schema.field(column).type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                continue
            if len(values) > _SKIP_VALUES_LIMIT:
                if pa.types.is_floating(values.type):
                    continue  # NaN sorts above every bound; leave it unbounded
                try:
                    low, high = (v.as_py() for v in pc.min_max(values).values())
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    continue
                ends = [_sql_literal(low), _sql_literal(high)]
                if any(e is None for e in ends):
                    continue
                text = f"{_quote_sql_ident(column)} BETWEEN {ends[0]} AND {ends[1]}"
            else:
                literals = [_sql_literal(v) for v in values.to_pylist()]
                if any(lit is None for lit in literals):
                    continue
                text = f"{_quote_sql_ident(column)} IN ({', '.join(literals)})"  # type: ignore[arg-type]
            try:
                sqlpred.parse(text)
            except Exception:
                continue
            parts.append(text)
        return parts

    def _evaluate(
        self, con: Any, schema: Any, target: Any, row_ids: bool
    ) -> tuple[Any, Any, dict[str, Any]]:
        import pyarrow as pa

        t, s = _quote(self._target_alias), _quote(self._source_alias)
        on = f"({self._on})"
        positions = (
            f"{t}.{_quote(_FILE)} AS {_quote(_POS_FILE)}, "
            f"{t}.{_quote(_INDEX)} AS {_quote(_POS_INDEX)}"
        )
        carried = f", {t}.{_quote(_ROW_ID)} AS {_quote(_ROW_ID)}" if row_ids else ""
        columns = [f.name for f in schema]

        clauses = {
            kind: [c for c in self._clauses if c[0] == kind]
            for kind in ("matched", "not_matched", "not_matched_by_source")
        }

        # A target row matched by several source rows cannot be updated
        # deterministically. As in Spark, only the pairs some MATCHED clause
        # would act on count: two source rows of which neither satisfies a
        # clause condition leave the row alone, which is no conflict.
        matched = clauses["matched"]
        only_deletes = all(verb == "DELETE" and cond is None for _, cond, verb, _ in matched)
        if matched and not only_deletes:
            conditions = [cond for _, cond, _, _ in matched]
            acting = (
                "TRUE"
                if any(c is None for c in conditions)
                else " OR ".join(f"COALESCE(({c}), FALSE)" for c in conditions)
            )
            dup = _one_statement(
                con,
                f"SELECT count(*) FROM (SELECT {t}.{_quote(_FILE)}, {t}.{_quote(_INDEX)} "
                f"FROM __target AS {t} JOIN __source AS {s} ON {on} WHERE {acting} "
                f"GROUP BY 1, 2 HAVING count(*) > 1)",
            ).fetchone()[0]
            if dup:
                raise InvalidArgumentError(
                    f"the MERGE matched {dup} target row(s) with more than one source row; "
                    "which source row updates it would be arbitrary. Deduplicate the source "
                    "on the join key"
                )

        deletions: list[Any] = []
        deleted: list[Any] = []
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
                    deleted.append(part)
                    continue
                exprs = self._assignments(verb, arg, columns, kind, schema)
                select = ", ".join(f"{e} AS {_quote(c)}" for c, e in exprs.items())
                part = _fetch(con, f"SELECT {positions}{carried}, {select} {sql_from}")
                deletions.append(part.select([_POS_FILE, _POS_INDEX]))
                outputs.append(part.drop_columns([_POS_FILE, _POS_INDEX]))
                counts["updated"] += part.num_rows

        not_matched = clauses["not_matched"]
        for index, (_, _, verb, arg) in enumerate(not_matched):
            condition = applies(not_matched, index)
            exprs = self._insert_values(verb, arg, columns, schema)
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

        positions_schema = pa.schema([(_POS_FILE, pa.string()), (_POS_INDEX, pa.int64())])
        deletion_schema = pa.schema([("path", pa.string()), ("row_index", pa.int64())])
        deletion_table = (
            pa.concat_tables([d.cast(positions_schema) for d in deletions]).rename_columns(
                deletion_schema.names
            )
            if deletions
            else deletion_schema.empty_table()
        )
        if deleted:
            # Several source rows may match one target row an unconditional
            # DELETE takes; the row is deleted once.
            rows = pa.concat_tables([d.cast(positions_schema) for d in deleted])
            counts["deleted"] = rows.group_by([_POS_FILE, _POS_INDEX]).aggregate([]).num_rows
        data = None
        if outputs:
            target_schema = pa.schema(
                list(schema) + ([pa.field(_ROW_ID, pa.int64())] if row_ids else [])
            )
            data = pa.concat_tables([_cast_to(o, target_schema, self._variants) for o in outputs])
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

    def _value(self, value: Any, schema: Any, column: str) -> str:
        if value is _DEFAULT:
            # A column named `default` wins over the keyword, as on Databricks.
            for alias, names in (
                (self._source_alias, self._source.column_names),
                (self._target_alias, schema.names),
            ):
                named = [n for n in names if n.lower() == "default"]
                if named:
                    return f"{_quote(alias)}.{_quote(named[0])}"
            return default_value_sql(schema.field(column), "SET ... = DEFAULT")
        return str(value)

    def _assignments(
        self, verb: str, arg: Any, columns: list[str], kind: str, schema: Any
    ) -> dict[str, str]:
        """The new value of every target column, for an UPDATE clause."""
        t, s = _quote(self._target_alias), _quote(self._source_alias)
        exprs = {c: f"{t}.{_quote(c)}" for c in columns}
        if verb == "UPDATE":
            for key, value in arg.items():
                column = self._target_column(key, columns)
                exprs[column] = f"({self._value(value, schema, column)})"
            return exprs
        # UPDATE_ALL: every column the source has, except `except_cols`.
        if kind != "matched":
            raise InvalidArgumentError("update_all is only valid for matched rows")
        for column, name in self._star_columns(arg, columns, "UPDATE SET *").items():
            exprs[column] = f"{s}.{_quote(name)}"
        return exprs

    def _insert_values(
        self, verb: str, arg: Any, columns: list[str], schema: Any
    ) -> dict[str, str]:
        s = _quote(self._source_alias)
        exprs = {c: "NULL" for c in columns}
        if verb == "INSERT":
            named = {self._target_column(key, columns): value for key, value in arg.items()}
            for column in columns:
                # A column the INSERT leaves out takes its DEFAULT, as on
                # Databricks; writing NULL lost 'dflt'.
                value = named.get(column, _DEFAULT)
                exprs[column] = f"({self._value(value, schema, column)})"
            return exprs
        for column, name in self._star_columns(arg, columns, "INSERT *").items():
            exprs[column] = f"{s}.{_quote(name)}"
        return exprs

    def _star_columns(self, except_cols: Any, columns: list[str], what: str) -> dict[str, str]:
        """Target column -> source column for `UPDATE SET *` / `INSERT *`.

        Databricks refuses either when the source lacks a target column that
        `except_cols` does not name ([DELTA_MERGE_UNRESOLVED_EXPRESSION]);
        filling it with NULL (or keeping the target's value) wrote rows the
        warehouse would not have.
        """
        excluded = {str(c).lower() for c in except_cols}
        sources = {n.lower(): n for n in self._source.column_names if n != _SRC_ROW}
        missing = [c for c in columns if c.lower() not in sources and c.lower() not in excluded]
        if missing:
            raise InvalidArgumentError(
                f"{what} needs every target column in the source, which lacks "
                f"{', '.join(missing)}; add them to the source, or spell the clause out"
            )
        return {c: sources[c.lower()] for c in columns if c.lower() not in excluded}


def _fetch(con: Any, sql: str) -> Any:
    """A query's result as a `pyarrow.Table`, on every supported DuckDB.

    `.arrow()` returned a Table before DuckDB 1.4 and a RecordBatchReader
    since; `pa.table` accepts both.
    """
    import pyarrow as pa

    return pa.table(_one_statement(con, sql).arrow())


def _one_statement(con: Any, sql: str) -> Any:
    """`con.execute(sql)`, refused unless `sql` is exactly one statement.

    The clauses are validated as single expressions before they get here;
    this is the last line should one still end the SELECT around it.
    """
    if len(con.extract_statements(sql)) != 1:
        raise InvalidArgumentError(
            "the MERGE clause text does not stay inside one expression; refused"
        )
    return con.execute(sql)


def _row_tracking_enabled(table: Any) -> bool:
    return str(table.properties.get("delta.enableRowTracking", "false")).lower() == "true"


def _cast_to(data: Any, schema: Any, variants: frozenset[str] = frozenset()) -> Any:
    """`data` with `schema`'s columns and types, by name."""
    import pyarrow as pa

    columns = []
    for field in schema:
        if field.name in data.column_names:
            column = data.column(field.name)
            if field.name.lower() in variants and (
                pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
            ):
                # Spark stores a STRING assigned to a VARIANT as a variant
                # string (`SET v = 'x'`); an object comes from parse_json.
                from .._variant import string_variant, variant_column

                texts = [None if t is None else string_variant(t) for t in column.to_pylist()]
                column = variant_column(pa, pa.array(texts, pa.string()))
            columns.append(store_cast(column, field.type, field.name))
        else:
            columns.append(pa.nulls(data.num_rows, field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def store_cast(column: Any, target: Any, name: str) -> Any:
    """`column` as `target`, the way Spark stores a value into a column.

    A bare NULL comes out of DuckDB typed INTEGER, which Arrow will not cast
    to a timestamp, struct or binary. A fraction stored into an integer
    column truncates and one stored into a narrower decimal rounds half-up,
    where Arrow's safe cast refuses both. Anything else that does not fit is
    the caller's mistake, reported as one.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if column.type == target:
        return column
    if column.null_count == len(column):
        return pa.nulls(len(column), target)
    try:
        if pa.types.is_nested(target):
            column = _placeholder_children(column, target)
        if pa.types.is_integer(target) and pa.types.is_floating(column.type):
            column = pc.trunc(column)
        elif pa.types.is_integer(target) and pa.types.is_decimal(column.type):
            column = pc.round(column, ndigits=0, round_mode="towards_zero")
        elif pa.types.is_decimal(target) and (
            pa.types.is_floating(column.type)
            or (pa.types.is_decimal(column.type) and column.type.scale > target.scale)
        ):
            column = pc.round(column, ndigits=target.scale, round_mode="half_up")
        return column.cast(target)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
        raise InvalidArgumentError(
            f"the value written to {name!r} is {column.type}, which cannot be stored in the "
            f"column's type {target}: {exc}"
        ) from exc


def _placeholder_children(column: Any, target: Any) -> Any:
    """`column` with a placeholder under every null struct whose `target` field is required.

    A VARIANT is a struct of two required binaries, and a NULL one holds
    b"" in both (`variant_column`). DuckDB hands a NULL struct back with
    null children, so a MERGE whose source ARRAY<VARIANT> held a NULL element
    failed the cast ("field 'metadata' ... has nulls") that the append of the
    same row passed. Only children of null parents are filled: a required
    field that is null in a present struct still fails, as it should.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(column, pa.ChunkedArray):
        chunks = [_placeholder_children(c, target) for c in column.chunks]
        return pa.chunked_array(chunks, type=chunks[0].type) if chunks else column
    kind = column.type
    if pa.types.is_struct(target) and pa.types.is_struct(kind):
        parent_null = column.is_null()
        names = [kind.field(i).name for i in range(kind.num_fields)]
        wanted = {target.field(i).name: target.field(i) for i in range(target.num_fields)}
        children = []
        for i, name in enumerate(names):
            child = column.field(i)
            want = wanted.get(name)
            if want is not None:
                child = _placeholder_children(child, want.type)
                zero = _zero(want.type) if not want.nullable else None
                if zero is not None and child.null_count:
                    child = pc.if_else(
                        pc.and_(parent_null, child.is_null()), pa.scalar(zero, child.type), child
                    )
            children.append(child)
        return pa.StructArray.from_arrays(
            children, fields=list(kind), mask=parent_null if column.null_count else None
        )
    if (
        (pa.types.is_list(target) or pa.types.is_large_list(target))
        and (pa.types.is_list(kind) or pa.types.is_large_list(kind))
        and pa.types.is_nested(target.value_type)
    ):
        values = _placeholder_children(column.values, target.value_type)
        field = kind.value_field.with_type(values.type)
        if pa.types.is_large_list(kind):
            cls, rebuilt = pa.LargeListArray, pa.large_list(field)
        else:
            cls, rebuilt = pa.ListArray, pa.list_(field)
        return cls.from_arrays(
            column.offsets,
            values,
            type=rebuilt,
            mask=column.is_null() if column.null_count else None,
        )
    if pa.types.is_map(target) and pa.types.is_map(kind) and pa.types.is_nested(target.item_type):
        items = _placeholder_children(column.items, target.item_type)
        return pa.MapArray.from_arrays(
            column.offsets,
            column.keys,
            items,
            type=pa.map_(kind.key_field, kind.item_field.with_type(items.type)),
            mask=column.is_null() if column.null_count else None,
        )
    return column


def _zero(datatype: Any) -> Any:
    """A placeholder value of `datatype`, or None where there is no plain one."""
    import pyarrow as pa

    t = pa.types
    if t.is_binary(datatype) or t.is_large_binary(datatype):
        return b""
    if t.is_string(datatype) or t.is_large_string(datatype):
        return ""
    if t.is_boolean(datatype):
        return False
    if t.is_integer(datatype) or t.is_floating(datatype):
        return 0
    return None


def _same_comparison_type(source: Any, target: Any) -> bool:
    """Whether a source key and a target column compare as one type, exactly.

    Only then are bounds cast to the target's type the values the ON clause
    compares: integers of any width (a lossless cast, checked), strings of
    any encoding, or one identical type.
    """
    import pyarrow as pa

    t = pa.types
    if source.equals(target):
        return True
    if t.is_integer(source) and t.is_integer(target):
        return True

    def stringy(x: Any) -> bool:
        view = getattr(t, "is_string_view", None)
        return bool(t.is_string(x) or t.is_large_string(x) or (view is not None and view(x)))

    return stringy(source) and stringy(target)


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


_KIND_NAMES = {
    "matched": "WHEN MATCHED",
    "not_matched": "WHEN NOT MATCHED",
    "not_matched_by_source": "WHEN NOT MATCHED BY SOURCE",
}


def _expression(value: Any) -> str:
    """A SET/INSERT value as SQL text; None is NULL, as on the warehouse."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _spark_sql(text: str) -> str:
    """Spark SQL clause text rewritten for DuckDB (see `dialect.to_duckdb`)."""
    from .dialect import to_duckdb

    return to_duckdb(text)


class _Default:
    """The DEFAULT keyword as a SET or INSERT value, resolved per column at execute."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "DEFAULT"


_DEFAULT = _Default()


def _is_default_keyword(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == "DEFAULT"


def default_value_sql(field: Any, what: str) -> str:
    """The DuckDB SQL for `field`'s DEFAULT, or NULL when it has none.

    Only a literal DEFAULT is evaluated here, as `append` does; one that is
    an expression (`current_timestamp()`) is Databricks' to evaluate, so the
    MERGE is refused before anything is written and another engine serves it.
    """
    from ..predicate import Literal, PredicateError, parse_value

    raw = (field.metadata or {}).get(b"CURRENT_DEFAULT")
    if raw is None:
        return "NULL"
    text = raw.decode()
    try:
        value = parse_value(text)
    except PredicateError:
        value = None
    if not isinstance(value, Literal) or value.type not in (
        "string",
        "long",
        "decimal",
        "boolean",
        "null",
        "date",
    ):
        raise EngineLimitError(
            f"merge on the kernel ({what})",
            f"column {field.name!r} defaults to the expression {text!r}, which only "
            "Databricks evaluates",
            "ds.connect(..., allow_sql_fallback=True) runs the MERGE on a SQL warehouse; "
            "or set the column explicitly",
        )
    return _spark_sql(text)
