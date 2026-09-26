"""A pyarrow Dataset that reads a table lazily, with projection and filter pushdown.

`to_duckdb()`, `to_polars(lazy=True)`, `to_pyarrow_dataset()` and
`Connection.sql()` used to read the whole table into memory before the
caller's filter ran: `t.to_duckdb().filter("id < 10")` read ten million rows
to return ten. DuckDB and Polars both push projections and filters into a
pyarrow Dataset by calling its ``scanner()`` / ``to_batches()`` with a
pyarrow Expression; this Dataset answers those calls with a scan through the
table's own engine (so deletion vectors, column mapping and catalog-managed
tables read correctly), passing the columns and -- where it translates -- the
filter down for file skipping.

The filter is also applied exactly, by pyarrow, to what the scan returns: the
translation to SQL only has to be no stronger than the expression, never
equal to it, and anything it cannot read is simply not pushed.
"""

from __future__ import annotations

import re
from typing import Any

import pyarrow as pa
import pyarrow.dataset as pds

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?![\d:. ])")
_COMPARISONS = {"==": "=", "!=": "<>", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


class _Unsupported(Exception):
    """An expression form the translation does not read."""


class _ExprParser:
    """Reads the textual form of a pyarrow Expression, for the forms engines push.

    Only what DuckDB and Polars send for ordinary WHERE clauses: comparisons
    of a column with a number, string, boolean or date, ``and``/``or``,
    ``invert``, ``is_null``/``is_valid`` and ``is_in``. Everything else raises
    `_Unsupported`.
    """

    def __init__(self, text: str, columns: set[str]) -> None:
        self.text = text
        self.i = 0
        self.columns = columns

    def parse(self) -> str:
        out = self.expr()
        self.skip()
        if self.i != len(self.text):
            raise _Unsupported(self.text[self.i :])
        return out

    def skip(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def take(self, token: str) -> bool:
        self.skip()
        if self.text.startswith(token, self.i):
            self.i += len(token)
            return True
        return False

    def expect(self, token: str) -> None:
        if not self.take(token):
            raise _Unsupported(token)

    def column(self) -> str:
        self.skip()
        match = _IDENT.match(self.text, self.i)
        if match is None or match.group() not in self.columns:
            raise _Unsupported("not a plain column")
        self.i = match.end()
        return f"`{match.group()}`"

    def literal(self) -> str:
        self.skip()
        text = self.text
        if text.startswith('"', self.i):
            j, chars = self.i + 1, []
            while j < len(text) and text[j] != '"':
                if text[j] == "\\":
                    if j + 1 >= len(text) or text[j + 1] not in '"\\':
                        raise _Unsupported("escape")
                    j += 1
                chars.append(text[j])
                j += 1
            if j >= len(text):
                raise _Unsupported("unterminated string")
            self.i = j + 1
            value = "".join(chars)
            # Spark SQL string syntax, which this library's parser reads: a
            # backslash escapes, and '' is two adjacent literals, not a quote.
            return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        for word, sql in (("true", "TRUE"), ("false", "FALSE")):
            if text.startswith(word, self.i) and not text[self.i + len(word) :][:1].isalnum():
                self.i += len(word)
                return sql
        date = _DATE.match(text, self.i)
        if date is not None:
            self.i = date.end()
            return f"DATE '{date.group()}'"
        number = _NUMBER.match(text, self.i)
        if number is not None and not text[number.end() :][:1].isalnum():
            self.i = number.end()
            body = number.group()
            # A double: SQL reads an unsuffixed 1.5 as DECIMAL, the same value.
            return body + ("D" if "e" in body.lower() else "")
        raise _Unsupported("literal")

    def operand(self) -> str:
        self.skip()
        match = _IDENT.match(self.text, self.i)
        if match is not None and match.group() in self.columns:
            return self.column()
        return self.literal()

    def expr(self) -> str:
        self.skip()
        if self.take("invert("):
            inner = self.expr()
            self.expect(")")
            return f"(NOT {inner})"
        if self.take("is_valid("):
            col = self.column()
            self.expect(")")
            return f"({col} IS NOT NULL)"
        if self.take("is_null("):
            col = self.column()
            # `{nan_is_null=false}` is SQL's IS NULL; true also matches NaN.
            self.expect(", {nan_is_null=false})")
            return f"({col} IS NULL)"
        if self.take("is_in("):
            col = self.column()
            self.expect(", {value_set=")
            match = _IDENT.match(self.text, self.i)
            if match is None or match.group() not in ("int64", "int32", "string", "double"):
                raise _Unsupported("value set type")
            self.i = match.end()
            self.expect(":[")
            items = []
            while not self.take("]"):
                items.append(self.literal())
                self.take(",")
            # MATCH only differs from SQL IN when the set holds a null,
            # which the literal reader refuses.
            self.expect(", null_matching_behavior=MATCH})")
            if not items:
                raise _Unsupported("empty value set")
            return f"({col} IN ({', '.join(items)}))"
        if self.take("("):
            left = self.expr() if self.text.startswith(("(", "invert(", "is_"), self.i) else None
            if left is not None:
                for word in (" and ", " or "):
                    if self.take(word.strip()):
                        right = self.expr()
                        self.expect(")")
                        return f"({left} {word.strip().upper()} {right})"
                raise _Unsupported("boolean operator")
            lhs = self.operand()
            self.skip()
            op = next((o for o in ("==", "!=", "<=", ">=", "<", ">") if self.take(o)), None)
            if op is None:
                raise _Unsupported("operator")
            rhs = self.operand()
            self.expect(")")
            if not (lhs.startswith("`") or rhs.startswith("`")):
                raise _Unsupported("no column")
            return f"({lhs} {_COMPARISONS[op]} {rhs})"
        # A bare boolean column.
        return self.column()


def expression_to_sql(expression: Any, columns: set[str]) -> str | None:
    """A SQL predicate no stronger than the pyarrow `expression`, or None.

    Top-level conjuncts that do not translate are dropped (a weaker
    predicate reads more, never less); anything else untranslatable drops the
    whole predicate.
    """
    if expression is None:
        return None
    text = str(expression)
    conjuncts = _split_and(text)
    parts = []
    for part in conjuncts:
        try:
            parts.append(_ExprParser(part, columns).parse())
        except (_Unsupported, IndexError):
            continue
    return " AND ".join(parts) if parts else None


def _split_and(text: str) -> list[str]:
    """`((a) and (b))` -> [`(a)`, `(b)`], recursively; anything else as it is."""
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return [text]
    depth, in_string, i = 0, False, 1
    inner = text[1:-1]
    while i <= len(inner):
        ch = inner[i - 1]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and inner.startswith(" and ", i - 1):
            return _split_and(inner[: i - 1]) + _split_and(inner[i + 4 :])
        i += 1
    return [text]


def referenced_columns(expression: Any, names: list[str]) -> set[str]:
    """Every column name that appears in the expression's text (a superset)."""
    text = str(expression)
    return {
        n for n in names if re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", text) is not None
    }


class TableDataset(pds.InMemoryDataset):  # type: ignore[misc]
    """A pyarrow Dataset over a `Table` that scans it only when read.

    Every read is a fresh scan through the table's engine at the handle's
    version, so the same relation or LazyFrame can be executed repeatedly.
    Methods pyarrow would answer from the (empty) in-memory placeholder
    materialize the table first instead.
    """

    def __init__(
        self,
        table: Any,
        schema: Any,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        scan_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(schema.empty_table())
        self._table = table
        self._columns = columns
        self._predicate = predicate
        self._scan_options = dict(scan_options or {})
        self._schema_ = schema
        self._base_filter: Any = None

    @property
    def schema(self) -> Any:
        return self._schema_

    def _reader(self, columns: list[str] | None, filter: Any) -> Any:
        names = list(self._schema_.names)
        pushed = expression_to_sql(filter, set(names)) if filter is not None else None
        predicate = " AND ".join(f"({p})" for p in (self._predicate, pushed) if p) or None
        read = None
        if columns is not None:
            wanted = set(columns) | (
                referenced_columns(filter, names) if filter is not None else set()
            )
            read = [n for n in names if n in wanted] or names[:1]
        stream = self._table.scan(columns=read, predicate=predicate, **self._scan_options)
        return pa.RecordBatchReader.from_stream(stream)

    def scanner(self, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        if self._base_filter is not None:
            filter = self._base_filter if filter is None else (self._base_filter & filter)
        project: Any
        if isinstance(columns, dict):
            # Computed projections: read every column and let pyarrow project.
            reader, project = self._reader(None, filter), columns
        else:
            project = list(columns) if columns is not None else None
            reader = self._reader(project, filter)
        return pds.Scanner.from_batches(
            reader,
            columns=project,
            filter=filter,
            **{k: v for k, v in kwargs.items() if k in ("batch_size", "use_threads")},
        )

    def to_batches(self, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        return self.scanner(columns=columns, filter=filter, **kwargs).to_batches()

    def to_table(self, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        return self.scanner(columns=columns, filter=filter, **kwargs).to_table()

    def head(self, num_rows: int, columns: Any = None, filter: Any = None, **kwargs: Any) -> Any:
        return self.scanner(columns=columns, filter=filter, **kwargs).head(num_rows)

    def count_rows(self, filter: Any = None, **kwargs: Any) -> int:
        if filter is None and self._base_filter is None and not self._scan_options:
            return int(self._table.count(predicate=self._predicate))
        return int(self.scanner(filter=filter, **kwargs).count_rows())

    def take(self, indices: Any, **kwargs: Any) -> Any:
        return self.to_table(**kwargs).take(indices)

    def filter(self, expression: Any) -> TableDataset:
        out = TableDataset(
            self._table,
            self._schema_,
            columns=self._columns,
            predicate=self._predicate,
            scan_options=self._scan_options,
        )
        base = self._base_filter
        out._base_filter = expression if base is None else (base & expression)
        return out

    def _materialized(self) -> Any:
        return pds.dataset(self.to_table())

    def sort_by(self, sorting: Any, **kwargs: Any) -> Any:
        return self._materialized().sort_by(sorting, **kwargs)

    def join(self, *args: Any, **kwargs: Any) -> Any:
        return self._materialized().join(*args, **kwargs)

    def join_asof(self, *args: Any, **kwargs: Any) -> Any:
        return self._materialized().join_asof(*args, **kwargs)

    def replace_schema(self, schema: Any) -> Any:
        return self._materialized().replace_schema(schema)

    def get_fragments(self, filter: Any = None) -> Any:
        return self._materialized().get_fragments(filter=filter)

    def __reduce__(self) -> Any:
        # The placeholder pickles as an empty table; ship the rows instead.
        return (pds.dataset, (self.to_table(),))
