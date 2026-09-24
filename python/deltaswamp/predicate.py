"""SQL predicates, parsed once and used two ways.

delta-rs takes a predicate as a SQL string and evaluates it with DataFusion. The
kernel takes a structured `Predicate` and uses it only to *skip files*: it never
promises to drop a non-matching row. So a predicate on the kernel path needs
both halves -- a structure the kernel can skip with, and an exact row filter
applied afterwards -- and both have to mean the same thing as the SQL string.

This module parses the boolean subset of Spark SQL that people actually write in
a `WHERE` clause into a small AST, then renders it:

* `to_kernel_json()` -- the skipping predicate handed to the native extension.
  Anything the kernel cannot express is dropped *conservatively*: an
  unexpressible conjunct of an AND is removed (skipping less is always safe),
  while an OR or NOT with an unexpressible child is abandoned entirely (skipping
  on part of a disjunction would skip files that match).
* `to_arrow(schema)` -- an exact pyarrow compute expression with SQL's
  three-valued logic: a row is kept only when the predicate is TRUE, so NULL
  comparisons drop the row just as a `WHERE` clause does.

What is deliberately out of scope: arithmetic, function calls other than the
comparisons below, and subqueries. Those raise `PredicateError` naming the
token, rather than being half-understood.
"""

from __future__ import annotations

import datetime as _dt
import decimal
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .errors import DeltaSwampError

__all__ = [
    "Column",
    "Literal",
    "Node",
    "PredicateError",
    "columns_of",
    "filter_stream",
    "filter_table",
    "parse",
    "parse_value",
]


class PredicateError(DeltaSwampError):
    """A predicate string could not be parsed, or uses unsupported SQL."""


# --------------------------------------------------------------------------- AST


@dataclass(frozen=True, slots=True)
class Column:
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Literal:
    value: Any
    #: Explicit SQL type from a typed literal (DATE '...'), else inferred.
    type: str


@dataclass(frozen=True, slots=True)
class Node:
    """One predicate node. `op` is and/or/not/eq/ne/lt/le/gt/ge/eq_null_safe/
    is_null/is_not_null/in/between/like/true/false/column."""

    op: str
    args: tuple[Any, ...] = ()
    negated: bool = False


# ------------------------------------------------------------------------- lexer

_TOKEN = re.compile(
    r"""
    \s*(?:
      (?P<number>-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[LlDdFf]?)
     |(?P<string>'(?:[^']|'')*')
     |(?P<dstring>"(?:[^"]|"")*")
     |(?P<quoted>`(?:[^`]|``)+`)
     |(?P<op><=>|<=|>=|<>|!=|==|=|<|>|\(|\)|,|\.)
     |(?P<word>[A-Za-z_][A-Za-z_0-9]*)
    )""",
    re.VERBOSE,
)

_KEYWORDS = frozenset({"AND", "OR", "NOT", "IS", "NULL", "IN", "BETWEEN", "LIKE", "TRUE", "FALSE"})
_TYPED_LITERALS = frozenset({"DATE", "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ"})


@dataclass(frozen=True, slots=True)
class _Tok:
    kind: str
    text: str


def _tokenize(text: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    pos = 0
    stripped = text.rstrip()
    while pos < len(stripped):
        match = _TOKEN.match(stripped, pos)
        if match is None or match.end() == pos:
            raise PredicateError(
                f"cannot parse predicate {text!r} at position {pos}: {stripped[pos : pos + 20]!r}"
            )
        kind = match.lastgroup or ""
        tokens.append(_Tok(kind, match.group(kind)))
        pos = match.end()
    return tokens


# ------------------------------------------------------------------------ parser


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self, offset: int = 0) -> _Tok | None:
        j = self.i + offset
        return self.tokens[j] if j < len(self.tokens) else None

    def keyword(self, word: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok.kind == "word" and tok.text.upper() == word

    def symbol(self, sym: str) -> bool:
        tok = self.peek()
        return tok is not None and tok.kind == "op" and tok.text == sym

    def take(self) -> _Tok:
        tok = self.peek()
        if tok is None:
            raise PredicateError(f"predicate {self.text!r} ends unexpectedly")
        self.i += 1
        return tok

    def expect_symbol(self, sym: str) -> None:
        if not self.symbol(sym):
            got = self.peek()
            raise PredicateError(
                f"expected {sym!r} in predicate {self.text!r}, got "
                f"{got.text if got else 'end of input'!r}"
            )
        self.i += 1

    def expect_keyword(self, word: str) -> None:
        if not self.keyword(word):
            got = self.peek()
            raise PredicateError(
                f"expected {word} in predicate {self.text!r}, got "
                f"{got.text if got else 'end of input'!r}"
            )
        self.i += 1

    # -- grammar --

    def parse(self) -> Node:
        node = self.or_expr()
        if self.peek() is not None:
            raise PredicateError(
                f"unexpected {self.peek().text!r} in predicate {self.text!r}"  # type: ignore[union-attr]
            )
        return node

    def or_expr(self) -> Node:
        parts = [self.and_expr()]
        while self.keyword("OR"):
            self.i += 1
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else Node("or", tuple(parts))

    def and_expr(self) -> Node:
        parts = [self.not_expr()]
        while self.keyword("AND"):
            self.i += 1
            parts.append(self.not_expr())
        return parts[0] if len(parts) == 1 else Node("and", tuple(parts))

    def not_expr(self) -> Node:
        if self.keyword("NOT"):
            self.i += 1
            return Node("not", (self.not_expr(),))
        return self.predicate()

    def predicate(self) -> Node:
        # A parenthesised boolean expression, unless it is a parenthesised value.
        if self.symbol("("):
            self.i += 1
            inner = self.or_expr()
            self.expect_symbol(")")
            return inner

        left = self.value()

        if self.keyword("IS"):
            self.i += 1
            negated = False
            if self.keyword("NOT"):
                self.i += 1
                negated = True
            self.expect_keyword("NULL")
            return Node("is_not_null" if negated else "is_null", (left,))

        negated = False
        if self.keyword("NOT") and (
            self.keyword("IN", 1) or self.keyword("BETWEEN", 1) or self.keyword("LIKE", 1)
        ):
            self.i += 1
            negated = True

        if self.keyword("IN"):
            self.i += 1
            self.expect_symbol("(")
            items = [self.value()]
            while self.symbol(","):
                self.i += 1
                items.append(self.value())
            self.expect_symbol(")")
            return Node("in", (left, *items), negated)

        if self.keyword("BETWEEN"):
            self.i += 1
            low = self.value()
            self.expect_keyword("AND")
            high = self.value()
            return Node("between", (left, low, high), negated)

        if self.keyword("LIKE"):
            self.i += 1
            pattern = self.value()
            if not (isinstance(pattern, Literal) and isinstance(pattern.value, str)):
                raise PredicateError(f"LIKE needs a string pattern in {self.text!r}")
            return Node("like", (left, pattern), negated)

        tok = self.peek()
        if tok is not None and tok.kind == "op" and tok.text in _COMPARISONS:
            self.i += 1
            right = self.value()
            return Node(_COMPARISONS[tok.text], (left, right))

        # A bare boolean column or literal.
        if isinstance(left, Column):
            return Node("column", (left,))
        if isinstance(left, Literal) and isinstance(left.value, bool):
            return Node("true" if left.value else "false")
        raise PredicateError(f"{self.text!r} is not a boolean predicate")

    def value(self) -> Column | Literal:
        tok = self.take()
        if tok.kind == "number":
            return _number(tok.text)
        if tok.kind == "string":
            return Literal(tok.text[1:-1].replace("''", "'"), "string")
        if tok.kind == "dstring":
            # Spark treats double quotes as a string literal by default.
            return Literal(tok.text[1:-1].replace('""', '"'), "string")
        if tok.kind == "op" and tok.text == "(":
            inner = self.value()
            self.expect_symbol(")")
            return inner
        if tok.kind in ("word", "quoted"):
            upper = tok.text.upper() if tok.kind == "word" else ""
            if upper in ("TRUE", "FALSE"):
                return Literal(upper == "TRUE", "boolean")
            if upper == "NULL":
                return Literal(None, "null")
            following = self.peek()
            if upper in _TYPED_LITERALS and following is not None and following.kind == "string":
                raw = self.take().text[1:-1]
                return _typed(upper, raw, self.text)
            if tok.kind == "word" and upper in _KEYWORDS:
                raise PredicateError(f"unexpected keyword {tok.text} in predicate {self.text!r}")
            if self.symbol("("):
                raise PredicateError(
                    f"function calls such as {tok.text}(...) are not supported in predicates "
                    "evaluated outside a SQL engine"
                )
            path = [_ident(tok)]
            while self.symbol("."):
                self.i += 1
                nxt = self.take()
                if nxt.kind not in ("word", "quoted"):
                    raise PredicateError(f"bad column reference in {self.text!r}")
                path.append(_ident(nxt))
            return Column(tuple(path))
        raise PredicateError(f"unexpected {tok.text!r} in predicate {self.text!r}")


_COMPARISONS = {
    "=": "eq",
    "==": "eq",
    "!=": "ne",
    "<>": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
    "<=>": "eq_null_safe",
}


def _ident(tok: _Tok) -> str:
    if tok.kind == "quoted":
        return tok.text[1:-1].replace("``", "`")
    return tok.text


def _number(text: str) -> Literal:
    suffix = text[-1].upper() if text[-1].isalpha() else ""
    body = text[:-1] if suffix else text
    if suffix == "L":
        return Literal(int(body), "long")
    if suffix in ("D", "F") or any(c in body for c in ".eE"):
        if suffix == "" and "." in body and not any(c in body for c in "eE"):
            # SQL: an unsuffixed 1.5 is a DECIMAL. Keep exactness for the
            # filter; the kernel coerces to the column type anyway.
            return Literal(decimal.Decimal(body), "decimal")
        return Literal(float(body), "double")
    return Literal(int(body), "long")


def _typed(kind: str, raw: str, text: str) -> Literal:
    try:
        if kind == "DATE":
            return Literal(_dt.date.fromisoformat(raw), "date")
        parsed = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredicateError(f"bad {kind} literal {raw!r} in {text!r}") from exc
    if kind == "TIMESTAMP_NTZ":
        return Literal(parsed.replace(tzinfo=None), "timestamp_ntz")
    if parsed.tzinfo is None:
        # Spark reads an unzoned TIMESTAMP literal in the session zone; with no
        # session here, UTC is the only defensible choice.
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return Literal(parsed, "timestamp")


def parse(text: str) -> Node:
    """Parse a SQL boolean expression. Raises `PredicateError` on anything else."""
    if not isinstance(text, str) or not text.strip():
        raise PredicateError("a predicate must be a non-empty string")
    return _Parser(text).parse()


def parse_value(text: str) -> Column | Literal:
    """Parse a single SQL value: a literal or a column reference."""
    try:
        parser = _Parser(text)
        value = parser.value()
    except PredicateError as exc:
        raise PredicateError(
            f"{text!r} is not a plain value; only a literal or a column is supported here"
        ) from exc
    if parser.peek() is not None:
        raise PredicateError(
            f"{text!r} is an expression; only a literal or a column is supported here"
        )
    return value


def columns_of(node: Node | Column | Literal) -> set[tuple[str, ...]]:
    """Every column path the predicate references."""
    if isinstance(node, Column):
        return {node.path}
    if isinstance(node, Literal):
        return set()
    out: set[tuple[str, ...]] = set()
    for arg in node.args:
        out |= columns_of(arg)
    return out


# ----------------------------------------------------------- kernel (skipping)


def _literal_json(lit: Literal) -> dict[str, Any]:
    value = lit.value
    if isinstance(value, _dt.datetime):
        return {"literal": value.isoformat(), "type": lit.type}
    if isinstance(value, _dt.date):
        return {"literal": value.isoformat(), "type": "date"}
    if isinstance(value, decimal.Decimal):
        return {"literal": str(value), "type": "decimal"}
    return {"literal": value, "type": lit.type}


def _expr_json(value: Column | Literal) -> dict[str, Any]:
    if isinstance(value, Column):
        return {"column": list(value.path)}
    return _literal_json(value)


def _skip(node: Node, exact: bool = False) -> dict[str, Any] | None:
    """The kernel form of `node`, or None where skipping must not be attempted.

    Dropping a conjunct weakens a predicate, which is safe for skipping -- a
    weaker predicate skips fewer files. Under a NOT the polarity flips and a
    weaker child makes a *stronger* predicate, so beneath a NOT every part must
    be rendered exactly or not at all. `exact` carries that requirement down.
    """
    op = node.op
    if op == "and":
        rendered = [_skip(a, exact) for a in node.args]
        if exact and any(p is None for p in rendered):
            return None
        parts = [p for p in rendered if p is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else {"op": "and", "args": parts}
    if op == "or":
        rendered = [_skip(a, exact) for a in node.args]
        disjuncts = [p for p in rendered if p is not None]
        if len(disjuncts) != len(rendered):
            return None
        return {"op": "or", "args": disjuncts}
    if op == "not":
        inner = _skip(node.args[0], exact=True)
        return None if inner is None else {"op": "not", "args": [inner]}
    if op in ("eq", "ne", "lt", "le", "gt", "ge"):
        left, right = node.args
        if any(isinstance(x, Literal) and x.value is None for x in (left, right)):
            # `x = NULL` is never true; not a skipping predicate worth sending.
            return None
        return {"op": op, "args": [_expr_json(left), _expr_json(right)]}
    if op in ("is_null", "is_not_null"):
        (target,) = node.args
        if not isinstance(target, Column):
            return None
        return {"op": op, "args": [_expr_json(target)]}
    if op == "in":
        target, *items = node.args
        if not isinstance(target, Column) or any(
            not isinstance(i, Literal) or i.value is None for i in items
        ):
            return None
        disjunction = {
            "op": "or",
            "args": [{"op": "eq", "args": [_expr_json(target), _expr_json(i)]} for i in items],
        }
        return {"op": "not", "args": [disjunction]} if node.negated else disjunction
    if op == "between":
        target, low, high = node.args
        rng = {
            "op": "and",
            "args": [
                {"op": "ge", "args": [_expr_json(target), _expr_json(low)]},
                {"op": "le", "args": [_expr_json(target), _expr_json(high)]},
            ],
        }
        return {"op": "not", "args": [rng]} if node.negated else rng
    # like, eq_null_safe, bare columns, constants: no skipping.
    return None


def to_kernel_json(node: Node) -> str | None:
    """The skipping predicate as the native extension's JSON, or None."""
    rendered = _skip(node)
    return None if rendered is None else json.dumps(rendered)


# ------------------------------------------------------------- arrow (exact)


def _field(pc: Any, path: tuple[str, ...]) -> Any:
    return pc.field(path[0]) if len(path) == 1 else pc.field(*path)


def _schema_type(schema: Any, path: tuple[str, ...]) -> Any:
    try:
        current = schema.field(path[0]).type
        for name in path[1:]:
            current = current.field(name).type
        return current
    except (KeyError, AttributeError):
        return None


def _scalar(pa: Any, lit: Literal, target: Any) -> Any:
    if lit.value is None:
        return pa.scalar(None, type=target) if target is not None else pa.scalar(None)
    scalar = pa.scalar(lit.value)
    if target is None or scalar.type == target:
        return scalar
    try:
        return scalar.cast(target)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        # Let Arrow's own comparison kernels decide; they widen numerics.
        return scalar


class _ArrowBuilder:
    def __init__(self, schema: Any) -> None:
        import pyarrow as pa
        import pyarrow.compute as pc

        self.pa = pa
        self.pc = pc
        self.schema = schema

    def value(self, value: Column | Literal, other: Column | Literal | None = None) -> Any:
        if isinstance(value, Column):
            if _schema_type(self.schema, value.path) is None:
                raise PredicateError(
                    f"the predicate references unknown column {'.'.join(value.path)!r}"
                )
            return _field(self.pc, value.path)
        target = _schema_type(self.schema, other.path) if isinstance(other, Column) else None
        return self.pc.scalar(_scalar(self.pa, value, target))

    def build(self, node: Node) -> Any:
        pc = self.pc
        op = node.op
        if op == "and":
            out = self.build(node.args[0])
            for arg in node.args[1:]:
                out = out & self.build(arg)
            return out
        if op == "or":
            out = self.build(node.args[0])
            for arg in node.args[1:]:
                out = out | self.build(arg)
            return out
        if op == "not":
            return ~self.build(node.args[0])
        if op == "true":
            return pc.scalar(True)
        if op == "false":
            return pc.scalar(False)
        if op == "column":
            return self.value(node.args[0])
        if op in ("eq", "ne", "lt", "le", "gt", "ge"):
            left, right = node.args
            a, b = self.value(left, right), self.value(right, left)
            return {
                "eq": a == b,
                "ne": a != b,
                "lt": a < b,
                "le": a <= b,
                "gt": a > b,
                "ge": a >= b,
            }[op]
        if op == "eq_null_safe":
            left, right = node.args
            a, b = self.value(left, right), self.value(right, left)
            both_null = a.is_null() & b.is_null()
            return pc.if_else(both_null, True, pc.coalesce(a == b, pc.scalar(False)))
        if op in ("is_null", "is_not_null"):
            target = self.value(node.args[0])
            return target.is_null() if op == "is_null" else target.is_valid()
        if op == "in":
            target, *items = node.args
            expr = self.value(target)
            clauses = [expr == self.value(i, target) for i in items]
            out = clauses[0]
            for clause in clauses[1:]:
                out = out | clause
            return ~out if node.negated else out
        if op == "between":
            target, low, high = node.args
            expr = self.value(target)
            out = (expr >= self.value(low, target)) & (expr <= self.value(high, target))
            return ~out if node.negated else out
        if op == "like":
            target, pattern = node.args
            out = pc.match_like(self.value(target), pattern.value)
            return ~out if node.negated else out
        raise PredicateError(f"unsupported predicate operator {op!r}")


def to_arrow(node: Node, schema: Any) -> Any:
    """An exact pyarrow compute expression bound to `schema`'s column types."""
    return _ArrowBuilder(schema).build(node)


def filter_table(table: Any, predicate: str | Node) -> Any:
    """Keep the rows of a pyarrow Table for which `predicate` is TRUE."""
    node = parse(predicate) if isinstance(predicate, str) else predicate
    return table.filter(to_arrow(node, table.schema))


def filter_stream(stream: Any, predicate: str | Node, *, keep: list[str] | None = None) -> Any:
    """Filter an Arrow stream batch by batch, optionally projecting afterwards.

    Returns a `pyarrow.RecordBatchReader`. Batches are filtered in order and
    never reordered. `keep` drops columns that were read only so the predicate
    could be evaluated.
    """
    import pyarrow as pa

    node = parse(predicate) if isinstance(predicate, str) else predicate
    reader = pa.RecordBatchReader.from_stream(stream)
    expr = to_arrow(node, reader.schema)
    out_schema = pa.schema([reader.schema.field(name) for name in keep]) if keep else reader.schema

    def batches() -> Iterator[Any]:
        for batch in reader:
            filtered = pa.Table.from_batches([batch]).filter(expr)
            if keep:
                filtered = filtered.select(keep)
            yield from filtered.to_batches()

    return pa.RecordBatchReader.from_batches(out_schema, batches())
