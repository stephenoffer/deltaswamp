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
import math
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
      (?P<number>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[LlDdFf]?)
     |(?P<string>'(?:[^']|'')*')
     |(?P<dstring>"(?:[^"]|"")*")
     |(?P<quoted>`(?:[^`]|``)+`)
     |(?P<op><=>|<=|>=|<>|!=|==|=|<|>|\(|\)|,|\.)
     |(?P<word>(?:[^\W\d]|_)\w*)
    )""",
    re.VERBOSE,
)

_KEYWORDS = frozenset(
    {"AND", "OR", "NOT", "IS", "NULL", "IN", "BETWEEN", "LIKE", "ILIKE", "TRUE", "FALSE"}
)
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
        return parts[0] if len(parts) == 1 else Node("or", _flatten("or", parts))

    def and_expr(self) -> Node:
        parts = [self.not_expr()]
        while self.keyword("AND"):
            self.i += 1
            parts.append(self.not_expr())
        return parts[0] if len(parts) == 1 else Node("and", _flatten("and", parts))

    def not_expr(self) -> Node:
        if self.keyword("NOT"):
            self.i += 1
            return Node("not", (self.not_expr(),))
        return self.predicate()

    def predicate(self) -> Node:
        # A parenthesised boolean expression, unless it is a parenthesised value
        # such as `(x) = 1`: then back up and parse it as the left operand.
        if self.symbol("(") and not self._continues_comparison(self._matching_paren() + 1):
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
            if self.keyword("DISTINCT"):
                # `IS [NOT] DISTINCT FROM` is the standard spelling of `<=>`.
                self.i += 1
                self.expect_keyword("FROM")
                same = Node("eq_null_safe", (left, self.value()))
                return same if negated else Node("not", (same,))
            if self.keyword("TRUE") or self.keyword("FALSE"):
                # `x IS TRUE` is never NULL: a NULL x is simply not TRUE.
                truth = self.take().text.upper() == "TRUE"
                same = Node("eq_null_safe", (left, Literal(truth, "boolean")))
                return Node("not", (same,)) if negated else same
            self.expect_keyword("NULL")
            return Node("is_not_null" if negated else "is_null", (left,))

        negated = False
        if self.keyword("NOT") and (
            self.keyword("IN", 1)
            or self.keyword("BETWEEN", 1)
            or self.keyword("LIKE", 1)
            or self.keyword("ILIKE", 1)
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

        if self.keyword("LIKE") or self.keyword("ILIKE"):
            op = "ilike" if self.take().text.upper() == "ILIKE" else "like"
            pattern = self.value()
            if not (isinstance(pattern, Literal) and isinstance(pattern.value, str)):
                raise PredicateError(f"LIKE needs a string pattern in {self.text!r}")
            if self.keyword("ESCAPE"):
                self.i += 1
                escape = self.value()
                if not (
                    isinstance(escape, Literal)
                    and isinstance(escape.value, str)
                    and len(escape.value) == 1
                ):
                    raise PredicateError(f"ESCAPE needs a single-character string in {self.text!r}")
                pattern = Literal(_backslash_escaped(pattern.value, escape.value), "string")
            return Node(op, (left, pattern), negated)

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

    def _matching_paren(self) -> int:
        """Index of the `)` closing the `(` at the cursor (past the end if none)."""
        depth = 0
        for j in range(self.i, len(self.tokens)):
            tok = self.tokens[j]
            if tok.kind == "op" and tok.text == "(":
                depth += 1
            elif tok.kind == "op" and tok.text == ")":
                depth -= 1
                if depth == 0:
                    return j
        return len(self.tokens)

    def _continues_comparison(self, at: int) -> bool:
        """Whether token `at` makes what precedes it a value, not a predicate."""
        offset = at - self.i
        tok = self.peek(offset)
        if tok is None:
            return False
        if tok.kind == "op" and tok.text in _COMPARISONS:
            return True
        if any(self.keyword(w, offset) for w in ("IS", "IN", "BETWEEN", "LIKE", "ILIKE")):
            return True
        return self.keyword("NOT", offset) and any(
            self.keyword(w, offset + 1) for w in ("IN", "BETWEEN", "LIKE", "ILIKE")
        )

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
            if (
                upper in _TYPED_LITERALS
                and following is not None
                and following.kind in ("string", "dstring")
            ):
                quote = following.text[0]
                raw = self.take().text[1:-1].replace(quote * 2, quote)
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


def _backslash_escaped(pattern: str, escape: str) -> str:
    """Rewrite a LIKE pattern using `escape` into the backslash form Arrow reads."""
    if escape == "\\":
        return pattern
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == escape and i + 1 < len(pattern):
            out.append("\\" + pattern[i + 1])
            i += 2
            continue
        out.append("\\\\" if ch == "\\" else ch)
        i += 1
    return "".join(out)


def _flatten(op: str, parts: list[Node]) -> tuple[Node, ...]:
    """Splice nested same-op children: `a OR (b OR c)` is one three-way OR.

    Generated SQL nests parentheses deeply, and the skipping JSON would nest
    as deep -- past serde_json's recursion limit, which rejects the whole
    predicate as invalid JSON.
    """
    out: list[Node] = []
    for part in parts:
        if part.op == op:
            out.extend(part.args)
        else:
            out.append(part)
    return tuple(out)


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


_LONG_MIN, _LONG_MAX = -(2**63), 2**63 - 1


def _number(text: str) -> Literal:
    suffix = text[-1].upper() if text[-1].isalpha() and text[-1] not in "eE" else ""
    body = text[:-1] if suffix else text
    if suffix == "L":
        if any(c in body for c in ".eE"):
            raise PredicateError(f"{text!r} is not a valid BIGINT literal")
        value = int(body)
        if not _LONG_MIN <= value <= _LONG_MAX:
            raise PredicateError(f"{text!r} is out of range for a BIGINT literal")
        return Literal(value, "long")
    if suffix in ("D", "F") or any(c in body for c in ".eE"):
        if suffix == "" and "." in body and not any(c in body for c in "eE"):
            # SQL: an unsuffixed 1.5 is a DECIMAL. Keep exactness for the
            # filter; the kernel coerces to the column type anyway.
            return Literal(decimal.Decimal(body), "decimal")
        return Literal(float(body), "double")
    value = int(body)
    if not _LONG_MIN <= value <= _LONG_MAX:
        # Spark types an integer literal too wide for BIGINT as a DECIMAL; it
        # cannot be a Python int handed to Arrow or the kernel as a long.
        return Literal(decimal.Decimal(body), "decimal")
    return Literal(value, "long")


_UTC_SUFFIX = re.compile(r"(?:\s*(?:[Zz]|UTC|utc|GMT|gmt))$")


def _parse_datetime(raw: str) -> _dt.datetime:
    """An ISO timestamp, accepting the `Z` / ` UTC` suffixes SQL users write."""
    text = raw.strip()
    if _UTC_SUFFIX.search(text):
        text = _UTC_SUFFIX.sub("", text) + "+00:00"
    return _dt.datetime.fromisoformat(text)


def _parse_date(raw: str) -> _dt.date:
    text = raw.strip()
    try:
        return _dt.date.fromisoformat(text)
    except ValueError:
        # Spark accepts unpadded fields: DATE '2024-1-5'.
        match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
        if match is None:
            raise
        return _dt.date(*(int(g) for g in match.groups()))


def _typed(kind: str, raw: str, text: str) -> Literal:
    try:
        if kind == "DATE":
            return Literal(_parse_date(raw), "date")
        parsed = _parse_datetime(raw)
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
    try:
        return _Parser(text).parse()
    except RecursionError:
        raise PredicateError(
            f"predicate is nested too deeply to parse ({len(text)} chars)"
        ) from None


def parse_value(text: str) -> Column | Literal:
    """Parse a single SQL value: a literal or a column reference."""
    if not isinstance(text, str) or not text.strip():
        raise PredicateError(
            f"{text!r} is not a SQL expression string; SQL assignments are strings such "
            'as "\'abc\'" or "other_col" (pass Python values through new_values=)'
        )
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


class _Unrenderable(Exception):
    """A literal with no JSON form the kernel can read (e.g. infinity)."""


def _literal_json(lit: Literal) -> dict[str, Any]:
    value = lit.value
    if isinstance(value, float) and not math.isfinite(value):
        # json.dumps would emit `Infinity`, which is not JSON: the native
        # parser would reject the whole predicate instead of skipping less.
        raise _Unrenderable(value)
    if isinstance(value, _dt.datetime):
        return {"literal": value.isoformat(), "type": lit.type}
    if isinstance(value, _dt.date):
        return {"literal": value.isoformat(), "type": "date"}
    if isinstance(value, decimal.Decimal):
        # Positional notation: str() gives 1E-7, which no decimal parser
        # downstream accepts.
        return {"literal": format(value, "f"), "type": "decimal"}
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
    return _skip_leaf(node)


def _skip_leaf_unchecked(node: Node) -> dict[str, Any] | None:
    op = node.op
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


def _skip_leaf(node: Node) -> dict[str, Any] | None:
    try:
        return _skip_leaf_unchecked(node)
    except _Unrenderable:
        return None


def to_kernel_json(node: Node) -> str | None:
    """The skipping predicate as the native extension's JSON, or None."""
    rendered = _skip(node)
    return None if rendered is None else json.dumps(rendered)


# ------------------------------------------------------------- arrow (exact)


def _field(pc: Any, path: tuple[str, ...]) -> Any:
    return pc.field(path[0]) if len(path) == 1 else pc.field(*path)


def _resolve_path(schema: Any, path: tuple[str, ...]) -> tuple[tuple[str, ...], Any] | None:
    """`path` spelled as `schema` spells it, and its type; None if absent.

    Spark resolves column names case-insensitively, and so does the kernel's
    skipping conversion, so `ID = 1` must find column `id` here too. An exact
    match wins; otherwise a unique case-insensitive one. Two candidates that
    differ only by case are ambiguous, and that is an error, not a guess.
    """
    import pyarrow as pa

    fields = list(schema)
    resolved: list[str] = []
    current: Any = None
    for i, name in enumerate(path):
        matches = [f for f in fields if f.name == name]
        if not matches:
            matches = [f for f in fields if f.name.lower() == name.lower()]
        if not matches:
            return None
        if len(matches) > 1:
            raise PredicateError(
                f"column reference {'.'.join(path)!r} is ambiguous: it matches "
                f"{', '.join(repr(f.name) for f in matches)}; quote the exact name"
            )
        resolved.append(matches[0].name)
        current = matches[0].type
        if i + 1 < len(path):
            if not pa.types.is_struct(current):
                return None
            fields = [current.field(j) for j in range(current.num_fields)]
    return tuple(resolved), current


def _schema_type(schema: Any, path: tuple[str, ...]) -> Any:
    resolved = _resolve_path(schema, path)
    return None if resolved is None else resolved[1]


def _arrow_errors(pa: Any) -> tuple[type[BaseException], ...]:
    return (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError)


def _to_scalar(pa: Any, value: Any, type: Any = None) -> Any:
    try:
        return pa.scalar(value) if type is None else pa.scalar(value, type=type)
    # ArrowInvalid/ArrowTypeError/ArrowNotImplementedError subclass these.
    except (OverflowError, TypeError, ValueError, NotImplementedError) as exc:
        raise PredicateError(f"literal {value!r} has no Arrow representation: {exc}") from exc


def _try_cast(pa: Any, scalar: Any, target: Any) -> Any:
    try:
        return scalar.cast(target)
    except _arrow_errors(pa):
        return None


def _is_text(pa: Any, target: Any) -> bool:
    t = pa.types
    return bool(
        t.is_string(target)
        or t.is_large_string(target)
        or (hasattr(t, "is_string_view") and t.is_string_view(target))
    )


def _string_to_temporal(pa: Any, text: str, target: Any) -> Any:
    """A string literal compared with a DATE/TIMESTAMP column, parsed as Spark casts it."""
    try:
        if pa.types.is_date(target):
            try:
                return _parse_date(text)
            except ValueError:
                return _parse_datetime(text).date()
        parsed = _parse_datetime(text)
    except ValueError as exc:
        raise PredicateError(
            f"string literal {text!r} is compared with a {target} column but is not a "
            "valid date/timestamp"
        ) from exc
    if target.tz is None:
        return parsed.replace(tzinfo=None)
    # An unzoned string against a zoned column: UTC, as the kernel reads it.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=_dt.UTC)


def _coerce(pa: Any, lit: Literal, target: Any) -> tuple[Any, Any]:
    """(cast for the column or None, scalar) to compare `lit` with a `target` column.

    The literal is converted to the column's type where that is exact. Where
    it is not, the *column* is widened instead -- narrowing the literal would
    move the comparison boundary (a DATE column against a TIMESTAMP with a
    time of day, a FLOAT column against the decimal 1.1).
    """
    value = lit.value
    if value is None:
        return None, (pa.scalar(None) if target is None else pa.scalar(None, type=target))
    if target is None:
        return None, _to_scalar(pa, value)
    types = pa.types

    if isinstance(value, str) and not _is_text(pa, target):
        if types.is_date(target) or types.is_timestamp(target):
            value = _string_to_temporal(pa, value, target)
        elif (
            types.is_integer(target)
            or types.is_floating(target)
            or types.is_decimal(target)
            or types.is_boolean(target)
        ):
            cast = _try_cast(pa, pa.scalar(value), target)
            if cast is not None:
                return None, cast
            try:
                number = decimal.Decimal(value.strip())
            except decimal.InvalidOperation:
                number = None
            if number is None or not number.is_finite() or types.is_boolean(target):
                raise PredicateError(
                    f"string literal {value!r} cannot be compared with a {target} column: "
                    "it is not a valid value of that type"
                ) from None
            value = number

    if isinstance(value, _dt.datetime):
        if types.is_date(target):
            wide = pa.timestamp("us", tz="UTC" if value.tzinfo is not None else None)
            return wide, _to_scalar(pa, value, wide)
        if types.is_timestamp(target):
            scalar = _to_scalar(pa, value)
            cast = _try_cast(pa, scalar, target)
            if cast is not None:
                return None, cast
            # The column's unit is coarser than the literal (a fractional
            # second against timestamp[s]): compare at microseconds.
            wide = pa.timestamp("us", tz=target.tz)
            return wide, scalar.cast(wide)

    if isinstance(value, (float, decimal.Decimal)) and (
        types.is_float16(target) or types.is_float32(target)
    ):
        # Spark compares FLOAT with a DOUBLE/DECIMAL literal as DOUBLE.
        return pa.float64(), pa.scalar(float(value), type=pa.float64())

    scalar = _to_scalar(pa, value)
    if scalar.type == target:
        return None, scalar
    cast = _try_cast(pa, scalar, target)
    # Otherwise let Arrow's comparison kernels decide; they widen numerics.
    return None, scalar if cast is None else cast


def _balanced(items: list[Any], combine: Any) -> Any:
    """Fold `items` pairwise into a tree of depth log2(n), not a chain of depth n.

    Arrow binds and evaluates expressions recursively; a left-deep chain from a
    long `OR` or `IN` list overflows the native stack and kills the process.
    """
    while len(items) > 1:
        paired = [combine(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]
        if len(items) % 2:
            paired.append(items[-1])
        items = paired
    return items[0]


class _ArrowBuilder:
    def __init__(self, schema: Any) -> None:
        import pyarrow as pa
        import pyarrow.compute as pc

        self.pa = pa
        self.pc = pc
        self.schema = schema

    def column(self, col: Column) -> tuple[Any, Any]:
        resolved = _resolve_path(self.schema, col.path)
        if resolved is None:
            raise PredicateError(f"the predicate references unknown column {'.'.join(col.path)!r}")
        path, type_ = resolved
        return _field(self.pc, path), type_

    def single(self, value: Column | Literal) -> Any:
        if isinstance(value, Column):
            return self.column(value)[0]
        return self.pc.scalar(_coerce(self.pa, value, None)[1])

    def pair(self, left: Column | Literal, right: Column | Literal) -> tuple[Any, Any]:
        """Both operands of a comparison, the literal typed to the column."""
        if isinstance(left, Column) and isinstance(right, Column):
            return self.column(left)[0], self.column(right)[0]
        if isinstance(left, Column):
            expr, type_ = self.column(left)
            cast, scalar = _coerce(self.pa, right, type_)  # type: ignore[arg-type]
            return (expr if cast is None else expr.cast(cast)), self.pc.scalar(scalar)
        if isinstance(right, Column):
            b, a = self.pair(right, left)
            return a, b
        return self.single(left), self.single(right)

    def membership(self, target: Column | Literal, items: list[Any]) -> Any:
        """`target IN (literals...)` as one set lookup, with SQL's NULL rules.

        A chain of thousands of `=` clauses is deep enough to overflow the
        native stack while Arrow binds it, killing the process. Returns None
        where the literals do not share one type with the column, and the
        caller falls back to equality clauses.
        """
        if not isinstance(target, Column) or not all(isinstance(i, Literal) for i in items):
            return None
        pa, pc = self.pa, self.pc
        expr, type_ = self.column(target)
        has_null = any(i.value is None for i in items)
        scalars = []
        for item in items:
            if item.value is None:
                continue
            cast, scalar = _coerce(pa, item, type_)
            if cast is not None or scalar.type != type_:
                return None
            scalars.append(scalar)
        null = pc.scalar(pa.scalar(None, type=pa.bool_()))
        if not scalars:
            # `x IN (NULL)` is never TRUE: NULL for every row.
            return pc.if_else(expr.is_valid(), null, null)
        try:
            values = pa.array([s.as_py() for s in scalars], type=type_)
        except (TypeError, ValueError, NotImplementedError):
            return None
        found = pc.is_in(expr, value_set=values)
        # SQL: a match is TRUE; no match is FALSE, or NULL if the list holds a
        # NULL; and a NULL target is NULL whatever the list holds.
        miss = null if has_null else pc.scalar(False)
        return pc.if_else(expr.is_valid(), pc.if_else(found, pc.scalar(True), miss), null)

    def disjuncts(self, args: tuple[Any, ...]) -> list[Any]:
        """The children of an OR, with `c = 1 OR c = 2 OR ...` folded into one lookup.

        Generated SQL spells a list that way, and Arrow cannot evaluate an OR of
        more than a few hundred clauses without overflowing its stack.
        """
        groups: dict[tuple[str, ...], tuple[Column, list[Literal]]] = {}
        rest: list[Node] = []
        for arg in args:
            if arg.op == "eq":
                left, right = arg.args
                if isinstance(right, Column) and isinstance(left, Literal):
                    left, right = right, left
                if isinstance(left, Column) and isinstance(right, Literal):
                    groups.setdefault(left.path, (left, []))[1].append(right)
                    continue
            rest.append(arg)
        out = [self.build(a) for a in rest]
        for column, literals in groups.values():
            lookup = self.membership(column, literals) if len(literals) > 1 else None
            if lookup is None:
                out.extend(self.build(Node("eq", (column, lit))) for lit in literals)
            else:
                out.append(lookup)
        return out

    def build(self, node: Node) -> Any:
        pc = self.pc
        op = node.op
        if op == "and":
            return _balanced([self.build(a) for a in node.args], lambda x, y: x & y)
        if op == "or":
            return _balanced(self.disjuncts(node.args), lambda x, y: x | y)
        if op == "not":
            return ~self.build(node.args[0])
        if op == "true":
            return pc.scalar(True)
        if op == "false":
            return pc.scalar(False)
        if op == "column":
            (col,) = node.args
            expr, type_ = self.column(col)
            if not self.pa.types.is_boolean(type_):
                raise PredicateError(
                    f"column {'.'.join(col.path)!r} is {type_}, not BOOLEAN, so it cannot "
                    "stand alone as a condition; compare it with a value"
                )
            return expr
        if op in ("eq", "ne", "lt", "le", "gt", "ge"):
            a, b = self.pair(*node.args)
            return {
                "eq": lambda: a == b,
                "ne": lambda: a != b,
                "lt": lambda: a < b,
                "le": lambda: a <= b,
                "gt": lambda: a > b,
                "ge": lambda: a >= b,
            }[op]()
        if op == "eq_null_safe":
            a, b = self.pair(*node.args)
            both_null = a.is_null() & b.is_null()
            return pc.if_else(both_null, True, pc.coalesce(a == b, pc.scalar(False)))
        if op in ("is_null", "is_not_null"):
            target = self.single(node.args[0])
            return target.is_null() if op == "is_null" else target.is_valid()
        if op == "in":
            target, *items = node.args
            out = self.membership(target, items)
            if out is None:
                clauses = []
                for item in items:
                    a, b = self.pair(target, item)
                    clauses.append(a == b)
                out = _balanced(clauses, lambda x, y: x | y)
            return ~out if node.negated else out
        if op == "between":
            target, low, high = node.args
            lo_a, lo_b = self.pair(target, low)
            hi_a, hi_b = self.pair(target, high)
            out = (lo_a >= lo_b) & (hi_a <= hi_b)
            return ~out if node.negated else out
        if op in ("like", "ilike"):
            target, pattern = node.args
            subject = self.single(target)
            if isinstance(target, Column) and not _is_text(self.pa, self.column(target)[1]):
                # Spark casts a non-string operand of LIKE to STRING.
                subject = subject.cast(self.pa.string())
            out = pc.match_like(subject, pattern.value, ignore_case=op == "ilike")
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
