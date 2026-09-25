"""SQL predicates, parsed once and used two ways.

delta-rs takes a predicate as a SQL string and evaluates it with DataFusion. The
kernel takes a structured `Predicate` and uses it only to *skip files*: it never
promises to drop a non-matching row. So a predicate on the kernel path needs
both halves -- a structure the kernel can skip with, and an exact row filter
applied afterward -- and both have to mean the same thing as the SQL string.

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

Out of scope: arithmetic, function calls other than the
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
     |(?P<string>'(?:[^'\\]|\\.|'')*')
     |(?P<dstring>"(?:[^"\\]|\\.|"")*")
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
        if tok.kind in ("string", "dstring"):
            # Spark treats double quotes as a string literal by default.
            return Literal(_unescape(tok.text), "string")
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
                raw = _unescape(self.take().text)
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


_SPARK_ESCAPES = {
    "'": "'",
    '"': '"',
    "b": "\b",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "Z": "\x1a",
    "\\": "\\",
    "0": "\x00",
    # Kept escaped, so LIKE still sees a literal % or _.
    "%": "\\%",
    "_": "\\_",
}


def _unescape(token: str) -> str:
    """The value of a quoted SQL string token, as Spark reads it.

    Spark processes backslash escapes in string literals (`'a\\b'` is `a\b`,
    `'it\'s'` is `it's`), which is what the SQL warehouse evaluates; a doubled
    quote is also accepted. The kernel and delta-rs paths used to take the
    backslashes literally, so the same predicate matched different rows.
    """
    quote, body = token[0], token[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == quote and i + 1 < len(body) and body[i + 1] == quote:
            out.append(quote)
            i += 2
            continue
        if ch != "\\" or i + 1 >= len(body):
            out.append(ch)
            i += 1
            continue
        nxt = body[i + 1]
        octal = body[i + 1 : i + 4]
        if re.fullmatch(r"[0-3][0-7][0-7]", octal):
            out.append(chr(int(octal, 8)))
            i += 4
        elif nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", body[i + 2 : i + 6]):
            out.append(chr(int(body[i + 2 : i + 6], 16)))
            i += 6
        else:
            out.append(_SPARK_ESCAPES.get(nxt, nxt))
            i += 2
    return "".join(out)


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


def _skip(
    node: Node, exact: bool = False, unsafe: dict[tuple[str, ...], str] | None = None
) -> dict[str, Any] | None:
    """The kernel form of `node`, or None where skipping must not be attempted.

    Dropping a conjunct weakens a predicate, which is safe for skipping -- a
    weaker predicate skips fewer files. Under a NOT the polarity flips and a
    weaker child makes a *stronger* predicate, so beneath a NOT every part must
    be rendered exactly or not at all. `exact` carries that requirement down.
    """
    op = node.op
    if op == "and":
        rendered = [_skip(a, exact, unsafe) for a in node.args]
        if exact and any(p is None for p in rendered):
            return None
        parts = [p for p in rendered if p is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else {"op": "and", "args": parts}
    if op == "or":
        rendered = [_skip(a, exact, unsafe) for a in node.args]
        disjuncts = [p for p in rendered if p is not None]
        if len(disjuncts) != len(rendered):
            return None
        return {"op": "or", "args": disjuncts}
    if op == "not":
        inner = _skip(node.args[0], exact=True, unsafe=unsafe)
        return None if inner is None else {"op": "not", "args": [inner]}
    if unsafe and not _stats_can_decide(node, exact, unsafe):
        return None
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


def _unsafe_columns(schema: Any) -> dict[tuple[str, ...], str]:
    """Column paths (lower-cased) whose min/max statistics cannot decide every comparison.

    * floating point: Parquet and Delta statistics leave NaN out of min/max,
      but NaN is a value SQL compares (equal to itself, above everything), so
      a file whose max is 1.0 may still hold a NaN that satisfies `f > 2`.
    * DECIMAL wider than 15 digits: delta-rs writes decimal statistics as
      JSON doubles, so 12345678901234.5678 is stored as ...234.568 and a
      file holding exactly the literal was skipped by `d <= 12345678901234.5678`.
    """
    import pyarrow as pa

    out: dict[tuple[str, ...], str] = {}

    def walk(fields: Any, prefix: tuple[str, ...]) -> None:
        for f in fields:
            path = (*prefix, f.name.lower())
            t = f.type
            if pa.types.is_floating(t):
                out[path] = "float"
            elif pa.types.is_decimal(t) and t.precision > 15:
                out[path] = "decimal"
            elif pa.types.is_struct(t):
                walk([t.field(i) for i in range(t.num_fields)], path)

    walk(list(schema), ())
    return out


def _is_nan_free(value: Any) -> bool:
    return isinstance(value, Literal) and (
        value.value is None
        or (isinstance(value.value, (int, decimal.Decimal)) and not isinstance(value.value, bool))
        or (isinstance(value.value, float) and not math.isnan(value.value))
    )


def _stats_can_decide(node: Node, exact: bool, unsafe: dict[tuple[str, ...], str]) -> bool:
    """Whether min/max statistics can skip for this leaf without losing rows."""
    kinds = {unsafe.get(tuple(p.lower() for p in c)) for c in columns_of(node)} - {None}
    if not kinds or node.op in ("is_null", "is_not_null"):
        return True
    if "decimal" in kinds or exact:
        # Under a NOT the complement of a NaN-free range includes NaN.
        return False
    op, args = node.op, node.args
    if op == "eq":
        return isinstance(args[0], Column) != isinstance(args[1], Column) and all(
            _is_nan_free(a) for a in args if isinstance(a, Literal)
        )
    if op in ("lt", "le", "gt", "ge"):
        left, right = args
        # Only the side of the range that NaN (the largest value) is not on.
        if isinstance(left, Column) and op in ("lt", "le"):
            return _is_nan_free(right)
        if isinstance(right, Column) and op in ("gt", "ge"):
            return _is_nan_free(left)
        return False
    if op in ("in", "between") and not node.negated:
        return isinstance(args[0], Column) and all(_is_nan_free(a) for a in args[1:])
    return False


def to_kernel_json(node: Node, schema: Any = None) -> str | None:
    """The skipping predicate as the native extension's JSON, or None.

    With the table's Arrow `schema`, comparisons that file statistics cannot
    decide safely (NaN in floating columns, lossy wide-decimal stats) are
    left out of skipping; the exact row filter still applies them.
    """
    unsafe = _unsafe_columns(schema) if schema is not None else None
    rendered = _skip(node, unsafe=unsafe)
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


def _is_numeric(pa: Any) -> Any:
    t = pa.types
    return lambda x: bool(t.is_integer(x) or t.is_floating(x) or t.is_decimal(x))


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


_INTEGER_DIGITS = {8: 3, 16: 5, 32: 10, 64: 20}


def _common_decimal(pa: Any, column: Any, literal: Any) -> Any:
    """A decimal type holding every value of `column` (integer or decimal) and `literal`."""
    if pa.types.is_integer(column):
        precision, scale = _INTEGER_DIGITS[column.bit_width], 0
    else:
        precision, scale = column.precision, column.scale
    lit_scale = literal.scale
    scale_out = max(scale, lit_scale)
    digits = max(precision - scale, literal.precision - lit_scale) + scale_out
    if digits <= 38:
        return pa.decimal128(digits, scale_out)
    return pa.decimal256(min(digits, 76), scale_out)


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

    is_number = isinstance(value, (int, float, decimal.Decimal)) and not isinstance(value, bool)
    if is_number and (types.is_float16(target) or types.is_float32(target)):
        # Spark compares FLOAT with a DOUBLE/DECIMAL literal as DOUBLE. An
        # integer literal is widened the same way: casting it to float32
        # fails (or rounds) past 2**24, and Arrow's own implicit cast of the
        # literal then raised "Integer value ... not in range".
        return pa.float64(), pa.scalar(float(value), type=pa.float64())
    if isinstance(value, float) and (types.is_integer(target) or types.is_decimal(target)):
        # Spark compares an integral or DECIMAL column with a DOUBLE as DOUBLE;
        # narrowing the double to the column's type moved the boundary
        # (`i64 <= 9007199254740992D`), and Arrow's implicit cast could fail.
        return pa.float64(), pa.scalar(value, type=pa.float64())
    if isinstance(value, int) and not isinstance(value, bool) and types.is_decimal(target):
        # Arrow refuses int64 -> decimal(5,2) for any value (19 digits do not
        # fit), and comparing the two raised "Precision is not great enough".
        value = decimal.Decimal(value)
    exact_number = types.is_integer(target) or types.is_decimal(target)
    if isinstance(value, decimal.Decimal) and exact_number:
        scalar = _to_scalar(pa, value)
        cast = _try_cast(pa, scalar, target)
        if cast is not None and cast.as_py() == value:
            return None, cast
        # Not representable in the column's type: compare both in a decimal
        # wide enough for each, as Spark's decimal promotion does.
        wide = _common_decimal(pa, target, scalar.type)
        return wide, scalar.cast(wide)

    scalar = _to_scalar(pa, value)
    if scalar.type == target:
        return None, scalar
    cast = _try_cast(pa, scalar, target)
    if (
        cast is None
        and types.is_integer(target)
        and types.is_int64(scalar.type)
        and target.bit_width < 64
    ):
        # `i8 >= 1000`: the literal does not fit the column, so the column
        # is widened. Left implicit, delta-rs DELETE/UPDATE failed with
        # "Can't cast value 1000 to type Int8".
        return pa.int64(), scalar
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
            (a, a_type), (b, b_type) = self.column(left), self.column(right)
            floating, numeric = self.pa.types.is_floating, _is_numeric(self.pa)
            if (floating(a_type) or floating(b_type)) and numeric(a_type) and numeric(b_type):
                # Spark compares a BIGINT/DECIMAL column with a DOUBLE one as
                # DOUBLE; Arrow's implicit cast is a safe one and raised
                # "Integer value ... not in range" past 2**53.
                wide = self.pa.float64()
                a = a if a_type == wide else a.cast(wide, safe=False)
                b = b if b_type == wide else b.cast(wide, safe=False)
            return a, b
        if isinstance(left, Column):
            expr, type_ = self.column(left)
            cast, scalar = _coerce(self.pa, right, type_)  # type: ignore[arg-type]
            # Unsafe: widening int64 to DOUBLE (Spark's comparison type) must
            # round, not raise, past 2**53.
            return (expr if cast is None else expr.cast(cast, safe=False)), self.pc.scalar(scalar)
        if isinstance(right, Column):
            b, a = self.pair(right, left)
            return a, b
        return self.single(left), self.single(right)

    def _floating(self, left: Column | Literal, right: Column | Literal) -> bool:
        """Whether the comparison of `left` and `right` happens in floating point."""
        floating = self.pa.types.is_floating
        types = []
        for side, other in ((left, right), (right, left)):
            if isinstance(side, Column):
                type_ = self.column(side)[1]
                if isinstance(other, Literal):
                    cast, scalar = _coerce(self.pa, other, type_)
                    types.append(cast if cast is not None else type_)
                    types.append(scalar.type)
                else:
                    types.append(type_)
        if not types:
            return any(isinstance(v, Literal) and isinstance(v.value, float) for v in (left, right))
        return any(floating(t) for t in types)

    def compare(self, op: str, left: Column | Literal, right: Column | Literal) -> Any:
        """`left <op> right` with Spark's floating-point semantics.

        Spark (and Delta) treat NaN as equal to itself and greater than every
        other value, while IEEE comparisons -- Arrow's -- make NaN unequal to
        everything, so `f = 'NaN'` matched nothing and `f > 1` dropped NaN.
        """
        a, b = self.pair(left, right)
        if not self._floating(left, right):
            return {
                "eq": lambda: a == b,
                "ne": lambda: a != b,
                "lt": lambda: a < b,
                "le": lambda: a <= b,
                "gt": lambda: a > b,
                "ge": lambda: a >= b,
            }[op]()
        pc = self.pc
        nan_a, nan_b = pc.is_nan(a), pc.is_nan(b)

        def eq() -> Any:
            return (a == b) | (nan_a & nan_b)

        def lt(x: Any, y: Any, nan_x: Any, nan_y: Any) -> Any:
            return (x < y) | (nan_y & ~nan_x)

        comparisons: dict[str, Any] = {
            "eq": eq,
            "ne": lambda: ~eq(),
            "lt": lambda: lt(a, b, nan_a, nan_b),
            "le": lambda: lt(a, b, nan_a, nan_b) | eq(),
            "gt": lambda: lt(b, a, nan_b, nan_a),
            "ge": lambda: lt(b, a, nan_b, nan_a) | eq(),
        }
        return comparisons[op]()

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
        items_py = [s.as_py() for s in scalars]
        if pa.types.is_floating(type_) and any(v == 0 for v in items_py):
            # is_in hashes -0.0 and 0.0 apart; SQL compares them equal.
            items_py += [0.0, -0.0]
        try:
            values = pa.array(items_py, type=type_)
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
            return self.compare(op, *node.args)
        if op == "eq_null_safe":
            a, b = self.pair(*node.args)
            both_null = a.is_null() & b.is_null()
            equal = self.compare("eq", *node.args)
            return pc.if_else(both_null, True, pc.coalesce(equal, pc.scalar(False)))
        if op in ("is_null", "is_not_null"):
            target = self.single(node.args[0])
            return target.is_null() if op == "is_null" else target.is_valid()
        if op == "in":
            target, *items = node.args
            out = self.membership(target, items)
            if out is None:
                clauses = [self.compare("eq", target, item) for item in items]
                out = _balanced(clauses, lambda x, y: x | y)
            return ~out if node.negated else out
        if op == "between":
            target, low, high = node.args
            out = self.compare("ge", target, low) & self.compare("le", target, high)
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


# ----------------------------------------------------- DataFusion (delta-rs)


class _DataFusionRenderer:
    """Render a parsed predicate as DataFusion SQL that means what Spark means.

    delta-rs evaluates a predicate string with DataFusion, whose dialect is not
    Spark's: identifiers are case-sensitive, `1.5` is a DOUBLE rather than a
    DECIMAL, `5L`/`1.5D`/TIMESTAMP_NTZ do not parse, and NOT IN with a NULL
    is mis-simplified. Rendering from the same AST the kernel path filters
    with -- columns spelled as the schema spells them and every literal
    already typed for its column by `_coerce` -- gives both engines one
    meaning.
    """

    def __init__(self, schema: Any) -> None:
        import pyarrow as pa

        self.pa = pa
        self.schema = schema

    def column(self, col: Column) -> tuple[str, Any]:
        resolved = _resolve_path(self.schema, col.path)
        if resolved is None:
            raise PredicateError(f"the predicate references unknown column {'.'.join(col.path)!r}")
        path, type_ = resolved
        text = '"' + path[0].replace('"', '""') + '"'
        for name in path[1:]:
            text += "[" + _df_string(name) + "]"
        pa = self.pa
        if pa.types.is_floating(type_) or (pa.types.is_decimal(type_) and type_.precision > 15):
            # delta-rs trusts file statistics for these, and they are not
            # exact: NaN is left out of min/max, and wide decimals are stored
            # as JSON doubles. A file whose min equals its max then had the
            # column replaced by that value -- a NaN read as 3.25, and
            # 12345678901234.5678 as ...5680, so a DELETE removed rows it
            # should have kept. Reading through a string round trip (exact
            # for both) keeps the statistics out of it.
            text = f"arrow_cast(CAST({text} AS VARCHAR), '{_df_type(pa, type_)}')"
            if pa.types.is_floating(type_):
                # Arrow-rs compares floats by total order, where -0.0 < 0.0;
                # SQL has them equal. Adding zero turns -0.0 into 0.0.
                text = f"({text} + arrow_cast('0.0', '{_df_type(pa, type_)}'))"
        return text, type_

    def scalar(self, scalar: Any) -> str:
        pa = self.pa
        t = scalar.type
        value = scalar.as_py()
        if value is None:
            return "NULL"
        if pa.types.is_boolean(t):
            return _df_bool(value)
        if pa.types.is_integer(t):
            if pa.types.is_int64(t):
                return str(value)
            return f"arrow_cast({value}, '{_df_type(pa, t)}')"
        if pa.types.is_floating(t):
            # `+ 0.0` turns -0.0 into 0.0, as the column side is normalised.
            text = repr(float(value) + 0.0) if math.isfinite(value) else str(value)
            return f"arrow_cast({_df_string(text)}, '{_df_type(pa, t)}')"
        if pa.types.is_decimal(t):
            return f"arrow_cast({_df_string(format(value, 'f'))}, '{_df_type(pa, t)}')"
        if _is_text(pa, t):
            return _df_string(value)
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            return f"X'{bytes(value).hex()}'"
        if pa.types.is_date32(t):
            return f"DATE '{value.isoformat()}'"
        if pa.types.is_timestamp(t):
            return f"arrow_cast({_df_string(value.isoformat())}, '{_df_type(pa, t)}')"
        raise _Unrenderable(t)

    def literal(self, lit: Literal) -> str:
        return self.scalar(_coerce(self.pa, lit, None)[1])

    def operands(self, left: Any, right: Any) -> tuple[str, str, Any]:
        """Both sides rendered and typed as the Arrow filter types them, plus
        the comparison type (None when two columns or two literals meet)."""
        if isinstance(left, Column) and isinstance(right, Column):
            return self.column(left)[0], self.column(right)[0], None
        if isinstance(left, Column):
            text, type_ = self.column(left)
            cast, scalar = _coerce(self.pa, right, type_)
            if cast is not None:
                text = f"arrow_cast({text}, '{_df_type(self.pa, cast)}')"
            return text, self.scalar(scalar), cast if cast is not None else type_
        if isinstance(right, Column):
            b, a, type_ = self.operands(right, left)
            return a, b, type_
        return self.literal(left), self.literal(right), None

    def null(self, *near: Any) -> str:
        """A NULL boolean DataFusion cannot fold to a constant.

        delta-rs mis-handles a predicate that simplifies to a NULL literal:
        DELETE ... WHERE id = NULL deleted every row (SQL deletes none), and a
        scan by `id > 0 AND CAST(NULL AS BOOLEAN)` returned every row. NULLIF
        of a column test with itself is NULL for every row but is not folded.
        """
        column = next((c for c in near if isinstance(c, Column)), None)
        if column is not None and _resolve_path(self.schema, column.path) is not None:
            text = self.column(column)[0]
        else:
            text = '"' + self.schema.names[0].replace('"', '""') + '"'
        return f"NULLIF(({text} IS NULL), ({text} IS NULL))"

    def compare(self, op: str, left: Any, right: Any) -> str:
        if any(isinstance(v, Literal) and v.value is None for v in (left, right)):
            # A comparison with NULL is NULL whatever the row holds.
            return self.null(left, right)
        a, b, type_ = self.operands(left, right)
        sym = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[op]
        plain = f"({a} {sym} {b})"
        if type_ is None or not self.pa.types.is_floating(type_):
            return plain
        # Spark: NaN equals NaN and sorts above every other value. File
        # statistics leave NaN out of min/max, so a comparison NaN satisfies
        # must not be prunable; CASE keeps DataFusion from skipping files
        # (under a NOT too) while giving Spark's answer row by row.
        col_first = isinstance(left, Column)
        column = a if col_first else b
        other = right if col_first else left
        # The truth value when the column is NaN, reading `column <op> other`.
        mirrored = op if col_first else {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}.get(op, op)
        if _literal_is_nan(other):
            at_nan = {"eq": True, "ne": False, "lt": False, "le": True, "gt": False, "ge": True}
            below = {"eq": False, "ne": True, "lt": True, "le": True, "gt": False, "ge": False}
            # Built from isnan() alone, which is NULL for a NULL column: a CASE
            # with a `THEN NULL` branch is mis-simplified by DataFusion into
            # one that keeps the NULL rows.
            nan = f"isnan({column})"
            return {
                (True, True): f"({nan} OR NOT {nan})",
                (False, False): f"({nan} AND NOT {nan})",
                (True, False): f"({nan})",
                (False, True): f"(NOT {nan})",
            }[(at_nan[mirrored], below[mirrored])]
        above = {"eq": False, "ne": True, "lt": False, "le": False, "gt": True, "ge": True}
        return f"(CASE WHEN isnan({column}) THEN {_df_bool(above[mirrored])} ELSE {plain} END)"

    def render(self, node: Node) -> str:
        op = node.op
        if op in ("and", "or"):
            word = op.upper()
            joined: str = _balanced(
                [self.render(a) for a in node.args], lambda x, y: f"({x} {word} {y})"
            )
            return joined
        if op == "not":
            return f"(NOT {self.render(node.args[0])})"
        if op in ("true", "false"):
            return op.upper()
        if op == "column":
            return self.column(node.args[0])[0]
        if op in ("eq", "ne", "lt", "le", "gt", "ge"):
            return self.compare(op, *node.args)
        if op == "eq_null_safe":
            left, right = node.args
            for this, other in ((left, right), (right, left)):
                if isinstance(other, Literal) and other.value is None:
                    if isinstance(this, Literal):
                        return "TRUE" if this.value is None else "FALSE"
                    return f"({self.column(this)[0]} IS NULL)"
            a, b, _ = self.operands(left, right)
            equal = self.compare("eq", left, right)
            return (
                f"(CASE WHEN {a} IS NULL AND {b} IS NULL THEN TRUE "
                f"WHEN {a} IS NULL OR {b} IS NULL THEN FALSE ELSE {equal} END)"
            )
        if op in ("is_null", "is_not_null"):
            (target,) = node.args
            text = self.column(target)[0] if isinstance(target, Column) else self.literal(target)
            return f"({text} IS {'NOT ' if op == 'is_not_null' else ''}NULL)"
        if op == "in":
            target, *items = node.args
            present = [i for i in items if not (isinstance(i, Literal) and i.value is None)]
            if present:
                out = _balanced(
                    [self.compare("eq", target, item) for item in present],
                    lambda x, y: f"({x} OR {y})",
                )
                if len(present) != len(items):
                    # SQL: no match against a list holding NULL is NULL, not
                    # FALSE; DataFusion simplifies `x IN (1, NULL)` to `x IN (1)`
                    # (and `.. OR NULL` likewise), and prunes partitions wrongly
                    # on a CASE with an ELSE NULL, keeping every row.
                    out = f"NULLIF({out}, FALSE)"
            else:
                out = self.null(target)
            return f"(NOT {out})" if node.negated else out
        if op == "between":
            target, low, high = node.args
            out = f"({self.compare('ge', target, low)} AND {self.compare('le', target, high)})"
            return f"(NOT {out})" if node.negated else out
        if op in ("like", "ilike"):
            target, pattern = node.args
            if isinstance(target, Column):
                subject, type_ = self.column(target)
                if not _is_text(self.pa, type_):
                    subject = f"CAST({subject} AS VARCHAR)"
            else:
                subject = self.literal(target)
            kw = ("NOT " if node.negated else "") + ("ILIKE" if op == "ilike" else "LIKE")
            return f"({subject} {kw} {_df_string(pattern.value)})"
        raise PredicateError(f"unsupported predicate operator {op!r}")


def _df_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _df_bool(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def _literal_is_nan(value: Any) -> bool:
    if not isinstance(value, Literal):
        return False
    if isinstance(value.value, float):
        return math.isnan(value.value)
    return isinstance(value.value, str) and value.value.strip().lower() in ("nan", "+nan", "-nan")


_DF_TYPES = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "halffloat": "Float16",
    "float": "Float32",
    "double": "Float64",
    "bool": "Boolean",
    "string": "Utf8",
    "large_string": "LargeUtf8",
    "date32[day]": "Date32",
}


def _df_type(pa: Any, t: Any) -> str:
    """`t` spelled as DataFusion's arrow_cast() spells an Arrow type."""
    if str(t) in _DF_TYPES:
        return _DF_TYPES[str(t)]
    if pa.types.is_decimal128(t):
        return f"Decimal128({t.precision}, {t.scale})"
    if pa.types.is_decimal256(t):
        return f"Decimal256({t.precision}, {t.scale})"
    if pa.types.is_timestamp(t):
        unit = {"s": "Second", "ms": "Millisecond", "us": "Microsecond", "ns": "Nanosecond"}[t.unit]
        tz = f'Some("{t.tz}")' if t.tz else "None"
        return f"Timestamp({unit}, {tz})"
    raise _Unrenderable(t)


def to_datafusion(node: Node, schema: Any) -> str | None:
    """`node` as a DataFusion predicate over `schema`, or None when a type has
    no DataFusion spelling here (the caller then passes its text through)."""
    # Validate as the kernel path does, so both engines raise the same
    # PredicateError for an unknown column or an impossible comparison.
    to_arrow(node, schema)
    try:
        return _DataFusionRenderer(schema).render(node)
    except _Unrenderable:
        return None


def to_datafusion_value(value: Column | Literal, schema: Any, target: Any) -> str | None:
    """An UPDATE right-hand side as DataFusion SQL, typed for a `target` column.

    A literal is typed the way the kernel path types it (so `'a\\b'`,
    `"text"`, `12345678901234.5678` into a DECIMAL(18,4), and TIMESTAMP_NTZ
    mean the same on both engines); a column is spelled as the schema spells
    it. None when there is no DataFusion spelling here.
    """
    import pyarrow as pa

    renderer = _DataFusionRenderer(schema)
    try:
        if isinstance(value, Column):
            resolved = _resolve_path(schema, value.path)
            if resolved is None:
                return None
            path = resolved[0]
            return (
                '"'
                + path[0].replace('"', '""')
                + '"'
                + "".join("[" + _df_string(p) + "]" for p in path[1:])
            )
        if value.value is None:
            return "NULL"
        _, scalar = _coerce(pa, value, target)
        return renderer.scalar(scalar)
    except (_Unrenderable, PredicateError):
        return None


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
