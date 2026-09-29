"""Spark SQL text, respelled for the engines that evaluate it outside Databricks.

The API takes Spark SQL everywhere -- the warehouse runs the same text -- but a
direct engine evaluates it in another dialect: delta-rs with DataFusion, the
kernel MERGE and the Delta Sharing filter with DuckDB. Where the dialects spell
one meaning differently, or give one spelling different meanings, the text is
rewritten here from a parse of Spark's expression grammar, so each engine
computes what the warehouse does. Each rule was checked against a SQL
warehouse (ANSI mode, as Databricks SQL runs):

* `"ab"` is a string, adjacent literals concatenate (`'it''s'` is `its`),
  and backslashes escape; both engines read `"ab"` as a column;
* `5 / 2` is 2.5 (DataFusion: 2); `7 DIV 2` truncates; `x RLIKE p` finds
  anywhere; `1.5D`, `7L`, `1.5BD` and `<=>` do not parse in one or both;
* a fraction CAST to an integral type truncates (DuckDB rounds 1.9 to 2),
  and `CAST('1.9' AS INT)` is an error (DuckDB: 2);
* `substring(s, 0, 2)` is `ab` (both: `a`), a negative position counts from
  the end (DataFusion: empty), and a negative length is empty;
* `concat` is NULL when any argument is (both skip NULLs);
* `log(x)` is the natural log (both: base 10);
* two-argument `trim`/`ltrim`/`rtrim` take the characters to trim first;
* `left`/`right` of a non-positive length are empty (both: all but some);
* `regexp_replace` replaces every match (both: the first);
* `^` is XOR (DuckDB: power);
* division and remainder by zero are errors (DuckDB: inf or NULL).

What cannot be made to agree without knowing the data -- an array subscript
(0-based in Spark, 1-based in both), `split` (a regex in Spark), a date
format pattern, `hash` -- is refused: `warehouse_reason` names it, so the call
routes to the warehouse instead of computing something else. Text outside the
grammar parsed here passes through with only its literals respelled; text that
also needs a rewrite the grammar was for (`DIV`, `RLIKE`) is refused.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..errors import EngineLimitError

__all__ = [
    "DUCKDB_MACROS",
    "check_expression",
    "install_duckdb_macros",
    "numeric_kinds",
    "parse_json_literal",
    "parses",
    "to_datafusion",
    "to_duckdb",
    "warehouse_reason",
]

_REMEDY = (
    "ds.connect(..., allow_sql_fallback=True) runs it on a SQL warehouse, which "
    "evaluates Spark SQL itself; or rewrite the expression"
)

# ------------------------------------------------------------------ tokens

_TOKEN = re.compile(
    r"""
      (?P<ws>\s+)
     |(?P<str>'(?:[^'\\]|\\.)*')
     |(?P<dstr>"(?:[^"\\]|\\.)*")
     |(?P<bq>`(?:[^`]|``)*`)
     |(?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?(?:[Bb][Dd]|[LlSsYyDdFf])?(?!\w))
     |(?P<word>(?:[^\W\d]|_)\w*)
     |(?P<op><=>|->|::|\|\||<<|>>>|>>|<=|>=|<>|!=|==|[-+*/%=<>!~^&|().,\[\]])
     |(?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class _Tok:
    kind: str
    text: str


def _tokenize(text: str) -> list[_Tok]:
    out: list[_Tok] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:  # pragma: no cover - `other` matches any character
            out.append(_Tok("other", text[pos:]))
            break
        kind = match.lastgroup or "other"
        out.append(_Tok(kind, match.group(0)))
        pos = match.end()
    return out


def _unescape(token: str) -> str:
    from .. import predicate as sqlpred

    return sqlpred._unescape(token)


def _string(value: str) -> str:
    """`value` as an ANSI string literal, which both engines read."""
    return "'" + value.replace("'", "''") + "'"


# ------------------------------------------------------------------ AST


@dataclass(slots=True)
class _Node:
    pass


@dataclass(slots=True)
class _Num(_Node):
    digits: str
    suffix: str  # "", L, S, Y, D, F, BD (upper case)


@dataclass(slots=True)
class _Str(_Node):
    value: str


@dataclass(slots=True)
class _Word(_Node):
    """TRUE, FALSE, NULL, or a keyword-like constant such as CURRENT_DATE."""

    text: str


@dataclass(slots=True)
class _Typed(_Node):
    """DATE '...', TIMESTAMP '...' and the like."""

    kind: str
    value: str


@dataclass(slots=True)
class _Ident(_Node):
    #: (name, quoted) per dotted part.
    parts: list[tuple[str, bool]]


@dataclass(slots=True)
class _Field(_Node):
    base: _Node
    name: str
    quoted: bool


@dataclass(slots=True)
class _Subscript(_Node):
    base: _Node
    index: _Node


@dataclass(slots=True)
class _Call(_Node):
    name: str
    args: list[_Node]
    distinct: bool = False


@dataclass(slots=True)
class _Star(_Node):
    pass


@dataclass(slots=True)
class _Unary(_Node):
    op: str  # -, +, ~, NOT
    operand: _Node


@dataclass(slots=True)
class _Binary(_Node):
    op: str
    left: _Node
    right: _Node


@dataclass(slots=True)
class _Case(_Node):
    operand: _Node | None
    whens: list[tuple[_Node, _Node]]
    default: _Node | None


@dataclass(slots=True)
class _Cast(_Node):
    operand: _Node
    type_tokens: list[_Tok]
    try_: bool = False


@dataclass(slots=True)
class _IsNull(_Node):
    operand: _Node
    negated: bool


@dataclass(slots=True)
class _IsBool(_Node):
    operand: _Node
    value: str  # TRUE, FALSE
    negated: bool


@dataclass(slots=True)
class _Distinct(_Node):
    """`a IS [NOT] DISTINCT FROM b`; `<=>` is the NOT form."""

    left: _Node
    right: _Node
    negated: bool  # True for IS NOT DISTINCT FROM


@dataclass(slots=True)
class _In(_Node):
    operand: _Node
    items: list[_Node]
    negated: bool


@dataclass(slots=True)
class _Between(_Node):
    operand: _Node
    low: _Node
    high: _Node
    negated: bool


@dataclass(slots=True)
class _Like(_Node):
    op: str  # LIKE, ILIKE
    operand: _Node
    pattern: _Node
    escape: _Node | None
    negated: bool


@dataclass(slots=True)
class _RLike(_Node):
    operand: _Node
    pattern: _Node
    negated: bool


@dataclass(slots=True)
class _Raw(_Node):
    """Tokens passed through with only their literals respelled."""

    tokens: list[_Tok] = field(default_factory=list)


class _Unparsed(Exception):
    """Text outside the grammar parsed here."""


# ------------------------------------------------------------------ parser

#: Words that end or join an expression, and so cannot be a column name.
_RESERVED = frozenset(
    {
        *("AND", "OR", "NOT", "IS", "IN", "BETWEEN", "LIKE", "ILIKE", "RLIKE", "REGEXP"),
        *("CASE", "WHEN", "THEN", "ELSE", "END", "ESCAPE", "DIV", "FROM", "AS", "SELECT"),
        *("DISTINCT", "EXISTS", "FOR", "WITH", "UNION", "WHERE", "ANY", "ALL", "SOME"),
    }
)
_TYPED = frozenset({"DATE", "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ"})
_INTERVAL_UNITS = frozenset(
    {
        *("YEAR", "YEARS", "MONTH", "MONTHS", "WEEK", "WEEKS", "DAY", "DAYS", "HOUR"),
        *("HOURS", "MINUTE", "MINUTES", "SECOND", "SECONDS", "MILLISECOND"),
        *("MILLISECONDS", "MICROSECOND", "MICROSECONDS", "TO"),
    }
)
#: Binary operators of Spark's valueExpression, by precedence (higher binds
#: tighter); all are left-associative.
_BINARY = {
    "*": 7,
    "/": 7,
    "%": 7,
    "DIV": 7,
    "+": 6,
    "-": 6,
    "||": 6,
    "<<": 5,
    ">>": 5,
    "&": 4,
    "^": 3,
    "|": 2,
    "=": 1,
    "==": 1,
    "<>": 1,
    "!=": 1,
    "<": 1,
    "<=": 1,
    ">": 1,
    ">=": 1,
    "<=>": 1,
}
#: A call whose arguments use keywords (`trim(BOTH 'x' FROM s)`,
#: `position('b' IN s)`) is passed through; the engines read these alike.
_CALL_KEYWORDS = frozenset({"FROM", "FOR", "IN", "BOTH", "LEADING", "TRAILING", "PLACING"})


class _Parser:
    def __init__(self, tokens: list[_Tok]) -> None:
        self.tokens = tokens
        self.i = 0

    # -- cursor --

    def _skip(self, j: int) -> int:
        while j < len(self.tokens) and self.tokens[j].kind == "ws":
            j += 1
        return j

    def peek(self, offset: int = 0) -> _Tok | None:
        j = self._skip(self.i)
        for _ in range(offset):
            j = self._skip(j + 1)
        return self.tokens[j] if j < len(self.tokens) else None

    def take(self) -> _Tok:
        j = self._skip(self.i)
        if j >= len(self.tokens):
            raise _Unparsed("unexpected end")
        self.i = j + 1
        return self.tokens[j]

    def word(self, *words: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok.kind == "word" and tok.text.upper() in words

    def op(self, *ops: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok.kind == "op" and tok.text in ops

    def expect_op(self, sym: str) -> None:
        if not self.op(sym):
            raise _Unparsed(f"expected {sym}")
        self.take()

    def expect_word(self, word: str) -> None:
        if not self.word(word):
            raise _Unparsed(f"expected {word}")
        self.take()

    # -- grammar --

    def parse(self) -> _Node:
        node = self.expr()
        if self.peek() is not None:
            raise _Unparsed(f"unexpected {self.peek()}")
        return node

    def expr(self) -> _Node:
        node = self.and_expr()
        while self.word("OR"):
            self.take()
            node = _Binary("OR", node, self.and_expr())
        return node

    def and_expr(self) -> _Node:
        node = self.not_expr()
        while self.word("AND"):
            self.take()
            node = _Binary("AND", node, self.not_expr())
        return node

    def not_expr(self) -> _Node:
        if self.word("NOT") or self.op("!"):
            self.take()
            return _Unary("NOT", self.not_expr())
        if self.word("EXISTS"):
            raise _Unparsed("EXISTS")
        return self.predicate()

    def predicate(self) -> _Node:
        left = self.value(0)
        negated = False
        if self.word("NOT") and self.word(
            "BETWEEN", "IN", "LIKE", "ILIKE", "RLIKE", "REGEXP", offset=1
        ):
            self.take()
            negated = True
        if self.word("BETWEEN"):
            self.take()
            low = self.value(0)
            self.expect_word("AND")
            return _Between(left, low, self.value(0), negated)
        if self.word("IN"):
            self.take()
            self.expect_op("(")
            if self.word("SELECT", "WITH"):
                raise _Unparsed("subquery")
            items = [self.expr()]
            while self.op(","):
                self.take()
                items.append(self.expr())
            self.expect_op(")")
            return _In(left, items, negated)
        if self.word("LIKE", "ILIKE"):
            kind = self.take().text.upper()
            if self.word("ANY", "ALL", "SOME"):
                raise _Unparsed("LIKE ANY")
            pattern = self.value(0)
            escape = None
            if self.word("ESCAPE"):
                self.take()
                escape = self.value(0)
            return _Like(kind, left, pattern, escape, negated)
        if self.word("RLIKE", "REGEXP"):
            self.take()
            return _RLike(left, self.value(0), negated)
        if negated:
            raise _Unparsed("NOT")
        if self.word("IS"):
            self.take()
            neg = False
            if self.word("NOT"):
                self.take()
                neg = True
            if self.word("NULL", "UNKNOWN"):
                self.take()
                return _IsNull(left, neg)
            if self.word("TRUE", "FALSE"):
                return _IsBool(left, self.take().text.upper(), neg)
            if self.word("DISTINCT"):
                self.take()
                self.expect_word("FROM")
                return _Distinct(left, self.value(0), neg)
            raise _Unparsed("IS")
        return left

    def value(self, min_prec: int) -> _Node:
        left = self.unary()
        while True:
            tok = self.peek()
            if tok is None:
                return left
            op = tok.text.upper() if tok.kind == "word" else tok.text if tok.kind == "op" else ""
            prec = _BINARY.get(op)
            if prec is None or prec < min_prec:
                return left
            self.take()
            right = self.value(prec + 1)
            left = _Distinct(left, right, True) if op == "<=>" else _Binary(op, left, right)

    def unary(self) -> _Node:
        if self.op("-", "+", "~"):
            op = self.take().text
            return _Unary(op, self.unary())
        return self.postfix(self.primary())

    def postfix(self, node: _Node) -> _Node:
        while True:
            if self.op("."):
                self.take()
                tok = self.take()
                if tok.kind not in ("word", "bq"):
                    raise _Unparsed("field")
                name, quoted = _ident_part(tok)
                if isinstance(node, _Ident):
                    node = _Ident([*node.parts, (name, quoted)])
                else:
                    node = _Field(node, name, quoted)
            elif self.op("["):
                self.take()
                index = self.expr()
                self.expect_op("]")
                node = _Subscript(node, index)
            elif self.op("::"):
                self.take()
                node = _Cast(node, self.type_tokens())
            else:
                return node

    def type_tokens(self) -> list[_Tok]:
        """A type's tokens: a name with optional `(...)` / `<...>` parameters."""
        tok = self.take()
        if tok.kind != "word":
            raise _Unparsed("type")
        out = [tok]
        if not self.op("(", "<"):
            return out
        depth = 0
        while True:
            nxt = self.take()
            out.append(nxt)
            if nxt.kind == "op" and nxt.text in ("(", "<"):
                depth += 1
            elif nxt.kind == "op" and nxt.text in (")", ">"):
                depth -= 1
            elif nxt.kind == "op" and nxt.text in (">>", ">>>"):
                depth -= len(nxt.text)
            if depth <= 0:
                return out

    def primary(self) -> _Node:
        tok = self.take()
        kind = tok.kind
        if kind == "num":
            return _Num(*_split_number(tok.text))
        if kind in ("str", "dstr"):
            # Both quote styles are string literals in Spark, and adjacent
            # literals concatenate: `'it''s'` is `its`.
            value = _unescape(tok.text)
            while (nxt := self.peek()) is not None and nxt.kind in ("str", "dstr"):
                value += _unescape(self.take().text)
            return _Str(value)
        if kind == "bq":
            return _Ident([_ident_part(tok)])
        if kind == "op" and tok.text == "(":
            if self.word("SELECT", "WITH"):
                raise _Unparsed("subquery")
            inner = self.expr()
            if self.op(","):
                raise _Unparsed("row constructor")
            self.expect_op(")")
            return inner
        if kind == "op" and tok.text == "*":
            return _Star()
        if kind != "word":
            raise _Unparsed(f"unexpected {tok.text!r}")
        upper = tok.text.upper()
        nxt = self.peek()
        if upper in ("TRUE", "FALSE", "NULL"):
            return _Word(upper)
        if upper in _TYPED and nxt is not None and nxt.kind == "str":
            return _Typed(upper, _unescape(self.take().text))
        if upper == "X" and nxt is not None and nxt.kind == "str" and self._adjacent():
            return _Raw([tok, self.take()])
        if upper == "INTERVAL":
            return self.interval(tok)
        if upper == "CASE":
            return self.case()
        if upper in ("CAST", "TRY_CAST") and self.op("("):
            self.take()
            operand = self.expr()
            self.expect_word("AS")
            type_tokens = self.type_tokens()
            self.expect_op(")")
            return _Cast(operand, type_tokens, upper == "TRY_CAST")
        if self.op("("):
            return self.call(tok)
        if upper in _RESERVED:
            raise _Unparsed(f"keyword {upper}")
        return _Ident([_ident_part(tok)])

    def _adjacent(self) -> bool:
        return self.i < len(self.tokens) and self.tokens[self.i].kind == "str"

    def interval(self, first: _Tok) -> _Node:
        tokens = [first]
        start = self.i
        while True:
            nxt = self.peek()
            if nxt is None:
                break
            unit = nxt.kind == "word" and nxt.text.upper() in _INTERVAL_UNITS
            if not (unit or nxt.kind in ("str", "num") or (nxt.kind == "op" and nxt.text == "-")):
                break
            self.take()
        tokens += self.tokens[start : self.i]
        return _Raw(tokens)

    def case(self) -> _Node:
        operand = None if self.word("WHEN") else self.expr()
        whens: list[tuple[_Node, _Node]] = []
        while self.word("WHEN"):
            self.take()
            cond = self.expr()
            self.expect_word("THEN")
            whens.append((cond, self.expr()))
        if not whens:
            raise _Unparsed("CASE")
        default = None
        if self.word("ELSE"):
            self.take()
            default = self.expr()
        self.expect_word("END")
        return _Case(operand, whens, default)

    def call(self, name: _Tok) -> _Node:
        open_at = self._skip(self.i)
        close = _matching(self.tokens, open_at)
        inner = [t for t in self.tokens[open_at + 1 : close] if t.kind != "ws"]
        depth = 0
        special = False
        for t in inner:
            if t.kind == "op" and t.text in ("(", "["):
                depth += 1
            elif t.kind == "op" and t.text in (")", "]"):
                depth -= 1
            elif depth == 0 and (
                (t.kind == "word" and t.text.upper() in _CALL_KEYWORDS)
                or (t.kind == "op" and t.text == "->")
            ):
                special = True
        if special or close >= len(self.tokens):
            # Keyword arguments or a lambda: pass the call through as written.
            raw = [name, *self.tokens[self.i : close + 1]]
            self.i = close + 1
            return _Raw(raw)
        self.take()  # (
        distinct = False
        if self.word("DISTINCT"):
            self.take()
            distinct = True
        args: list[_Node] = []
        if not self.op(")"):
            args.append(self.expr())
            while self.op(","):
                self.take()
                args.append(self.expr())
        self.expect_op(")")
        return _Call(name.text, args, distinct)


def _matching(tokens: list[_Tok], open_at: int) -> int:
    """Index of the `)` closing the `(` at `open_at` (past the end if none)."""
    depth = 0
    for j in range(open_at, len(tokens)):
        t = tokens[j]
        if t.kind == "op" and t.text == "(":
            depth += 1
        elif t.kind == "op" and t.text == ")":
            depth -= 1
            if depth == 0:
                return j
    return len(tokens)


def _split_number(text: str) -> tuple[str, str]:
    """A numeric token's digits and its upper-cased type suffix ('' if none)."""
    match = re.fullmatch(r"(.*?)(BD|[LSYDF])?", text, re.IGNORECASE)
    assert match is not None
    return match.group(1), (match.group(2) or "").upper()


def _ident_part(tok: _Tok) -> tuple[str, bool]:
    if tok.kind == "bq":
        return tok.text[1:-1].replace("``", "`"), True
    return tok.text, False


def _parse(text: str) -> _Node:
    try:
        return _Parser(_tokenize(text)).parse()
    except RecursionError:
        raise _Unparsed("too deep") from None


def parses(text: str) -> bool:
    """Whether `text` is exactly one expression of the grammar parsed here."""
    try:
        _parse(text)
    except _Unparsed:
        return False
    return True


#: Words Spark allows right after a complete operand (`s COLLATE UTF8_LCASE`,
#: a window or aggregate suffix); any other token there starts a second
#: expression.
_POSTFIX_WORDS = frozenset({"COLLATE", "OVER", "FILTER", "WITHIN", "IGNORE", "RESPECT", "NULLS"})
_CLOSING = {")": "(", "]": "["}


def check_expression(text: Any, what: str = "the SQL expression") -> None:
    """Refuse `text` unless it is structurally one Spark SQL expression.

    Every SQL fragment the API forwards to an engine -- a predicate, a SET or
    INSERT value, a MERGE ON or clause condition, a replaceWhere, a CHECK
    constraint, a generation expression -- goes through this before routing.
    DataFusion parses a prefix and ignores the rest, so ``id = 1) AND (...``
    or ``id = 1 extra`` acted on every ``id = 1`` row; DuckDB and the
    warehouse splice the text into a statement, where a stray ``)``, a ``;``
    or a comment would change the statement around it. So: literals and
    quoted names are closed, parentheses and brackets balance and never close
    early, there is no ``;``, no comment and no top-level ``,``, and when a
    complete expression parses from the start nothing but a postfix keyword
    follows it. Raises `PredicateError` (malformed in any SQL: no engine
    serves it). Legitimate Spark SQL beyond this module's grammar (a
    subquery, a lambda, a variant path) passes the structural checks.
    """
    from ..predicate import PredicateError

    if not isinstance(text, str):
        return

    def refuse(why: str) -> PredicateError:
        return PredicateError(f"{what} {text!r} is not a single SQL expression: {why}")

    tokens = _tokenize(text)
    stack: list[str] = []
    for i, t in enumerate(tokens):
        if t.kind == "other":
            if t.text == ";":
                raise refuse("';' outside a string literal ends a statement")
            if t.text in ("'", '"', "`"):
                raise refuse(f"an unterminated {t.text} literal or name")
            continue
        if t.kind != "op":
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and nxt.kind == "op" and (t.text, nxt.text) in (("-", "-"), ("/", "*")):
            raise refuse("it holds a SQL comment")
        if t.text in ("(", "["):
            stack.append(t.text)
        elif t.text in _CLOSING:
            if not stack or stack.pop() != _CLOSING[t.text]:
                raise refuse(f"an unbalanced {t.text!r}")
        elif t.text == "," and not stack:
            raise refuse("a ',' outside parentheses separates two expressions")
    if stack:
        raise refuse(f"an unclosed {stack[-1]!r}")
    if not text.strip():
        return
    try:
        _Parser(tokens).parse()
        return
    except (_Unparsed, RecursionError):
        pass
    parser = _Parser(tokens)
    try:
        parser.expr()
    except (_Unparsed, RecursionError):
        return  # beyond the grammar here; the structure above is sound
    tail = parser.peek()
    if tail is None:
        return
    if tail.kind == "word" and tail.text.upper() in _POSTFIX_WORDS:
        return
    if tail.kind in ("word", "num", "str", "dstr", "bq"):
        raise refuse(f"{tail.text!r} follows a complete expression")


# ------------------------------------------------------------ warehouse-only

#: Functions whose meaning differs from both engines' in a way no rewrite
#: here fixes: Spark's `split` splits on a regex, `hash` is Murmur3, a format
#: pattern is Java's. Each maps to why.
_WAREHOUSE_FUNCTIONS = {
    "split": "Spark's split() splits on a regular expression",
    "hash": "Spark's hash() is Murmur3, which neither engine computes",
    "xxhash64": "neither engine computes Spark's xxhash64()",
    "typeof": "type names differ between the engines",
    "decode": "Spark's decode() is a CASE-like lookup or a charset decode",
    "get": "get() indexes arrays from 0",
    "format_string": "format patterns are Java's",
    "printf": "format patterns are Java's",
    "format_number": "format patterns are Java's",
    "date_format": "date format patterns are Java's",
}
_SUBSCRIPT = (
    "an array subscript counts from 0 in Spark and from 1 in DuckDB and DataFusion "
    "(use element_at, which counts from 1 everywhere)"
)
#: Functions that take a Java format pattern as a later argument.
_FORMATTED = frozenset(
    {"to_date", "to_timestamp", "from_unixtime", "unix_timestamp", "to_unix_timestamp"}
)


def _walk(node: _Node) -> Any:
    yield node
    for f in dataclasses.fields(node):
        value = getattr(node, f.name)
        items = value if isinstance(value, list) else [value]
        for item in items:
            for part in item if isinstance(item, tuple) else (item,):
                if isinstance(part, _Node):
                    yield from _walk(part)


def _node_reason(node: _Node) -> str | None:
    for part in _walk(node):
        if isinstance(part, _Subscript) and not isinstance(part.index, _Str):
            return _SUBSCRIPT
        if isinstance(part, _Raw):
            reason = _raw_reason(part.tokens)
            if reason is not None:
                return reason
        if isinstance(part, _Call):
            name = part.name.lower()
            if name in _WAREHOUSE_FUNCTIONS:
                return f"{_WAREHOUSE_FUNCTIONS[name]}"
            if name in _FORMATTED and len(part.args) >= 2:
                return f"{name}() takes a Java date format pattern"
            if name == "regexp_replace" and not _plain_replacement(part.args):
                return (
                    "regexp_replace() refers to groups as $1 in Spark and differently in "
                    "DuckDB and DataFusion"
                )
        if isinstance(part, _Binary) and part.op == ">>>":
            return "neither engine has Spark's >>> (unsigned shift)"
    return None


def _plain_replacement(args: list[_Node]) -> bool:
    return (
        len(args) == 3
        and isinstance(args[2], _Str)
        and "$" not in args[2].value
        and "\\" not in args[2].value
    )


def _raw_reason(tokens: list[_Tok]) -> str | None:
    """Why text outside the grammar here must not pass through as written."""
    solid = [t for t in tokens if t.kind != "ws"]
    for i, t in enumerate(solid):
        word = t.text.upper() if t.kind == "word" else ""
        if word in ("DIV", "RLIKE", "REGEXP"):
            return f"{word} appears in SQL this library does not parse, so it cannot be translated"
        called = i + 1 < len(solid) and solid[i + 1].text == "("
        if called and word.lower() in _WAREHOUSE_FUNCTIONS:
            return _WAREHOUSE_FUNCTIONS[word.lower()]
        if t.kind == "op" and t.text == "[":
            # Spark has no `[...]` literal, so this is a subscript.
            key = i + 2 < len(solid) and solid[i + 1].kind == "str" and solid[i + 2].text == "]"
            if not key:
                return _SUBSCRIPT
    return None


def warehouse_reason(text: Any) -> str | None:
    """Why `text` (Spark SQL) must be evaluated by Databricks, or None.

    For SQL whose Spark meaning neither direct engine can be made to compute:
    routing then sends the call to the warehouse, or refuses it naming this.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        node = _parse(text)
    except _Unparsed:
        return _raw_reason(_tokenize(text))
    return _node_reason(node)


# ------------------------------------------------------------------ render

#: Integral type names as each engine spells them, from Spark's.
_INTEGRAL = {
    "TINYINT": "TINYINT",
    "BYTE": "TINYINT",
    "SMALLINT": "SMALLINT",
    "SHORT": "SMALLINT",
    "INT": "INTEGER",
    "INTEGER": "INTEGER",
    "BIGINT": "BIGINT",
    "LONG": "BIGINT",
}
_SIMPLE_TYPES = {
    "STRING": "VARCHAR",
    "DOUBLE": "DOUBLE",
    "FLOAT": "FLOAT",
    "REAL": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "DATE": "DATE",
    "TIMESTAMP_NTZ": "TIMESTAMP",
}

_TYPE_KINDS = {
    **dict.fromkeys(_INTEGRAL, "int"),
    "DOUBLE": "float",
    "FLOAT": "float",
    "REAL": "float",
    "DECIMAL": "decimal",
    "DEC": "decimal",
    "NUMERIC": "decimal",
}


class _Renderer:
    def __init__(self, target: str, kinds: Mapping[str, str] | None) -> None:
        self.target = target
        self.duck = target == "duckdb"
        self.kinds = kinds or {}

    # -- helpers --

    def ident(self, name: str, quoted: bool) -> str:
        if not quoted:
            return name
        if self.duck:
            return '"' + name.replace('"', '""') + '"'
        return "`" + name.replace("`", "``") + "`"

    def raw(self, tokens: list[_Tok]) -> str:
        return _respell(tokens, self.target, check=False)

    def kind(self, node: _Node) -> str | None:
        """The numeric kind of `node`: int, decimal, float, or None if unknown."""
        if isinstance(node, _Num):
            if node.suffix in ("L", "S", "Y"):
                return "int"
            if node.suffix in ("D", "F") or "e" in node.digits.lower():
                return "float"
            if node.suffix == "BD" or "." in node.digits:
                return "decimal"
            return "int"
        if isinstance(node, _Ident):
            return self.kinds.get(node.parts[-1][0].lower())
        if isinstance(node, _Unary) and node.op in ("-", "+"):
            return self.kind(node.operand)
        if isinstance(node, _Cast):
            base = node.type_tokens[0].text.upper()
            return _TYPE_KINDS.get(base)
        if isinstance(node, _Binary) and node.op in ("+", "-", "*", "%"):
            kinds = {self.kind(node.left), self.kind(node.right)}
            for k in ("float", "decimal"):
                if k in kinds:
                    return k
            return "int" if kinds == {"int"} else None
        if isinstance(node, _Binary) and node.op == "/":
            kinds = {self.kind(node.left), self.kind(node.right)}
            return "decimal" if "decimal" in kinds and "float" not in kinds else "float"
        if isinstance(node, _Binary) and node.op == "DIV":
            return "int"
        return None

    # -- nodes --

    def render(self, node: _Node) -> str:
        method: Callable[[Any], str] = getattr(self, "r_" + type(node).__name__[1:].lower())
        return method(node)

    def r_raw(self, node: _Raw) -> str:
        return self.raw(node.tokens)

    def r_num(self, node: _Num) -> str:
        if self.duck and node.suffix in ("", "L") and node.digits.isdigit():
            # Spark types an integer literal INT (BIGINT past INT, or with L).
            # DuckDB narrows one to the other side's type, so for a TINYINT
            # `b` = 127, `b + 1 - 1` overflowed TINYINT where Spark gives 127.
            value = int(node.digits)
            if node.suffix == "" and value <= 2**31 - 1:
                return f"CAST({node.digits} AS INTEGER)"
            if value <= 2**63 - 1:
                return f"CAST({node.digits} AS BIGINT)"
        return _number(node.digits, node.suffix, self.target)

    def r_str(self, node: _Str) -> str:
        return _string(node.value)

    def r_word(self, node: _Word) -> str:
        return node.text

    def r_typed(self, node: _Typed) -> str:
        kind = "TIMESTAMP" if node.kind in ("TIMESTAMP_NTZ", "TIMESTAMP_LTZ") else node.kind
        return f"{kind} {_string(node.value)}"

    def r_ident(self, node: _Ident) -> str:
        return ".".join(self.ident(n, q) for n, q in node.parts)

    def r_field(self, node: _Field) -> str:
        return f"{self.render(node.base)}.{self.ident(node.name, node.quoted)}"

    def r_subscript(self, node: _Subscript) -> str:
        return f"{self.render(node.base)}[{self.render(node.index)}]"

    def r_star(self, node: _Star) -> str:
        return "*"

    def r_unary(self, node: _Unary) -> str:
        operand = self.render(node.operand)
        if node.op == "NOT":
            return f"(NOT {operand})"
        return f"({node.op}{operand})"

    def r_binary(self, node: _Binary) -> str:
        op = node.op
        left, right = self.render(node.left), self.render(node.right)
        if op == "==":
            op = "="
        if op == "/":
            return self.divide(node, left, right)
        if op == "DIV":
            # Spark's integral division truncates toward zero.
            if self.duck:
                return f"__spark_div({left}, {right})"
            return f"arrow_cast(({left} / {right}), 'Int64')"
        if op == "%":
            return self.mod(left, right)
        if op == "^" and self.duck:
            return f"xor({left}, {right})"  # DuckDB's ^ is a power
        return f"({left} {op} {right})"

    def divide(self, node: _Binary, left: str, right: str) -> str:
        """Spark's `/`: a DOUBLE (or DECIMAL) quotient, an error when dividing by zero."""
        if self.duck:
            # DuckDB divides integers into a DOUBLE already; by zero it gives
            # inf or NULL where Spark raises.
            return f"__spark_divide({left}, {right})"
        kinds = {self.kind(node.left), self.kind(node.right)}
        if kinds & {"decimal", "float"}:
            return f"({left} / {right})"
        # DataFusion divides two integers into an integer: 5 / 2 was 2.
        quotient = f"arrow_cast({left}, 'Float64') / {right}"
        if self.kind(node.right) == "int":
            # Integer division by zero raises in DataFusion, as Spark does;
            # a DOUBLE quotient would be inf instead. `0 / right` keeps it.
            return f"({quotient} + (0 / {right}))"
        return f"({quotient})"

    def r_case(self, node: _Case) -> str:
        out = "CASE"
        if node.operand is not None:
            out += " " + self.render(node.operand)
        for cond, value in node.whens:
            out += f" WHEN {self.render(cond)} THEN {self.render(value)}"
        if node.default is not None:
            out += f" ELSE {self.render(node.default)}"
        return out + " END"

    def r_cast(self, node: _Cast) -> str:
        operand = self.render(node.operand)
        verb = "TRY_CAST" if node.try_ else "CAST"
        base = node.type_tokens[0].text.upper()
        if base in _INTEGRAL:
            type_text = _INTEGRAL[base]
            if self.duck:
                # DuckDB rounds a fraction cast to an integer (1.9 -> 2) and
                # reads '1.9' as one; Spark truncates, and refuses the text.
                helper = "__spark_try_integral" if node.try_ else "__spark_integral"
                operand = f"{helper}({operand})"
            return f"{verb}({operand} AS {type_text})"
        if len(node.type_tokens) == 1 and base in _SIMPLE_TYPES:
            return f"{verb}({operand} AS {_SIMPLE_TYPES[base]})"
        if base in ("DECIMAL", "DEC", "NUMERIC"):
            params = self.raw(node.type_tokens[1:]) if len(node.type_tokens) > 1 else "(10, 0)"
            # Spark's bare DECIMAL is DECIMAL(10,0); DuckDB's is (18,3).
            return f"{verb}({operand} AS DECIMAL{params})"
        if base == "BINARY" and len(node.type_tokens) == 1:
            return f"{verb}({operand} AS {'BLOB' if self.duck else 'BYTEA'})"
        return f"{verb}({operand} AS {self.raw(node.type_tokens)})"

    def r_isnull(self, node: _IsNull) -> str:
        return f"({self.render(node.operand)} IS {'NOT ' if node.negated else ''}NULL)"

    def r_isbool(self, node: _IsBool) -> str:
        return f"({self.render(node.operand)} IS {'NOT ' if node.negated else ''}{node.value})"

    def r_distinct(self, node: _Distinct) -> str:
        left, right = self.render(node.left), self.render(node.right)
        if self.duck:
            # A macro, so the text has no FROM for the sharing filter's screen.
            same = f"__spark_not_distinct({left}, {right})"
            return same if node.negated else f"(NOT {same})"
        return f"({left} IS {'NOT ' if node.negated else ''}DISTINCT FROM {right})"

    def r_in(self, node: _In) -> str:
        values = [node.operand, *node.items]
        rendered = [self.render(v) for v in values]
        kinds = [self.kind(v) for v in values]
        if not self.duck and "float" in kinds and any(k != "float" for k in kinds):
            # Spark compares the list in its common type: one DOUBLE makes it
            # all DOUBLE. DataFusion's file pruning refused BIGINT IN (.., 7D)
            # with "Invalid comparison operation".
            rendered = [
                r if k == "float" else f"arrow_cast({r}, 'Float64')"
                for r, k in zip(rendered, kinds, strict=True)
            ]
        items = ", ".join(rendered[1:])
        return f"({rendered[0]} {'NOT ' if node.negated else ''}IN ({items}))"

    def r_between(self, node: _Between) -> str:
        return (
            f"({self.render(node.operand)} {'NOT ' if node.negated else ''}BETWEEN "
            f"{self.render(node.low)} AND {self.render(node.high)})"
        )

    def r_like(self, node: _Like) -> str:
        from .. import predicate as sqlpred

        operand = self.render(node.operand)
        verb = ("NOT " if node.negated else "") + node.op
        pattern = self.render(node.pattern)
        if node.escape is None:
            # Spark's LIKE escapes with a backslash by default; DuckDB's has no
            # escape character unless one is named. DataFusion's is a backslash.
            escape = " ESCAPE '\\'" if self.duck else ""
            return f"({operand} {verb} {pattern}{escape})"
        if self.duck:
            return f"({operand} {verb} {pattern} ESCAPE {self.render(node.escape)})"
        if (
            isinstance(node.pattern, _Str)
            and isinstance(node.escape, _Str)
            and len(node.escape.value) == 1
        ):
            # DataFusion accepts only the backslash as an escape character.
            respelled = sqlpred._backslash_escaped(node.pattern.value, node.escape.value)
            return f"({operand} {verb} {_string(respelled)})"
        raise _Unparsed("LIKE ... ESCAPE with a non-literal pattern")

    def r_rlike(self, node: _RLike) -> str:
        operand, pattern = self.render(node.operand), self.render(node.pattern)
        if self.duck:
            found = f"regexp_matches({operand}, {pattern})"
            return f"(NOT {found})" if node.negated else found
        return f"({operand} {'!~' if node.negated else '~'} {pattern})"

    def r_call(self, node: _Call) -> str:
        name = node.name.lower()
        args = [self.render(a) for a in node.args]
        rewrite = _CALLS.get(name)
        if rewrite is not None and not node.distinct:
            out = rewrite(self, node, args)
            if out is not None:
                return out
        inner = ("DISTINCT " if node.distinct else "") + ", ".join(args)
        return f"{node.name}({inner})"

    # -- functions --

    def mod(self, left: str, right: str) -> str:
        # DuckDB's remainder by zero is NULL, where Spark raises.
        return f"__spark_mod({left}, {right})" if self.duck else f"({left} % {right})"

    def length(self, text: str) -> str:
        return f"length({text})" if self.duck else f"character_length({text})"

    def substring(self, s: str, p_node: _Node, p: str, n_node: _Node | None, n: str | None) -> str:
        """Spark's substring(s, p[, n]).

        A position of 0 counts as 1, a negative one from the end, and a length
        that ends before the start gives ''. DuckDB and DataFusion read 0 as
        the position before the first character (so `substring('abc', 0, 2)`
        was `a`), DataFusion returned '' for a negative position, and DuckDB
        counts a negative length backwards.
        """
        p_lit = _int_literal(p_node)
        n_lit = _int_literal(n_node) if n_node is not None else None
        if (
            p_lit is not None
            and p_lit >= 0
            and (n_node is None or (n_lit is not None and n_lit >= 0))
        ):
            first = max(p_lit, 1)
            return f"substr({s}, {first})" if n is None else f"substr({s}, {first}, {n})"
        # 0-based start, as Spark computes it; then clamped to the string.
        s0 = (
            f"(CASE WHEN {p} > 0 THEN {p} - 1 WHEN {p} < 0 THEN {self.length(s)} + {p} "
            f"WHEN {p} = 0 THEN 0 END)"
        )
        start = f"(CASE WHEN {s0} > 0 THEN {s0} WHEN {s0} <= 0 THEN 0 END)"
        if n is None:
            return f"substr({s}, {start} + 1)"
        end = f"({s0} + {n})"
        return (
            f"(CASE WHEN {end} <= {start} THEN '' "
            f"ELSE substr({s}, {start} + 1, {end} - {start}) END)"
        )


def _int_literal(node: _Node | None) -> int | None:
    if isinstance(node, _Num) and node.suffix in ("", "L", "S", "Y") and node.digits.isdigit():
        return int(node.digits)
    return None


def _number(digits: str, suffix: str, target: str) -> str:
    duck = target == "duckdb"
    if suffix == "D":
        return f"CAST({digits} AS DOUBLE)" if duck else f"arrow_cast('{digits}', 'Float64')"
    if suffix == "F":
        return f"CAST({digits} AS FLOAT)" if duck else f"arrow_cast('{digits}', 'Float32')"
    if suffix == "BD":
        if duck:
            return digits  # DuckDB reads a fraction as a DECIMAL already
        whole, _, frac = digits.partition(".")
        scale = len(frac)
        precision = max(len((whole + frac).lstrip("0")) or 1, scale, 1)
        if precision > 38:
            return digits
        return f"arrow_cast('{digits}', 'Decimal128({precision}, {scale})')"
    if suffix == "S":
        return f"CAST({digits} AS SMALLINT)" if duck else f"arrow_cast({digits}, 'Int16')"
    if suffix == "Y":
        return f"CAST({digits} AS TINYINT)" if duck else f"arrow_cast({digits}, 'Int8')"
    return digits  # L: both read an integer literal as a BIGINT where it needs one


def _concat(r: _Renderer, node: _Call, args: list[str]) -> str:
    # Spark's concat is NULL when any argument is; both engines' skip NULLs,
    # but `||` propagates them in each.
    if not args:
        return "''"
    return "(" + " || ".join(f"({a})" for a in args) + ")"


def _coalesce(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    return f"coalesce({', '.join(args)})" if len(args) == 2 else None


def _nvl2(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    if len(args) != 3:
        return None
    return f"(CASE WHEN ({args[0]}) IS NOT NULL THEN ({args[1]}) ELSE ({args[2]}) END)"


def _if(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    if len(args) != 3:
        return None
    return f"(CASE WHEN {args[0]} THEN {args[1]} ELSE {args[2]} END)"


def _substring(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    if len(args) == 2:
        return r.substring(args[0], node.args[1], args[1], None, None)
    if len(args) == 3:
        return r.substring(args[0], node.args[1], args[1], node.args[2], args[2])
    return None


def _left(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    # Spark: substring(s, 1, n); both engines return all but -n characters
    # for a negative n.
    if len(args) != 2:
        return None
    return r.substring(args[0], _Num("1", ""), "1", node.args[1], args[1])


def _right(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    if len(args) != 2:
        return None
    s, n = args
    tail = r.substring(s, _Unary("-", node.args[1]), f"(-{n})", None, None)
    return f"(CASE WHEN {s} IS NOT NULL AND {n} <= 0 THEN '' ELSE {tail} END)"


def _log(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    # One argument: the natural log in Spark, base 10 in both engines.
    return f"ln({args[0]})" if len(args) == 1 else None


def _trim(name: str) -> Callable[[_Renderer, _Call, list[str]], str | None]:
    def rewrite(r: _Renderer, node: _Call, args: list[str]) -> str | None:
        # Spark's two-argument form takes the characters to trim first.
        if len(args) != 2:
            return None
        return f"{name}({args[1]}, {args[0]})"

    return rewrite


def _btrim(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    if not r.duck:
        return None
    return f"trim({', '.join(args)})"


def _rename(name: str) -> Callable[[_Renderer, _Call, list[str]], str]:
    def rewrite(r: _Renderer, node: _Call, args: list[str]) -> str:
        return f"{name}({', '.join(args)})"

    return rewrite


def _mod(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    if len(args) != 2:
        return None
    return r.mod(args[0], args[1])


def _pmod(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    # Spark's positive modulus: the remainder, moved into [0, b) when negative.
    if len(args) != 2:
        return None
    a, b = args
    rem = r.mod(a, b)
    return f"(CASE WHEN {rem} < 0 THEN {r.mod(f'({rem} + {b})', b)} ELSE {rem} END)"


def _is_null(negated: bool) -> Callable[[_Renderer, _Call, list[str]], str | None]:
    def rewrite(r: _Renderer, node: _Call, args: list[str]) -> str | None:
        if len(args) != 1:
            return None
        return f"({args[0]} IS {'NOT ' if negated else ''}NULL)"

    return rewrite


def _locate(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    # locate(substr, str) / position(substr, str): 1-based, 0 when absent.
    if len(args) != 2:
        return None
    return f"strpos({args[1]}, {args[0]})"


def _array(r: _Renderer, node: _Call, args: list[str]) -> str:
    if r.duck:
        return "[" + ", ".join(args) + "]"
    return f"make_array({', '.join(args)})"


def _regexp_replace(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    # Spark replaces every match; both engines replace the first unless 'g'.
    if not _plain_replacement(node.args):
        return None
    return f"regexp_replace({args[0]}, {args[1]}, {args[2]}, 'g')"


def _regexp_like(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    if len(args) != 2:
        return None
    return r.render(_RLike(node.args[0], node.args[1], False))


def _parse_json(r: _Renderer, node: _Call, args: list[str]) -> str | None:
    # A literal's VARIANT encoding, as DuckDB has no parse_json. DataFusion's
    # struct literal cannot take a VARIANT column's type on its own; delta-rs
    # UPDATE spells that case out itself (`_variant.datafusion_literal`).
    text = parse_json_literal(node)
    if text is None or not r.duck:
        return None
    from .._variant import encode

    try:
        metadata, value = encode(text)
    except ValueError:
        return None  # DuckDB then reports the function missing
    return f"{{'metadata': unhex('{metadata.hex()}'), 'value': unhex('{value.hex()}')}}"


def parse_json_literal(node: Any) -> str | None:
    """The JSON text of `parse_json('<literal>')` -- a node, or SQL text -- else None."""
    if isinstance(node, str):
        try:
            node = _parse(node)
        except _Unparsed:
            return None
    if (
        isinstance(node, _Call)
        and node.name.lower() in ("parse_json", "try_parse_json")
        and len(node.args) == 1
        and isinstance(node.args[0], _Str)
        and not node.distinct
    ):
        return node.args[0].value
    return None


_CALLS: dict[str, Callable[[_Renderer, _Call, list[str]], str | None]] = {
    "concat": _concat,
    "nvl": _coalesce,
    "ifnull": _coalesce,
    "nvl2": _nvl2,
    "if": _if,
    "iff": _if,
    "substring": _substring,
    "substr": _substring,
    "left": _left,
    "right": _right,
    "log": _log,
    "trim": _trim("trim"),
    "ltrim": _trim("ltrim"),
    "rtrim": _trim("rtrim"),
    "btrim": _btrim,
    "startswith": _rename("starts_with"),
    "endswith": _rename("ends_with"),
    "lcase": _rename("lower"),
    "ucase": _rename("upper"),
    "len": _rename("length"),
    "mod": _mod,
    "pmod": _pmod,
    "isnull": _is_null(False),
    "isnotnull": _is_null(True),
    "locate": _locate,
    "position": _locate,
    "array": _array,
    "regexp_replace": _regexp_replace,
    "regexp_like": _regexp_like,
    "parse_json": _parse_json,
}


def _respell(tokens: list[_Tok], target: str, *, check: bool) -> str:
    """Tokens as written, with only literals and quoting respelled for `target`.

    For text outside the grammar parsed here. With `check`, a word whose
    translation needs that grammar (DIV, RLIKE) is refused rather than
    passed to an engine that reads it differently or not at all.
    """
    if check:
        reason = _raw_reason(tokens)
        if reason is not None:
            raise _Refused(reason)
    duck = target == "duckdb"
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.kind in ("str", "dstr"):
            value = _unescape(t.text)
            j = i + 1
            while True:
                k = j
                while k < len(tokens) and tokens[k].kind == "ws":
                    k += 1
                if k < len(tokens) and tokens[k].kind in ("str", "dstr"):
                    value += _unescape(tokens[k].text)
                    j = k + 1
                else:
                    break
            out.append(_string(value))
            i = j
            continue
        if t.kind == "bq":
            name = t.text[1:-1].replace("``", "`")
            out.append('"' + name.replace('"', '""') + '"' if duck else t.text)
        elif t.kind == "num":
            out.append(_number(*_split_number(t.text), target))
        elif t.kind == "op" and t.text == "<=>":
            out.append(" IS NOT DISTINCT FROM ")
        elif t.kind == "op" and t.text == "==":
            out.append("=")
        else:
            out.append(t.text)
        i += 1
    return "".join(out)


class _Refused(Exception):
    pass


def _translate(text: str, target: str, kinds: Mapping[str, str] | None) -> str:
    try:
        node = _parse(text)
    except _Unparsed:
        try:
            return _respell(_tokenize(text), target, check=True)
        except _Refused as exc:
            raise _refusal(text, target, str(exc)) from None
    reason = _node_reason(node)
    if reason is not None:
        raise _refusal(text, target, reason)
    try:
        return _Renderer(target, kinds).render(node)
    except _Unparsed:
        # A construct this renderer declines (LIKE with a computed pattern and
        # an ESCAPE, on DataFusion): as written, literals respelled.
        return _respell(_tokenize(text), target, check=False)


def _refusal(text: str, target: str, reason: str) -> EngineLimitError:
    engine = "DuckDB" if target == "duckdb" else "DataFusion (delta-rs)"
    return EngineLimitError(
        f"evaluate {text!r} with {engine}",
        f"{reason}, so {engine} would not compute what Databricks does",
        _REMEDY,
    )


def to_duckdb(text: str) -> str:
    """Spark SQL expression text as DuckDB evaluates it the same way.

    The result may call the helper macros in `DUCKDB_MACROS`; install them on
    the connection first (`install_duckdb_macros`). Raises `EngineLimitError`
    for SQL DuckDB cannot be made to compute as Spark does.
    """
    return _translate(text, "duckdb", None)


def to_datafusion(text: str, kinds: Mapping[str, str] | None = None) -> str:
    """Spark SQL expression text as DataFusion evaluates it the same way.

    `kinds` maps lower-cased column names to their numeric kind (see
    `numeric_kinds`), so `/` knows when both sides are integers. Raises
    `EngineLimitError` for SQL DataFusion cannot be made to compute as Spark
    does.
    """
    return _translate(text, "datafusion", kinds)


def numeric_kinds(*schemas: Any) -> dict[str, str]:
    """Lower-cased column name -> int/decimal/float, from pyarrow schemas.

    A name two schemas type differently is left out (unknown).
    """
    import pyarrow as pa

    out: dict[str, str] = {}
    clashing: set[str] = set()
    for schema in schemas:
        if schema is None:
            continue
        for f in schema:
            t = f.type
            if pa.types.is_integer(t):
                kind = "int"
            elif pa.types.is_decimal(t):
                kind = "decimal"
            elif pa.types.is_floating(t):
                kind = "float"
            else:
                kind = "other"
            name = f.name.lower()
            if out.get(name, kind) != kind:
                clashing.add(name)
            out[name] = kind
    return {k: v for k, v in out.items() if k not in clashing and v != "other"}


#: Helpers the DuckDB rendering calls, for what DuckDB spells differently or
#: not at all. Typed overloads pick the behaviour by argument type.
DUCKDB_MACROS = (
    # CAST(x AS <integral>): truncate a fraction; text must be an integer.
    "CREATE OR REPLACE MACRO __spark_integral(x VARCHAR) AS CASE WHEN x IS NULL THEN NULL "
    "WHEN regexp_full_match(trim(x), '[+-]?[0-9]+') THEN trim(x) ELSE "
    "error('[CAST_INVALID_INPUT] The value ''' || x || ''' of the type \"STRING\" cannot be "
    "cast to an integral type because it is malformed') END, "
    "(x BOOLEAN) AS x, (x) AS trunc(x)",
    "CREATE OR REPLACE MACRO __spark_try_integral(x VARCHAR) AS CASE "
    "WHEN regexp_full_match(trim(x), '[+-]?[0-9]+') THEN trim(x) END, "
    "(x BOOLEAN) AS x, (x) AS trunc(x)",
    "CREATE OR REPLACE MACRO __spark_divide(a, b) AS CASE WHEN b = 0 THEN "
    "error('[DIVIDE_BY_ZERO] Division by zero') ELSE a / b END",
    "CREATE OR REPLACE MACRO __spark_mod(a, b) AS CASE WHEN b = 0 THEN "
    "error('[REMAINDER_BY_ZERO] Remainder by zero') ELSE a % b END",
    "CREATE OR REPLACE MACRO __spark_div(a BIGINT, b BIGINT) AS CASE WHEN b = 0 THEN "
    "error('[DIVIDE_BY_ZERO] Division by zero') ELSE a // b END, "
    "(a, b) AS CASE WHEN b = 0 THEN error('[DIVIDE_BY_ZERO] Division by zero') "
    "ELSE CAST(trunc(a / b) AS BIGINT) END",
    "CREATE OR REPLACE MACRO __spark_not_distinct(a, b) AS a IS NOT DISTINCT FROM b",
)


def install_duckdb_macros(con: Any) -> None:
    """Define `DUCKDB_MACROS` on a DuckDB connection."""
    for statement in DUCKDB_MACROS:
        con.execute(statement)
