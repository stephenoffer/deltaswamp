"""Running SQL on a Databricks warehouse through the Statement Execution API.

The API comes with `databricks-sdk`, which is already a base dependency and
shares the connection's authentication, so no SQL connector is needed.

- Queries run with `format=ARROW_STREAM` and `disposition=EXTERNAL_LINKS`, so a
  large result streams from cloud storage instead of hitting the inline limit.
- Presigned result links are fetched with only the headers the link carries.
  The workspace `Authorization` header never goes to a storage bucket.
- User-supplied values are bound as named `:markers`, never spliced into
  statement text. Identifiers cannot be parameters and are quoted by the engine.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import decimal
import re
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .._util import enum_value
from ..errors import DeltaSwampError

__all__ = [
    "ParameterBinder",
    "SdkStatementBackend",
    "SqlParameter",
    "SqlStatementError",
    "StatementBackend",
    "parameters_from_mapping",
]

#: States from which a statement will not move again.
_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"})

#: `wait_timeout` must be 0 (return immediately) or between 5 and 50 seconds.
_MIN_WAIT, _MAX_WAIT = 5, 50


class SqlStatementError(DeltaSwampError):
    """A statement failed, was cancelled, or timed out on the warehouse.

    Carries the server's own message and codes, because "the query failed" is
    exactly the opaque error this library exists to avoid.
    """

    def __init__(
        self,
        message: str,
        *,
        statement_id: str | None = None,
        error_code: str | None = None,
        sql_state: str | None = None,
        state: str | None = None,
    ) -> None:
        self.statement_id = statement_id
        self.error_code = error_code
        self.sql_state = sql_state
        self.state = state
        detail = ", ".join(
            f"{k}={v}"
            for k, v in (
                ("state", state),
                ("error_code", error_code),
                ("sql_state", sql_state),
                ("statement_id", statement_id),
            )
            if v
        )
        super().__init__(f"{message} ({detail})" if detail else message)


class _DownloadError(SqlStatementError):
    """Fetching a presigned result link failed (expired, network, 403)."""


@dataclass(frozen=True, slots=True)
class SqlParameter:
    """One named parameter marker. `value=None` binds SQL NULL."""

    name: str
    value: str | None
    type: str = "STRING"


_BIGINT_MIN, _BIGINT_MAX = -(2**63), 2**63 - 1
_MAX_DECIMAL_PRECISION = 38


def _decimal_parameter(name: str, value: decimal.Decimal) -> SqlParameter:
    """A Decimal as DECIMAL(p,s), with p and s describing the value exactly.

    `str(Decimal("1E+5"))` is ``"1E+5"`` with one digit, which would bind as
    DECIMAL(1,0) and overflow; the value is rendered positionally instead, and
    a value no DECIMAL(38,s) can hold is refused rather than rounded.
    """
    if not value.is_finite():
        raise ValueError(f"cannot bind {value} as a SQL DECIMAL; it is not a finite number")
    _sign, digits, exponent = value.as_tuple()
    exp = int(exponent)
    ndigits = len(digits) + max(exp, 0)
    scale = max(0, -exp)
    precision = max(ndigits, scale, 1)
    if precision > _MAX_DECIMAL_PRECISION:
        raise ValueError(
            f"cannot bind {value} as a SQL DECIMAL: it needs {precision} digits of precision "
            f"and Databricks allows at most {_MAX_DECIMAL_PRECISION}"
        )
    return SqlParameter(name, format(value, "f"), f"DECIMAL({precision},{scale})")


def _float_text(value: float) -> str:
    # Spark's string-to-double cast spells the specials this way.
    if value != value:
        return "NaN"
    if value in (float("inf"), float("-inf")):
        return "Infinity" if value > 0 else "-Infinity"
    return repr(value)


def _is_pandas_missing(value: Any) -> bool:
    module = type(value).__module__ or ""
    return module.startswith("pandas") and type(value).__name__ in ("NaTType", "NAType")


def _parameter(name: str, value: Any) -> SqlParameter:
    """Render a Python value as a typed statement parameter."""
    if value is None or _is_pandas_missing(value):
        # pd.NaT is a datetime whose isoformat() is 'NaT', and pd.NA is no
        # Python type at all; both mean SQL NULL.
        return SqlParameter(name, None, "STRING")
    # A NumPy scalar (np.int64, np.bool_, np.float32 ...) is not a Python int
    # or bool; unwrap it to the Python value it stands for.
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        if type(value).__name__ == "datetime64":
            # A nanosecond datetime64 `.item()`s to a bare int; go through micros.
            value = value.astype("datetime64[us]")
        value = value.item()
        if value is None:
            return SqlParameter(name, None, "STRING")
    if isinstance(value, bool):  # before int: bool is an int subclass
        return SqlParameter(name, "true" if value else "false", "BOOLEAN")
    if isinstance(value, int):
        if _BIGINT_MIN <= value <= _BIGINT_MAX:
            return SqlParameter(name, str(value), "BIGINT")
        return _decimal_parameter(name, decimal.Decimal(value))
    if isinstance(value, float):
        return SqlParameter(name, _float_text(value), "DOUBLE")
    if isinstance(value, decimal.Decimal):
        return _decimal_parameter(name, value)
    if isinstance(value, _dt.datetime):
        kind = "TIMESTAMP" if value.tzinfo is not None else "TIMESTAMP_NTZ"
        return SqlParameter(name, value.isoformat(), kind)
    if isinstance(value, _dt.date):
        return SqlParameter(name, value.isoformat(), "DATE")
    if isinstance(value, str):
        return SqlParameter(name, value, "STRING")
    raise TypeError(
        f"cannot bind a {type(value).__name__} as a SQL parameter; pass str, int, float, "
        "bool, Decimal, date, datetime or None"
    )


def parameters_from_mapping(values: Mapping[str, Any] | None) -> list[SqlParameter]:
    """Build parameters for caller-written SQL that uses `:name` markers."""
    if not values:
        return []
    out = []
    for key, value in values.items():
        # {":id": 5} is how the marker is spelled in the SQL; the API wants the
        # bare name, and ":id" never matched, failing as UNBOUND_SQL_PARAMETER.
        name = str(key)[1:] if str(key).startswith(":") else str(key)
        if not name:
            raise ValueError("a statement parameter needs a name")
        out.append(value if isinstance(value, SqlParameter) else _parameter(name, value))
    return out


class ParameterBinder:
    """Hands out `:p0`, `:p1`, ... markers and remembers what they bind to."""

    def __init__(self, prefix: str = "p") -> None:
        self._prefix = prefix
        self.parameters: list[SqlParameter] = []

    def bind(self, value: Any, *, type: str | None = None) -> str:
        name = f"{self._prefix}{len(self.parameters)}"
        param = _parameter(name, value)
        if type is not None:
            param = SqlParameter(name, param.value, type)
        self.parameters.append(param)
        return f":{name}"


class StatementBackend(Protocol):
    """Something that runs one statement and returns Arrow, or nothing."""

    def execute(
        self,
        statement: str,
        parameters: Sequence[SqlParameter] = (),
        *,
        fetch: bool = True,
    ) -> Any:
        """Run `statement`; return a `pyarrow.Table` when `fetch`, else None."""
        ...


#: Opens a request and returns a readable response. Swappable for tests.
Opener = Callable[[urllib.request.Request, float], Any]


def _default_opener(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


class SdkStatementBackend:
    """Runs statements through `WorkspaceClient.statement_execution`.

    `wait_timeout` is how long the first call blocks server-side (``"0s"`` or
    5 to 50 seconds); past that, the statement keeps running and this polls
    `get_statement` until it reaches a terminal state or `timeout` seconds
    elapse, at which point it is cancelled. Ctrl-C cancels it too, so an
    interrupted script does not leave a warehouse query burning DBUs.
    """

    def __init__(
        self,
        client: Any,
        warehouse_id: str,
        *,
        wait_timeout: str = "30s",
        timeout: float | None = 3600.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 10.0,
        download_timeout: float = 300.0,
        opener: Opener | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_interval <= 0 or max_poll_interval <= 0:
            # 0 doubles to 0: a busy loop hammering get_statement.
            raise ValueError("poll_interval and max_poll_interval must be positive")
        _check_timeout(timeout)
        self._client = client
        self._warehouse_id = warehouse_id
        self._wait_timeout = _check_wait_timeout(wait_timeout)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_poll_interval = max_poll_interval
        self._download_timeout = download_timeout
        self._opener: Opener = opener or _default_opener
        self._sleep = sleep
        self._clock = clock

    @property
    def warehouse_id(self) -> str:
        return self._warehouse_id

    def execute(
        self,
        statement: str,
        parameters: Sequence[SqlParameter] = (),
        *,
        fetch: bool = True,
    ) -> Any:
        from databricks.sdk.service.sql import (
            Disposition,
            ExecuteStatementRequestOnWaitTimeout,
            Format,
            StatementParameterListItem,
        )

        api = self._client.statement_execution
        items = [
            StatementParameterListItem(name=p.name, value=p.value, type=p.type) for p in parameters
        ]
        # The deadline starts before the first call: that call itself blocks
        # for up to `wait_timeout`, which must not outlast `timeout`.
        deadline = None if self._timeout is None else self._clock() + self._timeout
        response = api.execute_statement(
            statement=statement,
            warehouse_id=self._warehouse_id,
            format=Format.ARROW_STREAM,
            disposition=Disposition.EXTERNAL_LINKS,
            wait_timeout=_bounded_wait(self._wait_timeout, self._timeout),
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            parameters=items or None,
        )
        statement_id = getattr(response, "statement_id", None)
        try:
            response = self._wait(response, deadline)
        except SqlStatementError as exc:
            if exc.state not in _TERMINAL:
                self._cancel(statement_id)
            raise
        except BaseException:
            # Ctrl-C, a failed poll, anything: the statement must not be left
            # running on the warehouse (and billing) after we stop watching it.
            self._cancel(statement_id)
            raise
        if not fetch:
            return None
        return self._arrow(response, statement_id)

    # ------------------------------------------------------------- polling

    def _wait(self, response: Any, deadline: float | None = None) -> Any:
        statement_id = getattr(response, "statement_id", None)
        interval = self._poll_interval
        while True:
            status = getattr(response, "status", None)
            state = enum_value(getattr(status, "state", None)) or "PENDING"
            if state in _TERMINAL:
                break
            if not statement_id:
                # Polling `get_statement(None)` would request
                # /api/2.0/sql/statements/None forever.
                raise SqlStatementError(
                    "the warehouse accepted the statement but returned no statement_id to "
                    "follow it by",
                    state=state,
                )
            now = self._clock()
            if deadline is not None and now >= deadline:
                # `execute` cancels it on the way out (state is not terminal).
                raise SqlStatementError(
                    f"the statement did not finish within {self._timeout:g}s and was cancelled",
                    statement_id=statement_id,
                    state=state,
                )
            # Never oversleep the deadline by a whole (up to 10s) poll interval.
            self._sleep(interval if deadline is None else max(0.0, min(interval, deadline - now)))
            interval = min(interval * 2, self._max_poll_interval)
            response = self._client.statement_execution.get_statement(statement_id)
            # A polled response may omit the id; keep following the one we have.
            statement_id = getattr(response, "statement_id", None) or statement_id

        if state == "SUCCEEDED":
            return response
        error = getattr(status, "error", None)
        message = getattr(error, "message", None) or f"the statement ended in state {state}"
        raise SqlStatementError(
            message,
            statement_id=statement_id,
            error_code=enum_value(getattr(error, "error_code", None)),
            sql_state=getattr(status, "sql_state", None),
            state=state,
        )

    def _cancel(self, statement_id: str | None) -> None:
        if not statement_id:
            return
        # Best effort: the interrupt or timeout is what the caller must see.
        with contextlib.suppress(Exception):
            self._client.statement_execution.cancel_execution(statement_id)

    # -------------------------------------------------------------- results

    def _arrow(self, response: Any, statement_id: str | None = None) -> Any:
        import pyarrow as pa

        # A polled response may omit the id; the one execute got is authoritative.
        statement_id = getattr(response, "statement_id", None) or statement_id
        manifest = getattr(response, "manifest", None)
        if getattr(manifest, "truncated", None):
            # We set no row_limit/byte_limit, so a truncated result means the
            # warehouse's own cap was hit; returning it would silently drop rows.
            raise SqlStatementError(
                "the warehouse truncated the result, so only part of it is available; "
                "narrow the query (columns, predicate or limit)",
                statement_id=statement_id,
                state="SUCCEEDED",
            )
        batches: list[Any] = []
        schema = None
        api = self._client.statement_execution

        chunk = getattr(response, "result", None)
        total_chunks = getattr(manifest, "total_chunk_count", None)
        if chunk is None and isinstance(total_chunks, int) and total_chunks > 0 and statement_id:
            # The status carried the manifest but not the first chunk.
            chunk = api.get_statement_result_chunk_n(statement_id, 0)
        seen: set[int] = set()
        while chunk is not None:
            links = list(getattr(chunk, "external_links", None) or [])
            index = getattr(chunk, "chunk_index", None)
            if index is None and links:
                # An EXTERNAL_LINKS chunk numbers its links, not itself.
                index = getattr(links[0], "chunk_index", None)
            if index is not None:
                if index in seen:
                    raise SqlStatementError(
                        f"the warehouse served result chunk {index} twice; refusing a result "
                        "that would duplicate or drop rows",
                        statement_id=statement_id,
                    )
                seen.add(index)
            next_index: int | None = getattr(chunk, "next_chunk_index", None)
            for position, link in enumerate(links):
                link_index = getattr(link, "chunk_index", None)
                if link_index is None:
                    data = self._download_fresh(link, index, position, statement_id)
                else:  # re-fetching chunk n puts this link first
                    data = self._download_fresh(link, link_index, 0, statement_id)
                reader = pa.ipc.open_stream(data)
                if schema is None:
                    schema = reader.schema
                batches.extend(reader)
                # The link's own pointer wins, but an absent one must not erase
                # the chunk's: that silently ended the result one chunk early.
                link_next = getattr(link, "next_chunk_index", None)
                if link_next is not None:
                    next_index = link_next
            if next_index is None:
                break
            chunk = api.get_statement_result_chunk_n(statement_id, next_index)

        if schema is None:
            schema = _schema_from_manifest(manifest)
        table = pa.Table.from_batches(batches, schema=schema)
        expected = getattr(manifest, "total_row_count", None)
        if isinstance(expected, int) and table.num_rows != expected:
            raise SqlStatementError(
                f"the result has {table.num_rows} rows but the warehouse reported {expected}; "
                "a result chunk was lost",
                statement_id=statement_id,
            )
        return table

    def _download_fresh(
        self, link: Any, chunk_index: int | None, position: int, statement_id: str | None
    ) -> bytes:
        """Download one link; if it fails (typically expired), re-fetch the chunk once.

        Presigned links live about 15 minutes, and the first chunk's link is
        minted when the statement finishes -- a slow earlier download or a long
        poll can outlive it. `get_statement_result_chunk_n` mints fresh ones.
        """
        try:
            return self._download(link)
        except _DownloadError:
            if chunk_index is None or not statement_id:
                raise
            fresh = self._client.statement_execution.get_statement_result_chunk_n(
                statement_id, chunk_index
            )
            links = list(getattr(fresh, "external_links", None) or [])
            if position >= len(links):
                raise
            return self._download(links[position])

    def _download(self, link: Any) -> bytes:
        url = getattr(link, "external_link", None)
        if not url or not str(url).lower().startswith("https://"):
            raise SqlStatementError(
                f"the warehouse returned a result link that is not https ({url!r}); "
                "refusing to fetch it"
            )
        # Only the headers the link carries. Never the workspace credentials:
        # the URL is presigned and points at cloud storage, not Databricks.
        headers = {str(k): str(v) for k, v in (getattr(link, "http_headers", None) or {}).items()}
        request = urllib.request.Request(str(url), headers=headers, method="GET")
        try:
            with self._opener(request, self._download_timeout) as response:
                data: bytes = response.read()
        except Exception as exc:
            raise _DownloadError(f"could not download a result chunk: {exc}") from exc
        return data


def _check_timeout(timeout: float | None) -> None:
    """None means "no limit"; anything else must be a positive, finite number."""
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise ValueError(f"timeout must be a number of seconds or None, got {timeout!r}")
    if not 0 < timeout < float("inf"):
        # 0 or a negative value cancelled every statement straight after
        # submitting it; NaN never compared past the deadline and hung forever.
        raise ValueError(f"timeout must be positive (or None for no limit), got {timeout!r}")


def _bounded_wait(wait_timeout: str, timeout: float | None) -> str:
    """`wait_timeout`, shortened so the blocking first call cannot outlast `timeout`.

    The API only takes 0s or 5s..50s, so a `timeout` under 5s makes the first
    call return at once and the rest is polled against the deadline.
    """
    if timeout is None:
        return wait_timeout
    seconds = int(wait_timeout[:-1])
    if seconds <= timeout:
        return wait_timeout
    return "0s" if timeout < _MIN_WAIT else f"{int(timeout)}s"


def _check_wait_timeout(value: str | int) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise ValueError(f"wait_timeout must look like '30s', got {value!r}")
    # A bare number of seconds is what people pass; accept it rather than
    # failing with AttributeError on `.strip()`.
    text = f"{value}s" if isinstance(value, int) else value.strip().lower()
    if not text[:-1].isascii():
        raise ValueError(f"wait_timeout must look like '30s', got {value!r}")
    if not text.endswith("s") or not text[:-1].isdigit():
        raise ValueError(f"wait_timeout must look like '30s', got {value!r}")
    seconds = int(text[:-1])
    if seconds != 0 and not _MIN_WAIT <= seconds <= _MAX_WAIT:
        raise ValueError(
            f"wait_timeout must be 0s or between {_MIN_WAIT}s and {_MAX_WAIT}s, got {value!r}"
        )
    return f"{seconds}s"


_TYPE_NAMES: dict[str, str] = {
    "BOOLEAN": "bool_",
    "BOOL": "bool_",
    "BYTE": "int8",
    "TINYINT": "int8",
    "SHORT": "int16",
    "SMALLINT": "int16",
    "INT": "int32",
    "INTEGER": "int32",
    "LONG": "int64",
    "BIGINT": "int64",
    "FLOAT": "float32",
    "REAL": "float32",
    "DOUBLE": "float64",
    "DATE": "date32",
    "BINARY": "binary",
    "STRING": "string",
    "CHAR": "string",
    "VARCHAR": "string",
}

#: Interval units that make a day-time interval (Arrow: a microsecond duration).
_DAY_TIME_UNITS = frozenset({"DAY", "HOUR", "MINUTE", "SECOND"})

_TYPE_TOKEN = re.compile(
    r"\s*(`(?:[^`]|``)*`|'(?:[^'\\]|\\.)*'|[A-Za-z_][A-Za-z0-9_]*|\d+|[<>(),:])"
)


class _TypeParser:
    """A small recursive-descent parser for Databricks `type_text`.

    ``ARRAY<STRUCT<a: INT, b: STRING>>``, ``MAP<STRING, DECIMAL(10,2)>`` and
    the like become the nested Arrow types a non-empty result carries, so an
    empty result has the same schema as a full one. Field names keep their
    case; keywords are compared case-insensitively. Anything it does not
    recognise is a string column, as before.
    """

    def __init__(self, text: str) -> None:
        self.tokens: list[str] = []
        pos = 0
        text = text.strip()
        while pos < len(text):
            match = _TYPE_TOKEN.match(text, pos)
            if match is None:
                raise ValueError(f"cannot parse type {text!r}")
            self.tokens.append(match.group(1))
            pos = match.end()
            while pos < len(text) and text[pos].isspace():
                pos += 1
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of type")
        self.i += 1
        return token

    def expect(self, token: str) -> None:
        if self.take() != token:
            raise ValueError(f"expected {token!r}")

    def parse(self) -> Any:
        result = self.type()
        if self.peek() is not None:
            raise ValueError(f"trailing text in type: {self.peek()!r}")
        return result

    def type(self) -> Any:
        import pyarrow as pa

        word = self.take().upper()
        if word == "ARRAY":
            self.expect("<")
            item = self.type()
            self.expect(">")
            return pa.list_(item)
        if word == "MAP":
            self.expect("<")
            key = self.type()
            self.expect(",")
            value = self.type()
            self.expect(">")
            return pa.map_(key, value)
        if word == "STRUCT":
            self.expect("<")
            fields = []
            while self.peek() != ">":
                name = self.take()
                if name.startswith("`"):
                    name = name[1:-1].replace("``", "`")
                if self.peek() == ":":
                    self.take()
                field_type = self.type()
                nullable = True
                while self.peek() is not None and self.peek().upper() in ("NOT", "COMMENT"):  # type: ignore[union-attr]
                    if self.take().upper() == "NOT":
                        self.take()  # NULL
                        nullable = False
                    else:
                        self.take()  # the comment literal
                fields.append(pa.field(name, field_type, nullable=nullable))
                if self.peek() == ",":
                    self.take()
            self.expect(">")
            return pa.struct(fields)
        if word in ("DECIMAL", "DEC", "NUMERIC"):
            precision, scale = 10, 0  # Databricks' default DECIMAL
            if self.peek() == "(":
                self.take()
                precision = int(self.take())
                if self.peek() == ",":
                    self.take()
                    scale = int(self.take())
                self.expect(")")
            if not 0 < precision <= 38:
                raise ValueError(f"DECIMAL({precision},{scale}) is out of range")
            return pa.decimal128(precision, scale)
        if word in ("CHAR", "VARCHAR") and self.peek() == "(":
            self.take()
            self.take()
            self.expect(")")
            return pa.string()
        if word == "TIMESTAMP_NTZ":
            return pa.timestamp("us")
        if word in ("TIMESTAMP", "TIMESTAMP_LTZ"):
            return pa.timestamp("us", tz="UTC")
        if word in ("VOID", "NULL"):
            return pa.null()
        if word == "INTERVAL":
            units = []
            while self.peek() is not None and self.peek() not in (",", ">", ")"):
                units.append(self.take().upper())
            if units and all(u in _DAY_TIME_UNITS or u == "TO" for u in units):
                return pa.duration("us")
            return pa.string()  # YEAR-MONTH: no pyarrow factory; the warehouse sends text
        factory = _TYPE_NAMES.get(word)
        if factory is not None:
            return getattr(pa, factory)()
        # VARIANT, GEOMETRY, a user-defined type, ...: text, as before.
        if self.peek() == "(":
            depth = 0
            while True:
                token = self.take()
                depth += token == "("
                depth -= token == ")"
                if depth == 0:
                    break
        return pa.string()


def _schema_from_manifest(manifest: Any) -> Any:
    """An empty result still has columns; keep their names and types."""
    import pyarrow as pa

    columns = getattr(getattr(manifest, "schema", None), "columns", None) or []
    fields = []
    for column in sorted(columns, key=lambda c: getattr(c, "position", 0) or 0):
        fields.append(pa.field(str(column.name), _manifest_arrow_type(column)))
    return pa.schema(fields)


def _manifest_arrow_type(column: Any) -> Any:
    """The Arrow type a non-empty result would have carried for this column.

    `type_name` alone is not enough: the SDK maps names it does not know
    (TIMESTAMP_NTZ, VARIANT, ...) to None, DECIMAL carries its precision and
    scale separately, and ARRAY/MAP/STRUCT say nothing about their elements.
    `type_text` (``ARRAY<STRUCT<a: INT>>``) is parsed when present.
    """
    import pyarrow as pa

    type_name = enum_value(getattr(column, "type_name", None))
    text = str(getattr(column, "type_text", None) or "").strip()
    if text:
        with contextlib.suppress(ValueError, IndexError):
            parsed = _TypeParser(text).parse()
            if type_name == "DECIMAL" and not re.search(r"\(", text):
                precision = getattr(column, "type_precision", None)
                scale = getattr(column, "type_scale", None)
                if isinstance(precision, int) and 0 < precision <= 38:
                    return pa.decimal128(precision, int(scale or 0))
            return parsed
    if type_name == "DECIMAL":
        precision = getattr(column, "type_precision", None)
        scale = getattr(column, "type_scale", None)
        if isinstance(precision, int) and 0 < precision <= 38:
            return pa.decimal128(precision, int(scale or 0))
        return pa.decimal128(10, 0)
    if type_name is None or type_name in ("ARRAY", "MAP", "STRUCT", "INTERVAL"):
        return pa.string()
    with contextlib.suppress(ValueError, IndexError):
        return _TypeParser(type_name).parse()
    return pa.string()
