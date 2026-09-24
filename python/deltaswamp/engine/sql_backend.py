"""Running SQL on a Databricks warehouse through the Statement Execution API.

Why this and not `databricks-sql-connector`: the SDK is already a base
dependency, it shares the connection's authentication, and the Statement
Execution API hands results back as Arrow IPC without an extra driver, a
Thrift session, or a second auth code path. The connector would add a heavy
optional install to buy nothing the fallback needs, so this is the only
backend and the `sql` extra is not required.

Three habits this module keeps deliberately:

* **Results are always Arrow.** Queries run with `format=ARROW_STREAM` and
  `disposition=EXTERNAL_LINKS`, so a large result streams from cloud storage
  instead of being squeezed through the JSON inline limit.
* **The Databricks token never leaves for storage.** Presigned result links are
  fetched with only the headers the link itself carries. Sending the workspace
  `Authorization` header to a cloud bucket would leak it to a third party, and
  some stores reject the request outright when it is present.
* **Values travel as parameters.** Callers bind user-supplied literals as named
  `:markers` (`StatementParameterListItem`), so a value is never spliced into
  statement text. Identifiers cannot be parameters and are quoted by the engine.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import decimal
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

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


@dataclass(frozen=True, slots=True)
class SqlParameter:
    """One named parameter marker. `value=None` binds SQL NULL."""

    name: str
    value: str | None
    type: str = "STRING"


def _parameter(name: str, value: Any) -> SqlParameter:
    """Render a Python value as a typed statement parameter."""
    if value is None:
        return SqlParameter(name, None, "STRING")
    if isinstance(value, bool):  # before int: bool is an int subclass
        return SqlParameter(name, "true" if value else "false", "BOOLEAN")
    if isinstance(value, int):
        return SqlParameter(name, str(value), "BIGINT")
    if isinstance(value, float):
        return SqlParameter(name, repr(value), "DOUBLE")
    if isinstance(value, decimal.Decimal):
        _sign, digits, exponent = value.as_tuple()
        scale = max(0, -int(exponent))
        precision = max(len(digits), scale, 1)
        return SqlParameter(name, str(value), f"DECIMAL({min(precision, 38)},{scale})")
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
    return [v if isinstance(v, SqlParameter) else _parameter(k, v) for k, v in values.items()]


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


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


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
        response = api.execute_statement(
            statement=statement,
            warehouse_id=self._warehouse_id,
            format=Format.ARROW_STREAM,
            disposition=Disposition.EXTERNAL_LINKS,
            wait_timeout=self._wait_timeout,
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            parameters=items or None,
        )
        statement_id = getattr(response, "statement_id", None)
        try:
            response = self._wait(response)
            if not fetch:
                return None
            return self._arrow(response)
        except KeyboardInterrupt:
            self._cancel(statement_id)
            raise

    # ------------------------------------------------------------- polling

    def _wait(self, response: Any) -> Any:
        statement_id = getattr(response, "statement_id", None)
        deadline = None if self._timeout is None else self._clock() + self._timeout
        interval = self._poll_interval
        while True:
            status = getattr(response, "status", None)
            state = _enum_value(getattr(status, "state", None)) or "PENDING"
            if state in _TERMINAL:
                break
            if deadline is not None and self._clock() >= deadline:
                self._cancel(statement_id)
                raise SqlStatementError(
                    f"the statement did not finish within {self._timeout:g}s and was cancelled",
                    statement_id=statement_id,
                    state=state,
                )
            self._sleep(interval)
            interval = min(interval * 2, self._max_poll_interval)
            response = self._client.statement_execution.get_statement(statement_id)

        if state == "SUCCEEDED":
            return response
        error = getattr(status, "error", None)
        message = getattr(error, "message", None) or f"the statement ended in state {state}"
        raise SqlStatementError(
            message,
            statement_id=statement_id,
            error_code=_enum_value(getattr(error, "error_code", None)),
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

    def _arrow(self, response: Any) -> Any:
        import pyarrow as pa

        statement_id = getattr(response, "statement_id", None)
        manifest = getattr(response, "manifest", None)
        batches: list[Any] = []
        schema = None

        chunk = getattr(response, "result", None)
        seen: set[int] = set()
        while chunk is not None:
            index = getattr(chunk, "chunk_index", None)
            if index is not None:
                if index in seen:  # defensive: never loop on a repeated chunk
                    break
                seen.add(index)
            next_index: int | None = getattr(chunk, "next_chunk_index", None)
            for link in getattr(chunk, "external_links", None) or []:
                reader = pa.ipc.open_stream(self._download(link))
                schema = schema or reader.schema
                batches.extend(reader)
                next_index = getattr(link, "next_chunk_index", next_index)
            if next_index is None:
                break
            chunk = self._client.statement_execution.get_statement_result_chunk_n(
                statement_id, next_index
            )

        if schema is None:
            schema = _schema_from_manifest(manifest)
        return pa.Table.from_batches(batches, schema=schema)

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
            raise SqlStatementError(f"could not download a result chunk: {exc}") from exc
        return data


def _check_wait_timeout(value: str) -> str:
    text = value.strip().lower()
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
    "BYTE": "int8",
    "SHORT": "int16",
    "INT": "int32",
    "LONG": "int64",
    "FLOAT": "float32",
    "DOUBLE": "float64",
    "DATE": "date32",
    "BINARY": "binary",
    "STRING": "string",
}


def _schema_from_manifest(manifest: Any) -> Any:
    """An empty result still has columns; keep their names and simple types."""
    import pyarrow as pa

    columns = getattr(getattr(manifest, "schema", None), "columns", None) or []
    fields = []
    for column in sorted(columns, key=lambda c: getattr(c, "position", 0) or 0):
        type_name = _enum_value(getattr(column, "type_name", None)) or "STRING"
        if type_name == "TIMESTAMP":
            arrow_type = pa.timestamp("us", tz="UTC")
        else:
            arrow_type = getattr(pa, _TYPE_NAMES.get(type_name, "string"))()
        fields.append(pa.field(str(column.name), arrow_type))
    return pa.schema(fields)
