"""The Statement Execution backend, against a scripted `WorkspaceClient`.

Results are real Arrow IPC bytes served through a stub opener, so no test
touches the network.
"""

from __future__ import annotations

import datetime as dt
import decimal
import io
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
from deltaswamp.engine.sql_backend import (
    ParameterBinder,
    SdkStatementBackend,
    SqlStatementError,
)

from tests.unit.sql_fakes import FakeStatements, status, succeeded


def arrow_ipc(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def link(
    url: str, chunk: int, next_chunk: int | None, headers: dict[str, str] | None = None
) -> Any:
    return SimpleNamespace(
        external_link=url,
        chunk_index=chunk,
        next_chunk_index=next_chunk,
        http_headers=headers,
    )


class StubOpener:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> Any:
        self.requests.append(request)
        return io.BytesIO(self.bodies[request.full_url])


def backend(statements: FakeStatements, **kwargs: Any) -> SdkStatementBackend:
    kwargs.setdefault("sleep", lambda _s: None)
    return SdkStatementBackend(SimpleNamespace(statement_execution=statements), "wh", **kwargs)


class TestStatementBackend:
    def test_request_shape_and_parameters(self) -> None:
        from databricks.sdk.service.sql import (
            Disposition,
            ExecuteStatementRequestOnWaitTimeout,
            Format,
            StatementParameterListItem,
        )

        statements = FakeStatements([succeeded("s1")])
        binder = ParameterBinder()
        marker = binder.bind("O'Hare")
        backend(statements, wait_timeout="10s").execute(
            f"SELECT {marker}", binder.parameters, fetch=False
        )
        (call,) = statements.executed
        assert call["statement"] == "SELECT :p0"
        assert call["warehouse_id"] == "wh"
        assert call["format"] is Format.ARROW_STREAM
        assert call["disposition"] is Disposition.EXTERNAL_LINKS
        assert call["wait_timeout"] == "10s"
        assert call["on_wait_timeout"] is ExecuteStatementRequestOnWaitTimeout.CONTINUE
        assert call["parameters"] == [
            StatementParameterListItem(name="p0", value="O'Hare", type="STRING")
        ]

    def test_external_links_are_downloaded_without_databricks_auth(self) -> None:
        first = pa.table({"id": [1, 2], "s": ["a", "b"]})
        second = pa.table({"id": [3], "s": ["c"]})
        result = SimpleNamespace(
            external_links=[link("https://bucket/c0", 0, 1, {"x-amz-sse": "AES256"})],
            next_chunk_index=1,
            chunk_index=0,
        )
        chunk1 = SimpleNamespace(
            external_links=[link("https://bucket/c1", 1, None)],
            next_chunk_index=None,
            chunk_index=1,
        )
        statements = FakeStatements([succeeded("s1", result=result)], chunks={1: chunk1})
        opener = StubOpener(
            {"https://bucket/c0": arrow_ipc(first), "https://bucket/c1": arrow_ipc(second)}
        )
        got = backend(statements, opener=opener).execute("SELECT 1")
        assert got.to_pydict() == {"id": [1, 2, 3], "s": ["a", "b", "c"]}
        assert statements.chunk_requests == [1]
        assert [r.full_url for r in opener.requests] == ["https://bucket/c0", "https://bucket/c1"]
        for request in opener.requests:
            assert not any(k.lower() == "authorization" for k, _ in request.header_items())
        assert opener.requests[0].get_header("X-amz-sse") == "AES256"

    def test_non_https_links_are_refused(self) -> None:
        result = SimpleNamespace(
            external_links=[link("file:///etc/passwd", 0, None)],
            next_chunk_index=None,
            chunk_index=0,
        )
        statements = FakeStatements([succeeded("s1", result=result)])
        with pytest.raises(SqlStatementError, match="not https"):
            backend(statements, opener=StubOpener({})).execute("SELECT 1")

    def test_empty_result_keeps_its_columns(self) -> None:
        manifest = SimpleNamespace(
            schema=SimpleNamespace(
                columns=[
                    SimpleNamespace(name="id", position=0, type_name=SimpleNamespace(value="LONG")),
                    SimpleNamespace(
                        name="s", position=1, type_name=SimpleNamespace(value="STRING")
                    ),
                ]
            )
        )
        statements = FakeStatements([succeeded("s1", manifest=manifest)])
        got = backend(statements).execute("SELECT * FROM t LIMIT 0")
        assert got.num_rows == 0
        assert got.schema == pa.schema([("id", pa.int64()), ("s", pa.string())])

    def test_polls_until_terminal(self) -> None:
        pending = SimpleNamespace(statement_id="s1", status=status("PENDING"))
        running = SimpleNamespace(statement_id="s1", status=status("RUNNING"))
        sleeps: list[float] = []
        statements = FakeStatements([pending], polls=[running, succeeded("s1")])
        backend(statements, sleep=sleeps.append).execute("OPTIMIZE t", fetch=False)
        assert sleeps == [1.0, 2.0]

    def test_failure_carries_the_server_message(self) -> None:
        failed = SimpleNamespace(
            statement_id="s1", status=status("FAILED", "[TABLE_OR_VIEW_NOT_FOUND] nope")
        )
        with pytest.raises(SqlStatementError, match="TABLE_OR_VIEW_NOT_FOUND") as info:
            backend(FakeStatements([failed])).execute("SELECT 1")
        assert info.value.statement_id == "s1"
        assert info.value.state == "FAILED"

    def test_timeout_cancels(self) -> None:
        pending = SimpleNamespace(statement_id="s1", status=status("PENDING"))
        statements = FakeStatements([pending], polls=[pending] * 10)
        clock = iter(float(i) for i in range(100))
        with pytest.raises(SqlStatementError, match="did not finish"):
            backend(statements, timeout=3, clock=lambda: next(clock)).execute("SELECT 1")
        assert statements.cancelled == ["s1"]

    def test_ctrl_c_cancels(self) -> None:
        pending = SimpleNamespace(statement_id="s1", status=status("PENDING"))

        def interrupt(_s: float) -> None:
            raise KeyboardInterrupt

        statements = FakeStatements([pending])
        with pytest.raises(KeyboardInterrupt):
            backend(statements, sleep=interrupt).execute("SELECT 1")
        assert statements.cancelled == ["s1"]

    def test_wait_timeout_is_validated(self) -> None:
        with pytest.raises(ValueError):
            backend(FakeStatements([]), wait_timeout="60s")

    def test_parameter_types(self) -> None:
        binder = ParameterBinder()
        for value in (None, 1.5, dt.datetime(2024, 1, 1), decimal.Decimal("12.345")):
            binder.bind(value)
        assert [(p.value, p.type) for p in binder.parameters] == [
            (None, "STRING"),
            ("1.5", "DOUBLE"),
            ("2024-01-01T00:00:00", "TIMESTAMP_NTZ"),
            ("12.345", "DECIMAL(5,3)"),
        ]
