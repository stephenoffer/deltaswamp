"""Fakes for the SQL warehouse engine and its statement backend.

Imported only after `pytest.importorskip("pyarrow")`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pyarrow as pa
from deltaswamp.engine.sql import SqlEngine
from deltaswamp.engine.sql_backend import SqlParameter, SqlStatementError


class RecordingBackend:
    """Records every statement; answers with canned Arrow tables."""

    def __init__(self, results: dict[str, pa.Table] | None = None) -> None:
        self.calls: list[tuple[str, list[SqlParameter], bool]] = []
        self.results = results or {}
        self.fail_on: str | None = None

    def execute(
        self, statement: str, parameters: Any = (), *, fetch: bool = True
    ) -> pa.Table | None:
        self.calls.append((statement, list(parameters), fetch))
        if self.fail_on is not None and self.fail_on in statement:
            raise SqlStatementError("boom from the server", state="FAILED")
        if not fetch:
            return None
        for prefix, table in self.results.items():
            if statement.startswith(prefix):
                return table
        return pa.table({"x": pa.array([], pa.int64())})

    @property
    def sql(self) -> list[str]:
        return [c[0] for c in self.calls]

    @property
    def last(self) -> str:
        return self.calls[-1][0]

    @property
    def params(self) -> dict[str, tuple[str | None, str]]:
        return {p.name: (p.value, p.type) for p in self.calls[-1][1]}


class FakeFiles:
    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.events: list[str] = []

    def upload(self, path: str, contents: Any, *, overwrite: bool | None = None) -> None:
        self.uploaded[path] = contents.read()
        self.events.append(f"upload {path}")

    def delete(self, path: str) -> None:
        self.deleted.append(path)
        self.events.append(f"delete {path}")


class FakeWarehouses:
    def __init__(self, warehouses: list[Any]) -> None:
        self._warehouses = warehouses
        self.calls = 0

    def list(self) -> list[Any]:
        self.calls += 1
        return list(self._warehouses)


def warehouse(wid: str, state: str, *, serverless: bool = False, name: str = "") -> Any:
    return SimpleNamespace(
        id=wid,
        name=name or wid,
        state=SimpleNamespace(value=state),
        enable_serverless_compute=serverless,
    )


class FakeClient:
    def __init__(self, warehouses: list[Any] | None = None, statements: Any = None) -> None:
        self.files = FakeFiles()
        self.warehouses = FakeWarehouses(warehouses or [])
        self.statement_execution = statements


def engine(
    backend: RecordingBackend | None = None, **kwargs: Any
) -> tuple[SqlEngine, RecordingBackend, FakeClient]:
    rec = backend or RecordingBackend()
    client = kwargs.pop("client", None) or FakeClient()
    kwargs.setdefault("staging_volume", "cat.sch.vol")
    kwargs.setdefault("warn_on_use", False)
    return SqlEngine(backend=rec, client=client, **kwargs), rec, client


def status(state: str, message: str | None = None) -> Any:
    error = SimpleNamespace(message=message, error_code="BAD_REQUEST") if message else None
    return SimpleNamespace(state=SimpleNamespace(value=state), error=error, sql_state=None)


def succeeded(sid: str, result: Any = None, manifest: Any = None) -> Any:
    return SimpleNamespace(
        statement_id=sid, status=status("SUCCEEDED"), result=result, manifest=manifest
    )


class FakeStatements:
    """`WorkspaceClient.statement_execution`, scripted."""

    def __init__(
        self,
        responses: list[Any],
        polls: list[Any] | None = None,
        chunks: dict[int, Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._polls = list(polls or [])
        self._chunks = chunks or {}
        self.executed: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.chunk_requests: list[int] = []

    def execute_statement(self, **kwargs: Any) -> Any:
        self.executed.append(kwargs)
        return self._responses.pop(0)

    def get_statement(self, statement_id: str) -> Any:
        return self._polls.pop(0)

    def get_statement_result_chunk_n(self, statement_id: str, chunk_index: int) -> Any:
        self.chunk_requests.append(chunk_index)
        return self._chunks[chunk_index]

    def cancel_execution(self, statement_id: str) -> None:
        self.cancelled.append(statement_id)
