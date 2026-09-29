"""Regressions for the single error-translation boundary (engine/boundary.py).

Every engine a connection routes to is handed out behind one boundary that
turns anything but a DeltaSwampError into this library's type -- including
what a returned stream or MERGE builder raises later. Each test here failed
before it: the error escaped raw, or its type depended on the engine.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")
if not ds.has_native():  # pragma: no cover
    pytest.skip("native extension not built", allow_module_level=True)

from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    DeltaSwampError,
    EngineError,
    EngineFallbackWarning,
    EnginePanicError,
    InvalidArgumentError,
)


@pytest.fixture
def table(conn: Any, tmp_path: Any) -> Any:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1, 2, 3]}))
    return conn.open_table(path)


def _engine(table: Any, kind: Engine) -> Any:
    return table._connection.router.engines[kind]


class TestTheBoundary:
    def test_both_engines_raise_a_library_error_for_the_same_call(self, table: Any) -> None:
        # delta-rs let its DeltaError out, the kernel an UnreachableTableError.
        for kind in (Engine.KERNEL, Engine.DELTARS):
            with pytest.raises(DeltaSwampError):
                _engine(table, kind).files(table.resolved, version=99)

    def test_an_unrecognised_error_is_an_engine_error_carrying_the_original(
        self, table: Any, monkeypatch: Any
    ) -> None:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise KeyError("engine bug")

        monkeypatch.setattr(_engine(table, Engine.DELTARS), "vacuum", broken)
        with pytest.raises(EngineError) as info:
            table.vacuum()
        error = info.value
        assert error.engine == "deltars" and error.operation == "vacuum"
        assert isinstance(error.original, KeyError) and error.__cause__ is error.original
        # Still the builtin, so `except KeyError` keeps working.
        assert isinstance(error, KeyError)

    def test_an_engine_error_pickles(self) -> None:
        from deltaswamp.engine.boundary import translate

        error = translate(Engine.KERNEL, "scan", OSError(2, "gone"))
        again = pickle.loads(pickle.dumps(error))
        assert isinstance(again, EngineError) and isinstance(again, FileNotFoundError)
        assert again.engine == "kernel" and str(again) == str(error)

    def test_a_kernel_panic_is_a_library_error(self, table: Any, monkeypatch: Any) -> None:
        class PanicException(BaseException):  # pyo3_runtime's, by name
            pass

        def panics(*args: Any, **kwargs: Any) -> Any:
            raise PanicException("index out of bounds")

        # The kernel's panics escaped `except Exception` entirely.
        monkeypatch.setattr(_engine(table, Engine.KERNEL), "detail", panics)
        with pytest.raises(EnginePanicError, match="detail"):
            _engine(table, Engine.KERNEL).detail(table.resolved)

    def test_the_extension_sets_a_stable_kind_code(self, table: Any) -> None:
        with pytest.raises(InvalidArgumentError) as info:
            _engine(table, Engine.KERNEL).scan(table.resolved, columns=["nope"])
        assert getattr(getattr(info.value, "original", None), "kind", None) == "invalid_input"


class TestLazyFailures:
    def test_a_stream_failing_mid_read_is_translated(self, table: Any, monkeypatch: Any) -> None:
        schema = pa.schema([("id", pa.int64())])

        def batches() -> Any:
            yield pa.record_batch([pa.array([1])], schema=schema)
            raise RuntimeError("engine bug mid-stream")

        def scan(*args: Any, **kwargs: Any) -> Any:
            return pa.RecordBatchReader.from_batches(schema, batches())

        monkeypatch.setattr(_engine(table, Engine.KERNEL), "scan", scan)
        monkeypatch.setattr(_engine(table, Engine.DELTARS), "scan", scan)
        with pytest.raises(EngineError, match="mid-stream"):
            table.to_arrow()

    def test_a_merge_builder_failing_at_execute_is_translated(
        self, table: Any, monkeypatch: Any
    ) -> None:
        class Builder:
            def when_matched_update(self, *args: Any, **kwargs: Any) -> Builder:
                return self

            def when_matched_update_all(self, *args: Any, **kwargs: Any) -> Builder:
                return self

            def execute(self) -> Any:
                raise ValueError("builder bug")

        for kind in (Engine.KERNEL, Engine.DELTARS):
            monkeypatch.setattr(_engine(table, kind), "merge", lambda *a, **k: Builder())
        merger = table.merge(pa.table({"id": [1]}), "t.id = s.id", source_alias="s")
        with pytest.raises(EngineError, match="builder bug") as info:
            merger.when_matched_update_all().execute()
        assert isinstance(info.value, ValueError)


class TestReadFallback:
    def test_an_engine_failure_still_moves_to_the_next_engine(
        self, table: Any, monkeypatch: Any
    ) -> None:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("cannot parse these statistics")

        monkeypatch.setattr(_engine(table, Engine.KERNEL), "scan", broken)
        with pytest.warns(EngineFallbackWarning, match="RuntimeError"):
            assert sorted(table.to_arrow().column("id").to_pylist()) == [1, 2, 3]

    def test_bad_input_does_not_move_to_the_next_engine(self, table: Any, monkeypatch: Any) -> None:
        served: list[str] = []
        real = _engine(table, Engine.DELTARS).scan

        def spy(*args: Any, **kwargs: Any) -> Any:
            served.append("deltars")
            return real(*args, **kwargs)

        monkeypatch.setattr(_engine(table, Engine.DELTARS), "scan", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("error", EngineFallbackWarning)
            with pytest.raises(InvalidArgumentError):
                table.to_arrow(columns=["nope"])
        assert served == []


class TestArgumentsAtTheEdge:
    def test_a_misspelt_keyword_is_an_invalid_argument(self, conn: Any, table: Any) -> None:
        with pytest.raises(InvalidArgumentError, match="colums"):
            table.to_arrow(colums=["id"])
        with pytest.raises(InvalidArgumentError, match="colour"):
            conn.write_table(table.resolved.location + "_2", pa.table({"id": [1]}), colour="r")

    def test_data_that_is_not_a_table(self, table: Any) -> None:
        for data in ("rows", {"id": 1}, 7):
            with pytest.raises(InvalidArgumentError):
                table.append(data)

    def test_conn_sql_errors_are_library_errors(self, conn: Any, table: Any) -> None:
        with pytest.raises(InvalidArgumentError):
            conn.sql("selec from", tables={"t": table})
        with pytest.raises(InvalidArgumentError):
            conn.sql(1)
