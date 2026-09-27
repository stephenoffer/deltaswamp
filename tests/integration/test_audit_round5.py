"""Round-5 regressions in the engine error boundary."""

from __future__ import annotations

import pickle
from typing import Any

import pytest
from deltaswamp.errors import DeltaSwampError, EngineError, EnginePanicError, engine_error


@pytest.mark.parametrize(
    "original",
    [
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        UnicodeEncodeError("ascii", "é", 0, 1, "no"),
        ExceptionGroup("several", [ValueError("a")]),
    ],
)
def test_builtins_needing_more_than_a_message_still_translate(original: BaseException) -> None:
    error = engine_error("boom", engine="kernel", operation="scan", original=original)
    assert isinstance(error, EngineError) and isinstance(error, DeltaSwampError)
    assert error.original is original
    assert isinstance(pickle.loads(pickle.dumps(error)), EngineError)


def test_a_panic_mid_stream_is_an_engine_panic_error() -> None:
    pa = pytest.importorskip("pyarrow")
    from deltaswamp.engine.base import TranslatingStream

    class PanicException(BaseException):
        pass

    class Reader:
        schema = pa.schema([("id", pa.int64())])

        def read_next_batch(self) -> Any:
            raise PanicException("index out of bounds")

    schema = pa.schema([("id", pa.int64())])
    stream = TranslatingStream(pa.RecordBatchReader.from_batches(schema, []), "scan")
    stream._reader = Reader()  # read lazily, once iteration starts
    with pytest.raises(EnginePanicError, match="panicked"):
        stream.read_next_batch()


def test_a_double_bound_at_two_to_the_53_skips_no_file(tmp_path: Any) -> None:
    pa = pytest.importorskip("pyarrow")
    import deltaswamp as ds

    conn = ds.connect("file://")
    path = str(tmp_path / "t")
    t = conn.create_table(path, pa.schema([("id", pa.int64())]))
    t.append(pa.table({"id": pa.array([2**53 + 1], pa.int64())}))
    t.append(pa.table({"id": pa.array([1], pa.int64())}))
    t = conn.open_table(path)
    # The D suffix makes a DOUBLE literal; without it 9007199254740992.0 is a
    # DECIMAL, compared exactly.
    for predicate in (f"id = {2**53}D", f"id >= {2**53}D"):
        rows = t.to_arrow(predicate=predicate).column("id").to_pylist()
        # Spark compares as DOUBLE: 2**53 + 1 rounds to 2**53.0.
        assert rows == [2**53 + 1], predicate


@pytest.mark.parametrize("op", ["delete", "count", "to_arrow"])
def test_a_semicolon_in_a_predicate_is_refused(tmp_path: Any, op: str) -> None:
    pa = pytest.importorskip("pyarrow")
    import deltaswamp as ds
    from deltaswamp import PredicateError

    conn = ds.connect("file://")
    t = conn.create_table(str(tmp_path / "t"), pa.schema([("id", pa.int64())]))
    t.append(pa.table({"id": pa.array([1, 2], pa.int64())}))
    with pytest.raises(PredicateError, match="';'"):
        if op == "delete":
            t.delete("id = 1; id = 2")
        else:
            getattr(t, op)(predicate="id = 1; id = 2")
    assert sorted(t.to_arrow().column("id").to_pylist()) == [1, 2]
    # Inside a string literal a ';' is data.
    assert t.count(predicate="'a;b' = 'a;b'") == 2
