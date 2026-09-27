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
