"""Docs and ergonomics audit: help() text, and errors that say how to fix them.

Each test failed before its fix.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402
from deltaswamp.errors import InvalidArgumentError  # noqa: E402


@pytest.fixture
def table(tmp_path: Any) -> Any:
    t = ds.connect("file://").create_table(
        str(tmp_path / "t"), pa.schema([("id", pa.int64()), ("city", pa.string())])
    )
    t.append(pa.table({"id": [1, 2], "city": ["a", "b"]}))
    return t


@pytest.mark.parametrize("cls", [ds.Table, ds.Connection])
def test_every_public_method_has_a_docstring(cls: type) -> None:
    # help(t.to_arrow), help(t.detail) and forty others printed no text.
    missing = [
        name
        for name, member in vars(cls).items()
        if not name.startswith("_")
        and (callable(member) or isinstance(member, property))
        and not inspect.getdoc(member)
    ]
    assert not missing, f"{cls.__name__} methods without a docstring: {missing}"


def test_an_extra_column_says_to_merge_the_schema(table: Any) -> None:
    # delta-rs said only "number of fields does not match: 3 vs 2".
    with pytest.raises(InvalidArgumentError, match="schema_mode='merge'"):
        table.append(pa.table({"id": [3], "city": ["c"], "extra": [1]}))


def test_base_install_appends_and_merges_arro3(tmp_path: Any, monkeypatch: Any) -> None:
    # The NaN check before a delta-rs write, and the MERGE key bound,
    # imported pyarrow unconditionally, so the base install's append of
    # arro3 data failed with EngineError[ImportError].
    import sys

    arro3 = pytest.importorskip("arro3.core")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    schema = arro3.Schema([arro3.Field("id", arro3.DataType.int64())])
    data = arro3.Table.from_pydict({"id": arro3.Array([1, 2], arro3.DataType.int64())})
    t = ds.connect("file://").create_table(str(tmp_path / "t"), schema)
    assert t.append(data)["num_rows"] == 2
    merged = t.merge(data, "target.id = source.id").when_matched_update_all().execute()
    assert merged["num_updated_rows"] == 2
    with pytest.raises(ImportError, match=r"deltaswamp\[pyarrow\]"):
        t.optimize()
