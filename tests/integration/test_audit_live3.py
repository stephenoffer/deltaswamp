"""Regressions for the third live re-verification, reproduced on local tables."""

from __future__ import annotations

from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")

from tests.integration.test_audit_live import _write_log  # noqa: E402


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _nested_variant_table(conn: Any, tmp_path: Any) -> Any:
    from deltaswamp._variant import encode, variant_type

    vt = variant_type(pa)

    def value(text: str) -> dict[str, bytes]:
        m, v = encode(text)
        return {"metadata": m, "value": v}

    def field(name: str, kind: Any) -> dict[str, Any]:
        return {"name": name, "type": kind, "nullable": True, "metadata": {}}

    schema = {
        "type": "struct",
        "fields": [
            field("id", "long"),
            field(
                "s", {"type": "struct", "fields": [field("v2", "variant"), field("n", "integer")]}
            ),
            field("arr", {"type": "array", "elementType": "variant", "containsNull": True}),
            field(
                "m",
                {
                    "type": "map",
                    "keyType": "string",
                    "valueType": "variant",
                    "valueContainsNull": True,
                },
            ),
        ],
    }
    data = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "s": pa.array(
                [{"v2": value('{"a":1}'), "n": 1}],
                pa.struct([pa.field("v2", vt), pa.field("n", pa.int32())]),
            ),
            "arr": pa.array([[value("[1,2]")]], pa.list_(pa.field("element", vt))),
            "m": pa.array(
                [[("k", value('{"q":true}'))]], pa.map_(pa.string(), pa.field("value", vt))
            ),
        }
    )
    protocol = {
        "minReaderVersion": 3,
        "minWriterVersion": 7,
        "readerFeatures": ["variantType"],
        "writerFeatures": ["variantType"],
    }
    path = str(tmp_path / "nested_variant")
    _write_log(path, schema, protocol, data)
    return conn.open_table(path)


class TestNestedVariants:
    """LV-02: a VARIANT in an array or map read as binary, while the warehouse sent text."""

    def test_every_nesting_level_reads_as_json_text(self, conn: Any, tmp_path: Any) -> None:
        t = _nested_variant_table(conn, tmp_path)
        got = pa.table(t.to_arrow())
        assert got.schema.field("arr").type == pa.list_(pa.field("element", pa.string()))
        assert got.schema.field("m").type.item_type == pa.string()
        assert t.schema().field("arr").type.value_type == pa.string()
        assert got.to_pylist() == [
            {"id": 1, "s": {"v2": '{"a":1}', "n": 1}, "arr": ["[1,2]"], "m": [("k", '{"q":true}')]}
        ]

    def test_the_text_round_trips(self, conn: Any, tmp_path: Any) -> None:
        t = _nested_variant_table(conn, tmp_path)
        t.append(pa.table(t.to_arrow()).set_column(0, "id", pa.array([2], pa.int64())))
        t.append(pa.table({"id": pa.array([3], pa.int64())}))
        rows = {
            r["id"]: r
            for r in pa.table(conn.open_table(t.resolved.location).to_arrow()).to_pylist()
        }
        assert rows[2]["arr"] == ["[1,2]"] and rows[2]["m"] == [("k", '{"q":true}')]
        assert rows[2]["s"] == {"v2": '{"a":1}', "n": 1}
        assert rows[3]["s"] is None and rows[3]["arr"] is None


def test_a_uint64_beyond_the_column_range_is_an_argument_error(conn: Any, tmp_path: Any) -> None:
    from deltaswamp.errors import InvalidArgumentError

    t = conn.write_table(str(tmp_path / "u"), pa.table({"u": pa.array([1], pa.int64())}))
    with pytest.raises(InvalidArgumentError, match="DECIMAL"):
        t.append(pa.table({"u": pa.array([2**63 + 5], pa.uint64())}))


def test_a_str_column_name_holding_a_dot_is_the_top_level_column(conn: Any, tmp_path: Any) -> None:
    # N2: a str was always split as a dotted path, so `dot.name` was the field
    # `name` of a struct `dot`, and `back`tick` an unterminated quote.
    t = conn.write_table(
        str(tmp_path / "dots"),
        pa.table({"id": [1], "dot.name": [2], "back`tick": [3]}),
        properties={"delta.columnMapping.mode": "name"},
    )
    t.set_column_comment("dot.name", "x")
    t.set_column_comment("back`tick", "y")
    t.rename_column("back`tick", "bt2")
    t.drop_column("dot.name")
    assert list(t.schema().names) == ["id", "bt2"]


def test_insert_takes_values_as_well_as_updates(conn: Any, tmp_path: Any) -> None:
    t = conn.write_table(str(tmp_path / "m"), pa.table({"id": [1], "v": ["a"]}))
    t.merge(pa.table({"id": [2], "v": ["b"]}), "target.id = source.id").when_not_matched_insert(
        values={"id": "source.id", "v": "source.v"}
    ).execute()
    assert sorted(pa.table(t.to_arrow()).column("id").to_pylist()) == [1, 2]


def test_can_incremental_agrees_with_changes(conn: Any, tmp_path: Any) -> None:
    # C7: can(INCREMENTAL) refused every table while changes() served it.
    t = conn.write_table(
        str(tmp_path / "cdf"),
        pa.table({"id": [1]}),
        properties={"delta.enableChangeDataFeed": "true"},
    )
    t.append(pa.table({"id": [2]}))
    verdict = t.can("incremental")
    assert verdict.ok and verdict.operation.value == "incremental"
    assert [v for v, _ in t.changes(t.version)] == [t.version]
