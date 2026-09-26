"""Spark SQL, column DEFAULTs and VARIANT through the direct engines, on real tables.

Each test failed before its fix. Expected results are what a Databricks SQL
warehouse did with the same statement (checked live): `"ab"` is a string,
`substring(s, 0, 2)` is `ab`, `5 / 2` is 2.5, an INSERT that leaves out a
column writes its DEFAULT, and a VARIANT reads back as `to_json` text.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from deltaswamp.capability import Operation
from deltaswamp.errors import (
    DeltaSwampError,
    EngineLimitError,
    InvalidArgumentError,
    UnreachableTableError,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
native = pytest.importorskip("deltaswamp._native")

DV = {"delta.enableDeletionVectors": "true"}


def _rows(conn: Any, path: str) -> list[dict[str, Any]]:
    rows = conn.open_table(path).to_arrow().to_pylist()
    return sorted(rows, key=lambda r: json.dumps(r, default=str, sort_keys=True))


def _table(conn: Any, tmp_path: Any, schema: Any, data: Any, dv: bool, name: str = "t") -> str:
    path = str(tmp_path / name)
    conn.create_table(path, schema, properties=DV if dv else None)
    conn.open_table(path).append(data)
    return path


def _write_log(
    path: str, schema: dict[str, Any], protocol: dict[str, Any], data: Any, config: Any = None
) -> str:
    """A one-commit table written by hand, for shapes the engines cannot create."""
    os.makedirs(os.path.join(path, "_delta_log"))
    pq.write_table(data, os.path.join(path, "part-0.parquet"))
    size = os.path.getsize(os.path.join(path, "part-0.parquet"))
    actions = [
        {"protocol": protocol},
        {
            "metaData": {
                "id": "00000000-0000-0000-0000-000000000001",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps(schema),
                "partitionColumns": [],
                "configuration": config or {},
                "createdTime": 0,
            }
        },
        {
            "add": {
                "path": "part-0.parquet",
                "partitionValues": {},
                "size": size,
                "modificationTime": 0,
                "dataChange": True,
                "stats": json.dumps({"numRecords": data.num_rows}),
            }
        },
    ]
    with open(os.path.join(path, "_delta_log", f"{0:020}.json"), "w") as f:
        f.write("\n".join(json.dumps(a) for a in actions))
    return path


def _field(name: str, kind: Any, metadata: Any = None) -> dict[str, Any]:
    return {"name": name, "type": kind, "nullable": True, "metadata": metadata or {}}


def _protocol(writer: list[str], reader: list[str] | None = None, dv: bool = False) -> Any:
    writer, reader = list(writer), list(reader or [])
    if dv:
        writer.append("deletionVectors")
        reader.append("deletionVectors")
    protocol: dict[str, Any] = {
        "minReaderVersion": 3 if reader else 1,
        "minWriterVersion": 7,
        "writerFeatures": writer,
    }
    if reader:
        protocol["readerFeatures"] = reader
    return protocol


# ------------------------------------------------------- CR-1, CR-6: "..."


@pytest.mark.parametrize("dv", [False, True])
def test_double_quoted_literal_is_a_string_in_a_function_delete(
    conn: Any, tmp_path: Any, dv: bool
) -> None:
    # A column named `ab`: DataFusion read "ab" as it, and deleted the row
    # where s equalled the ab column instead of the row s = 'ab'.
    schema = pa.schema([("s", pa.string()), ("ab", pa.string())])
    path = _table(conn, tmp_path, schema, pa.table({"s": ["ab", "zz"], "ab": ["x", "zz"]}), dv)
    conn.open_table(path).delete("concat(s, '') = \"ab\"")
    assert _rows(conn, path) == [{"s": "zz", "ab": "zz"}]


def test_double_quoted_literal_without_a_clashing_column(conn: Any, tmp_path: Any) -> None:
    # CR-6: the same text raised DeltaError "No field named ab".
    schema = pa.schema([("s", pa.string())])
    path = _table(conn, tmp_path, schema, pa.table({"s": ["ab", "zz"]}), False)
    conn.open_table(path).delete('upper(s) = "AB"')
    assert _rows(conn, path) == [{"s": "zz"}]


def test_double_quoted_literal_in_an_update_set(conn: Any, tmp_path: Any) -> None:
    schema = pa.schema([("s", pa.string()), ("ab", pa.string())])
    path = _table(conn, tmp_path, schema, pa.table({"s": ["zz"], "ab": ["y"]}), False)
    conn.open_table(path).update({"s": 'concat("ab", "!")'}, predicate="s = 'zz'")
    assert _rows(conn, path) == [{"s": "ab!", "ab": "y"}]


@pytest.mark.parametrize("dv", [False, True])
def test_double_quoted_literal_in_a_merge_condition(conn: Any, tmp_path: Any, dv: bool) -> None:
    schema = pa.schema([("id", pa.int64()), ("v", pa.string())])
    path = _table(conn, tmp_path, schema, pa.table({"id": [1, 2], "v": ["a", "b"]}), dv)
    source = pa.table({"id": [1, 2], "v": ["new1", "new2"], "tag": ["x", "v"]})
    conn.open_table(path).merge(source, "target.id = source.id").when_matched_update(
        {"v": "source.v"}, predicate='source.tag = "v"'
    ).execute()
    assert _rows(conn, path) == [{"id": 1, "v": "a"}, {"id": 2, "v": "new2"}]


# ------------------------------------------------ CR-3: function semantics


@pytest.mark.parametrize("dv", [False, True])
@pytest.mark.parametrize(
    ("expression", "spark"),
    [
        ("CAST(source.d AS BIGINT)", 1),  # DuckDB rounded 1.9 to 2
        ("length(substring(source.s, 0, 2))", 2),  # both engines: 1
        ("source.n / 2 * 2", 5),  # DataFusion: 4 (integer division)
        ("source.n DIV 2", 2),
    ],
)
def test_merge_set_computes_what_spark_does(
    conn: Any, tmp_path: Any, dv: bool, expression: str, spark: int
) -> None:
    schema = pa.schema([("id", pa.int64()), ("n", pa.int64())])
    path = _table(conn, tmp_path, schema, pa.table({"id": [1], "n": [0]}), dv)
    source = pa.table({"id": [1], "d": [1.9], "s": ["abc"], "n": [5]})
    conn.open_table(path).merge(source, "target.id = source.id").when_matched_update(
        {"n": expression}
    ).execute()
    assert _rows(conn, path) == [{"id": 1, "n": spark}]


@pytest.mark.parametrize(
    ("predicate", "kept"),
    [
        ("substring(s, 0, 2) = 'ab'", ["xyz"]),
        ("CAST(d AS BIGINT) = 1", ["xyz"]),
        ("abs(l) / 2 > 2", ["xyz"]),  # DataFusion: 5 / 2 = 2, so nothing matched
        ("abs(l) DIV 2 = 2", ["xyz"]),  # CR-5: DIV did not parse
        ("abs(l) = 5L", ["xyz"]),
        ("s RLIKE '^a.c$'", ["xyz"]),
        ("l IN (9007199254740992, 5D) AND abs(l) > 0", ["xyz"]),
        ("left(s, -1) = ''", []),  # both engines: all but the last character
        ("log(l) > 1.6", ["xyz"]),  # natural log: ln 5 = 1.609; log10 5 = 0.699
    ],
)
def test_delta_rs_delete_matches_what_spark_matches(
    conn: Any, tmp_path: Any, predicate: str, kept: list[str]
) -> None:
    schema = pa.schema([("s", pa.string()), ("d", pa.float64()), ("l", pa.int64())])
    data = pa.table({"s": ["abc", "xyz"], "d": [1.9, 2.5], "l": [5, 1]})
    path = _table(conn, tmp_path, schema, data, False)
    conn.open_table(path).delete(predicate)
    assert [r["s"] for r in _rows(conn, path)] == kept


def test_integer_division_in_a_delta_rs_update(conn: Any, tmp_path: Any) -> None:
    schema = pa.schema([("n", pa.int64()), ("x", pa.float64())])
    path = _table(conn, tmp_path, schema, pa.table({"n": [5], "x": [0.0]}), False)
    conn.open_table(path).update({"x": "n / 2"})
    assert _rows(conn, path) == [{"n": 5, "x": 2.5}]


def test_a_kernel_merge_runtime_error_is_this_librarys(conn: Any, tmp_path: Any) -> None:
    # Spark raises REMAINDER_BY_ZERO; DuckDB returned NULL, then its own error type.
    schema = pa.schema([("id", pa.int64()), ("n", pa.int64())])
    path = _table(conn, tmp_path, schema, pa.table({"id": [1], "n": [0]}), True)
    merge = conn.open_table(path).merge(pa.table({"id": [1], "z": [0]}), "target.id = source.id")
    with pytest.raises(InvalidArgumentError, match="REMAINDER_BY_ZERO"):
        merge.when_matched_update({"n": "7 % source.z"}).execute()
    assert _rows(conn, path) == [{"id": 1, "n": 0}]


@pytest.mark.parametrize(
    "predicate", ["arr[0] = 1", "split(s, ',')[0] = 'a'", "date_format(d, 'yyyy') = '2024'"]
)
def test_sql_no_direct_engine_matches_routes_to_the_warehouse(
    conn: Any, tmp_path: Any, predicate: str
) -> None:
    # An array subscript counts from 0 in Spark and 1 in both engines; split
    # takes a regex; a date pattern is Java's. Refused, and can() agrees.
    schema = pa.schema([("arr", pa.list_(pa.int64())), ("s", pa.string()), ("d", pa.date32())])
    data = pa.table({"arr": [[1, 2]], "s": ["a,b"], "d": [None]}, schema=schema)
    path = _table(conn, tmp_path, schema, data, False)
    table = conn.open_table(path)
    assert not table.can(Operation.DELETE, predicate=predicate).ok
    with pytest.raises(UnreachableTableError, match="spark_sql"):
        table.delete(predicate)
    assert len(_rows(conn, path)) == 1


def test_date_format_update_is_refused_not_written_as_text(conn: Any, tmp_path: Any) -> None:
    # delta-rs wrote the literal text 'yyyy-MM' for date_format(d, 'yyyy-MM').
    schema = pa.schema([("d", pa.date32()), ("s", pa.string())])
    path = _table(conn, tmp_path, schema, pa.table({"d": [None], "s": ["x"]}, schema=schema), False)
    table = conn.open_table(path)
    updates = {"s": "date_format(d, 'yyyy-MM')"}
    assert not table.can(Operation.UPDATE, updates=updates).ok
    with pytest.raises(UnreachableTableError):
        table.update(updates)
    assert _rows(conn, path) == [{"d": None, "s": "x"}]


def test_merge_with_untranslatable_sql_is_refused_before_writing(conn: Any, tmp_path: Any) -> None:
    schema = pa.schema([("id", pa.int64()), ("s", pa.string())])
    for dv in (False, True):
        path = _table(conn, tmp_path, schema, pa.table({"id": [1], "s": ["a"]}), dv, f"t{dv}")
        source = pa.table({"id": [1], "s": ["x,y"]})
        merge = conn.open_table(path).merge(source, "target.id = source.id")
        with pytest.raises(EngineLimitError, match="array subscript"):
            merge.when_matched_update({"s": "split(source.s, ',')[0]"}).execute()
        assert _rows(conn, path) == [{"id": 1, "s": "a"}]


# ------------------------------------------------------ CR-2, FZ-7: DEFAULT


def _defaults_table(tmp_path: Any, name: str, dv: bool, city_default: str = "'new'") -> str:
    schema = {
        "type": "struct",
        "fields": [
            _field("id", "long"),
            _field("city", "string", {"CURRENT_DEFAULT": city_default}),
            _field("n", "integer", {"CURRENT_DEFAULT": "42"}),
        ],
    }
    data = pa.table(
        {"id": pa.array([1, 2], pa.int64()), "city": ["x", "y"], "n": pa.array([1, 2], pa.int32())}
    )
    return _write_log(
        str(tmp_path / name),
        schema,
        _protocol(["allowColumnDefaults"], dv=dv),
        data,
        DV if dv else None,
    )


def test_kernel_merge_insert_writes_the_defaults_it_leaves_out(conn: Any, tmp_path: Any) -> None:
    path = _defaults_table(tmp_path, "t", dv=True)
    source = pa.table({"id": pa.array([3], pa.int64())})
    conn.open_table(path).merge(
        source, "t.id = s.id", source_alias="s", target_alias="t"
    ).when_not_matched_insert({"id": "s.id"}).execute()
    assert {"id": 3, "city": "new", "n": 42} in _rows(conn, path)


def test_kernel_merge_set_default(conn: Any, tmp_path: Any) -> None:
    path = _defaults_table(tmp_path, "t", dv=True)
    source = pa.table({"id": pa.array([2], pa.int64())})
    conn.open_table(path).merge(
        source, "t.id = s.id", source_alias="s", target_alias="t"
    ).when_matched_update({"city": "DEFAULT", "n": "default"}).execute()
    assert {"id": 2, "city": "new", "n": 42} in _rows(conn, path)


def test_kernel_merge_refuses_an_expression_default(conn: Any, tmp_path: Any) -> None:
    # `current_timestamp()` is Databricks' to evaluate; NULL was written.
    path = _defaults_table(tmp_path, "t", dv=True, city_default="current_user()")
    source = pa.table({"id": pa.array([3], pa.int64())})
    merge = (
        conn.open_table(path)
        .merge(source, "t.id = s.id", source_alias="s", target_alias="t")
        .when_not_matched_insert({"id": "s.id"})
    )
    with pytest.raises(UnreachableTableError, match="only Databricks evaluates"):
        merge.execute()
    assert len(_rows(conn, path)) == 2


@pytest.mark.parametrize("dv", [False, True])
def test_update_set_default(conn: Any, tmp_path: Any, dv: bool) -> None:
    path = _defaults_table(tmp_path, "t", dv=dv)
    table = conn.open_table(path)
    if not dv:
        # delta-rs cannot write this table; the kernel only by rewriting.
        pytest.skip("allowColumnDefaults without DVs has no direct UPDATE engine")
    table.update({"city": "DEFAULT"}, predicate="id = 1")
    assert {"id": 1, "city": "new", "n": 1} in _rows(conn, path)


def test_update_set_default_of_an_expression_needs_the_warehouse(conn: Any, tmp_path: Any) -> None:
    path = _defaults_table(tmp_path, "t", dv=True, city_default="current_user()")
    table = conn.open_table(path)
    assert not table.can(Operation.UPDATE, updates={"city": "DEFAULT"}).ok
    with pytest.raises(UnreachableTableError, match="sql_column_defaults"):
        table.update({"city": "DEFAULT"})


def test_set_default_without_a_default_is_null(conn: Any, tmp_path: Any) -> None:
    schema = pa.schema([("id", pa.int64()), ("s", pa.string())])
    for dv in (False, True):
        path = _table(conn, tmp_path, schema, pa.table({"id": [1], "s": ["a"]}), dv, f"t{dv}")
        conn.open_table(path).update({"s": "DEFAULT"})
        assert _rows(conn, path) == [{"id": 1, "s": None}]


def test_a_column_named_default_wins_over_the_keyword(conn: Any, tmp_path: Any) -> None:
    # Databricks: `SET c = DEFAULT` reads a column called `default`.
    schema = pa.schema([("id", pa.int64()), ("default", pa.string()), ("c", pa.string())])
    data = pa.table({"id": [1], "default": ["col"], "c": ["x"]})
    path = _table(conn, tmp_path, schema, data, True)
    conn.open_table(path).update({"c": "DEFAULT"})
    assert _rows(conn, path) == [{"id": 1, "default": "col", "c": "col"}]


# ------------------------------------------------------------- VARIANT


def _variant_table(tmp_path: Any, dv: bool, name: str = "v") -> str:
    from deltaswamp._variant import variant_column

    schema = {"type": "struct", "fields": [_field("id", "long"), _field("v", "variant")]}
    column = variant_column(pa, pa.array(['{"a":1}', '"s"']))
    data = pa.table({"id": pa.array([1, 2], pa.int64()), "v": column})
    protocol = _protocol(["variantType"], ["variantType"], dv=dv)
    return _write_log(str(tmp_path / name), schema, protocol, data, DV if dv else None)


def test_planned_scan_and_write_take_variant_as_json_text(conn: Any, tmp_path: Any) -> None:
    # CR-4: plan_scan().read() gave the binary struct, and plan_write().write()
    # of JSON text failed with a raw "Expected Struct, got Utf8".
    path = _variant_table(tmp_path, dv=False)
    table = conn.open_table(path)
    plan = table.plan_write()
    plan.commit([plan.write(pa.table({"id": pa.array([3], pa.int64()), "v": ['{"c":3}']}))])
    read = conn.open_table(path).plan_scan().read()
    assert read.schema.field("v").type == pa.string()
    assert sorted(read.to_pylist(), key=lambda r: r["id"]) == [
        {"id": 1, "v": '{"a":1}'},
        {"id": 2, "v": '"s"'},
        {"id": 3, "v": '{"c":3}'},
    ]


def test_a_struct_shaped_like_a_variant_is_not_decoded_as_one(conn: Any, tmp_path: Any) -> None:
    # V1: struct<metadata: binary, value: binary> was taken for a VARIANT.
    from deltaswamp._variant import variant_column

    blob_type = {
        "type": "struct",
        "fields": [
            {"name": "metadata", "type": "binary", "nullable": False, "metadata": {}},
            {"name": "value", "type": "binary", "nullable": False, "metadata": {}},
        ],
    }
    schema = {"type": "struct", "fields": [_field("v", "variant"), _field("blob", blob_type)]}
    column = variant_column(pa, pa.array(['{"a":1}']))
    blob = pa.StructArray.from_arrays(
        [pa.array([b"hdr"], pa.binary()), pa.array([b"\xff payload"], pa.binary())],
        fields=list(column.type),
    )
    path = _write_log(
        str(tmp_path / "t"),
        schema,
        _protocol(["variantType"], ["variantType"]),
        pa.table({"v": column, "blob": blob}),
    )
    table = conn.open_table(path)
    assert pa.types.is_struct(table.schema().field("blob").type)
    assert table.to_arrow().to_pylist() == [
        {"v": '{"a":1}', "blob": {"metadata": b"hdr", "value": b"\xff payload"}}
    ]


def test_a_variant_nested_in_a_struct_reads_and_writes_as_json_text(
    conn: Any, tmp_path: Any
) -> None:
    # V5: the nested VARIANT came back as the binary struct, and JSON text
    # appended into it failed with SchemaMismatchError.
    from deltaswamp._variant import variant_column

    inner = {"type": "struct", "fields": [_field("w", "variant")]}
    schema = {"type": "struct", "fields": [_field("id", "long"), _field("s", inner)]}
    w = variant_column(pa, pa.array(['{"a":1}']))
    data = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "s": pa.StructArray.from_arrays([w], fields=[pa.field("w", w.type)]),
        }
    )
    path = _write_log(
        str(tmp_path / "t"), schema, _protocol(["variantType"], ["variantType"]), data
    )
    table = conn.open_table(path)
    assert table.schema().field("s").type == pa.struct([pa.field("w", pa.string())])
    table.append(pa.table({"id": pa.array([2], pa.int64()), "s": [{"w": "[1,2]"}]}))
    assert _rows(conn, path) == [{"id": 1, "s": {"w": '{"a":1}'}}, {"id": 2, "s": {"w": "[1,2]"}}]


@pytest.mark.parametrize("dv", [False, True])
def test_update_new_values_takes_variant_json_text(conn: Any, tmp_path: Any, dv: bool) -> None:
    # FZ-6: delta-rs raised "Unsupported CAST from Utf8 to Struct", the kernel
    # "cannot be stored in the column's type".
    path = _variant_table(tmp_path, dv)
    conn.open_table(path).update(new_values={"v": '{"q":2}'}, predicate="id = 1")
    assert _rows(conn, path) == [{"id": 1, "v": '{"q":2}'}, {"id": 2, "v": '"s"'}]


@pytest.mark.parametrize("dv", [False, True])
def test_update_sql_variant_values_mean_what_spark_stores(
    conn: Any, tmp_path: Any, dv: bool
) -> None:
    # Databricks: a string literal is a variant string; parse_json an object.
    path = _variant_table(tmp_path, dv)
    table = conn.open_table(path)
    table.update({"v": "'{\"q\":2}'"}, predicate="id = 1")
    if not dv:
        table.update({"v": "parse_json('[1, 2]')"}, predicate="id = 2")
    rows = _rows(conn, path)
    assert rows[0] == {"id": 1, "v": '"{\\"q\\":2}"'}
    if not dv:
        assert rows[1] == {"id": 2, "v": "[1,2]"}


def test_kernel_merge_set_parse_json_literal(conn: Any, tmp_path: Any) -> None:
    path = _variant_table(tmp_path, dv=True)
    source = pa.table({"id": pa.array([1], pa.int64())})
    conn.open_table(path).merge(source, "target.id = source.id").when_matched_update(
        {"v": "parse_json('{\"z\": 0.5}')"}
    ).execute()
    assert {"id": 1, "v": '{"z":0.5}'} in _rows(conn, path)


def test_kernel_merge_translates_the_extensions_input_errors(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    # The commit runs after KernelEngine.merge() returned, outside the wrapper
    # that turns the extension's InvalidInputError into this library's error.
    error = getattr(native, "InvalidInputError", None)
    if error is None:
        pytest.skip("native build has no InvalidInputError")
    schema = pa.schema([("id", pa.int64()), ("n", pa.int64())])
    path = _table(conn, tmp_path, schema, pa.table({"id": [1], "n": [0]}), True)
    from deltaswamp.engine.kernel import KernelEngine

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise error("the extension refused its input")

    monkeypatch.setattr(KernelEngine, "_commit_dv_changes", refuse)
    merge = conn.open_table(path).merge(pa.table({"id": [1], "n": [5]}), "target.id = source.id")
    with pytest.raises(DeltaSwampError):
        merge.when_matched_update({"n": "source.n"}).execute()
