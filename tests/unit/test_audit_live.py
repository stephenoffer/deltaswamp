"""Regression tests for the live-matrix audit (LM-*), the parts that need no warehouse.

Each one reproduces something a real Databricks warehouse did differently from
what deltaswamp claimed or sent.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyarrow")

from deltaswamp.capability import Operation
from deltaswamp.engine.sql import SqlEngine

from tests.helpers import resolved_table as table

# ------------------------------------------------ LM-03: legacy-protocol ALTERs


@pytest.mark.parametrize(
    "operation", [Operation.RENAME_COLUMN, Operation.DROP_COLUMN, Operation.ALTER_COLUMN_TYPE]
)
def test_legacy_protocol_alters_are_refused(operation: Operation) -> None:
    # (1, 2) names no features; Databricks still refuses all three ALTERs.
    legacy = table(min_reader_version=1, min_writer_version=2)
    verdict = SqlEngine._table_type_refusal(operation, legacy)
    assert verdict is not None and not verdict.ok


def test_unknown_protocol_is_not_refused() -> None:
    # Before the log is read nothing is known, and nothing is refused.
    assert SqlEngine._table_type_refusal(Operation.RENAME_COLUMN, table()) is None


# ------------------------------------------------ LM-02: exact decimal stats


def test_files_keeps_decimal_stats_exact() -> None:
    from decimal import Decimal

    import pyarrow as pa
    from deltaswamp.table import _flat_files

    # The stats text Databricks writes for DECIMAL(38, 9) and DECIMAL(38, 0).
    big = 10**38 - 1
    bounds = f'{{"dec": 12345678901234567890.123456789, "big": {big}, "f": 1.1}}'
    text = (
        f'{{"numRecords": 1, "minValues": {bounds}, "maxValues": {bounds}, '
        '"nullCount": {"dec": 0, "big": 0, "f": 0}}'
    )
    files = pa.table(
        {
            "path": ["a.parquet"],
            "size": pa.array([1], pa.int64()),
            "modification_time": pa.array([0], pa.int64()),
            "num_records": pa.array([1], pa.int64()),
            "stats": [text],
            "partition_values": pa.array([[]], pa.map_(pa.string(), pa.string())),
        }
    )
    schema = pa.schema(
        [("dec", pa.decimal128(38, 9)), ("big", pa.decimal128(38, 0)), ("f", pa.float64())]
    )
    out = _flat_files(pa, files, schema)
    assert out.column("min.dec").to_pylist() == [Decimal("12345678901234567890.123456789")]
    assert out.column("max.big").to_pylist() == [Decimal(10**38 - 1)]
    assert out.schema.field("max.big").type == pa.decimal128(38, 0)
    assert out.column("min.f").to_pylist() == [1.1]


# ------------------------------------------------ the SQL warehouse's statements

import io  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from tests.unit.sql_fakes import RecordingBackend, engine  # noqa: E402

NAME = "`main`.`sales`.`orders`"


def _staged(client: object) -> pa.Table:
    (payload,) = client.files.uploaded.values()  # type: ignore[attr-defined]
    return pq.read_table(io.BytesIO(payload))


def test_dynamic_overwrite_treats_an_empty_partition_value_as_null() -> None:
    # LM-10: Databricks stores '' as the null partition, so `region = ''`
    # matched nothing it wrote and REPLACE WHERE's check refused the row.
    schema = pa.table({"id": pa.array([], pa.int64()), "region": pa.array([], pa.string())})
    eng, rec, _ = engine(RecordingBackend({"SELECT * FROM": schema}))
    data = pa.table({"id": [1, 2, 3], "region": ["", None, "eu"]})
    eng.overwrite(table(partition_columns=("region",)), data, partition_overwrite="dynamic")
    assert "REPLACE WHERE (`region` = :p0) OR (`region` IS NULL) SELECT" in rec.last
    assert [v for v, _ in rec.params.values()] == ["eu"]


def test_replace_where_lets_the_warehouse_fill_left_out_columns() -> None:
    # LM-13 / LM-12: a generated, identity or DEFAULT column left out of the
    # data was refused as "missing"; a column list lets Databricks compute it.
    schema = pa.table(
        {
            "id": pa.array([], pa.int64()),
            "ts": pa.array([], pa.timestamp("us", tz="UTC")),
            "day": pa.array([], pa.date32()),
        }
    )
    eng, rec, _ = engine(RecordingBackend({"SELECT * FROM": schema}))
    data = pa.table({"ts": pa.array([0], pa.timestamp("us", tz="UTC")), "id": [1]})
    eng.overwrite(table(), data, predicate="id = 1")
    assert rec.last.startswith(
        f"INSERT INTO {NAME} (`id`, `ts`) REPLACE WHERE id = 1 SELECT `id`, `ts` FROM read_files("
    )


def test_merge_insert_all_names_the_source_columns() -> None:
    # LM-13: `INSERT *` expands to every target column, so a source without
    # the identity column failed "cannot resolve id in INSERT clause".
    eng, rec, _ = engine()
    eng.merge(
        table(), pa.table({"city": ["q"]}), "target.city = source.city"
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    assert "THEN UPDATE SET `target`.`city` = `source`.`city`" in rec.last
    assert "THEN INSERT (`city`) VALUES (`source`.`city`)" in rec.last
    assert "*" not in rec.last


def _describe(**types: str) -> pa.Table:
    rows = [{"col_name": k, "data_type": v, "comment": None} for k, v in types.items()]
    rows += [
        {"col_name": "", "data_type": "", "comment": None},
        {"col_name": "# Partition Information", "data_type": "", "comment": None},
        {"col_name": "v", "data_type": "variant", "comment": None},
    ]
    return pa.Table.from_pylist(rows)


VARIANT_TABLE: dict[str, Any] = {
    "writer_features": frozenset({"variantType"}),
    "reader_features": frozenset({"variantType"}),
}


@pytest.mark.parametrize("how", ["append", "overwrite", "replace_where", "merge"])
def test_json_text_into_a_variant_column_is_parsed(how: str) -> None:
    # LM-08: a STRING staged into a VARIANT column became a string scalar, so a
    # read-then-append round trip turned every object into its JSON text.
    schema = pa.table({"id": pa.array([], pa.int64()), "v": pa.array([], pa.string())})
    backend = RecordingBackend(
        {"DESCRIBE TABLE": _describe(id="bigint", v="variant"), "SELECT * FROM": schema}
    )
    eng, rec, _ = engine(backend)
    data = pa.table({"id": [1], "v": ['{"a":1}']})
    t = table(**VARIANT_TABLE)
    if how == "append":
        eng.append(t, data)
    elif how == "overwrite":
        eng.overwrite(t, data)
    elif how == "replace_where":
        eng.overwrite(t, data, predicate="id = 1")
    else:
        eng.merge(t, data, "target.id = source.id").when_not_matched_insert_all().execute()
    assert "parse_json(`v`) AS `v`" in rec.last
    assert "parse_json(`id`)" not in rec.last
    assert rec.sql[0] == f"DESCRIBE TABLE {NAME}"


def test_variant_binary_is_staged_as_json_text() -> None:
    from deltaswamp._variant import encode, variant_column

    backend = RecordingBackend({"DESCRIBE TABLE": _describe(id="bigint", v="variant")})
    eng, rec, client = engine(backend)
    column = variant_column(pa, pa.array(['{"a":1}', None], pa.string()))
    eng.append(table(**VARIANT_TABLE), pa.table({"id": [1, 2], "v": column}))
    staged = _staged(client)
    assert staged.schema.field("v").type == pa.string()
    assert staged.column("v").to_pylist() == ['{"a":1}', None]
    assert encode('{"a":1}')  # the encoding round-trips through the decoder
    assert "parse_json(`v`) AS `v`" in rec.last


def test_a_table_without_variants_is_not_described() -> None:
    eng, rec, _ = engine()
    eng.append(table(), pa.table({"id": [1], "v": ["x"]}))
    assert not any(s.startswith("DESCRIBE") for s in rec.sql)
    assert "parse_json" not in rec.last


# ------------------------------------------------ LM-14: UPDATE values and paths


def test_update_sets_a_nested_field() -> None:
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("s", pa.struct([("a", pa.int32()), ("b", pa.string())])),
            ("x.y", pa.int64()),
        ]
    )
    eng, rec, _ = engine(RecordingBackend({"SELECT * FROM": schema.empty_table()}))
    eng.update(table(), new_values={"s.a": 6, "x.y": 1}, predicate="id = 1")
    assert rec.last.startswith(f"UPDATE {NAME} SET `s`.`a` = :p0, `x.y` = :p1 WHERE id = 1")
    eng.update(table(), updates={"S.A": "5"})
    assert rec.last == f"UPDATE {NAME} SET `S`.`A` = 5"


def test_update_renders_bytes_and_structs() -> None:
    eng, rec, _ = engine()
    eng.update(table(), new_values={"bin": b"\x01\x02", "s": {"a": 9, "b": "y'"}, "l": [1, None]})
    assert rec.last == (
        f"UPDATE {NAME} SET `bin` = X'0102', `s` = named_struct('a', :p0, 'b', :p1), "
        "`l` = array(:p2, NULL)"
    )
    assert rec.params["p1"] == ("y'", "STRING")


def test_update_with_an_unbindable_value_is_an_invalid_argument() -> None:
    from deltaswamp.errors import InvalidArgumentError

    eng, _, _ = engine()
    with pytest.raises(InvalidArgumentError):
        eng.update(table(), new_values={"x": object()})


# ------------------------------------------------ LM-08: the variant codec

#: (metadata, value) as Databricks wrote them, and its own to_json() of each.
DATABRICKS_VARIANTS = [
    (
        "AQUAAQIDBAVhYmNkZQ==",
        "AgMAAQIAAhMmDAEDBAABAggKBAAgARkAAAAFeAICAwQAAwwQ1P4YNRzc3wIAAAA=",
        '{"a":1,"b":[true,null,2.5,"x"],"c":{"d":-300,"e":12345678901}}',
    ),
    ("AQAA", "BXM=", '"s"'),
    ("AQAA", "JBFO8zCmS5u2AQ==", "1.23456789012345678"),
    (
        "AQYAAgQHCgsOZHR0c2RlY2JpbmZudHo=",
        "AgYDAgAEBQEUDgAbIAUpLOdPAAAwgNVO9l5HBgAgApYAAAA8AgAAAAECOAAAwD80QPNM9l5HBgA=",
        '{"bin":"AQI=","dec":1.5,"dt":"2026-01-02","f":1.5,"ntz":"2026-01-02 03:04:05",'
        '"ts":"2026-01-02 03:04:05.123456+00:00"}',
    ),
]


@pytest.mark.parametrize(("metadata", "value", "text"), DATABRICKS_VARIANTS)
def test_variant_decodes_as_databricks_renders_it(metadata: str, value: str, text: str) -> None:
    import base64

    from deltaswamp._variant import encode, to_json

    assert to_json(base64.b64decode(metadata), base64.b64decode(value)) == text
    assert to_json(*encode(text)) == text
