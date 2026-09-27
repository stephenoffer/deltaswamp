"""Regressions for the third live re-verification (LV-*, N*, C*), the parts that need no warehouse.

Each reproduces a statement or verdict a real Databricks warehouse
disagreed with.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from deltaswamp.capability import Operation  # noqa: E402
from deltaswamp.engine.sql import SqlEngine, _variant_tree  # noqa: E402

from tests.helpers import resolved_table as table  # noqa: E402
from tests.unit.sql_fakes import RecordingBackend, engine  # noqa: E402

VARIANT_TABLE: dict[str, Any] = {
    "writer_features": frozenset({"variantType"}),
    "reader_features": frozenset({"variantType"}),
}


def _describe(**types: str) -> Any:
    rows = [{"col_name": k, "data_type": v, "comment": None} for k, v in types.items()]
    return pa.Table.from_pylist(rows)


def _staged(client: Any) -> Any:
    (payload,) = client.files.uploaded.values()
    return pq.read_table(io.BytesIO(payload))


# ------------------------------------------------ LV-01/LV-03: nested VARIANT


def test_describe_type_strings_locate_nested_variants() -> None:
    assert _variant_tree("variant") == "variant"
    assert _variant_tree("int") is None
    assert _variant_tree("struct<a:int,b:string>") is None
    assert _variant_tree("struct<v2:variant,n:int>") == ("struct", {"v2": "variant"})
    assert _variant_tree("array<variant>") == ("array", "variant")
    assert _variant_tree("map<string,variant>") == ("map", "variant")
    nested = (
        "struct<`a b`:array<variant>,c:decimal(10,2),d:map<string,int> NOT NULL COMMENT 'x, >'>"
    )
    assert _variant_tree(nested) == ("struct", {"a b": ("array", "variant")})


NESTED = {
    "id": "bigint",
    "s": "struct<v2:variant,n:int>",
    "arr": "array<variant>",
    "m": "map<string,variant>",
}


def _nested_text() -> Any:
    return pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "s": pa.array(
                [{"v2": '{"a":1}', "n": 1}, None],
                pa.struct([("v2", pa.string()), ("n", pa.int32())]),
            ),
            "arr": pa.array([["[1,2]"], None], pa.list_(pa.string())),
            "m": pa.array([[("k", '{"q":true}')], None], pa.map_(pa.string(), pa.string())),
        }
    )


def test_nested_json_text_is_parsed_into_the_variant_fields() -> None:
    # The warehouse's own text for a nested VARIANT, appended back, was stored
    # as STRING scalars: only top-level columns were run through parse_json.
    eng, rec, _ = engine(RecordingBackend({"DESCRIBE TABLE": _describe(**NESTED)}))
    eng.append(table(**VARIANT_TABLE), _nested_text())
    sql = rec.last
    assert (
        "CASE WHEN `s` IS NULL THEN NULL ELSE named_struct('v2', parse_json(`s`.`v2`), "
        "'n', `s`.`n`) END AS `s`" in sql
    )
    assert "transform(`arr`, __e0 -> parse_json(__e0)) AS `arr`" in sql
    assert "transform_values(`m`, (__k0, __v0) -> parse_json(__v0)) AS `m`" in sql
    assert "parse_json(`id`)" not in sql


def test_nested_variant_binary_is_staged_as_text_even_under_a_null_parent() -> None:
    # A padded (null) struct holding a VARIANT failed staging with a raw
    # ArrowInvalid ("metadata is declared non-nullable but contains nulls").
    from deltaswamp._variant import encode, variant_type

    vt = variant_type(pa)

    def value(text: str) -> dict[str, bytes]:
        m, v = encode(text)
        return {"metadata": m, "value": v}

    data = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "s": pa.array(
                [{"v2": value('{"a":1}'), "n": 1}, None],
                pa.struct([pa.field("v2", vt), pa.field("n", pa.int32())]),
            ),
            "arr": pa.array([[value("[1]")], None], pa.list_(vt)),
            "m": pa.array([[("k", value("true"))], None], pa.map_(pa.string(), vt)),
        }
    )
    eng, _, client = engine(RecordingBackend({"DESCRIBE TABLE": _describe(**NESTED)}))
    eng.append(table(**VARIANT_TABLE), data)
    staged = _staged(client).to_pylist()
    assert staged[0]["s"] == {"v2": '{"a":1}', "n": 1}
    assert staged[0]["arr"] == ["[1]"]
    assert staged[0]["m"] == [("k", "true")]
    assert staged[1]["s"] is None and staged[1]["arr"] is None


# ------------------------------------------------ N1: MERGE *_ALL, extra source columns


def test_merge_all_clauses_skip_source_columns_the_target_lacks() -> None:
    schema = pa.table({"id": pa.array([], pa.int64()), "v": pa.array([], pa.string())})
    eng, rec, _ = engine(RecordingBackend({"SELECT * FROM": schema}))
    source = pa.table({"id": [1], "v": ["x"], "extra": [9]})
    eng.merge(
        table(), source, "target.id = source.id"
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    assert "`target`.`extra`" not in rec.last
    assert "INSERT (`id`, `v`) VALUES (`source`.`id`, `source`.`v`)" in rec.last


def test_merge_schema_still_sets_every_source_column() -> None:
    eng, rec, _ = engine()
    source = pa.table({"id": [1], "extra": [9]})
    eng.merge(
        table(), source, "target.id = source.id", merge_schema=True
    ).when_not_matched_insert_all().execute()
    assert "THEN INSERT *" in rec.last


# ------------------------------------------------ C1/C2/C4: can() honesty on the SQL path


def test_cluster_by_on_a_partitioned_table_is_refused() -> None:
    verdict = SqlEngine._table_type_refusal(
        Operation.CLUSTER_BY, table(partition_columns=("city",))
    )
    assert verdict is not None and "partitioned" in verdict.reason


def test_cdf_needs_the_change_feed_or_row_tracking() -> None:
    bare = table(writer_features=frozenset({"deletionVectors"}))
    verdict = SqlEngine._table_type_refusal(Operation.CDF, bare)
    assert verdict is not None and "enableChangeDataFeed" in verdict.reason
    tracked = table(
        writer_features=frozenset({"rowTracking"}),
        properties={"delta.enableRowTracking": "true"},
    )
    assert SqlEngine._table_type_refusal(Operation.CDF, tracked) is None
    fed = table(properties={"delta.enableChangeDataFeed": "true"})
    assert SqlEngine._table_type_refusal(Operation.CDF, fed) is None


def test_drop_feature_is_judged_per_feature() -> None:
    eng, _, _ = engine(warehouse_id="w")
    t = table(
        writer_features=frozenset({"deletionVectors", "typeWidening-preview", "appendOnly"}),
        reader_features=frozenset({"deletionVectors"}),
    )
    assert eng.supports(Operation.DROP_FEATURE, t, feature="deletionVectors").ok
    assert eng.supports(Operation.DROP_FEATURE, t, feature="typeWidening").ok
    refused = eng.supports(Operation.DROP_FEATURE, t, feature="appendOnly")
    assert not refused.ok and "NONREMOVABLE" in refused.reason
    absent = eng.supports(Operation.DROP_FEATURE, t, feature="v2Checkpoint")
    assert not absent.ok and "does not have" in absent.reason


# ------------------------------------------------ minor


def test_generated_partition_values_come_from_the_warehouse() -> None:
    # C5: a dynamic overwrite whose data leaves out a generated partition
    # column was refused "the data has no day column".
    import datetime as dt

    target = pa.table({"ts": pa.array([], pa.timestamp("us", tz="UTC"))})
    backend = RecordingBackend(
        {
            "SELECT DISTINCT": pa.table({"day": [dt.date(2026, 1, 1)]}),
            "SELECT * FROM": pa.table(
                {"ts": target.column("ts"), "day": pa.array([], pa.date32())}
            ),
        }
    )
    import json
    from types import SimpleNamespace

    eng, rec, client = engine(backend)
    field = {"metadata": {"delta.generationExpression": "CAST(ts AS DATE)"}}
    columns = [
        SimpleNamespace(name="ts", type_json="{}"),
        SimpleNamespace(name="day", type_json=json.dumps(field)),
    ]
    client.tables = SimpleNamespace(  # type: ignore[attr-defined]
        get=lambda name: SimpleNamespace(columns=columns)
    )
    t = table(partition_columns=("day",))
    data = pa.table({"ts": pa.array([0], pa.timestamp("us", tz="UTC"))})
    eng.overwrite(t, data, partition_overwrite="dynamic")
    assert any("SELECT DISTINCT (CAST(ts AS DATE)) AS `day` FROM read_files(" in s for s in rec.sql)
    assert "REPLACE WHERE (`day` = :p0) SELECT" in rec.last
