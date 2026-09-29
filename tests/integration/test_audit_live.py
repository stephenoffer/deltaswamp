"""Regressions for the live-matrix audit (LM-*), reproduced on local tables.

The defects were found against a Databricks warehouse; each test here builds
the same table shape on disk and failed before its fix.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
ds = pytest.importorskip("deltaswamp")


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _write_log(path: str, schema: dict[str, Any], protocol: dict[str, Any], data: Any) -> None:
    """A one-commit table written by hand, for shapes delta-rs cannot create."""
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
                "configuration": {},
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
            }
        },
    ]
    with open(os.path.join(path, "_delta_log", f"{0:020}.json"), "w") as f:
        f.write("\n".join(json.dumps(a) for a in actions))


class TestCollatedPredicates:
    """LM-04: every predicate on a collated table was refused on the direct path."""

    def _table(self, conn: Any, tmp_path: Any) -> Any:
        path = str(tmp_path / "coll")
        schema = {
            "type": "struct",
            "fields": [
                {"name": "id", "type": "long", "nullable": True, "metadata": {}},
                {
                    "name": "name",
                    "type": "string",
                    "nullable": True,
                    "metadata": {"__COLLATIONS": {"name": "spark.UTF8_LCASE"}},
                },
            ],
        }
        protocol = {"minReaderVersion": 1, "minWriterVersion": 7, "writerFeatures": ["collations"]}
        _write_log(path, schema, protocol, pa.table({"id": [1, 2], "name": ["Oslo", "oslo"]}))
        return conn.open_table(path)

    def test_predicate_on_an_uncollated_column_reads_directly(
        self, conn: Any, tmp_path: Any
    ) -> None:
        t = self._table(conn, tmp_path)
        assert t.can("scan", predicate="id = 1").ok
        assert pa.table(t.to_arrow(predicate="id = 1")).column("id").to_pylist() == [1]

    def test_predicate_on_a_collated_column_is_still_refused(
        self, conn: Any, tmp_path: Any
    ) -> None:
        t = self._table(conn, tmp_path)
        assert not t.can("scan", predicate="name = 'oslo'").ok
        assert not t.can("scan", predicate="id = 1 OR name = 'oslo'").ok


_VARIANT_SCHEMA = {
    "type": "struct",
    "fields": [
        {"name": "id", "type": "long", "nullable": True, "metadata": {}},
        {"name": "v", "type": "variant", "nullable": True, "metadata": {}},
    ],
}


class TestVariantColumns:
    """LM-08/LM-09: one representation for VARIANT, whichever engine reads it."""

    def _table(self, conn: Any, tmp_path: Any) -> Any:
        from deltaswamp._variant import variant_column

        path = str(tmp_path / "variant")
        column = variant_column(pa, pa.array(['{"a":1,"b":[true,null]}', '"s"', None]))
        protocol = {
            "minReaderVersion": 3,
            "minWriterVersion": 7,
            "readerFeatures": ["variantType"],
            "writerFeatures": ["variantType"],
        }
        _write_log(path, _VARIANT_SCHEMA, protocol, pa.table({"id": [1, 2, 3], "v": column}))
        return conn.open_table(path)

    def test_reads_give_json_text(self, conn: Any, tmp_path: Any) -> None:
        t = self._table(conn, tmp_path)
        got = pa.table(t.to_arrow())
        assert got.schema.field("v").type == pa.string()
        assert t.schema().field("v").type == pa.string()
        assert got.sort_by("id").column("v").to_pylist() == ['{"a":1,"b":[true,null]}', '"s"', None]

    def test_json_text_round_trips(self, conn: Any, tmp_path: Any) -> None:
        t = self._table(conn, tmp_path)
        t.append(pa.table(t.to_arrow()).set_column(0, "id", pa.array([4, 5, 6], pa.int64())))
        t.append(pa.table({"id": [7], "v": ['{"z": {"y": 2.5}}']}))
        t.append(pa.table({"id": [8]}))  # the VARIANT column left out: a null
        rows = {
            r["id"]: r["v"]
            for r in pa.table(conn.open_table(t.resolved.location).to_arrow()).to_pylist()
        }
        assert rows[4] == '{"a":1,"b":[true,null]}' and rows[5] == '"s"' and rows[6] is None
        assert rows[7] == '{"z":{"y":2.5}}'
        assert rows[8] is None

    def test_invalid_json_is_refused(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.errors import InvalidArgumentError

        t = self._table(conn, tmp_path)
        with pytest.raises(InvalidArgumentError, match="JSON"):
            t.append(pa.table({"id": [9], "v": ["{not json"]}))


class TestShreddedVariants:
    """LM-09: can(SCAN) said kernel, then the read raised a raw ArrowInvalid."""

    def _table(self, conn: Any, tmp_path: Any, shredding: bool) -> Any:
        from deltaswamp._variant import encode

        path = str(tmp_path / "shredded")
        metadata, _ = encode('{"a":1}')
        typed = pa.struct(
            [
                pa.field(
                    "a",
                    pa.struct(
                        [pa.field("value", pa.binary()), pa.field("typed_value", pa.int64())]
                    ),
                    nullable=False,
                )
            ]
        )
        shredded = pa.struct(
            [
                pa.field("metadata", pa.binary(), nullable=False),
                pa.field("value", pa.binary()),
                pa.field("typed_value", typed),
            ]
        )
        v = pa.array(
            [{"metadata": metadata, "value": None, "typed_value": {"a": {"typed_value": 1}}}],
            shredded,
        )
        features = ["variantType", "variantShredding"]
        protocol = {
            "minReaderVersion": 3,
            "minWriterVersion": 7,
            "readerFeatures": features,
            "writerFeatures": features,
        }
        _write_log(
            path, _VARIANT_SCHEMA, protocol, pa.table({"id": pa.array([1], pa.int64()), "v": v})
        )
        if shredding:
            log = pathlib.Path(path, "_delta_log", f"{0:020}.json")
            log.write_text(
                log.read_text().replace(
                    '"configuration": {}',
                    '"configuration": {"delta.enableVariantShredding": "true"}',
                )
            )
        return conn.open_table(path)

    def test_a_shredding_table_is_refused_before_the_read(self, conn: Any, tmp_path: Any) -> None:
        from deltaswamp.errors import DeltaSwampError

        t = self._table(conn, tmp_path, shredding=True)
        verdict = t.can("scan")
        assert not verdict.ok and "shred" in verdict.reason
        with pytest.raises(DeltaSwampError, match="shred"):
            t.to_arrow()
        # A read that opens no VARIANT column still goes direct.
        assert t.can("scan", columns=["id"]).ok
        assert pa.table(t.to_arrow(columns=["id"])).column("id").to_pylist() == [1]
        assert t.count() == 1

    def test_a_shredded_file_found_mid_read_is_a_library_error(
        self, conn: Any, tmp_path: Any
    ) -> None:
        from deltaswamp.errors import EngineLimitError

        t = self._table(conn, tmp_path, shredding=False)
        with pytest.raises(EngineLimitError, match="shredded"):
            t.to_arrow()


class TestColumnDefaults:
    """LM-12: a left-out column with a DEFAULT was written as NULL."""

    def _table(self, conn: Any, tmp_path: Any, default: str) -> Any:
        path = str(tmp_path / "defaults")
        schema = {
            "type": "struct",
            "fields": [
                {"name": "id", "type": "long", "nullable": True, "metadata": {}},
                {
                    "name": "city",
                    "type": "string",
                    "nullable": True,
                    "metadata": {"CURRENT_DEFAULT": default},
                },
                {
                    "name": "n",
                    "type": "integer",
                    "nullable": True,
                    "metadata": {"CURRENT_DEFAULT": "42"},
                },
            ],
        }
        protocol = {
            "minReaderVersion": 1,
            "minWriterVersion": 7,
            "writerFeatures": ["allowColumnDefaults"],
        }
        data = pa.table({"id": [1], "city": ["x"], "n": pa.array([1], pa.int32())})
        _write_log(path, schema, protocol, data)
        return conn.open_table(path)

    def test_a_literal_default_is_applied(self, conn: Any, tmp_path: Any) -> None:
        t = self._table(conn, tmp_path, "'new'")
        t.append(pa.table({"id": [2]}))
        t.overwrite(pa.table({"id": [1]}), predicate="id = 1")
        rows = sorted(pa.table(t.to_arrow()).to_pylist(), key=lambda r: r["id"])
        assert rows == [{"id": 1, "city": "new", "n": 42}, {"id": 2, "city": "new", "n": 42}]

    def test_an_expression_default_is_refused_on_a_direct_engine(
        self, conn: Any, tmp_path: Any
    ) -> None:
        from deltaswamp.errors import DeltaSwampError

        t = self._table(conn, tmp_path, "concat('a', 'b')")
        with pytest.raises(DeltaSwampError, match="sql_column_defaults"):
            t.append(pa.table({"id": [2]}))
        t.append(pa.table({"id": [3], "city": ["given"]}))  # nothing to default
        assert sorted(pa.table(t.to_arrow()).column("id").to_pylist()) == [1, 3]
