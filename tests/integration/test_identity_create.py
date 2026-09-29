"""Creating tables with identity and default columns locally, then writing them."""

from __future__ import annotations

import glob
import json
import pickle
from typing import Any
from urllib.parse import urlparse

import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.errors import DeltaSwampError  # noqa: E402

from tests.fake_uc import FakeUnityCatalog  # noqa: E402
from tests.helpers import direct_router  # noqa: E402


def _identity(start: int = 1, step: int = 1, explicit: bool = False) -> dict[str, str]:
    return {
        "delta.identity.start": str(start),
        "delta.identity.step": str(step),
        "delta.identity.allowExplicitInsert": str(explicit).lower(),
    }


def _schema(**identity: Any) -> Any:
    return pa.schema(
        [
            pa.field("id", pa.int64(), metadata=_identity(**identity)),
            ("v", pa.string()),
            pa.field("d", pa.string(), metadata={"CURRENT_DEFAULT": "'x'"}),
        ]
    )


def _v0(path: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with open(sorted(glob.glob(path + "/_delta_log/*.json"))[0]) as f:
        for line in f:
            out.update(json.loads(line))
    return out


@pytest.fixture
def conn() -> Any:
    return ds.connect()


class TestCreate:
    def test_version_zero_declares_the_features_as_databricks_does(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, _schema(start=5, step=-2, explicit=True))
        assert sorted(glob.glob(path + "/_delta_log/*.json")) == [
            path + "/_delta_log/00000000000000000000.json"
        ]
        v0 = _v0(path)
        assert v0["protocol"]["minWriterVersion"] == 7
        assert {"identityColumns", "allowColumnDefaults"} <= set(v0["protocol"]["writerFeatures"])
        fields = {f["name"]: f for f in json.loads(v0["metaData"]["schemaString"])["fields"]}
        # Typed as Spark reads them: getLong, getBoolean.
        assert fields["id"]["metadata"] == {
            "delta.identity.start": 5,
            "delta.identity.step": -2,
            "delta.identity.allowExplicitInsert": True,
        }
        assert fields["d"]["metadata"] == {"CURRENT_DEFAULT": "'x'"}

    def test_appends_and_distributed_writes_fill_both(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, _schema(start=1, step=1))
        conn.open_table(path).append(pa.table({"v": ["a", "b"]}))
        plan = conn.open_table(path).plan_write(identity_tasks=2, identity_rows_per_task=10)
        worker = pickle.loads(pickle.dumps(plan))
        plan.commit(
            [
                worker.write(pa.table({"v": ["c"]}), task_index=0),
                worker.write(pa.table({"v": ["e"]}), task_index=1),
            ]
        )
        rows = conn.open_table(path).to_arrow().to_pylist()
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == 4 and {1, 2} <= set(ids)
        assert {r["d"] for r in rows} == {"x"}

    def test_generated_always_refuses_a_given_value(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, _schema(explicit=False))
        with pytest.raises(DeltaSwampError, match="GENERATED ALWAYS"):
            conn.open_table(path).append(pa.table({"id": [9], "v": ["a"]}))

    @pytest.mark.parametrize(
        ("field", "match"),
        [
            (pa.field("id", pa.int32(), metadata=_identity()), "BIGINT"),
            (pa.field("id", pa.int64(), metadata=_identity(step=0)), "step cannot be 0"),
            (
                pa.field(
                    "id", pa.int64(), metadata={**_identity(), "delta.identity.highWaterMark": "3"}
                ),
                "high-water mark",
            ),
            (
                pa.field("id", pa.int64(), metadata={**_identity(), "delta.identity.start": "x"}),
                "not an integer",
            ),
        ],
    )
    def test_invalid_declarations_are_refused_before_anything_is_written(
        self, conn: Any, tmp_path: Any, field: Any, match: str
    ) -> None:
        path = tmp_path / "t"
        with pytest.raises(DeltaSwampError, match=match):
            conn.create_table(str(path), pa.schema([field, ("v", pa.string())]))
        assert not glob.glob(str(path / "_delta_log" / "*"))

    def test_an_existing_table_is_not_replaced(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        conn.create_table(path, _schema())
        with pytest.raises(DeltaSwampError):
            conn.create_table(path, _schema(start=100))
        assert (
            json.loads(_v0(path)["metaData"]["schemaString"])["fields"][0]["metadata"][
                "delta.identity.start"
            ]
            == 1
        )


def test_a_managed_table_types_its_identity_columns(tmp_path: Any) -> None:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog

    with FakeUnityCatalog(staging_root=tmp_path / "managed") as uc:
        catalog = OSSUnityCatalog(uc.url)
        catalog.create_catalog("main")
        catalog.create_schema("main", "sales")
        conn = ds.Connection(catalog=catalog, router=direct_router())
        t = conn.create_table("main.sales.ids", _schema(start=1, step=1))
        location = t.resolved.location
        v0 = _v0(urlparse(location).path if location.startswith("file:") else location)
        fields = {f["name"]: f for f in json.loads(v0["metaData"]["schemaString"])["fields"]}
        assert fields["id"]["metadata"]["delta.identity.start"] == 1
        assert fields["id"]["metadata"]["delta.identity.allowExplicitInsert"] is False
        assert "identityColumns" in v0["protocol"]["writerFeatures"]
