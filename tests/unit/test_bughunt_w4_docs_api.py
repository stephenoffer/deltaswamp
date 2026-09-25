"""Wave-4 docs/API audit, unit level: engine parity without a warehouse."""

from __future__ import annotations

import datetime as dt
import pickle

import pytest

pa = pytest.importorskip("pyarrow")

import deltaswamp as ds  # noqa: E402
from deltaswamp.errors import DeltaSwampError, InvalidArgumentError  # noqa: E402

from tests.unit.test_sql_engine import RecordingBackend, engine, table  # noqa: E402

# ------------------------------------------------------------ SQL parity


def test_warehouse_history_entries_match_delta_rs_shape() -> None:
    rows = pa.table(
        {
            "version": pa.array([3], pa.int64()),
            "timestamp": pa.array(
                [dt.datetime(2026, 1, 1, tzinfo=dt.UTC)], pa.timestamp("us", "UTC")
            ),
            "operation": ["WRITE"],
            "operationParameters": pa.array(
                [[("mode", "Append")]], pa.map_(pa.string(), pa.string())
            ),
            "operationMetrics": pa.array([[("numFiles", "1")]], pa.map_(pa.string(), pa.string())),
        }
    )
    eng, _, _ = engine(RecordingBackend({"DESCRIBE HISTORY": rows}))
    (entry,) = eng.history(table())
    assert entry["operationParameters"] == {"mode": "Append"}
    assert entry["operationMetrics"]["numFiles"] == "1"
    # epoch milliseconds, as delta-rs reports and scan(timestamp=) takes back
    assert entry["timestamp"] == 1767225600000
    assert entry["version"] == 3


def test_warehouse_delete_and_update_report_the_delta_rs_metric_names() -> None:
    affected = pa.table({"num_affected_rows": pa.array([4], pa.int64())})
    eng, _, _ = engine(RecordingBackend({"DELETE": affected, "UPDATE": affected}))
    assert eng.delete(table(), "id = 1")["num_deleted_rows"] == 4
    assert eng.update(table(), new_values={"a": 1})["num_updated_rows"] == 4


def test_warehouse_merge_metrics_have_delta_rs_aliases() -> None:
    from deltaswamp.engine.sql import _dml_metrics

    out = _dml_metrics(
        {
            "num_affected_rows": 3,
            "num_updated_rows": 1,
            "num_deleted_rows": 0,
            "num_inserted_rows": 2,
        },
        None,
    )
    assert out["num_target_rows_updated"] == 1
    assert out["num_target_rows_inserted"] == 2
    assert out["num_target_rows_deleted"] == 0


def test_engine_misuse_errors_are_deltaswamp_errors() -> None:
    eng, _, _ = engine()
    with pytest.raises(DeltaSwampError):
        eng.zorder(table(), [])
    with pytest.raises(ValueError):  # still a ValueError for existing callers
        eng.zorder(table(), [])
    from deltaswamp.engine.deltars import DeltaRsEngine

    with pytest.raises(InvalidArgumentError):
        DeltaRsEngine().zorder(table(), [])


# ---------------------------------------------------------- picklability


def test_unity_catalog_http_error_survives_pickling() -> None:
    from deltaswamp.catalog.ossuc import UnityCatalogHTTPError

    err = pickle.loads(pickle.dumps(UnityCatalogHTTPError("not found", 404)))
    assert isinstance(err, UnityCatalogHTTPError)
    assert err.status == 404 and str(err) == "not found"


# --------------------------------------------------------------- connect


def test_connect_refuses_a_uri_and_a_catalog_together() -> None:
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    with pytest.raises(InvalidArgumentError, match="not both"):
        ds.connect("hms://thrift://metastore:9083", catalog=FilesystemCatalog())


def test_non_string_uri_error_names_an_api_that_exists() -> None:
    import pathlib

    with pytest.raises(ds.InvalidReferenceError) as info:
        ds.connect(pathlib.Path("/tmp/x"))  # type: ignore[arg-type]
    assert "conn.path(" not in str(info.value)
    assert "open_table" in str(info.value)
    assert hasattr(ds.Connection, "open_table")


# ------------------------------------------------------------------ docs


def test_native_stub_lists_every_feature_the_build_reports() -> None:
    import pathlib

    if not ds.has_native():
        pytest.skip("native extension not built")
    from deltaswamp import _native

    stub = pathlib.Path(ds.__file__).with_name("_native.pyi").read_text()
    doc = stub.split("FEATURES: list[str]", 1)[1].split('"""', 2)[1]
    for feature in _native.FEATURES:
        assert f'"{feature}"' in doc, feature


def test_usage_errors_table_names_every_public_error() -> None:
    import pathlib

    usage = (pathlib.Path(ds.__file__).parents[2] / "docs" / "usage.md").read_text()
    for name in ds.__all__:
        obj = getattr(ds, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, DeltaSwampError)
            and obj is not DeltaSwampError
        ):
            assert f"| `{name}` |" in usage, name
