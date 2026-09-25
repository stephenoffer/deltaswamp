"""Regression tests for defects found auditing the delta-rs engine."""

from __future__ import annotations

import datetime as dt
import time
from decimal import Decimal
from typing import Any, Literal, cast

import pytest

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")

from deltalake import DeltaTable, write_deltalake  # noqa: E402
from deltaswamp.capability import Operation  # noqa: E402
from deltaswamp.catalog.base import ResolvedTable  # noqa: E402
from deltaswamp.engine import deltars  # noqa: E402
from deltaswamp.engine.deltars import DeltaRsEngine  # noqa: E402
from deltaswamp.errors import EnginePanicError, UnreachableTableError  # noqa: E402
from deltaswamp.identity import RefKind, TableRef  # noqa: E402


def _resolved(path: str, **kw: Any) -> ResolvedTable:
    return ResolvedTable(ref=TableRef(kind=RefKind.PATH, path=path), location=path, **kw)


def _rows(path: str) -> list[dict[str, Any]]:
    return sorted(pa.table(DeltaTable(path).scan()).to_pylist(), key=repr)


@pytest.fixture
def engine() -> DeltaRsEngine:
    return DeltaRsEngine()


@pytest.fixture
def parts(tmp_path: Any) -> str:
    path = str(tmp_path / "parts")
    write_deltalake(
        path,
        pa.table({"id": [1, 2, 3], "region": ["eu", "us", "apac"]}),
        partition_by=["region"],
    )
    return path


@pytest.fixture
def plain(tmp_path: Any) -> str:
    path = str(tmp_path / "plain")
    write_deltalake(path, pa.table({"id": [1, 2], "name": ["a", "b"]}))
    return path


# ------------------------------------------------------- dynamic overwrite


def test_dynamic_overwrite_with_stream_does_not_lose_data(
    engine: DeltaRsEngine, parts: str
) -> None:
    src = pa.table({"id": [9], "region": ["eu"]})
    reader = pa.RecordBatchReader.from_batches(src.schema, src.to_batches())
    engine.overwrite(
        _resolved(parts, partition_columns=("region",)), reader, partition_overwrite="dynamic"
    )
    assert _rows(parts) == sorted(
        [{"id": 9, "region": "eu"}, {"id": 2, "region": "us"}, {"id": 3, "region": "apac"}],
        key=repr,
    )


def test_dynamic_overwrite_with_null_partition(engine: DeltaRsEngine, tmp_path: Any) -> None:
    path = str(tmp_path / "n")
    write_deltalake(
        path, pa.table({"id": [1, 2, 3], "region": ["eu", None, "us"]}), partition_by=["region"]
    )
    data = pa.table({"id": [8, 9], "region": pa.array([None, "eu"], pa.string())})
    engine.overwrite(
        _resolved(path, partition_columns=("region",)), data, partition_overwrite="dynamic"
    )
    assert _rows(path) == sorted(
        [{"id": 8, "region": None}, {"id": 9, "region": "eu"}, {"id": 3, "region": "us"}],
        key=repr,
    )


def test_dynamic_overwrite_quotes_column_names(engine: DeltaRsEngine, tmp_path: Any) -> None:
    path = str(tmp_path / "q")
    write_deltalake(path, pa.table({"id": [1, 2], "my reg": ["a", "b"]}), partition_by=["my reg"])
    engine.overwrite(
        _resolved(path, partition_columns=("my reg",)),
        pa.table({"id": [9], "my reg": ["a"]}),
        partition_overwrite="dynamic",
    )
    assert _rows(path) == sorted([{"id": 9, "my reg": "a"}, {"id": 2, "my reg": "b"}], key=repr)


def test_dynamic_overwrite_nan_partition(engine: DeltaRsEngine, tmp_path: Any) -> None:
    path = str(tmp_path / "f")
    write_deltalake(path, pa.table({"id": [1, 2], "f": [float("nan"), 1.5]}), partition_by=["f"])
    engine.overwrite(
        _resolved(path, partition_columns=("f",)),
        pa.table({"id": [9], "f": [float("nan")]}),
        partition_overwrite="dynamic",
    )
    ids = sorted(r["id"] for r in _rows(path))
    assert ids == [2, 9]


def test_sql_literal_bytes_is_hex() -> None:
    assert deltars._sql_literal(b"hi") == "X'6869'"


def test_dynamic_overwrite_accepts_batch_list(engine: DeltaRsEngine, parts: str) -> None:
    batches = pa.table({"id": [9], "region": ["eu"]}).to_batches()
    engine.overwrite(
        _resolved(parts, partition_columns=("region",)), batches, partition_overwrite="dynamic"
    )
    assert {"id": 9, "region": "eu"} in _rows(parts)


# ------------------------------------------------------------------ pandas


def test_pandas_index_is_not_written_as_a_column(engine: DeltaRsEngine, plain: str) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"id": [3], "name": ["c"]}, index=[7])
    engine.append(_resolved(plain), df)
    assert "__index_level_0__" not in DeltaTable(plain).schema().to_arrow().names
    assert len(_rows(plain)) == 3


# --------------------------------------------------------- update values


def test_update_new_values_escapes_quotes(engine: DeltaRsEngine, plain: str) -> None:
    engine.update(_resolved(plain), new_values={"name": "O'Brien"}, predicate="id = 1")
    assert {"id": 1, "name": "O'Brien"} in _rows(plain)


def test_update_new_values_none_sets_null(engine: DeltaRsEngine, plain: str) -> None:
    engine.update(_resolved(plain), new_values={"name": None}, predicate="id = 1")
    assert {"id": 1, "name": None} in _rows(plain)


def test_update_naive_datetime_not_shifted_by_local_tz(
    engine: DeltaRsEngine, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        path = str(tmp_path / "ts")
        write_deltalake(
            path,
            pa.table({"id": [1], "ts": pa.array([dt.datetime(2020, 1, 1)], pa.timestamp("us"))}),
        )
        when = dt.datetime(2021, 2, 2, 3, 4, 5, 7)
        engine.update(_resolved(path), new_values={"ts": when}, predicate="id = 1")
        assert _rows(path)[0]["ts"] == when
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_update_new_values_date_decimal_nan(engine: DeltaRsEngine, tmp_path: Any) -> None:
    path = str(tmp_path / "types")
    write_deltalake(
        path,
        pa.table(
            {
                "id": [1],
                "d": [dt.date(2020, 1, 1)],
                "dec": pa.array([Decimal("1.10")], pa.decimal128(5, 2)),
                "f": [1.0],
            }
        ),
    )
    engine.update(
        _resolved(path),
        new_values={"d": dt.date(2021, 2, 2), "dec": Decimal("2.25"), "f": float("nan")},
        predicate="id = 1",
    )
    row = _rows(path)[0]
    assert row["d"] == dt.date(2021, 2, 2)
    assert row["dec"] == Decimal("2.25")
    assert row["f"] != row["f"]  # NaN


def test_update_with_nothing_to_set_is_refused(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(ValueError, match="at least one column"):
        engine.update(_resolved(plain), updates={}, predicate="id = 1")


def test_update_accepts_error_on_type_mismatch(engine: DeltaRsEngine, plain: str) -> None:
    engine.update(
        _resolved(plain), updates={"name": "'x'"}, predicate="id = 1", error_on_type_mismatch=False
    )
    assert {"id": 1, "name": "x"} in _rows(plain)


def test_delete_blank_predicate_is_refused(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        engine.delete(_resolved(plain), "  ")
    assert len(_rows(plain)) == 2


# ------------------------------------------------------------ time travel


def test_scan_refuses_version_and_timestamp(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(UnreachableTableError, match="not both"):
        engine.scan(_resolved(plain), version=0, timestamp="2020-01-01T00:00:00Z")


def test_scan_accepts_datetime_and_naive_timestamp(engine: DeltaRsEngine, plain: str) -> None:
    later = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
    for ts in (later, later.replace(tzinfo=None).isoformat(), later.date().isoformat()):
        assert pa.table(engine.scan(_resolved(plain), timestamp=ts)).num_rows == 2  # type: ignore[arg-type]


@pytest.fixture
def cdf_table(tmp_path: Any) -> tuple[str, dt.datetime]:
    path = str(tmp_path / "cdf")
    write_deltalake(
        path, pa.table({"id": [1]}), configuration={"delta.enableChangeDataFeed": "true"}
    )
    time.sleep(1.1)
    mid = dt.datetime.now(dt.UTC)
    time.sleep(1.1)
    write_deltalake(path, pa.table({"id": [2]}), mode="append")
    write_deltalake(path, pa.table({"id": [3]}), mode="append")
    return path, mid


def _cdf_props() -> dict[str, str]:
    return {"delta.enableChangeDataFeed": "true"}


def test_cdf_accepts_datetime_bounds(
    engine: DeltaRsEngine, cdf_table: tuple[str, dt.datetime]
) -> None:
    path, mid = cdf_table
    out = pa.table(engine.cdf(_resolved(path, properties=_cdf_props()), starting_timestamp=mid))  # type: ignore[arg-type]
    assert sorted(out.column("id").to_pylist()) == [2, 3]


def test_cdf_refuses_version_and_timestamp_on_same_bound(
    engine: DeltaRsEngine, cdf_table: tuple[str, dt.datetime]
) -> None:
    path, mid = cdf_table
    with pytest.raises(UnreachableTableError, match="start version and a start timestamp"):
        engine.cdf(
            _resolved(path, properties=_cdf_props()),
            starting_version=2,
            starting_timestamp=mid,  # type: ignore[arg-type]
        )


def test_restore_refuses_bool(engine: DeltaRsEngine, plain: str) -> None:
    write_deltalake(plain, pa.table({"id": [3], "name": ["c"]}), mode="append")
    with pytest.raises(TypeError):
        engine.restore(_resolved(plain), True)
    assert DeltaTable(plain).version() == 1


def test_restore_accepts_naive_timestamp(engine: DeltaRsEngine, plain: str) -> None:
    time.sleep(1.1)
    mid = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    time.sleep(1.1)
    write_deltalake(plain, pa.table({"id": [3], "name": ["c"]}), mode="append")
    engine.restore(_resolved(plain), mid.isoformat())
    assert len(_rows(plain)) == 2


# ------------------------------------------------------------ maintenance


def test_optimize_accepts_commit_metadata(engine: DeltaRsEngine, plain: str) -> None:
    write_deltalake(plain, pa.table({"id": [3], "name": ["c"]}), mode="append")
    engine.optimize(_resolved(plain), commit_metadata={"job": "nightly"})
    assert DeltaTable(plain).history(1)[0].get("job") == "nightly"


def test_vacuum_accepts_commit_metadata_and_fractional_hours(
    engine: DeltaRsEngine, plain: str
) -> None:
    files = engine.vacuum(_resolved(plain), retention_hours=200.5, commit_metadata={"a": "b"})
    assert files == []


def test_zorder_without_columns_is_refused(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(ValueError, match="at least one column"):
        engine.zorder(_resolved(plain), [])


def test_compact_logs_inverted_range_is_refused(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(ValueError, match="inverted"):
        engine.compact_logs(_resolved(plain), 3, 1)


def test_history_negative_limit(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(ValueError, match="zero or more"):
        engine.history(_resolved(plain), limit=-1)


def test_write_panic_becomes_catchable(
    engine: DeltaRsEngine, plain: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PanicException(BaseException):
        pass

    def boom(*a: Any, **k: Any) -> None:
        raise PanicException("oops")

    monkeypatch.setattr(deltalake, "write_deltalake", boom)
    with pytest.raises(EnginePanicError):
        engine.append(_resolved(plain), pa.table({"id": [3], "name": ["c"]}))


# ---------------------------------------------------------- capabilities


def test_detail_reports_metadata_id(engine: DeltaRsEngine, plain: str) -> None:
    assert engine.detail(_resolved(plain))["metadata_id"] == DeltaTable(plain).metadata().id


def test_files_not_blocked_by_writer_only_gates(engine: DeltaRsEngine, plain: str) -> None:
    uniform = _resolved(
        plain,
        properties={"delta.universalFormat.enabledFormats": "iceberg"},
        writer_features=frozenset({"icebergCompatV2", "columnMapping"}),
    )
    if not uniform.has_iceberg_compat:
        pytest.skip("has_iceberg_compat reads a different signal in this build")
    assert engine.supports(Operation.FILES, uniform).ok


def test_supports_refuses_non_delta(engine: DeltaRsEngine, plain: str) -> None:
    iceberg = _resolved(plain, data_source_format="ICEBERG")
    cap = engine.supports(Operation.SCAN, iceberg)
    assert not cap.ok and "not Delta" in cap.reason


# --------------------------------------------------------- CDF retention


@pytest.mark.parametrize(
    ("value", "days"),
    [
        ("interval 7 days", 7.0),
        ("interval 1 week", 7.0),
        ("interval 10080 minutes", 7.0),
        ("interval 86400 seconds", 1.0),
        ("interval 2 fortnights", None),
        ("garbage", None),
    ],
)
def test_duration_days(value: str, days: float | None) -> None:
    got = deltars._duration_days(value)
    assert got == pytest.approx(days) if days is not None else got is None


def test_cdf_guard_minutes_not_misread(
    engine: DeltaRsEngine, cdf_table: tuple[str, dt.datetime]
) -> None:
    path, _ = cdf_table
    props = {
        **_cdf_props(),
        "delta.deletedFileRetentionDuration": "interval 64800 minutes",  # 45 days
        "delta.logRetentionDuration": "interval 30 days",
    }
    out = pa.table(engine.cdf(_resolved(path, properties=props)))
    assert out.num_rows == 3


# ------------------------------------------------------------------ schema


def test_add_columns_accepts_arrow_fields(engine: DeltaRsEngine, plain: str) -> None:
    engine.add_columns(_resolved(plain), [pa.field("x", pa.int32())])
    engine.add_columns(_resolved(plain), pa.schema([("y", pa.string())]))
    assert DeltaTable(plain).schema().to_arrow().names == ["id", "name", "x", "y"]


def test_add_not_null_column_is_refused(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(UnreachableTableError, match="NOT NULL"):
        engine.add_columns(_resolved(plain), [pa.field("z", pa.int32(), nullable=False)])
    assert pa.table(DeltaTable(plain).scan()).num_rows == 2


def test_add_constraint_accepts_commit_metadata(engine: DeltaRsEngine, plain: str) -> None:
    engine.add_constraint(_resolved(plain), {"pos": "id > 0"}, commit_metadata={"who": "me"})
    assert DeltaTable(plain).history(1)[0].get("who") == "me"


def test_zorder_single_column_string(engine: DeltaRsEngine, plain: str) -> None:
    write_deltalake(plain, pa.table({"id": [3], "name": ["c"]}), mode="append")
    engine.zorder(_resolved(plain), "name")
    engine.optimize(_resolved(plain), zorder_by="id")
    assert len(_rows(plain)) == 3


def test_overwrite_blank_predicate_is_refused(engine: DeltaRsEngine, plain: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        engine.overwrite(_resolved(plain), pa.table({"id": [9], "name": ["z"]}), predicate=" ")
    assert len(_rows(plain)) == 2


def test_merge_accepts_max_commit_retries(engine: DeltaRsEngine, plain: str) -> None:
    src = pa.table({"id": [5], "name": ["z"]})
    engine.merge(
        _resolved(plain),
        src,
        "t.id = s.id",
        source_alias="s",
        target_alias="t",
        max_commit_retries=3,
        commit_metadata={"m": "1"},
    ).when_not_matched_insert_all().execute()
    assert DeltaTable(plain).history(1)[0].get("m") == "1"


@pytest.mark.parametrize("feature", ["deletionVectors", "columnMapping"])
def test_generate_refused_for_features_manifests_cannot_express(
    engine: DeltaRsEngine, plain: str, feature: str
) -> None:
    table = _resolved(plain, reader_features=frozenset({feature}))
    with pytest.raises(UnreachableTableError, match="manifest"):
        engine.generate(table)


def test_delete_and_update_accept_max_commit_retries(engine: DeltaRsEngine, plain: str) -> None:
    engine.update(
        _resolved(plain), updates={"name": "'q'"}, predicate="id = 1", max_commit_retries=2
    )
    engine.delete(_resolved(plain), "id = 2", max_commit_retries=2)
    assert _rows(plain) == [{"id": 1, "name": "q"}]


def test_column_comment_on_dotted_top_level_name(engine: DeltaRsEngine, tmp_path: Any) -> None:
    path = str(tmp_path / "dots")
    write_deltalake(path, pa.table({"a.b": [1]}))
    engine.set_column_comment(_resolved(path), "a.b", "dotted")
    field = DeltaTable(path).schema().fields[0]
    assert field.metadata.get("comment") == "dotted"


def test_convert_accepts_storage_options(engine: DeltaRsEngine, tmp_path: Any) -> None:
    import pyarrow.parquet as pq

    path = tmp_path / "pq"
    path.mkdir()
    pq.write_table(pa.table({"id": [1, 2]}), str(path / "a.parquet"))
    engine.convert(str(path), storage_options={"allow_unsafe_rename": "true"})
    assert DeltaTable(str(path)).version() == 0


# --------------------------------------------------- history / CDF start


def _strip_commit_info(path: str, version: int) -> None:
    log = f"{path}/_delta_log/{version:020d}.json"
    with open(log) as f:
        lines = [line for line in f.read().splitlines() if "commitInfo" not in line]
    with open(log, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_history_versions_come_from_commit_files(engine: DeltaRsEngine, tmp_path: Any) -> None:
    from deltalake import CommitProperties

    path = str(tmp_path / "h")
    for i, mode in enumerate(["error", "append", "append"]):
        write_deltalake(
            path,
            pa.table({"id": [i]}),
            mode=cast(Literal["error", "append"], mode),
            commit_properties=CommitProperties(custom_metadata={"tag": f"v{i}"}),
        )
    _strip_commit_info(path, 1)
    history = engine.history(_resolved(path))
    assert [(h["version"], h.get("tag")) for h in history] == [(2, "v2"), (1, None), (0, "v0")]
    limited = engine.history(_resolved(path), limit=2)
    assert [h["version"] for h in limited] == [2, 1]


def test_cdf_without_start_begins_where_cdf_was_enabled(
    engine: DeltaRsEngine, tmp_path: Any
) -> None:
    path = str(tmp_path / "late")
    write_deltalake(path, pa.table({"id": [1]}))
    write_deltalake(path, pa.table({"id": [2]}), mode="append")
    DeltaTable(path).alter.set_table_properties({"delta.enableChangeDataFeed": "true"})
    write_deltalake(path, pa.table({"id": [3]}), mode="append")
    out = pa.table(engine.cdf(_resolved(path, properties=_cdf_props())))
    assert out.column("id").to_pylist() == [3]


# ------------------------------------------------------------------ merge


def _merge_table(tmp_path: Any) -> str:
    path = str(tmp_path / "m")
    write_deltalake(path, pa.table({"id": [1, 2], "my col": [1, 2], "a.b": [1, 2]}))
    return path


def test_merge_except_cols_unknown_name_is_refused(engine: DeltaRsEngine, tmp_path: Any) -> None:
    path = _merge_table(tmp_path)
    src = pa.table({"id": [1], "my col": [99], "a.b": [99]})
    merger = engine.merge(_resolved(path), src, "t.id = s.id", source_alias="s", target_alias="t")
    with pytest.raises(ValueError, match="My Col"):
        merger.when_matched_update_all(except_cols=["My Col"])


def test_merge_except_cols_string_is_one_column(engine: DeltaRsEngine, tmp_path: Any) -> None:
    path = _merge_table(tmp_path)
    src = pa.table({"id": [1], "my col": [99], "a.b": [99]})
    engine.merge(
        _resolved(path), src, "t.id = s.id", source_alias="s", target_alias="t"
    ).when_matched_update_all(except_cols="a.b").when_not_matched_insert_all().execute()
    assert _rows(path)[0] == {"id": 1, "my col": 99, "a.b": 1}


def test_merge_through_table_wrapper_still_chains(engine: DeltaRsEngine, tmp_path: Any) -> None:
    from deltaswamp.table import _InvalidatingMerger

    path = _merge_table(tmp_path)
    calls: list[int] = []
    src = pa.table({"id": [3], "my col": [3], "a.b": [3]})
    merger = _InvalidatingMerger(
        engine.merge(_resolved(path), src, "t.id = s.id", source_alias="s", target_alias="t"),
        lambda: calls.append(1),
    )
    merger.when_matched_update_all().when_not_matched_insert_all().execute()
    assert calls == [1] and len(_rows(path)) == 3


def test_writer_properties_dict_is_accepted(engine: DeltaRsEngine, plain: str) -> None:
    engine.append(
        _resolved(plain),
        pa.table({"id": [3], "name": ["c"]}),
        writer_properties={"compression": "ZSTD"},
    )
    engine.optimize(_resolved(plain), writer_properties={"compression": "SNAPPY"})
    assert len(_rows(plain)) == 3


# ------------------------------------------------------- object store keys


def test_gcs_bearer_token_is_refused_clearly() -> None:
    with pytest.raises(UnreachableTableError, match="bearer token"):
        deltars._object_store_options({"google_bearer_token": "ya29.x"})


def test_gcs_with_vended_credentials_not_claimed(engine: DeltaRsEngine) -> None:
    class Provider:
        def credentials(self, op: Any) -> Any:
            raise AssertionError("supports() must not vend")

    table = _resolved("gs://bucket/t", credential_provider=Provider())
    cap = engine.supports(Operation.SCAN, table)
    assert not cap.ok and "GCS" in cap.reason


def test_r2_gets_conditional_put() -> None:
    opts = deltars._object_store_options(
        {"aws_endpoint": "https://acct.r2.cloudflarestorage.com", "aws_access_key_id": "k"}
    )
    assert opts is not None and opts["aws_conditional_put"] == "etag"


def test_range_like_named_index_is_kept_as_a_column() -> None:
    import pandas as pd  # type: ignore[import-untyped]
    from deltaswamp.engine.deltars import _plain_data

    frame = pd.DataFrame({"v": ["a", "b", "c"], "id": [1, 2, 3]}).set_index("id")
    out = _plain_data(frame)
    assert "id" in out.column_names
    assert out.column("id").to_pylist() == [1, 2, 3]


# ------------------------------------------------ review_engines follow-ups


def _set_mtime_ms(path: str, version: int, ms: int) -> None:
    import os

    log = f"{path}/_delta_log/{version:020d}.json"
    os.utime(log, ns=(ms * 1_000_000 + 400_000, ms * 1_000_000 + 400_000))


def test_history_timestamp_is_what_time_travel_resolves(
    engine: DeltaRsEngine, tmp_path: Any
) -> None:
    # commitInfo.timestamp can be a little before the commit file's mtime,
    # which is what delta-rs time travel compares.
    path = str(tmp_path / "tt")
    write_deltalake(path, pa.table({"id": [0]}))
    write_deltalake(path, pa.table({"id": [1]}), mode="append")
    info = DeltaTable(path).history(1)[0]["timestamp"]
    _set_mtime_ms(path, 0, info - 5000)
    _set_mtime_ms(path, 1, info + 3)  # the file landed 3 ms after the writer's clock
    history = engine.history(_resolved(path))
    assert history[0]["version"] == 1 and history[0]["timestamp"] == info + 3
    stamp = dt.datetime.fromtimestamp(history[0]["timestamp"] / 1000, dt.UTC)
    got = pa.table(engine.scan(_resolved(path), timestamp=stamp))  # type: ignore[arg-type]
    assert sorted(got.column("id").to_pylist()) == [0, 1]


def test_history_prefers_the_in_commit_timestamp(engine: DeltaRsEngine, tmp_path: Any) -> None:
    import json

    path = str(tmp_path / "ict")
    write_deltalake(path, pa.table({"id": [0]}))
    log = f"{path}/_delta_log/{0:020d}.json"
    with open(log) as f:
        actions = [json.loads(line) for line in f.read().splitlines() if line.strip()]
    for action in actions:
        if "commitInfo" in action:
            action["commitInfo"]["inCommitTimestamp"] = 1_700_000_000_123
    with open(log, "w") as f:
        f.write("\n".join(json.dumps(a) for a in actions) + "\n")
    assert engine.history(_resolved(path))[0]["timestamp"] == 1_700_000_000_123


def test_history_numbering_fix_still_uses_file_timestamps(
    engine: DeltaRsEngine, tmp_path: Any
) -> None:
    path = str(tmp_path / "h2")
    for i, mode in enumerate(["error", "append"]):
        write_deltalake(path, pa.table({"id": [i]}), mode=cast(Literal["error", "append"], mode))
    _strip_commit_info(path, 1)
    _set_mtime_ms(path, 1, 1_800_000_000_000)
    history = engine.history(_resolved(path))
    assert history[0] == {"version": 1, "timestamp": 1_800_000_000_000}


def test_files_under_column_mapping_use_logical_stat_names(
    engine: DeltaRsEngine, tmp_path: Any
) -> None:
    from deltaswamp.engine.kernel import KernelEngine

    path = str(tmp_path / "cm")
    schema = pa.schema(
        [("id", pa.int64()), ("part", pa.string()), ("s", pa.struct([("x", pa.int32())]))]
    )
    kernel = KernelEngine()
    kernel.create(
        _resolved(path),
        schema,
        partition_by=["part"],
        properties={"delta.columnMapping.mode": "name"},
    )
    kernel.append(
        _resolved(path),
        pa.table({"id": [1, 2], "part": ["a", "a"], "s": [{"x": 1}, {"x": 2}]}, schema=schema),
    )
    names = pa.table(engine.files(_resolved(path))).column_names
    assert "min.id" in names and "max.s.x" in names and "partition.part" in names
    assert not any("col-" in n for n in names)


def test_files_without_column_mapping_are_unchanged(engine: DeltaRsEngine, plain: str) -> None:
    names = pa.table(engine.files(_resolved(plain))).column_names
    assert "min.id" in names


def test_gcs_static_service_account_key_is_still_claimed(engine: DeltaRsEngine) -> None:
    # [regression] every gs:// table with a credential provider was refused,
    # including a static service-account key delta-rs can use.
    from deltaswamp.credentials.base import Cloud, Credentials, StaticCredentialProvider

    creds = Credentials(
        cloud=Cloud.GCP,
        url="gs://bucket/t",
        expires_at=None,
        secrets={"google_service_account_key": "{}"},
    )
    table = _resolved("gs://bucket/t", credential_provider=StaticCredentialProvider(creds))
    assert engine.supports(Operation.SCAN, table).ok
    bearer = Credentials(
        cloud=Cloud.GCP, url="gs://bucket/t", expires_at=None, secrets={"google_bearer_token": "t"}
    )
    table = _resolved("gs://bucket/t", credential_provider=StaticCredentialProvider(bearer))
    assert not engine.supports(Operation.SCAN, table).ok


def test_merge_except_cols_without_the_private_schema_hook_passes_through() -> None:
    class Merger:
        def when_matched_update_all(self, predicate: Any = None, except_cols: Any = None) -> Any:
            self.seen = except_cols
            return self

    merger = Merger()
    checked = deltars._CheckedMerger(merger)
    checked.when_matched_update_all(except_cols="a")
    assert merger.seen == ["a"]
