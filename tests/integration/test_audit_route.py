"""Regressions from the round-5 review of request routing and the error boundary.

`t.can(op, **args)` must route on the request the call makes (so an ok names
the engine that serves it, and a refusal is the call's), and the boundary must
keep the class of what it translates -- and leave the caller's own exceptions
alone. Each test failed before its fix.
"""

from __future__ import annotations

import glob
import os
import pickle
import warnings
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")
if not ds.has_native():  # pragma: no cover
    pytest.skip("native extension not built", allow_module_level=True)

import deltalake.exceptions as dle  # noqa: E402
from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.engine.boundary import Boundary, translate  # noqa: E402
from deltaswamp.errors import (  # noqa: E402
    CommitConflictError,
    CommitRefusedError,
    CorruptTableError,
    EngineError,
    EngineLimitError,
    InvalidArgumentError,
    StorageError,
    TableNotFoundError,
)


class MyError(Exception):
    """The caller's own exception, raised from the caller's own data source."""


@pytest.fixture
def feed(conn: Any, tmp_path: Any) -> Any:
    path = str(tmp_path / "cdf")
    conn.create_table(
        path, pa.schema([("id", pa.int64())]), properties={"delta.enableChangeDataFeed": "true"}
    )
    t = conn.open_table(path)
    t.append(pa.table({"id": [1, 2]}))
    t.append(pa.table({"id": [3]}))
    return t


@pytest.fixture
def table(conn: Any, tmp_path: Any) -> Any:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1, 2, 3]}))
    return conn.open_table(path)


# ------------------------------------------------------------------ routing


class TestChangeFeedRouting:
    """M1 / HE-7: cdf() and changes() route on their own options, as can() does."""

    @pytest.mark.parametrize("predicate", ["id % 2 = 1", "abs(id) > 1"])
    def test_a_predicate_beyond_the_kernel_grammar(self, feed: Any, predicate: str) -> None:
        cap = feed.can("cdf", starting_version=0, predicate=predicate)
        assert cap.ok and cap.engine is Engine.DELTARS
        stream = feed.cdf(starting_version=0, predicate=predicate)
        assert stream.engine_kind is Engine.DELTARS
        assert stream.read_all().num_rows >= 1

    def test_changes_with_such_a_predicate(self, feed: Any) -> None:
        assert feed.can("changes", predicate="id % 2 = 1").engine is Engine.DELTARS
        rows = [r for _, t in feed.changes(0, predicate="id % 2 = 1") for r in t.to_pylist()]
        assert sorted(r["id"] for r in rows) == [1, 3]

    def test_allow_out_of_range_names_the_engine_that_serves_it(self, feed: Any) -> None:
        cap = feed.can("cdf", starting_version=0, ending_version=99, allow_out_of_range=True)
        assert cap.engine is Engine.DELTARS
        stream = feed.cdf(starting_version=0, ending_version=99, allow_out_of_range=True)
        assert stream.engine_kind is Engine.DELTARS

    def test_a_plain_feed_stays_on_the_kernel(self, feed: Any) -> None:
        assert feed.can("cdf", starting_version=0, allow_out_of_range=False).engine is (
            Engine.KERNEL
        )
        assert feed.cdf(starting_version=0).engine_kind is Engine.KERNEL


def test_update_from_a_nested_field_on_a_dv_table(conn: Any, tmp_path: Any) -> None:
    """M2: the kernel's UPDATE reads top-level columns only; strict routing is on here."""
    path = str(tmp_path / "dv")
    st = pa.array([{"x": i, "y": str(i)} for i in range(3)])
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("v", pa.int64()), ("st", st.type)]),
        properties={"delta.enableDeletionVectors": "true"},
    )
    t = conn.open_table(path)
    t.append(pa.table({"id": [0, 1, 2], "v": [9, 9, 9], "st": st}))
    assert t.can("update", updates={"v": "1"}).engine is Engine.KERNEL
    assert t.can("update", updates={"v": "st.x"}).engine is Engine.DELTARS
    t.update({"v": "st.x"})
    assert sorted(t.to_arrow().column("v").to_pylist()) == [0, 1, 2]


def test_writes_through_a_pinned_handle_are_refused_by_can(conn: Any, tmp_path: Any) -> None:
    """M5: can() said ok for every write on a version-pinned handle."""
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1]}))
    conn.open_table(path).append(pa.table({"id": [2]}))
    t = conn.open_table(path, version=0)
    cases: list[tuple[str, dict[str, Any]]] = [
        ("append", {}),
        ("delete", {"predicate": "id = 1"}),
        ("merge", {}),
        ("optimize", {}),
        ("set_properties", {"properties": {"a.b": "1"}}),
        ("plan_write", {}),
    ]
    for op, args in cases:
        cap = t.can(op, **args)
        assert not cap.ok, op
        assert "pinned to version 0" in cap.reason
        assert "without version=" in (cap.remedy or "")
    with pytest.raises(InvalidArgumentError, match="pinned"):
        t.append(pa.table({"id": [9]}))
    assert t.can("scan").ok and t.can("vacuum").ok


@pytest.mark.parametrize("clauses", [{"when_matched_delete"}, "when_matched_delete"])
def test_merge_clauses_as_a_set_or_one_name(conn: Any, tmp_path: Any, clauses: Any) -> None:
    """L2: only a list or a tuple was read."""
    path = str(tmp_path / "ao")
    conn.write_table(path, pa.table({"id": [1]}), properties={"delta.appendOnly": "true"})
    assert not conn.open_table(path).can("merge", clauses=clauses).ok


def test_property_values_are_spelled_before_routing(table: Any) -> None:
    """r5rg M5: the kernel normalised True and '7 days', delta-rs refused them."""
    cap = table.can("set_properties", properties={"delta.appendOnly": True})
    assert cap.ok
    table.set_properties({"delta.appendOnly": True, "delta.logRetentionDuration": "7 days"})
    props = table.properties()
    assert props["delta.appendOnly"] == "true"
    assert props["delta.logRetentionDuration"] == "interval 7 days"


def test_no_alter_leaves_a_feature_table_unwritable(conn: Any, tmp_path: Any) -> None:
    """r5rg H3: a (3,7) table listing checkConstraints gains a feature delta-rs cannot write."""
    path = str(tmp_path / "v7")
    t = conn.create_table(path, pa.schema([("id", pa.int64()), ("city", pa.string())]))
    t.append(pa.table({"id": [1], "city": ["a"]}))
    t.set_properties({"delta.enableChangeDataFeed": "true"})
    conn.open_table(path).add_feature("deletionVectors")
    t = conn.open_table(path)
    assert "checkConstraints" in t.features()
    change = {"delta.enableTypeWidening": "true"}
    cap = t.can("set_properties", properties=change)
    assert not cap.ok and "no local engine" in cap.reason
    with pytest.raises(ds.UnreachableTableError, match="no local engine"):
        t.set_properties(change)
    with pytest.raises(ds.UnreachableTableError, match="no local engine"):
        t.cluster_by(["city"])
    t = conn.open_table(path)
    assert "typeWidening" not in t.features()
    t.append(pa.table({"id": [2], "city": ["b"]}))


# ------------------------------------------------------------------ the boundary


class TestClassFidelity:
    """M3: a translated error stays an instance of what it translates."""

    @pytest.mark.parametrize("exc", [TimeoutError("timed out"), ConnectionResetError(54, "reset")])
    def test_storage_errors_keep_their_class(self, exc: OSError) -> None:
        error = translate(Engine.DELTARS, "scan", exc)
        assert isinstance(error, StorageError) and isinstance(error, type(exc))
        assert error.errno == exc.errno
        again = pickle.loads(pickle.dumps(error))
        assert type(again) is type(error) and again.errno == exc.errno
        assert str(again) == str(error)

    def test_arrow_invalid_stays_catchable(self) -> None:
        error = translate(Engine.KERNEL, "scan", pa.ArrowInvalid("bad"))
        assert isinstance(error, EngineError) and isinstance(error, pa.ArrowInvalid)
        assert isinstance(pickle.loads(pickle.dumps(error)), pa.ArrowInvalid)

    def test_a_constructor_needing_more_falls_back_to_an_ancestor(self) -> None:
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        error = translate(Engine.SHARING, "scan", exc)
        assert isinstance(error, EngineError) and isinstance(error, ValueError)

    def test_key_error_text_is_not_quoted(self) -> None:
        error = translate(Engine.KERNEL, "scan", KeyError("gone"))
        assert isinstance(error, KeyError)
        assert str(error).startswith("the kernel failed")

    def test_file_not_found_keeps_errno_and_filename_through_pickle(self) -> None:
        exc = FileNotFoundError(2, "No such file", "/x/y.parquet")
        error = pickle.loads(pickle.dumps(translate(Engine.KERNEL, "scan", exc)))
        assert isinstance(error, FileNotFoundError)
        assert (error.errno, error.filename) == (2, "/x/y.parquet")

    def test_delta_rs_types(self) -> None:
        def tr(exc: BaseException) -> BaseException:
            return translate(Engine.DELTARS, "append", exc)

        assert isinstance(tr(dle.TableNotFoundError("no table")), TableNotFoundError)
        assert isinstance(tr(dle.DeltaProtocolError("reader features x")), EngineLimitError)
        assert isinstance(
            tr(dle.DeltaProtocolError("Invariant violations: [x]")), InvalidArgumentError
        )
        refused = tr(dle.CommitFailedError("Delta table is append-only"))
        assert isinstance(refused, CommitRefusedError) and isinstance(refused, EngineError)
        assert isinstance(
            tr(dle.CommitFailedError("Transaction failed: Version 3 already exists")),
            CommitConflictError,
        )


class TestCallersOwnErrors:
    """M3: what the caller's own data source raises reaches the caller as it was raised."""

    def test_from_arrow_c_stream(self, table: Any) -> None:
        class Source:
            def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
                raise MyError("from my source")

        with pytest.raises(MyError, match="from my source"):
            table.append(Source())

    def test_from_the_iterator_behind_a_reader(self, table: Any) -> None:
        def batches() -> Any:
            yield pa.record_batch({"id": [10]})
            raise MyError("mid-stream")

        reader = pa.RecordBatchReader.from_batches(pa.schema([("id", pa.int64())]), batches())
        with pytest.raises(MyError, match="mid-stream"):
            table.append(reader)
        assert table.count() == 3

    def test_a_reader_still_writes(self, table: Any) -> None:
        reader = pa.RecordBatchReader.from_batches(
            pa.schema([("id", pa.int64())]), iter([pa.record_batch({"id": [7, 8]})])
        )
        table.append(reader)
        assert table.count() == 5

    def test_a_generator_is_not_data(self, table: Any) -> None:
        def gen() -> Any:
            yield pa.record_batch({"id": [1]})

        with pytest.raises(InvalidArgumentError, match=r"RecordBatchReader\.from_batches"):
            table.append(gen())


class TestReadRulesAreScoped:
    """L3: a read's schema error is delta-rs failing on the table, not the caller's input."""

    def test_schema_error_on_a_read_is_an_engine_error(self) -> None:
        exc = dle.DeltaError("Schema error: invalid data type for stats")
        assert type(translate(Engine.DELTARS, "scan", exc)) is EngineError
        assert isinstance(translate(Engine.DELTARS, "update", exc), InvalidArgumentError)

    def test_an_unsupported_type_is_an_engine_limit(self) -> None:
        exc = dle.DeltaError("Kernel error: Schema error: Unsupported Delta table type: 'x'")
        assert isinstance(translate(Engine.DELTARS, "open the table", exc), EngineLimitError)

    def test_valid_fields_hint_is_kept(self) -> None:
        exc = dle.DeltaError("Schema error: No field named nope.\nValid fields are id, s.")
        assert "Valid fields are id, s" in str(translate(Engine.DELTARS, "update", exc))


def test_a_truncated_data_file_is_corrupt_not_storage(conn: Any, tmp_path: Any) -> None:
    """L4: it read as a storage outage."""
    path = str(tmp_path / "trunc")
    conn.write_table(path, pa.table({"id": list(range(10))}))
    (f,) = glob.glob(path + "/*.parquet")
    with open(f, "r+b") as handle:
        handle.truncate(os.path.getsize(f) // 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(CorruptTableError, match="damaged") as info:
            conn.open_table(path).to_arrow()
    assert not isinstance(info.value, StorageError)


class TestWrapper:
    def test_monkeypatching_leaves_plans_picklable(self, table: Any) -> None:
        """L1: undo wrote the boundary's closure onto the engine instance."""
        kernel = table._connection.router.engines[Engine.KERNEL]
        assert kernel.files is kernel.files
        mp = pytest.MonkeyPatch()
        mp.setattr(kernel, "files", lambda *a, **k: "fake")
        assert kernel.files(table.resolved) == "fake"
        pickle.dumps(table._connection.router.engines)  # the patch stays behind
        mp.undo()
        target = object.__getattribute__(kernel, "_boundary_target")
        assert "files" not in vars(target)
        pickle.dumps(table.plan_scan())

    def test_the_engines_mapping_cannot_be_bypassed(self) -> None:
        """L5: |=, reassignment and copy() handed out raw engines."""
        from deltaswamp.engine.deltars import DeltaRsEngine

        conn = ds.connect()
        conn.router.engines |= {Engine.DELTARS: DeltaRsEngine()}
        assert type(conn.router.engines[Engine.DELTARS]) is Boundary
        merged = conn.router.engines | {Engine.DELTARS: DeltaRsEngine()}
        assert type(merged[Engine.DELTARS]) is Boundary
        conn.router.engines = {Engine.DELTARS: DeltaRsEngine()}
        assert type(conn.router.engines[Engine.DELTARS]) is Boundary
        copied = conn.router.engines.copy()
        copied[Engine.KERNEL] = object()
        assert type(copied[Engine.KERNEL]) is Boundary


# ------------------------------------------------------------------ r5rg items


def test_a_conditional_last_insert_on_a_feed_table_is_refused_by_can(feed: Any) -> None:
    """r5rg M11: can("merge") named delta-rs, whose execute() then refused."""
    conditional = [("when_not_matched_insert_all", "source.id > 1")]
    cap = feed.can("merge", clauses=conditional)
    assert not cap.ok and "conditional WHEN NOT MATCHED" in cap.reason
    assert feed.can("merge", clauses=["when_not_matched_insert_all"]).ok
    builder = feed.merge(pa.table({"id": [5]}), "target.id = source.id")
    with pytest.raises(ds.UnreachableTableError, match="conditional WHEN NOT MATCHED"):
        builder.when_not_matched_insert_all(predicate="source.id > 1").execute()
    assert feed.count() == 3


def test_date_appends_into_a_column_widened_to_timestamp_ntz(conn: Any, tmp_path: Any) -> None:
    """r5rg M1: refused as "narrows or reinterprets"; Delta widens DATE to TIMESTAMP_NTZ."""
    path = str(tmp_path / "tw")
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("d", pa.date32())]),
        properties={"delta.enableTypeWidening": "true"},
    )
    conn.open_table(path).append(pa.table({"id": [1], "d": pa.array([19000], pa.date32())}))
    conn.open_table(path).alter_column_type("d", "timestamp_ntz")
    conn.open_table(path).append(pa.table({"id": [2], "d": pa.array([19001], pa.date32())}))
    rows = sorted(conn.open_table(path).to_arrow().to_pylist(), key=lambda r: r["id"])
    assert [r["d"].isoformat() for r in rows] == ["2022-01-08T00:00:00", "2022-01-09T00:00:00"]


@pytest.mark.parametrize("kind", ["generated", "check"])
def test_replace_refuses_to_keep_generated_columns_or_checks(
    conn: Any, tmp_path: Any, kind: str
) -> None:
    """r5rg M10: replace() carried them into the new table, or failed half-way."""
    path = str(tmp_path / kind)
    if kind == "generated":
        gen = pa.field("g", pa.int64(), metadata={"delta.generationExpression": "id * 2"})
        conn.create_table(path, pa.schema([("id", pa.int64()), gen]))
    else:
        conn.create_table(path, pa.schema([("id", pa.int64())]))
        conn.open_table(path).add_constraint({"idpos": "id > 0"})
    t = conn.open_table(path)
    t.append(pa.table({"id": [1]}))
    new = pa.table({"x": pa.array([1.5])})
    assert not t.can("overwrite", schema_mode="overwrite", data=new).ok
    with pytest.raises(ds.UnreachableTableError, match="would keep"):
        t.replace(new)
    assert conn.open_table(path).to_arrow().column("id").to_pylist() == [1]


def test_storage_options_must_be_a_dict() -> None:
    """r5rg low: a list of pairs raised a raw AttributeError."""
    with pytest.raises(InvalidArgumentError, match="storage_options"):
        ds.connect(storage_options=[("a", "b")])


def test_a_version_the_log_does_not_hold_is_not_a_contract_violation(table: Any) -> None:
    """r5rg low: under strict routing (on in this suite) it raised RoutingContractViolation."""
    with pytest.raises(ds.UnreachableTableError, match="no such version"):
        table.to_arrow(version=99)
