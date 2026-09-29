"""`can()` and the call derive one request, so they route the same (refactor R1).

Each test asks can() about a call with the call's own arguments, then makes
the call, and checks the two agree: the engine can() names serves it, or both
refuse before anything is written.
"""

from __future__ import annotations

import decimal
import pickle
from typing import Any

import pytest
from deltaswamp.errors import EngineLimitError, InvalidArgumentError, UnreachableTableError

pa = pytest.importorskip("pyarrow")


def _served(conn: Any) -> list[str]:
    """Records which engine's serving methods run, on this connection only."""
    calls: list[str] = []
    for engine in conn.router.engines.values():
        for name in ("append", "overwrite", "delete", "update", "add_feature", "scan", "merge"):
            method = getattr(engine, name, None)
            if method is None:
                continue

            def wrapper(*args: Any, _m: Any = method, _k: Any = engine.kind, **kwargs: Any) -> Any:
                calls.append(_k.value)
                return _m(*args, **kwargs)

            setattr(engine, name, wrapper)
    return calls


def _dv_table(conn: Any, path: str) -> Any:
    conn.create_table(
        path,
        pa.schema([("id", pa.int64()), ("v", pa.string())]),
        properties={"delta.enableDeletionVectors": "true"},
    )
    t = conn.open_table(path)
    t.append(pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}))
    return conn.open_table(path)


def _decimal_partitioned(conn: Any, path: str) -> Any:
    data = pa.table(
        {
            "id": [1, 2],
            "p": pa.array([decimal.Decimal("1.50"), decimal.Decimal("2.50")], pa.decimal128(5, 2)),
        }
    )
    conn.write_table(path, data, partition_by=["p"])
    return conn.open_table(path)


def test_can_with_data_routes_like_the_append(conn: Any, tmp_path: Any) -> None:
    # D1: a negative fractional decimal partition value is a need of the rows;
    # can() could not be given the rows, and named delta-rs, which the call
    # itself avoids.
    t = _decimal_partitioned(conn, str(tmp_path / "t"))
    negative = pa.table({"id": [3], "p": pa.array([decimal.Decimal("-1.50")], pa.decimal128(5, 2))})
    verdict = t.can("append", data=negative)
    served = _served(conn)
    t.append(negative)
    assert verdict.ok
    assert served == [verdict.engine.value] == ["kernel"]


def test_can_update_routes_like_the_update(conn: Any, tmp_path: Any) -> None:
    t = _decimal_partitioned(conn, str(tmp_path / "t"))
    verdict = t.can("update", updates={"p": "-1.5"}, predicate="id = 1")
    served = _served(conn)
    t.update(updates={"p": "-1.5"}, predicate="id = 1")
    assert served == [verdict.engine.value] == ["kernel"]


def test_dml_commit_metadata_is_served_by_the_kernel(conn: Any, tmp_path: Any) -> None:
    # C5 / D2: the kernel refused commit_metadata inside delete() and update()
    # after can() had named it. It now records it.
    t = _dv_table(conn, str(tmp_path / "t"))
    verdict = t.can("delete", predicate="id = 1", commit_metadata={"job": "r1"})
    served = _served(conn)
    t.delete("id = 1", commit_metadata={"job": "r1"})
    t.update({"v": "'z'"}, predicate="id = 2", commit_metadata={"job": "r1-update"})
    assert verdict.ok and served == ["kernel", "kernel"]
    history = conn.open_table(str(tmp_path / "t")).history()
    assert [h.get("job") for h in history[:2]] == ["r1-update", "r1"]


def test_an_option_the_kernel_lacks_falls_through(conn: Any, tmp_path: Any) -> None:
    # D2: max_commit_retries on a kernel DML rewrite was refused inside
    # delete(), and delta-rs, which takes it, was never tried. Refused in
    # supports(), it routes on to delta-rs, and can() says so.
    t = _dv_table(conn, str(tmp_path / "t"))
    plain = t.can("delete", predicate="id = 1")
    verdict = t.can("delete", predicate="id = 1", max_commit_retries=3)
    served = _served(conn)
    t.delete("id = 1", max_commit_retries=3)
    assert plain.engine.value == "kernel"
    assert served == [verdict.engine.value] == ["deltars"]
    assert conn.open_table(str(tmp_path / "t")).count() == 2


def test_add_feature_takes_the_calls_argument(conn: Any, tmp_path: Any) -> None:
    # C4: can(add_feature, feature=...) passed the call's spelling through,
    # no engine read it, and it named delta-rs while the call used the kernel.
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1]}))
    t = conn.open_table(path)
    verdict = t.can("add_feature", feature="deletionVectors", allow_protocol_versions_increase=True)
    assert t.can("add_feature", features=["deletionVectors"]).engine == verdict.engine
    served = _served(conn)
    t.add_feature("deletionVectors", allow_protocol_versions_increase=True)
    assert served == [verdict.engine.value]


def test_plan_write_can_be_asked(conn: Any, tmp_path: Any) -> None:
    # C3: plan_write needs distributed_write, which only the kernel has; where
    # the kernel does not write, can("append") said ok and plan_write refused.
    # An append-only table refuses a distributed overwrite.
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1]}), properties={"delta.appendOnly": "true"})
    t = conn.open_table(path)
    assert t.can("append").ok
    verdict = t.can("plan_write", mode="overwrite")
    assert not verdict.ok and "append-only" in verdict.reason
    with pytest.raises(UnreachableTableError):
        t.plan_write(mode="overwrite")


def test_dynamic_overwrite_of_an_unpartitioned_table(conn: Any, tmp_path: Any) -> None:
    # C9: refused inside delta-rs's write, after can() said ok.
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1, 2]}))
    t = conn.open_table(path)
    verdict = t.can("overwrite", partition_overwrite="dynamic")
    assert not verdict.ok and "not partitioned" in verdict.reason
    with pytest.raises(UnreachableTableError, match="not partitioned"):
        t.overwrite(pa.table({"id": [9]}), partition_overwrite="dynamic")
    assert conn.open_table(path).count() == 2


def test_merge_that_removes_rows_on_an_append_only_table(conn: Any, tmp_path: Any) -> None:
    # C6: delta-rs accepted it and failed at commit with a raw
    # CommitFailedError. The clauses decide, so can() takes them.
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1, 2]}), properties={"delta.appendOnly": "true"})
    t = conn.open_table(path)
    source = pa.table({"id": [2, 3]})
    args = {"source": source, "predicate": "t.id = s.id", "source_alias": "s", "target_alias": "t"}
    updating = t.can("merge", **args, clauses=["when_matched_update_all"])
    inserting = t.can("merge", **args, clauses=["when_not_matched_insert_all"])
    assert not updating.ok and "append-only" in updating.reason
    assert inserting.ok
    with pytest.raises(UnreachableTableError, match="append-only"):
        t.merge(**args).when_matched_update_all().when_not_matched_insert_all().execute()
    assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == [1, 2]
    t.merge(**args).when_not_matched_insert_all().execute()
    assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == [1, 2, 3]


def test_scan_beyond_the_kernel_grammar_routes_to_delta_rs(conn: Any, tmp_path: Any) -> None:
    # C1: can() named the kernel, whose scan raised PredicateError.
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1, 2, 8]}))
    t = conn.open_table(path)
    verdict = t.can("scan", predicate="id % 7 = 1")
    served = _served(conn)
    rows = pa.table(t.scan(predicate="id % 7 = 1"))
    assert served == [verdict.engine.value] == ["deltars"]
    assert sorted(rows.column("id").to_pylist()) == [1, 8]
    # Malformed in any SQL: the kernel's parser names it before routing, so
    # no engine reads a prefix of it (round-7 INJ-2).
    from deltaswamp.predicate import PredicateError

    with pytest.raises(PredicateError, match="==="):
        t.can("scan", predicate="id ===")


def test_can_convert_a_parquet_directory(conn: Any, tmp_path: Any) -> None:
    # C8: can() refused because the directory has no Delta log, which is the
    # premise of a conversion.
    import pyarrow.parquet as pq

    directory = tmp_path / "parquet"
    directory.mkdir()
    pq.write_table(pa.table({"id": [1, 2]}), directory / "part-0.parquet")
    assert conn.open_table(str(directory)).can("convert").ok
    assert conn.convert_to_delta(str(directory)).count() == 2


def test_can_takes_method_names_and_refuses_unknown_arguments(conn: Any, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    conn.write_table(path, pa.table({"id": [1]}))
    t = conn.open_table(path)
    assert t.can("z_order", columns=["id"]).operation.value == "zorder"
    assert t.can("compact_logs").operation.value == "log_compaction"
    with pytest.raises(InvalidArgumentError, match="predicat"):
        t.can("delete", predicat="id = 1")
    with pytest.raises(InvalidArgumentError, match="not an operation"):
        t.can("teleport")


class _Refusing:
    kind = None

    def delete(self) -> None:
        raise UnreachableTableError("delete", "refused after routing")

    def scan(self) -> None:
        raise EngineLimitError("scan", "a declared read-time limit")

    def snapshot(self) -> None:
        raise UnreachableTableError("snapshot", "a probe, not the service")


def test_strict_routing_flags_an_engine_refusal(monkeypatch: Any) -> None:
    from deltaswamp._request import RoutingContractViolation, strict_engine
    from deltaswamp.capability import Capability, Engine, Operation

    monkeypatch.setenv("DELTASWAMP_STRICT_ROUTING", "1")
    engine = _Refusing()
    served_elsewhere = Capability(Operation.DELETE, ok=True, engine=Engine.KERNEL)
    wrapped = strict_engine(engine, Operation.DELETE, lambda kind: served_elsewhere)
    assert isinstance(wrapped, _Refusing)
    with pytest.raises(RoutingContractViolation, match="refused after routing"):
        wrapped.delete()
    with pytest.raises(UnreachableTableError):
        wrapped.snapshot()
    with pytest.raises(EngineLimitError):
        strict_engine(engine, Operation.SCAN).scan()
    assert type(pickle.loads(pickle.dumps(wrapped))) is _Refusing
    # A refusal no other engine would avoid is about the request: left as it is.
    nowhere = Capability(Operation.DELETE, ok=False, reason="no engine")
    with pytest.raises(UnreachableTableError):
        strict_engine(engine, Operation.DELETE, lambda kind: nowhere).delete()
    monkeypatch.setenv("DELTASWAMP_STRICT_ROUTING", "0")
    assert strict_engine(engine, Operation.DELETE) is engine
