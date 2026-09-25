"""Regression tests for the distributed write / scan surface (wave-4 audit).

Each test is one defect found by attacking `plan_write` / `plan_scan` the way a
hostile caller or a flaky cluster would.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine
from deltaswamp.errors import (
    CommitConflictError,
    InvalidArgumentError,
    TransientCommitError,
    UnreachableTableError,
)

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")


@pytest.fixture
def conn() -> Any:
    from deltaswamp.catalog.filesystem import FilesystemCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Connection(
        catalog=FilesystemCatalog(),
        router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
    )


def _table(conn: Any, schema: Any = None, **kwargs: Any) -> str:
    location = os.path.join(tempfile.mkdtemp(), "t")
    conn.create_table(
        location, schema or pa.schema([("id", pa.int64()), ("region", pa.string())]), **kwargs
    )
    return location


def _rows(n: int = 1, start: int = 0) -> Any:
    return pa.table({"id": list(range(start, start + n)), "region": ["us"] * n})


# ------------------------------------------------------------------ idempotency


def test_a_txn_committed_after_planning_is_not_committed_twice(conn: Any) -> None:
    """Two runs of the same job both passed plan-time dedup and both committed."""
    loc = _table(conn)
    first = conn.open_table(loc).plan_write(txn=("job", 1))
    second = conn.open_table(loc).plan_write(txn=("job", 1))
    a, b = first.write(_rows(1, 1)), second.write(_rows(1, 2))
    first.commit([a])
    with pytest.raises(UnreachableTableError, match="already committed"):
        second.commit([b])
    assert conn.open_table(loc).to_arrow().to_pydict()["id"] == [1]


def test_a_retry_after_an_ambiguous_commit_does_not_add_the_files_again(
    conn: Any, monkeypatch: Any
) -> None:
    """A commit that landed but was reported as failed was re-committed on retry."""
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    fragment = plan.write(_rows(2))
    real = plan.engine.commit_files
    calls = []

    def lands_then_fails(*args: Any, **kwargs: Any) -> int:
        calls.append(1)
        real(*args, **kwargs)
        raise TransientCommitError("timed out after the put")

    monkeypatch.setattr(plan.engine, "commit_files", lands_then_fails)
    with pytest.raises(UnreachableTableError, match="already in the table"):
        plan.commit([fragment], retries=2)
    assert len(calls) == 1
    t = conn.open_table(loc)
    assert t.version == 1, "exactly one commit"
    assert t.to_arrow().num_rows == 2


# ------------------------------------------------------------------ overwrite


def test_a_guarded_overwrite_conflicts_even_when_the_check_cannot_run(
    conn: Any, monkeypatch: Any
) -> None:
    """The moved-table check swallows errors; the commit itself must still be pinned.

    It committed against a fresh snapshot, so a writer that landed after the
    check had its files removed with nothing in the log to say so.
    """
    loc = _table(conn)
    conn.open_table(loc).append(_rows(1, 1))
    plan = conn.open_table(loc).plan_write(mode="overwrite")
    fragment = plan.write(_rows(1, 9))
    conn.open_table(loc).append(_rows(1, 5))  # the racing writer

    def unreachable(*_: Any, **__: Any) -> Any:
        raise OSError("flaky network")

    monkeypatch.setattr(plan.engine, "detail", unreachable)
    with pytest.raises(CommitConflictError):
        plan.commit([fragment])
    monkeypatch.undo()
    assert 5 in conn.open_table(loc).to_arrow().to_pydict()["id"]


# ------------------------------------------------------------------ plan-time validation


def test_plan_write_refuses_a_handle_pinned_to_a_past_version(conn: Any) -> None:
    loc = _table(conn)
    conn.open_table(loc).append(_rows())
    with pytest.raises(InvalidArgumentError, match="pinned"):
        conn.open_table(loc, version=0).plan_write()


@pytest.mark.parametrize("txn", [("a", "1"), ("a", -1), "abc", ("", 1), ("a", 1, 2)])
def test_plan_write_refuses_a_malformed_txn(conn: Any, txn: Any) -> None:
    """A string version passed planning and failed only at commit, after the job."""
    loc = _table(conn)
    with pytest.raises(InvalidArgumentError):
        conn.open_table(loc).plan_write(txn=txn)


def test_plan_write_refuses_a_reserved_commit_metadata_key(conn: Any) -> None:
    loc = _table(conn)
    with pytest.raises(InvalidArgumentError, match="reserved"):
        conn.open_table(loc).plan_write(commit_metadata={"Operation": "mine"})


def test_plan_write_refuses_a_none_commit_metadata_value(conn: Any) -> None:
    """None was recorded in the log as the string 'None'."""
    loc = _table(conn)
    with pytest.raises(InvalidArgumentError, match="None"):
        conn.open_table(loc).plan_write(commit_metadata={"a": None})


def test_commit_metadata_values_are_recorded_as_strings(conn: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write(commit_metadata={"run": 7})
    plan.commit([plan.write(_rows())])
    assert conn.open_table(loc).history()[0]["run"] == "7"


# ------------------------------------------------------------------ fragments


def test_a_bare_fragment_is_refused_with_a_clear_message(conn: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    fragment = plan.write(_rows())
    with pytest.raises(InvalidArgumentError, match=r"commit\(\[fragment\]\)"):
        plan.commit(fragment)
    assert conn.open_table(loc).version == 0


@pytest.mark.parametrize("junk", [None, "abc", 12])
def test_a_fragment_that_is_not_bytes_is_named_by_index(conn: Any, junk: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    good = plan.write(_rows())
    with pytest.raises(InvalidArgumentError, match="fragment 1"):
        plan.commit([good, junk])
    assert conn.open_table(loc).version == 0, "nothing committed"


def test_the_same_fragment_twice_is_refused(conn: Any) -> None:
    """Two add actions for one path in one commit: the change feed counts the rows twice."""
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    fragment = plan.write(_rows())
    with pytest.raises(ValueError, match="more than one fragment"):
        plan.commit([fragment, fragment])
    assert conn.open_table(loc).version == 0


def test_bytearray_and_memoryview_fragments_are_accepted(conn: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    plan.commit([bytearray(plan.write(_rows(1, 1))), memoryview(plan.write(_rows(1, 2)))])
    assert conn.open_table(loc).count() == 2


# ------------------------------------------------------------------ retries


def test_a_transient_commit_error_is_retried(conn: Any, monkeypatch: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    fragment = plan.write(_rows())
    real = plan.engine.commit_files
    failures = [TransientCommitError("i/o hiccup")]

    def flaky(*args: Any, **kwargs: Any) -> int:
        if failures:
            raise failures.pop()
        result: int = real(*args, **kwargs)
        return result

    monkeypatch.setattr(plan.engine, "commit_files", flaky)
    assert plan.commit([fragment], retries=1) == 1
    assert conn.open_table(loc).count() == 1


def test_exhausted_transient_retries_do_not_claim_a_conflict(conn: Any, monkeypatch: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    fragment = plan.write(_rows())

    def always(*_: Any, **__: Any) -> int:
        raise TransientCommitError("i/o hiccup")

    monkeypatch.setattr(plan.engine, "commit_files", always)
    with pytest.raises(TransientCommitError):
        plan.commit([fragment], retries=2)


@pytest.mark.parametrize("retries", [-1, "2", True, 1.5])
def test_retries_must_be_a_non_negative_int(conn: Any, retries: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    with pytest.raises(InvalidArgumentError, match="retries"):
        plan.commit([], retries=retries)


# ------------------------------------------------------------------ worker input


def test_write_accepts_a_list_of_record_batches_and_row_dicts(conn: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    batches = _rows(2).to_batches()
    plan.commit(
        [plan.write(batches), plan.write([{"id": 7, "region": "eu"}, {"id": 8, "region": None}])]
    )
    assert sorted(conn.open_table(loc).to_arrow().to_pydict()["id"]) == [0, 1, 7, 8]


def test_write_refuses_none(conn: Any) -> None:
    loc = _table(conn)
    with pytest.raises(InvalidArgumentError, match="None"):
        conn.open_table(loc).plan_write().write(None)


# ------------------------------------------------------------------ scan plans


def test_splits_from_another_plan_are_refused(conn: Any) -> None:
    """A foreign split resolved against this table's root read nothing, silently."""
    a, b = _table(conn), _table(conn)
    conn.open_table(a).append(_rows(1, 1))
    conn.open_table(b).append(_rows(1, 2))
    plan_a = conn.open_table(a).plan_scan()
    plan_b = conn.open_table(b).plan_scan()
    with pytest.raises(InvalidArgumentError, match="not part of this plan"):
        plan_a.read(plan_b.splits)
    assert plan_a.read(plan_a.partitions(2)[0]).num_rows == 1


def test_an_empty_plan_reads_the_planned_schema_not_the_latest(conn: Any) -> None:
    loc = _table(conn)
    conn.open_table(loc).add_column(pa.field("amt", pa.float64()))
    plan = conn.open_table(loc).plan_scan(version=0)
    assert plan.splits == ()
    assert plan.version == 0
    assert plan.read().column_names == ["id", "region"]


# ------------------------------------------------------------------ error translation


def test_a_fragment_for_another_table_is_a_library_error(conn: Any) -> None:
    """It was a bare ValueError, so `except DeltaSwampError` around commit missed it."""
    a, b = _table(conn), _table(conn)
    fragment = conn.open_table(a).plan_write().write(_rows())
    with pytest.raises(ds.DeltaSwampError, match="different table"):
        conn.open_table(b).plan_write().commit([fragment])


def test_junk_fragment_bytes_say_what_they_are_not(conn: Any) -> None:
    loc = _table(conn)
    with pytest.raises(InvalidArgumentError, match="not a fragment produced by write_files"):
        conn.open_table(loc).plan_write().commit([b"\x00\x01garbage"])


def test_a_duplicated_data_column_is_named_as_a_duplicate(conn: Any) -> None:
    """A second `region` was reported as a column 'not in the table schema'."""
    loc = _table(conn)
    data = pa.Table.from_arrays(
        [pa.array([1]), pa.array(["a"]), pa.array(["b"])], names=["id", "region", "REGION"]
    )
    with pytest.raises(ValueError, match="more than once"):
        conn.open_table(loc).plan_write().write(data)


def test_commit_metadata_structured_values_are_recorded_as_json(conn: Any) -> None:
    """str() recorded Python's spelling -- "True", "{'a': 1}" -- in a JSON log."""
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write(commit_metadata={"flag": True, "tags": {"a": 1}})
    plan.commit([plan.write(_rows())])
    entry = conn.open_table(loc).history()[0]
    assert entry["flag"] == "true"
    assert entry["tags"] == '{"a": 1}'


def test_racing_distributed_appends_all_land_by_default(conn: Any) -> None:
    """With retries=0 by default, all but one of several racing appends raised."""
    from concurrent.futures import ThreadPoolExecutor

    loc = _table(conn)
    plans = [conn.open_table(loc).plan_write() for _ in range(6)]
    fragments = [plan.write(_rows(1, i)) for i, plan in enumerate(plans)]
    with ThreadPoolExecutor(6) as pool:
        versions = list(pool.map(lambda i: plans[i].commit([fragments[i]]), range(6)))
    assert sorted(versions) == [1, 2, 3, 4, 5, 6]
    assert conn.open_table(loc).count() == 6


# ------------------------------------------------------------------ catalog commit statuses


@pytest.fixture
def uc_conn(tmp_path: Any) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    from tests.fake_uc import FakeUnityCatalog

    with FakeUnityCatalog(staging_root=str(tmp_path)) as uc:
        conn = Connection(
            catalog=OSSUnityCatalog(uc.url),
            router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
        )
        conn.create_catalog("main")
        conn.create_schema("main.sales")
        conn.create_table("main.sales.cm", pa.schema([("id", pa.int64())]))
        yield uc, conn


@pytest.mark.parametrize(
    ("status", "native"),
    [
        (401, "CatalogPermissionError"),
        (403, "CatalogPermissionError"),
        (404, "CatalogNotFoundError"),
        (400, "CatalogCommitError"),
    ],
)
def test_catalog_refusals_carry_a_typed_native_error(
    uc_conn: Any, status: int, native: str
) -> None:
    """Only 409 and 429 were classified; every other status was a bare ValueError."""
    from deltaswamp import _native

    uc, conn = uc_conn
    plan = conn.table("main.sales.cm").plan_write()
    fragment = plan.write(pa.table({"id": [1]}))
    uc.next_commit_status = status
    with pytest.raises(ds.DeltaSwampError) as caught:
        plan.commit([fragment])
    chain = [caught.value, caught.value.__cause__]
    assert any(isinstance(e, getattr(_native, native)) for e in chain), chain
    assert conn.table("main.sales.cm").count() == 0


def test_a_catalog_5xx_is_transient_and_not_retried_blindly(uc_conn: Any) -> None:
    """A 5xx may have been ratified; it is transient, and a catalog table is not re-sent."""
    uc, conn = uc_conn
    plan = conn.table("main.sales.cm").plan_write()
    fragment = plan.write(pa.table({"id": [1]}))
    uc.next_commit_status = 503
    with pytest.raises(TransientCommitError, match="may or may not"):
        plan.commit([fragment], retries=3)


def test_native_input_errors_are_a_distinct_value_error(conn: Any) -> None:
    from deltaswamp import _native

    loc = _table(conn)
    snapshot = (
        conn.open_table(loc)
        .plan_write()
        .engine.snapshot(conn.open_table(loc)._resolved, write=True)
    )
    with pytest.raises(_native.InvalidInputError):
        snapshot.commit_files([b"junk"])
    assert issubclass(_native.InvalidInputError, ValueError)


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "CredentialError"), (403, "PreflightError"), (404, "InvalidReferenceError")],
)
def test_catalog_refusals_become_the_right_library_error(
    uc_conn: Any, status: int, kind: str
) -> None:
    """A 401 is worded "Authentication failed" with no status, and was a bare ValueError."""
    uc, conn = uc_conn
    plan = conn.table("main.sales.cm").plan_write()
    fragment = plan.write(pa.table({"id": [1]}))
    uc.next_commit_status = status
    with pytest.raises(ds.DeltaSwampError) as caught:
        plan.commit([fragment])
    assert type(caught.value).__name__ == kind


def test_plan_scan_refuses_an_unknown_column_on_the_driver(conn: Any) -> None:
    """It planned fine and then failed on every worker separately."""
    loc = _table(conn)
    conn.open_table(loc).append(_rows())
    with pytest.raises(InvalidArgumentError, match="nope"):
        conn.open_table(loc).plan_scan(columns=["nope"])
    assert conn.open_table(loc).plan_scan(columns=["ID"]).read().num_rows == 1


def test_plan_scan_refuses_an_unevaluable_predicate_on_the_driver(conn: Any) -> None:
    from deltaswamp.predicate import PredicateError

    loc = _table(conn)
    conn.open_table(loc).append(_rows())
    with pytest.raises(PredicateError, match="nope"):
        conn.open_table(loc).plan_scan(predicate="nope = 3")


def test_commit_operation_is_never_blank_or_unknown(conn: Any) -> None:
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    for bad in ("", "  ", 5):
        with pytest.raises(InvalidArgumentError, match="operation"):
            plan.commit([], operation=bad)
    plan.commit([], operation=None)
    assert conn.open_table(loc).history()[0]["operation"] == "WRITE"


def test_a_txn_given_as_a_list_commits(conn: Any) -> None:
    """The check accepts a list; the binding only a tuple -- TypeError at commit."""
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write(txn=["job", 1])
    plan.commit([plan.write(_rows())])
    assert conn.open_table(loc).txn_version("job") == 1


def test_a_catalog_managed_append_with_a_list_txn_commits(uc_conn: Any) -> None:
    """Table.append on a kernel-only table: the same list-vs-tuple TypeError."""
    _, conn = uc_conn
    t = conn.table("main.sales.cm")
    t.append(pa.table({"id": [1]}), txn=["job", 1])
    assert conn.table("main.sales.cm").txn_version("job") == 1


def test_ray_input_files_are_full_paths(conn: Any) -> None:
    """Ray's input_files() listed bare relative names from the log."""
    pytest.importorskip("ray")
    from deltaswamp.distributed import DeltaSwampDatasource

    loc = _table(conn, partition_by=["region"])
    conn.open_table(loc).append(pa.table({"id": [1], "region": ["a b"]}))
    source = DeltaSwampDatasource(conn.open_table(loc).plan_scan())
    (task,) = source.get_read_tasks(1)
    (path,) = task.metadata.input_files
    assert path.startswith(loc.rstrip("/") + "/region="), path
    assert os.path.exists(path), path


def test_committing_the_same_fragments_again_is_refused(conn: Any) -> None:
    """A restarted driver re-committing saved fragments re-added every file."""
    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    fragment = plan.write(_rows(2))
    plan.commit([fragment])
    with pytest.raises(UnreachableTableError, match="already in the table"):
        conn.open_table(loc).plan_write().commit([fragment])
    t = conn.open_table(loc)
    assert (t.version, t.count()) == (1, 2)


# ------------------------------------------------------------------ what a plan ships


_SECRET = "dapi-SECRET-CATALOG-TOKEN-0123456789"


@pytest.fixture
def token_conn(tmp_path: Any) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog
    from deltaswamp.connection import Connection
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    from tests.fake_uc import FakeUnityCatalog

    with FakeUnityCatalog(staging_root=str(tmp_path)) as uc:
        conn = Connection(
            catalog=OSSUnityCatalog(uc.url, token=_SECRET),
            router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
        )
        conn.create_catalog("main")
        conn.create_schema("main.sales")
        conn.create_table("main.sales.cm", pa.schema([("id", pa.int64())]))
        yield conn


@pytest.mark.parametrize("pickler", ["pickle", "cloudpickle"])
def test_a_pickled_plan_never_carries_the_catalog_token(token_conn: Any, pickler: str) -> None:
    """Providers pickled their catalog auth (a PAT, a client secret) into every task."""
    import pickle

    dumps: Any = pickle.dumps
    if pickler == "cloudpickle":
        dumps = pytest.importorskip("ray.cloudpickle").dumps
    t = token_conn.table("main.sales.cm")
    write_plan, scan_plan = t.plan_write(), t.plan_scan()
    for plan in (write_plan, scan_plan):
        assert _SECRET.encode() not in dumps(plan)

    # ...and the worker copy still works: it writes with the shipped storage
    # credential, while the driver's own copy commits through the catalog.
    worker = pickle.loads(pickle.dumps(write_plan))
    assert write_plan.commit([worker.write(pa.table({"id": [1]}))]) == 1
    fresh = token_conn.table("main.sales.cm").plan_scan()  # a Table captures its tail
    assert pickle.loads(pickle.dumps(fresh)).read().num_rows == 1


def test_a_worker_plan_cannot_commit_to_the_catalog(token_conn: Any) -> None:
    import pickle

    plan = token_conn.table("main.sales.cm").plan_write()
    worker = pickle.loads(pickle.dumps(plan))
    fragment = worker.write(pa.table({"id": [1]}))
    with pytest.raises(UnreachableTableError, match="Unity Catalog credentials"):
        worker.commit([fragment])


def test_catalog_auth_ships_only_when_asked(token_conn: Any) -> None:
    import pickle

    t = token_conn.table("main.sales.cm")
    assert _SECRET.encode() in pickle.dumps(t.plan_write(ship_catalog_auth=True))
    assert _SECRET.encode() in pickle.dumps(t.plan_scan(ship_catalog_auth=True))


def test_an_expiring_shipped_credential_is_a_clear_error_on_the_worker(
    token_conn: Any,
) -> None:
    import dataclasses
    import time

    from deltaswamp.distributed import ShippedCredentials
    from deltaswamp.errors import CredentialError

    plan = token_conn.table("main.sales.cm").plan_write()
    import pickle

    worker = pickle.loads(pickle.dumps(plan))
    shipped = worker.table.credential_provider
    assert isinstance(shipped, ShippedCredentials)
    stale = dataclasses.replace(shipped.credentials(), expires_at=time.time() + 5)
    worker = dataclasses.replace(
        worker,
        table=dataclasses.replace(
            worker.table, credential_provider=ShippedCredentials(stale, shipped.table_id)
        ),
    )
    with pytest.raises(CredentialError, match="Re-plan on the driver"):
        worker.write(pa.table({"id": [1]}))


# ------------------------------------------------------------------ orphans


def test_a_write_that_fails_partway_removes_the_files_it_wrote(conn: Any) -> None:
    """Groups written before the failing one were left behind, unowned."""
    import glob

    loc = _table(conn, pa.schema([("id", pa.int64()), ("p", pa.binary())]), partition_by=["p"])
    plan = conn.open_table(loc).plan_write()
    data = pa.table({"id": [1, 2, 3], "p": pa.array([b"a", b"b", b"\xff"], pa.binary())})
    with pytest.raises(ValueError, match="UTF-8"):
        plan.write(data)
    assert glob.glob(os.path.join(loc, "**", "*.parquet"), recursive=True) == []
    assert os.path.isdir(os.path.join(loc, "p=a")), "the test must have written a file first"


def test_a_databricks_provider_pat_does_not_travel_with_a_plan(conn: Any, monkeypatch: Any) -> None:
    """The provider pickles its host/token config; a plan must ship neither."""
    import dataclasses
    import pickle

    from deltaswamp.credentials import Credentials
    from deltaswamp.credentials.base import Cloud
    from deltaswamp.credentials.databricks import DatabricksCredentialProvider

    pat = "dapi0123456789abcdefSECRET"
    provider = DatabricksCredentialProvider(
        "tid", host="https://example.cloud.databricks.com", token=pat
    )
    assert pat.encode() in pickle.dumps(provider), "premise: the provider itself carries it"
    vended = Credentials(cloud=Cloud.LOCAL, url="file:///x", expires_at=None)
    monkeypatch.setattr(provider, "credentials", lambda *_a, **_k: vended)

    loc = _table(conn)
    plan = conn.open_table(loc).plan_write()
    plan = dataclasses.replace(
        plan, table=dataclasses.replace(plan.table, credential_provider=provider)
    )
    assert pat.encode() not in pickle.dumps(plan)
    scan = conn.open_table(loc).plan_scan()
    scan = dataclasses.replace(
        scan, table=dataclasses.replace(scan.table, credential_provider=provider)
    )
    assert pat.encode() not in pickle.dumps(scan)
