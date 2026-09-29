"""Distributed reads through the warehouse: result chunks as splits (`warehouse_scan`)."""

from __future__ import annotations

import dataclasses
import io
import pickle
import time
import urllib.error
from email.message import Message
from types import SimpleNamespace
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("databricks.sdk")

from deltaswamp import warehouse_scan  # noqa: E402
from deltaswamp.catalog import ResolvedTable  # noqa: E402
from deltaswamp.credentials import CredentialBroker  # noqa: E402
from deltaswamp.engine.sql import SqlEngine, SqlFallbackWarning  # noqa: E402
from deltaswamp.engine.sql_backend import SdkStatementBackend, SqlStatementError  # noqa: E402
from deltaswamp.errors import UnreachableTableError  # noqa: E402
from deltaswamp.identity import parse_ref  # noqa: E402

from tests.unit.sql_fakes import FakeClient, FakeStatements, succeeded  # noqa: E402

TABLE = ResolvedTable(ref=parse_ref("main.sales.orders"), location=None)
LATER = "2999-01-01T00:00:00Z"
PAST = "2000-01-01T00:00:00Z"


def _ipc(table: Any) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _chunk(i: int) -> Any:
    return pa.table({"id": pa.array([10 * i + j for j in range(i + 1)], pa.int64())})


def _link(i: int, expiration: str = LATER, host: str = "store.example.com") -> Any:
    return SimpleNamespace(
        external_link=f"https://{host}/r/{i}?sig=SECRET{i}",
        http_headers={"x-ms-blob-type": "BlockBlob"},
        expiration=expiration,
        chunk_index=i,
        next_chunk_index=None,
    )


def _manifest(n: int, *, truncated: bool = False) -> Any:
    chunks = [
        SimpleNamespace(chunk_index=i, row_count=i + 1, byte_count=100 * (i + 1)) for i in range(n)
    ]
    column = SimpleNamespace(name="id", type_name="LONG", type_text="BIGINT", position=0)
    return SimpleNamespace(
        chunks=chunks,
        total_chunk_count=n,
        total_row_count=sum(i + 1 for i in range(n)),
        truncated=truncated,
        schema=SimpleNamespace(columns=[column]),
    )


class _Store:
    """Serves chunk bytes by URL; can answer an expired link with 403."""

    def __init__(self, n: int) -> None:
        self.data = {f"https://store.example.com/r/{i}": _ipc(_chunk(i)) for i in range(n)}
        self.expired: set[str] = set()
        self.requests: list[Any] = []

    def __call__(self, request: Any, timeout: float) -> Any:
        self.requests.append(request)
        base = request.full_url.split("?")[0]
        if request.full_url in self.expired:
            raise urllib.error.HTTPError(request.full_url, 403, "expired", Message(), None)
        return io.BytesIO(self.data[base])


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    s = _Store(4)
    monkeypatch.setattr(warehouse_scan, "opener", s)
    return s


def _engine(n: int = 4, first: list[Any] | None = None, **manifest: Any) -> tuple[Any, Any]:
    response = succeeded(
        "stmt-1",
        result=SimpleNamespace(external_links=first if first is not None else [_link(0)]),
        manifest=_manifest(n, **manifest),
    )
    statements = FakeStatements(
        [response],
        chunks={i: SimpleNamespace(external_links=[_link(i)]) for i in range(n)},
    )
    backend = SdkStatementBackend(FakeClient(statements=statements), "wh")
    return SqlEngine(backend=backend, client=FakeClient(), warn_on_use=False), statements


def _worker(plan: Any) -> Any:
    return pickle.loads(pickle.dumps(plan))


def test_every_chunk_is_a_split_with_its_link(store: _Store) -> None:
    engine, statements = _engine()
    plan = engine.plan_scan_chunks(TABLE, columns=["id"], predicate="id > 0")
    assert [s.chunk_index for s in plan.splits] == [0, 1, 2, 3]
    assert [s.row_count for s in plan.splits] == [1, 2, 3, 4]
    # Chunk 0's link came with the response; the rest were fetched once each.
    assert sorted(statements.chunk_requests) == [1, 2, 3]
    assert all(len(s.links) == 1 for s in plan.splits)
    assert statements.executed[0]["statement"].startswith("SELECT `id` FROM")
    assert "WHERE id > 0" in statements.executed[0]["statement"]


def test_a_worker_reads_its_chunks_from_the_pickled_plan(store: _Store) -> None:
    engine, _ = _engine()
    plan = _worker(engine.plan_scan_chunks(TABLE))
    assert plan.driver_links is None and plan.catalog is None
    groups = plan.partitions(2)
    assert sorted(s.chunk_index for g in groups for s in g) == [0, 1, 2, 3]
    ids = sorted(i for g in groups for i in plan.read(g).column("id").to_pylist())
    assert ids == sorted(i for c in range(4) for i in _chunk(c).column("id").to_pylist())
    # Only the link's own headers went to the store.
    assert all(dict(r.header_items()) == {"X-ms-blob-type": "BlockBlob"} for r in store.requests), [
        dict(r.header_items()) for r in store.requests
    ]


def test_partitions_balance_bytes() -> None:
    splits = tuple(
        warehouse_scan.WarehouseSplit("s", i, 1, size) for i, size in enumerate([900, 100, 500])
    )
    plan = warehouse_scan.WarehouseScanPlan(splits=splits, schema_ipc=b"", statement_id="s")
    groups = plan.partitions(2)
    assert sorted(sum(s.size for s in g) for g in groups) == [600, 900]


def test_an_expired_link_without_a_way_to_refresh_says_how(store: _Store) -> None:
    engine, _ = _engine(first=[_link(0, PAST)])
    plan = _worker(engine.plan_scan_chunks(TABLE))
    with pytest.raises(UnreachableTableError, match="credential_source"):
        plan.read(plan.splits[:1])


def test_workers_refresh_links_through_a_broker(store: _Store) -> None:
    engine, statements = _engine()
    plan = engine.plan_scan_chunks(TABLE, prefetch_links=False)
    assert statements.chunk_requests == []  # only chunk 0's link, from the response
    broker = CredentialBroker()
    assert broker.add(plan) == "sql-statement:stmt-1"
    # As a worker holds it: no driver fetcher, a source reaching the broker
    # (in one process the broker itself; with Ray, an actor handle's wrapper).
    shipped = dataclasses.replace(_worker(plan), credential_source=broker, driver_links=None)
    assert shipped.read().num_rows == 1 + 2 + 3 + 4
    assert sorted(statements.chunk_requests) == [1, 2, 3]


def test_a_link_the_store_refuses_is_fetched_again(store: _Store) -> None:
    engine, statements = _engine()
    plan = engine.plan_scan_chunks(TABLE)
    store.expired.add(plan.splits[2].links[0].url)
    fresh = _link(2)
    fresh.external_link = "https://store.example.com/r/2?sig=FRESH"
    statements._chunks[2] = SimpleNamespace(external_links=[fresh])
    # On the driver, its own fetcher refreshes.
    assert plan.read(plan.splits[2:3]).num_rows == 3


@pytest.mark.parametrize(
    "url",
    [
        "http://store.example.com/r/0",
        "https://169.254.169.254/latest/meta-data",
        "https://127.0.0.1/r/0",
        "file:///etc/passwd",
    ],
)
def test_links_to_private_hosts_or_plain_http_are_refused(url: str, store: _Store) -> None:
    link = warehouse_scan.ChunkLink(url=url)
    split = warehouse_scan.WarehouseSplit("s", 0, 1, 1, links=(link,))
    plan = warehouse_scan.WarehouseScanPlan(splits=(split,), schema_ipc=b"", statement_id="s")
    with pytest.raises(UnreachableTableError, match=r"https|private"):
        plan.read()
    assert store.requests == []


def test_a_truncated_result_is_refused_at_planning(store: _Store) -> None:
    engine, _ = _engine(truncated=True)
    with pytest.raises(SqlStatementError, match="100 GiB"):
        engine.plan_scan_chunks(TABLE)


def test_an_empty_result_plans_no_splits_and_reads_its_columns() -> None:
    engine, _ = _engine(0, first=[])
    plan = engine.plan_scan_chunks(TABLE)
    assert plan.splits == ()
    empty = _worker(plan).read()
    assert empty.num_rows == 0 and empty.column_names == ["id"]


def test_a_chunk_short_of_its_rows_is_an_error(store: _Store) -> None:
    engine, _ = _engine()
    plan = engine.plan_scan_chunks(TABLE)
    store.data["https://store.example.com/r/3"] = _ipc(_chunk(0))
    with pytest.raises(UnreachableTableError, match="held 1 rows"):
        plan.read(plan.splits[3:])


def test_a_failed_statement_raises_before_planning() -> None:
    failed = SimpleNamespace(
        statement_id="stmt-1",
        status=SimpleNamespace(
            state=SimpleNamespace(value="FAILED"),
            error=SimpleNamespace(message="ROW_FILTER boom", error_code="BAD_REQUEST"),
            sql_state=None,
        ),
        result=None,
        manifest=None,
    )
    backend = SdkStatementBackend(FakeClient(statements=FakeStatements([failed])), "wh")
    engine = SqlEngine(backend=backend, client=FakeClient(), warn_on_use=False)
    with pytest.raises(SqlStatementError, match="ROW_FILTER boom"):
        engine.plan_scan_chunks(TABLE)


def test_the_fallback_is_announced(store: _Store) -> None:
    engine, _ = _engine()
    engine._warn_on_use = True
    with pytest.warns(SqlFallbackWarning):
        engine.plan_scan_chunks(TABLE)


def test_a_link_never_prints_its_signature() -> None:
    link = warehouse_scan.ChunkLink.of(_link(5))
    assert "SECRET" not in repr(link)
    split = warehouse_scan.WarehouseSplit("s", 5, 1, 1, links=(link,))
    assert "SECRET" not in repr(split)


def test_link_expiry_is_read_from_the_iso_text() -> None:
    link = warehouse_scan.ChunkLink.of(_link(0, "2030-01-01T00:00:00.000Z"))
    assert link.expires_at == pytest.approx(1893456000.0)
    assert not warehouse_scan.ChunkLink("https://x", expires_at=time.time() + 5).fresh()


def test_a_broker_serves_links_only_as_chunks(store: _Store) -> None:
    engine, _ = _engine()
    plan = engine.plan_scan_chunks(TABLE, prefetch_links=False)
    broker = CredentialBroker()
    key = broker.add(plan)
    assert broker(key, "chunk:2")["links"][0]["url"].endswith("SECRET2")
    from deltaswamp.errors import CredentialError

    with pytest.raises(CredentialError, match="chunk:<index>"):
        broker(key, "READ")
