"""End-to-end catalog flows through the fake Unity Catalog (wave 4).

Every test drives the public `Connection`/`Table` surface against
`tests/fake_uc_strict.py`, which adds the server-side checks a real metastore
makes (409 for a taken version, `assert-table-uuid`, the unbackfilled cap,
per-operation credential vending) to `tests/fake_uc.py`. Where both catalog
clients can speak to it -- OSS Unity Catalog over its REST API, Databricks
through the real databricks-sdk pointed at the fake -- the flow runs on both.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any
from urllib.parse import urlparse

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests.fake_uc import FakeTable, FakeUnityCatalog  # noqa: E402
from tests.fake_uc_strict import StrictUnityCatalog  # noqa: E402

SCHEMA = pa.schema([("id", pa.int64()), ("city", pa.string())])


def _rows(*ids: int) -> Any:
    return pa.table({"id": list(ids), "city": [f"c{i}" for i in ids]})


def _router() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()})


@pytest.fixture
def uc(tmp_path: pathlib.Path) -> Any:
    with StrictUnityCatalog(staging_root=tmp_path / "managed") as server:
        yield server


def _oss(uc: FakeUnityCatalog) -> Any:
    from deltaswamp.catalog.ossuc import OSSUnityCatalog
    from deltaswamp.connection import Connection

    return Connection(catalog=OSSUnityCatalog(uc.url), router=_router())


def _dbx(uc: StrictUnityCatalog, monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("databricks.sdk")
    from deltaswamp.catalog.databricks import DatabricksUnityCatalog
    from deltaswamp.connection import Connection

    for key in list(os.environ):
        if key.startswith("DATABRICKS"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent/databrickscfg")
    return Connection(
        catalog=DatabricksUnityCatalog(host=uc.url, token="dapi-fake"), router=_router()
    )


@pytest.fixture(params=["oss", "databricks"])
def conn(request: Any, uc: StrictUnityCatalog, monkeypatch: pytest.MonkeyPatch) -> Any:
    _oss(uc).create_catalog("main")
    _oss(uc).create_schema("main.sales")
    return _oss(uc) if request.param == "oss" else _dbx(uc, monkeypatch)


@pytest.fixture
def oss(uc: StrictUnityCatalog) -> Any:
    c = _oss(uc)
    c.create_catalog("main")
    c.create_schema("main.sales")
    return c


def _staged_files(location: str) -> list[str]:
    root = pathlib.Path(urlparse(location).path) / "_delta_log" / "_staged_commits"
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def _data_files(location: str) -> list[pathlib.Path]:
    root = pathlib.Path(urlparse(location).path)
    return [p for p in root.rglob("*.parquet") if "_delta_log" not in p.parts]


# --------------------------------------------------------- the same handle


class TestHandleSeesItsOwnCommits:
    def test_second_append_through_the_same_handle(self, conn: Any, uc: Any) -> None:
        t = conn.create_table("main.sales.cm", SCHEMA)
        t.append(_rows(1))
        t.append(_rows(2))  # used to re-commit version 1: a 409 every time
        t.append(_rows(3))
        assert uc.refusals == []
        assert t.count() == 3
        assert conn.table("main.sales.cm").count() == 3

    def test_the_handle_reads_its_own_write(self, conn: Any) -> None:
        t = conn.create_table("main.sales.cm", SCHEMA)
        t.append(_rows(1, 2))
        assert t.count() == 2
        assert t.version == 1

    def test_txn_replay_through_the_same_handle_is_skipped(self, conn: Any, uc: Any) -> None:
        t = conn.create_table("main.sales.cm", SCHEMA)
        t.append(_rows(1), txn=("job", 7))
        t.append(_rows(1), txn=("job", 7))  # the stale snapshot said "never committed"
        assert uc.refusals == []
        assert conn.table("main.sales.cm").count() == 1


class TestStaleHandles:
    def test_a_handle_opened_before_another_writer_can_still_append(
        self, conn: Any, uc: Any
    ) -> None:
        conn.create_table("main.sales.cm", SCHEMA)
        stale = conn.table("main.sales.cm")
        conn.table("main.sales.cm").append(_rows(1))
        stale.append(_rows(2))  # a guaranteed 409 from the captured tail
        stale.append(_rows(3))
        assert uc.refusals == []
        assert sorted(conn.table("main.sales.cm").to_arrow().to_pydict()["id"]) == [1, 2, 3]

    def test_delete_through_a_stale_handle_sees_the_other_writers_rows(self, conn: Any) -> None:
        conn.create_table("main.sales.cm", SCHEMA)
        stale = conn.table("main.sales.cm")
        conn.table("main.sales.cm").append(_rows(1, 2, 3))
        stale.delete("id = 2")
        assert sorted(conn.table("main.sales.cm").to_arrow().to_pydict()["id"]) == [1, 3]

    def test_writing_to_a_dropped_table_is_refused_before_any_file(self, conn: Any) -> None:
        from deltaswamp.errors import InvalidReferenceError

        t = conn.create_table("main.sales.cm", SCHEMA)
        conn.drop_table("main.sales.cm")
        before = _data_files(t.location)
        with pytest.raises(InvalidReferenceError):
            t.append(_rows(1))
        assert _data_files(t.location) == before

    def test_writing_to_a_recreated_table_is_refused(self, conn: Any, uc: Any) -> None:
        from deltaswamp.errors import CorruptTableError

        t = conn.create_table("main.sales.cm", SCHEMA)
        uc.recreate("main.sales.cm")
        with pytest.raises(CorruptTableError, match="re-created"):
            t.append(_rows(1))
        assert "uuid" not in uc.refusals  # refused before the commit reached the catalog


# --------------------------------------------------------------- backfill


class TestBackfillPressure:
    @pytest.fixture
    def capped(self, tmp_path: pathlib.Path) -> Any:
        with StrictUnityCatalog(staging_root=tmp_path / "m", max_unbackfilled=2) as server:
            c = _oss(server)
            c.create_catalog("main")
            c.create_schema("main.sales")
            yield server, c

    def test_a_backfill_demand_is_published_and_the_write_retried(self, capped: Any) -> None:
        uc, conn = capped
        t = conn.create_table("main.sales.cm", SCHEMA)
        for i in range(6):
            t.append(_rows(i))
        assert "backfill" in uc.refusals
        assert conn.table("main.sales.cm").count() == 6
        # The table never wedged: the tail stays under the cap.
        assert len(uc.tables["main.sales.cm"].commits) <= 2

    def test_delete_under_backfill_pressure(self, capped: Any) -> None:
        uc, conn = capped
        t = conn.create_table("main.sales.cm", SCHEMA)
        t.append(_rows(1, 2))
        t.append(_rows(3))
        t.delete("id = 1")
        assert "backfill" in uc.refusals
        assert sorted(conn.table("main.sales.cm").to_arrow().to_pydict()["id"]) == [2, 3]

    def test_a_stream_is_published_but_not_replayed(self, capped: Any) -> None:
        from deltaswamp.errors import BackfillRequiredError

        _, conn = capped
        t = conn.create_table("main.sales.cm", SCHEMA)
        t.append(_rows(1))
        t.append(_rows(2))
        with pytest.raises(BackfillRequiredError, match="published"):
            t.append(_rows(3).to_reader())
        # Published, so the next write goes through without another refusal.
        t.append(_rows(3))
        assert conn.table("main.sales.cm").count() == 3


# ----------------------------------------------------------- distributed


class TestDistributedCommit:
    def test_commit_after_a_concurrent_append(self, conn: Any, uc: Any) -> None:
        conn.create_table("main.sales.cm", SCHEMA)
        plan = conn.table("main.sales.cm").plan_write()
        fragment = plan.write(_rows(1))
        conn.table("main.sales.cm").append(_rows(2))
        plan.commit([fragment])  # the tail captured at planning was a guaranteed 409
        assert sorted(conn.table("main.sales.cm").to_arrow().to_pydict()["id"]) == [1, 2]

    def test_guarded_overwrite_sees_that_the_table_moved(self, conn: Any) -> None:
        from deltaswamp.errors import UnreachableTableError

        conn.create_table("main.sales.cm", SCHEMA)
        conn.table("main.sales.cm").append(_rows(1))
        plan = conn.table("main.sales.cm").plan_write(mode="overwrite")
        fragment = plan.write(_rows(9))
        conn.table("main.sales.cm").append(_rows(2))
        # The stale tail hid the new version, so this check always passed.
        with pytest.raises(UnreachableTableError, match="discard"):
            plan.commit([fragment])

    def test_commit_to_a_recreated_table_is_refused(self, conn: Any, uc: Any) -> None:
        from deltaswamp.errors import CorruptTableError

        conn.create_table("main.sales.cm", SCHEMA)
        plan = conn.table("main.sales.cm").plan_write()
        fragment = plan.write(_rows(1))
        uc.recreate("main.sales.cm")
        with pytest.raises(CorruptTableError, match="re-created"):
            plan.commit([fragment])


# ------------------------------------------------------------ time travel


class TestTimeTravel:
    def test_a_version_past_the_ratified_one_is_a_library_error(self, conn: Any) -> None:
        from deltaswamp.errors import UnreachableTableError

        t = conn.create_table("main.sales.cm", SCHEMA)
        t.append(_rows(1))
        with pytest.raises(UnreachableTableError, match="no such version"):
            t.to_arrow(version=5)


# ------------------------------------------------------ external tables


class TestExternalTables:
    def test_create_over_an_existing_log_is_a_library_error(
        self, conn: Any, tmp_path: pathlib.Path
    ) -> None:
        from deltaswamp.errors import UnreachableTableError

        location = f"file://{tmp_path / 'ext'}"
        conn.create_table("main.sales.a", SCHEMA, location=location)
        with pytest.raises(UnreachableTableError, match="register_table"):
            conn.create_table("main.sales.b", SCHEMA, location=location)


# ----------------------------------------------------------- shallow clones


class TestShallowClones:
    @pytest.mark.parametrize("kind", ["MANAGED_SHALLOW_CLONE", "EXTERNAL_SHALLOW_CLONE"])
    def test_rewrites_are_refused_like_reads(
        self, oss: Any, uc: Any, tmp_path: pathlib.Path, kind: str
    ) -> None:
        from deltalake import write_deltalake
        from deltaswamp.capability import Operation

        path = tmp_path / "clone"
        write_deltalake(str(path), _rows(1, 2))
        uc.add_table(
            "main.sales.clone",
            FakeTable(name="clone", location=str(path), table_id="c1", table_type=kind),
        )
        t = oss.table("main.sales.clone")
        assert not t.can(Operation.SCAN).ok
        for op in (Operation.DELETE, Operation.UPDATE, Operation.REPLACE_WHERE):
            verdict = t.can(op)
            assert not verdict.ok, op
            assert "shallow clone" in str(verdict.reason)


# ------------------------------------------------------------ credentials


class TestOssCredentials:
    def test_table_credentials_say_what_they_were_vended_for(self, oss: Any, uc: Any) -> None:
        from deltaswamp.credentials.base import Operation

        t = oss.create_table("main.sales.cm", SCHEMA)
        assert t.credentials().operation is Operation.READ
        assert t.credentials(write=True).operation is Operation.READ_WRITE

    def test_a_read_path_credential_refuses_a_write(self, oss: Any, tmp_path: pathlib.Path) -> None:
        from deltaswamp.credentials.base import Operation, StaticCredentialProvider
        from deltaswamp.errors import CredentialError

        creds = oss.catalog.path_credentials(f"file://{tmp_path}", "PATH_READ")
        assert creds.operation is Operation.READ
        with pytest.raises(CredentialError, match="read-only"):
            StaticCredentialProvider(creds).credentials(Operation.READ_WRITE)
        create = oss.catalog.path_credentials(f"file://{tmp_path}", "PATH_CREATE_TABLE")
        assert create.operation is Operation.READ_WRITE

    def test_a_short_lived_credential_is_not_revended_every_call(self, oss: Any, uc: Any) -> None:
        import warnings

        uc.credential_ttl = 60  # under the 300s default refresh margin
        t = oss.create_table("main.sales.cm", SCHEMA)
        before = len(uc.credential_operations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(3):
                t.append(_rows(i))
            assert t.count() == 3
        vends = uc.credential_operations[before:]
        assert sorted(set(vends)) == ["READ", "READ_WRITE"]
        assert len(vends) == 2, vends


class TestRejectedCredentialIsRevended:
    def test_storage_refusal_invalidates_and_retries_once(
        self, oss: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp import _native

        t = oss.create_table("main.sales.cm", SCHEMA)
        t.append(_rows(1))
        provider = t.resolved.credential_provider
        invalidated: list[bool] = []
        real_invalidate = provider.invalidate

        def recording_invalidate() -> Any:
            invalidated.append(True)
            return real_invalidate()

        monkeypatch.setattr(provider, "invalidate", recording_invalidate)
        real = _native.Snapshot.resolve
        calls: list[int] = []

        def flaky(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise OSError("Generic S3 error: ExpiredToken: The provided token has expired")
            return real(*args, **kwargs)

        from deltaswamp.capability import Engine

        monkeypatch.setattr(_native.Snapshot, "resolve", staticmethod(flaky))
        engine = oss.router.engines[Engine.KERNEL]
        assert engine.snapshot(t.resolved).version == 1
        assert invalidated == [True]
        assert len(calls) == 2


# ----------------------------------------------------- commit HTTP errors


class TestCommitHttpErrors:
    @pytest.mark.parametrize(
        ("status", "kind"),
        [(404, "InvalidReferenceError"), (401, "CredentialError"), (403, "PreflightError")],
    )
    def test_statuses_become_library_errors(self, status: int, kind: str) -> None:
        from deltaswamp.engine.kernel import _uc_commit_http_error

        message = (
            "Generic delta kernel error: UC update_table error: HTTP error "
            f'(status {status}): {{"message": "nope"}}'
        )
        error = _uc_commit_http_error(message)
        assert type(error).__name__ == kind

    def test_other_errors_are_left_alone(self) -> None:
        from deltaswamp.engine.kernel import _uc_commit_http_error

        assert _uc_commit_http_error("some unrelated ValueError") is None
        assert _uc_commit_http_error("UC update_table error: HTTP error (status 500)") is None

    def test_a_commit_refused_with_404_is_an_invalid_reference(self, oss: Any, uc: Any) -> None:
        from deltaswamp.errors import InvalidReferenceError

        t = oss.create_table("main.sales.cm", SCHEMA)
        uc.next_commit_status = 404
        with pytest.raises(InvalidReferenceError, match="dropped"):
            t.append(_rows(1))


# ----------------------------------------------------- resolution races


class TestRecreatedWhileResolving:
    def test_tables_api_and_delta_api_disagreeing_re_resolves(
        self, oss: Any, uc: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.identity import parse_ref

        oss.create_table("main.sales.cm", SCHEMA)
        real = uc._table_info
        flipped: list[str] = []

        def stale_once(table: Any, full_name: str | None = None) -> dict[str, Any]:
            info: dict[str, Any] = real(table, full_name)
            if full_name == "main.sales.cm" and not flipped:
                flipped.append(info["table_id"])
                info["table_id"] = "an-older-incarnation"
            return info

        monkeypatch.setattr(uc, "_table_info", stale_once)
        resolved = oss.catalog.resolve(parse_ref("main.sales.cm"))
        assert resolved.table_id == flipped[0]


# ------------------------------------------------------- lost commit races


class TestCatalogManagedAppendRace:
    def test_an_append_that_loses_the_race_is_restaged(self, conn: Any, uc: Any) -> None:
        t = conn.create_table("main.sales.cm", SCHEMA)
        uc.next_commit_status = 409  # someone else took the version first
        t.append(_rows(1))
        assert conn.table("main.sales.cm").count() == 1

    def test_without_retries_the_conflict_surfaces(self, conn: Any, uc: Any) -> None:
        from deltaswamp.errors import CommitConflictError

        t = conn.create_table("main.sales.cm", SCHEMA)
        uc.next_commit_status = 409
        with pytest.raises(CommitConflictError):
            t.append(_rows(1), max_commit_retries=0)

    def test_a_stream_is_not_replayed(self, conn: Any, uc: Any) -> None:
        from deltaswamp.errors import CommitConflictError

        t = conn.create_table("main.sales.cm", SCHEMA)
        uc.next_commit_status = 409
        with pytest.raises(CommitConflictError):
            t.append(_rows(1).to_reader())


# --------------------------------------------------- query-defined objects


class TestQueryDefinedTablesRefuseWrites:
    @pytest.mark.parametrize("kind", ["VIEW", "MATERIALIZED_VIEW", "METRIC_VIEW"])
    @pytest.mark.parametrize("fallback", [False, True])
    def test_a_write_names_the_real_cause(self, kind: str, fallback: bool) -> None:
        from deltaswamp.capability import Engine, Operation
        from deltaswamp.catalog.base import ResolvedTable, TableType
        from deltaswamp.engine.kernel import KernelEngine
        from deltaswamp.identity import parse_ref
        from deltaswamp.router import Router

        engines: dict[Any, Any] = {Engine.KERNEL: KernelEngine()}
        if fallback:
            engines[Engine.SQL] = object()  # never reached: the refusal comes first
        router = Router(engines=engines, allow_sql_fallback=fallback)
        view = ResolvedTable(
            ref=parse_ref("main.sales.v"),
            location="s3://b/v" if kind == "MATERIALIZED_VIEW" else None,
            table_type=TableType(kind),
            external_read_supported=True if kind == "MATERIALIZED_VIEW" else None,
        )
        for op in (Operation.APPEND, Operation.DELETE, Operation.MERGE):
            verdict = router.capability(op, view)
            assert not verdict.ok
            assert "defined by a query" in str(verdict.reason)
            assert "allow_sql_fallback" not in str(verdict.remedy or "")


# ------------------------------------------------------------ Delta Sharing


@pytest.fixture
def sharing_conn(tmp_path: pathlib.Path) -> Any:
    import json

    pytest.importorskip("delta_sharing")
    from tests.unit import test_sharing as ts

    server = ts.FakeSharingServer().start()
    server.add(
        ts.FakeSharedTable(
            share="retail",
            schema="sales",
            name="orders",
            schema_string=ts.SCHEMA_STRING,
            partition_columns=["day"],
            versions=[
                ts.FakeVersion(0, ts._ms(ts.T0), [ts._rows([1, 2], "2024-01-01")]),
                ts.FakeVersion(1, ts._ms(ts.T1), [ts._rows([3], "2024-01-02")]),
            ],
        )
    )
    profile = tmp_path / "config.share"
    profile.write_text(
        json.dumps(
            {"shareCredentialsVersion": 1, "endpoint": server.endpoint, "bearerToken": ts.TOKEN}
        )
    )
    try:
        yield ds.connect(f"sharing://{profile}")
    finally:
        server.stop()


class TestSharingFlows:
    def test_a_shared_table_exists(self, sharing_conn: Any) -> None:
        # No storage location by design; that alone reported every one absent.
        assert sharing_conn.table_exists("retail.sales.orders")
        assert not sharing_conn.table_exists("retail.sales.nope")

    def test_a_missing_version_is_not_blamed_on_history_sharing(self, sharing_conn: Any) -> None:
        from deltaswamp.errors import UnreachableTableError

        with pytest.raises(UnreachableTableError) as caught:
            sharing_conn.table("retail.sales.orders", version=9).count()
        assert "ALTER SHARE" not in str(caught.value)


class TestOssErrorsNameTheCause:
    def test_a_vending_refusal_is_a_credential_error(self, oss: Any, uc: Any) -> None:
        from deltaswamp.credentials.base import Operation
        from deltaswamp.errors import CredentialError

        t = oss.create_table("main.sales.cm", SCHEMA)
        uc.vending_refused.add("main.sales.cm")
        provider = t.resolved.credential_provider
        provider.invalidate()
        with pytest.raises(CredentialError, match="MODIFY"):
            provider.credentials(Operation.READ_WRITE)

    def test_vending_for_a_dropped_table_says_re_resolve(self, oss: Any) -> None:
        from deltaswamp.errors import CredentialError

        t = oss.create_table("main.sales.cm", SCHEMA)
        provider = t.resolved.credential_provider
        oss.drop_table("main.sales.cm")
        provider.invalidate()
        with pytest.raises(CredentialError, match="re-resolve"):
            provider.credentials()

    def test_a_401_is_not_reported_as_missing_privileges(
        self, oss: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.catalog import ossuc
        from deltaswamp.errors import PreflightError

        def rejected(*args: Any, **kwargs: Any) -> Any:
            raise ossuc.UnityCatalogHTTPError("GET x failed with HTTP 401: token expired", 401)

        monkeypatch.setattr(ossuc, "_request", rejected)
        with pytest.raises(PreflightError) as caught:
            oss.table("main.sales.cm")
        assert "rejected the credentials" in str(caught.value)
        assert "USE CATALOG" not in str(caught.value)


class TestDatabricksStaging:
    def test_staging_s3_credentials_carry_the_metastore_region(
        self, uc: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deltaswamp.identity import parse_ref

        _oss(uc).create_catalog("main")
        _oss(uc).create_schema("main.sales")
        uc.metastore_region = "eu-west-1"
        uc.staging_credential_config = {
            "s3.access-key-id": "AKIA",
            "s3.secret-access-key": "secret",
            "s3.session-token": "token",
        }
        dbx = _dbx(uc, monkeypatch)
        staged = dbx.catalog.create_staging_table(parse_ref("main.sales.fresh"))
        assert staged.storage_options["aws_region"] == "eu-west-1"


class TestBadDataIsALibraryError:
    @pytest.mark.parametrize(
        "data",
        [
            pa.table({"id": [1], "city": ["a"], "extra": [1]}),
            pa.table({"id": ["x"], "city": ["a"]}),
        ],
        ids=["extra-column", "uncastable"],
    )
    def test_on_a_catalog_managed_table(self, oss: Any, data: Any) -> None:
        from deltaswamp.errors import InvalidArgumentError

        t = oss.create_table("main.sales.cm", SCHEMA)
        with pytest.raises(InvalidArgumentError):
            t.append(data)


class TestIdentityCheck:
    def test_a_log_whose_metadata_id_differs_from_the_uc_id_still_opens(self, oss: Any) -> None:
        """Another writer's managed table: Metadata.id is its own UUID, and the
        catalog's id lives in io.unitycatalog.tableId (what UCCommitter checks)."""
        import json
        import uuid

        t = oss.create_table("main.sales.cm", SCHEMA)
        first = pathlib.Path(urlparse(t.location).path) / "_delta_log" / f"{0:020d}.json"
        actions = [json.loads(line) for line in first.read_text().splitlines() if line.strip()]
        for action in actions:
            if "metaData" in action:
                assert action["metaData"]["configuration"]["io.unitycatalog.tableId"]
                action["metaData"]["id"] = str(uuid.uuid4())
        first.write_text("\n".join(json.dumps(a) for a in actions) + "\n")
        reopened = oss.table("main.sales.cm")
        reopened.append(_rows(1))
        assert oss.table("main.sales.cm").count() == 1

    def test_a_re_created_table_is_still_caught(self, oss: Any, uc: Any) -> None:
        from deltaswamp.errors import CorruptTableError

        oss.create_table("main.sales.cm", SCHEMA)
        uc.recreate("main.sales.cm")  # new catalog id, same (old) log
        with pytest.raises(CorruptTableError, match="re-created"):
            oss.table("main.sales.cm").count()
