"""Credentials across a long distributed job: refresh, sharing, and plan-time refusals.

A Ray job outlives its vended storage credential (about an hour), runs many
tasks per worker process, and may target a table Unity Catalog will not vend
write credentials for. Each test here is one way that went wrong.
"""

from __future__ import annotations

import dataclasses
import http.server
import os
import pickle
import tempfile
import threading
import time
import warnings
from typing import Any, ClassVar

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine
from deltaswamp.credentials import Credentials, Operation
from deltaswamp.credentials.base import Cloud
from deltaswamp.errors import CredentialError, CredentialExpiryWarning, ExternalWriteNotAllowedError

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


def _table(conn: Any) -> Any:
    location = os.path.join(tempfile.mkdtemp(), "t")
    conn.create_table(location, pa.schema([("id", pa.int64())]))
    return conn.open_table(location)


def _local(
    expires_in: float | None, tag: str = "a", operation: Operation | None = None
) -> Credentials:
    return Credentials(
        cloud=Cloud.LOCAL,
        url="file:///x",
        expires_at=None if expires_in is None else time.time() + expires_in,
        secrets={"token": tag},
        operation=operation,
    )


class RotatingProvider:
    """A provider handing out a new credential each time it is asked."""

    table_id = "tid-rotating"

    def __init__(self, lifetime: float) -> None:
        self.lifetime = lifetime
        self.calls = 0
        self.lock = threading.Lock()

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        with self.lock:
            self.calls += 1
            return _local(self.lifetime, tag=f"v{self.calls}", operation=operation)

    def invalidate(self) -> None:
        pass


def _with_provider(table: Any, provider: Any) -> Any:
    table._resolved = dataclasses.replace(table._resolved, credential_provider=provider)
    return table


# ------------------------------------------------------------------ refresh


class TestRefresh:
    def test_a_vended_credential_names_a_slot_the_store_refreshes_from(self) -> None:
        from deltaswamp.credentials.refresh import _REFRESHER, SLOT_KEY, slot_options

        provider = RotatingProvider(lifetime=3.0)
        first = provider.credentials()
        options = slot_options(provider, Operation.READ, first, first.as_storage_options())
        slot = options[SLOT_KEY]
        assert _REFRESHER.entries()[slot].options == {"token": "v1"}
        # Due at half its three-second life: the refresher asks again and
        # publishes the new credential without anyone reading.
        deadline = time.time() + 10
        while _REFRESHER.entries()[slot].options == {"token": "v1"}:
            assert time.time() < deadline, "the slot was never refreshed"
            time.sleep(0.1)
        assert _REFRESHER.entries()[slot].options["token"] != "v1"

    def test_a_credential_without_expiry_needs_no_slot(self) -> None:
        from deltaswamp.credentials.refresh import SLOT_KEY, slot_options

        forever = _local(None)
        assert SLOT_KEY not in slot_options(object(), Operation.READ, forever, {"token": "a"})

    def test_a_refresh_does_not_miss_the_snapshot_cache(self) -> None:
        """The cache was keyed on the secret: each refresh replayed the log again."""
        from deltaswamp.credentials.refresh import SLOT_KEY
        from deltaswamp.engine.kernel import _store_fingerprint

        before = {"aws_access_key_id": "A", "aws_session_token": "1", SLOT_KEY: "s:READ"}
        after = {"aws_access_key_id": "B", "aws_session_token": "2", SLOT_KEY: "s:READ"}
        assert _store_fingerprint(before) == _store_fingerprint(after)
        other = {**after, SLOT_KEY: "t:READ"}
        assert _store_fingerprint(other) != _store_fingerprint(after)
        # Without a slot the secret still counts: two principals never share.
        assert _store_fingerprint({"aws_access_key_id": "A"}) != _store_fingerprint(
            {"aws_access_key_id": "B"}
        )

    def test_reads_and_writes_work_through_a_slot(self, conn: Any) -> None:
        t = _with_provider(_table(conn), RotatingProvider(lifetime=3600.0))
        t.append(pa.table({"id": [1, 2]}))
        assert t.to_arrow().num_rows == 2

    def test_a_store_refused_by_storage_gets_a_fresh_credential_at_once(self) -> None:
        """Storage refused the credential before its stated expiry (revoked, a
        clock apart): the store asks, and the refresher vends again then,
        not at the hour the credential said it would last."""
        from deltaswamp import _native
        from deltaswamp.credentials.refresh import _REFRESHER, SLOT_KEY, slot_options

        if "credential_retry" not in _native.FEATURES:
            pytest.skip("native build predates the retry")

        class Revoked(RotatingProvider):
            invalidated = 0

            def invalidate(self) -> None:
                self.invalidated += 1

        provider = Revoked(lifetime=3600.0)
        first = provider.credentials()
        slot = slot_options(provider, Operation.READ, first, first.as_storage_options())[SLOT_KEY]
        assert _REFRESHER.entries()[slot].options == {"token": "v1"}
        _native.request_credential_refresh(slot)
        deadline = time.time() + 10
        while _REFRESHER.entries()[slot].options == {"token": "v1"}:
            assert time.time() < deadline, "the refused slot was never re-vended"
            time.sleep(0.05)
        assert provider.invalidated == 1
        assert _REFRESHER.entries()[slot].options == {"token": "v2"}


# ------------------------------------------------------------------ sharing


class TestSharing:
    def _provider(self, **kwargs: Any) -> Any:
        from deltaswamp.credentials.databricks import DatabricksCredentialProvider

        return DatabricksCredentialProvider(
            kwargs.pop("table_id", "tid"), host="https://h.example", token="t", **kwargs
        )

    def test_identity_is_stable_across_copies_and_names_the_principal(self) -> None:
        from deltaswamp.credentials.databricks import shipping

        p = self._provider()
        # A shipped copy carries the token, so it is the same principal...
        shipped = pickle.loads(pickle.dumps(shipping(p)))
        assert shipped.credential_identity() == p.credential_identity()
        # ...and a default copy does not: its worker authenticates on its own,
        # perhaps as someone else, and must not share this one's credential.
        assert pickle.loads(pickle.dumps(p)).credential_identity() != p.credential_identity()
        assert self._provider(table_id="other").credential_identity() != p.credential_identity()
        from deltaswamp.credentials.databricks import DatabricksCredentialProvider

        other_principal = DatabricksCredentialProvider("tid", host="https://h.example", token="u")
        assert other_principal.credential_identity() != p.credential_identity()
        assert "t" not in p.credential_identity().split("-")[1]

    def test_tasks_unpickled_in_one_worker_share_one_provider(self) -> None:
        from deltaswamp.credentials.databricks import _shared_provider

        p = self._provider()
        cls, state, _pid = p.__reduce__()[1]
        # Pickled in another process (the driver): one copy per worker.
        first = _shared_provider(cls, dict(state), -1)
        second = _shared_provider(cls, dict(state), -1)
        assert first is second
        # A copy made in the pickling process stays a copy.
        assert _shared_provider(cls, dict(state), os.getpid()) is not first

    def test_one_workspace_client_per_authentication(self, monkeypatch: Any) -> None:
        from deltaswamp.credentials import databricks

        built: list[Any] = []

        def fake_client(**kwargs: Any) -> Any:
            built.append(kwargs)
            return object()

        monkeypatch.setattr(databricks, "workspace_client", fake_client)
        a = self._provider(table_id="a")
        b = self._provider(table_id="b")
        assert a._workspace() is b._workspace()
        assert len(built) == 1
        c = databricks.DatabricksCredentialProvider("c", host="https://h.example", token="z")
        assert c._workspace() is not a._workspace()

    def test_the_oss_provider_vends_once_for_racing_threads(self, monkeypatch: Any) -> None:
        from deltaswamp.catalog import ossuc
        from deltaswamp.identity import parse_ref

        calls: list[str] = []

        def fake_request(base: str, path: str, token: Any) -> Any:
            calls.append(path)
            time.sleep(0.2)
            return {
                "url": "s3://b/t",
                "expiration_time": int((time.time() + 3600) * 1000),
                "aws_temp_credentials": {"access_key_id": "A", "secret_access_key": "S"},
            }

        monkeypatch.setattr(ossuc, "_request", fake_request)
        provider = ossuc.OSSUnityCredentialProvider(
            "http://uc", parse_ref("main.s.t"), location="s3://b/t"
        )
        results: list[Any] = []

        def vend() -> None:
            try:
                results.append(provider.credentials(Operation.READ))
            except Exception as exc:  # the response shape is the fake's, not the point
                results.append(exc)

        threads = [threading.Thread(target=vend) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(isinstance(r, Credentials) for r in results), results
        assert len(calls) == 1, calls


# ------------------------------------------------------------------ credential source

#: Stand-ins for a broker in an actor: the source pickles a name, not the broker.
_BROKERS: dict[str, Any] = {}
_ASKED: dict[str, list[str]] = {}


class _Source:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, table_id: str, operation: str) -> Any:
        _ASKED.setdefault(self.name, []).append(operation)
        return _BROKERS[self.name].vend(table_id, operation)


class TestCredentialSource:
    def test_workers_refresh_from_the_broker_once_per_process(self, conn: Any) -> None:
        from deltaswamp.credentials import CredentialBroker
        from deltaswamp.distributed import ShippedCredentials

        t = _table(conn)
        provider = RotatingProvider(lifetime=120.0)  # inside the refresh margin
        _with_provider(t, provider)
        broker = CredentialBroker()
        broker.add(provider)
        _BROKERS["test"] = broker
        asked = _ASKED.setdefault("test", [])
        asked.clear()
        plan = t.plan_write(credential_source=_Source("test"))
        workers = [pickle.loads(pickle.dumps(plan)) for _ in range(4)]
        shipped = [w.table.credential_provider for w in workers]
        assert all(isinstance(s, ShippedCredentials) and s.refreshable for s in shipped)
        served = [s.credentials(Operation.READ_WRITE) for s in shipped]
        # Four tasks, one question to the driver.
        assert asked == ["READ_WRITE"]
        assert len({c.secrets["token"] for c in served}) == 1
        fragments = [w.write(pa.table({"id": [i]})) for i, w in enumerate(workers)]
        plan.commit(fragments)
        assert t.to_arrow().num_rows == 4

    def test_a_broker_refuses_to_travel_and_a_table_it_does_not_serve(self) -> None:
        from deltaswamp.credentials import CredentialBroker

        broker = CredentialBroker()
        with pytest.raises(TypeError, match="not picklable"):
            pickle.dumps(broker)
        with pytest.raises(CredentialError, match="serves no table"):
            broker.vend("nope", "READ")

    def test_a_plan_workers_cannot_refresh_warns_when_it_is_short_lived(self, conn: Any) -> None:
        t = _with_provider(_table(conn), RotatingProvider(lifetime=600.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = t.plan_scan()
        assert any(
            issubclass(w.category, CredentialExpiryWarning)
            and "credential_source" in str(w.message)
            for w in caught
        )
        assert plan.credential_expires_at is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t.plan_scan(credential_source=lambda *_: None)
        assert not [w for w in caught if issubclass(w.category, CredentialExpiryWarning)]


# ------------------------------------------------------------------ plan-time refusals


class TestExternalWriteNotAllowed:
    def test_the_vend_refusal_is_named(self) -> None:
        from deltaswamp.credentials.databricks import DatabricksCredentialProvider

        class Refusal(Exception):
            error_code = "EXTERNAL_WRITE_NOT_ALLOWED_FOR_TABLE"

        class Tables:
            def generate_temporary_table_credentials(self, **_: Any) -> Any:
                raise Refusal("Table is not eligible for external engine writes")

        class Workspace:
            temporary_table_credentials = Tables()

        provider = DatabricksCredentialProvider("tid", host="https://h.example", token="t")
        provider._client = Workspace()
        with pytest.raises(ExternalWriteNotAllowedError, match="catalog commits") as info:
            provider.credentials(Operation.READ_WRITE)
        assert "allow_sql_fallback=True" in str(info.value)
        assert "not a distributed" in str(info.value)

    def test_plan_write_refuses_before_any_worker_runs(self, conn: Any) -> None:
        class ReadOnly(RotatingProvider):
            def credentials(self, operation: Operation = Operation.READ) -> Credentials:
                if str(operation).upper() == Operation.READ_WRITE.value:
                    raise ExternalWriteNotAllowedError("EXTERNAL_WRITE_NOT_ALLOWED_FOR_TABLE")
                return super().credentials(operation)

        t = _with_provider(_table(conn), ReadOnly(lifetime=3600.0))
        with pytest.raises(ExternalWriteNotAllowedError):
            t.plan_write()


# ------------------------------------------------------------------ S3 region


class _RegionHandler(http.server.BaseHTTPRequestHandler):
    region = "eu-west-3"
    seen: ClassVar[list[str]] = []

    def do_HEAD(self) -> None:
        type(self).seen.append(self.path)
        self.send_response(301)
        self.send_header("x-amz-bucket-region", self.region)
        self.send_header("Location", "https://elsewhere")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


class TestBucketRegion:
    def test_vended_s3_keys_get_their_buckets_region(self, monkeypatch: Any) -> None:
        from deltaswamp import _storage
        from deltaswamp.credentials.databricks import credentials_from_response

        server = http.server.HTTPServer(("127.0.0.1", 0), _RegionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setenv(_storage.BUCKET_REGION_PROBE_ENV, "1")
            monkeypatch.setattr(
                _storage, "S3_REGION_ENDPOINT", f"http://127.0.0.1:{server.server_port}"
            )
            monkeypatch.setattr(_storage, "_BUCKET_REGIONS", {})
            response = {
                "url": "s3://paris-bucket/t",
                "aws_temp_credentials": {"access_key_id": "A", "secret_access_key": "S"},
            }
            for _ in range(3):
                creds = credentials_from_response(response, aws_region="us-west-2")
                # The bucket's region, not the metastore's.
                assert creds.secrets["aws_region"] == "eu-west-3"
            assert _RegionHandler.seen == ["/paris-bucket"], "asked once per bucket"
        finally:
            server.shutdown()

    def test_the_metastore_region_when_s3_will_not_say(self, monkeypatch: Any) -> None:
        from deltaswamp import _storage
        from deltaswamp.credentials.databricks import credentials_from_response

        monkeypatch.setenv(_storage.BUCKET_REGION_PROBE_ENV, "1")
        monkeypatch.setattr(_storage, "S3_REGION_ENDPOINT", "http://127.0.0.1:9")
        monkeypatch.setattr(_storage, "_BUCKET_REGIONS", {})
        creds = credentials_from_response(
            {
                "url": "s3://b/t",
                "aws_temp_credentials": {"access_key_id": "A", "secret_access_key": "S"},
            },
            aws_region="us-west-2",
        )
        assert creds.secrets["aws_region"] == "us-west-2"

    def test_an_environment_endpoint_does_not_take_vended_keys(self, monkeypatch: Any) -> None:
        from deltaswamp._storage import engine_options

        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
        vended = {"aws_access_key_id": "A", "aws_secret_access_key": "S", "aws_region": "eu-west-1"}
        options = engine_options(None, vended, "s3://b/t")
        assert options["aws_endpoint"] == "https://s3.eu-west-1.amazonaws.com"
        # The caller's own keys still take the environment's endpoint.
        mine = engine_options(
            {"aws_access_key_id": "A", "aws_secret_access_key": "S"}, None, "s3://b/t"
        )
        assert mine["aws_endpoint"] == "http://127.0.0.1:9"
