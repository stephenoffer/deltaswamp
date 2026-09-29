"""The kernel on GCS and Azure object stores, through emulators.

Both are opt-in. ``DELTASWAMP_TEST_GCS_EMULATOR=1`` runs GCS against
`tests.gcs_emulator`, started in-process: an XML-API emulator that honours
the create-only precondition and, like GCS, refuses an expired bearer token
with 401. ``DELTASWAMP_TEST_AZURITE`` names Azurite's blob endpoint (for
example ``http://127.0.0.1:10000``; its well-known ``devstoreaccount1`` key is
assumed) to run Azure; Azurite checks shared-key SAS signatures and their
expiry.

Each store gets the same checks: kernel reads and writes, DML, OPTIMIZE and
checkpoints; a distributed write whose worker is a separate interpreter
holding only the pickled plan; put-if-absent, by two plans racing to commit;
credentials in the shape Databricks vends them (``azure_user_delegation_sas``,
``gcp_oauth_token``) through a credential provider; and a planned read that
outlives its credential, finishing only because the store picked up the
refreshed one mid-read.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import hashlib
import hmac
import os
import pickle
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine
from deltaswamp.catalog.filesystem import FilesystemCatalog
from deltaswamp.credentials import Credentials, Operation
from deltaswamp.credentials.databricks import credentials_from_response

from tests.gcs_emulator import GcsEmulator

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

AZURITE_ACCOUNT = "devstoreaccount1"
AZURITE_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
)
ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ stores


class Store:
    """One object store under test: where tables go and how to reach them."""

    name: str

    def location(self) -> str:
        raise NotImplementedError

    def static_options(self) -> dict[str, str]:
        """A connection's options: a credential that outlives the test."""
        raise NotImplementedError

    def endpoint_options(self) -> dict[str, str]:
        """A connection's options when a provider vends the credential."""
        raise NotImplementedError

    def vend(self, location: str, ttl: float) -> dict[str, Any]:
        """A temporary-credentials response, shaped as Unity Catalog's."""
        raise NotImplementedError

    def short_lived_options(self, ttl: float) -> dict[str, str]:
        """Static options whose credential dies in `ttl` seconds."""
        raise NotImplementedError


class GcsStore(Store):
    name = "gcs"

    def __init__(self, emulator: GcsEmulator) -> None:
        self.emulator = emulator
        self.bucket = f"ds-{uuid.uuid4().hex[:8]}"
        emulator.create_bucket(self.bucket)
        emulator.allow("static-token")

    def location(self) -> str:
        return f"gs://{self.bucket}/t_{uuid.uuid4().hex[:8]}"

    def endpoint_options(self) -> dict[str, str]:
        return {"google_base_url": self.emulator.endpoint, "allow_http": "true"}

    def static_options(self) -> dict[str, str]:
        return {**self.endpoint_options(), "google_bearer_token": "static-token"}

    def _token(self, ttl: float) -> str:
        token = f"ya29.{uuid.uuid4().hex}"
        self.emulator.allow(token, ttl)
        return token

    def vend(self, location: str, ttl: float) -> dict[str, Any]:
        return {
            "url": location,
            "expiration_time": int((time.time() + ttl) * 1000),
            "gcp_oauth_token": {"oauth_token": self._token(ttl)},
        }

    def short_lived_options(self, ttl: float) -> dict[str, str]:
        return {**self.endpoint_options(), "google_bearer_token": self._token(ttl)}


def _account_sas(seconds: float) -> str:
    """An account SAS for Azurite's account, valid for `seconds`."""
    now = dt.datetime.now(dt.UTC)
    fields = {
        "sv": "2019-12-12",
        "ss": "b",
        "srt": "sco",
        "sp": "rwdlacup",
        "st": (now - dt.timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "se": (now + dt.timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spr": "https,http",
    }
    to_sign = "\n".join(
        [
            AZURITE_ACCOUNT,
            fields["sp"],
            fields["ss"],
            fields["srt"],
            fields["st"],
            fields["se"],
            "",
            fields["spr"],
            fields["sv"],
            "",
        ]
    )
    key = base64.b64decode(AZURITE_KEY)
    fields["sig"] = base64.b64encode(
        hmac.new(key, to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    return urllib.parse.urlencode(fields)


class AzureStore(Store):
    name = "azure"

    def __init__(self, blob_endpoint: str) -> None:
        self.endpoint = f"{blob_endpoint.rstrip('/')}/{AZURITE_ACCOUNT}"
        self.container = f"ds{uuid.uuid4().hex[:10]}"
        self._request("PUT", f"{self.endpoint}/{self.container}?restype=container")

    def _request(self, method: str, url: str) -> int:
        sas = _account_sas(600)
        request = urllib.request.Request(
            f"{url}&{sas}",
            method=method,
            data=b"" if method == "PUT" else None,
            headers={"x-ms-version": "2019-12-12"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            return exc.code

    def drop(self) -> None:
        self._request("DELETE", f"{self.endpoint}/{self.container}?restype=container")

    def location(self) -> str:
        # A real Azure URL: the endpoint override in the options routes it.
        return (
            f"abfss://{self.container}@{AZURITE_ACCOUNT}.dfs.core.windows.net/"
            f"t_{uuid.uuid4().hex[:8]}"
        )

    def endpoint_options(self) -> dict[str, str]:
        return {
            "azure_storage_account_name": AZURITE_ACCOUNT,
            "azure_storage_endpoint": self.endpoint,
            "azure_allow_http": "true",
        }

    def static_options(self) -> dict[str, str]:
        return {**self.endpoint_options(), "azure_storage_sas_key": _account_sas(3600)}

    def vend(self, location: str, ttl: float) -> dict[str, Any]:
        return {
            "url": location,
            "expiration_time": int((time.time() + ttl) * 1000),
            "azure_user_delegation_sas": {"sas_token": "?" + _account_sas(ttl)},
        }

    def short_lived_options(self, ttl: float) -> dict[str, str]:
        return {**self.endpoint_options(), "azure_storage_sas_key": _account_sas(ttl)}


@pytest.fixture(scope="module")
def gcs_emulator() -> Iterator[GcsEmulator]:
    emulator = GcsEmulator()
    yield emulator
    emulator.stop()


@pytest.fixture(params=["gcs", "azure"])
def store(request: Any, gcs_emulator: GcsEmulator) -> Iterator[Store]:
    if request.param == "gcs":
        if not os.environ.get("DELTASWAMP_TEST_GCS_EMULATOR"):
            pytest.skip("set DELTASWAMP_TEST_GCS_EMULATOR=1 to run against the GCS emulator")
        yield GcsStore(gcs_emulator)
        return
    endpoint = os.environ.get("DELTASWAMP_TEST_AZURITE")
    if not endpoint:
        pytest.skip("set DELTASWAMP_TEST_AZURITE to Azurite's blob endpoint to run")
    azure = AzureStore(endpoint)
    yield azure
    azure.drop()


def _connect(options: dict[str, str], *, kernel_only: bool = True) -> Any:
    conn = ds.connect(catalog=FilesystemCatalog(), storage_options=options)
    if kernel_only:
        # Every operation on the kernel: delta-rs serves some of them where it
        # can, which on Azure would leave the kernel's store path unchecked.
        conn.router.engines = {Engine.KERNEL: conn.router.engines[Engine.KERNEL]}
    return conn


def _ids(table: Any) -> list[int]:
    return sorted(table.to_arrow(columns=["id"]).column("id").to_pylist())


SCHEMA = pa.schema([("id", pa.int64()), ("s", pa.string())])


def _rows(start: int, n: int) -> Any:
    return pa.table(
        {"id": list(range(start, start + n)), "s": [f"v{i}" for i in range(start, start + n)]},
        schema=SCHEMA,
    )


# ------------------------------------------------------------- the checks


class TestKernelOnTheStore:
    @pytest.mark.parametrize("dv", [False, True], ids=["copy-on-write", "deletion-vectors"])
    def test_writes_dml_optimize_and_checkpoints(self, store: Store, dv: bool) -> None:
        conn = _connect(store.static_options())
        location = store.location()
        properties = {"delta.enableInCommitTimestamps": "true"}
        if dv:
            properties["delta.enableDeletionVectors"] = "true"
        conn.create_table(location, SCHEMA, properties=properties)
        for start in (0, 10, 20):
            conn.open_table(location).append(_rows(start, 10))
        table = conn.open_table(location)
        assert table.count() == 30

        table.delete("id % 10 = 3")
        table = conn.open_table(location)
        table.update({"s": "'changed'"}, predicate="id = 5")
        table = conn.open_table(location)
        assert table.to_arrow(predicate="id = 5").column("s").to_pylist() == ["changed"]
        expected = [i for i in range(30) if i % 10 != 3]
        assert _ids(table) == expected

        table.optimize()
        table = conn.open_table(location)
        assert _ids(table) == expected
        assert len(table.files()) == 1

        table.checkpoint()
        table = conn.open_table(location)
        assert _ids(table) == expected
        # Time travel reads through the checkpoint and the older commits.
        assert len(table.to_arrow(version=1)) == 10

        conn.open_table(location).overwrite(_rows(100, 5))
        assert _ids(conn.open_table(location)) == list(range(100, 105))

    def test_a_create_where_other_files_are_is_refused(self, store: Store) -> None:
        """The check lists the location through the kernel's store. Through
        delta-rs it could not use a GCS bearer token: it spent some ten
        seconds on the GCE metadata server, listed nothing, and let the
        create go ahead where a later VACUUM would delete the other files."""
        conn = _connect(store.static_options())
        location = store.location()
        conn.create_table(location + "/other", SCHEMA)
        with pytest.raises(ds.UnreachableTableError, match="not a Delta table"):
            conn.create_table(location, SCHEMA)

    def test_put_if_absent_is_honoured(self, store: Store) -> None:
        from deltaswamp import _native

        assert _native.probe_put_if_absent(store.location() + "/", store.static_options())

    def test_two_plans_racing_to_commit_both_land(self, store: Store) -> None:
        """The second commit loses version N+1 to the first (put-if-absent),
        rebases and lands at N+2; a store that ignored the condition would
        overwrite the first commit, and its rows would vanish."""
        conn = _connect(store.static_options())
        location = store.location()
        conn.create_table(location, SCHEMA)
        first = conn.open_table(location).plan_write()
        second = conn.open_table(location).plan_write()
        fragments = [first.write(_rows(0, 5)), second.write(_rows(5, 5))]
        v1 = first.commit([fragments[0]])
        v2 = second.commit([fragments[1]])
        assert (v1, v2) == (1, 2)
        assert _ids(conn.open_table(location)) == list(range(10))


_WORKER = """
import pickle, sys
import pyarrow as pa
plan = pickle.load(open(sys.argv[1], "rb"))
data = pa.table({"id": pa.array([1000, 1001], pa.int64()), "s": ["w", "w"]})
open(sys.argv[2], "wb").write(plan.write(data))
"""


def _write_in_a_worker(tmp_path: Path, plan: Any) -> bytes:
    """`plan.write()` in a fresh interpreter holding only the pickled plan."""
    job, out = tmp_path / "plan.pkl", tmp_path / "fragment.bin"
    job.write_bytes(pickle.dumps(plan))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AZURE_", "GOOGLE_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), str(ROOT)])
    subprocess.run(
        [sys.executable, "-c", _WORKER, str(job), str(out)],
        check=True,
        env=env,
        timeout=120,
    )
    return out.read_bytes()


class TestDistributedWrite:
    def test_a_worker_process_writes_what_the_driver_commits(
        self, store: Store, tmp_path: Path
    ) -> None:
        conn = _connect(store.static_options())
        location = store.location()
        conn.create_table(location, SCHEMA)
        conn.open_table(location).append(_rows(0, 3))
        plan = conn.open_table(location).plan_write()
        fragment = _write_in_a_worker(tmp_path, plan)
        assert plan.commit([fragment]) == 2
        assert _ids(conn.open_table(location)) == [0, 1, 2, 1000, 1001]


# ---------------------------------------------------- vended credentials


class VendingProvider:
    """Vends each credential as Unity Catalog does, through the same mapping."""

    def __init__(self, store: Store, location: str, ttl: float) -> None:
        self.store = store
        self.location = location
        self.ttl = ttl
        self.vends = 0
        self.lock = threading.Lock()

    @property
    def table_id(self) -> str:
        return f"tid-{id(self):x}"

    def credential_identity(self) -> str:
        return self.table_id

    def credentials(self, operation: Operation = Operation.READ) -> Credentials:
        with self.lock:
            self.vends += 1
        return credentials_from_response(
            self.store.vend(self.location, self.ttl),
            url=self.location,
            table_id=self.table_id,
            operation=operation,
        )

    def invalidate(self) -> None:
        pass

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("a test provider does not travel; plans ship its credential")


def _with_provider(table: Any, provider: Any) -> Any:
    table._resolved = dataclasses.replace(table._resolved, credential_provider=provider)
    table._enriched = False
    return table


class TestVendedCredentials:
    def test_the_databricks_shape_reads_writes_and_ships_to_a_worker(
        self, store: Store, tmp_path: Path
    ) -> None:
        location = store.location()
        _connect(store.static_options()).create_table(location, SCHEMA)
        # The connection names only the emulator's endpoint; the credential
        # comes from the provider, with the endpoint Databricks would derive
        # from the URL, which the connection's own must override.
        conn = _connect(store.endpoint_options())
        provider = VendingProvider(store, location, ttl=3600)
        table = _with_provider(conn.open_table(location), provider)
        table.append(_rows(0, 4))
        assert _ids(_with_provider(conn.open_table(location), provider)) == [0, 1, 2, 3]

        plan = _with_provider(conn.open_table(location), provider).plan_write()
        fragment = _write_in_a_worker(tmp_path, plan)
        plan.commit([fragment])
        assert _ids(_with_provider(conn.open_table(location), provider)) == [
            0,
            1,
            2,
            3,
            1000,
            1001,
        ]
        assert provider.vends >= 1

    # The plan's own workers could not refresh (no credential_source); this
    # test reads on the driver, where the provider does.
    @pytest.mark.filterwarnings("ignore::deltaswamp.errors.CredentialExpiryWarning")
    def test_a_read_outlives_its_credential(self, store: Store) -> None:
        """Each split is read after the credential the store was built with
        has expired; the slot hands the store the refreshed one."""
        ttl = 14.0
        location = store.location()
        writer = _connect(store.static_options())
        writer.create_table(location, SCHEMA)
        for start in range(0, 60, 10):
            writer.open_table(location).append(_rows(start, 10))

        conn = _connect(store.endpoint_options())
        provider = VendingProvider(store, location, ttl=ttl)
        table = _with_provider(conn.open_table(location), provider)
        plan = table.plan_scan()
        started = time.time()
        read: list[int] = []
        for split in plan.splits:
            read += plan.read([split]).column("id").to_pylist()
            time.sleep((ttl + 4) / len(plan.splits))
        assert time.time() - started > ttl
        assert sorted(read) == list(range(60))
        assert provider.vends >= 2
        if isinstance(store, GcsStore):
            # The requests themselves carried more than one token.
            assert len({t for t in store.emulator.tokens_seen if t.startswith("ya29.")}) >= 2

    def test_a_dead_credential_is_refused_by_the_store(self, store: Store) -> None:
        """The control for the test above: without a refresh, the store's
        credential expiry is enforced, so that test proves the refresh."""
        location = store.location()
        _connect(store.static_options()).create_table(location, SCHEMA)
        _connect(store.static_options()).open_table(location).append(_rows(0, 3))
        conn = _connect(store.short_lived_options(6.0))
        assert _ids(conn.open_table(location)) == [0, 1, 2]
        time.sleep(7.0)
        with pytest.raises(ds.DeltaSwampError, match=r"(?i)401|403|auth|expired|signature"):
            conn.open_table(location).to_arrow()


def test_a_gcs_create_lists_the_location_with_the_vended_token(gcs_emulator: GcsEmulator) -> None:
    """Always on (a second or two): the non-empty-location check on GCS.

    Through delta-rs the listing could not use a bearer token; off GCP it spent
    some ten seconds timing out on the metadata server, then listed nothing and
    let the create go ahead beside another table's files.
    """
    store = GcsStore(gcs_emulator)
    conn = _connect(store.static_options())
    location = store.location()
    conn.create_table(location + "/other", SCHEMA)
    started = time.time()
    with pytest.raises(ds.UnreachableTableError, match="not a Delta table"):
        conn.create_table(location, SCHEMA)
    assert time.time() - started < 8


def test_a_dead_gcs_token_is_reported_as_the_storage_refusal(gcs_emulator: GcsEmulator) -> None:
    """Always on: the kernel's 401 is the error, not delta-rs's refusal.

    Only the last engine's error was kept, and on GCS that is delta-rs saying
    it cannot use a bearer token, with the remedy to supply a service-account
    key: a dead vended token read as a configuration problem.
    """
    store = GcsStore(gcs_emulator)
    location = store.location()
    _connect(store.static_options()).create_table(location, SCHEMA)
    options = store.short_lived_options(60)
    conn = _connect(options, kernel_only=False)
    table = conn.open_table(location)
    gcs_emulator.allow(options["google_bearer_token"], -1)
    with pytest.raises(ds.DeltaSwampError) as caught:
        table.to_arrow()
    message = str(caught.value)
    assert "kernel:" in message and "401" in message, message
