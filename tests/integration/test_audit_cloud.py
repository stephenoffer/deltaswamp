"""Cloud/object-store audit fixes, end to end.

The concurrent-create test runs on the local filesystem. The Azurite and moto
tests run only when those emulators are listening (Azurite on 127.0.0.1:10000,
moto_server on 127.0.0.1:5055); otherwise they skip.
"""

from __future__ import annotations

import multiprocessing as mp
import socket
import uuid
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

import deltaswamp as ds  # noqa: E402
from deltaswamp.catalog.filesystem import FilesystemCatalog  # noqa: E402

AZURITE_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
)


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def _create(args: tuple[str, int, dict[str, str]]) -> str:
    location, i, options = args
    conn = ds.connect(catalog=FilesystemCatalog(), storage_options=options or None)
    try:
        conn.create_table(location, pa.schema([(f"c{i}", pa.int64())]))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "ok"


def _race(location: str, options: dict[str, str], writers: int = 6) -> list[str]:
    with mp.get_context("spawn").Pool(writers) as pool:
        return pool.map(_create, [(location, i, options) for i in range(writers)])


def test_concurrent_creates_have_exactly_one_winner(tmp_path: Path) -> None:
    """CL-01: delta-rs retried a lost version-0 CREATE at version 1, so two
    creates "succeeded" and the second replaced the first one's schema."""
    for trial in range(4):
        location = str(tmp_path / f"t{trial}")
        results = _race(location, {})
        assert results.count("ok") == 1, results
        losers = [r for r in results if r != "ok"]
        assert all("already exists" in r for r in losers), losers
        table = ds.connect(catalog=FilesystemCatalog()).table(location)
        assert table.version == 0


@pytest.mark.skipif(not _listening(10000), reason="Azurite is not running on :10000")
@pytest.mark.parametrize("suffix", ["core.chinacloudapi.cn", "core.usgovcloudapi.net"])
@pytest.mark.parametrize("properties", [{}, {"delta.enableInCommitTimestamps": "true"}])
def test_sovereign_cloud_tables_work_on_both_engines(
    suffix: str, properties: dict[str, str]
) -> None:
    """CL-02: a sovereign-cloud host with an explicit (Azurite) endpoint."""
    options = {
        "azure_storage_account_name": "devstoreaccount1",
        "azure_storage_account_key": AZURITE_KEY,
        "azure_storage_endpoint": "http://127.0.0.1:10000/devstoreaccount1",
        "azure_allow_http": "true",
    }
    conn = ds.connect(catalog=FilesystemCatalog(), storage_options=options)
    location = f"abfss://cont@devstoreaccount1.dfs.{suffix}/t_{uuid.uuid4().hex[:6]}"
    try:
        table = conn.create_table(location, pa.schema([("a", pa.int64())]), properties=properties)
    except Exception as exc:
        if "ContainerNotFound" in str(exc):
            pytest.skip("Azurite has no container named 'cont'")
        raise
    table.append(pa.table({"a": [1, 2]}))
    assert table.count() == 2


@pytest.mark.skipif(not _listening(5055), reason="moto_server is not running on :5055")
def test_a_store_that_honours_if_none_match_passes_the_probe() -> None:
    """CL-07: moto honours If-None-Match, so the probe lets writes through."""
    import contextlib
    import urllib.request

    # moto creates a bucket on an unsigned PUT; an existing one is fine.
    with contextlib.suppress(Exception):
        urllib.request.urlopen(
            urllib.request.Request("http://127.0.0.1:5055/dsauditcl", method="PUT"), timeout=5
        )
    options = {
        "aws_endpoint": "http://127.0.0.1:5055",
        "aws_access_key_id": "k",
        "aws_secret_access_key": "s",
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
    }
    conn = ds.connect(catalog=FilesystemCatalog(), storage_options=options)
    location = f"s3://dsauditcl/t_{uuid.uuid4().hex[:6]}"
    table = conn.create_table(location, pa.schema([("a", pa.int64())]))
    table.append(pa.table({"a": [1]}))
    assert table.can("append").ok
