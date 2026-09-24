"""The headline capability, proven end to end without Databricks.

A catalog-managed table keeps its newest commits as *staged* files that the
catalog has ratified but nobody has published. Listing `_delta_log/` finds only
the older versions, so a reader that does not consult the catalog silently sees
stale data. That is why delta-rs cannot open these tables at all.

These tests build exactly that situation -- a real commit moved into
`_delta_log/_staged_commits/`, served by a Unity Catalog server speaking the
real `/delta/v1` protocol -- and check that the kernel path reads through it
while delta-rs does not.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import uuid
from pathlib import Path
from typing import Any

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests.fake_uc import FakeTable, FakeUnityCatalog  # noqa: E402


def _make_catalog_managed(table_path: str) -> None:
    """Declare `catalogManaged` in the table's own protocol.

    The catalog saying a table is catalog-managed is not enough: the kernel
    checks the protocol, and rightly refuses a commit tail for a table that does
    not claim the feature. Real tables get this at creation through the UC
    staging flow, which needs a live catalog, so the fixture writes it directly.
    """
    log = pathlib.Path(table_path) / "_delta_log"
    first = log / f"{0:020d}.json"
    actions = [json.loads(line) for line in first.read_text().splitlines() if line.strip()]

    for action in actions:
        if "protocol" in action:
            action["protocol"] = {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                # catalogManaged is a ReaderWriter feature, so it must appear in
                # both lists; inCommitTimestamp is required alongside it.
                "readerFeatures": ["catalogManaged"],
                "writerFeatures": ["catalogManaged", "inCommitTimestamp"],
            }
        if "metaData" in action:
            configuration = action["metaData"].setdefault("configuration", {})
            # Only the enablement property belongs here. A `delta.feature.*`
            # signal is consumed at create time and must not persist in metadata.
            configuration["delta.enableInCommitTimestamps"] = "true"

    first.write_text("\n".join(json.dumps(a) for a in actions) + "\n")


def _metadata_id(table_path: str) -> str:
    """The table id from the log, which the catalog has to agree with."""
    first = pathlib.Path(table_path) / "_delta_log" / f"{0:020d}.json"
    for line in first.read_text().splitlines():
        action = json.loads(line) if line.strip() else {}
        if "metaData" in action:
            return str(action["metaData"]["id"])
    raise AssertionError("no metaData action in the first commit")


def _stage_latest_commit(table_path: str) -> dict[str, Any]:
    """Move the newest commit into `_staged_commits/`, as a catalog would.

    Returns the commit descriptor the catalog reports in its tail.
    """
    log = Path(table_path) / "_delta_log"
    commits = sorted(log.glob("[0-9]*.json"))
    newest = commits[-1]
    version = int(newest.stem)

    staged_dir = log / "_staged_commits"
    staged_dir.mkdir(exist_ok=True)
    staged_name = f"{version:020d}.{uuid.uuid4()}.json"
    shutil.move(str(newest), str(staged_dir / staged_name))

    return {
        "version": version,
        "file_name": staged_name,
        "file_size": (staged_dir / staged_name).stat().st_size,
        "timestamp": 1700000000000,
    }


@pytest.fixture
def catalog_managed(tmp_path: Any) -> Any:
    """A table whose version 1 exists only as a staged commit."""
    from deltalake import write_deltalake

    path = str(tmp_path / "cm")
    write_deltalake(path, pa.table({"id": [1, 2], "city": ["oslo", "lima"]}))
    write_deltalake(path, pa.table({"id": [3], "city": ["cairo"]}), mode="append")
    _make_catalog_managed(path)
    table_id = _metadata_id(path)
    commit = _stage_latest_commit(path)

    with FakeUnityCatalog() as uc:
        uc.add_table(
            "main.sales.cm",
            FakeTable(
                name="cm",
                location=path,
                table_id=table_id,
                properties={"delta.feature.catalogManaged": "supported"},
                commits=[commit],
                latest_version=commit["version"],
            ),
        )
        yield uc, path, commit, table_id


class TestResolution:
    def test_catalog_supplies_the_commit_tail(self, catalog_managed: Any) -> None:
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.identity import parse_ref

        uc, _path, commit, _id = catalog_managed
        resolved = OSSUnityCatalog(uc.url).resolve(parse_ref("main.sales.cm"))

        assert resolved.is_catalog_managed
        assert resolved.max_catalog_version == commit["version"]
        assert [entry.version for entry in resolved.log_tail] == [commit["version"]]
        assert resolved.log_tail[0].path == commit["file_name"]

    def test_table_uuid_is_carried_through(self, catalog_managed: Any) -> None:
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.identity import parse_ref

        uc, _path, _commit, table_id = catalog_managed
        assert OSSUnityCatalog(uc.url).resolve(parse_ref("main.sales.cm")).table_uuid == table_id


class TestTheGapThisCloses:
    def test_delta_rs_cannot_open_it_at_all(self, catalog_managed: Any) -> None:
        """delta-rs refuses a catalog-managed table outright.

        It has no way to obtain the ratified tail, so rather than reading a
        stale version it declines -- which is precisely the gap this project
        exists to close.
        """
        from deltalake import DeltaTable

        _uc, path, _commit, _id = catalog_managed
        with pytest.raises(Exception, match=r"[Cc]atalog"):
            DeltaTable(path).version()

    def test_kernel_reads_through_the_staged_commit(self, catalog_managed: Any) -> None:
        """With the tail, the same table reads at version 1."""
        from deltaswamp._native import Snapshot

        _uc, path, commit, _id = catalog_managed
        snapshot = Snapshot.resolve(
            path,
            log_tail=[
                (commit["version"], commit["file_name"], commit["timestamp"], commit["file_size"])
            ],
            max_catalog_version=commit["version"],
        )
        assert snapshot.version == commit["version"]
        assert pa.table(snapshot.scan()).num_rows == 3


class TestThroughThePublicAPI:
    @pytest.fixture
    def conn(self, catalog_managed: Any) -> Any:
        from deltaswamp.capability import Engine
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.engine.deltars import DeltaRsEngine
        from deltaswamp.engine.kernel import KernelEngine
        from deltaswamp.router import Router
        from deltaswamp.table import Connection

        uc, _path, _commit, _table_id = catalog_managed
        return Connection(
            catalog=OSSUnityCatalog(uc.url),
            router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
        )

    def test_read_routes_to_kernel_and_sees_the_staged_rows(self, conn: Any) -> None:
        from deltaswamp.capability import Engine, Operation

        table = conn.table("main.sales.cm")
        assert table.is_catalog_managed
        assert table.can(Operation.SCAN).engine is Engine.KERNEL
        assert table.to_arrow().num_rows == 3

    def test_delta_rs_is_refused_for_this_table(self, conn: Any) -> None:
        from deltaswamp.capability import Operation
        from deltaswamp.engine.deltars import DeltaRsEngine

        table = conn.table("main.sales.cm")
        verdict = DeltaRsEngine().supports(Operation.SCAN, table.resolved)
        assert not verdict.ok
        assert "catalogManaged" in verdict.reason

    def test_alter_is_refused_with_the_committer_reason(self, conn: Any) -> None:
        from deltaswamp.capability import Operation

        verdict = conn.table("main.sales.cm").can(Operation.SET_PROPERTIES)
        assert not verdict.ok
        assert "catalog-managed" in verdict.reason


class TestCommitErrorsAreDistinguished:
    """409 and 429 mean different things: restage versus publish."""

    def test_fake_catalog_can_produce_both(self, catalog_managed: Any) -> None:
        import urllib.error
        import urllib.request

        uc, _path, _commit, _table_id = catalog_managed
        for status in (409, 429):
            uc.next_commit_status = status
            request = urllib.request.Request(
                f"{uc.url}/api/2.1/unity-catalog/delta/v1/catalogs/main/schemas/sales/tables/cm",
                data=json.dumps({"requirements": [], "updates": []}).encode(),
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(request)
            assert excinfo.value.code == status


class TestLifecycle:
    """Discovery and drop, against the protocol rather than a mock."""

    @pytest.fixture
    def conn(self, catalog_managed: Any) -> Any:
        from deltaswamp.capability import Engine
        from deltaswamp.catalog.ossuc import OSSUnityCatalog
        from deltaswamp.engine.deltars import DeltaRsEngine
        from deltaswamp.engine.kernel import KernelEngine
        from deltaswamp.router import Router
        from deltaswamp.table import Connection

        uc, _path, _commit, _id = catalog_managed
        return Connection(
            catalog=OSSUnityCatalog(uc.url),
            router=Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()}),
            default_catalog="main",
            default_schema="sales",
        )

    def test_list_catalogs(self, conn: Any) -> None:
        assert conn.list_catalogs() == ["main"]

    def test_list_schemas(self, conn: Any) -> None:
        assert conn.list_schemas() == ["sales"]

    def test_list_schemas_for_an_explicit_catalog(self, conn: Any) -> None:
        assert conn.list_schemas("main") == ["sales"]

    def test_list_tables(self, conn: Any) -> None:
        assert [t.ref.table for t in conn.list_tables("main", "sales")] == ["cm"]

    def test_drop_table_removes_the_registration(self, conn: Any) -> None:
        conn.drop_table("main.sales.cm")
        assert conn.list_tables("main", "sales") == []

    def test_resolving_a_dropped_table_says_it_does_not_exist(self, conn: Any) -> None:
        from deltaswamp.errors import DeltaSwampError

        conn.drop_table("main.sales.cm")
        with pytest.raises(DeltaSwampError):
            conn.table("main.sales.cm")
