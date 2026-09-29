"""DML and maintenance on a catalog-managed table, racing other writers.

Against `tests/fake_uc_strict.py`, which refuses a commit whose version is
not the catalog's next (409), as the real server does. A DELETE, UPDATE,
MERGE or OPTIMIZE that read the table before a concurrent blind append was
ratified loses the race; the engine re-reads the catalog's commit tail and
rebases over the append where Delta's conflict rules allow, as it does on a
path table. The append's rows must survive, and the DML's effect with them.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import deltaswamp as ds
import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

from tests.fake_uc_strict import StrictUnityCatalog  # noqa: E402

SCHEMA = pa.schema([("id", pa.int64()), ("c", pa.string())])
NAME = "main.sales.cm"
_NO_DV = {"delta.enableDeletionVectors": "false"}
_RT = {"delta.enableRowTracking": "true"}
_RT_NO_DV = {**_RT, **_NO_DV}


def _router() -> Any:
    from deltaswamp.capability import Engine
    from deltaswamp.engine.deltars import DeltaRsEngine
    from deltaswamp.engine.kernel import KernelEngine
    from deltaswamp.router import Router

    return Router(engines={Engine.KERNEL: KernelEngine(), Engine.DELTARS: DeltaRsEngine()})


@pytest.fixture
def uc(tmp_path: pathlib.Path) -> Iterator[StrictUnityCatalog]:
    with StrictUnityCatalog(staging_root=tmp_path / "managed") as server:
        yield server


def _oss(uc: StrictUnityCatalog) -> Any:
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
    setup = _oss(uc)
    setup.create_catalog("main")
    setup.create_schema("main.sales")
    return setup if request.param == "oss" else _dbx(uc, monkeypatch)


@pytest.fixture
def oss(uc: StrictUnityCatalog) -> Any:
    c = _oss(uc)
    c.create_catalog("main")
    c.create_schema("main.sales")
    return c


def _table(conn: Any, properties: dict[str, str] | None = None, batches: int = 3) -> Any:
    conn.create_table(NAME, SCHEMA, properties=properties)
    for i in range(batches):
        conn.table(NAME).append(pa.table({"id": [i * 10 + 1, i * 10 + 2], "c": ["a", "b"]}))
    return conn.table(NAME)


def _rows(conn: Any) -> list[tuple[Any, Any]]:
    data = conn.table(NAME).to_arrow()
    return sorted(
        zip(data.column("id").to_pylist(), data.column("c").to_pylist(), strict=True), key=str
    )


@pytest.fixture
def race(monkeypatch: pytest.MonkeyPatch, uc: StrictUnityCatalog) -> Any:
    """Arm one concurrent commit, landed just before the next catalog commit.

    The engine builds the committer config as it commits, after it has read
    the table, so a commit made there is one the operation never saw: its
    staged version is then taken, and the catalog answers 409.
    """
    from deltaswamp.engine.kernel import KernelEngine

    armed: list[Any] = []
    original = KernelEngine._uc_commit_config

    def racing(self: Any, table: Any, *, staging: bool = False) -> Any:
        if armed and not staging:
            other = armed.pop()
            other()
        return original(self, table, staging=staging)

    monkeypatch.setattr(KernelEngine, "_uc_commit_config", racing)

    def arm(commit: Any) -> None:
        armed.append(commit)

    return arm


def _append_from_another_writer(uc: StrictUnityCatalog, *ids: int) -> Any:
    def commit() -> None:
        _oss(uc).table(NAME).append(pa.table({"id": list(ids), "c": ["x"] * len(ids)}))

    return commit


def _staged_text(location: str) -> str:
    root = pathlib.Path(urlparse(location).path) / "_delta_log"
    return "".join(p.read_text() for p in sorted(root.rglob("*.json")))


# ----------------------------------------------------------- losing a race


@pytest.mark.parametrize("properties", [None, _NO_DV], ids=["dv", "rewrite"])
class TestDmlRebasesOverABlindAppend:
    def test_delete(self, conn: Any, uc: Any, race: Any, properties: Any) -> None:
        table = _table(conn, properties)
        race(_append_from_another_writer(uc, 100))
        table.delete("id = 11")
        assert "version" in uc.refusals, "the delete lost the race"
        assert (100, "x") in _rows(conn), "the concurrent append survived"
        assert all(i != 11 for i, _ in _rows(conn))
        assert len(_rows(conn)) == 6

    def test_update(self, conn: Any, uc: Any, race: Any, properties: Any) -> None:
        table = _table(conn, properties)
        race(_append_from_another_writer(uc, 100))
        table.update(new_values={"c": "z"}, predicate="id = 21")
        assert "version" in uc.refusals
        rows = _rows(conn)
        assert (100, "x") in rows and (21, "z") in rows and len(rows) == 7

    def test_merge(self, conn: Any, uc: Any, race: Any, properties: Any) -> None:
        pytest.importorskip("duckdb")
        table = _table(conn, properties)
        race(_append_from_another_writer(uc, 100))
        (
            table.merge(
                pa.table({"id": [1, 7], "c": ["m", "n"]}),
                "t.id = s.id",
                source_alias="s",
                target_alias="t",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
        assert "version" in uc.refusals
        rows = _rows(conn)
        assert (100, "x") in rows and (1, "m") in rows and (7, "n") in rows
        assert len(rows) == 8


def test_a_concurrent_delete_of_the_same_rows_still_conflicts(
    conn: Any, uc: Any, race: Any
) -> None:
    """Not a blind append: it re-vectored the very file this DELETE touched."""
    table = _table(conn)

    def other() -> None:
        _oss(uc).table(NAME).delete("id = 12")

    race(other)
    with pytest.raises(ds.CommitConflictError):
        table.delete("id = 11")
    rows = _rows(conn)
    assert (11, "a") in rows and all(i != 12 for i, _ in rows)


def test_a_pinned_handle_deletes_through_the_catalog(conn: Any, uc: Any) -> None:
    """Read at a past version and rebased over the appends since, as on a path table."""
    _table(conn)
    pinned = conn.table(NAME, version=2)
    conn.table(NAME).append(pa.table({"id": [100], "c": ["x"]}))
    assert pinned.can("delete").ok, pinned.can("delete").reason
    pinned.delete("id = 1")
    rows = _rows(conn)
    assert (100, "x") in rows and all(i != 1 for i, _ in rows)


# ---------------------------------------------------------------- OPTIMIZE


def test_optimize_commits_through_the_catalog(conn: Any, uc: Any) -> None:
    table = _table(conn, batches=5)
    before = _rows(conn)
    assert table.can("optimize").ok, table.can("optimize").reason
    table.optimize()
    assert _rows(conn) == before
    assert conn.table(NAME).files().num_rows == 1
    assert uc.refusals == []
    assert '"OPTIMIZE"' in _staged_text(table.location)


def test_optimize_rebases_over_a_blind_append(conn: Any, uc: Any, race: Any) -> None:
    table = _table(conn, batches=4)
    race(_append_from_another_writer(uc, 100))
    table.optimize()
    assert "version" in uc.refusals
    rows = _rows(conn)
    assert (100, "x") in rows and len(rows) == 9


def test_zorder_through_the_catalog(conn: Any, uc: Any) -> None:
    table = _table(conn, batches=4)
    before = _rows(conn)
    table.optimize(zorder_by=["id"])
    assert _rows(conn) == before
    assert uc.refusals == []


# ------------------------------------------------------------ row tracking


def _row_ids(conn: Any) -> dict[int, int]:
    engine = conn.router.engines[ds.Engine.KERNEL]
    snapshot = engine.snapshot(conn.table(NAME).resolved)
    data = pa.table(snapshot.scan(row_ids=True, row_positions=True))
    return dict(
        zip(
            data.column("id").to_pylist(),
            data.column("__deltaswamp_row_id").to_pylist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("properties", [_RT, _RT_NO_DV], ids=["dv", "copy-on-write"])
def test_row_ids_survive_dml_on_a_catalog_managed_table(oss: Any, properties: Any) -> None:
    pytest.importorskip("duckdb")
    _table(oss, properties)
    before = _row_ids(oss)
    assert len(set(before.values())) == len(before)
    oss.table(NAME).update(new_values={"c": "z"}, predicate="id = 11")
    oss.table(NAME).delete("id = 21")
    (
        oss.table(NAME)
        .merge(
            pa.table({"id": [12, 99], "c": ["m", "n"]}),
            "t.id = s.id",
            source_alias="s",
            target_alias="t",
        )
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    after = _row_ids(oss)
    for key, rid in before.items():
        if key != 21:
            assert after[key] == rid, key
    assert 21 not in after
    assert after[99] not in before.values()
    assert len(set(after.values())) == len(after)


# ------------------------------------------------ copy-on-write and rewrites


def test_merge_without_deletion_vectors_rewrites_only_touched_files(oss: Any) -> None:
    pytest.importorskip("duckdb")
    table = _table(oss, _NO_DV)
    files_before = set(table.files().column("path").to_pylist())
    (
        oss.table(NAME)
        .merge(pa.table({"id": [1], "c": ["m"]}), "t.id = s.id", source_alias="s", target_alias="t")
        .when_matched_update_all()
        .execute()
    )
    files_after = set(oss.table(NAME).files().column("path").to_pylist())
    assert len(files_before - files_after) == 1, "only the file holding id 1 was rewritten"
    assert (1, "m") in _rows(oss) and len(_rows(oss)) == 6


def test_delete_without_deletion_vectors_rewrites_only_touched_files(oss: Any) -> None:
    table = _table(oss, _NO_DV)
    files_before = set(table.files().column("path").to_pylist())
    oss.table(NAME).delete("id = 11")
    files_after = set(oss.table(NAME).files().column("path").to_pylist())
    assert len(files_before - files_after) == 1
    assert len(_rows(oss)) == 5


# ------------------------------------------------------------ memory bounds


def _kernel(conn: Any) -> Any:
    return conn.router.engines[ds.Engine.KERNEL]


@pytest.mark.parametrize("properties", [None, _NO_DV], ids=["dv", "copy-on-write"])
def test_a_merge_past_the_read_bound_is_refused_before_writing(
    oss: Any, monkeypatch: pytest.MonkeyPatch, properties: Any
) -> None:
    pytest.importorskip("duckdb")
    _table(oss, properties)
    version = oss.table(NAME).version
    monkeypatch.setattr(_kernel(oss), "dml_max_bytes", 1)
    with pytest.raises(ds.DeltaSwampError, match="dml_max_bytes"):
        (
            oss.table(NAME)
            .merge(
                pa.table({"id": [1], "c": ["m"]}), "t.id = s.id", source_alias="s", target_alias="t"
            )
            .when_matched_update_all()
            .execute()
        )
    assert oss.table(NAME).version == version


def test_a_copy_on_write_delete_past_the_read_bound_is_refused(
    oss: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _table(oss, _NO_DV)
    version = oss.table(NAME).version
    monkeypatch.setattr(_kernel(oss), "dml_max_bytes", 1)
    with pytest.raises(ds.DeltaSwampError, match="dml_max_bytes"):
        oss.table(NAME).delete("id = 11")
    assert oss.table(NAME).version == version


def test_the_whole_table_bound_no_longer_limits_a_file_rewrite(
    oss: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DELETE reads only the files holding matching rows, so the table's size is no bound."""
    _table(oss, _NO_DV)
    monkeypatch.setattr(_kernel(oss), "rewrite_max_bytes", 1)
    assert oss.table(NAME).can("delete").ok
    oss.table(NAME).delete("id = 11")
    assert len(_rows(oss)) == 5


# --------------------------------------------------------- RESTORE, VACUUM


@pytest.mark.parametrize("properties", [None, _NO_DV], ids=["dv", "copy-on-write"])
def test_restore_commits_through_the_catalog(conn: Any, uc: Any, properties: Any) -> None:
    table = _table(conn, properties)
    at = table.version
    before = _rows(conn)
    conn.table(NAME).delete("id = 11")
    conn.table(NAME).update(new_values={"c": "z"}, predicate="id = 21")
    assert conn.table(NAME).can("restore").ok, conn.table(NAME).can("restore").reason
    result = conn.table(NAME).restore(at)
    assert _rows(conn) == before
    assert result["version"] == at + 3
    assert uc.refusals == []
    assert '"RESTORE"' in _staged_text(table.location)


def test_restore_rebases_over_a_concurrent_append(conn: Any, uc: Any, race: Any) -> None:
    table = _table(conn)
    at = table.version
    conn.table(NAME).delete("id = 11")
    race(_append_from_another_writer(uc, 100))
    conn.table(NAME).restore(at)
    assert "version" in uc.refusals
    rows = _rows(conn)
    # The restore was recomputed on the table as the append left it: back to
    # the files of version `at`, so the appended file is removed too.
    assert (11, "a") in rows and (100, "x") not in rows and len(rows) == 6


def test_vacuum_dry_run_on_a_catalog_managed_table(conn: Any) -> None:
    _table(conn)
    conn.table(NAME).delete("id = 11")
    listed = conn.table(NAME).vacuum(retention_hours=0, enforce_retention_duration=False)
    assert listed == []  # the deleted row's file is still live, with a vector


def test_vacuum_on_a_catalog_managed_table_needs_the_opt_in(conn: Any, uc: Any) -> None:
    table = _table(conn, _NO_DV)
    conn.table(NAME).delete("id = 11")  # the rewritten file's original is now garbage
    kwargs = {"retention_hours": 0, "enforce_retention_duration": False, "dry_run": False}
    assert not conn.table(NAME).can("vacuum", **kwargs).ok
    with pytest.raises(ds.DeltaSwampError, match="allow_catalog_managed"):
        conn.table(NAME).vacuum(**kwargs)
    listed = conn.table(NAME).vacuum(retention_hours=0, enforce_retention_duration=False)
    assert len(listed) == 1
    deleted = conn.table(NAME).vacuum(**kwargs, allow_catalog_managed=True)
    assert deleted == listed
    assert len(_rows(conn)) == 5
    text = _staged_text(table.location)
    assert '"VACUUM START"' in text and '"VACUUM END"' in text
    assert uc.refusals == []


def test_raw_commits_take_only_file_actions(tmp_path: pathlib.Path) -> None:
    """kernel writes the commitInfo, and the catalog refuses metaData or protocol changes."""
    conn = ds.connect()
    path = str(tmp_path / "t")
    conn.create_table(path, SCHEMA)
    snapshot = _kernel(conn).snapshot(conn.open_table(path).resolved, write=True)
    for action in ('{"metaData":{}}', '{"protocol":{}}', '{"commitInfo":{}}'):
        with pytest.raises(Exception, match="only add, remove, txn and domainMetadata"):
            snapshot.commit_actions([action])
