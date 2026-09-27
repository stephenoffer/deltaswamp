"""Provider-matrix audit regressions (PV-*) against local tables and, where
available, a real OSS Unity Catalog 0.6 server (skipped without java or the jar)."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import deltaswamp as ds
import pyarrow as pa
import pytest

SCHEMA = pa.schema([("id", pa.int64()), ("n32", pa.int32())])


def _rows() -> Any:
    return pa.table({"id": [1], "n32": pa.array([1], pa.int32())})


class TestCanFollowsTheCallShape:
    """PV-11."""

    def test_alter_column_type_without_type_widening(self, tmp_path: Path) -> None:
        t = ds.connect("file://").create_table(str(tmp_path / "t"), SCHEMA)
        t.append(_rows())
        verdict = t.can("alter_column_type")
        assert not verdict.ok and "delta.enableTypeWidening" in verdict.reason
        with pytest.raises(ds.UnreachableTableError):
            t.alter_column_type("n32", "bigint")

    def test_alter_column_type_with_type_widening(self, tmp_path: Path) -> None:
        t = ds.connect("file://").create_table(
            str(tmp_path / "t"), SCHEMA, properties={"delta.enableTypeWidening": "true"}
        )
        t.append(_rows())
        assert t.can("alter_column_type").ok
        t.alter_column_type("n32", "bigint")

    def test_vacuum_is_judged_as_the_default_dry_run(self, tmp_path: Path) -> None:
        t = ds.connect("file://").create_table(
            str(tmp_path / "t"), SCHEMA, properties={"delta.enableRowTracking": "true"}
        )
        t.append(_rows())
        assert t.can("vacuum").ok  # refused before: it judged a real VACUUM
        t.vacuum()


# ------------------------------------------------------- OSS UC 0.6 server

_UC_JAR = Path(
    os.environ.get(
        "DELTASWAMP_UC_CLASSPATH_FILE",
        str(Path.home() / ".cache/deltaswamp/uc-0.6.0.cp"),
    )
)
_JAVA = os.environ.get("DELTASWAMP_JAVA", "/opt/homebrew/opt/openjdk@21/bin/java")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def uc_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real OSS Unity Catalog server on local storage.

    Needs a JDK 21 (DELTASWAMP_JAVA) and a file holding the server's
    classpath (DELTASWAMP_UC_CLASSPATH_FILE, e.g. from ``cs fetch -p
    io.unitycatalog:unitycatalog-server:0.6.0``); skipped otherwise.
    """
    if not (os.access(_JAVA, os.X_OK) and _UC_JAR.is_file()):
        pytest.skip("no JDK 21 or OSS Unity Catalog classpath for the server test")
    root = tmp_path_factory.mktemp("uc")
    conf = root / "etc" / "conf"
    conf.mkdir(parents=True)
    (root / "storage").mkdir()
    (conf / "server.properties").write_text(
        "server.env=dev\nserver.authorization=disable\n"
        f"storage-root.tables=file://{root / 'storage'}\n"
        "server.managed-table.enabled=true\n"
    )
    (conf / "hibernate.properties").write_text(
        "hibernate.connection.driver_class=org.h2.Driver\n"
        "hibernate.connection.url=jdbc:h2:file:./etc/db/h2db;DB_CLOSE_DELAY=-1\n"
        "hibernate.hbm2ddl.auto=update\n"
    )
    port = _free_port()
    process = subprocess.Popen(
        [
            _JAVA,
            "-cp",
            _UC_JAR.read_text().strip(),
            "io.unitycatalog.server.UnityCatalogServer",
            "-p",
            str(port),
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/api/2.1/unity-catalog/catalogs", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.skip("the OSS Unity Catalog server did not start")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=30)


class TestOssUnityCatalogServer:
    """PV-1, PV-6, PV-7, PV-8, PV-10, PV-14 against the real server."""

    def test_local_storage_lifecycle(self, uc_server: str, tmp_path: Path) -> None:
        conn = ds.connect("uc://" + uc_server, default_catalog="c", default_schema="s")
        assert conn.preflight() == []
        conn.create_catalog("c")
        conn.create_schema("c.s")
        external = conn.create_table("c.s.e", SCHEMA, location=(tmp_path / "e").as_uri())
        external.append(_rows())
        path = tmp_path / "r"
        ds.connect("file://").create_table(str(path), SCHEMA)
        conn.register_table("c.s.r", path.as_uri())
        (tmp_path / "nolog").mkdir()
        with pytest.raises(ds.UnreachableTableError, match="no Delta log"):
            conn.register_table("c.s.nolog", (tmp_path / "nolog").as_uri())

        external.add_column([pa.field("extra", pa.string())])
        external.set_comment("hello")
        info = conn.table("c.s.e").info()
        assert [c.name for c in info.columns] == ["id", "n32", "extra"]
        assert info.comment == "hello"

        managed = conn.create_table("c.s.m", SCHEMA)
        verdict = managed.can("history")
        assert "allow_sql_fallback" not in verdict.remedy
        with pytest.raises(ds.InvalidReferenceError, match="knows no principal"):
            managed.grant("nobody@example.com", ["SELECT"])
        with pytest.raises(ds.InvalidArgumentError, match="SQL warehouse"):
            ds.connect("uc://" + uc_server, allow_sql_fallback=True)

        for name in ("c.s.e", "c.s.r", "c.s.m"):
            conn.drop_table(name)
        conn.drop_schema("c.s", force=True)
        conn.drop_catalog("c", force=True)
        assert "c" not in conn.list_catalogs()
