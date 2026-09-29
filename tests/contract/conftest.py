"""Session-scoped tables, catalog and engine instrumentation for the contract suites.

Templates are built once; a test that writes gets a copy (a directory copy of a
six-row table costs a millisecond). Catalog-managed copies are registered with
one fake Unity Catalog server for the whole session.
"""

from __future__ import annotations

import itertools
import os
import pathlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")
if not ds.has_native():  # pragma: no cover - the whole suite needs the engines
    pytest.skip("native extension not built", allow_module_level=True)

from tests import helpers  # noqa: E402
from tests.contract.tables import FIXTURES, copy_table  # noqa: E402
from tests.fake_uc import FakeTable, FakeUnityCatalog  # noqa: E402

#: Engine methods that are probes, not service: the router and the Table call
#: them to decide or to resolve, whichever engine then serves the operation.
_PROBES = frozenset(
    {
        "available",
        "supports",
        "need_refusal",
        "snapshot",
        "legacy_calendar_files",
        "checkpoint_interval",
        "txn_version",
    }
)


@dataclass
class Served:
    """Which engines did the work of a call: (engine kind, method) in call order."""

    calls: list[tuple[str, str]] = field(default_factory=list)
    active: bool = False

    def kinds_of(self, methods: frozenset[str] | None = None) -> list[str]:
        """The engines that made any of `methods` (every call when None), in order."""
        seen: list[str] = []
        for kind, name in self.calls:
            if (methods is None or name in methods) and kind not in seen:
                seen.append(kind)
        return seen


def instrument(conn: Any) -> Served:
    """Record every serving call on the connection's engines.

    The wrappers live on the engine instances of this one connection, so
    nothing outside the suite sees them, and `isinstance` checks the Table
    makes on the engine still hold.
    """
    served = Served()
    for engine in conn.router.engines.values():
        kind = str(getattr(engine, "kind", type(engine).__name__))
        # engine.__class__, not type(engine): the router hands engines out
        # behind the error boundary, a wrapper that reports the engine's class.
        for name in dir(engine.__class__):
            if name.startswith("_") or name in _PROBES:
                continue
            method = getattr(engine, name, None)
            if not callable(method):
                continue

            def wrapper(
                *args: Any, _m: Any = method, _n: str = name, _k: str = kind, **kwargs: Any
            ) -> Any:
                if served.active:
                    served.calls.append((_k, _n))
                return _m(*args, **kwargs)

            setattr(engine, name, wrapper)
    return served


def _make_catalog_managed(path: str, *, vacuum_protocol_check: bool = True) -> None:
    """Declare `catalogManaged` in the table's own protocol, as Unity Catalog creates it.

    Unity Catalog also puts `vacuumProtocolCheck` on every managed table and
    its table id in the configuration; the kernel's committer refuses a
    catalog-managed table without either.
    """
    import json

    first = pathlib.Path(path) / "_delta_log" / f"{0:020d}.json"
    actions = [json.loads(line) for line in first.read_text().splitlines() if line.strip()]
    extra = ["vacuumProtocolCheck"] if vacuum_protocol_check else []
    for action in actions:
        if "protocol" in action:
            protocol = action["protocol"]
            readers = set(protocol.get("readerFeatures") or []) | {"catalogManaged", *extra}
            writers = set(protocol.get("writerFeatures") or []) | {
                "catalogManaged",
                "inCommitTimestamp",
                *extra,
            }
            action["protocol"] = {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": sorted(readers),
                "writerFeatures": sorted(writers),
            }
        if "metaData" in action:
            configuration = action["metaData"].setdefault("configuration", {})
            configuration["delta.enableInCommitTimestamps"] = "true"
            configuration["io.unitycatalog.tableId"] = action["metaData"]["id"]
    first.write_text("\n".join(json.dumps(a) for a in actions) + "\n")
    # Checksums record the old protocol; readers trust them over the log.
    for crc in first.parent.glob("*.crc"):
        crc.unlink()


@dataclass
class Tables:
    """Hands out tables: a shared read-only template, or a private writable copy."""

    root: pathlib.Path
    templates: dict[str, str]
    uc: FakeUnityCatalog
    counter: Iterator[int] = field(default_factory=itertools.count)

    def fresh_dir(self, stem: str) -> str:
        return str(self.root / "work" / f"{stem}-{next(self.counter)}")

    def path(self, fixture: str, *, writable: bool) -> str:
        template = self.templates[fixture]
        if not writable:
            return template
        return copy_table(template, self.fresh_dir(fixture))

    def catalog_managed(self, *, writable: bool, vacuum_protocol_check: bool = True) -> str:
        """A fully qualified name for a catalog-managed table with a staged tail."""
        from tests.integration.test_catalog_managed import _metadata_id, _stage_latest_commit

        shared = not writable and vacuum_protocol_check
        if shared and "main.contract.cm_shared" in self.uc.tables:
            return "main.contract.cm_shared"
        n = next(self.counter)
        path = copy_table(self.templates["_cm_base"], self.fresh_dir("uc"))
        _make_catalog_managed(path, vacuum_protocol_check=vacuum_protocol_check)
        commit = _stage_latest_commit(path)
        name = "cm_shared" if shared else f"cm_{n}"
        self.uc.add_table(
            f"main.contract.{name}",
            FakeTable(
                name=name,
                location=path,
                table_id=_metadata_id(path),
                properties={"delta.feature.catalogManaged": "supported"},
                commits=[commit],
                latest_version=commit["version"],
                table_type="MANAGED",
            ),
        )
        return f"main.contract.{name}"


@pytest.fixture(scope="session")
def contract_tables(tmp_path_factory: Any) -> Iterator[Tables]:
    from deltaswamp import Connection
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    root = tmp_path_factory.mktemp("contract")
    (root / "work").mkdir()
    conn = Connection(catalog=FilesystemCatalog(), router=helpers.direct_router())
    templates: dict[str, str] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fixture in FIXTURES:
            path = root / "templates" / fixture.name
            os.makedirs(path)
            fixture.build(conn, str(path))
            templates[fixture.name] = str(path)
        # The catalog-managed base: rows whose second commit is staged. A
        # catalog-managed table has in-commit timestamps on every commit.
        base = root / "templates" / "_cm_base"
        os.makedirs(base)
        next(f for f in FIXTURES if f.name == "ict").build(conn, str(base))
        templates["_cm_base"] = str(base)
    with FakeUnityCatalog() as uc:
        uc.add_schema("main", "contract")
        yield Tables(root, templates, uc)


def connect_path() -> Any:
    from deltaswamp import Connection
    from deltaswamp.catalog.filesystem import FilesystemCatalog

    return Connection(catalog=FilesystemCatalog(), router=helpers.direct_router())


def connect_uc(uc: FakeUnityCatalog) -> Any:
    from deltaswamp import Connection
    from deltaswamp.catalog.ossuc import OSSUnityCatalog

    return Connection(catalog=OSSUnityCatalog(uc.url), router=helpers.direct_router())
