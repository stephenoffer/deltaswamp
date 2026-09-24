"""A minimal Unity Catalog server, speaking the real `/delta/v1` protocol.

Docker is not available here and Databricks needs an account, so the
catalog-managed path would otherwise be untestable -- which is exactly the
capability this project exists for. This serves the endpoints from open-source
Unity Catalog's `ManagedTablesSpec.md` over stdlib HTTP.

It is deliberately a real protocol implementation rather than a mock: the
client code under test issues genuine requests and parses genuine responses,
and the error cases that matter (409 on a lost commit, 429 demanding backfill)
can be produced on demand, which a live server will not do to order.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

UC_API = "/api/2.1/unity-catalog"
DELTA_API = f"{UC_API}/delta/v1"


@dataclass
class FakeTable:
    """One table as the catalog knows it."""

    name: str
    location: str
    table_id: str
    properties: dict[str, str] = field(default_factory=dict)
    #: Ratified-but-unpublished commits: the log tail.
    commits: list[dict[str, Any]] = field(default_factory=list)
    latest_version: int = 0
    data_source_format: str = "DELTA"
    table_type: str = "EXTERNAL"


class FakeUnityCatalog:
    """Serves a set of tables. Start it, point a catalog at `.url`, stop it."""

    def __init__(self) -> None:
        self.tables: dict[str, FakeTable] = {}
        #: Set to a status code to make the next commit fail that way.
        self.next_commit_status: int | None = None
        #: Every commit body received, for assertions.
        self.commit_log: list[dict[str, Any]] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- lifecycle

    def start(self) -> FakeUnityCatalog:
        catalog = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass  # keep the test output readable

            def _send(self, code: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path == f"{DELTA_API}/config":
                    return self._send(
                        200,
                        {
                            "endpoints": [
                                "GET /delta/v1/config",
                                "GET /delta/v1/catalogs/{c}/schemas/{s}/tables/{t}",
                                "POST /delta/v1/catalogs/{c}/schemas/{s}/tables/{t}",
                                "POST /delta/v1/catalogs/{c}/schemas/{s}/staging-tables",
                            ],
                            "protocol-version": "1.0",
                        },
                    )
                if path == f"{UC_API}/catalogs":
                    names = sorted({k.split(".")[0] for k in catalog.tables})
                    return self._send(200, {"catalogs": [{"name": n} for n in names]})
                if path == f"{UC_API}/schemas":
                    wanted = self.path.partition("catalog_name=")[2].split("&")[0]
                    names = sorted(
                        {k.split(".")[1] for k in catalog.tables if k.split(".")[0] == wanted}
                    )
                    return self._send(200, {"schemas": [{"name": n} for n in names]})
                if path == f"{UC_API}/tables":
                    query = self.path.partition("?")[2]
                    params = dict(pair.split("=", 1) for pair in query.split("&") if "=" in pair)
                    prefix = f"{params.get('catalog_name')}.{params.get('schema_name')}."
                    listed = [
                        catalog._table_info(t)
                        for k, t in catalog.tables.items()
                        if k.startswith(prefix)
                    ]
                    return self._send(200, {"tables": listed})
                if path.startswith(f"{UC_API}/tables/"):
                    full = path.rsplit("/", 1)[-1].replace("%2E", ".")
                    table = catalog.tables.get(full)
                    if table is None:
                        return self._send(404, {"message": f"{full} does not exist"})
                    return self._send(200, catalog._table_info(table))
                if path.endswith("/credentials"):
                    return self._send(
                        200,
                        {
                            "aws_temp_credentials": {
                                "access_key_id": "AKIAFAKE",
                                "secret_access_key": "secret",
                                "session_token": "token",
                            },
                            "expiration_time": 9999999999000,
                            "url": "s3://fake/t",
                        },
                    )
                if path.startswith(f"{DELTA_API}/catalogs/"):
                    parts = path.split("/")
                    full = f"{parts[-5]}.{parts[-3]}.{parts[-1]}"
                    table = catalog.tables.get(full)
                    if table is None:
                        return self._send(404, {"message": f"{full} does not exist"})
                    return self._send(
                        200,
                        {
                            "table_id": table.table_id,
                            "location": table.location,
                            "commits": table.commits,
                            "latest_table_version": table.latest_version,
                            "metadata": {"properties": table.properties},
                        },
                    )
                return self._send(404, {"message": f"no route for {path}"})

            def do_DELETE(self) -> None:
                full = self.path.split("?")[0].rsplit("/", 1)[-1].replace("%2E", ".")
                if catalog.tables.pop(full, None) is None:
                    return self._send(404, {"message": f"{full} does not exist"})
                return self._send(200, {})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                catalog.commit_log.append({"path": self.path, "body": body})

                if catalog.next_commit_status is not None:
                    code = catalog.next_commit_status
                    catalog.next_commit_status = None
                    messages = {
                        409: "a different commit with the same version already exists",
                        429: "the maximum number of unbackfilled commits has been reached",
                    }
                    return self._send(code, {"message": messages.get(code, "error")})
                return self._send(200, {})

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def __enter__(self) -> FakeUnityCatalog:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def url(self) -> str:
        assert self._server is not None, "start() the server first"
        host, port = self._server.server_address[:2]
        # server_address gives bytes for the host on some platforms.
        hostname = host.decode() if isinstance(host, bytes) else str(host)
        return f"http://{hostname}:{port}"

    # ----------------------------------------------------------------- state

    def add_table(self, full_name: str, table: FakeTable) -> FakeTable:
        self.tables[full_name] = table
        return table

    @staticmethod
    def _table_info(table: FakeTable) -> dict[str, Any]:
        return {
            "name": table.name,
            "table_id": table.table_id,
            "storage_location": table.location,
            "table_type": table.table_type,
            "data_source_format": table.data_source_format,
            "properties": table.properties,
        }
