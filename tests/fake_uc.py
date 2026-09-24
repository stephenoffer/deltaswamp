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
import tempfile
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
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
    owner: str | None = None
    comment: str | None = None
    #: Unity Catalog column dicts (name, type_text, position, ...).
    columns: list[dict[str, Any]] = field(default_factory=list)
    created_at: int | None = None
    created_by: str | None = None


#: What `staging-tables` demands of version 0, beyond the protocol. A None value
#: would mean "any value, but present"; `delta.feature.*` entries are satisfied
#: by the typed protocol, never by properties. `{table_id}` is substituted.
STAGING_REQUIRED_PROPERTIES: dict[str, str | None] = {
    "delta.feature.catalogManaged": "supported",
    "delta.feature.inCommitTimestamp": "supported",
    "delta.enableInCommitTimestamps": "true",
    "io.unitycatalog.tableId": "{table_id}",
}
STAGING_SUGGESTED_PROPERTIES: dict[str, str | None] = {
    "delta.enableDeletionVectors": "true",
    "delta.checkpoint.writeStatsAsStruct": "true",
}
STAGING_REQUIRED_PROTOCOL: dict[str, Any] = {
    "min-reader-version": 3,
    "min-writer-version": 7,
    "reader-features": ["catalogManaged"],
    "writer-features": ["catalogManaged", "inCommitTimestamp"],
}


class FakeUnityCatalog:
    """Serves a set of tables. Start it, point a catalog at `.url`, stop it."""

    def __init__(self, staging_root: str | Path | None = None) -> None:
        self.tables: dict[str, FakeTable] = {}
        #: Set to a status code to make the next commit fail that way.
        self.next_commit_status: int | None = None
        #: Every commit body received, for assertions.
        self.commit_log: list[dict[str, Any]] = []
        #: Every request, as (method, path-with-query, body), for assertions.
        self.requests: list[tuple[str, str, Any]] = []

        #: Where staging-tables allocates managed-table locations. A temporary
        #: directory is made on first use when none is given.
        self.staging_root: Path | None = Path(staging_root) if staging_root else None
        #: The `config` of the one storage credential staging returns. Empty,
        #: as for local `file://` storage; set s3.*/azure.*/gcs.* keys to
        #: exercise cloud credential mapping.
        self.staging_credential_config: dict[str, str] = {}
        #: Staging tables allocated but not yet created, by "catalog.schema.name".
        self.staged: dict[str, dict[str, Any]] = {}

        #: Explicitly created namespaces. Catalogs and schemas are also implied
        #: by the tables present, so `add_table` alone still lists them.
        self.catalogs: dict[str, dict[str, Any]] = {}
        self.schemas: dict[str, dict[str, Any]] = {}
        self.volumes: dict[str, dict[str, Any]] = {}
        self.functions: dict[str, dict[str, Any]] = {}
        #: (securable_type, full_name) -> principal -> privileges (OSS spelling).
        self.permissions: dict[tuple[str, str], dict[str, list[str]]] = {}
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

            def _query(self) -> dict[str, str]:
                return {
                    k: v[0] for k, v in urllib.parse.parse_qs(self.path.partition("?")[2]).items()
                }

            def _body(self) -> Any:
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length) or b"{}")

            def do_GET(self) -> None:
                path = urllib.parse.unquote(self.path.split("?")[0])
                catalog.requests.append(("GET", self.path, None))
                routed = catalog._route_get(path, self._query())
                if routed is not None:
                    return self._send(*routed)
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
                    names = sorted(catalog._catalog_names())
                    return self._send(200, {"catalogs": [{"name": n} for n in names]})
                if path == f"{UC_API}/schemas":
                    wanted = self._query().get("catalog_name", "")
                    names = sorted(
                        k.split(".")[1]
                        for k in catalog._schema_names()
                        if k.split(".")[0] == wanted
                    )
                    return self._send(200, {"schemas": [{"name": n} for n in names]})
                if path == f"{UC_API}/tables":
                    query = self.path.partition("?")[2]
                    params = dict(pair.split("=", 1) for pair in query.split("&") if "=" in pair)
                    prefix = f"{params.get('catalog_name')}.{params.get('schema_name')}."
                    listed = [
                        catalog._table_info(t, k)
                        for k, t in catalog.tables.items()
                        if k.startswith(prefix)
                    ]
                    return self._send(200, {"tables": listed})
                if path.startswith(f"{UC_API}/tables/"):
                    full = path.rsplit("/", 1)[-1].replace("%2E", ".")
                    table = catalog.tables.get(full)
                    if table is None:
                        return self._send(404, {"message": f"{full} does not exist"})
                    return self._send(200, catalog._table_info(table, full))
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
                path = urllib.parse.unquote(self.path.split("?")[0])
                catalog.requests.append(("DELETE", self.path, None))
                return self._send(*catalog._route_delete(path, self._query()))

            def do_PATCH(self) -> None:
                path = urllib.parse.unquote(self.path.split("?")[0])
                body = self._body()
                catalog.requests.append(("PATCH", self.path, body))
                return self._send(*catalog._route_patch(path, body))

            def do_POST(self) -> None:
                path = urllib.parse.unquote(self.path.split("?")[0])
                body = self._body()
                catalog.requests.append(("POST", self.path, body))
                routed = catalog._route_post(path, body)
                if routed is not None:
                    return self._send(*routed)

                # Anything else is a commit to an existing table.
                catalog.commit_log.append({"path": self.path, "body": body})
                if catalog.next_commit_status is not None:
                    code = catalog.next_commit_status
                    catalog.next_commit_status = None
                    messages = {
                        409: "a different commit with the same version already exists",
                        429: "the maximum number of unbackfilled commits has been reached",
                    }
                    return self._send(code, {"message": messages.get(code, "error")})
                catalog._ratify(self.path, body)
                return self._send(200, {})

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def _ratify(self, path: str, body: dict[str, Any]) -> None:
        """Apply a commit the way UC does: record it in the table's tail.

        `add-commit` joins the ratified-but-unpublished tail and advances the
        latest version; `set-latest-backfilled-version` drops commits the
        client has since published. A 409 for a version already taken is the
        server's conflict detection.
        """
        parts = path.split("?")[0].split("/")
        if "tables" not in parts or not isinstance(body.get("updates"), list):
            return
        try:
            full = f"{parts[-5]}.{parts[-3]}.{parts[-1]}"
        except IndexError:
            return
        table = self.tables.get(full)
        if table is None:
            return
        for update in body["updates"]:
            if update.get("action") == "add-commit":
                commit = update["commit"]
                table.commits.append(
                    {
                        "version": commit["version"],
                        "timestamp": commit["timestamp"],
                        "file_name": commit["file-name"],
                        "file_size": commit["file-size"],
                        "file_modification_timestamp": commit["file-modification-timestamp"],
                    }
                )
                table.latest_version = max(table.latest_version, int(commit["version"]))
            if update.get("action") == "set-latest-backfilled-version":
                published = int(update["latest-published-version"])
                table.commits = [c for c in table.commits if int(c["version"]) > published]

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

    def add_schema(self, catalog: str, schema: str) -> None:
        """Declare a namespace without a table in it."""
        self.catalogs.setdefault(catalog, {"name": catalog})
        self.schemas.setdefault(f"{catalog}.{schema}", {"name": schema, "catalog_name": catalog})

    def add_function(self, full_name: str, **info: Any) -> None:
        c, s, n = full_name.split(".")
        self.functions[full_name] = {
            "name": n,
            "catalog_name": c,
            "schema_name": s,
            "full_name": full_name,
            **info,
        }

    @staticmethod
    def _table_info(table: FakeTable, full_name: str | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": table.name,
            "table_id": table.table_id,
            "storage_location": table.location,
            "table_type": table.table_type,
            "data_source_format": table.data_source_format,
            "properties": table.properties,
            "columns": table.columns,
        }
        if full_name:
            c, s, _ = full_name.split(".")
            info.update(catalog_name=c, schema_name=s, full_name=full_name)
        for key in ("owner", "comment", "created_at", "created_by"):
            if getattr(table, key) is not None:
                info[key] = getattr(table, key)
        return info

    # --------------------------------------------------------------- routing
    # Each router returns (status, payload), or None to fall through to the
    # original handlers above.

    def _catalog_names(self) -> set[str]:
        return {k.split(".")[0] for k in self.tables} | set(self.catalogs)

    def _schema_names(self) -> set[str]:
        return {".".join(k.split(".")[:2]) for k in self.tables} | set(self.schemas)

    def _permission_body(self, key: tuple[str, str]) -> dict[str, Any]:
        grants = self.permissions.get(key, {})
        return {
            "privilege_assignments": [
                {"principal": p, "privileges": sorted(privs)}
                for p, privs in sorted(grants.items())
                if privs
            ]
        }

    def _route_get(self, path: str, query: dict[str, str]) -> tuple[int, Any] | None:
        if path.startswith(f"{UC_API}/permissions/"):
            securable_type, _, name = path[len(f"{UC_API}/permissions/") :].partition("/")
            body = self._permission_body((securable_type, name))
            if "principal" in query:
                body["privilege_assignments"] = [
                    a for a in body["privilege_assignments"] if a["principal"] == query["principal"]
                ]
            return 200, body
        prefix = f"{query.get('catalog_name')}.{query.get('schema_name')}."
        if path == f"{UC_API}/functions":
            found = [f for k, f in sorted(self.functions.items()) if k.startswith(prefix)]
            return 200, {"functions": found}
        if path == f"{UC_API}/volumes":
            found = [v for k, v in sorted(self.volumes.items()) if k.startswith(prefix)]
            return 200, {"volumes": found}
        return None

    def _route_delete(self, path: str, query: dict[str, str]) -> tuple[int, Any]:
        force = query.get("force") == "true"
        for kind, store in (("catalogs", self.catalogs), ("schemas", self.schemas)):
            if path.startswith(f"{UC_API}/{kind}/"):
                name = path.rsplit("/", 1)[-1]
                if name not in (
                    self._catalog_names() if kind == "catalogs" else self._schema_names()
                ):
                    return 404, {"message": f"{name} does not exist"}
                children = [k for k in [*self.tables, *self.schemas] if k.startswith(name + ".")]
                if children and not force:
                    return 400, {"message": f"{name} is not empty: {sorted(children)}"}
                for child in children:
                    self.tables.pop(child, None)
                    self.schemas.pop(child, None)
                store.pop(name, None)
                return 200, {}
        if path.startswith(f"{UC_API}/volumes/"):
            name = path.rsplit("/", 1)[-1]
            if self.volumes.pop(name, None) is None:
                return 404, {"message": f"{name} does not exist"}
            return 200, {}
        name = path.rsplit("/", 1)[-1]
        if self.tables.pop(name, None) is None:
            return 404, {"message": f"{name} does not exist"}
        return 200, {}

    def _route_patch(self, path: str, body: Any) -> tuple[int, Any]:
        if not path.startswith(f"{UC_API}/permissions/"):
            return 404, {"message": f"no route for PATCH {path}"}
        securable_type, _, name = path[len(f"{UC_API}/permissions/") :].partition("/")
        grants = self.permissions.setdefault((securable_type, name), {})
        for change in body.get("changes") or []:
            held = set(grants.get(change["principal"], []))
            held |= set(change.get("add") or [])
            held -= set(change.get("remove") or [])
            grants[change["principal"]] = sorted(held)
        return 200, self._permission_body((securable_type, name))

    def _route_post(self, path: str, body: Any) -> tuple[int, Any] | None:
        if path == f"{UC_API}/catalogs":
            if body["name"] in self._catalog_names():
                return 409, {"message": f"catalog {body['name']} already exists"}
            self.catalogs[body["name"]] = dict(body)
            return 200, dict(body)
        if path == f"{UC_API}/schemas":
            full = f"{body['catalog_name']}.{body['name']}"
            if body["catalog_name"] not in self._catalog_names():
                return 404, {"message": f"catalog {body['catalog_name']} does not exist"}
            if full in self._schema_names():
                return 409, {"message": f"schema {full} already exists"}
            self.schemas[full] = {**body, "full_name": full}
            return 200, self.schemas[full]
        if path == f"{UC_API}/volumes":
            parent = f"{body['catalog_name']}.{body['schema_name']}"
            if parent not in self._schema_names():
                return 404, {"message": f"schema {parent} does not exist"}
            full = f"{parent}.{body['name']}"
            info = {
                **body,
                "full_name": full,
                "volume_id": str(uuid.uuid4()),
                "storage_location": body.get("storage_location")
                or f"file:///volumes/{full.replace('.', '/')}",
            }
            self.volumes[full] = info
            return 200, info
        if path == f"{UC_API}/temporary-path-credentials":
            return 200, {
                "aws_temp_credentials": {
                    "access_key_id": "AKIAFAKEPATH",
                    "secret_access_key": "path-secret",
                    "session_token": "path-token",
                },
                "expiration_time": 9999999999000,
                "url": body.get("url", ""),
            }
        if path == f"{UC_API}/tables":
            return self._register_external(body)
        if path.startswith(f"{DELTA_API}/catalogs/"):
            parts = path[len(f"{DELTA_API}/catalogs/") :].split("/")
            # {c}/schemas/{s}/staging-tables  or  {c}/schemas/{s}/tables
            if len(parts) == 4 and parts[1] == "schemas":
                c, s, leaf = parts[0], parts[2], parts[3]
                if leaf == "staging-tables":
                    return self._stage(c, s, body)
                if leaf == "tables":
                    return self._create_managed(c, s, body)
        return None

    def _register_external(self, body: dict[str, Any]) -> tuple[int, Any]:
        parent = f"{body.get('catalog_name')}.{body.get('schema_name')}"
        full = f"{parent}.{body.get('name')}"
        if parent not in self._schema_names():
            return 404, {"message": f"schema {parent} does not exist"}
        if full in self.tables:
            return 409, {"message": f"table {full} already exists"}
        if body.get("table_type") != "EXTERNAL" or body.get("data_source_format") != "DELTA":
            return 400, {"message": "only EXTERNAL DELTA tables can be created through this API"}
        if not body.get("storage_location"):
            return 400, {"message": "storage_location is required"}
        table = FakeTable(
            name=body["name"],
            location=body["storage_location"],
            table_id=str(uuid.uuid4()),
            properties=dict(body.get("properties") or {}),
            table_type="EXTERNAL",
            comment=body.get("comment"),
            columns=list(body.get("columns") or []),
            created_at=int(time.time() * 1000),
        )
        self.tables[full] = table
        return 200, self._table_info(table, full)

    def _stage(self, c: str, s: str, body: dict[str, Any]) -> tuple[int, Any]:
        """Allocate a managed table: id, location and write credentials."""
        full = f"{c}.{s}.{body.get('name')}"
        if f"{c}.{s}" not in self._schema_names():
            return 404, {"message": f"schema {c}.{s} does not exist"}
        if full in self.tables:
            return 409, {"message": f"table {full} already exists"}
        if self.staging_root is None:
            self.staging_root = Path(tempfile.mkdtemp(prefix="fake-uc-staging-"))
        table_id = str(uuid.uuid4())
        directory = self.staging_root / table_id
        directory.mkdir(parents=True, exist_ok=True)
        location = directory.resolve().as_uri()
        required = {
            k: (v.replace("{table_id}", table_id) if v is not None else None)
            for k, v in STAGING_REQUIRED_PROPERTIES.items()
        }
        response = {
            "table-id": table_id,
            "table-type": "MANAGED",
            "location": location,
            "storage-credentials": [
                {
                    "prefix": location + "/",
                    "operation": "READ_WRITE",
                    "expiration-time-ms": 9999999999000,
                    "config": dict(self.staging_credential_config),
                }
            ],
            "required-protocol": STAGING_REQUIRED_PROTOCOL,
            "required-properties": required,
            "suggested-properties": dict(STAGING_SUGGESTED_PROPERTIES),
        }
        self.staged[full] = {"table_id": table_id, "location": location, "required": required}
        return 200, response

    def _create_managed(self, c: str, s: str, body: dict[str, Any]) -> tuple[int, Any]:
        """Promote a staging table, checking what the real server checks."""
        name = body.get("name")
        full = f"{c}.{s}.{name}"
        staged = self.staged.get(full)
        if staged is None:
            return 404, {"message": f"no staging table for {full}; call staging-tables first"}
        if str(body.get("location", "")).rstrip("/") != staged["location"].rstrip("/"):
            return 400, {
                "message": f"location {body.get('location')!r} is not the staged "
                f"location {staged['location']!r}"
            }
        if body.get("table-type", "MANAGED") != "MANAGED":
            return 400, {"message": "table-type must be MANAGED"}

        protocol = body.get("protocol") or {}
        features = set(protocol.get("reader-features") or []) | set(
            protocol.get("writer-features") or []
        )
        properties = dict(body.get("properties") or {})
        for key, value in staged["required"].items():
            if key.startswith("delta.feature."):
                feature = key[len("delta.feature.") :]
                if feature not in features:
                    return 400, {"message": f"required table feature {feature} is missing"}
                if key in properties:
                    return 400, {
                        "message": f"{key} must come from the typed protocol, not properties"
                    }
            elif key not in properties or (value is not None and properties[key] != value):
                return 400, {
                    "message": f"required property {key}={value!r} is missing or wrong "
                    f"(got {properties.get(key)!r})"
                }

        location = staged["location"]
        if location.startswith("file://"):
            local = Path(urllib.parse.urlparse(location).path)
            first = local / "_delta_log" / f"{0:020d}.json"
            if not first.exists():
                return 400, {"message": "version 0 has not been written at the staged location"}
            # The catalog's table id and the log's Metadata.id are one identity.
            for line in first.read_text().splitlines():
                action = json.loads(line) if line.strip() else {}
                log_id = (action.get("metaData") or {}).get("id")
                if log_id is not None and log_id != staged["table_id"]:
                    return 400, {
                        "message": f"version 0 has Metadata.id {log_id!r}, but the staged "
                        f"table id is {staged['table_id']!r}"
                    }

        columns = [
            {"name": f["name"], "type_text": str(f["type"]), "position": i}
            for i, f in enumerate((body.get("columns") or {}).get("fields") or [])
        ]
        self.tables[full] = FakeTable(
            name=str(name),
            location=location,
            table_id=staged["table_id"],
            # How the catalog reports a catalog-managed table in its properties.
            properties={**properties, "delta.feature.catalogManaged": "supported"},
            table_type="MANAGED",
            comment=body.get("comment"),
            columns=columns,
            latest_version=0,
            created_at=int(time.time() * 1000),
        )
        del self.staged[full]
        return 200, {
            "metadata": {
                "etag": str(uuid.uuid4()),
                "table-type": "MANAGED",
                "table-uuid": staged["table_id"],
                "location": location,
                "columns": body.get("columns"),
                "partition-columns": body.get("partition-columns") or [],
                "properties": properties,
                "last-commit-version": 0,
            },
            "commits": [],
            "latest-table-version": 0,
        }
