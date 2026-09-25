"""A stricter Unity Catalog fake, layered on `tests/fake_uc.py`.

`FakeUnityCatalog` ratifies every commit it is sent. A real server does not:
it refuses a version that is already taken (409), a commit whose
`assert-table-uuid` names a table that has since been dropped and re-created
(the requirement exists for exactly that), and a commit past its cap on
unbackfilled commits (429). It vends table credentials through the Delta API
in the `storage-credentials` shape (unity-catalog-delta-client-api
`CredentialsResponse`), per operation, with an expiry. This subclass models
those contracts so the end-to-end flows can be checked against them.
"""

from __future__ import annotations

import time
import urllib.parse
from typing import Any

from tests.fake_uc import DELTA_API, UC_API, FakeUnityCatalog


class StrictUnityCatalog(FakeUnityCatalog):
    """`FakeUnityCatalog` with the server-side checks a real metastore makes."""

    def __init__(self, *args: Any, max_unbackfilled: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: The server's cap on ratified-but-unpublished commits (429 past it).
        self.max_unbackfilled = max_unbackfilled
        #: Operations asked of the Delta API credentials endpoint, in order.
        self.credential_operations: list[str] = []
        #: Lifetime, in seconds, of each vended table credential.
        self.credential_ttl: float = 3600.0
        #: Full names whose credential vending is refused (row filter / mask).
        self.vending_refused: set[str] = set()
        #: Commit requirement/validation failures, for assertions.
        self.refusals: list[str] = []
        #: Extra table-info fields by full name (row_filter, securable_kind...).
        self.table_extras: dict[str, dict[str, Any]] = {}
        #: Served by GET metastore_summary (Databricks), when set.
        self.metastore_region: str | None = None

    def _table_info(self, table: Any, full_name: str | None = None) -> dict[str, Any]:
        info = FakeUnityCatalog._table_info(table, full_name)
        info.update(self.table_extras.get(full_name or "", {}))
        return info

    # ------------------------------------------------------------- commits

    def _check_commit(self, path: str, body: dict[str, Any]) -> tuple[int, Any] | None:
        parts = path.split("?")[0].split("/")
        try:
            full = f"{parts[-5]}.{parts[-3]}.{parts[-1]}"
        except IndexError:
            return None
        table = self.tables.get(full)
        if table is None:
            return 404, {"message": f"{full} does not exist"}
        for req in body.get("requirements") or []:
            if req.get("type") == "assert-table-uuid" and req.get("uuid") != table.table_id:
                self.refusals.append("uuid")
                return 409, {
                    "error_code": "INVALID_STATE",
                    "message": f"table uuid {req.get('uuid')} does not match {table.table_id}",
                }
        updates = body.get("updates") or []
        published = next(
            (
                int(u["latest-published-version"])
                for u in updates
                if u.get("action") == "set-latest-backfilled-version"
            ),
            None,
        )
        for update in updates:
            if update.get("action") != "add-commit":
                continue
            version = int(update["commit"]["version"])
            if version != table.latest_version + 1:
                self.refusals.append("version")
                return 409, {
                    "message": f"commit version {version} is not the next version "
                    f"({table.latest_version + 1})"
                }
            pending = [
                c for c in table.commits if published is None or int(c["version"]) > published
            ]
            if self.max_unbackfilled is not None and len(pending) >= self.max_unbackfilled:
                self.refusals.append("backfill")
                return 429, {
                    "message": "the maximum number of unbackfilled commits has been reached"
                }
        return None

    def _route_post(self, path: str, body: Any) -> tuple[int, Any] | None:
        if path == "/api/2.0/unity-catalog/temporary-path-credentials":
            # The SDK's path vending lives under 2.0; the base fake serves 2.1.
            self.credential_operations.append(str(body.get("operation", "")))
            return super()._route_post(f"{UC_API}/temporary-path-credentials", body)
        if path.endswith("/unity-catalog/temporary-table-credentials"):
            # Databricks' table vending (SDK: POST /api/2.0/...), by table id.
            table = next(
                (t for t in self.tables.values() if t.table_id == body.get("table_id")), None
            )
            operation = str(body.get("operation", ""))
            self.credential_operations.append(operation)
            if table is None:
                return 404, {
                    "error_code": "TABLE_DOES_NOT_EXIST",
                    "message": f"table {body.get('table_id')} does not exist",
                }
            full = next(k for k, t in self.tables.items() if t is table)
            if full in self.vending_refused:
                return 403, {
                    "error_code": "PERMISSION_DENIED",
                    "message": "credential vending is not allowed for tables with row "
                    "filters or column masks",
                }
            return 200, {
                "aws_temp_credentials": {
                    "access_key_id": "AKIAFAKE",
                    "secret_access_key": "secret",
                    "session_token": "token",
                },
                "expiration_time": int((time.time() + self.credential_ttl) * 1000),
                "url": table.location,
            }
        routed = super()._route_post(path, body)
        if routed is not None:
            return routed
        if path.startswith(f"{DELTA_API}/catalogs/") and isinstance(body, dict):
            checked = self._check_commit(path, body)
            if checked is not None:
                self.commit_log.append({"path": path, "body": body, "refused": checked[0]})
                return checked
        return None

    def _ratify(self, path: str, body: dict[str, Any]) -> None:
        # The base fake splits the raw request path, so a table whose name
        # needs escaping (a space, non-ASCII) was answered 200 and never
        # recorded -- the commit silently vanished.
        super()._ratify(urllib.parse.unquote(path.split("?")[0]), body)

    # --------------------------------------------------------- credentials

    def _route_get(self, path: str, query: dict[str, str]) -> tuple[int, Any] | None:
        if path == f"{UC_API}/metastore_summary" and self.metastore_region:
            return 200, {"metastore_id": "m1", "region": self.metastore_region}
        if path.startswith(f"{UC_API}/tables/") and path.endswith("/exists"):
            full = path[len(f"{UC_API}/tables/") : -len("/exists")]
            parent = full.rsplit(".", 1)[0]
            if parent not in self._schema_names():
                return 404, {"error_code": "SCHEMA_DOES_NOT_EXIST", "message": parent}
            return 200, {"table_exists": full in self.tables}
        if path.startswith(f"{DELTA_API}/catalogs/") and path.endswith("/credentials"):
            parts = path.split("/")
            full = f"{parts[-6]}.{parts[-4]}.{parts[-2]}"
            table = self.tables.get(full)
            if table is None:
                return 404, {"message": f"{full} does not exist"}
            operation = query.get("operation", "")
            self.credential_operations.append(operation)
            if operation not in ("READ", "READ_WRITE"):
                return 400, {"message": f"invalid operation {operation!r}"}
            if full in self.vending_refused:
                return 403, {
                    "error_code": "PERMISSION_DENIED",
                    "message": "credential vending is not allowed for tables with row "
                    "filters or column masks",
                }
            return 200, {
                "storage-credentials": [
                    {
                        "prefix": table.location.rstrip("/") + "/",
                        "operation": operation,
                        "expiration-time-ms": int((time.time() + self.credential_ttl) * 1000),
                        "config": {},
                    }
                ]
            }
        return super()._route_get(path, query)

    # ---------------------------------------------------------- lifecycle

    def recreate(self, full_name: str) -> str:
        """Drop `full_name` and re-create it under a new table id (same name)."""
        table = self.tables[full_name]
        import dataclasses
        import uuid

        new_id = str(uuid.uuid4())
        self.tables[full_name] = dataclasses.replace(
            table, table_id=new_id, commits=[], latest_version=0
        )
        return new_id

    def table_path(self, full_name: str) -> str:
        c, s, t = full_name.split(".")
        q = urllib.parse.quote
        return f"{DELTA_API}/catalogs/{q(c)}/schemas/{q(s)}/tables/{q(t)}"


__all__ = ["UC_API", "StrictUnityCatalog"]
