"""The Databricks SQL warehouse fallback.

Opt-in, never automatic. Everything works here -- views, materialized views,
row-filtered tables, catalog-managed tables, DROP FEATURE, REORG, CLONE, column
rename -- because the work happens server-side. That is also why it is off by
default: rerouting a scan through a warehouse changes latency, egress and DBU
cost by orders of magnitude, and a silent reroute would turn a performance cliff
into a mystery.

When this engine serves an operation, it says so.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..capability import OPERATION_ENGINES, Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable
from ..errors import UnreachableTableError
from ..identity import RefKind
from .base import missing_method

__all__ = ["SqlEngine", "SqlFallbackWarning"]


def _quote(identifier: str) -> str:
    """Backtick-quote an identifier, doubling any embedded backtick."""
    return "`" + identifier.replace("`", "``") + "`"


class SqlFallbackWarning(UserWarning):
    """Raised when an operation is served by a SQL warehouse rather than directly."""


class SqlEngine:
    """Executes operations as SQL against a Databricks warehouse."""

    kind = EngineKind.SQL
    supports_distributed_scan = False
    supports_predicates = True
    supports_timestamp_travel = True
    supports_schema_merge = True
    supports_schema_overwrite = True
    supports_idempotent_txn = True
    supports_commit_metadata = True
    supports_writer_properties = False
    supports_dynamic_overwrite = True

    def __init__(
        self,
        *,
        warehouse_id: str | None = None,
        http_path: str | None = None,
        profile: str | None = None,
        host: str | None = None,
        token: str | None = None,
        config: Any = None,
        warn_on_use: bool = True,
    ) -> None:
        self._warehouse_id = warehouse_id
        self._http_path = http_path
        self._profile = profile
        self._host = host
        self._token = token
        self._config = config
        self._warn_on_use = warn_on_use

    # ----------------------------------------------------------- capabilities

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the databricks-sql-connector package is not installed",
                remedy="pip install 'deltaswamp[sql]'",
            )
        if self._http_path is None and self._warehouse_id is None:
            return Capability(
                operation,
                ok=False,
                reason="no SQL warehouse configured",
                remedy="ds.connect(..., allow_sql_fallback=True, warehouse_id='<id>')",
            )
        # Structural guard: never claim an operation with no method behind it.
        gap = missing_method(self, operation)
        if gap is not None:
            return gap

        routing = OPERATION_ENGINES.get(operation)
        if routing is None or self.kind not in routing.engines:
            return Capability(
                operation, ok=False, reason=f"{operation.value} is not expressible as SQL here"
            )
        if table.ref.kind is not RefKind.CATALOG:
            return Capability(
                operation,
                ok=False,
                reason="a SQL warehouse addresses tables by name; this is a path reference",
            )
        return Capability(operation, ok=True, engine=self.kind)

    @staticmethod
    def available() -> bool:
        try:
            import databricks.sql  # noqa: F401
        except ImportError:
            return False
        return True

    # --------------------------------------------------------------- plumbing

    def _connection(self) -> Any:
        from databricks import sql as dbsql
        from databricks.sdk.core import Config

        cfg = self._config or Config(profile=self._profile, host=self._host, token=self._token)
        http_path = self._http_path or f"/sql/1.0/warehouses/{self._warehouse_id}"
        return dbsql.connect(
            server_hostname=cfg.host.replace("https://", "").rstrip("/"),
            http_path=http_path,
            credentials_provider=lambda: cfg.authenticate,
        )

    def _notify(self, operation: Operation) -> None:
        if self._warn_on_use:
            warnings.warn(
                f"{operation.value} is being served by a Databricks SQL warehouse rather "
                "than by direct object-store access. Expect materially different latency "
                "and cost.",
                SqlFallbackWarning,
                stacklevel=3,
            )

    def execute(self, operation: Operation, statement: str, fetch: bool = True) -> Any:
        self._notify(operation)
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(statement)
            if not fetch:
                return None
            return cur.fetchall_arrow()

    # ------------------------------------------------------------------- read

    def scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
    ) -> Any:
        name = table.ref.full_name
        # Quote identifiers: a column named `order` or `my col` breaks otherwise.
        projection = ", ".join(_quote(c) for c in columns) if columns else "*"
        sql = f"SELECT {projection} FROM {name}"
        if version is not None:
            sql += f" VERSION AS OF {int(version)}"
        elif timestamp is not None:
            sql += f" TIMESTAMP AS OF '{timestamp}'"
        if predicate:
            sql += f" WHERE {predicate}"
        return self.execute(Operation.SCAN, sql)

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = f"DESCRIBE HISTORY {table.ref.full_name}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        result = self.execute(Operation.HISTORY, sql)
        return result.to_pylist() if hasattr(result, "to_pylist") else list(result)

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        result = self.execute(Operation.DETAIL, f"DESCRIBE DETAIL {table.ref.full_name}")
        rows = result.to_pylist() if hasattr(result, "to_pylist") else list(result)
        return rows[0] if rows else {}

    # ------------------------------------------------------------------ write

    def append(self, table: ResolvedTable, data: Any, **kwargs: Any) -> None:
        raise UnreachableTableError(
            "append via SQL",
            "bulk-loading data through a SQL warehouse means staging files and running "
            "COPY INTO, which this engine does not implement",
            "write to a table that supports direct object-store access, or load via Databricks",
        )

    overwrite = append

    def delete(self, table: ResolvedTable, predicate: str | None = None) -> dict[str, Any]:
        sql = f"DELETE FROM {table.ref.full_name}"
        if predicate:
            sql += f" WHERE {predicate}"
        self.execute(Operation.DELETE, sql, fetch=False)
        return {"status": "ok"}

    def update(
        self,
        table: ResolvedTable,
        *,
        updates: dict[str, str] | None = None,
        predicate: str | None = None,
    ) -> dict[str, Any]:
        if not updates:
            raise UnreachableTableError("update", "no updates given")
        assignments = ", ".join(f"{k} = {v}" for k, v in updates.items())
        sql = f"UPDATE {table.ref.full_name} SET {assignments}"
        if predicate:
            sql += f" WHERE {predicate}"
        self.execute(Operation.UPDATE, sql, fetch=False)
        return {"status": "ok"}

    # ------------------------------------------------- Databricks-only surface

    def reorg(self, table: ResolvedTable, *, purge: bool = True) -> dict[str, Any]:
        clause = "APPLY (PURGE)" if purge else "APPLY (UPGRADE UNIFORM)"
        self.execute(Operation.REORG, f"REORG TABLE {table.ref.full_name} {clause}", fetch=False)
        return {"status": "ok"}

    def drop_feature(
        self, table: ResolvedTable, feature: str, *, truncate_history: bool = False
    ) -> dict[str, Any]:
        sql = f"ALTER TABLE {table.ref.full_name} DROP FEATURE {feature}"
        if truncate_history:
            sql += " TRUNCATE HISTORY"
        self.execute(Operation.DROP_FEATURE, sql, fetch=False)
        return {"status": "ok"}

    def rename_column(self, table: ResolvedTable, old: str, new: str) -> dict[str, Any]:
        self.execute(
            Operation.RENAME_COLUMN,
            f"ALTER TABLE {table.ref.full_name} RENAME COLUMN {_quote(old)} TO {_quote(new)}",
            fetch=False,
        )
        return {"status": "ok"}

    def drop_column(self, table: ResolvedTable, column: str) -> dict[str, Any]:
        self.execute(
            Operation.DROP_COLUMN,
            f"ALTER TABLE {table.ref.full_name} DROP COLUMN {_quote(column)}",
            fetch=False,
        )
        return {"status": "ok"}

    def sync_iceberg_metadata(self, table: ResolvedTable) -> dict[str, Any]:
        """Regenerate Iceberg metadata after an external write.

        Required when anything other than Databricks writes to a table with
        Iceberg reads enabled; without it the Iceberg view silently goes stale.
        """
        self.execute(
            Operation.REORG,
            f"MSCK REPAIR TABLE {table.ref.full_name} SYNC METADATA",
            fetch=False,
        )
        return {"status": "ok"}

    def set_properties(self, table: ResolvedTable, properties: dict[str, str]) -> dict[str, Any]:
        pairs = ", ".join(f"'{k}' = '{v}'" for k, v in properties.items())
        self.execute(
            Operation.SET_PROPERTIES,
            f"ALTER TABLE {table.ref.full_name} SET TBLPROPERTIES ({pairs})",
            fetch=False,
        )
        return {"status": "ok"}

    def add_feature(self, table: ResolvedTable, feature: str) -> dict[str, Any]:
        self.execute(
            Operation.ADD_FEATURE,
            f"ALTER TABLE {table.ref.full_name} "
            f"SET TBLPROPERTIES ('delta.feature.{feature}' = 'supported')",
            fetch=False,
        )
        return {"status": "ok"}

    def add_constraint(self, table: ResolvedTable, constraints: dict[str, str]) -> dict[str, Any]:
        for name, expression in constraints.items():
            self.execute(
                Operation.ADD_CONSTRAINT,
                f"ALTER TABLE {table.ref.full_name} ADD CONSTRAINT {_quote(name)} "
                f"CHECK ({expression})",
                fetch=False,
            )
        return {"status": "ok"}

    def add_columns(self, table: ResolvedTable, fields: Any) -> dict[str, Any]:
        """`fields` is a mapping of column name to SQL type."""
        if not isinstance(fields, dict):
            raise UnreachableTableError(
                "add columns via SQL",
                "the SQL engine needs a {name: sql_type} mapping, not an Arrow schema",
            )
        pairs = ", ".join(f"{_quote(n)} {t}" for n, t in fields.items())
        self.execute(
            Operation.ADD_COLUMN,
            f"ALTER TABLE {table.ref.full_name} ADD COLUMNS ({pairs})",
            fetch=False,
        )
        return {"status": "ok"}

    def clone(
        self, table: ResolvedTable, target: str, *, shallow: bool = True, replace: bool = False
    ) -> dict[str, Any]:
        verb = "CREATE OR REPLACE TABLE" if replace else "CREATE TABLE"
        kind = "SHALLOW CLONE" if shallow else "DEEP CLONE"
        self.execute(Operation.CLONE, f"{verb} {target} {kind} {table.ref.full_name}", fetch=False)
        return {"status": "ok"}

    def cdf(
        self,
        table: ResolvedTable,
        *,
        starting_version: int | None = None,
        ending_version: int | None = None,
        columns: list[str] | None = None,
    ) -> Any:
        projection = ", ".join(_quote(c) for c in columns) if columns else "*"
        start = 0 if starting_version is None else int(starting_version)
        args = f"{start}" if ending_version is None else f"{start}, {int(ending_version)}"
        return self.execute(
            Operation.CDF,
            f"SELECT {projection} FROM table_changes('{table.ref.full_name}', {args})",
        )

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError("the SQL engine has no split-planning surface")

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")
