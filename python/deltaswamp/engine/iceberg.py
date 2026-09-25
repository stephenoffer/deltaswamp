"""The Iceberg engine: tables behind a catalog's Iceberg REST endpoint, via PyIceberg.

Unity Catalog serves the Iceberg REST catalog protocol for its Iceberg tables
and for UniForm Delta tables (whose Iceberg metadata it generates from the Delta
log). Databricks serves it at ``{host}/api/2.1/unity-catalog/iceberg-rest`` and
open-source Unity Catalog at ``{base}/api/2.1/unity-catalog/iceberg``; either
way the warehouse is the UC catalog name and the Iceberg identifier is
``(schema, table)``. The catalog sets `ResolvedTable.iceberg_rest_uri`, and this
engine serves only tables that carry one.

What it serves:

* **native Iceberg tables** -- scan, time travel (a snapshot id passed as
  `version`, or a timestamp), history (the snapshot log), detail, append, and
  overwrite: whole-table, by predicate (``overwrite_filter``), or dynamic
  partition overwrite.
* **UniForm Delta tables** -- reads only. Their Iceberg metadata is derived from
  the Delta log, so a write through Iceberg would diverge from the Delta table;
  writes belong to the Delta engines.

Auth: the catalog token comes from ``credential_provider.workspace_auth()`` on
every catalog build. The SDK refreshes that token on its own clock, so a cached
catalog is rebuilt as soon as the token it was built with changes, rather than
holding one until it expires. Storage credentials are vended per table by the
REST server (``X-Iceberg-Access-Delegation: vended-credentials``).
"""

from __future__ import annotations

import importlib
import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

from ..capability import Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable
from ..errors import SQL_FALLBACK_REMEDY, UnreachableTableError
from .base import missing_method
from .sharing import filter_arrow_exact

__all__ = ["ACCESS_DELEGATION_HEADER", "IcebergEngine", "iceberg_rest_uri"]

ACCESS_DELEGATION_HEADER = "header.X-Iceberg-Access-Delegation"

_READS: frozenset[Operation] = frozenset(
    {Operation.SCAN, Operation.TIME_TRAVEL, Operation.HISTORY, Operation.DETAIL}
)
_WRITES: frozenset[Operation] = frozenset(
    {Operation.APPEND, Operation.OVERWRITE, Operation.REPLACE_WHERE}
)

#: The snapshot-summary key UniForm records the source Delta version under.
_UNIFORM_DELTA_VERSION = "delta-version"

CatalogFactory = Callable[[str, dict[str, str]], Any]


def iceberg_rest_uri(base_url: str, *, databricks: bool) -> str:
    """The Iceberg REST endpoint of a Unity Catalog server.

    Databricks: ``{host}/api/2.1/unity-catalog/iceberg-rest``.
    Open-source Unity Catalog: ``{base}/api/2.1/unity-catalog/iceberg``.
    """
    suffix = "iceberg-rest" if databricks else "iceberg"
    return f"{base_url.rstrip('/')}/api/2.1/unity-catalog/{suffix}"


def _default_factory(name: str, properties: dict[str, str]) -> Any:
    # Looked up at call time, so `pyiceberg.catalog.load_catalog` can be patched.
    return importlib.import_module("pyiceberg.catalog").load_catalog(name, **properties)


def _timestamp_ms(timestamp: str) -> int:
    text = timestamp.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UnreachableTableError(
            "time travel by timestamp",
            f"{timestamp!r} is not an ISO-8601 timestamp or epoch milliseconds",
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def _to_arrow_table(data: Any) -> Any:
    pa = importlib.import_module("pyarrow")
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all()
    return pa.table(data)  # PyCapsule streams, pandas, dict-of-columns


def _snapshot_properties(commit_metadata: dict[str, Any] | None) -> dict[str, str]:
    if not commit_metadata:
        return {}
    return {
        str(k): v if isinstance(v, str) else json.dumps(v, default=str)
        for k, v in commit_metadata.items()
    }


def _as_json(model: Any) -> Any:
    return json.loads(model.model_dump_json())


class IcebergEngine:
    """Reads and writes Iceberg tables through a catalog's Iceberg REST endpoint."""

    kind = EngineKind.ICEBERG
    supports_distributed_scan = False
    #: Pushed down when PyIceberg's parser accepts it, else applied exactly after.
    supports_predicates = True
    supports_timestamp_travel = True
    #: Recorded as Iceberg snapshot summary properties.
    supports_commit_metadata = True
    #: PyIceberg's dynamic_partition_overwrite.
    supports_dynamic_overwrite = True
    supports_schema_merge = False
    supports_schema_overwrite = False
    supports_idempotent_txn = False
    supports_writer_properties = False

    def __init__(
        self,
        *,
        properties: dict[str, str] | None = None,
        token: str | None = None,
        catalog_factory: CatalogFactory | None = None,
    ) -> None:
        """
        `properties` are extra PyIceberg REST catalog properties (TLS settings,
        ``rest.signing-*``, ...), applied over the defaults this engine sets.
        `token` is used only for tables whose credential provider offers no
        ``workspace_auth()``. `catalog_factory(name, properties)` replaces
        ``pyiceberg.catalog.load_catalog``.
        """
        self._properties = dict(properties or {})
        self._token = token
        self._factory: CatalogFactory = catalog_factory or _default_factory
        # (uri, warehouse) -> (token it was built with, catalog)
        self._catalogs: dict[tuple[str, str], tuple[str | None, Any]] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_catalogs"] = {}
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ----------------------------------------------------------- capabilities

    @staticmethod
    def available() -> bool:
        try:
            importlib.import_module("pyiceberg.catalog")
        except ImportError:
            return False
        return True

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        if not self.available():
            return Capability(
                operation,
                ok=False,
                reason="the pyiceberg package is not installed",
                remedy="pip install 'deltaswamp[iceberg]'",
            )

        gap = missing_method(self, operation)
        if gap is not None:
            return gap

        if not table.iceberg_rest_uri:
            return Capability(
                operation,
                ok=False,
                reason="the table's catalog advertises no Iceberg REST endpoint",
                remedy="reach the table through a Unity Catalog connection (Databricks or "
                "open-source), which serves one",
            )

        uniform = not table.is_iceberg
        if uniform and not table.has_iceberg_compat:
            return Capability(
                operation,
                ok=False,
                reason="the table is Delta without UniForm Iceberg metadata, so the Iceberg "
                "REST endpoint does not serve it",
                remedy="the Delta engines serve it",
            )

        if table.external_read_supported is False:
            return Capability(
                operation,
                ok=False,
                reason="the catalog reports no external-engine read support for this table "
                "(usually a row filter or column mask)",
                remedy=SQL_FALLBACK_REMEDY,
            )

        if operation in _WRITES:
            if uniform:
                return Capability(
                    operation,
                    ok=False,
                    reason="the table is UniForm: its Iceberg metadata is generated from the "
                    "Delta log, so the Iceberg view is read-only and a write through it would "
                    "diverge from the Delta table",
                    remedy="write through the Delta engines",
                )
            if table.external_write_supported is False:
                return Capability(
                    operation,
                    ok=False,
                    reason="the catalog reports no external-engine write support for this table",
                    remedy=SQL_FALLBACK_REMEDY,
                )
        elif operation not in _READS:
            return Capability(
                operation,
                ok=False,
                reason=f"the Iceberg engine does not implement {operation.value}",
            )

        return Capability(operation, ok=True, engine=self.kind)

    # ---------------------------------------------------------------- catalog

    @staticmethod
    def _identifier(table: ResolvedTable) -> tuple[str, str, str]:
        ref = table.ref
        if not (ref.catalog and ref.schema and ref.table):
            raise UnreachableTableError(
                "open the table through Iceberg REST",
                f"{ref} is not a catalog.schema.table name, which the endpoint needs "
                "(warehouse = catalog, identifier = schema.table)",
            )
        return ref.catalog, ref.schema, ref.table

    def _fresh_token(self, table: ResolvedTable) -> str | None:
        """The catalog token, asked for anew on every catalog lookup."""
        auth = getattr(table.credential_provider, "workspace_auth", None)
        if auth is not None:
            _, token = auth()
            return str(token) or None
        return self._token

    def rest_properties(self, table: ResolvedTable) -> dict[str, str]:
        """The PyIceberg REST catalog properties for `table`, with a fresh token."""
        if not table.iceberg_rest_uri:
            raise UnreachableTableError(
                "open the table through Iceberg REST",
                "the table's catalog advertises no Iceberg REST endpoint",
            )
        warehouse, _, _ = self._identifier(table)
        properties = {
            "type": "rest",
            "uri": table.iceberg_rest_uri,
            "warehouse": warehouse,
            ACCESS_DELEGATION_HEADER: "vended-credentials",
        }
        token = self._fresh_token(table)
        if token:
            properties["token"] = token
        properties.update(self._properties)
        return properties

    def _catalog(self, table: ResolvedTable) -> Any:
        properties = self.rest_properties(table)
        key = (properties["uri"], properties["warehouse"])
        token = properties.get("token")
        with self._lock:
            cached = self._catalogs.get(key)
            if cached is not None and cached[0] == token:
                return cached[1]
        catalog = self._factory(f"deltaswamp-{key[1]}", properties)
        with self._lock:
            self._catalogs[key] = (token, catalog)
        return catalog

    def _load(self, table: ResolvedTable) -> Any:
        _, schema, name = self._identifier(table)
        catalog = self._catalog(table)
        exceptions = importlib.import_module("pyiceberg.exceptions")
        try:
            return catalog.load_table((schema, name))
        except exceptions.NoSuchTableError as exc:
            raise UnreachableTableError(
                "open the table through Iceberg REST",
                f"the Iceberg endpoint has no table {schema}.{name} in warehouse "
                f"{table.ref.catalog} ({exc})",
            ) from exc

    # ------------------------------------------------------------------- read

    @staticmethod
    def _row_filter(predicate: str | None) -> tuple[Any, str | None]:
        """PyIceberg's own parse of `predicate`, or no pushdown plus a residual.

        Returns ``(row_filter, residual)``: when PyIceberg cannot parse the SQL
        the scan reads everything and `residual` is applied exactly afterwards.
        """
        expressions = importlib.import_module("pyiceberg.expressions")
        if not predicate:
            return expressions.AlwaysTrue(), None
        parser = importlib.import_module("pyiceberg.expressions.parser")
        try:
            return parser.parse(predicate), None
        except Exception:
            return expressions.AlwaysTrue(), predicate

    @staticmethod
    def _snapshot_id(
        iceberg: Any, table: ResolvedTable, version: int | None, timestamp: str | None
    ) -> int | None:
        if version is not None:
            if not table.is_iceberg:
                # UniForm: `version` is a Delta version. Map it through the
                # snapshot summary rather than misreading it as a snapshot id.
                for snapshot in iceberg.snapshots():
                    summary = snapshot.summary
                    if summary is not None and summary.get(_UNIFORM_DELTA_VERSION) == str(version):
                        return int(snapshot.snapshot_id)
                raise UnreachableTableError(
                    f"read Delta version {version} through the table's Iceberg metadata",
                    "no Iceberg snapshot records that Delta version (UniForm converts only "
                    "some commits, and expired snapshots are gone)",
                    "time travel through the Delta engines",
                )
            if iceberg.snapshot_by_id(version) is None:
                raise UnreachableTableError(
                    f"read snapshot {version}",
                    "the Iceberg table has no snapshot with that id. Iceberg time travel takes "
                    "a snapshot id, not a sequential version number",
                    "history() lists the snapshot ids",
                )
            return version
        if timestamp is not None:
            snapshot = iceberg.snapshot_as_of_timestamp(_timestamp_ms(timestamp))
            if snapshot is None:
                raise UnreachableTableError(
                    f"read the table as of {timestamp}",
                    "the Iceberg table has no snapshot at or before that time",
                    "history() lists the snapshot timestamps",
                )
            return int(snapshot.snapshot_id)
        return None

    def scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """Read as a `pyarrow.RecordBatchReader`. `version` is an Iceberg snapshot id."""
        pa = importlib.import_module("pyarrow")
        iceberg = self._load(table)
        snapshot_id = self._snapshot_id(iceberg, table, version, timestamp)
        row_filter, residual = self._row_filter(predicate)
        # A residual may reference columns the caller did not select.
        selected = ("*",) if columns is None or residual else tuple(columns)
        reader = iceberg.scan(
            row_filter=row_filter, selected_fields=selected, snapshot_id=snapshot_id
        ).to_arrow_batch_reader()
        if residual is None:
            return reader

        out_schema = (
            reader.schema
            if columns is None
            else pa.schema([reader.schema.field(c) for c in columns])
        )

        def batches() -> Iterator[Any]:
            for batch in reader:
                rows = filter_arrow_exact(pa.Table.from_batches([batch]), residual)
                if columns is not None:
                    rows = rows.select(columns)
                yield from rows.to_batches()

        return pa.RecordBatchReader.from_batches(out_schema, batches())

    def history(self, table: ResolvedTable, *, limit: int | None = None) -> list[dict[str, Any]]:
        """The snapshot log, newest first. ``version`` is the snapshot id."""
        iceberg = self._load(table)
        out: list[dict[str, Any]] = []
        for entry in reversed(iceberg.history()):
            snapshot = iceberg.snapshot_by_id(entry.snapshot_id)
            summary = snapshot.summary if snapshot is not None else None
            out.append(
                {
                    "version": entry.snapshot_id,
                    "snapshot_id": entry.snapshot_id,
                    "timestamp": entry.timestamp_ms,
                    "parent_snapshot_id": snapshot.parent_snapshot_id if snapshot else None,
                    "sequence_number": snapshot.sequence_number if snapshot else None,
                    "operation": summary.operation.value if summary is not None else None,
                    "summary": dict(summary.additional_properties) if summary is not None else {},
                    "schema_id": snapshot.schema_id if snapshot else None,
                }
            )
            if limit is not None and len(out) >= limit:
                break
        return out

    def detail(self, table: ResolvedTable, *, version: int | None = None) -> dict[str, Any]:
        """Format version, snapshot, schema, partition spec, properties, location.

        ``version`` is the current (or requested) snapshot id, None for a table
        with no snapshots yet.
        """
        iceberg = self._load(table)
        metadata = iceberg.metadata
        snapshot_id = self._snapshot_id(iceberg, table, version, None)
        snapshot = (
            iceberg.snapshot_by_id(snapshot_id)
            if snapshot_id is not None
            else iceberg.current_snapshot()
        )
        schemas = iceberg.schemas()
        schema = (
            schemas.get(snapshot.schema_id, iceberg.schema())
            if snapshot is not None and snapshot.schema_id is not None
            else iceberg.schema()
        )
        spec = iceberg.spec()
        partition_columns = [
            schema.find_column_name(f.source_id) or str(f.source_id) for f in spec.fields
        ]
        summary = snapshot.summary if snapshot is not None else None
        return {
            "version": snapshot.snapshot_id if snapshot is not None else None,
            "format": "iceberg",
            "format_version": metadata.format_version,
            "table_uuid": str(metadata.table_uuid),
            "location": iceberg.location(),
            "metadata_location": iceberg.metadata_location,
            "current_snapshot_id": metadata.current_snapshot_id,
            "snapshot_timestamp": snapshot.timestamp_ms if snapshot is not None else None,
            "sequence_number": snapshot.sequence_number if snapshot is not None else None,
            "snapshot_summary": dict(summary.additional_properties) if summary else {},
            "schema": _as_json(schema),
            "arrow_schema": schema.as_arrow(),
            "partition_spec": _as_json(spec),
            "partition_columns": partition_columns,
            "sort_order": _as_json(iceberg.sort_order()),
            "properties": dict(metadata.properties),
            "is_uniform": not table.is_iceberg,
        }

    # ------------------------------------------------------------------ write

    @staticmethod
    def _refuse_unsupported(what: str, **options: Any) -> None:
        given = sorted(k for k, v in options.items() if v is not None)
        if given:
            raise UnreachableTableError(
                what,
                f"the Iceberg engine does not support {', '.join(given)}",
                "drop the option, or write through a Delta engine if the table is Delta",
            )

    def append(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        commit_metadata: dict[str, Any] | None = None,
        schema_mode: str | None = None,
        partition_by: list[str] | None = None,
        writer_properties: Any = None,
        txn: tuple[str, int] | None = None,
        **_: Any,
    ) -> None:
        """Append rows. `data` is anything Arrow-shaped (PyCapsule, pandas, ...)."""
        self._refuse_unsupported(
            "append to an Iceberg table",
            schema_mode=schema_mode,
            partition_by=partition_by,
            writer_properties=writer_properties,
            txn=txn,
        )
        self._load(table).append(
            _to_arrow_table(data), snapshot_properties=_snapshot_properties(commit_metadata)
        )

    def overwrite(
        self,
        table: ResolvedTable,
        data: Any,
        *,
        predicate: str | None = None,
        partition_overwrite: str = "static",
        commit_metadata: dict[str, Any] | None = None,
        schema_mode: str | None = None,
        writer_properties: Any = None,
        txn: tuple[str, int] | None = None,
        **_: Any,
    ) -> None:
        """Replace the whole table, the rows matching `predicate`, or (with
        ``partition_overwrite="dynamic"``) exactly the partitions in `data`."""
        self._refuse_unsupported(
            "overwrite an Iceberg table",
            schema_mode=schema_mode,
            writer_properties=writer_properties,
            txn=txn,
        )
        iceberg = self._load(table)
        rows = _to_arrow_table(data)
        properties = _snapshot_properties(commit_metadata)

        if partition_overwrite == "dynamic":
            if predicate:
                raise UnreachableTableError(
                    "overwrite an Iceberg table",
                    "a predicate and dynamic partition overwrite are mutually exclusive",
                )
            iceberg.dynamic_partition_overwrite(rows, snapshot_properties=properties)
            return

        if predicate:
            row_filter, residual = self._row_filter(predicate)
            if residual is not None:
                # An overwrite filter decides which rows are DELETED, so it
                # cannot be approximated the way a read filter can.
                raise UnreachableTableError(
                    "overwrite the rows matching a predicate",
                    f"PyIceberg cannot parse {predicate!r} as an Iceberg expression",
                    "use comparisons, IN, IS [NOT] NULL, AND/OR/NOT, or LIKE 'prefix%'",
                )
            iceberg.overwrite(rows, overwrite_filter=row_filter, snapshot_properties=properties)
            return

        iceberg.overwrite(rows, snapshot_properties=properties)

    # ------------------------------------------------------ distributed path

    def plan_scan(self, table: ResolvedTable, **kwargs: Any) -> list[Any]:
        raise NotImplementedError(
            "the Iceberg engine does not expose split planning; PyIceberg plans files "
            "per scan in-process"
        )

    def execute_scan(self, table: ResolvedTable, splits: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError("see plan_scan")
