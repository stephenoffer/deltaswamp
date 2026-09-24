"""Unity Catalog governance and table lifecycle, as plain data.

Everything a catalog returns from its governance methods is one of the frozen
dataclasses below, never an SDK object. Two reasons. Databricks and open-source
Unity Catalog describe the same things with different wire shapes (SDK enums
versus JSON, underscores versus spaces in privilege names), and a caller should
not care which server answered. And an SDK object keeps a reference to live
client state, which is exactly what must not travel into logs or task payloads.

The normalisers here work on the REST JSON shape. A Databricks SDK object is
first turned into that shape with its own ``as_dict()``, so one code path serves
both catalogs and the tests exercise the same parsing either way.

Nothing in this module talks to a server, except `Volume`, which is handed the
SDK's Files API by its catalog.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .errors import DeltaSwampError, InvalidReferenceError
from .identity import TableRef

__all__ = [
    "Column",
    "ColumnLineage",
    "ColumnLineageEntry",
    "ColumnMask",
    "Constraint",
    "FileEntry",
    "FunctionSummary",
    "Grant",
    "Lineage",
    "LineageEntry",
    "RowFilter",
    "StagingTable",
    "TableInfo",
    "TableSummary",
    "Volume",
    "VolumeSummary",
    "create_table_body",
    "delta_schema_to_columns",
    "normalize_privilege",
    "path_operation",
    "staging_storage_options",
]

_T = TypeVar("_T")


# --------------------------------------------------------------------- helpers


def _plain(obj: Any) -> dict[str, Any]:
    """REST-JSON dict for either an SDK dataclass or an already-plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        out = as_dict()
        return dict(out) if isinstance(out, Mapping) else {}
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _pick(d: Mapping[str, Any], *keys: str) -> Any:
    """First present key. OSS UC and the lineage API mix snake and camel case."""
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _str(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_privilege(privilege: Any) -> str:
    """``"external use schema"`` / ``Privilege.EXTERNAL_USE_SCHEMA`` -> ``"EXTERNAL_USE_SCHEMA"``.

    Databricks spells privileges with underscores and OSS Unity Catalog with
    spaces. This is the one public spelling; each catalog converts on the wire.
    """
    raw = str(getattr(privilege, "value", privilege)).strip()
    return "_".join(raw.replace("-", " ").replace("_", " ").upper().split())


# ------------------------------------------------------------------ table info


@dataclass(frozen=True, slots=True)
class ColumnMask:
    """A column mask: the function applied, and the extra columns it reads."""

    function_name: str
    using_column_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RowFilter:
    """A row filter: the function applied, and the columns passed to it."""

    function_name: str
    input_column_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    type_text: str | None = None
    type_name: str | None = None
    nullable: bool | None = None
    comment: str | None = None
    position: int | None = None
    #: Position among the partition columns; None when not a partition column.
    partition_index: int | None = None
    mask: ColumnMask | None = None


@dataclass(frozen=True, slots=True)
class Constraint:
    """An informational constraint. UC records these; nothing enforces them."""

    kind: str  # "PRIMARY_KEY" | "FOREIGN_KEY" | "NAMED"
    name: str
    columns: tuple[str, ...] = ()
    parent_table: str | None = None
    parent_columns: tuple[str, ...] = ()
    rely: bool | None = None


@dataclass(frozen=True, slots=True)
class TableInfo:
    """What the catalog says about one table, normalised across servers.

    Timestamps are epoch milliseconds, as the catalog reports them.
    """

    full_name: str
    catalog_name: str | None = None
    schema_name: str | None = None
    name: str | None = None
    table_id: str | None = None
    table_type: str | None = None
    data_source_format: str | None = None
    storage_location: str | None = None
    owner: str | None = None
    comment: str | None = None
    created_at: int | None = None
    created_by: str | None = None
    updated_at: int | None = None
    updated_by: str | None = None
    columns: tuple[Column, ...] = ()
    properties: dict[str, str] = field(default_factory=dict)
    #: Databricks' separately reported delta.* runtime settings.
    runtime_properties: dict[str, str] = field(default_factory=dict)
    row_filter: RowFilter | None = None
    constraints: tuple[Constraint, ...] = ()
    #: "ENABLE" / "DISABLE", with where it was inherited from.
    predictive_optimization: str | None = None
    predictive_optimization_inherited_from: str | None = None
    securable_kind: str | None = None
    capabilities: frozenset[str] = frozenset()
    #: The Lakeflow pipeline that maintains a materialized view or streaming table.
    pipeline_id: str | None = None
    #: True when the caller can see the table exists but not read its metadata.
    browse_only: bool | None = None
    view_definition: str | None = None
    metastore_id: str | None = None
    storage_credential_name: str | None = None
    access_point: str | None = None
    deleted_at: int | None = None

    @property
    def partition_columns(self) -> tuple[str, ...]:
        parts = [c for c in self.columns if c.partition_index is not None]
        return tuple(c.name for c in sorted(parts, key=lambda c: c.partition_index or 0))

    @property
    def column_masks(self) -> dict[str, ColumnMask]:
        return {c.name: c.mask for c in self.columns if c.mask is not None}

    @property
    def has_row_filter_or_mask(self) -> bool:
        """Credential vending refuses a table with either."""
        return self.row_filter is not None or bool(self.column_masks)

    @classmethod
    def from_api(cls, obj: Any) -> TableInfo:
        d = _plain(obj)
        catalog, schema, name = d.get("catalog_name"), d.get("schema_name"), d.get("name")
        full = d.get("full_name") or ".".join(p for p in (catalog, schema, name) if p)

        columns = tuple(_column(c) for c in (d.get("columns") or []))

        row_filter = None
        rf = d.get("row_filter")
        if rf and rf.get("function_name"):
            row_filter = RowFilter(
                function_name=rf["function_name"],
                input_column_names=tuple(rf.get("input_column_names") or ()),
            )

        runtime: dict[str, str] = {}
        kv = d.get("delta_runtime_properties_kvpairs")
        if isinstance(kv, Mapping):
            runtime = {
                str(k): str(v) for k, v in (kv.get("delta_runtime_properties") or {}).items()
            }

        po = d.get("effective_predictive_optimization_flag") or {}
        po_from = None
        if po.get("inherited_from_type") or po.get("inherited_from_name"):
            po_from = f"{po.get('inherited_from_type') or ''}:{po.get('inherited_from_name') or ''}"

        manifest = d.get("securable_kind_manifest") or {}
        return cls(
            full_name=str(full),
            catalog_name=catalog,
            schema_name=schema,
            name=name,
            table_id=d.get("table_id"),
            table_type=_str(d.get("table_type")),
            data_source_format=_str(d.get("data_source_format")),
            storage_location=d.get("storage_location"),
            owner=d.get("owner"),
            comment=d.get("comment"),
            created_at=_int(d.get("created_at")),
            created_by=d.get("created_by"),
            updated_at=_int(d.get("updated_at")),
            updated_by=d.get("updated_by"),
            columns=columns,
            properties={str(k): str(v) for k, v in (d.get("properties") or {}).items()},
            runtime_properties=runtime,
            row_filter=row_filter,
            constraints=tuple(
                c for c in (_constraint(x) for x in (d.get("table_constraints") or [])) if c
            ),
            predictive_optimization=_str(po.get("value")),
            predictive_optimization_inherited_from=po_from,
            securable_kind=_str(manifest.get("securable_kind") or d.get("securable_kind")),
            capabilities=frozenset(str(c) for c in (manifest.get("capabilities") or [])),
            pipeline_id=d.get("pipeline_id"),
            browse_only=d.get("browse_only"),
            view_definition=d.get("view_definition"),
            metastore_id=d.get("metastore_id"),
            storage_credential_name=d.get("storage_credential_name"),
            access_point=d.get("access_point"),
            deleted_at=_int(d.get("deleted_at")),
        )


def _column(c: Mapping[str, Any]) -> Column:
    mask = None
    m = c.get("mask")
    if m and m.get("function_name"):
        mask = ColumnMask(
            function_name=m["function_name"],
            using_column_names=tuple(m.get("using_column_names") or ()),
        )
    return Column(
        name=str(c.get("name")),
        type_text=c.get("type_text"),
        type_name=_str(c.get("type_name")),
        nullable=c.get("nullable"),
        comment=c.get("comment"),
        position=_int(c.get("position")),
        partition_index=_int(c.get("partition_index")),
        mask=mask,
    )


def _constraint(c: Mapping[str, Any]) -> Constraint | None:
    if pk := c.get("primary_key_constraint"):
        return Constraint(
            kind="PRIMARY_KEY",
            name=pk.get("name", ""),
            columns=tuple(pk.get("child_columns") or ()),
            rely=pk.get("rely"),
        )
    if fk := c.get("foreign_key_constraint"):
        return Constraint(
            kind="FOREIGN_KEY",
            name=fk.get("name", ""),
            columns=tuple(fk.get("child_columns") or ()),
            parent_table=fk.get("parent_table"),
            parent_columns=tuple(fk.get("parent_columns") or ()),
            rely=fk.get("rely"),
        )
    if named := c.get("named_table_constraint"):
        return Constraint(kind="NAMED", name=named.get("name", ""))
    return None


# ----------------------------------------------------------------- permissions


@dataclass(frozen=True, slots=True)
class Grant:
    """Privileges one principal holds on one securable.

    For an effective grant, `inherited_from_*` says which parent it came from;
    both are None when it was granted on the securable itself.
    """

    principal: str
    privileges: tuple[str, ...]
    inherited_from_type: str | None = None
    inherited_from_name: str | None = None

    @staticmethod
    def list_from_api(obj: Any) -> list[Grant]:
        """Parse a get/update (or get-effective) permissions response."""
        out: list[Grant] = []
        for assignment in _plain(obj).get("privilege_assignments") or []:
            principal = str(assignment.get("principal") or "")
            groups: dict[tuple[str | None, str | None], list[str]] = {}
            for p in assignment.get("privileges") or []:
                if isinstance(p, Mapping):  # effective: {privilege, inherited_from_*}
                    key = (_str(p.get("inherited_from_type")), p.get("inherited_from_name"))
                    groups.setdefault(key, []).append(normalize_privilege(p.get("privilege")))
                else:
                    groups.setdefault((None, None), []).append(normalize_privilege(p))
            for (from_type, from_name), privileges in groups.items():
                out.append(
                    Grant(
                        principal=principal,
                        privileges=tuple(sorted(privileges)),
                        inherited_from_type=from_type,
                        inherited_from_name=from_name,
                    )
                )
        return out


# --------------------------------------------------------------------- lineage


@dataclass(frozen=True, slots=True)
class LineageEntry:
    """One neighbour in the lineage graph: a table, notebook, job, query, file..."""

    kind: str
    #: Full name for a table, path for a file, the id for everything else.
    name: str
    table_type: str | None = None
    lineage_timestamp: str | None = None
    workspace_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class Lineage:
    table: str
    upstream: tuple[LineageEntry, ...] = ()
    downstream: tuple[LineageEntry, ...] = ()

    @property
    def upstream_tables(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.upstream if e.kind == "table")

    @property
    def downstream_tables(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.downstream if e.kind == "table")

    @classmethod
    def from_api(cls, table: str, body: Mapping[str, Any], direction: str = "both") -> Lineage:
        _check_direction(direction)
        up = _lineage_entries(body.get("upstreams") or [])
        down = _lineage_entries(body.get("downstreams") or [])
        return cls(
            table=table,
            upstream=up if direction in ("both", "upstream") else (),
            downstream=down if direction in ("both", "downstream") else (),
        )


_LINEAGE_ID_KEYS = (
    "notebook_id",
    "job_id",
    "query_id",
    "dashboard_id",
    "pipeline_id",
    "model_name",
    "path",
    "name",
)


def _lineage_kind(key: str) -> str:
    """``tableInfo`` / ``notebook_infos`` / ``dashboardV3Infos`` -> table / notebook / dashboard."""
    snake = "".join(f"_{ch.lower()}" if ch.isupper() else ch for ch in key).lstrip("_")
    for suffix in ("_infos", "_info"):
        if snake.endswith(suffix):
            snake = snake[: -len(suffix)]
            break
    return snake.split("_v")[0] if "_v" in snake and snake.rsplit("_v", 1)[1].isdigit() else snake


def _lineage_entries(items: Iterable[Mapping[str, Any]]) -> tuple[LineageEntry, ...]:
    out: list[LineageEntry] = []
    for item in items:
        for key, value in item.items():
            kind = _lineage_kind(key)
            for info in value if isinstance(value, list) else [value]:
                if not isinstance(info, Mapping):
                    continue
                if kind == "table":
                    name = ".".join(
                        str(p)
                        for p in (
                            info.get("catalog_name"),
                            info.get("schema_name"),
                            info.get("name"),
                        )
                        if p
                    )
                else:
                    ident = _pick(info, *_LINEAGE_ID_KEYS)
                    name = "" if ident is None else str(ident)
                out.append(
                    LineageEntry(
                        kind=kind,
                        name=name,
                        table_type=_str(info.get("table_type")),
                        lineage_timestamp=_str(info.get("lineage_timestamp")),
                        workspace_id=_int(info.get("workspace_id")),
                        details=dict(info),
                    )
                )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class ColumnLineageEntry:
    table: str
    column: str
    table_type: str | None = None
    lineage_timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnLineage:
    table: str
    column: str
    upstream: tuple[ColumnLineageEntry, ...] = ()
    downstream: tuple[ColumnLineageEntry, ...] = ()

    @classmethod
    def from_api(
        cls, table: str, column: str, body: Mapping[str, Any], direction: str = "both"
    ) -> ColumnLineage:
        _check_direction(direction)

        def entries(items: Iterable[Mapping[str, Any]]) -> tuple[ColumnLineageEntry, ...]:
            return tuple(
                ColumnLineageEntry(
                    table=".".join(
                        str(p)
                        for p in (c.get("catalog_name"), c.get("schema_name"), c.get("table_name"))
                        if p
                    ),
                    column=str(c.get("name") or ""),
                    table_type=_str(c.get("table_type")),
                    lineage_timestamp=_str(c.get("lineage_timestamp")),
                )
                for c in items
            )

        up = entries(_pick(body, "upstream_cols", "upstreamCols") or [])
        down = entries(_pick(body, "downstream_cols", "downstreamCols") or [])
        return cls(
            table=table,
            column=column,
            upstream=up if direction in ("both", "upstream") else (),
            downstream=down if direction in ("both", "downstream") else (),
        )


def _check_direction(direction: str) -> None:
    if direction not in ("both", "upstream", "downstream"):
        raise InvalidReferenceError(
            f"lineage direction must be 'upstream', 'downstream' or 'both', not {direction!r}"
        )


# ------------------------------------------------------------------ namespaces


@dataclass(frozen=True, slots=True)
class TableSummary:
    full_name: str
    table_type: str | None = None
    securable_kind: str | None = None
    #: The capability manifest flags, when the catalog returned one.
    capabilities: frozenset[str] | None = None

    @classmethod
    def from_api(cls, obj: Any) -> TableSummary:
        d = _plain(obj)
        manifest = d.get("securable_kind_manifest")
        caps = None
        if isinstance(manifest, Mapping) and manifest.get("capabilities") is not None:
            caps = frozenset(str(c) for c in manifest["capabilities"])
        full = d.get("full_name") or ".".join(
            str(p) for p in (d.get("catalog_name"), d.get("schema_name"), d.get("name")) if p
        )
        return cls(
            full_name=str(full),
            table_type=_str(d.get("table_type")),
            securable_kind=_str((manifest or {}).get("securable_kind")),
            capabilities=caps,
        )


@dataclass(frozen=True, slots=True)
class FunctionSummary:
    full_name: str
    name: str | None = None
    data_type: str | None = None
    full_data_type: str | None = None
    routine_body: str | None = None
    comment: str | None = None
    owner: str | None = None
    function_id: str | None = None

    @classmethod
    def from_api(cls, obj: Any) -> FunctionSummary:
        d = _plain(obj)
        full = d.get("full_name") or ".".join(
            str(p) for p in (d.get("catalog_name"), d.get("schema_name"), d.get("name")) if p
        )
        return cls(
            full_name=str(full),
            name=d.get("name"),
            data_type=_str(d.get("data_type")),
            full_data_type=d.get("full_data_type"),
            routine_body=_str(d.get("routine_body")),
            comment=d.get("comment"),
            owner=d.get("owner"),
            function_id=d.get("function_id"),
        )


@dataclass(frozen=True, slots=True)
class VolumeSummary:
    full_name: str
    name: str | None = None
    volume_type: str | None = None
    storage_location: str | None = None
    owner: str | None = None
    comment: str | None = None
    volume_id: str | None = None

    @classmethod
    def from_api(cls, obj: Any) -> VolumeSummary:
        d = _plain(obj)
        full = d.get("full_name") or ".".join(
            str(p) for p in (d.get("catalog_name"), d.get("schema_name"), d.get("name")) if p
        )
        return cls(
            full_name=str(full),
            name=d.get("name"),
            volume_type=_str(d.get("volume_type")),
            storage_location=d.get("storage_location"),
            owner=d.get("owner"),
            comment=d.get("comment"),
            volume_id=d.get("volume_id"),
        )


# --------------------------------------------------------------------- volumes


@dataclass(frozen=True, slots=True)
class FileEntry:
    path: str
    name: str
    is_directory: bool
    size: int | None = None
    #: Epoch milliseconds.
    last_modified: int | None = None


class Volume:
    """Files in one Unity Catalog volume, through the Databricks Files API.

    Paths are relative to the volume root. ``..`` is refused rather than
    normalised, so a path can never climb out of the volume it was opened on.
    `on_error` turns an SDK error into one that names the missing privilege.
    """

    def __init__(
        self,
        full_name: str,
        files_api: Any,
        *,
        on_error: Callable[[Exception, str], Exception] | None = None,
    ) -> None:
        parts = full_name.split(".")
        if len(parts) != 3 or not all(parts):
            raise InvalidReferenceError(
                f"a volume is named catalog.schema.volume, not {full_name!r}"
            )
        self.full_name = full_name
        self.root = "/Volumes/" + "/".join(parts)
        self._files = files_api
        self._on_error = on_error

    def __repr__(self) -> str:
        return f"Volume({self.full_name!r})"

    def _run(self, action: str, fn: Callable[[], _T]) -> _T:
        try:
            return fn()
        except DeltaSwampError:
            raise
        except Exception as exc:
            if self._on_error is None:
                raise
            raise self._on_error(exc, action) from exc

    def path(self, path: str = "") -> str:
        """The absolute ``/Volumes/...`` path for a volume-relative one."""
        parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
        if ".." in parts:
            raise InvalidReferenceError(f"{path!r} climbs out of volume {self.full_name}")
        return "/".join([self.root, *parts])

    def list(self, path: str = "") -> list[FileEntry]:
        directory = self.path(path).rstrip("/") + "/"
        entries = self._run(
            "list volume files",
            lambda: list(self._files.list_directory_contents(directory_path=directory)),
        )
        out: list[FileEntry] = []
        for entry in entries:
            d = _plain(entry)
            full = str(d.get("path") or "")
            out.append(
                FileEntry(
                    path=full[len(self.root) :].lstrip("/") if full.startswith(self.root) else full,
                    name=str(d.get("name") or full.rstrip("/").rsplit("/", 1)[-1]),
                    is_directory=bool(d.get("is_directory")),
                    size=_int(d.get("file_size")),
                    last_modified=_int(d.get("last_modified")),
                )
            )
        return out

    def read(self, path: str) -> bytes:
        target = self.path(path)

        def download() -> bytes:
            contents = getattr(self._files.download(file_path=target), "contents", None)
            return b"" if contents is None else bytes(contents.read())

        return self._run("read a volume file", download)

    def write(self, path: str, data: bytes, *, overwrite: bool = False) -> None:
        target = self.path(path)
        self._run(
            "write a volume file",
            lambda: self._files.upload(
                file_path=target, contents=io.BytesIO(data), overwrite=overwrite
            ),
        )

    def delete(self, path: str) -> None:
        target = self.path(path)
        self._run("delete a volume file", lambda: self._files.delete(file_path=target))

    def mkdir(self, path: str) -> None:
        target = self.path(path)
        self._run(
            "create a volume directory", lambda: self._files.create_directory(directory_path=target)
        )


# -------------------------------------------------------------------- creation


@dataclass(frozen=True, slots=True)
class StagingTable:
    """A managed table the catalog has allocated but not yet registered.

    Write version 0 at `location` using `storage_options`, with the log's
    ``metaData.id`` set to `table_id`, the `required_protocol` features, and
    every `required_properties` entry in ``metaData.configuration`` (a None
    value means "any value, but present"; ``delta.feature.*`` entries are met
    by the protocol, not written as properties). Then finalise through the
    catalog. `storage_options` holds live secrets and
    is kept out of the repr.
    """

    full_name: str
    table_id: str
    table_type: str
    location: str
    storage_options: dict[str, str] = field(default_factory=dict, repr=False)
    #: Epoch seconds, from the credential's ``expiration-time-ms``; None if unstated.
    expires_at: float | None = None
    required_protocol: dict[str, Any] = field(default_factory=dict)
    suggested_protocol: dict[str, Any] | None = None
    required_properties: dict[str, str | None] = field(default_factory=dict)
    suggested_properties: dict[str, str | None] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"StagingTable(full_name={self.full_name!r}, table_id={self.table_id!r}, "
            f"location={self.location!r}, storage_options=keys{sorted(self.storage_options)})"
        )

    @classmethod
    def from_api(cls, full_name: str, body: Mapping[str, Any]) -> StagingTable:
        location = str(_pick(body, "location", "storage-location", "storage_location") or "")
        table_id = _pick(body, "table-id", "table_id")
        if not table_id or not location:
            raise DeltaSwampError(
                f"the staging-table response for {full_name} carried no table id or location "
                f"(fields: {sorted(body)})"
            )
        credentials = _pick(body, "storage-credentials", "storage_credentials") or []
        options, expires_at = staging_storage_options(location, credentials)
        return cls(
            full_name=full_name,
            table_id=str(table_id),
            table_type=str(_pick(body, "table-type", "table_type") or "MANAGED"),
            location=location,
            storage_options=options,
            expires_at=expires_at,
            required_protocol=dict(_pick(body, "required-protocol", "required_protocol") or {}),
            suggested_protocol=_pick(body, "suggested-protocol", "suggested_protocol"),
            required_properties=dict(
                _pick(body, "required-properties", "required_properties") or {}
            ),
            suggested_properties=dict(
                _pick(body, "suggested-properties", "suggested_properties") or {}
            ),
        )


_STAGING_KEYS = {
    "s3.access-key-id": "aws_access_key_id",
    "s3.secret-access-key": "aws_secret_access_key",
    "s3.session-token": "aws_session_token",
    "s3.region": "aws_region",
    "s3.endpoint": "aws_endpoint",
    "azure.sas-token": "azure_storage_sas_key",
    "gcs.oauth-token": "google_bearer_token",
}


def staging_storage_options(
    location: str, credentials: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, str], float | None]:
    """Render UC Delta API ``storage-credentials`` as object_store options.

    Picks the credential whose prefix covers `location` most specifically,
    preferring READ_WRITE (version 0 has to be written). Unknown config keys
    are dropped rather than passed through, since object_store rejects keys it
    does not know. Azure gets an explicit endpoint, never account inference.
    """
    creds = list(credentials)
    if not creds:
        return {}, None

    def score(c: Mapping[str, Any]) -> tuple[int, int, int]:
        prefix = str(c.get("prefix") or "")
        covers = location.rstrip("/").startswith(prefix.rstrip("/"))
        return (int(covers), int(c.get("operation") == "READ_WRITE"), len(prefix))

    chosen = max(creds, key=score)
    config = chosen.get("config") or {}
    options = {_STAGING_KEYS[k]: str(v) for k, v in config.items() if k in _STAGING_KEYS}
    if "azure_storage_sas_key" in options:
        from .credentials.databricks import azure_endpoint_for

        endpoint = azure_endpoint_for(str(chosen.get("prefix") or location))
        if endpoint is None:
            raise DeltaSwampError(
                f"cannot derive an Azure endpoint from {location!r}; account-name inference "
                "breaks Azurite, private-link DNS and sovereign clouds"
            )
        options["azure_endpoint"] = endpoint
    expiry = _pick(chosen, "expiration-time-ms", "expiration_time_ms")
    return options, (float(expiry) / 1000.0 if expiry else None)


# ------------------------------------------------------------ schema -> columns

#: Delta primitive -> (UC type_name, Spark catalog type string).
_PRIMITIVES = {
    "string": ("STRING", "string"),
    "long": ("LONG", "bigint"),
    "integer": ("INT", "int"),
    "short": ("SHORT", "smallint"),
    "byte": ("BYTE", "tinyint"),
    "float": ("FLOAT", "float"),
    "double": ("DOUBLE", "double"),
    "boolean": ("BOOLEAN", "boolean"),
    "binary": ("BINARY", "binary"),
    "date": ("DATE", "date"),
    "timestamp": ("TIMESTAMP", "timestamp"),
    "timestamp_ntz": ("TIMESTAMP_NTZ", "timestamp_ntz"),
    "variant": ("VARIANT", "variant"),
}


def _type_text(dtype: Any) -> str:
    if isinstance(dtype, str):
        if dtype in _PRIMITIVES:
            return _PRIMITIVES[dtype][1]
        if dtype.startswith("decimal"):
            return dtype.replace(" ", "")
        raise DeltaSwampError(f"unrecognised Delta type {dtype!r} in the table schema")
    kind = dtype.get("type")
    if kind == "struct":
        inner = ",".join(f"{f['name']}:{_type_text(f['type'])}" for f in dtype.get("fields", []))
        return f"struct<{inner}>"
    if kind == "array":
        return f"array<{_type_text(dtype['elementType'])}>"
    if kind == "map":
        return f"map<{_type_text(dtype['keyType'])},{_type_text(dtype['valueType'])}>"
    raise DeltaSwampError(f"unrecognised Delta type {dtype!r} in the table schema")


def delta_schema_to_columns(
    schema: str | Mapping[str, Any], partition_columns: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Delta ``schemaString`` -> Unity Catalog ``columns`` (REST JSON dicts).

    The create-table API does not validate columns, and Databricks Runtime
    cannot read a table whose spec is not Spark-compatible, so this emits the
    Spark catalog spelling (``bigint``, ``struct<a:int>``) that Spark would.
    """
    parsed: Mapping[str, Any] = json.loads(schema) if isinstance(schema, str) else schema
    if parsed.get("type") != "struct":
        raise DeltaSwampError("a Delta table schema must be a struct type")
    partitions = list(partition_columns)
    names = [f["name"] for f in parsed.get("fields", [])]
    missing = [p for p in partitions if p not in names]
    if missing:
        raise DeltaSwampError(f"partition columns {missing} are not in the schema {names}")

    out: list[dict[str, Any]] = []
    for position, f in enumerate(parsed.get("fields", [])):
        dtype = f["type"]
        column: dict[str, Any] = {
            "name": f["name"],
            "type_text": _type_text(dtype),
            "type_json": json.dumps(f, separators=(",", ":")),
            "position": position,
            "nullable": bool(f.get("nullable", True)),
        }
        if isinstance(dtype, str) and dtype.startswith("decimal"):
            precision, _, scale = dtype[dtype.index("(") + 1 : -1].partition(",")
            column.update(
                type_name="DECIMAL", type_precision=int(precision), type_scale=int(scale or 0)
            )
        elif isinstance(dtype, str):
            column["type_name"] = _PRIMITIVES[dtype][0]
        else:
            column["type_name"] = str(dtype["type"]).upper()
        comment = (f.get("metadata") or {}).get("comment")
        if comment:
            column["comment"] = str(comment)
        if f["name"] in partitions:
            column["partition_index"] = partitions.index(f["name"])
        out.append(column)
    return out


# ------------------------------------------------------------ request helpers

_PATH_OPERATIONS = {
    "PATH_READ": "PATH_READ",
    "READ": "PATH_READ",
    "PATH_READ_WRITE": "PATH_READ_WRITE",
    "READ_WRITE": "PATH_READ_WRITE",
    "PATH_CREATE_TABLE": "PATH_CREATE_TABLE",
    "CREATE_TABLE": "PATH_CREATE_TABLE",
}


def path_operation(operation: Any) -> str:
    """Normalise to the UC `PathOperation` spelling."""
    key = str(getattr(operation, "value", operation)).upper()
    try:
        return _PATH_OPERATIONS[key]
    except KeyError:
        raise InvalidReferenceError(
            f"path credential operation must be PATH_READ, PATH_READ_WRITE or "
            f"PATH_CREATE_TABLE, not {operation!r}"
        ) from None


def create_table_body(ref: TableRef, request_body: Mapping[str, Any]) -> dict[str, Any]:
    """Check a CreateTableRequest names the table it is being sent for."""
    body = dict(request_body)
    named = body.setdefault("name", ref.table)
    if named != ref.table:
        raise InvalidReferenceError(
            f"the create-table request names {named!r} but is being sent for {ref.table!r}"
        )
    return body
