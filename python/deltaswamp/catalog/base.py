"""The catalog abstraction: name -> everything an engine needs to open a table.

`ResolvedTable` is deliberately fat. The router must decide an engine *before*
touching the Delta log, because several decisions cannot be made afterwards:

* A shallow clone must be detected from catalog metadata first. Its `add` actions
  carry absolute paths into the source table, and kernel resolves those to
  absolute URLs before a connector sees them -- so by log-read time you can no
  longer tell borrowed files from owned ones (kernel#2411).
* Row filters and column masks make UC credential vending refuse outright. You
  learn this from the capability manifest, not from a failed read.
* A `catalogManaged` table needs its commit tail and ratified version supplied at
  snapshot-construction time; there is no retrofitting it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..capability import TableFeature, feature_from_wire
from ..credentials import CredentialProvider
from ..identity import TableRef

if TYPE_CHECKING:  # pragma: no cover
    from ..credentials import Credentials
    from ..governance import (
        ColumnLineage,
        FunctionSummary,
        Grant,
        Lineage,
        StagingTable,
        TableInfo,
        TableSummary,
        Volume,
        VolumeSummary,
    )

__all__ = [
    "Catalog",
    "GovernedCatalog",
    "LogTailEntry",
    "NamespaceCatalog",
    "ResolvedTable",
    "TableLifecycleCatalog",
    "TableType",
    "parse_commit_tail",
]


class TableType(StrEnum):
    """UC `table_type`. Nine values; there is no ONLINE member.

    Online/synced tables surface through `securable_kind` instead, and are not
    file-addressable at all.
    """

    MANAGED = "MANAGED"
    EXTERNAL = "EXTERNAL"
    VIEW = "VIEW"
    MATERIALIZED_VIEW = "MATERIALIZED_VIEW"
    STREAMING_TABLE = "STREAMING_TABLE"
    MANAGED_SHALLOW_CLONE = "MANAGED_SHALLOW_CLONE"
    EXTERNAL_SHALLOW_CLONE = "EXTERNAL_SHALLOW_CLONE"
    FOREIGN = "FOREIGN"
    METRIC_VIEW = "METRIC_VIEW"


@dataclass(frozen=True, slots=True)
class LogTailEntry:
    """One catalog-ratified but possibly unpublished commit.

    Maps to a kernel `LogPath`. These live at
    ``_delta_log/_staged_commits/<20-digit version>.<uuid>.json`` until published.
    """

    version: int
    path: str
    size: int
    timestamp: int | None = None


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    """The first present key among `names`.

    The UC Delta API spells its JSON in kebab-case (`latest-table-version`,
    `file-name`), which is what a real Databricks metastore returns. Some
    servers and older drafts use snake_case or camelCase for the same fields, so
    every spelling is accepted rather than guessed at.
    """
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def parse_commit_tail(
    body: Mapping[str, Any], fallback_location: str | None = None
) -> tuple[tuple[LogTailEntry, ...], int | None, str | None]:
    """Read a `/delta/v1` table response into (log tail, max version, location).

    Shared by both Unity Catalog backends so one spelling fix covers both.
    Reading `latest-table-version` under the wrong spelling leaves the maximum
    ratified version unset, and the kernel then refuses the table outright with
    "Max catalog version is required when loading a catalog-managed table" --
    which is every catalog-managed table, the one thing no other Python library
    can open.
    """
    commits = [c for c in (body.get("commits") or []) if isinstance(c, Mapping)]
    for c in commits:
        missing = [
            what
            for what, names in (
                ("version", ("version", "commit-version", "commitVersion")),
                # Without a file name the kernel is handed an empty path and
                # fails far from the cause.
                ("file name", ("file-name", "file_name", "fileName")),
            )
            if not str(_first(c, *names) or "").strip() and _first(c, *names) != 0
        ]
        if missing:
            from ..errors import CorruptTableError

            raise CorruptTableError(
                f"a catalog commit carries no {' or '.join(missing)}: {dict(c)!r:.300}"
            )
    # Sorted and de-duplicated: kernel wants the tail ascending and contiguous,
    # and the API does not promise an order.
    commits = list(
        {int(_first(c, "version", "commit-version", "commitVersion")): c for c in commits}.values()
    )
    commits.sort(key=lambda c: int(_first(c, "version", "commit-version", "commitVersion")))
    entries = tuple(
        LogTailEntry(
            version=int(_first(c, "version", "commit-version", "commitVersion")),
            path=_first(c, "file-name", "file_name", "fileName") or "",
            size=int(_first(c, "file-size", "file_size", "fileSize") or 0),
            timestamp=(
                int(ts)
                if (
                    ts := _first(
                        c,
                        "file-modification-timestamp",
                        "timestamp",
                        "file_modification_timestamp",
                    )
                )
                is not None
                else None
            ),
        )
        for c in commits
    )
    latest = _first(body, "latest-table-version", "latest_table_version", "latestTableVersion")
    if latest is None and entries:
        # Commits are published oldest-first, so the newest unpublished one is
        # the latest ratified version. Leaving it unset made the kernel refuse
        # the table ("Max catalog version is required").
        latest = entries[-1].version
    metadata = body.get("metadata") or {}
    location = (
        body.get("location")
        or (metadata.get("location") if metadata else None)
        or fallback_location
    )
    return entries, (int(latest) if latest is not None else None), location


@dataclass(frozen=True, slots=True)
class ResolvedTable:
    """Everything needed to open a table, gathered before the log is read."""

    ref: TableRef
    location: str | None
    table_type: TableType | None = None
    data_source_format: str | None = None
    securable_kind: str | None = None
    table_id: str | None = None

    # Protocol state. Unrecognised names are kept verbatim so we can report them
    # precisely instead of dropping them.
    min_reader_version: int | None = None
    min_writer_version: int | None = None
    reader_features: frozenset[str] = field(default_factory=frozenset)
    writer_features: frozenset[str] = field(default_factory=frozenset)
    properties: dict[str, str] = field(default_factory=dict)
    # Empty for an unpartitioned table, or before the log has been read.
    partition_columns: tuple[str, ...] = ()

    # Set when reading the log failed on every engine. The router then refuses
    # direct-storage operations with this as the reason, instead of routing on
    # an empty feature list and claiming success.
    open_error: str | None = None

    #: The catalog's own table UUID. Checked against the log's Metadata.id,
    #: because a table dropped and re-created under the same name gets a new
    #: one, and a stale id silently reads the wrong table.
    table_uuid: str | None = None
    #: Optimistic-concurrency token from the catalog, where it supplies one.
    etag: str | None = None

    # From ListTables(include_manifest_capabilities=true). None means "not asked".
    # This is the authoritative pre-flight -- a row filter silently removes an
    # ordinary managed Delta table from the eligible set, and table_type will
    # not tell you.
    external_read_supported: bool | None = None
    external_write_supported: bool | None = None

    # Catalog-managed state, supplied to SnapshotBuilder.
    log_tail: tuple[LogTailEntry, ...] = ()
    max_catalog_version: int | None = None

    credential_provider: CredentialProvider | None = None

    #: The catalog's Iceberg REST endpoint, when it has one. Set for Iceberg
    #: tables and UniForm tables, and consumed by the Iceberg engine.
    iceberg_rest_uri: str | None = None
    #: A Delta Sharing profile (a path, URL or JSON document). Set only for
    #: tables reached through a share, whose files are served as presigned URLs
    #: and are not addressable by a storage location at all.
    sharing_profile: str | None = field(default=None, repr=False)

    @property
    def features(self) -> frozenset[str]:
        return self.reader_features | self.writer_features

    @property
    def known_features(self) -> frozenset[TableFeature]:
        out = set()
        for name in self.features:
            f = feature_from_wire(name)
            if f is not None:
                out.add(f)
        return frozenset(out)

    @property
    def unknown_features(self) -> frozenset[str]:
        """Feature names this release does not recognise.

        Tolerated on reads when writer-only; they block writes.
        """
        return frozenset(n for n in self.features if feature_from_wire(n) is None)

    @property
    def is_catalog_managed(self) -> bool:
        return (
            TableFeature.CATALOG_MANAGED.value in self.features
            or TableFeature.CATALOG_OWNED_PREVIEW.value in self.features
            # Databricks signals it as a table property too (either spelling).
            or any(
                str(self.properties.get(f"delta.feature.{f.value}", "")).lower()
                in ("supported", "enabled")
                for f in (TableFeature.CATALOG_MANAGED, TableFeature.CATALOG_OWNED_PREVIEW)
            )
        )

    @property
    def is_shallow_clone(self) -> bool:
        """Shallow clones borrow the source's data files via absolute paths.

        Detected from catalog metadata because it is undetectable later.
        """
        if self.table_type in (
            TableType.MANAGED_SHALLOW_CLONE,
            TableType.EXTERNAL_SHALLOW_CLONE,
        ):
            return True
        return bool(self.securable_kind and "SHALLOW_CLONE" in self.securable_kind)

    @property
    def is_delta(self) -> bool:
        """True when the catalog says this holds Delta data.

        `None` means the catalog did not say, which we treat as Delta because
        every path-based table resolves that way.
        """
        fmt = (self.data_source_format or "").upper()
        return fmt in ("", "DELTA", "DELTA_UNIFORM_ICEBERG", "DELTA_UNIFORM_HUDI")

    @property
    def is_shared(self) -> bool:
        """Reached through Delta Sharing rather than by storage location."""
        return self.sharing_profile is not None

    @property
    def is_iceberg(self) -> bool:
        return (self.data_source_format or "").upper() == "ICEBERG"

    @property
    def is_view_like(self) -> bool:
        """Views, MVs, metric views and streaming tables have no writable file surface."""
        return self.table_type in (
            TableType.VIEW,
            TableType.MATERIALIZED_VIEW,
            TableType.METRIC_VIEW,
            TableType.STREAMING_TABLE,
        )

    @property
    def has_iceberg_compat(self) -> bool:
        """Iceberg-reads / UniForm tables.

        Writing to one requires ``MSCK REPAIR TABLE ... SYNC METADATA`` afterwards,
        which only Databricks can run -- so an external write silently diverges
        the Iceberg view. We refuse rather than diverge.
        """
        return (
            any(
                f in self.features
                for f in (
                    TableFeature.ICEBERG_COMPAT_V1.value,
                    TableFeature.ICEBERG_COMPAT_V2.value,
                    TableFeature.ICEBERG_COMPAT_V3.value,
                )
            )
            or "iceberg"
            in str(self.properties.get("delta.universalFormat.enabledFormats", "")).lower()
            # The catalog reports the enabling properties before the log has
            # been read, when the feature lists are still empty.
            or any(
                str(self.properties.get(f"delta.enableIcebergCompatV{n}", "")).lower() == "true"
                for n in (1, 2, 3)
            )
        )

    @property
    def compatibility_mode_location(self) -> str | None:
        """Path of the read-only v1-metadata copy, when Compatibility Mode is on.

        This is the friendliest surface in the entire taxonomy: a plain
        v1-protocol Delta table at a known path, readable by anything.
        """
        fmts = self.properties.get("delta.universalFormat.enabledFormats", "")
        if "compatibility" not in fmts:
            return None
        return self.properties.get("delta.universalFormat.compatibility.location")


@runtime_checkable
class Catalog(Protocol):
    """Resolves references to tables. Registered via the
    ``deltaswamp.catalogs`` entry point group."""

    name: str

    def resolve(self, ref: TableRef) -> ResolvedTable:
        """Resolve one reference, including its capability manifest."""
        ...

    def list_catalogs(self) -> list[str]:
        """Catalog names visible to this principal."""
        ...

    def list_schemas(self, catalog: str) -> list[str]:
        """Schema names in `catalog`."""
        ...

    def drop_table(self, ref: TableRef) -> None:
        """Remove a table from the catalog."""
        ...

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        """List a schema's tables.

        Implementations should request capability manifests here and cache them:
        one call per schema is both cheaper and more reliable than N per-table
        lookups, and per-table credential vending has no batch endpoint.
        """
        ...


# ---------------------------------------------------------------------------
# Optional catalog capabilities. A catalog implements these in addition to
# `Catalog`; check with isinstance. A catalog may satisfy a protocol and still
# raise NotImplementedError from a method its server lacks -- it then lists the
# method name in an `unsupported_operations` frozenset attribute, so a caller
# can report support without making the call.
# ---------------------------------------------------------------------------


@runtime_checkable
class GovernedCatalog(Protocol):
    """Unity Catalog governance: metadata, permissions, tags, lineage, constraints.

    `target` in the permission methods is a `TableRef` for a table, or a dotted
    name for another securable, whose kind `securable_type` names ("TABLE",
    "SCHEMA", "CATALOG", "VOLUME", "FUNCTION"). Privileges use the underscore
    spelling, e.g. "EXTERNAL_USE_SCHEMA".
    """

    def table_info(self, ref: TableRef) -> TableInfo: ...

    def grants(
        self,
        target: TableRef | str,
        principal: str | None = None,
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]: ...

    def effective_grants(
        self,
        target: TableRef | str,
        principal: str | None = None,
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]: ...

    def grant(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]: ...

    def revoke(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]: ...

    def tags(self, ref: TableRef, column: str | None = None) -> dict[str, str]: ...

    def set_tags(
        self, ref: TableRef, tags: Mapping[str, str], column: str | None = None
    ) -> None: ...

    def unset_tags(self, ref: TableRef, keys: Iterable[str], column: str | None = None) -> None: ...

    def set_owner(self, ref: TableRef, principal: str) -> None: ...

    def lineage(self, ref: TableRef, direction: str = "both") -> Lineage: ...

    def column_lineage(
        self, ref: TableRef, column: str, direction: str = "both"
    ) -> ColumnLineage: ...

    def add_primary_key(
        self, ref: TableRef, name: str, columns: Iterable[str], *, rely: bool = False
    ) -> None: ...

    def add_foreign_key(
        self,
        ref: TableRef,
        name: str,
        columns: Iterable[str],
        parent_ref: TableRef,
        parent_columns: Iterable[str],
        *,
        rely: bool = False,
    ) -> None: ...

    def drop_table_constraint(self, ref: TableRef, name: str, *, cascade: bool = False) -> None: ...


@runtime_checkable
class NamespaceCatalog(Protocol):
    """Creating and dropping catalogs, schemas and volumes, and searching them."""

    def create_catalog(
        self, name: str, comment: str | None = None, storage_root: str | None = None
    ) -> None: ...

    def drop_catalog(self, name: str, force: bool = False) -> None: ...

    def create_schema(
        self,
        catalog: str,
        name: str,
        comment: str | None = None,
        storage_root: str | None = None,
    ) -> None: ...

    def drop_schema(self, catalog: str, name: str, force: bool = False) -> None: ...

    def table_exists(self, ref: TableRef) -> bool: ...

    def search_tables(
        self,
        catalog: str,
        schema_pattern: str | None = None,
        table_pattern: str | None = None,
    ) -> list[TableSummary]: ...

    def list_functions(self, catalog: str, schema: str) -> list[FunctionSummary]: ...

    def list_volumes(self, catalog: str, schema: str) -> list[VolumeSummary]: ...

    def create_volume(
        self,
        catalog: str,
        schema: str,
        name: str,
        volume_type: str = "MANAGED",
        storage_location: str | None = None,
        comment: str | None = None,
    ) -> VolumeSummary: ...

    def drop_volume(self, catalog: str, schema: str, name: str) -> None: ...

    def volume(self, ref: TableRef | str) -> Volume: ...


@runtime_checkable
class TableLifecycleCatalog(Protocol):
    """Bringing a table into existence in the catalog.

    External: write the Delta log with `path_credentials(location,
    "PATH_CREATE_TABLE")`, then `register_table`. Managed (catalog-managed):
    `create_staging_table`, write version 0 at its location with its storage
    options and required properties, then `finalize_managed_table` with the
    UC Delta API CreateTableRequest body.
    """

    def register_table(
        self,
        ref: TableRef,
        location: str,
        *,
        columns_schema_json: str | Mapping[str, Any] | None = None,
        partition_columns: Iterable[str] | None = None,
        properties: Mapping[str, str] | None = None,
        comment: str | None = None,
    ) -> ResolvedTable: ...

    def path_credentials(self, url: str, operation: str = "PATH_READ") -> Credentials: ...

    def create_staging_table(self, ref: TableRef) -> StagingTable: ...

    def finalize_managed_table(
        self, ref: TableRef, request_body: Mapping[str, Any]
    ) -> ResolvedTable: ...
