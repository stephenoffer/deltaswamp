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

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..capability import TableFeature, feature_from_wire
from ..credentials import CredentialProvider
from ..identity import TableRef

__all__ = [
    "Catalog",
    "LogTailEntry",
    "ResolvedTable",
    "TableType",
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
            # Databricks signals it as a table property too.
            or self.properties.get("delta.feature.catalogManaged") == "supported"
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
            or self.properties.get("delta.universalFormat.enabledFormats", "").find("iceberg") >= 0
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
