"""Databricks Unity Catalog.

Resolution happens in two stages, because some decisions must be made before the
Delta log is touched and others are only knowable afterwards.

Stage one, here: `tables.get` plus the **capability manifest**. The manifest is
the authoritative pre-flight -- a row filter or column mask silently removes an
otherwise ordinary managed Delta table from the set eligible for credential
vending, and `table_type` will not tell you that. Shallow clones must also be
caught here, because once the log is open their borrowed absolute paths are
indistinguishable from owned ones.

Stage two, in the engine: opening the snapshot yields the real protocol and
feature lists. For a `catalogManaged` table that requires the commit tail, which
comes from the UC Delta API below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..credentials.databricks import DatabricksCredentialProvider
from ..errors import InvalidReferenceError, PreflightError
from ..identity import RefKind, TableRef
from .base import LogTailEntry, ResolvedTable, TableType

if TYPE_CHECKING:  # pragma: no cover
    from databricks.sdk.core import Config

__all__ = ["UC_DELTA_API_BASE", "DatabricksUnityCatalog"]

# The current generation of the UC Delta API, as specified in OSS Unity
# Catalog's ManagedTablesSpec.md and as delta-kernel-rs's UC client calls it.
# Databricks serves this but publishes no endpoint reference for it.
#
# `/delta/preview/commits` is the DEAD previous generation -- do not use it.
UC_DELTA_API_BASE = "/api/2.1/unity-catalog/delta/v1"

# Capability flags from the securable-kind manifest. Treated as an open world:
# Databricks does not publish the full enum, so an unrecognised flag is ignored
# rather than fatal.
CAP_EXTERNAL_READ = "HAS_DIRECT_EXTERNAL_ENGINE_READ_SUPPORT"
CAP_EXTERNAL_WRITE = "HAS_DIRECT_EXTERNAL_ENGINE_WRITE_SUPPORT"


class DatabricksUnityCatalog:
    """Resolves tables in a Databricks-hosted Unity Catalog metastore."""

    name = "databricks"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> DatabricksUnityCatalog:
        """Build from a connection URI. ``databricks://host`` or None."""
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        host = kwargs.pop("host", None)
        if uri and uri.startswith("databricks://"):
            host = host or uri[len("databricks://") :] or None
        return cls(host=host, **kwargs)

    def __init__(
        self,
        *,
        config: Config | None = None,
        profile: str | None = None,
        host: str | None = None,
        token: str | None = None,
        **config_kwargs: Any,
    ) -> None:
        self._explicit_config = config
        self._profile = profile
        self._host = host
        self._token = token
        self._config_kwargs = config_kwargs
        self._client: Any = None
        # Manifest capabilities are cached per schema: one list call is both
        # cheaper and more reliable than N per-table lookups.
        self._manifest_cache: dict[tuple[str, str], dict[str, frozenset[str]]] = {}

    # ------------------------------------------------------------------ client

    @property
    def workspace(self) -> Any:
        if self._client is None:
            try:
                from databricks.sdk import WorkspaceClient
            except ImportError as exc:  # pragma: no cover
                raise PreflightError(
                    "the databricks-sdk package is required for Unity Catalog access"
                ) from exc
            if self._explicit_config is not None:
                self._client = WorkspaceClient(config=self._explicit_config)
            else:
                kwargs = dict(self._config_kwargs)
                if self._profile:
                    kwargs["profile"] = self._profile
                if self._host:
                    kwargs["host"] = self._host
                if self._token:
                    kwargs["token"] = self._token
                self._client = WorkspaceClient(**kwargs)
        return self._client

    def _provider_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._config_kwargs)
        if self._profile:
            kwargs["profile"] = self._profile
        if self._host:
            kwargs["host"] = self._host
        if self._token:
            kwargs["token"] = self._token
        if self._explicit_config is not None:
            kwargs["config"] = self._explicit_config
        return kwargs

    # ----------------------------------------------------------------- resolve

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG:
            raise InvalidReferenceError(
                f"{ref} is a path reference; use the filesystem catalog for paths"
            )
        assert ref.catalog and ref.schema and ref.table

        info = self._get_table(ref)
        capabilities = self._capabilities_for(ref, info)

        properties = dict(getattr(info, "properties", None) or {})
        # Databricks also exposes delta runtime properties separately; they carry
        # the delta.* settings that decide routing.
        runtime = getattr(info, "delta_runtime_properties_kvpairs", None)
        if runtime is not None:
            properties.update(getattr(runtime, "delta_runtime_properties", None) or {})

        table_type = self._table_type(info)
        resolved = ResolvedTable(
            ref=ref,
            location=getattr(info, "storage_location", None),
            table_type=table_type,
            data_source_format=self._enum_value(getattr(info, "data_source_format", None)),
            securable_kind=self._securable_kind(info),
            table_id=getattr(info, "table_id", None),
            table_uuid=getattr(info, "table_id", None),
            etag=getattr(info, "etag", None),
            properties=properties,
            external_read_supported=(
                None if capabilities is None else CAP_EXTERNAL_READ in capabilities
            ),
            external_write_supported=(
                None if capabilities is None else CAP_EXTERNAL_WRITE in capabilities
            ),
            credential_provider=(
                DatabricksCredentialProvider(
                    table_id=info.table_id,
                    table_url=getattr(info, "storage_location", None),
                    **self._provider_kwargs(),
                )
                if getattr(info, "table_id", None)
                else None
            ),
        )

        # A catalog-managed table is unreadable without the catalog's commit
        # tail, so fetch it as part of resolution rather than lazily.
        if resolved.is_catalog_managed:
            resolved = self._with_catalog_commits(resolved)

        return resolved

    def _get_table(self, ref: TableRef) -> Any:
        """Fetch a table, asking for the manifest in the same round trip.

        Without `include_manifest_capabilities` every resolve falls back to
        listing the whole schema just to learn the capability flags, which is
        slower and fails outright where list is denied but get is allowed.
        """
        try:
            return self.workspace.tables.get(
                full_name=ref.full_name,
                include_manifest_capabilities=True,
                include_delta_metadata=True,
            )
        except TypeError:
            # An older SDK without these parameters; degrade rather than fail.
            return self.workspace.tables.get(full_name=ref.full_name)
        except Exception as exc:
            raise self._classify(exc, ref) from exc

    def _classify(self, exc: Exception, ref: TableRef) -> Exception:
        """Turn an SDK error into something that names the cause.

        A bare 403 from Unity Catalog does not say which privilege is missing,
        so we ask `grants.get_effective` and name it.
        """
        text = str(exc)
        if "404" in text or "does not exist" in text.lower():
            return InvalidReferenceError(
                f"{ref.full_name} does not exist in Unity Catalog. If it was recently "
                "dropped and re-created, re-resolve it: a cached table_id no longer "
                f"matches. Underlying error: {exc}"
            )
        if "403" in text or "permission" in text.lower():
            return PreflightError(
                f"access to {ref.full_name} was denied. {self._missing_privileges(ref)} "
                f"Underlying error: {exc}"
            )
        return PreflightError(f"could not resolve {ref.full_name}: {exc}")

    def _missing_privileges(self, ref: TableRef) -> str:
        """Name the privileges the principal actually holds, when we can read them."""
        try:
            effective = self.workspace.grants.get_effective(
                securable_type="TABLE", full_name=ref.full_name
            )
            held = sorted(
                {
                    str(getattr(p, "privilege", p))
                    for assignment in (getattr(effective, "privilege_assignments", None) or [])
                    for p in (getattr(assignment, "privileges", None) or [])
                }
            )
        except Exception:
            return (
                "Reading a table's files needs SELECT, plus EXTERNAL USE SCHEMA on the "
                "schema (which only the catalog owner can grant) and external data "
                "access enabled on the metastore."
            )
        missing = [p for p in ("SELECT", "EXTERNAL USE SCHEMA") if p not in held]
        if missing:
            return f"Effective privileges are {held or 'none'}; missing {', '.join(missing)}."
        return f"Effective privileges are {held}, so the block is likely metastore-level."

    def _with_catalog_commits(self, resolved: ResolvedTable) -> ResolvedTable:
        """Attach the ratified commit tail and the max trustworthy version."""
        import dataclasses

        ref = resolved.ref
        assert ref.catalog and ref.schema and ref.table
        path = f"{UC_DELTA_API_BASE}/catalogs/{ref.catalog}/schemas/{ref.schema}/tables/{ref.table}"
        try:
            body = self.workspace.api_client.do("GET", path)
        except Exception as exc:
            raise PreflightError(
                f"could not load catalog commits for {ref.full_name} via the UC Delta API "
                f"({path}). Catalog-managed tables cannot be opened without the catalog's "
                "ratified commit tail, so there is no fallback to direct log listing. "
                "External access to catalog-commit tables is gated on a workspace preview "
                f"that an admin must enable. Underlying error: {exc}"
            ) from exc

        commits = body.get("commits") or []
        entries = tuple(
            LogTailEntry(
                version=int(c["version"]),
                path=c.get("file_name") or c.get("fileName") or "",
                size=int(c.get("file_size") or c.get("fileSize") or 0),
                timestamp=int(c["timestamp"]) if c.get("timestamp") is not None else None,
            )
            for c in commits
        )
        latest = body.get("latest_table_version", body.get("latestTableVersion"))
        location = body.get("location") or resolved.location

        return dataclasses.replace(
            resolved,
            log_tail=entries,
            max_catalog_version=int(latest) if latest is not None else None,
            location=location,
        )

    # ------------------------------------------------------------------ listing

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        out: list[ResolvedTable] = []
        for info in self.workspace.tables.list(
            catalog_name=catalog, schema_name=schema, include_manifest_capabilities=True
        ):
            ref = TableRef(
                kind=RefKind.CATALOG,
                catalog=catalog,
                schema=schema,
                table=info.name,
                scheme="uc",
                raw=f"{catalog}.{schema}.{info.name}",
            )
            caps = self._manifest_capabilities(info)
            out.append(
                ResolvedTable(
                    ref=ref,
                    location=getattr(info, "storage_location", None),
                    table_type=self._table_type(info),
                    data_source_format=self._enum_value(getattr(info, "data_source_format", None)),
                    securable_kind=self._securable_kind(info),
                    table_id=getattr(info, "table_id", None),
                    external_read_supported=(None if caps is None else CAP_EXTERNAL_READ in caps),
                    external_write_supported=(None if caps is None else CAP_EXTERNAL_WRITE in caps),
                )
            )
        return out

    def list_catalogs(self) -> list[str]:
        return [c.name for c in self.workspace.catalogs.list() if c.name]

    def list_schemas(self, catalog: str) -> list[str]:
        return [s.name for s in self.workspace.schemas.list(catalog_name=catalog) if s.name]

    def drop_table(self, ref: TableRef) -> None:
        """Drop a table from Unity Catalog.

        Note this removes the catalog entry. For an EXTERNAL table the files
        stay where they are, so the data outlives the registration.
        """
        try:
            self.workspace.tables.delete(full_name=ref.full_name)
        except Exception as exc:
            raise self._classify(exc, ref) from exc

    def _capabilities_for(self, ref: TableRef, info: Any) -> frozenset[str] | None:
        """Capability flags for one table, from its own manifest or the schema's."""
        own = self._manifest_capabilities(info)
        if own is not None:
            return own

        assert ref.catalog and ref.schema
        key = (ref.catalog, ref.schema)
        if key not in self._manifest_cache:
            cache: dict[str, frozenset[str]] = {}
            try:
                for listed in self.workspace.tables.list(
                    catalog_name=ref.catalog,
                    schema_name=ref.schema,
                    include_manifest_capabilities=True,
                ):
                    caps = self._manifest_capabilities(listed)
                    if caps is not None:
                        cache[listed.name] = caps
            except Exception:
                # Listing may be denied where a direct get is allowed. Absence of
                # a manifest is reported as "unknown", never as "unsupported".
                cache = {}
            self._manifest_cache[key] = cache
        return self._manifest_cache[key].get(ref.table or "")

    @staticmethod
    def _manifest_capabilities(info: Any) -> frozenset[str] | None:
        manifest = getattr(info, "securable_kind_manifest", None)
        if manifest is None:
            return None
        caps = getattr(manifest, "capabilities", None)
        if caps is None:
            return None
        return frozenset(str(c) for c in caps)

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        return getattr(value, "value", None) or str(value)

    @classmethod
    def _table_type(cls, info: Any) -> TableType | None:
        raw = cls._enum_value(getattr(info, "table_type", None))
        if raw is None:
            return None
        try:
            return TableType(raw)
        except ValueError:
            # Forward compatibility: an unrecognised table_type must not crash
            # resolution. The router refuses it by name instead.
            return None

    @classmethod
    def _securable_kind(cls, info: Any) -> str | None:
        manifest = getattr(info, "securable_kind_manifest", None)
        if manifest is not None:
            kind = cls._enum_value(getattr(manifest, "securable_kind", None))
            if kind:
                return kind
        return cls._enum_value(getattr(info, "securable_kind", None))

    # ---------------------------------------------------------------- preflight

    def preflight(self) -> list[str]:
        """Check the two admin prerequisites that block credential vending.

        Returns a list of human-readable problems; empty means ready. These two
        settings account for most first-contact failures, and both are gated on
        someone other than the caller.
        """
        problems: list[str] = []
        try:
            summary = self.workspace.metastores.summary()
            if getattr(summary, "external_access_enabled", None) is False:
                problems.append(
                    "the metastore has external data access disabled (it is off by "
                    "default). An account admin must enable it before any external "
                    "client can read table data."
                )
        except Exception as exc:
            problems.append(f"could not read metastore settings: {exc}")
        return problems
