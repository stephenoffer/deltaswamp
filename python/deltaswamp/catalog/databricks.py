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

import dataclasses
import functools
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from .._sdk import PRODUCT
from ..credentials.base import Credentials, Operation
from ..credentials.databricks import DatabricksCredentialProvider
from ..errors import DeltaSwampError, InvalidReferenceError, PreflightError
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
    create_table_body,
    delta_schema_to_columns,
    normalize_privilege,
    path_operation,
)
from ..identity import RefKind, TableRef
from .base import ResolvedTable, TableType, parse_commit_tail

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

LINEAGE_API = "/api/2.0/lineage-tracking"
_JSON_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

_T = TypeVar("_T")

# What each governance call needs, named in the error when it is denied. A bare
# 403 from Unity Catalog does not say, and guessing wastes an admin round trip.
_NEEDS = {
    "read": "Reading table metadata needs USE_CATALOG, USE_SCHEMA and SELECT or ownership.",
    "grants": "Reading grants needs USE_CATALOG and USE_SCHEMA on the parents; another "
    "principal's grants may need ownership or MANAGE.",
    "grant": "Changing grants needs ownership of the securable or MANAGE on it, plus "
    "USE_CATALOG and USE_SCHEMA on its parents.",
    "tags": "Setting tags needs APPLY_TAG on the table plus USE_SCHEMA and USE_CATALOG; "
    "governed tags also need ASSIGN on the tag policy.",
    "owner": "Transferring ownership needs current ownership or MANAGE.",
    "lineage": "Lineage needs BROWSE or SELECT on the table and shows only entities the "
    "caller can see.",
    "constraint": "Constraints need ownership of the table (and of the parent table for a "
    "foreign key), plus USE_CATALOG and USE_SCHEMA.",
    "catalog": "Creating a catalog needs CREATE_CATALOG on the metastore; dropping one "
    "needs ownership.",
    "schema": "Creating a schema needs CREATE_SCHEMA and USE_CATALOG; dropping one needs "
    "ownership.",
    "volume": "Creating a volume needs CREATE_VOLUME, USE_SCHEMA and USE_CATALOG (an "
    "external volume also CREATE_EXTERNAL_VOLUME on the external location); reading and "
    "writing files needs READ_VOLUME / WRITE_VOLUME.",
    "register": "Registering an external table needs EXTERNAL_USE_SCHEMA on the schema "
    "(granted explicitly: ownership and ALL_PRIVILEGES do not imply it), CREATE_TABLE and "
    "USE_SCHEMA on the schema, USE_CATALOG, and CREATE_EXTERNAL_TABLE plus "
    "EXTERNAL_USE_LOCATION on the external location.",
    "path": "Path credentials need EXTERNAL_USE_LOCATION on the external location covering "
    "the path, plus READ_FILES, WRITE_FILES or CREATE_EXTERNAL_TABLE for the operation, "
    "and external data access enabled on the metastore.",
    "staging": "Creating a managed table needs CREATE_TABLE, USE_SCHEMA and "
    "EXTERNAL_USE_SCHEMA on the schema and USE_CATALOG; creation of catalog-managed "
    "tables by external clients is gated on a workspace preview.",
}


def _dotted(ref: TableRef) -> str:
    """The unquoted three-part name the REST API takes. Backticks are SQL-only."""
    if ref.kind is not RefKind.CATALOG or not (ref.catalog and ref.schema and ref.table):
        raise InvalidReferenceError(f"{ref} is not a catalog.schema.table reference")
    return f"{ref.catalog}.{ref.schema}.{ref.table}"


def _segment(part: str) -> str:
    return urllib.parse.quote(part, safe="")


def _securable_name(target: TableRef | str) -> str:
    return _dotted(target) if isinstance(target, TableRef) else str(target)


def _three_part(name: TableRef | str, what: str) -> tuple[str, str, str]:
    full = _dotted(name) if isinstance(name, TableRef) else name
    parts = full.split(".")
    if len(parts) != 3 or not all(parts):
        raise InvalidReferenceError(f"a {what} is named catalog.schema.name, not {full!r}")
    return parts[0], parts[1], parts[2]


class _RawPrivilege:
    """A privilege name this SDK's enum does not know yet.

    The SDK serialises privileges with ``.value``, so this is all it needs;
    refusing an unknown name would make every new privilege ungrantable until
    the SDK caught up.
    """

    def __init__(self, value: str) -> None:
        self.value = value


def _sdk_privileges(privileges: Iterable[str]) -> list[Any]:
    from databricks.sdk.service.catalog import Privilege

    out: list[Any] = []
    for p in privileges:
        name = normalize_privilege(p)
        try:
            out.append(Privilege(name))
        except ValueError:
            out.append(_RawPrivilege(name))
    if not out:
        raise InvalidReferenceError("grant/revoke needs at least one privilege")
    return out


class DatabricksUnityCatalog:
    """Resolves tables in a Databricks-hosted Unity Catalog metastore.

    Also implements the optional `GovernedCatalog`, `NamespaceCatalog` and
    `TableLifecycleCatalog` protocols from `catalog.base`.
    """

    name = "databricks"
    #: Every governance method works here; see OSSUnityCatalog for the contrast.
    unsupported_operations: frozenset[str] = frozenset()

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
        # The metastore region, fetched at most once. `_region_cached` is
        # separate from `_region` so a workspace that answers None is not
        # re-asked on every resolve.
        self._region: str | None = None
        self._region_cached = False

    # ------------------------------------------------------------------ client

    @property
    def workspace(self) -> Any:
        if self._client is None:
            try:
                from .._sdk import workspace_client
            except ImportError as exc:  # pragma: no cover
                raise PreflightError(
                    "the databricks-sdk package is required for Unity Catalog access"
                ) from exc
            if self._explicit_config is not None:
                self._client = workspace_client(config=self._explicit_config)
            else:
                kwargs = dict(self._config_kwargs)
                if self._profile:
                    kwargs["profile"] = self._profile
                if self._host:
                    kwargs["host"] = self._host
                if self._token:
                    kwargs["token"] = self._token
                self._client = workspace_client(**kwargs)
        return self._client

    def _metastore_region(self) -> str | None:
        """The current metastore's region, fetched once per catalog.

        UC vends S3 keys without a region and object_store then assumes
        us-east-1, so a bucket anywhere else fails with a redirect that carries
        no Location header. The metastore is the authoritative source for a
        UC-managed location. A workspace that cannot answer (no permission, or a
        non-AWS deployment) leaves this None and the provider falls back to the
        environment.
        """
        if self._region_cached:
            return self._region
        self._region_cached = True
        try:
            current = self.workspace.metastores.current()
            if current is not None and current.metastore_id:
                info = self.workspace.metastores.get(current.metastore_id)
                self._region = getattr(info, "region", None) or None
        except Exception:
            self._region = None
        return self._region

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
            # table_uuid is deliberately unset. It means the Delta log's
            # Metadata.id, and Databricks' table_id is a different thing: the UC
            # securable's own UUID, which names the storage directory. Setting
            # it here made the identity check compare two unrelated namespaces,
            # so every UC managed table raised CorruptTableError. Databricks
            # exposes no Delta metadata id, so there is nothing to compare.
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
                    region=self._metastore_region(),
                    **self._provider_kwargs(),
                )
                if getattr(info, "table_id", None)
                else None
            ),
        )

        # A catalog-managed table is unreadable without the catalog's commit
        # tail, so fetch it as part of resolution rather than lazily.
        # Every Unity Catalog table is reachable through the catalog's Iceberg
        # REST endpoint when it has Iceberg metadata (managed Iceberg, foreign
        # Iceberg, UniForm); the Iceberg engine decides whether it applies.
        import dataclasses as _dc

        from ..engine.iceberg import iceberg_rest_uri

        host = getattr(getattr(self.workspace, "config", None), "host", None) or self._host
        if host:
            resolved = _dc.replace(
                resolved, iceberg_rest_uri=iceberg_rest_uri(host, databricks=True)
            )

        if resolved.is_catalog_managed:
            resolved = self._with_catalog_commits(resolved)

        return resolved

    def _get_table(self, ref: TableRef) -> Any:
        """Fetch a table, asking for the manifest in the same round trip.

        Without `include_manifest_capabilities` every resolve falls back to
        listing the whole schema just to learn the capability flags, which is
        slower and fails outright where list is denied but get is allowed.
        """
        # _dotted, not ref.full_name: full_name is SQL-quoted, and a backtick in
        # the REST path is a literal character, so any table whose name needs
        # quoting (a hyphen, a dot, a space) 404s on an otherwise valid lookup.
        name = _dotted(ref)
        try:
            return self.workspace.tables.get(
                full_name=name,
                include_manifest_capabilities=True,
                include_delta_metadata=True,
            )
        except TypeError:
            # An older SDK without these parameters; degrade rather than fail.
            return self.workspace.tables.get(full_name=name)
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
        held = self._held_privileges("TABLE", _dotted(ref))
        if held is None:
            return (
                "Reading a table's files needs SELECT, plus EXTERNAL USE SCHEMA on the "
                "schema (which only the catalog owner can grant) and external data "
                "access enabled on the metastore."
            )
        missing = [p for p in ("SELECT", "EXTERNAL_USE_SCHEMA") if p not in held]
        if missing:
            return f"Effective privileges are {held or 'none'}; missing {', '.join(missing)}."
        return f"Effective privileges are {held}, so the block is likely metastore-level."

    def _held_privileges(self, securable_type: str, full_name: str) -> list[str] | None:
        """Effective privileges visible on a securable, or None when unreadable.

        Normalised to the underscore spelling: the SDK returns `Privilege` enums,
        whose `str()` is ``"Privilege.SELECT"`` and never matches a bare name.
        """
        try:
            effective = self.workspace.grants.get_effective(
                securable_type=securable_type, full_name=full_name
            )
        except Exception:
            return None
        return sorted(
            {
                normalize_privilege(getattr(p, "privilege", p))
                for assignment in (getattr(effective, "privilege_assignments", None) or [])
                for p in (getattr(assignment, "privileges", None) or [])
            }
        )

    def _error(
        self,
        exc: Exception,
        action: str,
        full_name: str,
        *,
        securable_type: str = "TABLE",
        needs: str = "",
    ) -> Exception:
        """Classify a failed governance call, naming what it needs when denied."""
        text = str(exc)
        kind = type(exc).__name__
        # Databricks allowlists which connectors may WRITE through the UC Delta
        # API, by product User-Agent. deltaswamp sets one (see _sdk.py), but an
        # unregistered name is still refused, and the raw 400 reads like a bug
        # in the caller's setup rather than a Databricks-side gate.
        if "User-Agent" in text and "insufficient" in text:
            return PreflightError(
                f"cannot {action}: Databricks restricts writes through the Unity Catalog "
                "Delta API to connectors it has allowlisted, and it does not recognise "
                f"{PRODUCT!r}. This is a Databricks-side registration, not a "
                "misconfiguration here: reads of catalog-managed tables go through the "
                "same API and are unaffected. Ask Databricks support to allowlist the "
                "connector, or use ds.connect(..., allow_sql_fallback=True) to create "
                f"and write managed tables through a SQL warehouse. Underlying error: {exc}"
            )
        if (
            kind in ("NotFound", "ResourceDoesNotExist")
            or "404" in text
            or "does not exist" in text.lower()
        ):
            return InvalidReferenceError(
                f"cannot {action}: {full_name} does not exist in Unity Catalog, or is not "
                f"visible to this principal. Underlying error: {exc}"
            )
        if (
            kind in ("PermissionDenied", "Unauthenticated")
            or "403" in text
            or ("permission" in text.lower())
        ):
            held = self._held_privileges(securable_type, full_name)
            seen = (
                "" if held is None else f"Effective privileges on {full_name}: {held or 'none'}. "
            )
            return PreflightError(
                f"cannot {action}: access to {full_name} was denied. {needs} {seen}"
                f"Underlying error: {exc}".replace("  ", " ")
            )
        return PreflightError(f"cannot {action} ({full_name}): {exc}")

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

        entries, latest, location = parse_commit_tail(body, resolved.location)
        return dataclasses.replace(
            resolved,
            log_tail=entries,
            max_catalog_version=latest,
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
            # Unquoted: see _get_table_info.
            self.workspace.tables.delete(full_name=_dotted(ref))
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

    # =============================================================== governance

    def _call(
        self,
        action: str,
        full_name: str,
        needs: str,
        fn: Callable[[], _T],
        *,
        securable_type: str = "TABLE",
    ) -> _T:
        try:
            return fn()
        except DeltaSwampError:
            raise
        except Exception as exc:
            raise self._error(
                exc, action, full_name, securable_type=securable_type, needs=_NEEDS[needs]
            ) from exc

    def table_info(self, ref: TableRef) -> TableInfo:
        """Everything Unity Catalog records about a table, as plain data."""
        name = _dotted(ref)

        def fetch() -> Any:
            try:
                return self.workspace.tables.get(
                    full_name=name,
                    include_browse=True,
                    include_delta_metadata=True,
                    include_manifest_capabilities=True,
                )
            except TypeError:  # an older SDK without the include_* flags
                return self.workspace.tables.get(full_name=name)

        return TableInfo.from_api(self._call("read table metadata", name, "read", fetch))

    # ------------------------------------------------------------ permissions

    def _grant_pages(
        self, method: Any, securable_type: str, full_name: str, principal: str | None
    ) -> list[Grant]:
        out: list[Grant] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"securable_type": securable_type, "full_name": full_name}
            if principal:
                kwargs["principal"] = principal
            if token:
                kwargs["page_token"] = token
            response = method(**kwargs)
            out.extend(Grant.list_from_api(response))
            token = getattr(response, "next_page_token", None)
            if not token:
                return out

    def grants(
        self,
        target: TableRef | str,
        principal: str | None = None,
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        """Privileges granted directly on a securable (not inherited ones)."""
        kind, name = securable_type.upper(), _securable_name(target)
        return self._call(
            "read grants",
            name,
            "grants",
            lambda: self._grant_pages(self.workspace.grants.get, kind, name, principal),
            securable_type=kind,
        )

    def effective_grants(
        self,
        target: TableRef | str,
        principal: str | None = None,
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        """Privileges in force, including those inherited from catalog and schema."""
        kind, name = securable_type.upper(), _securable_name(target)
        return self._call(
            "read effective grants",
            name,
            "grants",
            lambda: self._grant_pages(self.workspace.grants.get_effective, kind, name, principal),
            securable_type=kind,
        )

    def _change_grants(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        securable_type: str,
        *,
        add: bool,
    ) -> list[Grant]:
        from databricks.sdk.service.catalog import PermissionsChange

        kind, name = securable_type.upper(), _securable_name(target)
        sdk = _sdk_privileges(privileges)
        change = (
            PermissionsChange(principal=principal, add=sdk)
            if add
            else PermissionsChange(principal=principal, remove=sdk)
        )
        response = self._call(
            "grant privileges" if add else "revoke privileges",
            name,
            "grant",
            lambda: self.workspace.grants.update(
                securable_type=kind, full_name=name, changes=[change]
            ),
            securable_type=kind,
        )
        return Grant.list_from_api(response)

    def grant(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        """Grant privileges; returns the securable's grants afterwards."""
        return self._change_grants(target, principal, privileges, securable_type, add=True)

    def revoke(
        self,
        target: TableRef | str,
        principal: str,
        privileges: Iterable[str],
        *,
        securable_type: str = "TABLE",
    ) -> list[Grant]:
        """Revoke privileges; returns the securable's grants afterwards."""
        return self._change_grants(target, principal, privileges, securable_type, add=False)

    # ------------------------------------------------------------------- tags

    @staticmethod
    def _tag_entity(ref: TableRef, column: str | None) -> tuple[str, str]:
        name = _dotted(ref)
        return ("columns", f"{name}.{column}") if column else ("tables", name)

    def tags(self, ref: TableRef, column: str | None = None) -> dict[str, str]:
        """Tags on a table, or on one of its columns. Key-only tags map to ""."""
        entity_type, entity_name = self._tag_entity(ref, column)
        assignments = self._call(
            "read tags",
            entity_name,
            "tags",
            lambda: list(
                self.workspace.entity_tag_assignments.list(
                    entity_type=entity_type, entity_name=entity_name
                )
            ),
        )
        return {str(a.tag_key): str(a.tag_value or "") for a in assignments}

    def set_tags(self, ref: TableRef, tags: Mapping[str, str], column: str | None = None) -> None:
        """Set tags, creating new keys and updating existing ones."""
        from databricks.sdk.service.catalog import EntityTagAssignment

        entity_type, entity_name = self._tag_entity(ref, column)
        existing = self.tags(ref, column)
        api = self.workspace.entity_tag_assignments
        for key, value in tags.items():
            assignment = EntityTagAssignment(
                entity_name=entity_name,
                tag_key=key,
                entity_type=entity_type,
                tag_value=value or None,
            )
            call: Callable[[], Any] = (
                functools.partial(
                    api.update,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    tag_key=key,
                    tag_assignment=assignment,
                    update_mask="tag_value",
                )
                if key in existing
                else functools.partial(api.create, tag_assignment=assignment)
            )
            self._call("set tags", entity_name, "tags", call)

    def unset_tags(self, ref: TableRef, keys: Iterable[str], column: str | None = None) -> None:
        entity_type, entity_name = self._tag_entity(ref, column)
        api = self.workspace.entity_tag_assignments
        for key in keys:
            self._call(
                "unset tags",
                entity_name,
                "tags",
                functools.partial(
                    api.delete, entity_type=entity_type, entity_name=entity_name, tag_key=key
                ),
            )

    # -------------------------------------------------------- owner & lineage

    def set_owner(self, ref: TableRef, principal: str) -> None:
        name = _dotted(ref)
        self._call(
            "change the owner",
            name,
            "owner",
            lambda: self.workspace.tables.update(full_name=name, owner=principal),
        )

    def lineage(self, ref: TableRef, direction: str = "both") -> Lineage:
        """Upstream and downstream tables, plus notebooks, jobs and queries."""
        name = _dotted(ref)
        body = self._call(
            "read lineage",
            name,
            "lineage",
            lambda: self.workspace.api_client.do(
                "GET",
                f"{LINEAGE_API}/table-lineage",
                query={"table_name": name, "include_entity_lineage": True},
                headers={"Accept": "application/json"},
            ),
        )
        return Lineage.from_api(name, body or {}, direction)

    def column_lineage(self, ref: TableRef, column: str, direction: str = "both") -> ColumnLineage:
        name = _dotted(ref)
        body = self._call(
            "read column lineage",
            name,
            "lineage",
            lambda: self.workspace.api_client.do(
                "GET",
                f"{LINEAGE_API}/column-lineage",
                query={"table_name": name, "column_name": column},
                headers={"Accept": "application/json"},
            ),
        )
        return ColumnLineage.from_api(name, column, body or {}, direction)

    # ------------------------------------------------------------ constraints

    def add_primary_key(
        self, ref: TableRef, name: str, columns: Iterable[str], *, rely: bool = False
    ) -> None:
        """Add an informational (unenforced) primary key."""
        from databricks.sdk.service.catalog import PrimaryKeyConstraint, TableConstraint

        full = _dotted(ref)
        constraint = TableConstraint(
            primary_key_constraint=PrimaryKeyConstraint(
                name=name, child_columns=list(columns), rely=rely
            )
        )
        self._call(
            "add a primary key",
            full,
            "constraint",
            lambda: self.workspace.table_constraints.create(
                full_name_arg=full, constraint=constraint
            ),
        )

    def add_foreign_key(
        self,
        ref: TableRef,
        name: str,
        columns: Iterable[str],
        parent_ref: TableRef,
        parent_columns: Iterable[str],
        *,
        rely: bool = False,
    ) -> None:
        """Add an informational (unenforced) foreign key."""
        from databricks.sdk.service.catalog import ForeignKeyConstraint, TableConstraint

        full = _dotted(ref)
        child, parent = list(columns), list(parent_columns)
        if len(child) != len(parent):
            raise InvalidReferenceError(
                f"a foreign key pairs columns one to one; got {child} -> {parent}"
            )
        constraint = TableConstraint(
            foreign_key_constraint=ForeignKeyConstraint(
                name=name,
                child_columns=child,
                parent_table=_dotted(parent_ref),
                parent_columns=parent,
                rely=rely,
            )
        )
        self._call(
            "add a foreign key",
            full,
            "constraint",
            lambda: self.workspace.table_constraints.create(
                full_name_arg=full, constraint=constraint
            ),
        )

    def drop_table_constraint(self, ref: TableRef, name: str, *, cascade: bool = False) -> None:
        full = _dotted(ref)
        self._call(
            "drop a constraint",
            full,
            "constraint",
            lambda: self.workspace.table_constraints.delete(
                full_name=full, constraint_name=name, cascade=cascade
            ),
        )

    # ============================================================= namespaces

    def create_catalog(
        self, name: str, comment: str | None = None, storage_root: str | None = None
    ) -> None:
        self._call(
            "create a catalog",
            name,
            "catalog",
            lambda: self.workspace.catalogs.create(
                name=name, comment=comment, storage_root=storage_root
            ),
            securable_type="METASTORE",
        )

    def drop_catalog(self, name: str, force: bool = False) -> None:
        self._call(
            "drop a catalog",
            name,
            "catalog",
            lambda: self.workspace.catalogs.delete(name=name, force=force),
            securable_type="CATALOG",
        )

    def create_schema(
        self,
        catalog: str,
        name: str,
        comment: str | None = None,
        storage_root: str | None = None,
    ) -> None:
        self._call(
            "create a schema",
            f"{catalog}.{name}",
            "schema",
            lambda: self.workspace.schemas.create(
                name=name, catalog_name=catalog, comment=comment, storage_root=storage_root
            ),
            securable_type="CATALOG",
        )

    def drop_schema(self, catalog: str, name: str, force: bool = False) -> None:
        full = f"{catalog}.{name}"
        self._call(
            "drop a schema",
            full,
            "schema",
            lambda: self.workspace.schemas.delete(full_name=full, force=force),
            securable_type="SCHEMA",
        )
        self._manifest_cache.pop((catalog, name), None)

    def table_exists(self, ref: TableRef) -> bool:
        name = _dotted(ref)
        try:
            response = self.workspace.tables.exists(full_name=name)
        except Exception as exc:
            error = self._error(exc, "check existence", name, needs=_NEEDS["read"])
            if isinstance(error, InvalidReferenceError):
                return False  # the parent schema or catalog is missing
            raise error from exc
        return bool(getattr(response, "table_exists", False))

    def search_tables(
        self,
        catalog: str,
        schema_pattern: str | None = None,
        table_pattern: str | None = None,
    ) -> list[TableSummary]:
        """Tables in a catalog matching SQL LIKE patterns (``%`` and ``_``).

        One paginated call for the whole catalog, carrying the capability
        manifest, rather than a get per table.
        """
        return self._call(
            "search tables",
            catalog,
            "read",
            lambda: [
                TableSummary.from_api(t)
                for t in self.workspace.tables.list_summaries(
                    catalog_name=catalog,
                    schema_name_pattern=schema_pattern,
                    table_name_pattern=table_pattern,
                    include_manifest_capabilities=True,
                )
            ],
            securable_type="CATALOG",
        )

    def list_functions(self, catalog: str, schema: str) -> list[FunctionSummary]:
        return self._call(
            "list functions",
            f"{catalog}.{schema}",
            "read",
            lambda: [
                FunctionSummary.from_api(f)
                for f in self.workspace.functions.list(catalog_name=catalog, schema_name=schema)
            ],
            securable_type="SCHEMA",
        )

    def list_volumes(self, catalog: str, schema: str) -> list[VolumeSummary]:
        return self._call(
            "list volumes",
            f"{catalog}.{schema}",
            "volume",
            lambda: [
                VolumeSummary.from_api(v)
                for v in self.workspace.volumes.list(catalog_name=catalog, schema_name=schema)
            ],
            securable_type="SCHEMA",
        )

    def create_volume(
        self,
        catalog: str,
        schema: str,
        name: str,
        volume_type: str = "MANAGED",
        storage_location: str | None = None,
        comment: str | None = None,
    ) -> VolumeSummary:
        from databricks.sdk.service.catalog import VolumeType

        kind = VolumeType(volume_type.upper())
        if kind is VolumeType.EXTERNAL and not storage_location:
            raise InvalidReferenceError("an EXTERNAL volume needs a storage_location")
        info = self._call(
            "create a volume",
            f"{catalog}.{schema}.{name}",
            "volume",
            lambda: self.workspace.volumes.create(
                catalog_name=catalog,
                schema_name=schema,
                name=name,
                volume_type=kind,
                comment=comment,
                storage_location=storage_location,
            ),
            securable_type="SCHEMA",
        )
        return VolumeSummary.from_api(info)

    def drop_volume(self, catalog: str, schema: str, name: str) -> None:
        full = f"{catalog}.{schema}.{name}"
        self._call(
            "drop a volume",
            full,
            "volume",
            lambda: self.workspace.volumes.delete(name=full),
            securable_type="VOLUME",
        )

    def volume(self, ref: TableRef | str) -> Volume:
        """Files in a volume, through the Files API at ``/Volumes/c/s/v``."""
        full = ".".join(_three_part(ref, "volume"))

        def classify(exc: Exception, action: str) -> Exception:
            return self._error(exc, action, full, securable_type="VOLUME", needs=_NEEDS["volume"])

        return Volume(full, self.workspace.files, on_error=classify)

    # ============================================================== lifecycle

    def register_table(
        self,
        ref: TableRef,
        location: str,
        *,
        columns_schema_json: str | Mapping[str, Any] | None = None,
        partition_columns: Iterable[str] | None = None,
        properties: Mapping[str, str] | None = None,
        comment: str | None = None,
    ) -> ResolvedTable:
        """Register an existing Delta log at `location` as an EXTERNAL table.

        Write the log first (see `path_credentials`); the catalog records what
        it is told and does not read the log to check. Returns the table
        resolved afresh, so its id and credential provider are the catalog's.
        """
        from databricks.sdk.service.catalog import ColumnInfo, DataSourceFormat
        from databricks.sdk.service.catalog import TableType as SdkTableType

        name = _dotted(ref)
        assert ref.catalog and ref.schema and ref.table
        partitions = list(partition_columns or ())
        if partitions and columns_schema_json is None:
            raise InvalidReferenceError(
                "partition columns are recorded on the column list, so registering a "
                "partitioned table needs columns_schema_json"
            )
        columns = (
            delta_schema_to_columns(columns_schema_json, partitions)
            if columns_schema_json is not None
            else None
        )

        def create() -> Any:
            if comment is None:
                return self.workspace.tables.create(
                    name=ref.table,
                    catalog_name=ref.catalog,
                    schema_name=ref.schema,
                    table_type=SdkTableType.EXTERNAL,
                    data_source_format=DataSourceFormat.DELTA,
                    storage_location=location,
                    columns=[ColumnInfo.from_dict(c) for c in columns] if columns else None,
                    properties=dict(properties) if properties else None,
                )
            # The SDK's create has no comment parameter, though the REST body
            # shares the TableInfo shape that carries one.
            body: dict[str, Any] = {
                "name": ref.table,
                "catalog_name": ref.catalog,
                "schema_name": ref.schema,
                "table_type": "EXTERNAL",
                "data_source_format": "DELTA",
                "storage_location": location,
                "comment": comment,
            }
            if columns:
                body["columns"] = columns
            if properties:
                body["properties"] = dict(properties)
            return self.workspace.api_client.do(
                "POST", "/api/2.1/unity-catalog/tables", body=body, headers=_JSON_HEADERS
            )

        self._call("register an external table", name, "register", create)
        self._manifest_cache.pop((ref.catalog, ref.schema), None)
        return self.resolve(ref)

    def path_credentials(self, url: str, operation: str = "PATH_READ") -> Credentials:
        """Short-lived storage credentials for a path under an external location.

        ``PATH_CREATE_TABLE`` is the one to write a new external table's log
        before `register_table`: it is the only operation that works where no
        table is registered yet.
        """
        from databricks.sdk.service.catalog import PathOperation

        op = path_operation(operation)
        response = self._call(
            f"vend {op} credentials",
            url,
            "path",
            lambda: self.workspace.temporary_path_credentials.generate_temporary_path_credentials(
                url=url, operation=PathOperation(op)
            ),
            securable_type="EXTERNAL_LOCATION",
        )
        # Same response shape as table vending; reuse its parsing, including
        # the explicit Azure endpoint.
        parser = DatabricksCredentialProvider(table_id="", table_url=url)
        creds = parser._to_credentials(
            response, Operation.READ if op == "PATH_READ" else Operation.READ_WRITE
        )
        return dataclasses.replace(creds, table_id=None, scope_prefix=creds.url or url)

    def _delta_api_tables_path(self, ref: TableRef, leaf: str) -> str:
        assert ref.catalog and ref.schema
        return (
            f"{UC_DELTA_API_BASE}/catalogs/{_segment(ref.catalog)}"
            f"/schemas/{_segment(ref.schema)}/{leaf}"
        )

    def create_staging_table(self, ref: TableRef) -> StagingTable:
        """Reserve a managed table: its id, location and write credentials.

        Nothing is visible in the catalog until `finalize_managed_table`.
        """
        name = _dotted(ref)
        body = self._call(
            "create a staging table",
            name,
            "staging",
            lambda: self.workspace.api_client.do(
                "POST",
                self._delta_api_tables_path(ref, "staging-tables"),
                body={"name": ref.table},
                headers=_JSON_HEADERS,
            ),
            securable_type="SCHEMA",
        )
        return StagingTable.from_api(name, body or {})

    def finalize_managed_table(
        self, ref: TableRef, request_body: Mapping[str, Any]
    ) -> ResolvedTable:
        """Register a staged table after its version 0 is written.

        `request_body` is the UC Delta API CreateTableRequest (kebab-case).
        """
        name = _dotted(ref)
        body = create_table_body(ref, request_body)
        self._call(
            "finalize a managed table",
            name,
            "staging",
            lambda: self.workspace.api_client.do(
                "POST",
                self._delta_api_tables_path(ref, "tables"),
                body=body,
                headers=_JSON_HEADERS,
            ),
            securable_type="SCHEMA",
        )
        assert ref.catalog and ref.schema
        self._manifest_cache.pop((ref.catalog, ref.schema), None)
        return self.resolve(ref)
