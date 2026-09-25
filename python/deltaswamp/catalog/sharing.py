"""Delta Sharing: tables a provider has shared with you, over the open protocol.

A share is addressed as ``share.schema.table``, the same three-level shape as a
Unity Catalog name, so a sharing connection slots into `parse_ref` unchanged:
the share takes the catalog position.

Connection URIs:

* ``sharing:///abs/path/config.share`` -- a profile file (any fsspec URL works
  after the ``sharing://`` prefix, e.g. ``sharing://s3://bucket/config.share``).
* ``sharing://`` plus ``profile=`` -- a path, a `dict`, or a JSON string holding
  the profile document (bearer-token or OAuth client-credential profiles, as the
  ``delta-sharing`` client supports them).
* ``sharing+https://host/api/2.0/delta-sharing/`` plus ``token=`` -- builds a
  version-1 bearer-token profile, for when there is no profile file.

What a recipient can see is limited by design. There is no storage location
(files arrive as presigned URLs), no credential vending, and nothing to write,
so `resolve` fills in the protocol and properties from the server's metadata
endpoint and leaves `location` empty. The router then sends every operation on
the table to the sharing engine or refuses it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .._util import http_error_text
from ..errors import InvalidReferenceError, UnreachableTableError
from ..identity import RefKind, TableRef
from .base import ResolvedTable

__all__ = ["SHARING_SCHEMES", "SharingCatalog", "load_profile", "profile_document"]

#: URI schemes this catalog answers to. `registry.scheme_to_catalog` maps each
#: of them to ``"sharing"``.
SHARING_SCHEMES: tuple[str, ...] = ("sharing", "sharing+https", "sharing+http")

_INSTALL_REMEDY = "pip install 'deltaswamp[sharing]'"


def sharing_module(name: str = "delta_sharing") -> Any:
    """Import a module of the delta-sharing client, or say which extra provides it.

    Imported dynamically because the client ships no type information.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            "Delta Sharing needs the delta-sharing package, which is not installed. "
            f"Install it with: {_INSTALL_REMEDY}"
        ) from exc


def http_error_type() -> type[BaseException]:
    """`requests.exceptions.HTTPError`, which the sharing client raises."""
    error: type[BaseException] = importlib.import_module("requests.exceptions").HTTPError
    return error


def profile_document(profile: str | Path | dict[str, Any]) -> str:
    """Normalise a profile to the string `ResolvedTable.sharing_profile` holds.

    A path or URL stays a path (so no secret is copied around); a mapping or a
    JSON string becomes a compact JSON document.
    """
    if isinstance(profile, dict):
        return json.dumps(profile, separators=(",", ":"))
    if isinstance(profile, Path):
        return str(profile)
    text = profile.strip()
    if not text:
        raise InvalidReferenceError("the Delta Sharing profile is empty")
    if text.startswith("{"):
        try:
            json.loads(text)
        except ValueError as exc:
            raise InvalidReferenceError(
                f"the Delta Sharing profile looks like JSON but does not parse: {exc}"
            ) from exc
    return text


def load_profile(profile: str | Path | dict[str, Any]) -> Any:
    """Parse a profile given as a path/URL, a JSON document, or a mapping.

    Returns a `delta_sharing.protocol.DeltaSharingProfile`.
    """
    DeltaSharingProfile = sharing_module("delta_sharing.protocol").DeltaSharingProfile

    document = profile_document(profile)
    try:
        if document.startswith("{"):
            return DeltaSharingProfile.from_json(document)
        return DeltaSharingProfile.read_from_file(document)
    except FileNotFoundError as exc:
        raise InvalidReferenceError(
            f"the Delta Sharing profile file {document!r} does not exist"
        ) from exc
    except (KeyError, ValueError, TypeError) as exc:
        where = "document" if document.startswith("{") else f"file {document!r}"
        raise InvalidReferenceError(
            f"the Delta Sharing profile {where} is not a valid profile ({exc!r}). It needs "
            "shareCredentialsVersion and endpoint, plus bearerToken (version 1) or a "
            "'type' with its credentials (version 2)"
        ) from exc


class SharingCatalog:
    """Resolves ``share.schema.table`` against a Delta Sharing server."""

    name = "sharing"

    def __init__(self, profile: str | Path | dict[str, Any]) -> None:
        self._profile = profile_document(profile)
        # Parse now so a bad profile fails at connect(), not at first read.
        self._parsed = load_profile(self._profile)

    @property
    def profile(self) -> str:
        """The profile as stored on every table this catalog resolves."""
        return self._profile

    @property
    def endpoint(self) -> str:
        return str(self._parsed.endpoint)

    def __repr__(self) -> str:  # never print a bearer token
        return f"SharingCatalog(endpoint={self.endpoint!r})"

    @classmethod
    def from_uri(
        cls,
        uri: str | None,
        *,
        profile: str | Path | dict[str, Any] | None = None,
        token: str | None = None,
        **_: Any,
    ) -> SharingCatalog:
        """Build from a ``sharing://`` / ``sharing+https://`` URI.

        `host` and `config` (which `connect()` always passes) are Databricks
        settings and are ignored here.
        """
        scheme, body = "sharing", ""
        if uri is not None:
            head, sep, rest = uri.partition("://")
            scheme = head.lower() if sep else uri.lower()
            body = rest if sep else ""

        if scheme in ("sharing+https", "sharing+http"):
            if profile is not None:
                raise InvalidReferenceError(
                    f"{uri!r} names a sharing endpoint and profile= was passed too; "
                    "give one or the other"
                )
            if not token:
                raise InvalidReferenceError(
                    f"{uri!r} names a Delta Sharing endpoint but no bearer token was given. "
                    "Pass token=..., or use sharing:///path/to/config.share with a profile file"
                )
            endpoint = f"{scheme.split('+', 1)[1]}://{body}".rstrip("/")
            return cls({"shareCredentialsVersion": 1, "endpoint": endpoint, "bearerToken": token})

        if scheme != "sharing":
            raise InvalidReferenceError(
                f"{uri!r} is not a Delta Sharing URI. Known schemes: {', '.join(SHARING_SCHEMES)}"
            )
        if body and profile is not None:
            raise InvalidReferenceError(
                f"{uri!r} names a profile file and profile= was passed too; give one or the other"
            )
        if body:
            return cls(body)
        if profile is None:
            raise InvalidReferenceError(
                "a Delta Sharing connection needs a profile: sharing:///path/to/config.share, "
                "or sharing:// with profile= (a path, a dict, or the profile's JSON)"
            )
        return cls(profile)

    # ---------------------------------------------------------------- client

    def _rest_client(self) -> Any:
        return sharing_module("delta_sharing.rest_client").DataSharingRestClient(self._parsed)

    def _client(self) -> Any:
        return sharing_module().SharingClient(self._parsed)

    # --------------------------------------------------------------- resolve

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG or not (ref.catalog and ref.schema and ref.table):
            raise InvalidReferenceError(
                f"{ref} is not a shared table name; Delta Sharing tables are share.schema.table"
            )
        protocol_module = sharing_module("delta_sharing.protocol")
        HTTPError = http_error_type()

        shared = protocol_module.Table(name=ref.table, share=ref.catalog, schema=ref.schema)
        client = self._rest_client()
        try:
            # Advertise delta responses so the server reports the full protocol
            # (reader/writer features), not just minReaderVersion.
            client.set_sharing_capabilities_header()
            response = client.query_table_metadata(shared)
        except HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (403, 404):
                raise InvalidReferenceError(
                    f"{ref.full_name} is not visible through this share "
                    f"({http_error_text(exc)}). Recipients see only the tables the provider "
                    "added to the share"
                ) from exc
            raise UnreachableTableError(
                f"resolve the shared table {ref.full_name}", http_error_text(exc)
            ) from exc
        finally:
            client.close()

        protocol, metadata = response.protocol, response.metadata
        return ResolvedTable(
            ref=ref,
            location=None,
            data_source_format="DELTA",
            table_id=metadata.id,
            table_uuid=metadata.id,
            min_reader_version=protocol.min_reader_version,
            min_writer_version=protocol.min_writer_version,
            reader_features=frozenset(protocol.reader_features or ()),
            writer_features=frozenset(protocol.writer_features or ()),
            properties={str(k): str(v) for k, v in (metadata.configuration or {}).items()},
            partition_columns=tuple(metadata.partition_columns or ()),
            # A share is read-only to its recipients, and the router consults
            # these before any engine, so say so here too.
            external_read_supported=True,
            external_write_supported=False,
            sharing_profile=self._profile,
        )

    # ------------------------------------------------------------- discovery

    def list_catalogs(self) -> list[str]:
        """The shares visible to this recipient."""
        return [share.name for share in self._client().list_shares()]

    def list_schemas(self, catalog: str) -> list[str]:
        share = sharing_module("delta_sharing.protocol").Share(name=catalog)
        return [schema.name for schema in self._client().list_schemas(share)]

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        """The shared tables in one schema.

        Returned without protocol details -- the protocol has no batch
        metadata endpoint, so filling them would cost one request per table.
        `resolve` a table to get them.
        """
        shared_schema = sharing_module("delta_sharing.protocol").Schema(name=schema, share=catalog)
        tables = self._client().list_tables(shared_schema)
        return [
            ResolvedTable(
                ref=TableRef(
                    kind=RefKind.CATALOG,
                    catalog=t.share,
                    schema=t.schema,
                    table=t.name,
                    scheme="sharing",
                    raw=f"{t.share}.{t.schema}.{t.name}",
                ),
                location=None,
                data_source_format="DELTA",
                external_read_supported=True,
                external_write_supported=False,
                sharing_profile=self._profile,
            )
            for t in tables
        ]

    def drop_table(self, ref: TableRef) -> None:
        raise NotImplementedError(
            f"cannot drop {ref}: a Delta Sharing recipient has read-only access. The provider "
            "owns the table and removes it from the share (ALTER SHARE ... REMOVE TABLE)"
        )
