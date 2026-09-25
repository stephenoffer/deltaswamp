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
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from ..errors import InvalidReferenceError, UnreachableTableError
from ..identity import RefKind, TableRef, _quote
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


def request_error_types() -> tuple[type[BaseException], ...]:
    """Everything a sharing request can fail with once the client gives up.

    `HTTPError` is only one of them: a refused connection, a timeout or a TLS
    failure raise other `requests` exceptions (after the client's retries), and
    the client raises `LookupError` (or `KeyError`, a subclass) for a response
    missing the ``delta-table-version`` header or an expected field.
    """
    requests_exceptions = importlib.import_module("requests.exceptions")
    return (requests_exceptions.RequestException, LookupError)


def quote_name(name: str) -> str:
    """Percent-encode one share/schema/table name for the request path.

    The client interpolates names into the URL verbatim, so a name holding
    ``/``, ``#``, ``?``, ``%`` or a space would address another endpoint (or
    none at all) instead of the table.
    """
    return quote(name, safe="")


def _expiration(profile: Any) -> datetime | None:
    text = getattr(profile, "expiration_time", None)
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def check_not_expired(profile: Any) -> None:
    """Refuse a bearer-token profile whose ``expirationTime`` has passed.

    The server would answer every request with a bare HTTP 401, and the
    client's own expired-token detection never fires, so say why up front.
    """
    expires = _expiration(profile)
    if expires is not None and expires <= datetime.now(UTC):
        raise InvalidReferenceError(
            f"the Delta Sharing profile's bearer token expired at {profile.expiration_time}. "
            "Ask the provider for a new credential file (or rotate the recipient token)"
        )


def error_text(exc: BaseException, profile: Any = None) -> str:
    """A sharing failure as one line: the HTTP status, or the transport error.

    For a 401 on a profile that carries an ``expirationTime``, the expiry is
    named, because that is the usual cause.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    message = str(exc).strip() or type(exc).__name__
    text = f"HTTP {status}: {message}" if status is not None else message
    if status == 401 and profile is not None:
        expires = getattr(profile, "expiration_time", None)
        if expires:
            text += f" (the profile's bearer token expires at {expires})"
    return text


_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _is_url(text: str) -> bool:
    return bool(_URL_SCHEME.match(text))


def _display(document: str) -> str:
    """A profile location safe to print.

    A profile is often fetched from a presigned URL or one with credentials in
    it; the query string and user info are the secret, so they are dropped.
    """
    if not _is_url(document):
        return repr(document)
    parts = urlsplit(document)
    host = parts.hostname or ""
    if parts.port is not None:
        host += f":{parts.port}"
    shown = f"{parts.scheme}://{host}{parts.path}"
    if parts.query or parts.username or parts.password:
        shown += " (credentials hidden)"
    return repr(shown)


def profile_document(profile: str | Path | dict[str, Any]) -> str:
    """Normalise a profile to the string `ResolvedTable.sharing_profile` holds.

    A path or URL stays a path (so no secret is copied around); a mapping or a
    JSON string becomes a compact JSON document. A local path is made absolute:
    the profile is re-read on every request, and a relative one would break as
    soon as the working directory changed.
    """
    if isinstance(profile, dict):
        try:
            return json.dumps(profile, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise InvalidReferenceError(
                f"the Delta Sharing profile mapping is not JSON-serialisable ({exc})"
            ) from exc
    if isinstance(profile, Path):
        return str(profile.expanduser().absolute())
    if not isinstance(profile, str):
        raise InvalidReferenceError(
            "a Delta Sharing profile is a path, a URL, a dict or a JSON document, "
            f"not {type(profile).__name__}"
        )
    text = profile.strip()
    if not text:
        raise InvalidReferenceError("the Delta Sharing profile is empty")
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise InvalidReferenceError(
                f"the Delta Sharing profile looks like JSON but does not parse: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise InvalidReferenceError("the Delta Sharing profile JSON must be an object")
        return text
    if _is_url(text):
        return text  # s3://, abfss://, https://, file://...: fsspec opens it as given
    # A local path: fsspec expands neither ~ nor a relative path against a
    # working directory that may change later.
    return os.path.abspath(os.path.expanduser(text))


def _read_profile_text(document: str) -> str:
    """Read a profile file from a local path or any fsspec URL."""
    fsspec = importlib.import_module("fsspec")
    where = _display(document)
    try:
        with fsspec.open(document, "rt", encoding="utf-8") as handle:
            text: str = handle.read()
            return text
    except FileNotFoundError as exc:
        raise InvalidReferenceError(
            f"the Delta Sharing profile file {where} does not exist"
        ) from exc
    except (ImportError, ValueError) as exc:
        # fsspec: an unknown protocol (ValueError) or its backend not installed.
        scheme = document.split("://", 1)[0] if _is_url(document) else "file"
        raise InvalidReferenceError(
            f"the Delta Sharing profile file {where} cannot be opened: no usable fsspec "
            f"filesystem for {scheme!r} ({type(exc).__name__}: {exc}). Install the fsspec "
            "backend for it (s3fs, gcsfs, adlfs, aiohttp...)"
        ) from exc
    except OSError as exc:  # a directory, no permission, an unreachable URL
        raise InvalidReferenceError(
            f"the Delta Sharing profile file {where} cannot be read ({type(exc).__name__})"
        ) from exc
    except UnicodeDecodeError as exc:
        raise InvalidReferenceError(
            f"the Delta Sharing profile file {where} is not UTF-8 text"
        ) from exc


def load_profile(profile: str | Path | dict[str, Any]) -> Any:
    """Parse a profile given as a path/URL, a JSON document, or a mapping.

    Returns a `delta_sharing.protocol.DeltaSharingProfile`.
    """
    DeltaSharingProfile = sharing_module("delta_sharing.protocol").DeltaSharingProfile

    document = profile_document(profile)
    inline = document.startswith("{")
    text = document if inline else _read_profile_text(document)
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("the profile is not a JSON object")
        return DeltaSharingProfile.from_json(parsed)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        where = "document" if inline else f"file {_display(document)}"
        # KeyError/ValueError text names a field or a version, never a value.
        detail = f"missing {exc}" if isinstance(exc, KeyError) else type(exc).__name__
        if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
            detail = str(exc)
        raise InvalidReferenceError(
            f"the Delta Sharing profile {where} is not a valid profile ({detail}). It needs "
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
        check_not_expired(self._parsed)

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
            if not body.strip("/") or body.startswith("/"):
                raise InvalidReferenceError(
                    f"{uri!r} names no Delta Sharing host; write "
                    f"{scheme}://host/api/2.0/delta-sharing/"
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

    def _failure(self, what: str, exc: BaseException) -> UnreachableTableError:
        return UnreachableTableError(
            what, f"the sharing server did not answer ({error_text(exc, self._parsed)})"
        )

    def _paged(self, what: str, fetch: Any, items: str) -> list[Any]:
        """Follow ``nextPageToken`` to the end, on one client that is then closed.

        (`SharingClient` builds a rest client whose HTTP session it never
        closes, and loops forever on a server that repeats a page token.)
        """
        client = self._rest_client()
        out: list[Any] = []
        seen: set[str] = set()
        token: str | None = None
        try:
            while True:
                response = fetch(client, token)
                out.extend(getattr(response, items))
                token = response.next_page_token
                if not token:
                    return out
                if token in seen:
                    raise UnreachableTableError(
                        what, f"the sharing server repeated the page token {token!r}"
                    )
                seen.add(token)
        except request_error_types() as exc:
            raise self._failure(what, exc) from exc
        finally:
            client.close()

    # --------------------------------------------------------------- resolve

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.CATALOG or not (ref.catalog and ref.schema and ref.table):
            raise InvalidReferenceError(
                f"{ref} is not a shared table name; Delta Sharing tables are share.schema.table"
            )
        protocol_module = sharing_module("delta_sharing.protocol")
        HTTPError = http_error_type()

        shared = protocol_module.Table(
            name=quote_name(ref.table),
            share=quote_name(ref.catalog),
            schema=quote_name(ref.schema),
        )
        what = f"resolve the shared table {ref.full_name}"
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
                    f"({error_text(exc, self._parsed)}). Recipients see only the tables the "
                    "provider added to the share"
                ) from exc
            raise UnreachableTableError(what, error_text(exc, self._parsed)) from exc
        except request_error_types() as exc:
            raise self._failure(what, exc) from exc
        except ValueError as exc:
            # delta_sharing.protocol.Protocol refuses a reader version it does not know.
            raise UnreachableTableError(what, str(exc), "pip install -U delta-sharing") from exc
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
        shares = self._paged(
            "list the shares",
            lambda client, token: client.list_shares(page_token=token),
            "shares",
        )
        return [share.name for share in shares]

    def list_schemas(self, catalog: str) -> list[str]:
        share = sharing_module("delta_sharing.protocol").Share(name=quote_name(catalog))
        schemas = self._paged(
            f"list the schemas of the share {catalog!r}",
            lambda client, token: client.list_schemas(share, page_token=token),
            "schemas",
        )
        return [schema.name for schema in schemas]

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        """The shared tables in one schema.

        Returned without protocol details -- the protocol has no batch
        metadata endpoint, so filling them would cost one request per table.
        `resolve` a table to get them.
        """
        shared_schema = sharing_module("delta_sharing.protocol").Schema(
            name=quote_name(schema), share=quote_name(catalog)
        )
        tables = self._paged(
            f"list the tables of {catalog}.{schema}",
            lambda client, token: client.list_tables(shared_schema, page_token=token),
            "tables",
        )
        out = []
        for t in tables:
            # Older servers omit share/schema on each item; they are the ones asked for.
            share_name, schema_name = t.share or catalog, t.schema or schema
            out.append(
                ResolvedTable(
                    ref=TableRef(
                        kind=RefKind.CATALOG,
                        catalog=share_name,
                        schema=schema_name,
                        table=t.name,
                        scheme="sharing",
                        # Quoted, so a name holding a dot parses back to the same parts.
                        raw=".".join(_quote(p) for p in (share_name, schema_name, t.name)),
                    ),
                    location=None,
                    data_source_format="DELTA",
                    external_read_supported=True,
                    external_write_supported=False,
                    sharing_profile=self._profile,
                )
            )
        return out

    def drop_table(self, ref: TableRef) -> None:
        raise NotImplementedError(
            f"cannot drop {ref}: a Delta Sharing recipient has read-only access. The provider "
            "owns the table and removes it from the share (ALTER SHARE ... REMOVE TABLE)"
        )
