"""Parsing table references into a single normalised form.

One entry point (`parse_ref`) accepts every shape a user might reasonably hand
us -- a Databricks three-level name, a `uc://` URI, an `hms://` or `glue://`
locator, or a raw object-store path -- and yields a `TableRef`.

Two deliberate behaviours:

* Backtick-quoted identifiers are supported, because Databricks names can
  legally contain dots and spaces. ``main.`my.schema`.tbl`` is three parts,
  not four.
* ``dbfs:/`` and ``/mnt/`` are parsed successfully and then refused with a
  specific message. They are never reachable from outside Databricks, and a
  vague "file not found" would send people hunting in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import InvalidReferenceError

__all__ = ["RefKind", "TableRef", "parse_ref", "split_identifier"]

# Object-store schemes we can hand to an engine directly.
_STORAGE_SCHEMES = frozenset(
    {
        "s3",
        "s3a",
        "s3n",
        "gs",
        "gcs",
        "abfs",
        "abfss",
        "az",
        "adl",
        "wasb",
        "wasbs",
        "file",
        "memory",
    }
)

# Schemes that name a catalog rather than a location.
_CATALOG_SCHEMES = frozenset({"uc", "unity", "hms", "hive", "glue", "deltasharing"})

# Reachable only from inside Databricks; refused with an explanation.
_UNREACHABLE_PREFIXES = ("dbfs:/", "/dbfs/", "/mnt/")


class RefKind(StrEnum):
    """Whether a reference names a catalog entry or a storage location."""

    CATALOG = "catalog"
    PATH = "path"


@dataclass(frozen=True, slots=True)
class TableRef:
    """A normalised table reference.

    Exactly one of (catalog, schema, table) or (path,) is populated, per `kind`.
    """

    kind: RefKind
    catalog: str | None = None
    schema: str | None = None
    table: str | None = None
    path: str | None = None
    scheme: str | None = None
    endpoint: str | None = None
    raw: str = ""

    @property
    def full_name(self) -> str:
        """The dotted three-level name, quoting parts that need it."""
        if self.kind is not RefKind.CATALOG:
            raise InvalidReferenceError(f"{self.raw!r} is a path reference, not a catalog name")
        return ".".join(_quote(p) for p in (self.catalog, self.schema, self.table) if p is not None)

    def __str__(self) -> str:
        return self.raw or (self.path if self.kind is RefKind.PATH else self.full_name) or ""


def _quote(part: str) -> str:
    """Re-quote an identifier if it contains characters needing it."""
    if part and all(c.isalnum() or c == "_" for c in part):
        return part
    return "`" + part.replace("`", "``") + "`"


def split_identifier(name: str) -> list[str]:
    """Split a dotted identifier, honouring backtick quoting.

    ``main.`my.schema`.tbl`` -> ``["main", "my.schema", "tbl"]``.
    A doubled backtick inside a quoted part is a literal backtick.
    """
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(name)
    in_quotes = False

    while i < n:
        ch = name[i]
        if in_quotes:
            if ch == "`":
                # A doubled backtick is an escaped literal.
                if i + 1 < n and name[i + 1] == "`":
                    buf.append("`")
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            buf.append(ch)
            i += 1
        elif ch == "`":
            in_quotes = True
            i += 1
        elif ch == ".":
            parts.append("".join(buf))
            buf.clear()
            i += 1
        else:
            buf.append(ch)
            i += 1

    if in_quotes:
        raise InvalidReferenceError(f"unterminated backtick quote in {name!r}")
    parts.append("".join(buf))

    if any(p == "" for p in parts):
        raise InvalidReferenceError(f"empty identifier part in {name!r}")
    return parts


def parse_ref(
    ref: str,
    *,
    default_catalog: str | None = None,
    default_schema: str | None = None,
) -> TableRef:
    """Parse any supported reference shape into a `TableRef`.

    `default_catalog` / `default_schema` fill in one- and two-part names, which
    is how ``conn.table("orders")`` works against a connection bound to a schema.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidReferenceError("table reference must be a non-empty string")
    ref = ref.strip()

    lowered = ref.lower()
    if lowered.startswith(_UNREACHABLE_PREFIXES):
        raise InvalidReferenceError(
            f"{ref!r} is a DBFS path. DBFS root and mounts are not reachable from outside "
            "Databricks -- there is no cloud URI and no credential vending for them. "
            "Use the table's three-level Unity Catalog name, or its external cloud path."
        )

    scheme = _scheme_of(ref)

    if scheme in _CATALOG_SCHEMES:
        return _parse_catalog_uri(ref, scheme, default_catalog, default_schema)
    if scheme in _STORAGE_SCHEMES:
        return TableRef(kind=RefKind.PATH, path=ref, scheme=scheme, raw=ref)
    if scheme is not None:
        raise InvalidReferenceError(
            f"unsupported URI scheme {scheme!r} in {ref!r}. "
            f"Known schemes: {', '.join(sorted(_STORAGE_SCHEMES | _CATALOG_SCHEMES))}"
        )

    # No scheme: an absolute filesystem path, or a dotted catalog name.
    if ref.startswith("/") or ref.startswith("./") or ref.startswith("../"):
        return TableRef(kind=RefKind.PATH, path=ref, scheme="file", raw=ref)

    return _parse_name(ref, default_catalog, default_schema, raw=ref)


def _scheme_of(ref: str) -> str | None:
    head, sep, _ = ref.partition("://")
    if sep:
        return head.lower()
    # Single-colon forms such as `dbfs:/path` are handled before this point;
    # a bare `a:b` is treated as a name, not a scheme.
    return None


def _parse_catalog_uri(
    ref: str, scheme: str, default_catalog: str | None, default_schema: str | None
) -> TableRef:
    body = ref[len(scheme) + 3 :]
    if not body:
        raise InvalidReferenceError(f"{ref!r} has no body after the scheme")

    if scheme in ("uc", "unity", "deltasharing"):
        # `uc://catalog.schema.table` -- the delta-rs precedent.
        parsed = _parse_name(body, default_catalog, default_schema, raw=ref)
        return TableRef(
            kind=RefKind.CATALOG,
            catalog=parsed.catalog,
            schema=parsed.schema,
            table=parsed.table,
            scheme=scheme,
            raw=ref,
        )

    if scheme in ("hms", "hive"):
        # `hms://host:9083/db/table` or `hms://host:9083/db.table`
        endpoint, _, rest = body.partition("/")
        if not rest:
            raise InvalidReferenceError(
                f"{ref!r} names a metastore but no table; expected hms://host:port/db/table"
            )
        parts = split_identifier(rest.replace("/", ".")) if rest else []
        if len(parts) != 2:
            raise InvalidReferenceError(
                f"{ref!r}: expected exactly db and table after the metastore endpoint, got {parts}"
            )
        return TableRef(
            kind=RefKind.CATALOG,
            catalog="hive_metastore",
            schema=parts[0],
            table=parts[1],
            scheme="hms",
            endpoint=endpoint,
            raw=ref,
        )

    if scheme == "glue":
        # `glue://database.table` or `glue://catalog_id/database.table`
        catalog_id: str | None = None
        rest = body
        if "/" in body:
            catalog_id, _, rest = body.partition("/")
        parts = split_identifier(rest)
        if len(parts) != 2:
            raise InvalidReferenceError(
                f"{ref!r}: expected database.table, got {len(parts)} part(s)"
            )
        return TableRef(
            kind=RefKind.CATALOG,
            catalog=catalog_id or "glue",
            schema=parts[0],
            table=parts[1],
            scheme="glue",
            endpoint=catalog_id,
            raw=ref,
        )

    raise InvalidReferenceError(f"unhandled catalog scheme {scheme!r}")


def _parse_name(
    name: str, default_catalog: str | None, default_schema: str | None, *, raw: str
) -> TableRef:
    parts = split_identifier(name)
    if len(parts) == 3:
        catalog, schema, table = parts
    elif len(parts) == 2:
        if default_catalog is None:
            raise InvalidReferenceError(
                f"{raw!r} is a two-part name and the connection has no default catalog; "
                "use catalog.schema.table"
            )
        catalog, schema, table = default_catalog, parts[0], parts[1]
    elif len(parts) == 1:
        if default_catalog is None or default_schema is None:
            raise InvalidReferenceError(
                f"{raw!r} is a bare table name and the connection has no default "
                "catalog/schema; use catalog.schema.table"
            )
        catalog, schema, table = default_catalog, default_schema, parts[0]
    else:
        raise InvalidReferenceError(
            f"{raw!r} has {len(parts)} parts; Unity Catalog names have at most three "
            "(quote dots inside a name with backticks)"
        )

    return TableRef(
        kind=RefKind.CATALOG, catalog=catalog, schema=schema, table=table, scheme="uc", raw=raw
    )
