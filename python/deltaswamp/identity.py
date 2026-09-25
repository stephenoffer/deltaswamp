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

import os
import re
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
    """Re-quote an identifier if it contains characters needing it.

    Only ASCII letters, digits and underscores are safe bare, and not a leading
    digit: `str.isalnum` would pass `café` or `²`, which SQL parsers reject
    unquoted, and `123` would parse as a number.
    """
    if (
        part
        and part.isascii()
        and not part[0].isdigit()
        and all(c.isalnum() or c == "_" for c in part)
    ):
        return part
    return "`" + part.replace("`", "``") + "`"


def split_identifier(name: str) -> list[str]:
    """Split a dotted identifier, honouring backtick quoting.

    ``main.`my.schema`.tbl`` -> ``["main", "my.schema", "tbl"]``.
    A doubled backtick inside a quoted part is a literal backtick.
    """
    parts: list[str] = []
    # (character, came from inside backticks). Whitespace outside the quotes
    # around a part is insignificant -- `main . sales` names `main`, not
    # `main ` -- while whitespace inside them is part of the name.
    buf: list[tuple[str, bool]] = []
    i = 0
    n = len(name)
    in_quotes = False

    def finish() -> None:
        while buf and not buf[0][1] and buf[0][0].isspace():
            buf.pop(0)
        while buf and not buf[-1][1] and buf[-1][0].isspace():
            buf.pop()
        parts.append("".join(c for c, _ in buf))
        buf.clear()

    while i < n:
        ch = name[i]
        if in_quotes:
            if ch == "`":
                # A doubled backtick is an escaped literal.
                if i + 1 < n and name[i + 1] == "`":
                    buf.append(("`", True))
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            buf.append((ch, True))
            i += 1
        elif ch == "`":
            in_quotes = True
            i += 1
        elif ch == ".":
            finish()
            i += 1
        else:
            buf.append((ch, False))
            i += 1

    if in_quotes:
        raise InvalidReferenceError(f"unterminated backtick quote in {name!r}")
    finish()

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
    # A pathlib.Path is always a filesystem location, never a catalog name.
    force_path = isinstance(ref, os.PathLike)
    if force_path:
        ref = os.fsdecode(os.fspath(ref))
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidReferenceError("table reference must be a non-empty string")
    ref = ref.strip()

    lowered = ref.lower()
    if lowered.startswith(_UNREACHABLE_PREFIXES) and not (
        # `/dbfs/` is a real FUSE mount on a Databricks cluster, and `/mnt/` is
        # an ordinary local directory on Linux and WSL; only refuse what is
        # genuinely not there.
        ref.startswith("/") and _exists_locally(ref)
    ):
        raise InvalidReferenceError(
            f"{ref!r} is a DBFS path. DBFS root and mounts are not reachable from outside "
            "Databricks -- there is no cloud URI and no credential vending for them. "
            "Use the table's three-level Unity Catalog name, or its external cloud path."
        )

    if lowered.startswith("file:") and not lowered.startswith("file://"):
        # Hadoop/Java spell local URIs `file:/tmp/t`; object_store wants three slashes.
        rest = ref[len("file:") :]
        path = "file:///" + rest.lstrip("/")
        return TableRef(kind=RefKind.PATH, path=path, scheme="file", raw=ref)

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

    # No scheme: a filesystem path, or a dotted catalog name. A catalog name
    # never contains an unquoted slash, so any path separator outside
    # backticks -- `data/tbl`, `C:\data\tbl`, `~/tbl` -- makes it a path.
    if force_path or _looks_like_path(ref):
        return TableRef(kind=RefKind.PATH, path=_local_path(ref), scheme="file", raw=ref)

    return _parse_name(ref, default_catalog, default_schema, raw=ref)


def _exists_locally(path: str) -> bool:
    """`path`, or a directory under the mount root it would be created in, exists here.

    `/mnt/data/new_table` on a machine with a real `/mnt/data` is a local
    table about to be created; the bare roots `/mnt` and `/dbfs` prove nothing
    (Linux always has an empty `/mnt`).
    """
    current = os.path.normpath(path)
    while current not in ("/", "/mnt", "/dbfs", ""):
        if os.path.exists(current):
            return True
        current = os.path.dirname(current)
    return False


_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_DRIVE = re.compile(r"^[A-Za-z]:(?:[\\/]|$)")


def _scheme_of(ref: str) -> str | None:
    # A scheme is the RFC 3986 token before `://`; a backtick-quoted name that
    # happens to contain `://` (main.`a://b`.t) is not a URI.
    match = _SCHEME.match(ref)
    if match is None:
        return None
    # Single-colon forms such as `dbfs:/path` are handled before this point;
    # a bare `a:b` is treated as a name, not a scheme.
    return match.group(1).lower()


def _looks_like_path(ref: str) -> bool:
    if ref in (".", "..") or ref.startswith(("~", "/", "./", "../", "\\")):
        return True
    if _DRIVE.match(ref):
        return True
    return any(sep in _unquoted(ref) for sep in ("/", "\\"))


def _unquoted(text: str) -> str:
    """`text` with backtick-quoted sections removed."""
    out: list[str] = []
    in_quotes = False
    for ch in text:
        if ch == "`":
            in_quotes = not in_quotes
        elif not in_quotes:
            out.append(ch)
    return "".join(out)


def _local_path(path: str) -> str:
    """A local path, `~` expanded (no engine does it, and `~` is never a directory)."""
    return os.path.expanduser(path) if path.startswith("~") else path


def _split_unquoted(text: str, sep: str) -> list[str]:
    """Split on `sep` outside backticks, keeping the quotes in each piece."""
    pieces: list[str] = []
    buf: list[str] = []
    in_quotes = False
    for ch in text:
        if ch == "`":
            in_quotes = not in_quotes
        if ch == sep and not in_quotes:
            pieces.append("".join(buf))
            buf.clear()
        else:
            buf.append(ch)
    pieces.append("".join(buf))
    return pieces


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
        # `hms://host:9083/db/table` or `hms://host:9083/db.table`, optionally
        # with the `thrift://` transport that `connect()` also accepts.
        if body.lower().startswith("thrift://"):
            body = body[len("thrift://") :]
        endpoint, _, rest = body.partition("/")
        rest = rest.rstrip("/")
        if not rest:
            raise InvalidReferenceError(
                f"{ref!r} names a metastore but no table; expected hms://host:port/db/table"
            )
        # Split on `/` outside backticks, and each segment on dots: rewriting
        # `/` to `.` first would corrupt a quoted name that contains a slash.
        parts = [p for seg in _split_unquoted(rest, "/") for p in split_identifier(seg)]
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
            # `hms:///db/t` names no endpoint: the connection's own metastore.
            endpoint=endpoint or None,
            raw=ref,
        )

    if scheme == "glue":
        # `glue://database.table` or `glue://catalog_id/database.table`
        catalog_id: str | None = None
        segments = _split_unquoted(body.rstrip("/"), "/")
        if len(segments) > 2:
            raise InvalidReferenceError(f"{ref!r}: expected glue://[catalog_id/]database.table")
        if len(segments) == 2:
            catalog_id = segments[0].strip() or None
        rest = segments[-1]
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
