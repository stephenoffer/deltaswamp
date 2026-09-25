"""Plain object-store paths: no catalog, no vending, no governance."""

from __future__ import annotations

import os
import re
from typing import Any

from ..errors import InvalidReferenceError
from ..identity import RefKind, TableRef
from .base import ResolvedTable

__all__ = ["FilesystemCatalog"]

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


class FilesystemCatalog:
    """Resolves a path reference to itself.

    Credentials come from the ambient environment (instance profile, ADC, env
    vars) or from explicit storage options, so there is no provider to attach.
    """

    name = "filesystem"

    @classmethod
    def from_uri(cls, uri: str | None, **kwargs: Any) -> FilesystemCatalog:
        return cls()

    def resolve(self, ref: TableRef) -> ResolvedTable:
        if ref.kind is not RefKind.PATH:
            raise InvalidReferenceError(
                f"{ref} names a catalog entry; the filesystem catalog handles paths only"
            )
        location = ref.path
        if (
            location
            and "://" not in location
            and not location.startswith("~")
            and not _WINDOWS_DRIVE.match(location)
        ):
            # Pin a relative local path to the directory it was named from.
            # Engines, Ray workers and a later os.chdir() would otherwise each
            # resolve it against their own working directory -- a different
            # table, or none.
            location = os.path.abspath(location)
        return ResolvedTable(ref=ref, location=location)

    def list_tables(self, catalog: str, schema: str) -> list[ResolvedTable]:
        raise NotImplementedError(
            "a filesystem path has no schema to enumerate; pass table paths directly"
        )

    def list_catalogs(self) -> list[str]:
        raise NotImplementedError("FilesystemCatalog has no catalog namespace to enumerate")

    def list_schemas(self, catalog: str) -> list[str]:
        raise NotImplementedError("FilesystemCatalog does not implement schema discovery")

    def drop_table(self, ref: TableRef) -> None:
        raise NotImplementedError("FilesystemCatalog does not implement dropping tables")
