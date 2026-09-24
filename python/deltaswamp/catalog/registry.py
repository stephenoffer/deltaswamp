"""Catalog discovery.

Catalogs are looked up through the ``deltaswamp.catalogs`` entry-point group, so
a third party can ship ``deltaswamp-gravitino`` out of tree and have it work
without a change here. A dotted import path is accepted too, as an escape hatch
for a class that is not installed as a distribution.

PyIceberg solves the same problem with config alone and people end up working
around it; entry points plus a dotted-path override covers both cases.

The plugin contract is one classmethod::

    class MyCatalog:
        name = "mine"

        @classmethod
        def from_uri(cls, uri: str | None, **kwargs: Any) -> "MyCatalog": ...

plus the `Catalog` protocol (`resolve`, `list_tables`).
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from ..errors import InvalidReferenceError
from .base import Catalog

__all__ = [
    "ENTRY_POINT_GROUP",
    "available_catalogs",
    "catalog_for_uri",
    "load_catalog_class",
    "scheme_to_catalog",
]

ENTRY_POINT_GROUP = "deltaswamp.catalogs"

#: Connection-URI scheme -> registered catalog name. `None` means "no URI given",
#: which we take to mean Databricks, since that needs no endpoint.
scheme_to_catalog: dict[str | None, str] = {
    None: "databricks",
    "databricks": "databricks",
    "uc": "unity",
    "unity": "unity",
    "hms": "hive",
    "hive": "hive",
    "glue": "glue",
    "file": "filesystem",
}


def available_catalogs() -> dict[str, str]:
    """Registered catalog names mapped to the object they load."""
    return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}


def load_catalog_class(name: str) -> type:
    """Resolve a catalog name, or a dotted ``module:Class`` / ``module.Class`` path.

    Tries the dotted path first only when it looks like one, so a registered name
    always wins over an accidental collision.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            loaded = ep.load()
            if not isinstance(loaded, type):
                raise InvalidReferenceError(
                    f"the {name!r} catalog entry point loaded a "
                    f"{type(loaded).__name__}, not a class"
                )
            return loaded

    if ":" in name or "." in name:
        return _import_dotted(name)

    known = sorted(available_catalogs())
    raise InvalidReferenceError(
        f"no catalog named {name!r} is registered. Known catalogs: "
        f"{', '.join(known) if known else '(none -- is deltaswamp installed?)'}. "
        "A dotted module:Class path is also accepted."
    )


def _import_dotted(path: str) -> type:
    import importlib

    module_name, _, attribute = path.partition(":")
    if not attribute:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise InvalidReferenceError(f"{path!r} is not a module:Class path")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise InvalidReferenceError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        loaded = getattr(module, attribute)
    except AttributeError as exc:
        raise InvalidReferenceError(f"{module_name!r} has no attribute {attribute!r}") from exc
    if not isinstance(loaded, type):
        raise InvalidReferenceError(f"{path!r} is not a class")
    return loaded


def catalog_for_uri(uri: str | None, **kwargs: Any) -> Catalog:
    """Build the catalog a connection URI names.

    ``deltaswamp.catalogs`` entry points are consulted, so this resolves
    third-party catalogs as readily as the built-in ones.
    """
    scheme: str | None = None
    if uri is not None:
        scheme = uri.split("://", 1)[0].lower() if "://" in uri else uri.lower()

    name = kwargs.pop("catalog_name", None) or scheme_to_catalog.get(scheme)
    if name is None:
        known = ", ".join(sorted(k for k in scheme_to_catalog if k))
        raise InvalidReferenceError(
            f"unrecognised connection URI {uri!r}. Known schemes: {known}. "
            "Pass catalog=... to supply a catalog object directly."
        )

    cls = load_catalog_class(name)
    factory = getattr(cls, "from_uri", None)
    if factory is None:
        raise InvalidReferenceError(
            f"the {name!r} catalog class does not implement from_uri(uri, **kwargs), "
            "which the catalog plugin contract requires"
        )
    catalog: Catalog = factory(uri, **kwargs)
    return catalog
