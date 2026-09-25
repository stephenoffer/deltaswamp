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
    "BUILTIN_CATALOGS",
    "ENTRY_POINT_GROUP",
    "available_catalogs",
    "catalog_for_uri",
    "load_catalog_class",
    "scheme_to_catalog",
]

ENTRY_POINT_GROUP = "deltaswamp.catalogs"

#: The first-party catalogs, as dotted paths.
#:
#: These ship inside this package, so they are resolved without consulting
#: installed distribution metadata. Entry points are the *extension* mechanism,
#: not the way deltaswamp finds its own modules: a source checkout, a vendored
#: copy, a zipapp and several freezers (PyInstaller, py2app) all lose entry-point
#: metadata, and losing it used to leave every built-in catalog unregistered and
#: the library unusable. Entry points are layered on top of this map, so a third
#: party can still register a new name -- or deliberately shadow a built-in one.
#:
#: `tests/unit/test_registry.py` asserts this agrees with pyproject.toml.
BUILTIN_CATALOGS: dict[str, str] = {
    "databricks": "deltaswamp.catalog.databricks:DatabricksUnityCatalog",
    "unity": "deltaswamp.catalog.ossuc:OSSUnityCatalog",
    "hive": "deltaswamp.catalog.hms:HiveMetastoreCatalog",
    "glue": "deltaswamp.catalog.glue:GlueCatalog",
    "filesystem": "deltaswamp.catalog.filesystem:FilesystemCatalog",
    "sharing": "deltaswamp.catalog.sharing:SharingCatalog",
}

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
    # Delta Sharing: `sharing:///path/config.share`, or an endpoint plus token=.
    "sharing": "sharing",
    "sharing+https": "sharing",
    "sharing+http": "sharing",
}


def available_catalogs() -> dict[str, str]:
    """Registered catalog names mapped to the object they load.

    Built-ins first, then entry points, so an installed plugin can shadow a
    built-in name and an absent entry-point index costs nothing.
    """
    found = dict(BUILTIN_CATALOGS)
    found.update({ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)})
    return found


def load_catalog_class(name: str) -> type:
    """Resolve a catalog name, or a dotted ``module:Class`` / ``module.Class`` path.

    Tries the dotted path first only when it looks like one, so a registered name
    always wins over an accidental collision.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            try:
                loaded = ep.load()
            except Exception as exc:
                # A broken plugin must fail with its name attached, not as an
                # anonymous ImportError from deep inside importlib.
                raise InvalidReferenceError(
                    f"the {name!r} catalog entry point ({ep.value}) failed to load: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(loaded, type):
                raise InvalidReferenceError(
                    f"the {name!r} catalog entry point loaded a "
                    f"{type(loaded).__name__}, not a class"
                )
            return loaded

    # Only after the entry points, so a plugin registering an existing name
    # still wins; before the dotted path, so a name never falls through to an
    # accidental import.
    builtin = BUILTIN_CATALOGS.get(name)
    if builtin is not None:
        return _import_dotted(builtin)

    if ":" in name or "." in name:
        return _import_dotted(name)

    known = sorted(available_catalogs())
    raise InvalidReferenceError(
        f"no catalog named {name!r} is registered. Known catalogs: "
        f"{', '.join(known)}. "
        "A dotted module:Class path is also accepted."
    )


def _import_dotted(path: str) -> type:
    import importlib

    module_name, colon, attribute = path.partition(":")
    if not colon:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise InvalidReferenceError(f"{path!r} is not a module:Class path")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise InvalidReferenceError(f"cannot import {module_name!r}: {exc}") from exc
    # `module:Outer.Inner`, as entry-point syntax allows.
    loaded: Any = module
    try:
        for name in attribute.split("."):
            loaded = getattr(loaded, name)
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
        if not isinstance(uri, str):
            raise InvalidReferenceError(
                f"a connection URI must be a string, not {type(uri).__name__}; for a "
                "local table use conn.path(...) or ds.connect('file://')"
            )
        # An empty URI (an unset environment variable, typically) means "no
        # URI", not a scheme named "".
        uri = uri.strip() or None
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
