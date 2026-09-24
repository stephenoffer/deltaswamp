"""Catalog discovery through entry points.

These matter for packaging: the entry points are declared in `pyproject.toml`,
so if the distribution metadata is wrong these tests fail rather than the
declaration quietly doing nothing.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from deltaswamp.catalog.registry import (
    ENTRY_POINT_GROUP,
    available_catalogs,
    catalog_for_uri,
    load_catalog_class,
)
from deltaswamp.errors import InvalidReferenceError


class TestEntryPoints:
    def test_group_name(self) -> None:
        assert ENTRY_POINT_GROUP == "deltaswamp.catalogs"

    @pytest.mark.parametrize("name", ["databricks", "unity", "hive", "glue", "filesystem"])
    def test_every_builtin_catalog_is_registered(self, name: str) -> None:
        """Fails if the distribution metadata was not installed correctly."""
        assert name in available_catalogs()

    @pytest.mark.parametrize("name", ["databricks", "unity", "hive", "glue", "filesystem"])
    def test_every_registered_catalog_loads(self, name: str) -> None:
        cls = load_catalog_class(name)
        assert isinstance(cls, type)
        assert hasattr(cls, "from_uri"), f"{name} does not satisfy the plugin contract"
        assert hasattr(cls, "resolve")

    def test_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(InvalidReferenceError, match="Known catalogs"):
            load_catalog_class("nope")


class TestDottedPathEscapeHatch:
    def test_module_colon_class(self) -> None:
        cls = load_catalog_class("deltaswamp.catalog.filesystem:FilesystemCatalog")
        assert cls.__name__ == "FilesystemCatalog"

    def test_module_dot_class(self) -> None:
        cls = load_catalog_class("deltaswamp.catalog.filesystem.FilesystemCatalog")
        assert cls.__name__ == "FilesystemCatalog"

    def test_missing_module_is_reported(self) -> None:
        with pytest.raises(InvalidReferenceError, match="cannot import"):
            load_catalog_class("no.such.module:Thing")

    def test_missing_attribute_is_reported(self) -> None:
        with pytest.raises(InvalidReferenceError, match="has no attribute"):
            load_catalog_class("deltaswamp.catalog.filesystem:Nope")

    def test_non_class_is_rejected(self) -> None:
        with pytest.raises(InvalidReferenceError, match="not a class"):
            load_catalog_class("deltaswamp.catalog.registry:ENTRY_POINT_GROUP")


class TestUriDispatch:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("uc://http://localhost:8080", "OSSUnityCatalog"),
            ("unity://localhost:8080", "OSSUnityCatalog"),
            ("hms://thrift://host:9083", "HiveMetastoreCatalog"),
            ("hive://host:9083", "HiveMetastoreCatalog"),
            ("glue://", "GlueCatalog"),
            ("glue://123456789012", "GlueCatalog"),
        ],
    )
    def test_scheme_selects_the_right_catalog(self, uri: str, expected: str) -> None:
        catalog = catalog_for_uri(uri, profile=None, host=None, config=None)
        assert type(catalog).__name__ == expected

    def test_unknown_scheme_lists_known_schemes(self) -> None:
        with pytest.raises(InvalidReferenceError, match="Known schemes"):
            catalog_for_uri("ftp://host", profile=None, host=None, config=None)

    def test_glue_catalog_id_is_taken_from_the_uri(self) -> None:
        catalog = catalog_for_uri("glue://123456789012", profile=None, host=None, config=None)
        assert catalog._catalog_id == "123456789012"  # type: ignore[attr-defined]

    def test_explicit_catalog_name_overrides_the_scheme(self) -> None:
        catalog = catalog_for_uri(
            None, catalog_name="filesystem", profile=None, host=None, config=None
        )
        assert type(catalog).__name__ == "FilesystemCatalog"


class TestNoCredentialShapedLiterals:
    """No tracked file may contain something shaped like a real credential.

    A fake token that matches a provider's pattern is still rejected by GitHub
    push protection, and a scanner cannot know it is fake. This caught a
    hard-coded `dapi` + 32 hex test token that blocked a push, so it is asserted
    rather than left to reviewer attention.
    """

    PATTERNS: ClassVar[tuple[tuple[str, str], ...]] = (
        (r"dapi[0-9a-fA-F]{32}", "Databricks PAT"),
        (r"AKIA[0-9A-Z]{16}", "AWS access key"),
        (r"ghp_[0-9A-Za-z]{36}", "GitHub PAT"),
        (r"xox[baprs]-[0-9A-Za-z-]{10,}", "Slack token"),
        (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
        (r"glpat-[0-9A-Za-z_-]{20}", "GitLab PAT"),
    )

    def test_no_tracked_file_matches_a_credential_pattern(self) -> None:
        import re
        import subprocess
        from pathlib import Path

        root = Path(__file__).parents[2]
        listed = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.split()

        offenders: list[str] = []
        for name in listed:
            path = root / name
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary asset
            for pattern, label in self.PATTERNS:
                # Skip this file: it necessarily contains the patterns.
                if path == Path(__file__):
                    continue
                if re.search(pattern, text):
                    offenders.append(f"{name}: looks like a {label}")

        assert not offenders, (
            "credential-shaped literals found. Build such values at runtime "
            "instead:\n  " + "\n  ".join(offenders)
        )
