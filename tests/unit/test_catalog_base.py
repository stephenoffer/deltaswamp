"""Catalog protocols and the commit-tail wire format shared by both Unity Catalogs."""

from __future__ import annotations

import pytest
from deltaswamp.catalog.base import (
    GovernedCatalog,
    NamespaceCatalog,
    TableLifecycleCatalog,
    parse_commit_tail,
)
from deltaswamp.catalog.ossuc import OSSUnityCatalog


def test_both_catalogs_satisfy_the_protocols() -> None:
    pytest.importorskip("databricks.sdk")
    from deltaswamp.catalog.databricks import DatabricksUnityCatalog

    dbx = DatabricksUnityCatalog(host="https://example.cloud.databricks.com", token="t")
    oss = OSSUnityCatalog("http://localhost:1")
    for catalog in (dbx, oss):
        assert isinstance(catalog, GovernedCatalog)
        assert isinstance(catalog, NamespaceCatalog)
        assert isinstance(catalog, TableLifecycleCatalog)
    assert dbx.unsupported_operations == frozenset()
    assert "tags" in oss.unsupported_operations


class TestCommitTail:
    def test_kebab_case_wire_format(self) -> None:
        """The `/delta/v1` shape Databricks returns: kebab-case, `location` under `metadata`."""
        body = {
            "metadata": {
                "table-uuid": "aca332b7-3250-465e-a9f2-05743fc27cc5",
                "location": "s3://bucket/tables/aca332b7",
            },
            "commits": [
                {
                    "version": 3,
                    "file-name": "00000000000000000003.uuid.json",
                    "file-size": 1234,
                    "file-modification-timestamp": 1790301244842,
                }
            ],
            "latest-table-version": 3,
        }
        entries, latest, location = parse_commit_tail(body, "s3://fallback")
        assert latest == 3
        assert location == "s3://bucket/tables/aca332b7"
        assert len(entries) == 1
        assert entries[0].version == 3
        assert entries[0].path == "00000000000000000003.uuid.json"
        assert entries[0].size == 1234
        assert entries[0].timestamp == 1790301244842

    def test_snake_and_camel_case(self) -> None:
        """Other servers spell the same fields in snake_case or camelCase."""
        for latest_key, name_key, size_key in (
            ("latest_table_version", "file_name", "file_size"),
            ("latestTableVersion", "fileName", "fileSize"),
        ):
            body = {
                "commits": [{"version": 1, name_key: "f.json", size_key: 10, "timestamp": 5}],
                latest_key: 1,
                "location": "s3://b/t",
            }
            entries, latest, location = parse_commit_tail(body)
            assert latest == 1
            assert location == "s3://b/t"
            assert (entries[0].path, entries[0].size, entries[0].timestamp) == ("f.json", 10, 5)

    def test_missing_version_is_none_not_zero(self) -> None:
        """None means "the catalog did not say"; 0 is a real version."""
        entries, latest, location = parse_commit_tail({"commits": []}, "s3://fallback")
        assert entries == ()
        assert latest is None
        assert location == "s3://fallback"
