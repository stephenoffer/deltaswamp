"""Reference parsing."""

from __future__ import annotations

import pytest
from deltaswamp.errors import InvalidReferenceError
from deltaswamp.identity import RefKind, parse_ref, split_identifier


class TestSplitIdentifier:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a.b.c", ["a", "b", "c"]),
            ("main.sales.orders", ["main", "sales", "orders"]),
            ("`my catalog`.sch.tbl", ["my catalog", "sch", "tbl"]),
            # A dot inside backticks is part of the name, not a separator.
            ("main.`my.schema`.tbl", ["main", "my.schema", "tbl"]),
            # Doubled backtick is an escaped literal.
            ("main.`we``ird`.tbl", ["main", "we`ird", "tbl"]),
            ("solo", ["solo"]),
        ],
    )
    def test_splits(self, raw: str, expected: list[str]) -> None:
        assert split_identifier(raw) == expected

    @pytest.mark.parametrize("raw", ["a..b", "a.", ".b", "`unterminated"])
    def test_rejects_malformed(self, raw: str) -> None:
        with pytest.raises(InvalidReferenceError):
            split_identifier(raw)


class TestParseRef:
    def test_three_level_name(self) -> None:
        ref = parse_ref("main.sales.orders")
        assert ref.kind is RefKind.CATALOG
        assert (ref.catalog, ref.schema, ref.table) == ("main", "sales", "orders")

    def test_two_level_name_uses_default_catalog(self) -> None:
        ref = parse_ref("sales.orders", default_catalog="main")
        assert (ref.catalog, ref.schema, ref.table) == ("main", "sales", "orders")

    def test_bare_name_uses_both_defaults(self) -> None:
        ref = parse_ref("orders", default_catalog="main", default_schema="sales")
        assert (ref.catalog, ref.schema, ref.table) == ("main", "sales", "orders")

    def test_two_level_without_default_is_refused(self) -> None:
        with pytest.raises(InvalidReferenceError, match="no default catalog"):
            parse_ref("sales.orders")

    def test_four_parts_refused(self) -> None:
        with pytest.raises(InvalidReferenceError, match="at most three"):
            parse_ref("a.b.c.d")

    @pytest.mark.parametrize(
        "uri",
        ["s3://bucket/t", "gs://bucket/t", "abfss://c@a.dfs.core.windows.net/t", "file:///tmp/t"],
    )
    def test_storage_uris(self, uri: str) -> None:
        ref = parse_ref(uri)
        assert ref.kind is RefKind.PATH
        assert ref.path == uri

    def test_uc_uri(self) -> None:
        ref = parse_ref("uc://main.sales.orders")
        assert ref.kind is RefKind.CATALOG
        assert ref.full_name == "main.sales.orders"
        assert ref.scheme == "uc"

    def test_hms_uri(self) -> None:
        ref = parse_ref("hms://metastore:9083/analytics/events")
        assert (ref.catalog, ref.schema, ref.table) == ("hive_metastore", "analytics", "events")
        assert ref.endpoint == "metastore:9083"

    def test_glue_uri_with_catalog_id(self) -> None:
        ref = parse_ref("glue://123456789012/analytics.events")
        assert (ref.schema, ref.table) == ("analytics", "events")
        assert ref.endpoint == "123456789012"

    def test_absolute_local_path(self) -> None:
        ref = parse_ref("/data/tables/orders")
        assert ref.kind is RefKind.PATH
        assert ref.scheme == "file"

    @pytest.mark.parametrize("raw", ["dbfs:/mnt/data/t", "/dbfs/tmp/t", "/mnt/raw/t"])
    def test_dbfs_refused_with_an_explanation(self, raw: str) -> None:
        """DBFS is unreachable externally; the error must say so, not 404."""
        with pytest.raises(InvalidReferenceError, match="not reachable from outside"):
            parse_ref(raw)

    def test_unknown_scheme_lists_known_ones(self) -> None:
        with pytest.raises(InvalidReferenceError, match="unsupported URI scheme"):
            parse_ref("ftp://host/t")

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_refused(self, raw: object) -> None:
        with pytest.raises(InvalidReferenceError):
            parse_ref(raw)  # type: ignore[arg-type]

    def test_full_name_requotes_awkward_identifiers(self) -> None:
        ref = parse_ref("main.`my.schema`.tbl")
        assert ref.full_name == "main.`my.schema`.tbl"

    def test_full_name_on_path_ref_is_an_error(self) -> None:
        with pytest.raises(InvalidReferenceError, match="path reference"):
            _ = parse_ref("s3://bucket/t").full_name
