"""Regression tests for defects found auditing the core modules.

identity, predicate, errors, properties, _sdk, catalog.registry and
catalog.filesystem -- one small test per defect.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import pathlib
import pickle
import subprocess
import sys
import warnings
from typing import Any

import pytest
from deltaswamp import predicate as P
from deltaswamp.capability import Engine, Operation
from deltaswamp.catalog import registry
from deltaswamp.catalog.filesystem import FilesystemCatalog
from deltaswamp.errors import (
    CommitConflictError,
    FallbackRequiredError,
    InvalidArgumentError,
    InvalidReferenceError,
    PropertyNotSupportedError,
    UnreachableTableError,
)
from deltaswamp.identity import RefKind, TableRef, parse_ref, split_identifier
from deltaswamp.properties import validate_properties

pa = pytest.importorskip("pyarrow")

D = decimal.Decimal


@pytest.fixture
def t() -> Any:
    return pa.table(
        {
            "id": pa.array([1, 2, None], pa.int64()),
            "j": pa.array([1, None, None], pa.int64()),
            "f": pa.array([1.1, 2.0, None], pa.float32()),
            "day": pa.array([dt.date(2024, 1, 1)] * 3),
            "amount": pa.array([D("1.50")] * 3, pa.decimal128(10, 2)),
            "s": ["a%b", "AXB", None],
            "ts": pa.array([dt.datetime(2024, 1, 1, 12)] * 3, pa.timestamp("us", tz="UTC")),
            "tss": pa.array([dt.datetime(2024, 1, 1, 12)] * 3, pa.timestamp("s")),
            "b": pa.array([True, False, None]),
            "st": pa.array([{"x": 1}, {"x": 2}, None]),
        }
    )


def rows(table: Any, predicate: str) -> int:
    return int(P.filter_table(table, predicate).num_rows)


# ------------------------------------------------------------------ predicate


class TestPredicateExactFilter:
    def test_column_names_resolve_case_insensitively(self, t: Any) -> None:
        # The kernel's skipping conversion already resolved ID -> id; the
        # exact filter raised "unknown column".
        assert rows(t, "ID = 1") == 1
        assert rows(t, "ST.X = 2") == 1

    def test_case_ambiguous_column_is_an_error_not_a_guess(self) -> None:
        table = pa.table({"ab": [1], "AB": [2]})
        assert rows(table, "ab = 1") == 1  # an exact match wins
        with pytest.raises(P.PredicateError, match="ambiguous"):
            rows(table, "Ab = 1")

    def test_float_column_is_widened_not_the_literal_narrowed(self, t: Any) -> None:
        # f32(1.1) > 1.1 as DOUBLE (Spark); narrowing 1.1 to f32 made it equal.
        assert rows(t, "f > 1.1") == 2
        assert rows(t, "f > 1.1D") == 2

    def test_date_column_against_timestamp_literal_keeps_time_of_day(self, t: Any) -> None:
        # The timestamp was truncated to a date, so 2024-01-01 >= 12:00 held.
        assert rows(t, "day >= TIMESTAMP '2024-01-01 12:00:00'") == 0
        assert rows(t, "day < TIMESTAMP '2024-01-01 12:00:00'") == 3

    def test_integer_literal_beyond_bigint_is_a_decimal(self, t: Any) -> None:
        assert rows(t, "id = 99999999999999999999") == 0
        node = P.parse("id < 99999999999999999999")
        assert json.loads(P.to_kernel_json(node) or "{}")["args"][1]["type"] == "decimal"
        assert rows(t, "id < 99999999999999999999") == 2

    def test_fractional_bigint_suffix_is_a_predicate_error(self) -> None:
        with pytest.raises(P.PredicateError, match="BIGINT"):
            P.parse("id = 1.5L")

    def test_unparseable_string_against_numeric_column_is_named(self, t: Any) -> None:
        with pytest.raises(P.PredicateError, match="'abc'"):
            rows(t, "id = 'abc'")
        with pytest.raises(P.PredicateError, match="'maybe'"):
            rows(t, "b = 'maybe'")

    def test_numeric_string_against_integer_column(self, t: Any) -> None:
        assert rows(t, "id = '1.0'") == 1

    def test_string_against_timestamp_column(self, t: Any) -> None:
        assert rows(t, "ts = '2024-01-01 12:00:00'") == 3
        assert rows(t, "ts = '2024-01-01T12:00:00z'") == 3

    def test_fractional_timestamp_against_second_unit_column(self, t: Any) -> None:
        assert rows(t, "tss = TIMESTAMP '2024-01-01 12:00:00.5'") == 0
        assert rows(t, "tss < TIMESTAMP '2024-01-01 12:00:00.5'") == 3

    def test_parenthesised_value_on_the_left(self, t: Any) -> None:
        assert rows(t, "(id) = 1") == 1
        assert rows(t, "(1) = id") == 1
        assert rows(t, "(id) IS NULL") == 1
        assert rows(t, "((id = 1) OR (id = 2))") == 2

    def test_malformed_nesting_fails_fast(self) -> None:
        with pytest.raises(P.PredicateError):
            P.parse("(" * 22 + "id = " + ")" * 22)

    def test_deep_nesting_is_a_predicate_error(self) -> None:
        text = "id = 0"
        for i in range(1, 2000):
            text = f"(id = {i} OR {text})"
        with pytest.raises(P.PredicateError, match="nested too deeply"):
            P.parse(text)

    def test_nested_ors_flatten(self) -> None:
        text = "id = 0"
        for i in range(1, 150):
            text = f"(id = {i} OR {text})"
        node = P.parse(text)
        assert node.op == "or" and len(node.args) == 150
        # Flat JSON: no nesting for serde_json's recursion limit to trip on.
        assert json.dumps(json.loads(P.to_kernel_json(node) or "{}")).count('"or"') == 1

    def test_typed_literal_with_double_quotes(self, t: Any) -> None:
        assert rows(t, 'DATE "2024-01-01" = day') == 3

    def test_unpadded_date_literal(self, t: Any) -> None:
        assert rows(t, "day = DATE '2024-1-1'") == 3

    def test_is_distinct_from(self, t: Any) -> None:
        assert rows(t, "id IS NOT DISTINCT FROM j") == 2
        assert rows(t, "id IS DISTINCT FROM j") == 1

    def test_is_true_false(self, t: Any) -> None:
        assert rows(t, "b IS TRUE") == 1
        assert rows(t, "b IS NOT TRUE") == 2
        assert rows(t, "b IS FALSE") == 1

    def test_bare_non_boolean_column_is_named(self, t: Any) -> None:
        with pytest.raises(P.PredicateError, match="not BOOLEAN"):
            rows(t, "id = 1 AND id")

    def test_unicode_identifier(self) -> None:
        assert rows(pa.table({"café": [1, 2]}), "café = 2") == 1

    def test_unary_plus(self, t: Any) -> None:
        assert rows(t, "id = +1") == 1

    def test_ilike_and_escape(self, t: Any) -> None:
        assert rows(t, "s ILIKE 'axb'") == 1
        assert rows(t, "s LIKE 'a!%b' ESCAPE '!'") == 1
        assert rows(t, "s LIKE 'a!%x' ESCAPE '!'") == 0

    def test_like_on_non_string_column_casts(self, t: Any) -> None:
        assert rows(t, "id LIKE '1%'") == 1

    def test_between_with_literal_target_types_against_bounds(self, t: Any) -> None:
        assert rows(t, "'2024-01-01' BETWEEN day AND day") == 3


class TestPredicateLongLists:
    """Long IN lists / OR chains used to overflow Arrow's stack (SIGBUS)."""

    def _run(self, predicate_text: str) -> int:
        code = (
            "import pyarrow as pa\n"
            "from deltaswamp import predicate as P\n"
            "t = pa.table({'id': pa.array(list(range(5000)) + [None], pa.int64())})\n"
            f"print(P.filter_table(t, {predicate_text!r}).num_rows)\n"
        )
        env = {**os.environ, "PYTHONPATH": str(pathlib.Path(P.__file__).parents[1])}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
        )
        assert out.returncode == 0, out.stderr[-2000:]
        return int(out.stdout.strip())

    def test_long_in_list(self) -> None:
        items = ",".join(str(i) for i in range(3000))
        assert self._run(f"id IN ({items})") == 3000

    def test_long_or_chain(self) -> None:
        assert self._run(" OR ".join(f"id = {i}" for i in range(3000))) == 3000

    def test_in_list_null_semantics(self) -> None:
        table = pa.table({"id": pa.array([1, 2, None], pa.int64())})
        assert rows(table, "id IN (1, 3)") == 1
        assert rows(table, "id NOT IN (1, 3)") == 1
        assert rows(table, "id NOT IN (1, NULL)") == 0
        assert rows(table, "id IN (NULL)") == 0
        assert rows(table, "NOT (id IN (NULL))") == 0
        assert rows(table, "id = 1 OR id = 3") == 1
        assert rows(table, "NOT (id = 1 OR id = NULL)") == 0


class TestPredicateKernelJson:
    def test_infinite_literal_is_dropped_not_invalid_json(self) -> None:
        node = P.parse("id > 1e999 AND id = 1")
        rendered = P.to_kernel_json(node)
        assert rendered is not None and "Infinity" not in rendered
        json.loads(rendered)  # strict JSON
        assert P.to_kernel_json(P.parse("id > 1e999")) is None

    def test_small_decimal_rendered_positionally(self) -> None:
        rendered = json.loads(P.to_kernel_json(P.parse("x > 0.0000001")) or "{}")
        assert rendered["args"][1]["literal"] == "0.0000001"


class TestParseValue:
    def test_non_string_is_a_predicate_error(self) -> None:
        with pytest.raises(P.PredicateError, match="new_values"):
            P.parse_value(5)  # type: ignore[arg-type]


# ------------------------------------------------------------------ identity


class TestIdentity:
    def test_hms_quoted_name_with_slash(self) -> None:
        ref = parse_ref("hms://h:9083/`my/db`/t")
        assert (ref.schema, ref.table) == ("my/db", "t")

    def test_hms_without_endpoint(self) -> None:
        assert parse_ref("hms:///db/t").endpoint is None

    def test_hms_thrift_transport(self) -> None:
        ref = parse_ref("hms://thrift://h:9083/db/t")
        assert (ref.endpoint, ref.schema, ref.table) == ("h:9083", "db", "t")

    def test_relative_path_without_dot_slash(self) -> None:
        ref = parse_ref("data/tbl")
        assert ref.kind is RefKind.PATH and ref.path == "data/tbl"

    @pytest.mark.parametrize("raw", ["C:\\data\\t", "~/t", ".", ".."])
    def test_other_local_path_shapes(self, raw: str) -> None:
        assert parse_ref(raw).kind is RefKind.PATH

    def test_tilde_is_expanded(self) -> None:
        assert parse_ref("~/t").path == os.path.expanduser("~/t")

    def test_pathlib_path(self, tmp_path: pathlib.Path) -> None:
        ref = parse_ref(tmp_path / "t")  # type: ignore[arg-type]
        assert ref.kind is RefKind.PATH and ref.path == str(tmp_path / "t")

    def test_glue_empty_catalog_id_is_none(self) -> None:
        ref = parse_ref("glue:///db.t")
        assert ref.endpoint is None and ref.catalog == "glue"

    def test_glue_quoted_name_with_slash(self) -> None:
        ref = parse_ref("glue://`a/b`.t")
        assert (ref.endpoint, ref.schema, ref.table) == (None, "a/b", "t")

    def test_quoted_name_containing_scheme_separator(self) -> None:
        ref = parse_ref("main.`a://b`.t")
        assert ref.kind is RefKind.CATALOG and ref.schema == "a://b"

    def test_single_slash_file_uri(self) -> None:
        ref = parse_ref("file:/tmp/x")
        assert ref.kind is RefKind.PATH and ref.path == "file:///tmp/x"

    def test_whitespace_around_dots(self) -> None:
        assert split_identifier("main . sales . orders") == ["main", "sales", "orders"]
        assert split_identifier("` a `.b") == [" a ", "b"]

    def test_existing_mnt_directory_is_a_local_path(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/mnt/c/data/t")
        ref = parse_ref("/mnt/c/data/t")
        assert ref.kind is RefKind.PATH
        with pytest.raises(InvalidReferenceError, match="not reachable"):
            parse_ref("/mnt/raw/other")

    def test_new_table_under_an_existing_mnt_directory_is_a_local_path(
        self, monkeypatch: Any
    ) -> None:
        # Creating a table under a real local mount: the table does not exist
        # yet, its parent does. The bare /mnt root proves nothing.
        monkeypatch.setattr(os.path, "exists", lambda p: p in ("/mnt", "/mnt/data"))
        assert parse_ref("/mnt/data/new_tbl").kind is RefKind.PATH
        with pytest.raises(InvalidReferenceError, match="not reachable"):
            parse_ref("/mnt/raw/other")

    @pytest.mark.parametrize("part", ["café", "123", "1abc"])
    def test_full_name_quotes_non_ascii_and_digit_leading(self, part: str) -> None:
        ref = TableRef(kind=RefKind.CATALOG, catalog="main", schema="s", table=part)
        assert ref.full_name == f"main.s.`{part}`"
        assert split_identifier(ref.full_name)[2] == part


# ------------------------------------------------------------ filesystem


class TestFilesystemCatalog:
    def test_relative_path_is_pinned_to_cwd(self, tmp_path: pathlib.Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        resolved = FilesystemCatalog().resolve(parse_ref("data/tbl"))
        assert resolved.location == str(tmp_path / "data" / "tbl")

    def test_uri_is_left_alone(self) -> None:
        resolved = FilesystemCatalog().resolve(parse_ref("s3://b/t"))
        assert resolved.location == "s3://b/t"


# ------------------------------------------------------------------ errors


class TestErrorPickling:
    @pytest.mark.parametrize(
        "error",
        [
            UnreachableTableError("read", "no engine", "enable x"),
            FallbackRequiredError("read", "fallback off"),
            PropertyNotSupportedError("set", "rejected", None),
            CommitConflictError(7, "lost the race"),
        ],
    )
    def test_round_trips(self, error: Exception) -> None:
        clone = pickle.loads(pickle.dumps(error))
        assert type(clone) is type(error) and str(clone) == str(error)
        assert clone.__dict__ == error.__dict__

    def test_invalid_argument_error_is_exported(self) -> None:
        import deltaswamp

        assert deltaswamp.InvalidArgumentError is InvalidArgumentError
        assert issubclass(InvalidArgumentError, ValueError)


# ------------------------------------------------------------------ properties


class TestPropertyValues:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("delta.appendOnly", "True"),
            ("delta.appendOnly", True),
            ("delta.enableChangeDataFeed", "yes"),
            ("delta.logRetentionDuration", "30 days"),
            ("delta.deletedFileRetentionDuration", "interval 1 month"),
            ("delta.checkpointInterval", "0"),
            ("delta.dataSkippingNumIndexedCols", "x"),
            # int() accepts these; the engines' i64 parsers do not.
            ("delta.dataSkippingNumIndexedCols", "5_0"),
            ("delta.dataSkippingNumIndexedCols", " 5"),
            ("delta.columnMapping.mode", "Name"),
        ],
    )
    def test_invalid_values_are_refused_by_key(self, key: str, value: Any) -> None:
        with pytest.raises(PropertyNotSupportedError, match=key.replace(".", r"\.")):
            validate_properties({key: value}, Engine.DELTARS, Operation.CREATE)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("delta.appendOnly", "true"),
            ("delta.logRetentionDuration", "interval 30 days"),
            ("delta.checkpointInterval", "10"),
            ("delta.dataSkippingNumIndexedCols", "-1"),
            ("delta.dataSkippingNumIndexedCols", "+32"),
            ("delta.columnMapping.mode", "name"),
            ("delta.targetFileSize", "128mb"),
        ],
    )
    def test_valid_values_pass(self, key: str, value: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_properties({key: value}, Engine.DELTARS, Operation.CREATE)

    def test_wrong_case_key_gets_a_spelling_hint(self) -> None:
        with pytest.raises(PropertyNotSupportedError, match=r"did you mean delta\.appendOnly"):
            validate_properties({"delta.appendonly": "true"}, Engine.DELTARS, Operation.CREATE)

    def test_kernel_remedy_does_not_blame_delta_rs(self) -> None:
        with pytest.raises(PropertyNotSupportedError) as info:
            validate_properties({"delta.targetFileSize": "1"}, Engine.KERNEL, Operation.CREATE)
        assert "delta-rs rejects" not in str(info.value)


# ------------------------------------------------------------------ _sdk


class TestSdk:
    def test_explicit_none_product_keeps_the_stamp(self, monkeypatch: Any) -> None:
        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        import databricks.sdk

        monkeypatch.setattr(databricks.sdk, "WorkspaceClient", FakeClient)
        from deltaswamp import _sdk

        _sdk.workspace_client(host="h", product=None, product_version=None)
        assert captured["product"] == "deltaswamp" and captured["product_version"]


# ------------------------------------------------------------------ registry


class TestRegistry:
    def test_broken_entry_point_is_named(self, monkeypatch: Any) -> None:
        class BrokenEP:
            name = "broken"
            value = "nowhere.module:Cls"

            def load(self) -> Any:
                raise ImportError("No module named 'nowhere'")

        monkeypatch.setattr(registry, "entry_points", lambda group: [BrokenEP()])
        with pytest.raises(InvalidReferenceError, match="'broken' catalog entry point"):
            registry.load_catalog_class("broken")

    def test_nested_attribute_path(self) -> None:
        cls = registry.load_catalog_class("deltaswamp.errors:UnreachableTableError")
        assert cls is UnreachableTableError
        with pytest.raises(InvalidReferenceError, match="not a class"):
            registry.load_catalog_class("deltaswamp.errors:UnreachableTableError.__init__")

    def test_non_string_uri_is_named(self) -> None:
        with pytest.raises(InvalidReferenceError, match="must be a string"):
            registry.catalog_for_uri(pathlib.Path("x"))  # type: ignore[arg-type]

    def test_empty_uri_means_no_uri(self, monkeypatch: Any) -> None:
        seen: list[str | None] = []

        def load_catalog_class(name: str | None) -> type[FilesystemCatalog]:
            seen.append(name)
            return FilesystemCatalog

        monkeypatch.setattr(registry, "load_catalog_class", load_catalog_class)
        registry.catalog_for_uri("  ")
        assert seen == ["databricks"]

    def test_uri_whitespace_is_ignored(self) -> None:
        catalog = registry.catalog_for_uri(" glue://123 ", profile=None, host=None, config=None)
        assert type(catalog).__name__ == "GlueCatalog"
