"""The kernel read path, end to end against a real on-disk Delta table.

These are the tests that prove the headline capability actually works, rather
than merely compiling. They write a table with delta-rs and read it back with
our kernel binding, which also guards against the two implementations drifting.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest

pytestmark = pytest.mark.skipif(
    not ds.has_native(), reason="native extension not built (run `maturin develop`)"
)

pa = pytest.importorskip("pyarrow")
deltalake = pytest.importorskip("deltalake")


@pytest.fixture
def table_path(tmp_path: object) -> str:
    """A two-commit Delta table: 5 rows, then 2 appended."""
    from deltalake import write_deltalake

    path = str(tmp_path)
    write_deltalake(
        path,
        pa.table(
            {"id": [1, 2, 3, 4, 5], "name": list("abcde"), "amount": [1.5, 2.5, 3.5, 4.5, 5.5]}
        ),
    )
    write_deltalake(
        path,
        pa.table({"id": [6, 7], "name": ["f", "g"], "amount": [6.5, 7.5]}),
        mode="append",
    )
    return path


def _snapshot(path: str, **kwargs: Any) -> Any:
    from deltaswamp._native import Snapshot

    return Snapshot.resolve(path, **kwargs)


class TestSnapshotMetadata:
    def test_version_reflects_both_commits(self, table_path: str) -> None:
        assert _snapshot(table_path).version == 1

    def test_table_root_is_normalised_with_trailing_slash(self, table_path: str) -> None:
        root = _snapshot(table_path).table_root
        assert root.startswith("file://")
        assert root.endswith("/")

    def test_plain_table_is_not_catalog_managed(self, table_path: str) -> None:
        assert _snapshot(table_path).is_catalog_managed is False

    def test_protocol_shape(self, table_path: str) -> None:
        min_reader, min_writer, reader_features, writer_features = _snapshot(table_path).protocol()
        assert min_reader >= 1
        assert min_writer >= 1
        assert isinstance(reader_features, list)
        assert isinstance(writer_features, list)

    def test_schema_matches_what_was_written(self, table_path: str) -> None:
        schema = _snapshot(table_path).schema()
        assert [f.name for f in schema] == ["id", "name", "amount"]

    def test_agrees_with_delta_rs_on_version(self, table_path: str) -> None:
        """Cross-engine check: the two implementations must see the same table."""
        from deltalake import DeltaTable

        assert _snapshot(table_path).version == DeltaTable(table_path).version()


class TestScan:
    def test_exports_arrow_c_stream(self, table_path: str) -> None:
        """The PyCapsule interface is the interop contract -- it is what lets
        Polars/DuckDB/pandas consume a scan without us depending on pyarrow."""
        assert hasattr(_snapshot(table_path).scan(), "__arrow_c_stream__")

    def test_reads_all_rows(self, table_path: str) -> None:
        assert pa.table(_snapshot(table_path).scan()).num_rows == 7

    def test_values_round_trip(self, table_path: str) -> None:
        got = pa.table(_snapshot(table_path).scan()).to_pydict()
        # Delta guarantees no row order across files, so compare as sets.
        assert set(got["id"]) == {1, 2, 3, 4, 5, 6, 7}
        assert set(got["name"]) == set("abcdefg")

    def test_column_projection(self, table_path: str) -> None:
        assert pa.table(_snapshot(table_path).scan(columns=["id", "amount"])).column_names == [
            "id",
            "amount",
        ]

    def test_projection_of_unknown_column_errors(self, table_path: str) -> None:
        with pytest.raises((ValueError, OSError, RuntimeError)):
            _snapshot(table_path).scan(columns=["nonexistent"])


class TestTimeTravel:
    def test_reads_an_earlier_version(self, table_path: str) -> None:
        assert pa.table(_snapshot(table_path, version=0).scan()).num_rows == 5

    def test_latest_version_is_the_default(self, table_path: str) -> None:
        assert pa.table(_snapshot(table_path).scan()).num_rows == 7

    def test_version_beyond_the_log_errors(self, table_path: str) -> None:
        with pytest.raises((ValueError, OSError, RuntimeError)):
            _snapshot(table_path, version=99)


class TestLogTailValidation:
    def test_non_contiguous_tail_is_refused_with_an_actionable_message(
        self, table_path: str
    ) -> None:
        """Kernel requires an unbroken run of commits. A gap means the caller's
        catalog response was stale or partial, so say that rather than failing
        deep inside log replay."""
        with pytest.raises(ValueError, match="not contiguous"):
            _snapshot(
                table_path,
                log_tail=[
                    (2, "00000000000000000002.uuid.json", 0, 10),
                    (4, "00000000000000000004.uuid.json", 0, 10),
                ],
                max_catalog_version=4,
            )
