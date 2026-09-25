"""Table-level helpers that need no engine."""

from __future__ import annotations

from deltaswamp.table import _protocol_from_properties

from tests.helpers import resolved_table


def test_protocol_is_recovered_from_catalog_properties() -> None:
    """When the log cannot be read, the catalog's properties still name the protocol."""
    table = resolved_table(
        data_source_format="DELTA",
        properties={
            "delta.minReaderVersion": "3",
            "delta.minWriterVersion": "7",
            "delta.feature.geospatial": "supported",
            "delta.feature.rowTracking": "supported",
            "delta.feature.deletionVectors": "supported",
            "delta.enableRowTracking": "true",
        },
    )
    got = _protocol_from_properties(table)
    assert got["min_reader_version"] == 3
    assert got["writer_features"] == {"geospatial", "rowTracking", "deletionVectors"}
    # rowTracking is writer-only, so it must not be reported as a reader feature.
    assert got["reader_features"] == {"geospatial", "deletionVectors"}
