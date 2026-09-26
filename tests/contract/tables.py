"""Template tables for the contract suites, one per feature permutation.

Each builder writes a small table (six rows: ``id``, ``v``, ``grp``, ``w`` plus the
columns its feature needs) once per session; a test that mutates gets its own
copy. Shapes the engines can create are created through the public API, as a
user would; the rest (legacy protocols, Spark's legacy calendar, column
defaults, generated columns, VARIANT) are written by hand, the way another
writer leaves them on disk.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

IDS = [1, 2, 3, 4, 5, 6]
VS = ["a", "b", "c", "d", "e", "f"]
GRPS = ["x", "x", "y", "y", "z", "z"]


def base_data() -> pa.Table:
    return pa.table(
        {
            "id": pa.array(IDS, pa.int64()),
            "v": pa.array(VS, pa.string()),
            "grp": pa.array(GRPS, pa.string()),
            "w": pa.array(IDS, pa.int32()),
        }
    )


def _field(name: str, kind: Any, metadata: Any = None, nullable: bool = True) -> dict[str, Any]:
    return {"name": name, "type": kind, "nullable": nullable, "metadata": metadata or {}}


BASE_FIELDS = [
    _field("id", "long"),
    _field("v", "string"),
    _field("grp", "string"),
    _field("w", "integer"),
]


def write_log(
    path: str,
    fields: list[dict[str, Any]],
    protocol: dict[str, Any],
    data: pa.Table,
    config: dict[str, str] | None = None,
    *,
    file_meta: dict[str, str] | None = None,
) -> str:
    """A one-commit table written by hand, for shapes the engines cannot create."""
    os.makedirs(os.path.join(path, "_delta_log"))
    if file_meta:
        data = data.replace_schema_metadata(file_meta)
    pq.write_table(data, os.path.join(path, "part-0.parquet"))
    size = os.path.getsize(os.path.join(path, "part-0.parquet"))
    # Two commits, as a writer that creates and then loads leaves them: the
    # table at version 0 is empty, so there is a version to restore to.
    create = [
        {"commitInfo": {"timestamp": 1_700_000_000_000, "operation": "CREATE TABLE"}},
        {"protocol": protocol},
        {
            "metaData": {
                "id": str(uuid.uuid4()),
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": [],
                "configuration": config or {},
                "createdTime": 0,
            }
        },
    ]
    load = [
        {"commitInfo": {"timestamp": 1_700_000_001_000, "operation": "WRITE"}},
        {
            "add": {
                "path": "part-0.parquet",
                "partitionValues": {},
                "size": size,
                "modificationTime": 0,
                "dataChange": True,
                "stats": json.dumps({"numRecords": data.num_rows}),
            }
        },
    ]
    for version, actions in enumerate((create, load)):
        pathlib.Path(path, "_delta_log", f"{version:020}.json").write_text(
            "\n".join(json.dumps(a) for a in actions) + "\n"
        )
    return path


def _features(writer: list[str], reader: list[str] | None = None) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "minReaderVersion": 3 if reader else 1,
        "minWriterVersion": 7,
        "writerFeatures": writer,
    }
    if reader:
        protocol["readerFeatures"] = reader
    return protocol


def _through_api(
    properties: dict[str, str] | None = None,
    *,
    partition_by: list[str] | None = None,
    cluster_by: list[str] | None = None,
    after: Callable[[Any], None] | None = None,
) -> Callable[[Any, str], None]:
    """Create through the public API and append the base rows in a second commit."""

    def build(conn: Any, path: str) -> None:
        conn.create_table(
            path,
            base_data().schema,
            properties=properties,
            partition_by=partition_by,
            cluster_by=cluster_by,
        )
        table = conn.open_table(path)
        table.append(base_data())
        if after is not None:
            after(conn.open_table(path))

    return build


def _legacy(reader: int, writer: int, config: dict[str, str] | None = None) -> Callable[..., None]:
    def build(conn: Any, path: str) -> None:
        protocol = {"minReaderVersion": reader, "minWriterVersion": writer}
        write_log(path, BASE_FIELDS, protocol, base_data(), config)

    return build


def _legacy_cm(conn: Any, path: str) -> None:
    """Protocol (2, 5) with column mapping by name: physical names differ from logical."""
    physical = {name: f"col-{uuid.uuid4()}" for name in ("id", "v", "grp", "w")}
    fields = [
        _field(
            f["name"],
            f["type"],
            {
                "delta.columnMapping.id": i + 1,
                "delta.columnMapping.physicalName": physical[f["name"]],
            },
        )
        for i, f in enumerate(BASE_FIELDS)
    ]
    data = base_data().rename_columns([physical[n] for n in ("id", "v", "grp", "w")])
    write_log(
        path,
        fields,
        {"minReaderVersion": 2, "minWriterVersion": 5},
        data,
        {"delta.columnMapping.mode": "name", "delta.columnMapping.maxColumnId": "4"},
    )


def _timestamp_ntz(conn: Any, path: str) -> None:
    import datetime as dt

    data = base_data().append_column(
        "ntz", pa.array([dt.datetime(2024, 1, i) for i in IDS], pa.timestamp("us"))
    )
    fields = [*BASE_FIELDS, _field("ntz", "timestamp_ntz")]
    write_log(path, fields, _features(["timestampNtz"], ["timestampNtz"]), data)


def _defaults(conn: Any, path: str) -> None:
    data = base_data().append_column("n", pa.array([1, 2, 3, 4, 5, 6], pa.int32()))
    fields = [*BASE_FIELDS, _field("n", "integer", {"CURRENT_DEFAULT": "42"})]
    write_log(path, fields, _features(["allowColumnDefaults"]), data)


def _generated(conn: Any, path: str) -> None:
    data = base_data().append_column("id2", pa.array([i * 2 for i in IDS], pa.int64()))
    fields = [*BASE_FIELDS, _field("id2", "long", {"delta.generationExpression": "id * 2"})]
    write_log(path, fields, _features(["generatedColumns"]), data)


def _check_constraints(conn: Any, path: str) -> None:
    write_log(
        path,
        BASE_FIELDS,
        _features(["checkConstraints"]),
        base_data(),
        {"delta.constraints.id_positive": "id > 0"},
    )


def _append_only(conn: Any, path: str) -> None:
    write_log(
        path,
        BASE_FIELDS,
        _features(["appendOnly"]),
        base_data(),
        {"delta.appendOnly": "true"},
    )


def _variant(conn: Any, path: str) -> None:
    from deltaswamp._variant import variant_column

    column = variant_column(pa, pa.array([json.dumps({"a": i}) for i in IDS]))
    data = base_data().append_column("var", column)
    fields = [*BASE_FIELDS, _field("var", "variant")]
    write_log(path, fields, _features(["variantType"], ["variantType"]), data)


def _legacy_calendar(conn: Any, path: str) -> None:
    """Spark's LEGACY rebase mode: pre-1582 dates stored in the hybrid calendar."""
    from tests.integration.test_audit_read import _spark_table

    os.rmdir(path)
    _spark_table(pathlib.Path(path))


def _materialize_dv(t: Any) -> None:
    # A deletion vector on disk, not merely the feature: id 6 is deleted.
    t.delete("id = 6")


def _v2_checkpoint(t: Any) -> None:
    t.checkpoint()


@dataclass(frozen=True)
class Fixture:
    name: str
    build: Callable[[Any, str], None]
    #: Columns the table has beyond id, v, grp (a test may need to skip them).
    extra: tuple[str, ...] = ()
    #: Rows ids present after building (DV fixtures delete one).
    ids: tuple[int, ...] = tuple(IDS)
    #: A datetime column the fixture carries (legacy calendar tables have no v/grp).
    calendar: bool = False


FIXTURES: list[Fixture] = [
    Fixture("plain", _through_api()),
    Fixture("partitioned", _through_api(partition_by=["grp"])),
    Fixture("cm_name", _through_api({"delta.columnMapping.mode": "name"})),
    Fixture("cm_id", _through_api({"delta.columnMapping.mode": "id"})),
    Fixture(
        "dv",
        _through_api({"delta.enableDeletionVectors": "true"}, after=_materialize_dv),
        ids=(1, 2, 3, 4, 5),
    ),
    Fixture("cdf", _through_api({"delta.enableChangeDataFeed": "true"})),
    Fixture(
        "dv_cdf",
        _through_api(
            {"delta.enableDeletionVectors": "true", "delta.enableChangeDataFeed": "true"},
            after=_materialize_dv,
        ),
        ids=(1, 2, 3, 4, 5),
    ),
    Fixture("row_tracking", _through_api({"delta.enableRowTracking": "true"})),
    Fixture("ict", _through_api({"delta.enableInCommitTimestamps": "true"})),
    Fixture("clustered", _through_api(cluster_by=["id"])),
    Fixture("type_widening", _through_api({"delta.enableTypeWidening": "true"})),
    Fixture("v2_checkpoint", _through_api({"delta.checkpointPolicy": "v2"}, after=_v2_checkpoint)),
    Fixture("generated", _generated, extra=("id2",)),
    Fixture("defaults", _defaults, extra=("n",)),
    Fixture("check_constraints", _check_constraints),
    Fixture("timestamp_ntz", _timestamp_ntz, extra=("ntz",)),
    Fixture("variant", _variant, extra=("var",)),
    Fixture("legacy_1_2", _legacy(1, 2)),
    Fixture("legacy_1_4", _legacy(1, 4)),
    Fixture("legacy_2_5", _legacy_cm),
    Fixture("append_only", _append_only),
    Fixture("legacy_calendar", _legacy_calendar, ids=(1, 2, 3), calendar=True),
]

BY_NAME = {f.name: f for f in FIXTURES}


def copy_table(template: str, target: str) -> str:
    shutil.copytree(template, target)
    return target
