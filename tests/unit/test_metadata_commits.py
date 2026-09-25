"""Metadata-only commits: each rule, checked on its own.

These run against hand-built protocol/metadata dicts, so every refusal and
every protocol upgrade is visible without a table on disk. The end-to-end path
through the native commit is covered in tests/integration/test_native_ddl.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from deltaswamp.engine import metadata as m
from deltaswamp.errors import UnreachableTableError


def _schema(*fields: dict[str, Any]) -> str:
    return json.dumps({"type": "struct", "fields": list(fields)})


def _field(name: str, dtype: Any = "long", nullable: Any = True, **meta: Any) -> dict[str, Any]:
    return {"name": name, "type": dtype, "nullable": nullable, "metadata": dict(meta)}


def state(
    *fields: dict[str, Any],
    configuration: dict[str, str] | None = None,
    protocol: dict[str, Any] | None = None,
    partition_columns: list[str] | None = None,
    clustering: list[list[str]] | None = None,
    version: int = 3,
) -> m.TableState:
    return m.TableState(
        version=version,
        protocol=protocol or {"minReaderVersion": 1, "minWriterVersion": 2},
        metadata={
            "id": "abc",
            "format": {"provider": "parquet", "options": {}},
            "schemaString": _schema(*(fields or (_field("id"), _field("city", "string")))),
            "partitionColumns": partition_columns or [],
            "configuration": configuration or {},
            "createdTime": 1,
        },
        timestamp=1_000,
        clustering={"clusteringColumns": clustering} if clustering is not None else None,
    )


def schema_of(change: m.Change) -> dict[str, Any]:
    assert change.metadata is not None
    parsed: dict[str, Any] = json.loads(change.metadata["schemaString"])
    return parsed


CM = {"delta.columnMapping.mode": "name", "delta.columnMapping.maxColumnId": "2"}


def cm_state(**kwargs: Any) -> m.TableState:
    return state(
        _field("id", **{"delta.columnMapping.id": 1, "delta.columnMapping.physicalName": "col-a"}),
        _field(
            "city",
            "string",
            **{"delta.columnMapping.id": 2, "delta.columnMapping.physicalName": "col-b"},
        ),
        configuration={**CM, **kwargs.pop("configuration", {})},
        protocol={
            "minReaderVersion": 2,
            "minWriterVersion": 5,
        },
        **kwargs,
    )


class TestProtocolUpgrades:
    def test_legacy_features_survive_an_upgrade(self) -> None:
        """Dropping a feature a legacy version implied would invalidate the table."""
        upgraded = m.with_features(
            {"minReaderVersion": 1, "minWriterVersion": 4}, ["domainMetadata"]
        )
        assert upgraded is not None
        assert upgraded["minWriterVersion"] == 7
        assert {
            "appendOnly",
            "invariants",
            "checkConstraints",
            "changeDataFeed",
            "generatedColumns",
            "domainMetadata",
        } <= set(upgraded["writerFeatures"])
        assert upgraded["minReaderVersion"] == 1

    def test_reader_writer_feature_raises_the_reader_version(self) -> None:
        upgraded = m.with_features(
            {"minReaderVersion": 1, "minWriterVersion": 2}, ["deletionVectors"]
        )
        assert upgraded is not None
        assert upgraded["minReaderVersion"] == 3
        assert upgraded["readerFeatures"] == ["deletionVectors"]

    def test_already_supported_is_no_change(self) -> None:
        assert (
            m.with_features({"minReaderVersion": 1, "minWriterVersion": 4}, ["changeDataFeed"])
            is None
        )

    def test_dependencies_come_along(self) -> None:
        upgraded = m.with_features({"minReaderVersion": 1, "minWriterVersion": 2}, ["clustering"])
        assert upgraded is not None
        assert "domainMetadata" in upgraded["writerFeatures"]

    def test_legacy_column_mapping_reader_stays_legacy_for_writer_features(self) -> None:
        upgraded = m.with_features(
            {"minReaderVersion": 2, "minWriterVersion": 5}, ["changeDataFeed"]
        )
        assert upgraded is None  # writer 5 already implies changeDataFeed


class TestProperties:
    def test_enabling_cdf_on_writer_2_upgrades_the_protocol(self) -> None:
        change = m.set_properties(state(), {"delta.enableChangeDataFeed": "true"})
        assert change.protocol is not None
        assert "changeDataFeed" in change.protocol["writerFeatures"]
        assert change.metadata is not None
        assert change.metadata["configuration"]["delta.enableChangeDataFeed"] == "true"

    @pytest.mark.parametrize("key", ["delta.minReaderVersion", "delta.minWriterVersion"])
    def test_protocol_versions_are_never_set_by_hand(self, key: str) -> None:
        with pytest.raises(UnreachableTableError, match="enabling a feature"):
            m.set_properties(state(), {key: "3"})

    def test_row_tracking_needs_a_backfill(self) -> None:
        with pytest.raises(UnreachableTableError, match="backfilled"):
            m.set_properties(state(), {"delta.enableRowTracking": "true"})

    def test_unknown_delta_key_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="knows"):
            m.set_properties(state(), {"delta.madeUp": "1"})

    def test_custom_keys_are_stored(self) -> None:
        change = m.set_properties(state(), {"owner.team": "data"})
        assert change.protocol is None
        assert change.metadata is not None
        assert change.metadata["configuration"]["owner.team"] == "data"

    def test_boolean_values_are_validated(self) -> None:
        with pytest.raises(UnreachableTableError, match="'true' or 'false'"):
            m.set_properties(state(), {"delta.appendOnly": "yes"})

    def test_feature_signal_adds_the_feature(self) -> None:
        change = m.set_properties(state(), {"delta.feature.deletionVectors": "supported"})
        assert change.protocol is not None
        assert "deletionVectors" in change.protocol["readerFeatures"]

    @pytest.mark.parametrize(
        "name", ["rowTracking", "catalogManaged", "clustering", "icebergCompatV3"]
    )
    def test_features_needing_more_than_metadata_are_refused(self, name: str) -> None:
        with pytest.raises(UnreachableTableError, match="cannot add"):
            m.set_properties(state(), {f"delta.feature.{name}": "supported"})

    def test_column_mapping_none_to_name_assigns_physical_names(self) -> None:
        nested = _field("addr", {"type": "struct", "fields": [_field("zip", "string")]})
        change = m.set_properties(state(_field("id"), nested), {"delta.columnMapping.mode": "name"})
        schema = schema_of(change)
        ids = [f["metadata"]["delta.columnMapping.id"] for f in schema["fields"]]
        assert ids == [1, 2]
        zip_meta = schema["fields"][1]["type"]["fields"][0]["metadata"]
        assert zip_meta["delta.columnMapping.id"] == 3
        # Existing files hold the logical names, so those become physical names.
        assert zip_meta["delta.columnMapping.physicalName"] == "zip"
        assert change.metadata is not None
        assert change.metadata["configuration"]["delta.columnMapping.maxColumnId"] == "3"
        assert change.protocol is not None
        assert "columnMapping" in change.protocol["writerFeatures"]

    def test_column_mapping_name_to_id_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="none -> name"):
            m.set_properties(cm_state(), {"delta.columnMapping.mode": "id"})

    def test_in_commit_timestamp_enablement_is_recorded(self) -> None:
        s = state()
        change = m.set_properties(s, {"delta.enableInCommitTimestamps": "true"})
        actions = [json.loads(a) for a in m.build_actions(s, change, engine_info="t", now_ms=500)]
        info = actions[0]["commitInfo"]
        # Monotonic: strictly after the previous commit even with a slow clock.
        assert info["inCommitTimestamp"] == 1_001
        config = next(a for a in actions if "metaData" in a)["metaData"]["configuration"]
        assert config["delta.inCommitTimestampEnablementVersion"] == "4"
        assert config["delta.inCommitTimestampEnablementTimestamp"] == "1001"

    def test_unset_removes_and_tolerates_missing(self) -> None:
        change = m.unset_properties(state(configuration={"a": "1"}), ["a", "missing"])
        assert change.metadata is not None
        assert "a" not in change.metadata["configuration"]
        with pytest.raises(UnreachableTableError, match="no such property"):
            m.unset_properties(state(), ["missing"], if_exists=False)

    def test_unsetting_column_mapping_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="orphan"):
            m.unset_properties(cm_state(), ["delta.columnMapping.mode"])


class TestColumns:
    def test_rename_needs_column_mapping(self) -> None:
        with pytest.raises(UnreachableTableError, match="column mapping"):
            m.rename_column(state(), "city", "town")

    def test_rename_keeps_the_physical_name(self) -> None:
        change = m.rename_column(cm_state(), "city", "town")
        field = schema_of(change)["fields"][1]
        assert field["name"] == "town"
        assert field["metadata"]["delta.columnMapping.physicalName"] == "col-b"

    def test_rename_updates_partition_columns(self) -> None:
        change = m.rename_column(cm_state(partition_columns=["city"]), "city", "town")
        assert change.metadata is not None
        assert change.metadata["partitionColumns"] == ["town"]

    def test_rename_to_an_existing_name_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="already exists"):
            m.rename_column(cm_state(), "city", "ID")

    def test_rename_referenced_by_a_constraint_is_refused(self) -> None:
        s = cm_state(configuration={"delta.constraints.c": "city <> ''"})
        with pytest.raises(UnreachableTableError, match="CHECK constraint c"):
            m.rename_column(s, "city", "town")

    def test_drop_column(self) -> None:
        change = m.drop_column(cm_state(), "city")
        assert [f["name"] for f in schema_of(change)["fields"]] == ["id"]

    def test_drop_partition_column_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="partition column"):
            m.drop_column(cm_state(partition_columns=["city"]), "city")

    def test_drop_clustering_column_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="clustering column"):
            m.drop_column(cm_state(clustering=[["col-b"]]), "city")

    def test_drop_last_column_is_refused(self) -> None:
        s = state(
            _field("id", **{"delta.columnMapping.id": 1, "delta.columnMapping.physicalName": "c"}),
            configuration={"delta.columnMapping.mode": "name"},
        )
        with pytest.raises(UnreachableTableError, match="at least one column"):
            m.drop_column(s, "id")

    def test_add_columns_under_column_mapping_gets_fresh_ids(self) -> None:
        change = m.add_columns(cm_state(), [_field("zip", "string")])
        added = schema_of(change)["fields"][-1]["metadata"]
        assert added["delta.columnMapping.id"] == 3
        assert added["delta.columnMapping.physicalName"].startswith("col-")
        assert change.metadata is not None
        assert change.metadata["configuration"]["delta.columnMapping.maxColumnId"] == "3"

    def test_add_not_null_column_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="nullable"):
            m.add_columns(state(), [_field("zip", "string", nullable=False)])

    def test_add_duplicate_is_refused(self) -> None:
        with pytest.raises(UnreachableTableError, match="already exists"):
            m.add_columns(state(), [_field("CITY", "string")])

    def test_nullability(self) -> None:
        change = m.set_nullability(state(), "city", False)
        assert schema_of(change)["fields"][1]["nullable"] is False

    def test_column_comment_on_a_nested_field(self) -> None:
        nested = _field("addr", {"type": "struct", "fields": [_field("zip", "string")]})
        change = m.set_column_comment(state(_field("id"), nested), "addr.zip", "postal code")
        zip_field = schema_of(change)["fields"][1]["type"]["fields"][0]
        assert zip_field["metadata"]["comment"] == "postal code"

    def test_unknown_column_is_named(self) -> None:
        with pytest.raises(UnreachableTableError, match="no column 'nope'"):
            m.set_column_comment(state(), "nope", "x")


class TestTypeWidening:
    def test_requires_the_feature(self) -> None:
        with pytest.raises(UnreachableTableError, match="enableTypeWidening"):
            m.alter_column_type(state(_field("n", "integer")), "n", "long")

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("byte", "short"),
            ("integer", "long"),
            ("float", "double"),
            ("short", "double"),
            ("date", "timestamp_ntz"),
            ("decimal(10,2)", "decimal(12,4)"),
            ("integer", "decimal(10,0)"),
        ],
    )
    def test_allowed_widenings(self, old: str, new: str) -> None:
        s = state(_field("n", old), configuration={"delta.enableTypeWidening": "true"})
        change = m.alter_column_type(s, "n", new)
        field = schema_of(change)["fields"][0]
        assert field["type"] == new
        assert field["metadata"]["delta.typeChanges"] == [{"fromType": old, "toType": new}]

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("long", "integer"),
            ("double", "float"),
            ("long", "double"),
            ("decimal(10,2)", "decimal(10,4)"),
            ("string", "long"),
        ],
    )
    def test_narrowing_and_lossy_changes_are_refused(self, old: str, new: str) -> None:
        s = state(_field("n", old), configuration={"delta.enableTypeWidening": "true"})
        with pytest.raises(UnreachableTableError, match="widening"):
            m.alter_column_type(s, "n", new)

    def test_sql_type_aliases(self) -> None:
        s = state(_field("n", "integer"), configuration={"delta.enableTypeWidening": "true"})
        assert schema_of(m.alter_column_type(s, "n", "BIGINT"))["fields"][0]["type"] == "long"


class TestConstraintsAndClustering:
    def test_drop_constraint(self) -> None:
        change = m.drop_constraint(state(configuration={"delta.constraints.pos": "id > 0"}), "POS")
        assert change.metadata is not None
        assert change.metadata["configuration"] == {}

    def test_drop_missing_constraint(self) -> None:
        with pytest.raises(UnreachableTableError, match="no such constraint"):
            m.drop_constraint(state(), "nope")
        assert m.drop_constraint(state(), "nope", if_exists=True).metadata is None

    def test_cluster_by_uses_physical_names_and_adds_features(self) -> None:
        change = m.cluster_by(cm_state(), ["city"])
        assert change.domains == [
            {
                "domainMetadata": {
                    "domain": "delta.clustering",
                    "configuration": json.dumps({"clusteringColumns": [["col-b"]]}),
                    "removed": False,
                }
            }
        ]
        assert change.protocol is not None
        assert {"clustering", "domainMetadata"} <= set(change.protocol["writerFeatures"])

    def test_cluster_by_none_clears_keys(self) -> None:
        change = m.cluster_by(state(clustering=[["id"]]), None)
        config = json.loads(change.domains[0]["domainMetadata"]["configuration"])
        assert config == {"clusteringColumns": []}

    def test_cluster_by_none_on_an_unclustered_table_adds_no_features(self) -> None:
        change = m.cluster_by(state(), None)
        assert change.domains == []
        assert change.protocol is None

    def test_partitioned_tables_cannot_be_clustered(self) -> None:
        with pytest.raises(UnreachableTableError, match="partitioned or clustered"):
            m.cluster_by(state(partition_columns=["city"]), ["id"])

    def test_clustering_on_a_struct_is_refused(self) -> None:
        nested = _field("addr", {"type": "struct", "fields": [_field("zip", "string")]})
        with pytest.raises(UnreachableTableError, match="primitive"):
            m.cluster_by(state(_field("id"), nested), ["addr"])


class TestRendering:
    def test_commit_info_is_first_and_metadata_follows(self) -> None:
        s = state()
        change = m.set_comment(s, "hello")
        actions = [json.loads(a) for a in m.build_actions(s, change, engine_info="t", now_ms=5)]
        assert list(actions[0]) == ["commitInfo"]
        assert actions[0]["commitInfo"]["timestamp"] == 5
        assert "inCommitTimestamp" not in actions[0]["commitInfo"]
        assert actions[1]["metaData"]["description"] == "hello"
        # Identity must never change on an ALTER.
        assert actions[1]["metaData"]["id"] == "abc"

    def test_arrow_types_convert(self) -> None:
        pa = pytest.importorskip("pyarrow")
        schema = pa.schema(
            [
                ("a", pa.int32()),
                ("b", pa.timestamp("us", tz="UTC")),
                ("c", pa.timestamp("us")),
                ("d", pa.decimal128(12, 3)),
                ("e", pa.list_(pa.string())),
                ("f", pa.map_(pa.string(), pa.int64())),
                ("g", pa.struct([("h", pa.bool_())])),
            ]
        )
        types = [f["type"] for f in m.arrow_to_delta_schema(schema)["fields"]]
        assert types[:4] == ["integer", "timestamp", "timestamp_ntz", "decimal(12,3)"]
        assert types[4]["type"] == "array"
        assert types[5]["keyType"] == "string"
        assert types[6]["fields"][0]["type"] == "boolean"
