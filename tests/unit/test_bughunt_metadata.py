"""Regression tests for defects found auditing engine/metadata.py."""

from __future__ import annotations

import json
from typing import Any

import pytest
from deltaswamp.engine import metadata as m
from deltaswamp.errors import UnreachableTableError


def _field(name: str, dtype: Any = "long", nullable: bool = True, /, **meta: Any) -> dict[str, Any]:
    return {"name": name, "type": dtype, "nullable": nullable, "metadata": dict(meta)}


def _struct(*fields: dict[str, Any]) -> dict[str, Any]:
    return {"type": "struct", "fields": list(fields)}


def state(
    *fields: dict[str, Any],
    configuration: dict[str, str] | None = None,
    protocol: dict[str, Any] | None = None,
    partition_columns: list[str] | None = None,
    clustering: list[list[str]] | None = None,
    extra: dict[str, Any] | None = None,
) -> m.TableState:
    metadata = {
        "id": "abc",
        "format": {"provider": "parquet", "options": {}},
        "schemaString": json.dumps(_struct(*(fields or (_field("id"), _field("city", "string"))))),
        "partitionColumns": partition_columns or [],
        "configuration": configuration or {},
        "createdTime": 1,
        **(extra or {}),
    }
    return m.TableState(
        version=3,
        protocol=protocol or {"minReaderVersion": 1, "minWriterVersion": 2},
        metadata=metadata,
        timestamp=1_000,
        clustering={"clusteringColumns": clustering} if clustering is not None else None,
    )


def _cm(name: str, fid: int, phys: str, dtype: Any = "long") -> dict[str, Any]:
    return _field(
        name,
        dtype,
        **{"delta.columnMapping.id": fid, "delta.columnMapping.physicalName": phys},
    )


CM_PROTO = {"minReaderVersion": 2, "minWriterVersion": 5}


def cm_state(
    *fields: dict[str, Any], configuration: dict[str, str] | None = None, **kw: Any
) -> m.TableState:
    return state(
        *(fields or (_cm("id", 1, "col-a"), _cm("city", 2, "col-b", "string"))),
        configuration={
            "delta.columnMapping.mode": "name",
            "delta.columnMapping.maxColumnId": "2",
            **(configuration or {}),
        },
        protocol=CM_PROTO,
        **kw,
    )


def schema_of(change: m.Change) -> dict[str, Any]:
    assert change.metadata is not None
    parsed: dict[str, Any] = json.loads(change.metadata["schemaString"])
    return parsed


TW = {"delta.enableTypeWidening": "true"}


# ------------------------------------------------------------------ protocol


def test_preview_feature_satisfies_stable_name() -> None:
    proto = {
        "minReaderVersion": 3,
        "minWriterVersion": 7,
        "readerFeatures": ["typeWidening-preview"],
        "writerFeatures": ["typeWidening-preview"],
    }
    assert m.with_features(proto, ["typeWidening"]) is None


# --------------------------------------------------------------- add columns


def test_add_timestamp_ntz_column_adds_feature() -> None:
    change = m.add_columns(state(), [_field("ts", "timestamp_ntz")])
    assert change.protocol is not None
    assert "timestampNtz" in change.protocol["readerFeatures"]
    assert "timestampNtz" in change.protocol["writerFeatures"]


def test_add_nested_variant_column_adds_feature() -> None:
    arr = {"type": "array", "elementType": "variant", "containsNull": True}
    change = m.add_columns(state(), [_field("v", arr)])
    assert change.protocol is not None
    assert "variantType" in change.protocol["writerFeatures"]


def test_add_column_with_parquet_invalid_name_without_cm_refused() -> None:
    with pytest.raises(UnreachableTableError, match="Parquet does not allow"):
        m.add_columns(state(), [_field("first name", "string")])


def test_add_column_with_space_allowed_under_cm() -> None:
    change = m.add_columns(cm_state(), [_field("first name", "string")])
    assert schema_of(change)["fields"][-1]["name"] == "first name"


def test_add_struct_with_duplicate_nested_names_refused() -> None:
    bad = _field("s", _struct(_field("a"), _field("A")))
    with pytest.raises(UnreachableTableError, match="appears twice"):
        m.add_columns(state(), [bad])


def test_add_column_ids_do_not_collide_when_max_id_missing() -> None:
    s = state(
        _cm("id", 1, "col-a"),
        _cm("city", 7, "col-b", "string"),
        configuration={"delta.columnMapping.mode": "name"},
        protocol=CM_PROTO,
    )
    change = m.add_columns(s, [_field("zip", "string")])
    assert schema_of(change)["fields"][-1]["metadata"]["delta.columnMapping.id"] == 8
    assert change.metadata is not None
    assert change.metadata["configuration"]["delta.columnMapping.maxColumnId"] == "8"


def test_add_no_columns_commits_nothing() -> None:
    change = m.add_columns(state(), [])
    assert change.metadata is None and change.protocol is None


def test_add_cdf_reserved_column_refused_when_cdf_on() -> None:
    s = state(configuration={"delta.enableChangeDataFeed": "true"})
    with pytest.raises(UnreachableTableError, match="reserves"):
        m.add_columns(s, [_field("_change_type", "string")])


def test_add_malformed_field_gives_clear_error() -> None:
    with pytest.raises(UnreachableTableError, match="not a Delta schema field"):
        m.add_columns(state(), [{"name": "x"}])


# ------------------------------------------------------------- drop / rename


def test_drop_nested_clustering_column_refused() -> None:
    s = cm_state(
        _cm("id", 1, "col-a"),
        _cm("addr", 2, "col-s", _struct(_cm("zip", 3, "col-z", "string"), _cm("n", 4, "col-n"))),
        clustering=[["col-s", "col-z"]],
    )
    with pytest.raises(UnreachableTableError, match="clustering column"):
        m.drop_column(s, "addr.zip")


def test_drop_last_field_of_struct_refused() -> None:
    s = cm_state(_cm("id", 1, "col-a"), _cm("addr", 2, "col-s", _struct(_cm("zip", 3, "col-z"))))
    with pytest.raises(UnreachableTableError, match="struct must keep"):
        m.drop_column(s, "addr.zip")


def test_rename_nested_with_full_new_path() -> None:
    s = cm_state(
        _cm("id", 1, "col-a"),
        _cm("addr", 2, "col-s", _struct(_cm("zip", 3, "col-z", "string"))),
    )
    change = m.rename_column(s, "addr.zip", "addr.postcode")
    assert schema_of(change)["fields"][1]["type"]["fields"][0]["name"] == "postcode"


def test_rename_nested_into_other_struct_refused() -> None:
    s = cm_state(
        _cm("id", 1, "col-a"),
        _cm("addr", 2, "col-s", _struct(_cm("zip", 3, "col-z", "string"))),
    )
    with pytest.raises(UnreachableTableError, match="keeps the column in its struct"):
        m.rename_column(s, "addr.zip", "other.zip")


def test_rename_to_empty_refused() -> None:
    with pytest.raises(UnreachableTableError, match="empty"):
        m.rename_column(cm_state(), "city", "")


def test_rename_to_cdf_reserved_refused() -> None:
    s = cm_state(configuration={"delta.enableChangeDataFeed": "true"})
    with pytest.raises(UnreachableTableError, match="reserves"):
        m.rename_column(s, "city", "_commit_version")


def test_rename_to_same_name_commits_nothing() -> None:
    assert m.rename_column(cm_state(), "city", "city").metadata is None


# ------------------------------------------------------------- type widening


def test_date_to_timestamp_ntz_adds_timestamp_ntz_feature() -> None:
    s = state(_field("d", "date"), configuration=TW)
    change = m.alter_column_type(s, "d", "timestamp_ntz")
    assert change.protocol is not None
    assert {"timestampNtz", "typeWidening"} <= set(change.protocol["writerFeatures"])


def test_widening_adds_type_widening_feature_if_protocol_lacks_it() -> None:
    change = m.alter_column_type(state(_field("n", "integer"), configuration=TW), "n", "long")
    assert change.protocol is not None
    assert "typeWidening" in change.protocol["readerFeatures"]


def test_decimal_beyond_38_refused() -> None:
    s = state(_field("n", "decimal(30,2)"), configuration=TW)
    with pytest.raises(UnreachableTableError, match="widening"):
        m.alter_column_type(s, "n", "decimal(40,2)")


@pytest.mark.parametrize(
    ("old", "given", "expected"),
    [("integer", "NUMERIC(12, 2)", "decimal(12,2)"), ("byte", "DECIMAL", "decimal(10,0)")],
)
def test_sql_decimal_aliases(old: str, given: str, expected: str) -> None:
    s = state(_field("n", old), configuration=TW)
    assert schema_of(m.alter_column_type(s, "n", given))["fields"][0]["type"] == expected


# ---------------------------------------------------------------- clustering


def test_cluster_by_duplicate_column_refused() -> None:
    with pytest.raises(UnreachableTableError, match="twice"):
        m.cluster_by(state(), ["city", "CITY"])


def test_cluster_by_too_many_columns_refused() -> None:
    s = state(*(_field(f"c{i}") for i in range(5)))
    with pytest.raises(UnreachableTableError, match="at most 4"):
        m.cluster_by(s, [f"c{i}" for i in range(5)])


def test_cluster_by_boolean_refused() -> None:
    s = state(_field("id"), _field("flag", "boolean"))
    with pytest.raises(UnreachableTableError, match="no min/max"):
        m.cluster_by(s, ["flag"])


def test_cluster_by_column_without_stats_refused() -> None:
    s = state(
        _field("id"),
        _field("city", "string"),
        configuration={"delta.dataSkippingNumIndexedCols": "1"},
    )
    with pytest.raises(UnreachableTableError, match="no statistics"):
        m.cluster_by(s, ["city"])
    listed = state(
        _field("id"),
        _field("city", "string"),
        configuration={"delta.dataSkippingStatsColumns": "city"},
    )
    assert m.cluster_by(listed, ["city"]).domains


def test_cluster_by_single_string_is_one_column() -> None:
    change = m.cluster_by(state(), "city")
    config = json.loads(change.domains[0]["domainMetadata"]["configuration"])
    assert config == {"clusteringColumns": [["city"]]}


# ---------------------------------------------------------------- properties


def _config(change: m.Change) -> dict[str, str]:
    assert change.metadata is not None
    config: dict[str, str] = change.metadata["configuration"]
    return config


def test_boolean_values_stored_lowercase() -> None:
    assert _config(m.set_properties(state(), {"delta.appendOnly": "TRUE"}))["delta.appendOnly"] == (
        "true"
    )


def test_python_bool_value_stored_as_true() -> None:
    change = m.set_properties(state(), {"delta.enableChangeDataFeed": True})
    assert _config(change)["delta.enableChangeDataFeed"] == "true"


def test_delta_keys_are_canonicalised() -> None:
    change = m.set_properties(state(), {"delta.ENABLECHANGEDATAFEED": "true"})
    assert _config(change)["delta.enableChangeDataFeed"] == "true"
    assert "delta.ENABLECHANGEDATAFEED" not in _config(change)
    assert change.protocol is not None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("delta.checkpointInterval", "0"),
        ("delta.checkpointInterval", "ten"),
        ("delta.dataSkippingNumIndexedCols", "-2"),
        ("delta.logRetentionDuration", "30 fortnights"),
        ("delta.isolationLevel", "ReadCommitted"),
        ("delta.parquet.compression.codec", "rar"),
        ("delta.targetFileSize", "big"),
    ],
)
def test_bad_values_refused(key: str, value: str) -> None:
    with pytest.raises(UnreachableTableError, match=key.replace(".", r"\.")):
        m.set_properties(state(), {key: value})


def test_duration_gets_interval_prefix() -> None:
    change = m.set_properties(state(), {"delta.logRetentionDuration": "30 days"})
    assert _config(change)["delta.logRetentionDuration"] == "interval 30 days"


def test_isolation_level_canonical_spelling() -> None:
    change = m.set_properties(state(), {"delta.isolationLevel": "serializable"})
    assert _config(change)["delta.isolationLevel"] == "Serializable"


def test_stats_columns_must_exist() -> None:
    with pytest.raises(UnreachableTableError, match="no column 'nope'"):
        m.set_properties(state(), {"delta.dataSkippingStatsColumns": "id,nope"})


def test_none_value_refused() -> None:
    with pytest.raises(UnreachableTableError, match="no value"):
        m.set_properties(state(), {"owner": None})


def test_column_mapping_mode_stored_lowercase() -> None:
    change = m.set_properties(state(), {"delta.columnMapping.mode": "Name"})
    assert _config(change)["delta.columnMapping.mode"] == "name"


def test_checkpoint_policy_stored_lowercase() -> None:
    change = m.set_properties(state(), {"delta.checkpointPolicy": "V2"})
    assert _config(change)["delta.checkpointPolicy"] == "v2"


def test_enabling_cdf_with_reserved_column_refused() -> None:
    s = state(_field("id"), _field("_change_type", "string"))
    with pytest.raises(UnreachableTableError, match="reserves"):
        m.set_properties(s, {"delta.enableChangeDataFeed": "true"})


def test_setting_same_value_commits_nothing() -> None:
    s = state(configuration={"owner": "a"})
    assert m.set_properties(s, {"owner": "a"}).metadata is None


def test_feature_signal_is_protocol_only() -> None:
    change = m.set_properties(state(), {"delta.feature.deletionVectors": "supported"})
    assert change.protocol is not None and change.metadata is None


def test_disabling_row_tracking_is_accepted() -> None:
    change = m.set_properties(state(), {"delta.enableRowTracking": "false"})
    assert _config(change)["delta.enableRowTracking"] == "false"


def test_unset_with_generator_records_keys() -> None:
    s = state(configuration={"a": "1", "b": "2"})
    change = m.unset_properties(s, (k for k in ["a", "b"]))
    assert json.loads(change.parameters["properties"]) == ["a", "b"]
    assert _config(change) == {}


def test_unset_nothing_commits_nothing() -> None:
    assert m.unset_properties(state(), ["missing"]).metadata is None


def test_unset_is_case_insensitive_for_delta_keys() -> None:
    s = state(configuration={"delta.appendOnly": "true"})
    assert _config(m.unset_properties(s, ["delta.appendonly"])) == {}


def test_unchanged_comment_commits_nothing() -> None:
    # The kernel serialises an absent description as null.
    s = state(extra={"description": None, "name": None})
    assert m.set_comment(s, None).metadata is None


def test_unchanged_nullability_commits_nothing() -> None:
    assert m.set_nullability(state(), "city", True).metadata is None


def test_nullability_needs_a_bool() -> None:
    with pytest.raises(UnreachableTableError, match="True or False"):
        m.set_nullability(state(), "city", "false")  # type: ignore[arg-type]


# ----------------------------------------------------------------- rendering


def test_build_actions_refuses_catalog_managed_table() -> None:
    s = state(
        protocol={
            "minReaderVersion": 3,
            "minWriterVersion": 7,
            "readerFeatures": ["catalogManaged"],
            "writerFeatures": ["catalogManaged", "inCommitTimestamp"],
        }
    )
    with pytest.raises(UnreachableTableError, match="catalog-managed"):
        m.build_actions(s, m.set_comment(s, "x"), engine_info="t")


# ---------------------------------------------------------------- arrow


def test_arrow_dictionary_and_fixed_size_list() -> None:
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema(
        [
            ("d", pa.dictionary(pa.int32(), pa.string())),
            ("f", pa.list_(pa.int64(), 3)),
        ]
    )
    types = [f["type"] for f in m.arrow_to_delta_schema(schema)["fields"]]
    assert types[0] == "string"
    assert types[1]["type"] == "array" and types[1]["elementType"] == "long"


def test_arrow_decimal256_beyond_38_refused() -> None:
    pa = pytest.importorskip("pyarrow")
    with pytest.raises(UnreachableTableError, match="decimal range"):
        m.arrow_to_delta_type(pa.decimal256(50, 2))


def test_arrow_column_mapping_metadata_is_not_carried() -> None:
    pa = pytest.importorskip("pyarrow")
    f = pa.field(
        "x",
        pa.int64(),
        metadata={"delta.columnMapping.id": "5", "comment": "keep", "PARQUET:field_id": "5"},
    )
    assert m.arrow_to_delta_field(f)["metadata"] == {"comment": "keep"}


# ---------------------------------------------------------------- version 0


def _v0(**kw: Any) -> list[dict[str, Any]]:
    args: dict[str, Any] = {
        "table_id": "tid",
        "schema": _struct(_field("id"), _field("city", "string")),
        "required_protocol": {},
        "configuration": {},
        "engine_info": "t",
        "now_ms": 42,
    }
    args.update(kw)
    return [json.loads(a) for a in m.initial_actions(**args)]


def _part(actions: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    found: dict[str, Any] = next(a[kind] for a in actions if kind in a)
    return found


def test_v0_timestamp_ntz_column_adds_feature() -> None:
    actions = _v0(schema=_struct(_field("ts", "timestamp_ntz")))
    assert "timestampNtz" in _part(actions, "protocol")["readerFeatures"]


def test_v0_keeps_unknown_required_reader_features() -> None:
    actions = _v0(
        required_protocol={"reader-features": ["futureThing"], "writer-features": ["futureThing"]}
    )
    proto = _part(actions, "protocol")
    assert "futureThing" in proto["readerFeatures"]
    assert "futureThing" in proto["writerFeatures"]


def test_v0_catalog_managed_enables_ict_and_stamps_commit() -> None:
    actions = _v0(configuration={"delta.feature.catalogManaged": "supported"})
    assert actions[0]["commitInfo"]["inCommitTimestamp"] == 42
    config = _part(actions, "metaData")["configuration"]
    assert config["delta.enableInCommitTimestamps"] == "true"
    assert "inCommitTimestamp" in _part(actions, "protocol")["writerFeatures"]


def test_v0_config_values_are_strings() -> None:
    actions = _v0(configuration={"delta.appendOnly": True, "delta.checkpointInterval": 10})
    config = _part(actions, "metaData")["configuration"]
    assert config["delta.appendOnly"] == "true"
    assert config["delta.checkpointInterval"] == "10"
    with pytest.raises(UnreachableTableError, match="no value"):
        _v0(configuration={"x": None})


def test_v0_omits_null_name_and_description() -> None:
    meta = _part(_v0(), "metaData")
    assert "name" not in meta and "description" not in meta
    assert _part(_v0(description="d"), "metaData")["description"] == "d"


def test_v0_partition_columns_validated_and_canonicalised() -> None:
    assert _part(_v0(partition_columns=["CITY"]), "metaData")["partitionColumns"] == ["city"]
    with pytest.raises(UnreachableTableError, match="not in the schema"):
        _v0(partition_columns=["nope"])
    nested = _struct(_field("id"), _field("s", _struct(_field("a"))))
    with pytest.raises(UnreachableTableError, match="primitive"):
        _v0(schema=nested, partition_columns=["s"])


def test_v0_row_tracking_gets_materialized_column_names() -> None:
    config = _part(_v0(configuration={"delta.enableRowTracking": "true"}), "metaData")[
        "configuration"
    ]
    assert config["delta.rowTracking.materializedRowIdColumnName"].startswith("_row-id-col-")
    assert "delta.rowTracking.materializedRowCommitVersionColumnName" in config


def test_v0_invalid_column_name_without_cm_refused() -> None:
    with pytest.raises(UnreachableTableError, match="Parquet does not allow"):
        _v0(schema=_struct(_field("a,b")))


def test_v0_cluster_by_honours_stats_settings() -> None:
    with pytest.raises(UnreachableTableError, match="no statistics"):
        _v0(cluster_by=["city"], configuration={"delta.dataSkippingNumIndexedCols": "1"})
