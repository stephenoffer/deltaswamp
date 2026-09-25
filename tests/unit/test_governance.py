"""Catalog-neutral governance helpers: schema conversion, staging options, privileges."""

from __future__ import annotations

import json

import pytest
from deltaswamp.errors import DeltaSwampError
from deltaswamp.governance import (
    delta_schema_to_columns,
    normalize_privilege,
    staging_storage_options,
)

from tests.helpers import DELTA_SCHEMA


class TestSchemaToColumns:
    def test_spark_spelling_and_partitions(self) -> None:
        cols = delta_schema_to_columns(json.dumps(DELTA_SCHEMA), ["region"])
        assert [(c["name"], c["type_text"], c["type_name"]) for c in cols] == [
            ("id", "bigint", "LONG"),
            ("amount", "decimal(10,2)", "DECIMAL"),
            ("tags", "array<string>", "ARRAY"),
            ("region", "string", "STRING"),
        ]
        assert cols[0]["nullable"] is False and cols[0]["comment"] == "key"
        assert (cols[1]["type_precision"], cols[1]["type_scale"]) == (10, 2)
        assert cols[3]["partition_index"] == 0
        assert "partition_index" not in cols[0]
        assert json.loads(cols[2]["type_json"])["type"]["elementType"] == "string"

    def test_nested_struct_and_map(self) -> None:
        schema = {
            "type": "struct",
            "fields": [
                {
                    "name": "m",
                    "type": {
                        "type": "map",
                        "keyType": "string",
                        "valueType": {
                            "type": "struct",
                            "fields": [{"name": "x", "type": "integer", "nullable": True}],
                        },
                        "valueContainsNull": True,
                    },
                    "nullable": True,
                }
            ],
        }
        (col,) = delta_schema_to_columns(schema)
        assert col["type_text"] == "map<string,struct<x:int>>"

    def test_unknown_partition_column(self) -> None:
        with pytest.raises(DeltaSwampError, match="not in the schema"):
            delta_schema_to_columns(DELTA_SCHEMA, ["nope"])


class TestStagingOptions:
    def test_azure_and_gcs(self) -> None:
        azure, _ = staging_storage_options(
            "abfss://c@acct.dfs.core.windows.net/t",
            [
                {
                    "prefix": "abfss://c@acct.dfs.core.windows.net/t",
                    "operation": "READ_WRITE",
                    "config": {"azure.sas-token": "sv=x"},
                }
            ],
        )
        assert azure == {
            "azure_storage_sas_key": "sv=x",
            "azure_endpoint": "https://acct.dfs.core.windows.net",
        }
        gcs, expires = staging_storage_options(
            "gs://b/t",
            [
                {
                    "prefix": "gs://b/t/",
                    "operation": "READ_WRITE",
                    "config": {"gcs.oauth-token": "y"},
                }
            ],
        )
        assert gcs == {"google_bearer_token": "y"} and expires is None

    def test_local_storage_has_no_options(self) -> None:
        assert staging_storage_options("file:///tmp/t", []) == ({}, None)


def test_privilege_spellings_converge() -> None:
    uc = pytest.importorskip("databricks.sdk.service.catalog")
    assert normalize_privilege("use schema") == "USE_SCHEMA"
    assert normalize_privilege(uc.Privilege.EXTERNAL_USE_SCHEMA) == "EXTERNAL_USE_SCHEMA"
    assert normalize_privilege("ALL-PRIVILEGES") == "ALL_PRIVILEGES"
