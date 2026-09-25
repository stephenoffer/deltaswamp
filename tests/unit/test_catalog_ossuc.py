"""The open-source Unity Catalog client, against `tests/fake_uc.py`.

The fake speaks the REST protocol over a real socket, so requests are checked
as they go over the wire.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from deltaswamp.catalog.ossuc import OSSUnityCatalog
from deltaswamp.errors import InvalidReferenceError, PreflightError
from deltaswamp.governance import Grant, StagingTable
from deltaswamp.identity import parse_ref

from tests.fake_uc import FakeTable, FakeUnityCatalog
from tests.helpers import DELTA_SCHEMA

REF = parse_ref("main.sales.orders")


@pytest.fixture
def fake(tmp_path: Path) -> Iterator[FakeUnityCatalog]:
    with FakeUnityCatalog(staging_root=tmp_path / "staging") as server:
        server.add_table(
            "main.sales.orders",
            FakeTable(
                name="orders",
                location="s3://bucket/orders",
                table_id="tid-9",
                owner="alice",
                comment="all orders",
                columns=[
                    {"name": "id", "type_text": "bigint", "type_name": "LONG", "position": 0},
                    {
                        "name": "region",
                        "type_text": "string",
                        "type_name": "STRING",
                        "position": 1,
                        "partition_index": 0,
                    },
                ],
                created_at=1700000000000,
            ),
        )
        yield server


@pytest.fixture
def oss(fake: FakeUnityCatalog) -> OSSUnityCatalog:
    return OSSUnityCatalog(fake.url, token="tok")


class TestOSSGovernance:
    def test_table_info(self, oss: OSSUnityCatalog) -> None:
        info = oss.table_info(REF)
        assert (info.full_name, info.owner, info.comment) == (
            "main.sales.orders",
            "alice",
            "all orders",
        )
        assert info.partition_columns == ("region",)
        assert info.created_at == 1700000000000

    def test_grant_revoke_round_trip_uses_space_spelling(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        after = oss.grant(REF, "analysts", ["SELECT", "USE_SCHEMA"])
        assert after == [Grant("analysts", ("SELECT", "USE_SCHEMA"))]
        method, path, body = fake.requests[-1]
        assert method == "PATCH"
        assert path == "/api/2.1/unity-catalog/permissions/table/main.sales.orders"
        assert body == {"changes": [{"principal": "analysts", "add": ["SELECT", "USE SCHEMA"]}]}

        oss.revoke(REF, "analysts", ["use schema"])
        assert oss.grants(REF) == [Grant("analysts", ("SELECT",))]
        assert oss.grants(REF, principal="nobody") == []
        oss.grant("main", "analysts", ["USE_CATALOG"], securable_type="CATALOG")
        assert oss.grants("main", securable_type="catalog") == [Grant("analysts", ("USE_CATALOG",))]

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("effective_grants", (REF,)),
            ("tags", (REF,)),
            ("set_tags", (REF, {"a": "b"})),
            ("unset_tags", (REF, ["a"])),
            ("set_owner", (REF, "bob")),
            ("lineage", (REF,)),
            ("column_lineage", (REF, "id")),
            ("add_primary_key", (REF, "pk", ["id"])),
            ("add_foreign_key", (REF, "fk", ["id"], REF, ["id"])),
            ("drop_table_constraint", (REF, "pk")),
            ("volume", ("main.sales.v",)),
        ],
    )
    def test_unsupported_says_what_oss_lacks(
        self, oss: OSSUnityCatalog, method: str, args: tuple[Any, ...]
    ) -> None:
        assert method in oss.unsupported_operations
        with pytest.raises(NotImplementedError, match="OSS Unity Catalog"):
            getattr(oss, method)(*args)


class TestOSSNamespaces:
    def test_catalog_and_schema_lifecycle(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        oss.create_catalog("sandbox", comment="play")
        oss.create_schema("sandbox", "s1")
        assert "sandbox" in oss.list_catalogs()
        assert oss.list_schemas("sandbox") == ["s1"]
        assert fake.requests[0][2] == {"name": "sandbox", "comment": "play"}

        with pytest.raises(PreflightError, match="not empty"):
            oss.drop_catalog("sandbox")
        oss.drop_schema("sandbox", "s1")
        oss.drop_catalog("sandbox")
        assert "sandbox" not in oss.list_catalogs()

    def test_force_drop(self, oss: OSSUnityCatalog) -> None:
        oss.drop_schema("main", "sales", force=True)
        assert oss.list_tables("main", "sales") == []

    def test_missing_is_an_invalid_reference(self, oss: OSSUnityCatalog) -> None:
        with pytest.raises(InvalidReferenceError):
            oss.drop_catalog("nope")

    def test_table_exists(self, oss: OSSUnityCatalog) -> None:
        assert oss.table_exists(REF) is True
        assert oss.table_exists(parse_ref("main.sales.nope")) is False

    def test_search_tables_emulates_like(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        fake.add_table(
            "main.hr.people", FakeTable(name="people", location="s3://b/p", table_id="x")
        )
        assert sorted(t.full_name for t in oss.search_tables("main")) == [
            "main.hr.people",
            "main.sales.orders",
        ]
        assert [t.full_name for t in oss.search_tables("main", "sa%")] == ["main.sales.orders"]
        assert [t.full_name for t in oss.search_tables("main", None, "p_ople")] == [
            "main.hr.people"
        ]

    def test_functions_and_volumes(self, oss: OSSUnityCatalog, fake: FakeUnityCatalog) -> None:
        fake.add_function("main.sales.f", data_type="INT", comment="adds")
        assert [(f.full_name, f.data_type) for f in oss.list_functions("main", "sales")] == [
            ("main.sales.f", "INT")
        ]
        created = oss.create_volume(
            "main", "sales", "landing", volume_type="EXTERNAL", storage_location="s3://b/v"
        )
        assert created.full_name == "main.sales.landing" and created.volume_type == "EXTERNAL"
        assert [v.full_name for v in oss.list_volumes("main", "sales")] == ["main.sales.landing"]
        oss.drop_volume("main", "sales", "landing")
        assert oss.list_volumes("main", "sales") == []


class TestOSSLifecycle:
    def test_register_external_then_resolve(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        ref = parse_ref("main.sales.events")
        resolved = oss.register_table(
            ref,
            "s3://bucket/events",
            columns_schema_json=DELTA_SCHEMA,
            partition_columns=["region"],
            properties={"k": "v"},
            comment="events",
        )
        _, path, body = next(r for r in fake.requests if r[0] == "POST")
        assert path == "/api/2.1/unity-catalog/tables"
        assert body["table_type"] == "EXTERNAL" and body["data_source_format"] == "DELTA"
        assert body["storage_location"] == "s3://bucket/events"
        assert body["comment"] == "events"
        assert body["columns"][3]["partition_index"] == 0
        assert resolved.location == "s3://bucket/events"
        assert resolved.table_id == fake.tables["main.sales.events"].table_id
        assert oss.table_info(ref).comment == "events"

    def test_register_twice_is_refused(self, oss: OSSUnityCatalog) -> None:
        with pytest.raises(PreflightError, match="409"):
            oss.register_table(REF, "s3://bucket/orders")

    def test_path_credentials(self, oss: OSSUnityCatalog, fake: FakeUnityCatalog) -> None:
        creds = oss.path_credentials("s3://bucket/new", "PATH_CREATE_TABLE")
        assert fake.requests[-1][1:] == (
            "/api/2.1/unity-catalog/temporary-path-credentials",
            {"url": "s3://bucket/new", "operation": "PATH_CREATE_TABLE"},
        )
        assert creds.url == "s3://bucket/new"
        assert creds.as_storage_options()["aws_secret_access_key"] == "path-secret"
        assert "path-secret" not in repr(creds)

    def _write_v0(self, staged: StagingTable) -> Path:
        from urllib.parse import urlparse

        root = Path(urlparse(staged.location).path)
        (root / "_delta_log").mkdir(parents=True)
        (root / "_delta_log" / f"{0:020d}.json").write_text("{}\n")
        return root

    def _request_body(self, staged: StagingTable, **overrides: Any) -> dict[str, Any]:
        properties = {
            k: v
            for k, v in staged.required_properties.items()
            if v is not None and not k.startswith("delta.feature.")
        }
        body: dict[str, Any] = {
            "name": "fresh",
            "location": staged.location,
            "table-type": "MANAGED",
            "columns": {"type": "struct", "fields": [{"name": "id", "type": "long"}]},
            "partition-columns": [],
            "protocol": {
                "min-reader-version": 3,
                "min-writer-version": 7,
                "reader-features": ["catalogManaged"],
                "writer-features": ["catalogManaged", "inCommitTimestamp"],
            },
            "properties": properties,
            "last-commit-timestamp-ms": 1700000000000,
        }
        body.update(overrides)
        return body

    def test_staging_then_finalize(self, oss: OSSUnityCatalog, fake: FakeUnityCatalog) -> None:
        ref = parse_ref("main.sales.fresh")
        staged = oss.create_staging_table(ref)
        assert fake.requests[-1][1] == (
            "/api/2.1/unity-catalog/delta/v1/catalogs/main/schemas/sales/staging-tables"
        )
        assert staged.location.startswith("file://")
        assert staged.storage_options == {}
        assert staged.required_properties["io.unitycatalog.tableId"] == staged.table_id
        assert staged.required_properties["delta.feature.catalogManaged"] == "supported"
        assert not oss.table_exists(ref)  # staged, not registered

        self._write_v0(staged)
        resolved = oss.finalize_managed_table(ref, self._request_body(staged))

        assert fake.requests[-3][1] == (
            "/api/2.1/unity-catalog/delta/v1/catalogs/main/schemas/sales/tables"
        )
        assert resolved.table_type is not None and resolved.table_type.value == "MANAGED"
        assert resolved.table_id == staged.table_id
        assert resolved.is_catalog_managed
        assert resolved.max_catalog_version == 0
        assert resolved.location == staged.location

    def test_finalize_without_v0_is_refused(self, oss: OSSUnityCatalog) -> None:
        ref = parse_ref("main.sales.fresh")
        staged = oss.create_staging_table(ref)
        with pytest.raises(PreflightError, match="version 0"):
            oss.finalize_managed_table(ref, self._request_body(staged))

    def test_finalize_without_the_feature_is_refused(self, oss: OSSUnityCatalog) -> None:
        ref = parse_ref("main.sales.fresh")
        staged = oss.create_staging_table(ref)
        self._write_v0(staged)
        body = self._request_body(
            staged, protocol={"min-reader-version": 1, "min-writer-version": 2}
        )
        with pytest.raises(PreflightError, match="catalogManaged"):
            oss.finalize_managed_table(ref, body)

    def test_staging_maps_cloud_credentials(
        self, oss: OSSUnityCatalog, fake: FakeUnityCatalog
    ) -> None:
        fake.staging_credential_config = {
            "s3.access-key-id": "AKIA",
            "s3.secret-access-key": "SUPERSECRET",
            "s3.session-token": "TOKEN",
        }
        staged = oss.create_staging_table(parse_ref("main.sales.fresh"))
        assert staged.storage_options["aws_secret_access_key"] == "SUPERSECRET"
        assert staged.expires_at == 9999999999.0
        assert "SUPERSECRET" not in repr(staged)

    def test_staging_into_a_missing_schema(self, oss: OSSUnityCatalog) -> None:
        with pytest.raises(InvalidReferenceError):
            oss.create_staging_table(parse_ref("main.nope.t"))
