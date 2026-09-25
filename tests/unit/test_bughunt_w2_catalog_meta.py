"""Second-pass regression tests: metadata commits, catalog base, OSS UC, router.

No network: the OSS UC tests replace `urllib.request.urlopen` inside the ossuc
module with a scripted fake.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from email.message import Message
from typing import Any

import pytest
from deltaswamp.capability import Capability, Operation
from deltaswamp.capability import Engine as EngineKind
from deltaswamp.catalog.base import ResolvedTable, parse_commit_tail
from deltaswamp.catalog.ossuc import OSSUnityCatalog
from deltaswamp.engine import metadata as m
from deltaswamp.errors import (
    CorruptTableError,
    FallbackRequiredError,
    InvalidReferenceError,
    PreflightError,
    UnreachableTableError,
)
from deltaswamp.identity import RefKind, TableRef, parse_ref
from deltaswamp.router import Router

# ------------------------------------------------------------------ helpers


def _field(name: str, dtype: Any = "long", nullable: bool = True, /, **meta: Any) -> dict[str, Any]:
    return {"name": name, "type": dtype, "nullable": nullable, "metadata": dict(meta)}


def state(
    *fields: dict[str, Any],
    configuration: dict[str, str] | None = None,
    protocol: dict[str, Any] | None = None,
    partition_columns: list[str] | None = None,
    clustering: list[list[str]] | None = None,
) -> m.TableState:
    return m.TableState(
        version=3,
        protocol=protocol or {"minReaderVersion": 1, "minWriterVersion": 2},
        metadata={
            "id": "abc",
            "format": {"provider": "parquet", "options": {}},
            "schemaString": json.dumps(
                {
                    "type": "struct",
                    "fields": list(fields or (_field("id"), _field("city", "string"))),
                }
            ),
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


def v0(**kwargs: Any) -> list[dict[str, Any]]:
    args: dict[str, Any] = {
        "table_id": "tid",
        "schema": {"type": "struct", "fields": [_field("id"), _field("v", "string")]},
        "required_protocol": {},
        "configuration": {},
        "engine_info": "test",
        "now_ms": 1_000,
    }
    args.update(kwargs)
    return [json.loads(line) for line in m.initial_actions(**args)]


def _action(actions: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    return next(a[kind] for a in actions if kind in a)


TW = {"delta.enableTypeWidening": "true"}
CLUSTERED = {
    "minReaderVersion": 1,
    "minWriterVersion": 7,
    "writerFeatures": ["clustering", "domainMetadata"],
}


# --------------------------------------------------------------- metadata: v0


def test_v0_catalog_managed_refuses_ict_disabled() -> None:
    with pytest.raises(UnreachableTableError, match="in-commit timestamps"):
        v0(
            required_protocol={
                "writer-features": ["catalogManaged"],
                "reader-features": ["catalogManaged"],
            },
            configuration={"delta.enableInCommitTimestamps": "False"},
        )


def test_v0_catalog_managed_still_enables_ict_by_default() -> None:
    actions = v0(
        required_protocol={
            "writer-features": ["catalogManaged"],
            "reader-features": ["catalogManaged"],
        },
    )
    config = _action(actions, "metaData")["configuration"]
    assert config["delta.enableInCommitTimestamps"] == "true"
    assert "inCommitTimestamp" in _action(actions, "protocol")["writerFeatures"]


def test_v0_delta_keys_canonicalised_and_values_normalised() -> None:
    actions = v0(
        configuration={
            "delta.enablechangedatafeed": "TRUE",
            "delta.logRetentionDuration": "30 days",
            "delta.columnMapping.mode": "Name",
        }
    )
    config = _action(actions, "metaData")["configuration"]
    assert config["delta.enableChangeDataFeed"] == "true"
    assert "delta.enablechangedatafeed" not in config
    assert config["delta.logRetentionDuration"] == "interval 30 days"
    assert config["delta.columnMapping.mode"] == "name"
    assert "changeDataFeed" in _action(actions, "protocol")["writerFeatures"]


def test_v0_bad_property_value_refused() -> None:
    with pytest.raises(UnreachableTableError, match="checkpointInterval"):
        v0(configuration={"delta.checkpointInterval": "0"})


def test_v0_bad_column_mapping_mode_refused() -> None:
    with pytest.raises(UnreachableTableError, match="none, name or id"):
        v0(configuration={"delta.columnMapping.mode": "names"})


def test_v0_cdf_with_reserved_column_refused() -> None:
    with pytest.raises(UnreachableTableError, match="reserves"):
        v0(
            schema={"type": "struct", "fields": [_field("id"), _field("_change_type", "string")]},
            configuration={"delta.enableChangeDataFeed": "true"},
        )


def test_v0_feature_signal_needs_supported() -> None:
    with pytest.raises(UnreachableTableError, match="only accepts 'supported'"):
        v0(configuration={"delta.feature.deletionVectors": "disabled"})
    actions = v0(configuration={"delta.feature.DeletionVectors": "supported"})
    assert "deletionVectors" in _action(actions, "protocol")["writerFeatures"]


def test_v0_invalid_schema_type_refused() -> None:
    with pytest.raises(UnreachableTableError, match="not a Delta type"):
        v0(schema={"type": "struct", "fields": [_field("id", "bigint")]})


# ------------------------------------------------------ metadata: add columns


@pytest.mark.parametrize("bad", ["int", "bigint", "int64", "varchar", "Long", "decimal(40,2)"])
def test_add_column_invalid_type_refused(bad: str) -> None:
    with pytest.raises(UnreachableTableError):
        m.add_columns(state(), [_field("x", bad)])


def test_add_column_invalid_nested_type_refused() -> None:
    nested = {"type": "array", "elementType": {"type": "struct", "fields": [_field("a", "int")]}}
    with pytest.raises(UnreachableTableError, match="'integer'"):
        m.add_columns(state(), [_field("x", nested)])


def test_add_column_array_and_map_flags_defaulted() -> None:
    change = m.add_columns(
        state(),
        [
            _field("arr", {"type": "array", "elementType": "long"}),
            _field("mp", {"type": "map", "keyType": "string", "valueType": "long"}),
            _field("d", "decimal(10, 2)"),
        ],
    )
    fields = {f["name"]: f for f in schema_of(change)["fields"]}
    assert fields["arr"]["type"]["containsNull"] is True
    assert fields["mp"]["type"]["valueContainsNull"] is True
    assert fields["d"]["type"] == "decimal(10,2)"


def test_add_generated_or_identity_column_refused() -> None:
    with pytest.raises(UnreachableTableError, match="generated and identity"):
        m.add_columns(state(), [_field("g", "long", **{"delta.generationExpression": "id + 1"})])
    with pytest.raises(UnreachableTableError, match="generated and identity"):
        m.add_columns(state(), [_field("i", "long", **{"delta.identity.start": 1})])


def test_add_column_default_needs_allow_column_defaults() -> None:
    with pytest.raises(UnreachableTableError, match="allowColumnDefaults"):
        m.add_columns(state(), [_field("x", "long", CURRENT_DEFAULT="0")])
    proto = {
        "minReaderVersion": 1,
        "minWriterVersion": 7,
        "writerFeatures": ["allowColumnDefaults"],
    }
    m.add_columns(state(protocol=proto), [_field("x", "long", CURRENT_DEFAULT="0")])


def test_add_column_strips_foreign_column_mapping_metadata() -> None:
    change = m.add_columns(
        state(),
        [
            _field(
                "x",
                "long",
                **{"delta.columnMapping.id": 7, "delta.columnMapping.physicalName": "c"},
            )
        ],
    )
    added = schema_of(change)["fields"][-1]
    assert not any(k.startswith("delta.columnMapping.") for k in added["metadata"])


# ---------------------------------------------------- metadata: type widening


@pytest.mark.parametrize("old", ["byte", "short"])
def test_small_int_to_narrow_decimal_is_not_a_widening(old: str) -> None:
    with pytest.raises(UnreachableTableError, match="only widening"):
        m.alter_column_type(state(_field("a", old), configuration=TW), "a", "decimal(5,0)")
    change = m.alter_column_type(state(_field("a", old), configuration=TW), "a", "decimal(10,0)")
    assert schema_of(change)["fields"][0]["type"] == "decimal(10,0)"


def test_decimal_with_precision_only_means_scale_zero() -> None:
    change = m.alter_column_type(
        state(_field("a", "integer"), configuration=TW), "a", "DECIMAL(12)"
    )
    assert schema_of(change)["fields"][0]["type"] == "decimal(12,0)"


def test_type_change_under_constraint_or_generated_column_refused() -> None:
    constrained = state(
        _field("a", "integer"), configuration={**TW, "delta.constraints.pos": "a > 0"}
    )
    with pytest.raises(UnreachableTableError, match="CHECK constraint"):
        m.alter_column_type(constrained, "a", "long")
    generated = state(
        _field("a", "integer"),
        _field("b", "integer", **{"delta.generationExpression": "a * 2"}),
        configuration=TW,
    )
    with pytest.raises(UnreachableTableError, match="generated column"):
        m.alter_column_type(generated, "a", "long")


# ------------------------------------------------------- metadata: clustering


def test_cluster_by_none_still_clears_existing_keys() -> None:
    change = m.cluster_by(state(protocol=CLUSTERED, clustering=[["id"]]), None)
    config = json.loads(change.domains[0]["domainMetadata"]["configuration"])
    assert config == {"clusteringColumns": []}


def test_cluster_by_unchanged_keys_commits_nothing() -> None:
    change = m.cluster_by(state(protocol=CLUSTERED, clustering=[["id"]]), ["id"])
    assert change.protocol is None and not change.domains


def test_stats_setting_that_strands_a_clustering_key_refused() -> None:
    clustered = state(
        _field("a"), _field("b"), _field("id"), protocol=CLUSTERED, clustering=[["id"]]
    )
    with pytest.raises(UnreachableTableError, match="clustering column id"):
        m.set_properties(clustered, {"delta.dataSkippingNumIndexedCols": "2"})
    with pytest.raises(UnreachableTableError, match="clustering column id"):
        m.set_properties(clustered, {"delta.dataSkippingStatsColumns": "a,b"})
    # Keeping the key covered is fine.
    m.set_properties(clustered, {"delta.dataSkippingStatsColumns": "id"})


def test_unset_stats_columns_that_strands_a_clustering_key_refused() -> None:
    fields = [_field(f"c{i}") for i in range(33)] + [_field("id")]
    clustered = state(
        *fields,
        protocol=CLUSTERED,
        clustering=[["id"]],
        configuration={"delta.dataSkippingStatsColumns": "id"},
    )
    with pytest.raises(UnreachableTableError, match="clustering column id"):
        m.unset_properties(clustered, ["delta.dataSkippingStatsColumns"])


# --------------------------------------------------- metadata: other commits


def test_feature_signal_is_case_insensitive() -> None:
    change = m.set_properties(state(), {"delta.feature.deletionvectors": "supported"})
    assert change.protocol is not None
    assert "deletionVectors" in change.protocol["writerFeatures"]


def test_unsetting_row_tracking_column_names_refused() -> None:
    st = state(configuration={"delta.rowTracking.materializedRowIdColumnName": "_row-id-col-x"})
    with pytest.raises(UnreachableTableError, match="row tracking"):
        m.unset_properties(st, ["delta.rowTracking.materializedRowIdColumnName"])


def test_drop_last_non_partition_column_refused() -> None:
    cm = {"delta.columnMapping.mode": "name", "delta.columnMapping.maxColumnId": "2"}
    st = state(
        _field(
            "p", "string", **{"delta.columnMapping.id": 1, "delta.columnMapping.physicalName": "c1"}
        ),
        _field(
            "v", "long", **{"delta.columnMapping.id": 2, "delta.columnMapping.physicalName": "c2"}
        ),
        configuration=cm,
        protocol={"minReaderVersion": 2, "minWriterVersion": 5},
        partition_columns=["p"],
    )
    with pytest.raises(UnreachableTableError, match="last non-partition column"):
        m.drop_column(st, "v")


# ------------------------------------------------------------- catalog base


def test_commit_without_file_name_is_corrupt() -> None:
    with pytest.raises(CorruptTableError, match="file name"):
        parse_commit_tail({"commits": [{"version": 1}], "latest-table-version": 1})


def test_version_zero_commit_is_not_missing() -> None:
    entries, _, _ = parse_commit_tail(
        {"commits": [{"version": 0, "file-name": "0.json"}], "latest-table-version": 0}
    )
    assert entries[0].version == 0


def test_latest_version_derived_from_tail_when_absent() -> None:
    body = {
        "commits": [
            {"version": 5, "file-name": "5.u.json"},
            {"version": 4, "file-name": "4.u.json"},
        ]
    }
    _, latest, _ = parse_commit_tail(body)
    assert latest == 5


def test_iceberg_compat_detected_from_catalog_properties() -> None:
    table = ResolvedTable(
        ref=parse_ref("/tmp/t"),
        location="/tmp/t",
        properties={"delta.enableIcebergCompatV2": "true"},
    )
    assert table.has_iceberg_compat
    table = ResolvedTable(
        ref=parse_ref("/tmp/t"),
        location="/tmp/t",
        properties={"delta.universalFormat.enabledFormats": "ICEBERG"},
    )
    assert table.has_iceberg_compat


# ------------------------------------------------------------------- router


class _Engine:
    def __init__(self, ok: bool) -> None:
        self.ok = ok

    def supports(self, operation: Operation, table: Any, **shape: Any) -> Capability:
        return Capability(operation, ok=self.ok, engine=EngineKind.DELTARS, reason="nope")


def test_fallback_disabled_is_the_reason_not_engine_not_configured() -> None:
    router = Router(engines={EngineKind.DELTARS: _Engine(False)}, allow_sql_fallback=False)
    # A named Unity Catalog table: the warehouse could serve it. (A path table
    # it cannot, and there the reason says so instead -- see test_bughunt_w3.)
    table = ResolvedTable(ref=parse_ref("main.s.t"), location="/tmp/t")
    verdict = router.capability(Operation.OPTIMIZE, table)
    assert "fallback is disabled" in verdict.reason
    assert "sql: engine not configured" not in verdict.reason
    with pytest.raises(FallbackRequiredError):
        router.engine_for(Operation.OPTIMIZE, table)


# ------------------------------------------------------------------- OSS UC

BASE = "http://uc.test"
UC = "/api/2.1/unity-catalog"


def ref(c: str = "main", s: str = "sch", t: str = "tbl") -> TableRef:
    return TableRef(kind=RefKind.CATALOG, catalog=c, schema=s, table=t, raw=f"{c}.{s}.{t}")


class _Resp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class FakeServer:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[dict[str, str], Any], Any]] = {}
        self.requests: list[Any] = []

    def on(self, method: str, path: str, handler: Callable[[dict[str, str], Any], Any]) -> None:
        self.routes[(method, path)] = handler

    def urlopen(self, request: Any, timeout: float | None = None) -> _Resp:
        self.requests.append(request)
        parsed = urllib.parse.urlsplit(request.full_url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        body = json.loads(request.data) if request.data else None
        handler = self.routes.get((request.get_method(), urllib.parse.unquote(parsed.path)))
        if handler is None:
            raise urllib.error.HTTPError(
                request.full_url, 404, "nf", Message(), io.BytesIO(b'{"message":"no route"}')
            )
        status, payload = handler(query, body)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        if status >= 400:
            raise urllib.error.HTTPError(
                request.full_url, status, "err", Message(), io.BytesIO(data)
            )
        return _Resp(data)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    return fake


def test_from_uri_keeps_https() -> None:
    catalog = OSSUnityCatalog.from_uri("https://uc.example.com:8443")
    assert catalog._base_url == "https://uc.example.com:8443"
    assert OSSUnityCatalog.from_uri("uc://https://h")._base_url == "https://h"


def test_commit_tail_404_is_not_reported_as_missing_table(server: FakeServer) -> None:
    server.on(
        "GET",
        f"{UC}/tables/main.sch.tbl",
        lambda q, b: (
            200,
            {
                "name": "tbl",
                "table_id": "id1",
                "storage_location": "s3://b/t",
                "properties": {"delta.feature.catalogManaged": "supported"},
            },
        ),
    )
    with pytest.raises(PreflightError, match=r"0\.5"):
        OSSUnityCatalog(BASE).resolve(ref())


def test_list_calls_name_a_missing_parent(server: FakeServer) -> None:
    catalog = OSSUnityCatalog(BASE)
    with pytest.raises(InvalidReferenceError):
        catalog.list_tables("main", "nope")
    with pytest.raises(InvalidReferenceError):
        catalog.list_schemas("nope")


def test_list_denied_names_privilege(server: FakeServer) -> None:
    server.on("GET", f"{UC}/catalogs", lambda q, b: (403, {"message": "denied"}))
    with pytest.raises(PreflightError, match="denied"):
        OSSUnityCatalog(BASE).list_catalogs()


def test_drop_missing_table_is_invalid_reference(server: FakeServer) -> None:
    with pytest.raises(InvalidReferenceError, match="does not exist"):
        OSSUnityCatalog(BASE).drop_table(ref())


def test_paged_non_object_body_is_preflight_error(server: FakeServer) -> None:
    server.on("GET", f"{UC}/catalogs", lambda q, b: (200, [{"name": "x"}]))
    with pytest.raises(PreflightError, match="not a JSON object"):
        OSSUnityCatalog(BASE).list_catalogs()


def test_register_then_failed_readback_says_the_table_exists(server: FakeServer) -> None:
    server.on("POST", f"{UC}/tables", lambda q, b: (200, {"name": "tbl"}))
    # GET of the new table 404s (e.g. read-after-write lag).
    with pytest.raises(PreflightError, match="was registered"):
        OSSUnityCatalog(BASE).register_table(ref(), "s3://b/t")


def test_search_skips_schema_dropped_mid_listing(server: FakeServer) -> None:
    server.on(
        "GET", f"{UC}/schemas", lambda q, b: (200, {"schemas": [{"name": "a"}, {"name": "b"}]})
    )

    def tables(q: dict[str, str], b: Any) -> Any:
        if q.get("schema_name") == "a":
            return 404, {"message": "schema a does not exist"}
        return 200, {"tables": [{"name": "t", "table_type": "MANAGED"}]}

    server.on("GET", f"{UC}/tables", tables)
    found = OSSUnityCatalog(BASE).search_tables("main")
    assert [t.full_name for t in found] == ["main.b.t"]


def test_search_summary_carries_schema_when_entry_omits_it(server: FakeServer) -> None:
    server.on("GET", f"{UC}/schemas", lambda q, b: (200, {"schemas": [{"name": "s"}]}))
    server.on("GET", f"{UC}/tables", lambda q, b: (200, {"tables": [{"name": "t"}]}))
    (found,) = OSSUnityCatalog(BASE).search_tables("main")
    assert found.full_name == "main.s.t"
