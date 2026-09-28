"""Regressions for the round-7 security review, each proven on a real local table.

Every test here failed before its fix.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import pytest
from deltaswamp.errors import DeltaSwampError, InvalidArgumentError
from deltaswamp.predicate import PredicateError

pa = pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
ds = pytest.importorskip("deltaswamp")

SCHEMA = pa.schema([("id", pa.int64()), ("s", pa.string())])


@pytest.fixture
def conn() -> Any:
    return ds.connect("file://")


def _table(conn: Any, tmp_path: Any, *, dv: bool = False, name: str = "t") -> str:
    path = str(tmp_path / name)
    props = {"delta.enableDeletionVectors": "true"} if dv else None
    conn.create_table(path, SCHEMA, properties=props)
    conn.open_table(path).append(
        pa.table({"id": [1, 1, 2], "s": ["keep", "hit", "z"]}, schema=SCHEMA)
    )
    return path


def _rows(conn: Any, path: str) -> list[str]:
    return sorted(r["s"] for r in conn.open_table(path).to_arrow().to_pylist())


class TestKernelMergeSandbox:
    """INJ-1: the kernel MERGE evaluated clause text on an unsandboxed DuckDB."""

    def test_a_set_value_cannot_read_local_files(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path, dv=True)
        local = tmp_path / "local.txt"
        local.write_text("local file contents\n")
        merge = conn.open_table(path).merge(pa.table({"id": [2]}), "target.id = source.id")
        merge = merge.when_matched_update({"s": f"(SELECT content FROM read_text('{local}'))"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(DeltaSwampError):
                merge.execute()
        assert _rows(conn, path) == ["hit", "keep", "z"]

    def test_the_merge_connection_runs_one_statement(self) -> None:
        import duckdb
        from deltaswamp.engine.kernel_merge import _one_statement

        con = duckdb.connect()
        try:
            with pytest.raises(InvalidArgumentError, match="one expression"):
                _one_statement(con, "SELECT 1; SELECT 2")
        finally:
            con.close()

    def test_an_on_condition_that_closes_early_is_refused_before_any_engine(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = _table(conn, tmp_path, dv=True)
        marker = str(tmp_path / "marker")
        on = f"target.id = source.id) WHERE TRUE); COPY (SELECT 1) TO '{marker}'; SELECT (TRUE"
        with pytest.raises(PredicateError):
            conn.open_table(path).merge(pa.table({"id": [1]}), on)
        assert not os.path.exists(marker)
        assert _rows(conn, path) == ["hit", "keep", "z"]


class TestOneExpression:
    """INJ-2: delta-rs read a prefix of a malformed predicate and acted on more rows."""

    BAD = (
        "id = 1) AND (s = 'hit'",
        "id = 1 s = 'hit'",
        "id = 1 -- AND s = 'hit'",
        "id = 1, s = 'hit'",
        "(id = 1 AND s = 'hit'",
        "id = 1 AND s = 'hit",
    )

    @pytest.mark.parametrize("dv", [False, True])
    @pytest.mark.parametrize("text", BAD)
    def test_delete_refuses(self, conn: Any, tmp_path: Any, dv: bool, text: str) -> None:
        path = _table(conn, tmp_path, dv=dv)
        with pytest.raises(PredicateError):
            conn.open_table(path).delete(text)
        assert _rows(conn, path) == ["hit", "keep", "z"]

    def test_update_refuses(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        with pytest.raises(PredicateError):
            conn.open_table(path).update({"s": "'UPD'"}, predicate=self.BAD[0])
        with pytest.raises(PredicateError):
            conn.open_table(path).update({"s": "'UPD') , (s"}, predicate="id = 2")
        assert _rows(conn, path) == ["hit", "keep", "z"]

    def test_replace_where_refuses(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        data = pa.table({"id": [1], "s": ["new"]}, schema=SCHEMA)
        with pytest.raises(PredicateError):
            conn.open_table(path).overwrite(data, predicate=self.BAD[0])
        assert _rows(conn, path) == ["hit", "keep", "z"]

    def test_merge_on_and_clauses_refuse(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        src = pa.table({"id": [1], "s": ["hit"]})
        t = conn.open_table(path)
        with pytest.raises(PredicateError):
            t.merge(src, "target.id = source.id) AND (target.s = source.s")
        with pytest.raises(PredicateError):
            t.merge(src, "target.id = source.id; SELECT 1")
        with pytest.raises(PredicateError):
            t.merge(src, "target.id = source.id").when_matched_delete("target.s = 'hit')")
        with pytest.raises(PredicateError):
            t.merge(src, "target.id = source.id").when_matched_update({"s": "'x' extra"})
        assert _rows(conn, path) == ["hit", "keep", "z"]

    def test_constraints_refuse(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        with pytest.raises(PredicateError):
            conn.open_table(path).add_constraint({"c": "id > 0) OR (TRUE"})

    def test_spark_sql_still_passes(self, conn: Any, tmp_path: Any) -> None:
        path = _table(conn, tmp_path)
        t = conn.open_table(path)
        t.delete("s RLIKE '^hi' AND CASE WHEN id > 0 THEN TRUE ELSE FALSE END")
        assert _rows(conn, path) == ["keep", "z"]
        t.update({"s": "concat(s, '-', 'x')"}, predicate="lower(s) IN ('z')")
        assert _rows(conn, path) == ["keep", "z-x"]

    @pytest.mark.parametrize(
        "text",
        [
            "id IN (SELECT id FROM x)",
            "v:a::int = 1",
            "exists(arr, x -> x > 1)",
            "s = 'a; -- /* ) ('",
            "`a;b` = 1",
            "trim(BOTH 'x' FROM s) = 'a'",
            "x - -1 > 0",
            "s COLLATE UTF8_LCASE = 'a'",
            "struct(1 AS a).a = 1",
            "s LIKE ANY ('a%', 'b%')",
        ],
    )
    def test_the_validator_accepts_spark_expressions(self, text: str) -> None:
        from deltaswamp.engine.dialect import check_expression

        check_expression(text)


class TestAzureEndpointAllowlist:
    """SEC-C1: an Azure location naming any host sent the connection's secret there."""

    def test_an_unknown_host_is_refused_without_an_explicit_endpoint(self) -> None:
        from deltaswamp._storage import azure_store_location
        from deltaswamp.credentials.databricks import azure_endpoint_for

        loc = "abfss://c@acct.dfs.evil.example/t"
        assert azure_endpoint_for(loc) is None
        with pytest.raises(InvalidArgumentError, match="not an Azure Storage domain"):
            azure_store_location(loc, {"azure_storage_sas_key": "sv=1&sig=x"})
        # Named by the caller, the host is theirs to choose.
        _, options = azure_store_location(
            loc, {"azure_storage_sas_key": "s", "azure_storage_endpoint": "https://x.example"}
        )
        assert options["azure_storage_endpoint"] == "https://x.example"

    def test_another_account_than_the_options_name_is_refused(self) -> None:
        from deltaswamp._storage import azure_store_location

        opts = {"azure_storage_account_name": "mine", "azure_storage_sas_key": "s"}
        with pytest.raises(InvalidArgumentError, match="another account"):
            azure_store_location("abfss://c@other.dfs.core.windows.net/t", opts)
        with pytest.raises(InvalidArgumentError, match="another account"):
            azure_store_location("abfss://c@other.dfs.core.chinacloudapi.cn/t", opts)
        loc = "abfss://c@mine.dfs.core.windows.net/t"
        assert azure_store_location(loc, opts) == (loc, opts)

    def test_sovereign_private_link_and_fabric_hosts_still_derive(self) -> None:
        from deltaswamp.credentials.databricks import azure_endpoint_for

        for host, endpoint in [
            ("a.dfs.core.usgovcloudapi.net", "https://a.blob.core.usgovcloudapi.net"),
            ("a.privatelink.blob.core.windows.net", "https://a.privatelink.blob.core.windows.net"),
            ("onelake.dfs.fabric.microsoft.com", "https://onelake.blob.fabric.microsoft.com"),
        ]:
            assert azure_endpoint_for(f"abfss://c@{host}/t") == endpoint

    def test_the_native_store_refuses_the_unknown_host(self, tmp_path: Any) -> None:
        from deltaswamp import _native

        with pytest.raises(Exception, match="not an Azure Storage domain"):
            _native.probe_put_if_absent(
                "abfss://c@acct.dfs.evil.example/t", {"azure_storage_sas_key": "s"}
            )

    def test_r2_keys_go_only_to_cloudflare(self) -> None:
        from deltaswamp.credentials.databricks import r2_endpoint_for

        assert r2_endpoint_for("r2://b@acct.r2.cloudflarestorage.com/t") == (
            "https://acct.r2.cloudflarestorage.com"
        )
        assert r2_endpoint_for("r2://b@acct.evil.example/t") is None
