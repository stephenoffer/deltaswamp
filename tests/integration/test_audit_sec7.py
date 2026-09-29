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


def _old_file(path: Any, days: int = 30) -> None:
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unrelated")
    old = time.time() - days * 86400
    os.utime(path, (old, old))


class TestForeignFilesInTheTableRoot:
    """SEC-P1: a table created in a non-empty folder, and a VACUUM deleting its files."""

    @pytest.mark.parametrize("how", ["create_table", "write_table"])
    def test_create_refuses_a_non_empty_directory(self, conn: Any, tmp_path: Any, how: str) -> None:
        from deltaswamp.errors import UnreachableTableError

        root = tmp_path / "docs"
        _old_file(root / "notes.txt")
        with pytest.raises(UnreachableTableError, match="not a Delta table"):
            if how == "create_table":
                conn.create_table(str(root), SCHEMA)
            else:
                conn.write_table(str(root), pa.table({"id": [1], "s": ["a"]}), mode="append")
        assert sorted(os.listdir(root)) == ["notes.txt"]

    def test_an_empty_log_directory_is_not_foreign(self, conn: Any, tmp_path: Any) -> None:
        root = tmp_path / "t"
        (root / "_delta_log").mkdir(parents=True)
        conn.create_table(str(root), SCHEMA)

    @pytest.mark.parametrize("dv", [False, True])
    def test_vacuum_deletes_only_delta_named_files(
        self, conn: Any, tmp_path: Any, dv: bool
    ) -> None:
        from deltaswamp.errors import DeltaSwampWarning

        path = _table(conn, tmp_path, dv=dv)
        root = tmp_path / "t"
        _old_file(root / "resume.docx")
        _old_file(root / "taxes" / "2025.pdf")
        _old_file(root / "part-00000-orphan-c000.snappy.parquet")
        with pytest.warns(DeltaSwampWarning, match="not deleted"):
            removed = conn.open_table(path).vacuum(dry_run=False)
        assert removed == ["part-00000-orphan-c000.snappy.parquet"]
        assert (root / "resume.docx").exists()
        assert (root / "taxes" / "2025.pdf").exists()
        assert not (root / "part-00000-orphan-c000.snappy.parquet").exists()
        assert _rows(conn, path) == ["hit", "keep", "z"]

    def test_a_remote_catalog_cannot_place_a_table_on_local_disk(self) -> None:
        from deltaswamp.connection import _refuse_local_catalog_location
        from deltaswamp.errors import UnreachableTableError

        class Remote:
            _base_url = "https://uc.example.com/api/2.1/unity-catalog"

        class Local:
            _base_url = "http://localhost:8080/api/2.1/unity-catalog"

        with pytest.raises(UnreachableTableError, match="this machine's filesystem"):
            _refuse_local_catalog_location(Remote(), "file:///home/u/Documents")
        with pytest.raises(UnreachableTableError):
            _refuse_local_catalog_location(Remote(), "/home/u/Documents")
        _refuse_local_catalog_location(Local(), "file:///tmp/uc/t")
        _refuse_local_catalog_location(Remote(), "s3://bucket/t")


def _serve(handler: Any) -> Any:
    import http.server
    import threading

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestOssUcRedirects:
    """SEC-C2 / SEC-C5: the OSS UC token followed redirects to other origins."""

    def test_the_token_is_not_forwarded_to_another_origin(self) -> None:
        import http.server

        from deltaswamp.catalog.ossuc import _request

        seen: list[Any] = []

        class Other(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                seen.append(self.headers.get("Authorization"))
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        other = _serve(Other)
        port = other.server_address[1]

        class Catalog(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{port}/moved")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        catalog = _serve(Catalog)
        try:
            base = f"http://127.0.0.1:{catalog.server_address[1]}"
            assert _request(base, "/api/2.1/unity-catalog/catalogs", "tok-SECRET") == {}
        finally:
            other.shutdown()
            catalog.shutdown()
        assert seen == [None]

    def test_a_downgrade_to_http_is_refused(self) -> None:
        import urllib.error
        import urllib.request

        from deltaswamp.catalog.ossuc import _AuthSafeRedirect

        request = urllib.request.Request("https://uc.example.com/api")
        request.add_header("Authorization", "Bearer tok")
        with pytest.raises(urllib.error.HTTPError, match="https to http"):
            _AuthSafeRedirect().redirect_request(
                request, None, 302, "Found", {}, "http://uc.example.com/api"
            )
        same = _AuthSafeRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://uc.example.com/api/v2"
        )
        assert same.get_header("Authorization") == "Bearer tok"

    def test_a_token_over_plain_http_warns(self) -> None:
        from deltaswamp.catalog import ossuc

        ossuc._WARNED_PLAIN_HTTP.discard("http://uc.example.com")
        with pytest.warns(UserWarning, match="plain http"):
            ossuc._warn_plain_http("http://uc.example.com/api/2.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ossuc._warn_plain_http("http://localhost:8080/api/2.1")


class TestLogPathsStayInTheTable:
    """SEC-P2 / SEC-P4: log paths outside the table root were read, and compacted in."""

    @staticmethod
    def _escaping(conn: Any, tmp_path: Any, add: str) -> tuple[str, str]:
        import json

        import pyarrow.parquet as pq

        (tmp_path / "outside").mkdir()
        secret = str(tmp_path / "outside" / "secret.parquet")
        pq.write_table(pa.table({"id": pa.array([4242], pa.int64())}), secret)
        root = str(tmp_path / "tbl")
        conn.write_table(root, pa.table({"id": pa.array([1, 2], pa.int64())}))
        conn.open_table(root).append(pa.table({"id": pa.array([3], pa.int64())}))
        entry = {
            "path": add.format(secret=secret),
            "size": os.path.getsize(secret),
            "partitionValues": {},
            "modificationTime": 1,
            "dataChange": True,
        }
        with open(os.path.join(root, "_delta_log", "00000000000000000002.json"), "w") as f:
            f.write(json.dumps({"commitInfo": {"operation": "WRITE", "timestamp": 1}}) + "\n")
            f.write(json.dumps({"add": entry}) + "\n")
        return root, secret

    @pytest.mark.parametrize(
        "add",
        ["../outside/secret.parquet", "%2E%2E/outside/secret.parquet", "file://{secret}"],
    )
    def test_reads_and_compaction_refuse(self, conn: Any, tmp_path: Any, add: str) -> None:
        root, _ = self._escaping(conn, tmp_path, add)
        t = conn.open_table(root)
        with pytest.raises(DeltaSwampError, match="outside the table root"):
            t.to_arrow()
        with pytest.raises(DeltaSwampError, match="outside the table root"):
            t.to_pyarrow_dataset().to_table()
        with pytest.raises(DeltaSwampError, match="outside the table root"):
            t.z_order("id")
        files = [f for f in os.listdir(root) if f.endswith(".parquet")]
        assert len(files) == 2  # nothing was written into the table


class TestSharingPresignedUrls:
    """SEC-P3 / SEC-C5: Sharing replay fetched any http(s) URL, internal hosts included."""

    @pytest.fixture(autouse=True)
    def _no_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DELTASWAMP_SHARING_ALLOW_PRIVATE_URLS", raising=False)

    @pytest.mark.parametrize(
        "url",
        [
            "http://s3.amazonaws.com/b/x.parquet?sig=1",
            "https://127.0.0.1/latest/meta-data/x.parquet",
            "https://169.254.169.254/latest/meta-data/iam/",
            "https://10.0.0.5/x.parquet",
            "https://[::1]/x.parquet",
            "https://[::ffff:127.0.0.1]/x.parquet",
        ],
    )
    def test_plain_http_and_internal_addresses_are_refused(self, url: str) -> None:
        from deltaswamp.engine.sharing import SharingEngine
        from deltaswamp.errors import UnreachableTableError

        with pytest.raises(UnreachableTableError):
            SharingEngine._check_url(url)

    def test_public_https_passes(self) -> None:
        from deltaswamp.engine.sharing import SharingEngine

        SharingEngine._check_url("https://bucket.s3.amazonaws.com/x.parquet?X-Amz-Signature=1")

    def test_a_name_that_resolves_to_loopback_is_refused_on_connect(self) -> None:
        import socket

        from deltaswamp.engine.sharing import _refuse_private_peer
        from deltaswamp.errors import UnreachableTableError

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.create_connection(listener.getsockname())
        try:
            with pytest.raises(UnreachableTableError, match=r"resolves to 127\.0\.0\.1"):
                _refuse_private_peer(client, "innocent.example")
        finally:
            client.close()
            listener.close()

    def test_every_redirect_hop_is_checked(self) -> None:
        import urllib.request

        from deltaswamp.engine.sharing import _CheckedRedirect
        from deltaswamp.errors import UnreachableTableError

        request = urllib.request.Request("https://bucket.s3.amazonaws.com/x.parquet")
        with pytest.raises(UnreachableTableError):
            _CheckedRedirect().redirect_request(
                request, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/"
            )


class TestSdkDebugRedaction:
    """SEC-C3: secrets the SDK logs at DEBUG that the redaction filter missed."""

    @pytest.fixture
    def captured(self) -> Any:
        import io
        import logging

        from deltaswamp import _sdk

        _sdk._install_log_redaction()
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logger = logging.getLogger("databricks.sdk")
        old = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        yield buf
        logger.removeHandler(handler)
        logger.setLevel(old)

    def test_hyphenated_and_nested_secret_fields(self, captured: Any) -> None:
        import json
        import logging

        body = {
            "config": {
                "s3.access-key-id": "ASIAKEYID",
                "s3.secret-access-key": "SECRETKEY1",
                "s3.session-token": "SESSIONTOK",
                "adls.sas-token.acct.dfs.core.windows.net": "sig=ADLSSAS",
                "gcs.oauth2.token": "GCSTOKEN",
            },
            "azure_aad": {"aad_token": "AADTOKEN"},
            "location": "s3://b/t",
        }
        logging.getLogger("databricks.sdk").debug("< %s", json.dumps(body))
        out = captured.getvalue()
        for secret in ("ASIAKEYID", "SECRETKEY1", "SESSIONTOK", "ADLSSAS", "GCSTOKEN", "AADTOKEN"):
            assert secret not in out, secret
        assert "s3://b/t" in out

    def test_headers_signatures_and_child_loggers(self, captured: Any) -> None:
        import logging

        logging.getLogger("databricks.sdk").debug(
            "POST /api\n> * Authorization: Bearer dapiPAT123\n> * Content-Length: 2"
        )
        logging.getLogger("databricks.sdk.oauth").debug('{"access_token": "CHILDTOKEN"}')
        logging.getLogger("databricks.sdk").debug(
            "GET https://a.blob.core.windows.net/c/x?sv=1&sig=SASSIG%3D&se=2"
            " https://b.s3.amazonaws.com/x?X-Amz-Signature=AMZSIG&X-Amz-Date=1"
        )
        out = captured.getvalue()
        for secret in ("dapiPAT123", "CHILDTOKEN", "SASSIG", "AMZSIG"):
            assert secret not in out, secret
        assert "Content-Length: 2" in out and "se=2" in out


class TestDdlFromTableMetadata:
    """INJ-3: generation / DEFAULT expressions from a log were spliced into CREATE TABLE."""

    @pytest.mark.parametrize(
        "key, text",
        [
            ("delta.generationExpression", "id + 1), `smuggled` STRING COMMENT 'x', `g2` BIGINT"),
            ("delta.generationExpression", "id + 1) GENERATED ALWAYS AS (id"),
            ("CURRENT_DEFAULT", "1 COMMENT 'smuggled'"),
            ("CURRENT_DEFAULT", "1, `smuggled` STRING"),
        ],
    )
    def test_a_column_expression_that_is_not_one_expression_is_refused(
        self, key: str, text: str
    ) -> None:
        from deltaswamp.engine.sql import _column_ddl

        field = pa.field("g", pa.int64(), metadata={key: text})
        with pytest.raises(InvalidArgumentError, match="single SQL expression"):
            _column_ddl("c.s.t", "g", "BIGINT", field)

    def test_legitimate_expressions_render(self) -> None:
        from deltaswamp.engine.sql import _column_ddl

        gen = pa.field(
            "g", pa.int64(), metadata={"delta.generationExpression": "CAST(id + 1 AS BIGINT)"}
        )
        clause, _ = _column_ddl("c.s.t", "g", "BIGINT", gen)
        assert clause.endswith("GENERATED ALWAYS AS (CAST(id + 1 AS BIGINT))")
        default = pa.field("d", pa.string(), metadata={"CURRENT_DEFAULT": "'a, b'"})
        clause, has_default = _column_ddl("c.s.t", "d", "STRING", default)
        assert has_default and clause.endswith("DEFAULT 'a, b'")


class TestDotSegmentNames:
    """INJ-4: a name part '.' or '..' became a dot segment in UC REST paths."""

    def test_dot_names_are_encoded(self) -> None:
        from deltaswamp.catalog.databricks import _segment
        from deltaswamp.catalog.ossuc import _q

        for quote in (_q, _segment):
            assert quote("..") == "%2E%2E"
            assert quote(".") == "%2E"
            assert quote("a.b") == "a.b"
            assert quote("x/y") == "x%2Fy"

    def test_the_server_sees_no_dot_segment(self) -> None:
        import http.server

        from deltaswamp.catalog.ossuc import OSSUnityCredentialProvider
        from deltaswamp.identity import parse_ref

        seen: list[str] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                seen.append(self.path)
                self.send_response(404)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args: Any) -> None:
                pass

        server = _serve(Handler)
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            ref = parse_ref("cat.`..`.`..`")
            with pytest.raises(DeltaSwampError):
                OSSUnityCredentialProvider(base, ref, "tok", "tid", "file:///tmp/x").credentials()
        finally:
            server.shutdown()
        assert seen and all("/../" not in p and not p.endswith("/..") for p in seen), seen
        assert all("%2E%2E" in p for p in seen), seen


class TestPickledSecrets:
    """SEC-C4: a pickled Databricks Table carried the PAT, even one from the environment."""

    def test_provider_and_catalog_pickle_no_literal_secret(self) -> None:
        import pickle

        from deltaswamp.catalog.databricks import DatabricksUnityCatalog
        from deltaswamp.credentials.databricks import DatabricksCredentialProvider, shipping

        pat = "dapiSECRET0123456789"
        provider = DatabricksCredentialProvider(
            "tid", host="https://h.cloud.databricks.com", token=pat, client_secret="CSECRET"
        )
        catalog = DatabricksUnityCatalog(host="https://h.cloud.databricks.com", token=pat)
        for obj in (provider, catalog):
            payload = pickle.dumps(obj)
            assert pat.encode() not in payload
            assert b"CSECRET" not in payload
            assert b"h.cloud.databricks.com" in payload
            assert pat.encode() in pickle.dumps(shipping(obj))

    def test_connect_ship_credentials_opts_in(self) -> None:
        import pickle

        pytest.importorskip("databricks.sdk")
        pat = "dapiSHIPPED0123456789"
        plain = ds.connect(host="https://h.cloud.databricks.com", token=pat)
        assert pat.encode() not in pickle.dumps(plain)
        shipped = ds.connect(
            host="https://h.cloud.databricks.com", token=pat, ship_credentials=True
        )
        assert pat.encode() in pickle.dumps(shipped)
