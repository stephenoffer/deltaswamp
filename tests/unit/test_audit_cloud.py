"""Cloud/object-store audit fixes (CL-01..CL-11) that need no emulator."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import pytest
from deltaswamp import _storage
from deltaswamp._storage import (
    azure_store_location,
    canonical_options,
    commit_refusal,
    engine_options,
    iceberg_fileio_properties,
    merge_options,
)
from deltaswamp.capability import Operation
from deltaswamp.catalog import ResolvedTable
from deltaswamp.credentials.base import Cloud, Credentials, StaticCredentialProvider
from deltaswamp.errors import InvalidArgumentError, InvalidReferenceError
from deltaswamp.identity import parse_ref


class TestCanonicalKeys:
    """CL-04: one spelling per setting, so object_store never picks at random."""

    def test_aliases_collapse_to_the_canonical_name(self) -> None:
        out = canonical_options(
            {"AWS_REGION": "eu-central-1", "AWS_ENDPOINT_URL": "http://x", "token": "t"},
            "s3://b/t",
        )
        assert out == {
            "aws_region": "eu-central-1",
            "aws_endpoint": "http://x",
            "aws_session_token": "t",
        }

    def test_endpoint_means_the_cloud_of_the_url(self) -> None:
        assert canonical_options({"endpoint": "e"}, "s3://b/t") == {"aws_endpoint": "e"}
        assert canonical_options({"endpoint": "e"}, "abfss://c@a.dfs.core.windows.net/t") == {
            "azure_storage_endpoint": "e"
        }

    def test_one_setting_twice_with_different_values_is_refused(self) -> None:
        with pytest.raises(InvalidArgumentError, match="aws_endpoint twice"):
            canonical_options({"aws_endpoint": "a", "AWS_ENDPOINT_URL": "b"}, "s3://b/t")

    def test_unknown_keys_pass_through(self) -> None:
        out = canonical_options({"AWS_S3_ALLOW_UNSAFE_RENAME": "true"}, "s3://b/t")
        assert out == {"AWS_S3_ALLOW_UNSAFE_RENAME": "true"}


class TestPrecedence:
    """CL-04: vended secrets win; the caller's region and endpoint win otherwise."""

    VENDED: ClassVar[dict[str, str]] = {
        "aws_access_key_id": "V",
        "aws_secret_access_key": "S",
        "aws_region": "us-west-2",
    }

    def test_user_region_overrides_the_vended_metastore_region(self) -> None:
        out = merge_options({"AWS_REGION": "eu-central-1"}, self.VENDED, "s3://b/t")
        assert out["aws_region"] == "eu-central-1"
        assert "AWS_REGION" not in out

    def test_vended_credentials_replace_every_user_credential_key(self) -> None:
        out = merge_options(
            {"aws_access_key_id": "U", "aws_session_token": "UT"}, self.VENDED, "s3://b/t"
        )
        assert out["aws_access_key_id"] == "V"
        assert "aws_session_token" not in out

    def test_a_vended_endpoint_wins_with_its_region(self) -> None:
        vended = {
            **self.VENDED,
            "aws_endpoint_url": "https://ap.s3-accesspoint.eu-west-1.amazonaws.com",
        }
        out = merge_options(
            {"aws_region": "us-east-1", "endpoint": "http://mine"}, vended, "s3://b/t"
        )
        assert out["aws_endpoint"].startswith("https://ap.")
        assert out["aws_region"] == "us-west-2"

    def test_user_azure_endpoint_wins_over_the_derived_one(self) -> None:
        vended = {
            "azure_storage_sas_key": "sig",
            "azure_endpoint": "https://a.blob.core.windows.net",
        }
        out = merge_options(
            {"azure_storage_endpoint": "https://private.link"},
            vended,
            "abfss://c@a.dfs.core.windows.net/t",
        )
        assert out["azure_storage_endpoint"] == "https://private.link"

    def test_merge_is_the_same_whatever_the_dict_order(self) -> None:
        user = {"aws_endpoint_url": "http://e", "REGION": "eu-west-1"}
        a = engine_options(user, self.VENDED, "s3://b/t")
        b = engine_options(dict(reversed(list(user.items()))), self.VENDED, "s3://b/t")
        assert a == b

    def test_both_engines_merge_alike(self) -> None:
        from deltaswamp.engine.deltars import DeltaRsEngine
        from deltaswamp.engine.kernel import KernelEngine

        table = _table("s3://b/t", Cloud.AWS, self.VENDED)
        user = {"AWS_REGION": "eu-central-1"}
        k = KernelEngine(storage_options=user)._options(table, write=False)
        d = DeltaRsEngine(storage_options=user)._storage_options(table, write=False)
        assert k == d
        assert k["aws_region"] == "eu-central-1"


class TestChinaEndpoint:
    """CL-03: the kernel gets the amazonaws.com.cn endpoint delta-rs had."""

    def test_kernel_options_pin_the_china_endpoint(self) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        vended = {
            "aws_access_key_id": "A",
            "aws_secret_access_key": "x",
            "aws_region": "cn-north-1",
        }
        table = _table("s3://b/t", Cloud.AWS, vended)
        options = KernelEngine(storage_options={"aws_region": "us-east-1"})._options(
            table, write=False
        )
        assert options["aws_endpoint"] == "https://s3.cn-north-1.amazonaws.com.cn"
        assert options["aws_region"] == "cn-north-1"

    def test_static_china_keys_get_the_endpoint_too(self) -> None:
        out = engine_options(
            {
                "aws_access_key_id": "A",
                "aws_secret_access_key": "x",
                "aws_region": "cn-northwest-1",
            },
            None,
            "s3://b/t",
        )
        assert out["aws_endpoint"] == "https://s3.cn-northwest-1.amazonaws.com.cn"


class TestAzureSovereign:
    """CL-02: the az:// form delta-rs's object store can parse."""

    @pytest.mark.parametrize(
        ("host", "endpoint"),
        [
            ("acct.dfs.core.chinacloudapi.cn", "https://acct.blob.core.chinacloudapi.cn"),
            ("acct.dfs.core.usgovcloudapi.net", "https://acct.blob.core.usgovcloudapi.net"),
        ],
    )
    def test_sovereign_hosts_are_rewritten(self, host: str, endpoint: str) -> None:
        uri, options = azure_store_location(
            f"abfss://cont@{host}/a/t", {"azure_storage_sas_key": "s"}
        )
        assert uri == "az://cont/a/t"
        assert options["azure_storage_account_name"] == "acct"
        assert options["azure_storage_endpoint"] == endpoint

    def test_public_cloud_urls_are_left_alone(self) -> None:
        loc = "abfss://cont@acct.dfs.core.windows.net/t"
        assert azure_store_location(loc, {}) == (loc, {})

    def test_an_explicit_endpoint_is_kept(self) -> None:
        _, options = azure_store_location(
            "abfss://cont@acct.dfs.core.chinacloudapi.cn/t",
            {"azure_storage_endpoint": "http://127.0.0.1:10000/devstoreaccount1"},
        )
        assert options["azure_storage_endpoint"] == "http://127.0.0.1:10000/devstoreaccount1"

    @pytest.mark.parametrize(
        "location",
        [
            "abfss://cont@zzqqnotreal98765x.dfs.core.chinacloudapi.cn/t",
            # CL-06: the host is explicit, so the kernel no longer refuses.
            "abfss://cont@zzqqnotreal98765x.dfs.core.windows.net/t",
        ],
    )
    def test_the_kernel_store_builds_with_no_endpoint_option(self, location: str) -> None:
        pytest.importorskip("deltaswamp._native")
        from deltaswamp.engine.kernel import KernelEngine

        engine = KernelEngine(
            storage_options={
                "azure_storage_sas_key": "sv=1&sig=x",
                "timeout": "1s",
                "max_retries": "0",
            }
        )
        table = ResolvedTable(ref=parse_ref(location), location=location)
        with pytest.raises(Exception) as err:
            engine.snapshot(table)
        # It fails on the network (the account does not exist), not on the URL.
        text = str(err.value)
        assert "known pattern" not in text and "no explicit endpoint" not in text, text
        assert "Error performing GET" in text, text


class TestGcsBearer:
    """CL-05: every token alias routes to the kernel; none is silently dropped."""

    @pytest.mark.parametrize("key", ["google_bearer_token", "gcp_oauth_token", "BEARER_TOKEN"])
    def test_delta_rs_refuses_a_path_table_bearer_token(self, key: str) -> None:
        pytest.importorskip("deltalake")
        from deltaswamp.engine.deltars import DeltaRsEngine

        table = ResolvedTable(
            ref=parse_ref("gs://b/t"), location="gs://b/t", data_source_format="DELTA"
        )
        verdict = DeltaRsEngine(storage_options={key: "ya29.x"}).supports(Operation.CREATE, table)
        assert not verdict.ok
        assert "storage options carry a GCS OAuth bearer token" in verdict.reason

    def test_every_alias_reaches_delta_rs_options_as_one_key(self) -> None:
        out = canonical_options({"gcp_oauth_token": "ya29.x"}, "gs://b/t")
        assert out == {"google_bearer_token": "ya29.x"}


class TestCommitSafety:
    """CL-07 / CL-10: writes refused before any side effect."""

    def test_dynamodb_locking_is_refused(self) -> None:
        refused = commit_refusal("s3://b/t", {"AWS_S3_LOCKING_PROVIDER": "dynamodb"})
        assert refused is not None and "DynamoDB" in refused[0]

    def test_conditional_put_disabled_is_refused(self) -> None:
        refused = commit_refusal("s3://b/t", {"aws_conditional_put": "disabled"})
        assert refused is not None and "put-if-absent" in refused[0]

    def test_a_store_ignoring_if_none_match_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_storage, "_PROBED", {})
        monkeypatch.setattr(_storage, "_probe", lambda *a: False)
        refused = commit_refusal("s3://b/t", {"aws_endpoint": "http://minio.local:9000"})
        assert refused is not None and "ignores If-None-Match" in refused[0]
        assert _storage.OPT_OUT_PROBE in refused[1]

    def test_the_opt_out_and_aws_hosts_skip_the_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object) -> bool:
            raise AssertionError("probed")

        monkeypatch.setattr(_storage, "_probe", boom)
        assert (
            commit_refusal(
                "s3://b/t", {"aws_endpoint": "http://minio:9000", _storage.OPT_OUT_PROBE: "true"}
            )
            is None
        )
        assert (
            commit_refusal("s3://b/t", {"aws_endpoint": "https://s3.us-west-2.amazonaws.com"})
            is None
        )

    def test_can_agrees_with_the_refusal(self) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        table = ResolvedTable(
            ref=parse_ref("s3://b/t"), location="s3://b/t", data_source_format="DELTA"
        )
        engine = KernelEngine(storage_options={"aws_conditional_put": "disabled"})
        verdict = engine.supports(Operation.APPEND, table)
        assert not verdict.ok and "aws_conditional_put=disabled" in verdict.reason
        assert "aws_conditional_put" not in engine.supports(Operation.SCAN, table).reason

    def test_the_opt_out_key_never_reaches_a_store(self) -> None:
        from deltaswamp.engine.kernel import KernelEngine

        table = ResolvedTable(ref=parse_ref("s3://b/t"), location="s3://b/t")
        options = KernelEngine(storage_options={_storage.OPT_OUT_PROBE: "true"})._options(
            table, write=True
        )
        assert _storage.OPT_OUT_PROBE not in options


def test_wasb_is_refused_at_parse() -> None:
    """CL-08."""
    with pytest.raises(InvalidReferenceError, match="abfss://"):
        parse_ref("wasbs://c@a.blob.core.windows.net/t")


def test_vacuum_paths_are_table_relative() -> None:
    """CL-09: the three forms delta-rs reported, normalised."""
    from deltaswamp.engine.deltars import _table_relative

    assert _table_relative(["/tmp/tbl/part-1.parquet"], "/tmp/tbl") == ["part-1.parquet"]
    assert _table_relative(["t/p=a/part-1.parquet"], "az://cont/t") == ["p=a/part-1.parquet"]
    assert _table_relative(["t/part-1.parquet"], "gs://b/t") == ["part-1.parquet"]
    assert _table_relative(["part-1.parquet"], "s3://b/t") == ["part-1.parquet"]
    # A lite/dry run is already table-relative; a path that happens to start
    # with the table's directory name is not stripped there.
    assert _table_relative(["t/x.parquet"], "az://cont/t", full=False) == ["t/x.parquet"]


class TestOtherEngines:
    """CL-11: storage_options reach PyIceberg and Delta Sharing."""

    def test_fileio_properties(self) -> None:
        props = iceberg_fileio_properties(
            {
                "AWS_ENDPOINT_URL": "http://minio",
                "aws_region": "eu-west-1",
                "proxy_url": "http://p",
                "timeout": "30s",
            }
        )
        assert props["s3.endpoint"] == "http://minio"
        assert props["s3.region"] == "eu-west-1"
        assert props["s3.proxy-uri"] == "http://p"
        assert props["s3.request-timeout"] == "30.0"

    def test_connect_passes_them_on(self) -> None:
        pytest.importorskip("pyiceberg")
        import deltaswamp as ds
        from deltaswamp.capability import Engine
        from deltaswamp.catalog.filesystem import FilesystemCatalog

        conn = ds.connect(
            catalog=FilesystemCatalog(),
            storage_options={"aws_endpoint": "http://minio", "proxy_url": "http://p"},
            iceberg_properties={"s3.endpoint": "http://override"},
        )
        iceberg = conn.router.engines[Engine.ICEBERG]
        assert iceberg._properties["s3.endpoint"] == "http://override"  # type: ignore[attr-defined]
        assert iceberg._properties["s3.proxy-uri"] == "http://p"  # type: ignore[attr-defined]
        sharing = conn.router.engines.get(Engine.SHARING)
        if sharing is not None:
            assert sharing._proxy_url == "http://p"  # type: ignore[attr-defined]


def test_staging_credentials_with_no_known_key_are_refused() -> None:
    """CL-11b: a vended config nothing maps must not fall back to ambient credentials."""
    from deltaswamp.errors import DeltaSwampError
    from deltaswamp.governance import staging_storage_options

    with pytest.raises(DeltaSwampError, match="none of which"):
        staging_storage_options("s3://b/t", [{"prefix": "s3://b/t", "config": {"x.unknown": "1"}}])
    options, _ = staging_storage_options(
        "abfss://c@a.dfs.core.windows.net/t",
        [
            {
                "prefix": "abfss://c@a.dfs.core.windows.net/t",
                "config": {"adls.sas-token.a.dfs.core.windows.net": "sv=1"},
            }
        ],
    )
    assert options["azure_storage_sas_key"] == "sv=1"


def _table(location: str, cloud: Cloud, secrets: dict[str, str]) -> ResolvedTable:
    cred = Credentials(cloud=cloud, url=location, expires_at=None, secrets=secrets)
    base = ResolvedTable(ref=parse_ref(location), location=location, data_source_format="DELTA")
    return dataclasses.replace(base, credential_provider=StaticCredentialProvider(cred))


class TestCreateCommitsOnce:
    """CL-01: a create that must not replace anything gets no delta-rs retry."""

    def _engine_and_calls(
        self, monkeypatch: pytest.MonkeyPatch, fail: bool
    ) -> tuple[Any, list[dict[str, Any]]]:
        deltalake = pytest.importorskip("deltalake")
        from deltaswamp.engine.deltars import DeltaRsEngine

        calls: list[dict[str, Any]] = []

        def create(*args: object, **kwargs: object) -> None:
            calls.append(kwargs)
            if fail:
                raise deltalake.exceptions.CommitFailedError("Failed to commit transaction: 0")

        monkeypatch.setattr(deltalake.DeltaTable, "create", staticmethod(create))
        return DeltaRsEngine(), calls

    def test_no_retries_for_error_and_ignore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._engine_and_calls(monkeypatch, fail=False)
        table = ResolvedTable(ref=parse_ref("/tmp/x"), location="/tmp/x")
        schema = pytest.importorskip("pyarrow").schema([("a", "int64")])
        engine.create(table, schema, mode="error")
        assert calls[-1]["commit_properties"].max_commit_retries == 0
        engine.create(table, schema, mode="overwrite")
        assert calls[-1]["commit_properties"] is None

    def test_the_loser_is_told_the_table_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from deltaswamp.errors import UnreachableTableError

        engine, _ = self._engine_and_calls(monkeypatch, fail=True)
        table = ResolvedTable(ref=parse_ref("/tmp/x"), location="/tmp/x")
        schema = pytest.importorskip("pyarrow").schema([("a", "int64")])
        with pytest.raises(UnreachableTableError, match="another writer created it first"):
            engine.create(table, schema, mode="error")
