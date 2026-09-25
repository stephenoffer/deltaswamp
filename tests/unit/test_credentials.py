"""Credentials vended by Databricks Unity Catalog, mapped to object-store options."""

from __future__ import annotations

import pathlib
import pickle
from types import SimpleNamespace

import pytest
from deltaswamp.credentials.base import Cloud
from deltaswamp.credentials.base import Operation as CredOp
from deltaswamp.credentials.databricks import DatabricksCredentialProvider


class TestVendedCredentials:
    def test_aws_keys_carry_the_table_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """UC vends keys but no region; without one object_store assumes us-east-1."""
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

        provider = DatabricksCredentialProvider(
            table_id="tid-1",
            table_url="s3://databricks-unitycatalog-default-uw2/x",
            region="us-west-2",
            host="https://example.cloud.databricks.com",
            token="t",
        )
        response = SimpleNamespace(
            url="s3://databricks-unitycatalog-default-uw2/x",
            expiration_time=None,
            aws_temp_credentials=SimpleNamespace(
                access_key_id="AK",
                secret_access_key="SK",
                session_token="ST",
                access_point=None,
            ),
        )
        creds = provider._to_credentials(response, CredOp.READ)
        assert creds.secrets["aws_region"] == "us-west-2"

    def test_region_falls_back_to_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        provider = DatabricksCredentialProvider(table_id="t", host="h", token="t")
        assert provider._aws_region() == "eu-west-1"

    def test_gcs_token_becomes_the_store_layer_key(self) -> None:
        """A vended GCS OAuth2 token maps to `google_bearer_token`, never to
        `google_application_credentials` (which object_store reads as a file path)."""
        provider = DatabricksCredentialProvider(
            table_id="tid-1", table_url="gs://bucket/t", host="h", token="t"
        )
        response = SimpleNamespace(
            url="gs://bucket/t",
            expiration_time=None,
            gcp_oauth_token=SimpleNamespace(oauth_token="ya29.TOKEN"),
        )
        creds = provider._to_credentials(response, CredOp.READ)
        assert creds.cloud is Cloud.GCP
        assert creds.secrets == {"google_bearer_token": "ya29.TOKEN"}
        assert "google_application_credentials" not in creds.secrets

    def test_the_store_layer_accepts_the_gcs_key(self) -> None:
        """The Python key and the Rust alias list in store.rs must not drift apart."""
        store_rs = (
            pathlib.Path(__file__).parents[2] / "crates" / "native" / "src" / "store.rs"
        ).read_text()
        assert '"google_bearer_token"' in store_rs

    def test_azure_sas_gets_an_explicit_endpoint(self) -> None:
        """Account-name inference breaks on private-link and sovereign hosts."""
        provider = DatabricksCredentialProvider(
            table_id="tid-1",
            table_url="abfss://c@acct.dfs.core.windows.net/t",
            host="h",
            token="t",
        )
        response = SimpleNamespace(
            url="abfss://c@acct.dfs.core.windows.net/t",
            expiration_time=None,
            azure_user_delegation_sas=SimpleNamespace(sas_token="sv=SECRET"),
        )
        creds = provider._to_credentials(response, CredOp.READ)
        assert creds.cloud is Cloud.AZURE
        assert creds.secrets["azure_endpoint"] == "https://acct.dfs.core.windows.net"
        # Azure SAS is path-scoped, so the store registry must key on the path
        # rather than the bucket, or the second table in a container gets the
        # first one's signature and a 403.
        assert creds.scope_prefix == "abfss://c@acct.dfs.core.windows.net/t"

    def test_the_provider_stays_picklable_with_a_region(self) -> None:
        provider = DatabricksCredentialProvider(
            table_id="tid-1", region="us-west-2", host="h", token="t"
        )
        assert pickle.loads(pickle.dumps(provider))._region == "us-west-2"
