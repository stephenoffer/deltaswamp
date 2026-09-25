"""The delta-rs engine's own verdicts and storage options, without a table."""

from __future__ import annotations

import pytest
from deltaswamp.capability import Operation
from deltaswamp.catalog import ResolvedTable
from deltaswamp.credentials.base import Cloud, Credentials, StaticCredentialProvider
from deltaswamp.engine.deltars import DeltaRsEngine, _pin_s3_endpoint
from deltaswamp.identity import parse_ref

from tests.helpers import resolved_table


class TestS3Endpoint:
    """An explicit regional endpoint keeps delta-rs from probing EC2 metadata for a region."""

    def test_vended_keys_get_a_regional_endpoint(self) -> None:
        options = {"aws_access_key_id": "k", "aws_region": "us-west-2"}
        _pin_s3_endpoint(options)
        assert options["aws_endpoint"] == "https://s3.us-west-2.amazonaws.com"

    def test_china_regions_use_their_own_domain(self) -> None:
        options = {"aws_access_key_id": "k", "aws_region": "cn-north-1"}
        _pin_s3_endpoint(options)
        assert options["aws_endpoint"] == "https://s3.cn-north-1.amazonaws.com.cn"

    @pytest.mark.parametrize("key", ["aws_endpoint", "AWS_ENDPOINT_URL", "endpoint"])
    def test_an_explicit_endpoint_is_never_overridden(self, key: str) -> None:
        options = {"aws_access_key_id": "k", "aws_region": "us-west-2", key: "http://minio"}
        _pin_s3_endpoint(options)
        assert options[key] == "http://minio"
        assert len(options) == 3

    def test_no_region_or_no_keys_means_no_change(self) -> None:
        for options in ({"aws_access_key_id": "k"}, {"aws_region": "us-west-2"}):
            before = dict(options)
            _pin_s3_endpoint(options)
            assert options == before


class TestSupports:
    """Refusals made in `supports()`, so the router can try the next engine."""

    def test_column_mapped_cdf_is_refused(self) -> None:
        pytest.importorskip("deltalake")
        table = resolved_table(
            data_source_format="DELTA",
            properties={"delta.enableChangeDataFeed": "true", "delta.columnMapping.mode": "name"},
        )
        verdict = DeltaRsEngine().supports(Operation.CDF, table)
        assert not verdict.ok
        assert "column mapping" in verdict.reason

    def test_restore_on_deletion_vectors_is_refused(self) -> None:
        pytest.importorskip("deltalake")
        table = resolved_table(
            data_source_format="DELTA",
            reader_features=frozenset({"deletionVectors"}),
            writer_features=frozenset({"deletionVectors"}),
        )
        verdict = DeltaRsEngine().supports(Operation.RESTORE, table)
        assert not verdict.ok
        assert "delta-rs#4613" in verdict.reason

    def test_vended_gcs_bearer_tokens_are_refused(self) -> None:
        """deltalake ignores the token and falls back to the GCE metadata server."""
        pytest.importorskip("deltalake")
        credentials = Credentials(
            cloud=Cloud.GCP, url="gs://b/t", expires_at=None, secrets={"google_bearer_token": "x"}
        )
        table = ResolvedTable(
            ref=parse_ref("gs://b/t"),
            location="gs://b/t",
            data_source_format="DELTA",
            credential_provider=StaticCredentialProvider(credentials),
        )
        verdict = DeltaRsEngine().supports(Operation.MERGE, table)
        assert not verdict.ok
        assert "bearer token" in verdict.reason


class TestAddFeature:
    """delta-rs adds only features it can go on writing afterwards."""

    def test_takes_features_it_can_then_write(self) -> None:
        pytest.importorskip("deltalake")
        verdict = DeltaRsEngine().supports(
            Operation.ADD_FEATURE, resolved_table(), features=["changeDataFeed", "v2Checkpoint"]
        )
        assert verdict.ok

    @pytest.mark.parametrize("feature", ["rowTracking", "domainMetadata", "typeWidening"])
    def test_declines_features_it_cannot_write(self, feature: str) -> None:
        verdict = DeltaRsEngine().supports(
            Operation.ADD_FEATURE, resolved_table(), features=[feature]
        )
        assert not verdict.ok
