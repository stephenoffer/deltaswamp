"""The kernel engine's decisions that need no table on disk."""

from __future__ import annotations

import time
import warnings
from typing import Any

import pytest
from deltaswamp.capability import Operation
from deltaswamp.credentials.base import Cloud, Credentials
from deltaswamp.engine.kernel import KernelEngine
from deltaswamp.errors import CredentialExpiryWarning

from tests.helpers import resolved_table


def test_catalog_managed_cdf_is_refused_in_supports() -> None:
    table = resolved_table(
        data_source_format="DELTA",
        reader_features=frozenset({"catalogManaged"}),
        writer_features=frozenset({"catalogManaged", "inCommitTimestamp"}),
        properties={"delta.enableChangeDataFeed": "true"},
    )
    verdict = KernelEngine().supports(Operation.CDF, table)
    assert not verdict.ok
    assert "commit tail" in verdict.reason


def test_cdf_without_the_feed_enabled_is_refused_in_supports() -> None:
    """A table with no change feed has nothing for any engine to read.

    delta-rs already reported this from `supports`, so the router moved on to
    the kernel, which only checked for catalog-managed and said yes. The call
    then refused -- making `capabilities()` claim a feed that is not there,
    which is the whole reason to ask before calling.
    """
    table = resolved_table(data_source_format="DELTA", properties={})
    verdict = KernelEngine().supports(Operation.CDF, table)
    assert not verdict.ok
    assert "enableChangeDataFeed" in verdict.reason
    assert "not retroactive" in verdict.reason


def test_cdf_is_allowed_once_the_feed_is_enabled() -> None:
    table = resolved_table(
        data_source_format="DELTA", properties={"delta.enableChangeDataFeed": "true"}
    )
    assert KernelEngine().supports(Operation.CDF, table).ok


def test_renaming_without_column_mapping_is_refused_in_supports() -> None:
    """Without column mapping a column's name is also its name in every file.

    The metadata-commit path refused this from inside the commit, so `can()`
    said yes and the call said no -- on every table that does not use column
    mapping, which is most of them. The mode is a table property, so the
    verdict costs nothing.
    """
    table = resolved_table(data_source_format="DELTA", properties={})
    for operation in (Operation.RENAME_COLUMN, Operation.DROP_COLUMN):
        verdict = KernelEngine().supports(operation, table)
        assert not verdict.ok
        assert "column mapping" in verdict.reason
        assert "columnMapping.mode" in verdict.remedy


def test_renaming_is_allowed_once_column_mapping_is_on() -> None:
    for mode in ("name", "id"):
        table = resolved_table(
            data_source_format="DELTA", properties={"delta.columnMapping.mode": mode}
        )
        assert KernelEngine().supports(Operation.RENAME_COLUMN, table).ok, mode
        assert KernelEngine().supports(Operation.DROP_COLUMN, table).ok, mode


def test_shredded_variants_fail_inside_the_call(monkeypatch: Any) -> None:
    """A shredded-variant failure surfaces from `scan()`, where the read can be retried,
    not partway through the caller's iteration."""
    pa = pytest.importorskip("pyarrow")
    features = frozenset({"variantType", "variantShredding"})
    table = resolved_table(
        data_source_format="DELTA", reader_features=features, writer_features=features
    )
    assert KernelEngine().supports(Operation.SCAN, table).ok

    def shredded(*_: Any, **__: Any) -> Any:
        schema = pa.schema([("id", pa.int64())])

        def batches() -> Any:
            if schema is not None:
                raise RuntimeError("The field v presumed to be of Variant type might be shredded")
            yield pa.record_batch([], schema=schema)

        return pa.RecordBatchReader.from_batches(schema, batches())

    monkeypatch.setattr(KernelEngine, "_scan", shredded)
    with pytest.raises(Exception, match="shredded"):
        KernelEngine().scan(table)


class TestCredentialExpiry:
    """The object store holds one credential for the whole scan, so a short TTL warns first."""

    def test_a_short_lived_credential_warns_before_the_read(self) -> None:
        soon = Credentials(
            cloud=Cloud.AWS, url="s3://b/t", expires_at=time.time() + 30, secrets={"k": "v"}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            KernelEngine()._warn_if_short_lived(soon)
        assert len(caught) == 1
        assert issubclass(caught[0].category, CredentialExpiryWarning)
        assert "plan_scan" in str(caught[0].message)

    def test_a_long_lived_credential_is_silent(self) -> None:
        engine = KernelEngine()
        for expires_at in (time.time() + 86400, None):  # plenty of time, and "no TTL given"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                engine._warn_if_short_lived(
                    Credentials(cloud=Cloud.AWS, url="s3://b/t", expires_at=expires_at)
                )
            assert caught == [], f"expires_at={expires_at} should not warn"
