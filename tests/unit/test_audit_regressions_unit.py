"""Small API guarantees added after the audit."""

import deltaswamp as ds


def test_every_warning_shares_one_base() -> None:
    for name in (
        "SqlFallbackWarning",
        "EngineFallbackWarning",
        "IgnoredPropertyWarning",
        "CredentialExpiryWarning",
    ):
        assert issubclass(getattr(ds, name), ds.DeltaSwampWarning)
    assert issubclass(ds.DeltaSwampWarning, UserWarning)
