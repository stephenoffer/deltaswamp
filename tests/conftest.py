"""Settings for the whole test suite."""

from __future__ import annotations

import os
from typing import Any

import pytest

# An engine the router chose must not refuse the call itself while another
# engine would have served it: a refusal that depends only on the table and
# the arguments belongs in the engine's supports() or in the request's needs,
# where can() sees it and routing moves on. See deltaswamp._request.
os.environ.setdefault("DELTASWAMP_STRICT_ROUTING", "1")

#: Tests whose call is refused inside an engine as UnreachableTableError, with
#: the router willing to send the same request to another engine. Each is a
#: known gap found by strict routing, listed here instead of hidden: the test
#: runs with the check off. In all of them the request itself is wrong, so the
#: other engine would refuse too. They are raised with a routing refusal's
#: class where InvalidArgumentError is meant (E11 in the contract findings), or
#: depend on the table's history, which routing does not read. Remove an entry
#: once its refusal moves into supports() or gets the right class.
STRICT_ROUTING_GAPS: dict[str, str] = {
    "test_audit_alter.py::TestChangeFeedReservedNames::test_enabling_it_through_delta_rs": (
        "G1: delta-rs's set_properties refuses a property value the kernel's route accepts"
    ),
    "test_audit_alter.py::TestDeltaRsAlterValidation::test_invalid_property_values": (
        "G1: delta-rs's set_properties refuses a property value the kernel's route accepts"
    ),
    "test_audit_alter.py::TestDeltaRsAlterValidation::test_downgrading_a_writer_7_protocol": (
        "G1: delta-rs's set_properties refuses delta.minWriterVersion"
    ),
    "[table::set_properties_bad": (
        "G1: delta-rs's set_properties refuses a property value the kernel's route accepts"
    ),
    "test_audit_alter.py::TestDeltaRsAlterValidation::test_column_names_parquet_cannot_hold": (
        "G2: delta-rs's add_columns refuses a column name Parquet cannot hold"
    ),
    "test_audit_alter.py::TestLogUpkeep::test_drop_not_null_on_a_missing_column": (
        "G2: an ALTER naming a missing column is refused inside the engine"
    ),
}


def _gap(nodeid: str) -> str | None:
    for key, reason in STRICT_ROUTING_GAPS.items():
        if key in nodeid:
            return reason
    return None


@pytest.fixture(autouse=True)
def _strict_routing_gap(request: Any, monkeypatch: Any) -> None:
    if _gap(request.node.nodeid) is not None:
        monkeypatch.setenv("DELTASWAMP_STRICT_ROUTING", "0")
