"""The plans a Ray Data read_delta/write_delta runs, against a real workspace.

Each read is planned on the driver, pickled, and read in a separate Python
process -- a stand-in for a Ray worker: no snapshot cache, no catalog, only
what the plan carries. See docs/ray-data.md.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pickle
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from deltaswamp.capability import Operation
from deltaswamp.errors import DeltaSwampError

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.databricks

_WORKER = textwrap.dedent(
    """
    import pickle, sys
    import pyarrow as pa
    plan, parts = pickle.load(open(sys.argv[1], "rb"))
    tables = [plan.read(part) for part in parts]
    out = pa.concat_tables([t for t in tables if t.num_rows]) if tables else None
    with open(sys.argv[2], "wb") as f:
        pickle.dump(out.to_pylist() if out is not None else [], f)
    """
)


def _read_in_a_worker(tmp_path: Path, plan: Any, parts: list[Any]) -> list[dict[str, Any]]:
    """Read `parts` of `plan` in a fresh interpreter, as a Ray task would."""
    job, result = tmp_path / "job.pkl", tmp_path / "rows.pkl"
    with open(job, "wb") as f:
        pickle.dump((plan, parts), f)
    env = dict(os.environ)
    # The worker authenticates with nothing but what the plan carries.
    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        env.pop(key, None)
    subprocess.run(
        [sys.executable, "-c", _WORKER, str(job), str(result)],
        check=True,
        env=env,
        timeout=300,
    )
    with open(result, "rb") as f:
        rows: list[dict[str, Any]] = pickle.load(f)
    return rows


def _ids(rows: list[dict[str, Any]]) -> list[int]:
    return sorted(int(r["id"]) for r in rows)


class TestPlannedRead:
    def test_a_worker_reads_the_managed_table(
        self, live_connection: Any, scratch_sql: Any, tmp_path: Path
    ) -> None:
        name, _run = scratch_sql
        table = live_connection.table(name)
        if not table.can(Operation.SCAN).ok:
            pytest.skip(table.can(Operation.SCAN).reason)
        plan = table.plan_scan()
        assert plan.credential_expires_at is not None
        assert all(getattr(s, "scan_row", None) for s in plan.splits)
        rows = _read_in_a_worker(tmp_path, plan, plan.partitions(2))
        assert _ids(rows) == [1, 2, 3]

    def test_a_worker_applies_databricks_deletion_vectors(
        self, live_connection: Any, scratch_sql: Any, tmp_path: Path
    ) -> None:
        name, run = scratch_sql
        run(f"ALTER TABLE {name} SET TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')")
        run(f"DELETE FROM {name} WHERE id = 2")
        table = live_connection.table(name)
        if not table.can(Operation.SCAN).ok:
            pytest.skip(table.can(Operation.SCAN).reason)
        plan = table.plan_scan(columns=["id"])
        assert _ids(_read_in_a_worker(tmp_path, plan, plan.partitions(1))) == [1, 3]

    def test_a_worker_reads_the_change_feed(
        self, live_connection: Any, scratch_sql: Any, tmp_path: Path
    ) -> None:
        name, run = scratch_sql
        run(f"ALTER TABLE {name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
        start = live_connection.table(name).version + 1
        run(f"INSERT INTO {name} VALUES (4, 'accra'), (5, 'hanoi')")
        run(f"DELETE FROM {name} WHERE id = 1")
        table = live_connection.table(name)
        plan = table.plan_changes(start)
        rows = _read_in_a_worker(tmp_path, plan, plan.partitions(2))
        expected = table.cdf(starting_version=start).read_all().to_pylist()

        def key(r: dict[str, Any]) -> tuple[Any, ...]:
            return (r["_commit_version"], r["_change_type"], r["id"])

        assert sorted(map(key, rows)) == sorted(map(key, expected))

    def test_a_catalog_managed_table_is_read_by_a_worker(
        self, live_connection: Any, scratch_sql: Any, tmp_path: Path
    ) -> None:
        name, run = scratch_sql
        try:
            run(
                f"ALTER TABLE {name} SET TBLPROPERTIES "
                "('delta.feature.catalogManaged' = 'supported')"
            )
        except Exception as exc:
            pytest.skip(f"catalog commits are not available on this workspace ({exc})")
        run(f"INSERT INTO {name} VALUES (6, 'lagos')")
        table = live_connection.table(name)
        if not table.is_catalog_managed:
            pytest.skip("the table did not report catalogManaged after ALTER")
        plan = table.plan_scan()
        assert _ids(_read_in_a_worker(tmp_path, plan, plan.partitions(2))) == [1, 2, 3, 6]


class TestPlannedWrite:
    """This workspace refuses direct external writes to managed tables, so the
    refusal must come from plan_write, before any worker runs; where it is
    allowed, the job's commit must land."""

    def _plan_or_refusal(self, table: Any) -> Any:
        try:
            return table.plan_write(), None
        except DeltaSwampError as exc:
            return None, exc

    def test_a_managed_table_is_written_or_refused_at_planning(
        self, live_connection: Any, scratch_sql: Any
    ) -> None:
        name, _run = scratch_sql
        table = live_connection.table(name)
        plan, refusal = self._plan_or_refusal(table)
        if refusal is not None:
            text = str(refusal)
            assert any(
                hint in text
                for hint in ("catalog commits", "external", "allow_sql_fallback", "catalog-managed")
            ), text
            assert live_connection.table(name).to_arrow().num_rows == 3
            return
        fragment = plan.write(pa.table({"id": [10], "city": ["perth"]}))
        plan.commit([fragment])
        assert _ids(live_connection.table(name).to_arrow().to_pylist()) == [1, 2, 3, 10]

    def test_an_aborted_write_leaves_no_rows(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        plan, refusal = self._plan_or_refusal(live_connection.table(name))
        if refusal is not None:
            pytest.skip(f"direct writes are refused here: {refusal}")
        fragment = plan.write(pa.table({"id": [11], "city": ["kyiv"]}))
        plan.abort([fragment])
        assert live_connection.table(name).to_arrow().num_rows == 3


class TestGovernedRead:
    """A table only the warehouse can read: its result chunks are the splits."""

    @pytest.fixture
    def governed(self, live_config: Any, scratch_sql: Any) -> Iterator[tuple[str, str]]:
        """The scratch table behind a row filter (id <> 2) and a column mask on city."""
        name, run = scratch_sql
        suffix = uuid.uuid4().hex[:8]
        keep = f"{live_config.prefix}.dsw_keep_{suffix}"
        hide = f"{live_config.prefix}.dsw_hide_{suffix}"
        view = f"{live_config.prefix}.dsw_view_{suffix}"
        run(f"CREATE FUNCTION {keep}(id BIGINT) RETURN id <> 2")
        run(f"CREATE FUNCTION {hide}(city STRING) RETURN concat('masked-', length(city))")
        try:
            run(f"ALTER TABLE {name} SET ROW FILTER {keep} ON (id)")
            run(f"ALTER TABLE {name} ALTER COLUMN city SET MASK {hide}")
            run(f"CREATE VIEW {view} AS SELECT id, city FROM {name} WHERE id > 1")
            yield name, view
        finally:
            for statement in (
                f"DROP VIEW IF EXISTS {view}",
                f"ALTER TABLE {name} DROP ROW FILTER",
                f"ALTER TABLE {name} ALTER COLUMN city DROP MASK",
                f"DROP FUNCTION IF EXISTS {keep}",
                f"DROP FUNCTION IF EXISTS {hide}",
            ):
                with contextlib.suppress(Exception):
                    run(statement)

    @staticmethod
    def _rows(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
        return sorted((int(r["id"]), str(r["city"])) for r in rows)

    def test_a_worker_reads_the_filtered_and_masked_result(
        self, live_connection: Any, governed: Any, tmp_path: Path
    ) -> None:
        name, _view = governed
        table = live_connection.table(name)
        plan = table.plan_scan()
        assert getattr(plan, "is_warehouse_plan", False), plan
        expected = self._rows(table.to_arrow().to_pylist())
        assert expected == [(1, "masked-4"), (3, "masked-5")]
        assert self._rows(_read_in_a_worker(tmp_path, plan, plan.partitions(4))) == expected

    def test_projection_and_predicate_run_in_the_query(
        self, live_connection: Any, governed: Any, tmp_path: Path
    ) -> None:
        name, _view = governed
        plan = live_connection.table(name).plan_scan(columns=["id"], predicate="id >= 3")
        assert _read_in_a_worker(tmp_path, plan, plan.partitions(2)) == [{"id": 3}]

    def test_a_view_is_read_by_a_worker_with_shipped_auth(
        self, live_connection: Any, governed: Any, tmp_path: Path
    ) -> None:
        _name, view = governed
        plan = live_connection.table(view).plan_scan(ship_catalog_auth=True)
        assert plan.shipped_links is not None
        # Links as a worker finds them after the ones planned have expired:
        # none. It asks the warehouse for fresh ones with the shipped auth.
        stale = dataclasses.replace(
            plan, splits=tuple(dataclasses.replace(s, links=()) for s in plan.splits)
        )
        rows = _read_in_a_worker(tmp_path, stale, stale.partitions(2))
        assert self._rows(rows) == [(3, "masked-5")]
