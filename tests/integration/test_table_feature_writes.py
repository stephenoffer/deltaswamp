"""Writes to tables whose features delta-kernel 0.28 refuses to write.

Each feature either works on both write paths a connector uses -- a local
`Table.append` / `overwrite`, and a distributed `plan_write()` -> workers'
`plan.write()` -> the driver's `plan.commit()` -- or is refused when the write
is planned, before any worker has written a file:

* `checkpointProtection` (Delta 4.0), which binds only log cleanup.
* Generated columns, computed on the workers, and checked where given.
* Column invariants, evaluated with the table's CHECK constraints.
* Identity columns, numbered from values reserved above the high-water mark.
* The change data feed on a full overwrite, which needs no CDC files.
* User domain metadata set by the job's commit.
* In-commit timestamps written as the protocol has them, first in commitInfo.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import pickle
import threading
from typing import Any

import pyarrow as pa
import pytest
from deltaswamp.errors import InvalidArgumentError, UnreachableTableError


def _write_log(path: str, protocol: dict[str, Any], fields: list[Any], **config: str) -> None:
    """Version 0 of a table, as a newer or Databricks writer leaves it."""
    os.makedirs(os.path.join(path, "_delta_log"))
    actions = [
        {"commitInfo": {"timestamp": 1, "operation": "CREATE TABLE"}},
        {"protocol": protocol},
        {
            "metaData": {
                "id": "5b3c0a1e-0000-4000-8000-00000000000a",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": [],
                "configuration": config,
                "createdTime": 1,
            }
        },
    ]
    with open(os.path.join(path, "_delta_log", f"{0:020d}.json"), "w") as fh:
        fh.write("\n".join(json.dumps(a) for a in actions) + "\n")


def _commit(path: str, version: int) -> list[dict[str, Any]]:
    with open(os.path.join(path, "_delta_log", f"{version:020d}.json")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _schema(path: str, version: int | None = None) -> dict[str, Any]:
    """The newest schemaString at or below `version`."""
    logs = sorted(glob.glob(os.path.join(path, "_delta_log", "*.json")))
    found: dict[str, Any] = {}
    for log in logs:
        v = int(os.path.basename(log).split(".")[0])
        if version is not None and v > version:
            break
        for action in _commit(path, v):
            if "metaData" in action:
                found = json.loads(action["metaData"]["schemaString"])
    return found


def _parquet(path: str) -> set[str]:
    return set(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))


def _field(schema: dict[str, Any], name: str) -> dict[str, Any]:
    return next(f for f in schema["fields"] if f["name"] == name)


def _worker(plan: Any) -> Any:
    """The plan as a worker process receives it."""
    return pickle.loads(pickle.dumps(plan))


def _long(name: str, **metadata: Any) -> dict[str, Any]:
    return {"name": name, "type": "long", "nullable": True, "metadata": metadata}


# ---------------------------------------------------------------- checkpointProtection


class TestCheckpointProtection:
    """Kernel 0.28 has no variant for it and refused every write. It binds only
    which history a writer may delete, so every commit is written past it."""

    @pytest.fixture
    def path(self, tmp_path: Any) -> str:
        path = str(tmp_path / "t")
        _write_log(
            path,
            {
                "minReaderVersion": 1,
                "minWriterVersion": 7,
                "writerFeatures": ["checkpointProtection", "appendOnly"],
            },
            [_long("id")],
            **{"delta.requireCheckpointProtectionBeforeVersion": "0"},
        )
        return path

    def test_appends_overwrites_and_distributed_writes_commit(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        assert t.can("append").ok
        t.append(pa.table({"id": [1, 2]}))
        plan = conn.open_table(path).plan_write()
        plan.commit([_worker(plan).write(pa.table({"id": [3]}))])
        assert sorted(conn.open_table(path).to_arrow().column("id").to_pylist()) == [1, 2, 3]
        # The protocol the commits carry is still the table's own.
        protocols = [a["protocol"] for v in (1, 2) for a in _commit(path, v) if "protocol" in a]
        assert protocols == []

    def test_a_metadata_change_commits(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        assert t.can("add_column").ok
        t.add_column([pa.field("n", pa.string())])
        assert "n" in conn.open_table(path).schema().names

    def test_a_checkpoint_keeps_the_feature(self, conn: Any, path: str) -> None:
        t = conn.open_table(path)
        t.append(pa.table({"id": [1]}))
        conn.open_table(path).checkpoint()
        assert glob.glob(os.path.join(path, "_delta_log", "*.checkpoint.parquet"))
        fresh = conn.open_table(path)
        assert "checkpointProtection" in fresh._enrich().writer_features
        assert fresh.to_arrow().num_rows == 1

    def test_log_cleanup_is_still_refused(self, conn: Any, path: str) -> None:
        verdict = conn.open_table(path).can("cleanup_metadata")
        assert not verdict.ok
        assert "checkpointProtection" in verdict.reason


# ---------------------------------------------------------------- generated / invariants


class TestGeneratedColumnsAndInvariants:
    @pytest.fixture
    def path(self, tmp_path: Any) -> str:
        path = str(tmp_path / "t")
        _write_log(
            path,
            {
                "minReaderVersion": 1,
                "minWriterVersion": 7,
                "writerFeatures": ["invariants", "checkConstraints", "generatedColumns"],
            },
            [
                _long(
                    "v", **{"delta.invariants": json.dumps({"expression": {"expression": "v > 0"}})}
                ),
                _long("g", **{"delta.generationExpression": "v * 2"}),
                {
                    "name": "d",
                    "type": "string",
                    "nullable": True,
                    "metadata": {"delta.generationExpression": "concat('v', CAST(v AS STRING))"},
                },
            ],
            **{"delta.constraints.small": "v < 100"},
        )
        return path

    @staticmethod
    def _kernel(conn: Any, path: str) -> Any:
        """The table, its writes served by the kernel (delta-rs, preferred for
        local appends, evaluates all three itself)."""
        from deltaswamp.capability import Engine

        t = conn.open_table(path)
        return t, conn.router.engines[Engine.KERNEL]

    def test_a_local_kernel_append_computes_them(self, conn: Any, path: str) -> None:
        t, kernel = self._kernel(conn, path)
        kernel.append(t._enrich(), pa.table({"v": [1, 2]}))
        rows = conn.open_table(path).to_arrow().sort_by("v").to_pylist()
        assert rows == [{"v": 1, "g": 2, "d": "v1"}, {"v": 2, "g": 4, "d": "v2"}]

    def test_a_given_value_is_checked(self, conn: Any, path: str) -> None:
        t, kernel = self._kernel(conn, path)
        kernel.append(t._enrich(), pa.table({"v": [3], "g": [6]}))
        with pytest.raises(InvalidArgumentError, match="generated column g"):
            kernel.append(t._enrich(), pa.table({"v": [3], "g": [7]}))
        assert conn.open_table(path).to_arrow().num_rows == 1

    @pytest.mark.parametrize(("value", "names"), [(0, "invariant on v"), (100, "small")])
    def test_invariants_and_constraints_are_enforced(
        self, conn: Any, path: str, value: int, names: str
    ) -> None:
        t, kernel = self._kernel(conn, path)
        with pytest.raises(InvalidArgumentError, match=names):
            kernel.append(t._enrich(), pa.table({"v": [value]}))
        # Every engine refuses it.
        with pytest.raises(InvalidArgumentError):
            t.append(pa.table({"v": [value]}))

    def test_workers_compute_them_and_a_violation_writes_no_file(
        self, conn: Any, path: str
    ) -> None:
        plan = conn.open_table(path).plan_write()
        worker = _worker(plan)
        fragment = worker.write(pa.table({"v": [5, 6]}))
        before = _parquet(path)
        with pytest.raises(InvalidArgumentError, match="invariant on v"):
            worker.write(pa.table({"v": [7, -1]}))
        with pytest.raises(InvalidArgumentError, match="generated column g"):
            worker.write(pa.table({"v": [7], "g": [0]}))
        assert _parquet(path) == before
        plan.commit([fragment])
        rows = conn.open_table(path).to_arrow().sort_by("v").to_pylist()
        assert rows == [{"v": 5, "g": 10, "d": "v5"}, {"v": 6, "g": 12, "d": "v6"}]

    def test_an_expression_duckdb_cannot_evaluate_is_refused_at_plan_time(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "u")
        _write_log(
            path,
            {"minReaderVersion": 1, "minWriterVersion": 7, "writerFeatures": ["generatedColumns"]},
            [_long("v"), _long("g", **{"delta.generationExpression": "no_such_function(v)"})],
        )
        t = conn.open_table(path)
        with pytest.raises(UnreachableTableError, match="generated column g"):
            t.plan_write()

    def test_dml_is_still_left_to_other_engines(self, conn: Any, path: str) -> None:
        conn.open_table(path).append(pa.table({"v": [1]}))
        verdict = conn.open_table(path).can("update")
        assert "kernel" not in (verdict.engine.value if verdict.ok and verdict.engine else "")


# ---------------------------------------------------------------- identity


def _identity_table(path: str, *, explicit: bool = False, step: int = 1, start: int = 1) -> None:
    _write_log(
        path,
        {"minReaderVersion": 1, "minWriterVersion": 7, "writerFeatures": ["identityColumns"]},
        [
            _long(
                "id",
                **{
                    "delta.identity.start": start,
                    "delta.identity.step": step,
                    "delta.identity.allowExplicitInsert": explicit,
                },
            ),
            _long("v"),
        ],
    )


def _hwm(path: str, version: int | None = None) -> Any:
    return _field(_schema(path, version), "id")["metadata"].get("delta.identity.highWaterMark")


class TestIdentityColumnsLocally:
    def test_values_follow_the_high_water_mark_committed_with_them(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        _identity_table(path, start=10, step=5)
        conn.open_table(path).append(pa.table({"v": [1, 2, 3]}))
        # The mark moved in the very commit that used the values.
        assert any("metaData" in a for a in _commit(path, 1))
        assert _hwm(path) == 20
        conn.open_table(path).overwrite(pa.table({"v": [4]}))
        rows = conn.open_table(path).to_arrow().to_pylist()
        assert rows == [{"id": 25, "v": 4}]
        assert _hwm(path) == 25

    def test_generated_always_refuses_a_given_value(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _identity_table(path)
        with pytest.raises(UnreachableTableError, match="IDENTITY"):
            conn.open_table(path).append(pa.table({"id": [7], "v": [1]}))

    def test_by_default_keeps_a_given_value_and_the_mark(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _identity_table(path, explicit=True)
        conn.open_table(path).append(pa.table({"v": [1]}))
        conn.open_table(path).append(pa.table({"id": [100], "v": [2]}))
        assert _hwm(path) == 1  # as Spark: SYNC IDENTITY moves it, an insert does not
        conn.open_table(path).append(pa.table({"v": [3]}))
        ids = sorted(conn.open_table(path).to_arrow().column("id").to_pylist())
        assert ids == [1, 2, 100]

    def test_concurrent_appends_get_disjoint_values(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _identity_table(path)
        conn.open_table(path).append(pa.table({"v": [0]}))
        errors: list[Exception] = []

        def append(i: int) -> None:
            try:
                conn.open_table(path).append(pa.table({"v": [i] * 3}))
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=append, args=(i,)) for i in range(1, 5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        ids = conn.open_table(path).to_arrow().column("id").to_pylist()
        assert len(ids) == len(set(ids)) == 13
        assert max(ids) <= _hwm(path)


class TestIdentityColumnsDistributed:
    def test_a_plan_without_identity_tasks_is_refused(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _identity_table(path)
        with pytest.raises(UnreachableTableError, match="identity_tasks"):
            conn.open_table(path).plan_write()
        with pytest.raises(UnreachableTableError, match="GENERATED ALWAYS"):
            conn.open_table(path).plan_write(identity_tasks=0)

    def test_workers_number_rows_from_disjoint_slots(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _identity_table(path, step=-2, start=-1)
        plan = conn.open_table(path).plan_write(identity_tasks=3, identity_rows_per_task=4)
        # The reservation is a metadata-only commit, before any worker runs.
        assert _hwm(path) == -1 + (-2) * 11
        a, b = _worker(plan), _worker(plan)
        fragments = [
            a.write(pa.table({"v": [1, 2]}), task_index=0),
            a.write(pa.table({"v": [3]}), task_index=0),  # the same task's next block
            b.write(pa.table({"v": [4]}), task_index=2),
        ]
        version = plan.commit(fragments)
        # The data commit changes no metadata.
        assert not any("metaData" in x for x in _commit(path, version))
        rows = {r["v"]: r["id"] for r in conn.open_table(path).to_arrow().to_pylist()}
        assert rows == {1: -1, 2: -3, 3: -5, 4: -17}

    def test_a_task_that_overruns_its_slot_writes_nothing(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _identity_table(path)
        plan = conn.open_table(path).plan_write(identity_tasks=2, identity_rows_per_task=2)
        worker = _worker(plan)
        with pytest.raises(InvalidArgumentError, match="task_index"):
            worker.write(pa.table({"v": [1]}))
        with pytest.raises(InvalidArgumentError, match="identity_rows_per_task"):
            worker.write(pa.table({"v": [1, 2, 3]}), task_index=1)
        with pytest.raises(InvalidArgumentError, match="outside"):
            worker.write(pa.table({"v": [1]}), task_index=2)
        assert _parquet(path) == set()

    def test_two_jobs_reserve_disjoint_blocks(self, conn: Any, tmp_path: Any) -> None:
        path = str(tmp_path / "t")
        _identity_table(path)
        first = conn.open_table(path).plan_write(identity_tasks=1, identity_rows_per_task=5)
        second = conn.open_table(path).plan_write(identity_tasks=1, identity_rows_per_task=5)
        f1 = _worker(first).write(pa.table({"v": [1]}), task_index=0)
        f2 = _worker(second).write(pa.table({"v": [2]}), task_index=0)
        second.commit([f2])
        first.commit([f1])
        rows = {r["v"]: r["id"] for r in conn.open_table(path).to_arrow().to_pylist()}
        assert rows == {1: 1, 2: 6}

    def test_by_default_data_that_gives_every_value_reserves_none(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "t")
        _identity_table(path, explicit=True)
        plan = conn.open_table(path).plan_write(identity_tasks=0)
        plan.commit([_worker(plan).write(pa.table({"id": [42], "v": [1]}))])
        assert conn.open_table(path).to_arrow().to_pylist() == [{"id": 42, "v": 1}]
        assert _hwm(path) is None


# ---------------------------------------------------------------- change data feed


def test_a_distributed_overwrite_of_a_change_feed_table_reads_back_as_changes(
    conn: Any, tmp_path: Any
) -> None:
    path = str(tmp_path / "t")
    t = conn.create_table(
        path,
        pa.schema([("id", pa.int64())]),
        properties={"delta.enableChangeDataFeed": "true"},
    )
    t.append(pa.table({"id": [1, 2]}))
    plan = conn.open_table(path).plan_write(mode="overwrite")
    version = plan.commit([_worker(plan).write(pa.table({"id": [3]}))])
    feed = conn.open_table(path).cdf(starting_version=version, ending_version=version)
    changes = sorted((r["_change_type"], r["id"]) for r in pa.table(feed).to_pylist())
    assert changes == [("delete", 1), ("delete", 2), ("insert", 3)]


# ---------------------------------------------------------------- domain metadata


class TestDomainMetadata:
    @pytest.fixture
    def path(self, conn: Any, tmp_path: Any) -> str:
        path = str(tmp_path / "t")
        conn.create_table(
            path,
            pa.schema([("id", pa.int64())]),
            properties={"delta.feature.domainMetadata": "supported"},
        )
        return path

    def test_the_commit_sets_the_domain(self, conn: Any, path: str) -> None:
        plan = conn.open_table(path).plan_write(domain_metadata={"app.job": '{"run": 7}'})
        version = plan.commit([_worker(plan).write(pa.table({"id": [1]}))])
        domains = [a["domainMetadata"] for a in _commit(path, version) if "domainMetadata" in a]
        assert domains == [{"domain": "app.job", "configuration": '{"run": 7}', "removed": False}]
        assert any("add" in a for a in _commit(path, version))

    def test_a_system_domain_is_refused_at_plan_time(self, conn: Any, path: str) -> None:
        with pytest.raises(UnreachableTableError, match="system domain"):
            conn.open_table(path).plan_write(domain_metadata={"delta.rowTracking": "{}"})

    def test_a_table_without_the_feature_is_refused_at_plan_time(
        self, conn: Any, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "u")
        conn.create_table(
            path,
            pa.schema([("id", pa.int64())]),
            properties={"delta.enableDeletionVectors": "true"},
        )
        with pytest.raises(UnreachableTableError, match="domainMetadata"):
            conn.open_table(path).plan_write(domain_metadata={"app.job": "{}"})


# ---------------------------------------------------------------- in-commit timestamps


class TestInCommitTimestamps:
    @pytest.fixture
    def path(self, conn: Any, tmp_path: Any) -> str:
        path = str(tmp_path / "t")
        conn.create_table(
            path,
            pa.schema([("id", pa.int64())]),
            properties={
                "delta.enableInCommitTimestamps": "true",
                "delta.enableRowTracking": "true",
            },
        )
        return path

    def test_a_distributed_commit_writes_the_timestamp_first(self, conn: Any, path: str) -> None:
        plan = conn.open_table(path).plan_write(commit_metadata={"userMetadata": "job"})
        version = plan.commit([_worker(plan).write(pa.table({"id": [1]}))])
        with open(os.path.join(path, "_delta_log", f"{version:020d}.json")) as fh:
            first = fh.readline()
        info = json.loads(first)["commitInfo"]
        assert next(iter(info)) == "inCommitTimestamp"
        assert (
            info["inCommitTimestamp"]
            >= _commit(path, version - 1)[0]["commitInfo"]["inCommitTimestamp"]
        )

    def test_concurrent_distributed_appends_keep_row_ids_and_times_in_order(
        self, conn: Any, path: str
    ) -> None:
        plans = [conn.open_table(path).plan_write() for _ in range(3)]
        fragments = [
            _worker(p).write(pa.table({"id": list(range(i * 10, i * 10 + 4))}))
            for i, p in enumerate(plans)
        ]
        versions = [p.commit([f]) for p, f in zip(plans, fragments, strict=True)]
        assert versions == sorted(versions) and len(set(versions)) == 3
        stamps = [_commit(path, v)[0]["commitInfo"]["inCommitTimestamp"] for v in versions]
        assert stamps == sorted(stamps) and len(set(stamps)) == 3
        adds = [a["add"] for v in versions for a in _commit(path, v) if "add" in a]
        spans = sorted(
            (a["baseRowId"], a["baseRowId"] + json.loads(a["stats"])["numRecords"]) for a in adds
        )
        assert all(end <= nxt for (_, end), (nxt, _) in itertools.pairwise(spans))
        marks = [
            json.loads(a["domainMetadata"]["configuration"])["rowIdHighWaterMark"]
            for v in versions
            for a in _commit(path, v)
            if "domainMetadata" in a
        ]
        assert marks[-1] == spans[-1][1] - 1
