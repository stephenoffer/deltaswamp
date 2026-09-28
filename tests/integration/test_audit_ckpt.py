"""Checkpoints and checksums of tables only the kernel writes, once they carry
a CHECK constraint (or a generated column).

delta-rs refuses a table with in-commit timestamps (and row tracking,
clustering, type widening, column defaults), and the kernel's checkpoint and
checksum writers ran its write-protocol check, which refuses checkConstraints
and generatedColumns. So after ``add_constraint`` no engine could checkpoint
such a table: its log grew forever, ``cleanup_metadata`` could expire nothing,
and the ``.crc`` chain stopped. The kernel now writes both from a snapshot
whose checked protocol sets the value-constraint features aside
(crates/native/src/restate.rs, ``log_writing_snapshot``); the files hold the
table's own protocol and metadata, read from the log.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
deltalake = pytest.importorskip("deltalake")
ds = pytest.importorskip("deltaswamp")

from deltaswamp.capability import Engine  # noqa: E402
from deltaswamp.errors import InvalidArgumentError, UnreachableTableError  # noqa: E402

pytestmark = pytest.mark.skipif(not ds.has_native(), reason="native extension not built")

DAY = 86400.0
ICT = {"delta.enableInCommitTimestamps": "true"}


def _rows(n: int, off: int = 0) -> Any:
    return pa.table({"id": pa.array(range(off, off + n), pa.int64())})


def _log(path: str) -> pathlib.Path:
    return pathlib.Path(path, "_delta_log")


def _versions_of(path: str, suffix: str) -> list[int]:
    return sorted(
        int(f.name[:20])
        for f in _log(path).iterdir()
        if f.name[:20].isdigit() and f.name[20:] == suffix
    )


def _actions(path: str, version: int) -> list[dict[str, Any]]:
    text = (_log(path) / f"{version:020d}.json").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _latest(path: str, key: str, upto: int) -> dict[str, Any]:
    """The newest `key` action (protocol or metaData) at or below `upto`, from the JSON log."""
    found: dict[str, Any] | None = None
    for version in range(upto + 1):
        for action in _actions(path, version):
            if key in action:
                found = action[key]
    assert found is not None
    return found


def _checkpoint_rows(path: str, version: int) -> dict[str, Any]:
    rows = pq.read_table(_log(path) / f"{version:020d}.checkpoint.parquet").to_pylist()
    many = ("add", "remove", "txn", "domainMetadata", "sidecar")
    out: dict[str, Any] = {key: [] for key in many}
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            if key in many:
                out[key].append(value)
            else:
                assert key not in out, f"two {key} rows"
                out[key] = value
    return out


def _as_log_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """A checkpoint's metaData row in the log's JSON form (parquet maps are pair lists)."""
    return {
        "id": row["id"],
        "schemaString": row["schemaString"],
        "partitionColumns": list(row["partitionColumns"]),
        "configuration": dict(row["configuration"] or []),
        "createdTime": row["createdTime"],
        "format": {"provider": row["format"]["provider"]},
    }


def _norm_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "minReaderVersion": protocol["minReaderVersion"],
        "minWriterVersion": protocol["minWriterVersion"],
        "readerFeatures": sorted(protocol.get("readerFeatures") or []),
        "writerFeatures": sorted(protocol.get("writerFeatures") or []),
    }


def _norm_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": metadata["id"],
        "schemaString": metadata["schemaString"],
        "partitionColumns": list(metadata.get("partitionColumns") or []),
        "configuration": dict(metadata.get("configuration") or {}),
        "createdTime": metadata.get("createdTime"),
        "format": {"provider": metadata["format"]["provider"]},
    }


def _constrained(conn: Any, tmp_path: Any, *, appends: int = 3, **properties: str) -> str:
    """A kernel-only table (in-commit timestamps) with a CHECK constraint."""
    path = str(tmp_path / "t")
    conn.create_table(path, _rows(0).schema, properties={**ICT, **properties})
    conn.open_table(path).add_constraint({"pos": "id >= 0"})
    for i in range(appends):
        conn.open_table(path).append(_rows(2, 2 * i))
    return path


def _ids(conn: Any, path: str, version: int | None = None) -> list[int]:
    return sorted(conn.open_table(path).to_arrow(version=version).column("id").to_pylist())


def _deltars_ids(path: str) -> list[int]:
    dt = deltalake.DeltaTable(path)
    got = deltalake.QueryBuilder().register("t", dt).execute("select id from t").read_all()
    return sorted(got.column("id").to_pylist())


def _assert_checkpoint_is_the_log(path: str, version: int) -> dict[str, Any]:
    """The checkpoint at `version` holds exactly the log's protocol and metaData."""
    rows = _checkpoint_rows(path, version)
    assert _norm_protocol(rows["protocol"]) == _norm_protocol(_latest(path, "protocol", version))
    assert _as_log_metadata(rows["metaData"]) == _norm_metadata(_latest(path, "metaData", version))
    last = json.loads((_log(path) / "_last_checkpoint").read_text())
    assert last["version"] == version
    assert last["numOfAddFiles"] == len(rows["add"])
    return rows


def _assert_checksums(path: str, versions: range) -> None:
    """Every version's .crc holds the log's protocol and metaData and the live files."""
    sizes: dict[str, int] = {}
    for version in versions:
        for action in _actions(path, version):
            if "add" in action:
                sizes[action["add"]["path"]] = action["add"]["size"]
            elif "remove" in action:
                sizes.pop(action["remove"]["path"], None)
        crc_file = _log(path) / f"{version:020d}.crc"
        assert crc_file.exists(), f"no checksum at {version}"
        crc = json.loads(crc_file.read_text())
        assert _norm_protocol(crc["protocol"]) == _norm_protocol(_latest(path, "protocol", version))
        assert _norm_metadata(crc["metadata"]) == _norm_metadata(_latest(path, "metaData", version))
        assert crc["numFiles"] == len(sizes), version
        assert crc["tableSizeBytes"] == sum(sizes.values()), version
        histogram = crc.get("fileSizeHistogram")
        if histogram:
            assert sum(histogram["fileCounts"]) == len(sizes)
            assert sum(histogram["totalBytes"]) == sum(sizes.values())
        commit = _actions(path, version)[0]["commitInfo"]
        assert crc["inCommitTimestampOpt"] == commit["inCommitTimestamp"]


@pytest.mark.parametrize("policy", ["classic", "v2"])
def test_a_constrained_kernel_only_table_checkpoints(
    conn: Any, tmp_path: Any, policy: str, monkeypatch: Any
) -> None:
    """The reported gap: UnreachableTableError, on classic and v2 checkpoints."""
    monkeypatch.setenv("DELTASWAMP_STRICT_ROUTING", "1")
    path = _constrained(conn, tmp_path, **{"delta.checkpointPolicy": policy})
    t = conn.open_table(path)
    verdict = t.can("checkpoint")
    assert verdict.ok and verdict.engine is Engine.KERNEL, verdict
    t.checkpoint()
    assert _versions_of(path, ".checkpoint.parquet") == [4]
    rows = _assert_checkpoint_is_the_log(path, 4)
    assert "checkConstraints" in rows["protocol"]["writerFeatures"]
    assert dict(rows["metaData"]["configuration"])["delta.constraints.pos"] == "id >= 0"
    assert ("checkpointMetadata" in rows) is (policy == "v2")
    # Every commit's checksum, including the appends after ADD CONSTRAINT,
    # holds the table's real protocol and files.
    _assert_checksums(path, range(0, 5))
    assert _ids(conn, path) == list(range(6))
    assert _deltars_ids(path) == list(range(6))


@pytest.mark.parametrize("policy", ["classic", "v2"])
def test_the_checkpoint_alone_reads_as_the_table(conn: Any, tmp_path: Any, policy: str) -> None:
    """A copy with every commit below the checkpoint deleted: the kernel and
    delta-rs load it from the checkpoint, and the constraint still binds."""
    path = _constrained(conn, tmp_path, **{"delta.checkpointPolicy": policy})
    conn.open_table(path).checkpoint()
    copy = str(tmp_path / "copy")
    shutil.copytree(path, copy)
    for f in _log(copy).iterdir():
        if f.name[:20].isdigit() and int(f.name[:20]) < 4:
            f.unlink()
    t = conn.open_table(copy)
    assert t.version == 4
    assert _ids(conn, copy) == list(range(6))
    assert t.properties()["delta.constraints.pos"] == "id >= 0"
    from deltaswamp._native import Snapshot

    snapshot = Snapshot.resolve(copy)
    reader, writer, readers, writers = snapshot.protocol()
    assert _norm_protocol(
        {
            "minReaderVersion": reader,
            "minWriterVersion": writer,
            "readerFeatures": readers,
            "writerFeatures": writers,
        }
    ) == _norm_protocol(_latest(path, "protocol", 4))
    assert json.loads(snapshot.metadata_json())["configuration"]["delta.constraints.pos"] == (
        "id >= 0"
    )
    assert _deltars_ids(copy) == list(range(6))
    with pytest.raises(InvalidArgumentError, match="pos"):
        conn.open_table(copy).append(_rows(1, -5))
    conn.open_table(copy).append(_rows(1, 100))
    assert _ids(conn, copy) == [*range(6), 100]


def test_checkpoints_land_at_the_checkpoint_interval(conn: Any, tmp_path: Any) -> None:
    """Kernel writes checkpoint at delta.checkpointInterval; they failed silently here."""
    path = _constrained(conn, tmp_path, appends=6, **{"delta.checkpointInterval": "3"})
    assert _versions_of(path, ".checkpoint.parquet") == [3, 6]
    _assert_checkpoint_is_the_log(path, 6)
    _assert_checksums(path, range(0, 8))


def test_optimize_then_checkpoint(conn: Any, tmp_path: Any) -> None:
    path = _constrained(conn, tmp_path, appends=4)
    t = conn.open_table(path)
    t.optimize()
    conn.open_table(path).checkpoint()
    version = conn.open_table(path).version
    rows = _assert_checkpoint_is_the_log(path, version)
    assert len(rows["add"]) == 1
    _assert_checksums(path, range(0, version + 1))
    assert _ids(conn, path) == list(range(8))


def _shift_ict(path: str, days: float) -> None:
    """Move every in-commit timestamp `days` into the past (commits and checksums)."""
    shift = int(days * DAY * 1000)
    for f in sorted(_log(path).iterdir()):
        if f.name.endswith(".json") and f.name[:20].isdigit():
            actions = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            for action in actions:
                info = action.get("commitInfo")
                if info and "inCommitTimestamp" in info:
                    info["inCommitTimestamp"] -= shift
                config = (action.get("metaData") or {}).get("configuration") or {}
                key = "delta.inCommitTimestampEnablementTimestamp"
                if key in config:
                    config[key] = str(int(config[key]) - shift)
            f.write_text("\n".join(json.dumps(a) for a in actions) + "\n")
        elif f.name.endswith(".crc"):
            crc = json.loads(f.read_text())
            if crc.get("inCommitTimestampOpt") is not None:
                crc["inCommitTimestampOpt"] -= shift
            f.write_text(json.dumps(crc))


def test_cleanup_metadata_expires_the_log_behind_the_checkpoint(conn: Any, tmp_path: Any) -> None:
    """With no checkpoint, the kernel's log cleanup had nothing it could delete."""
    path = _constrained(conn, tmp_path, appends=5)
    conn.open_table(path).checkpoint()  # at 6
    conn.open_table(path).append(_rows(1, 50))
    _shift_ict(path, 40)
    for f in _log(path).iterdir():
        os.utime(f)
    conn.open_table(path).cleanup_metadata()
    assert _versions_of(path, ".json") == [6, 7]
    assert _ids(conn, path) == [*range(10), 50]
    assert _ids(conn, path, version=6) == list(range(10))
    with pytest.raises(InvalidArgumentError, match="pos"):
        conn.open_table(path).append(_rows(1, -1))


def _add_generated_column(path: str) -> int:
    """Commit, by hand, a generated column `g` (and the generatedColumns feature)."""
    version = max(_versions_of(path, ".json"))
    protocol = dict(_latest(path, "protocol", version))
    protocol["writerFeatures"] = sorted({*protocol["writerFeatures"], "generatedColumns"})
    metadata = dict(_latest(path, "metaData", version))
    schema = json.loads(metadata["schemaString"])
    schema["fields"].append(
        {
            "name": "g",
            "type": "long",
            "nullable": True,
            "metadata": {"delta.generationExpression": "id * 2"},
        }
    )
    metadata["schemaString"] = json.dumps(schema)
    ict = _actions(path, version)[0]["commitInfo"]["inCommitTimestamp"] + 1
    info = {"inCommitTimestamp": ict, "timestamp": ict, "operation": "ADD COLUMNS"}
    body = [{"commitInfo": info}, {"protocol": protocol}, {"metaData": metadata}]
    (_log(path) / f"{version + 1:020d}.json").write_text(
        "\n".join(json.dumps(a) for a in body) + "\n"
    )
    return version + 1


def test_a_real_generated_column_is_checkpointed_but_not_written(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """A checkpoint computes no generated value, so the kernel takes it; a write
    would have to, so it is still refused (and delta-rs lacks inCommitTimestamp)."""
    monkeypatch.setenv("DELTASWAMP_STRICT_ROUTING", "1")
    path = _constrained(conn, tmp_path, appends=2)
    version = _add_generated_column(path)
    t = conn.open_table(path)
    assert not t.can("append").ok
    with pytest.raises(UnreachableTableError):
        t.append(pa.table({"id": pa.array([9], pa.int64()), "g": pa.array([18], pa.int64())}))
    verdict = t.can("checkpoint")
    assert verdict.ok and verdict.engine is Engine.KERNEL, verdict
    t.checkpoint()
    rows = _assert_checkpoint_is_the_log(path, version)
    assert {"generatedColumns", "checkConstraints"} <= set(rows["protocol"]["writerFeatures"])
    field = json.loads(rows["metaData"]["schemaString"])["fields"][-1]
    assert field["metadata"]["delta.generationExpression"] == "id * 2"
    assert (_log(path) / f"{version:020d}.crc").exists()
    assert _ids(conn, path) == list(range(4))


def test_write_checksum_counts_a_constrained_table(conn: Any, tmp_path: Any) -> None:
    """The explicit call: a checksum lost at a checkpointed version is rebuilt
    by the kernel from the checkpoint, as the carried one had it."""
    from deltaswamp._native import Snapshot

    path = _constrained(conn, tmp_path, appends=3)
    conn.open_table(path).checkpoint()
    carried = json.loads((_log(path) / f"{4:020d}.crc").read_text())
    for v in (3, 4):
        (_log(path) / f"{v:020d}.crc").unlink()
    assert Snapshot.resolve(path).write_checksum(always=True)
    rebuilt = json.loads((_log(path) / f"{4:020d}.crc").read_text())
    for key in ("numFiles", "tableSizeBytes", "fileSizeHistogram", "inCommitTimestampOpt"):
        assert rebuilt[key] == carried[key], key
    assert _norm_protocol(rebuilt["protocol"]) == _norm_protocol(carried["protocol"])
    assert _norm_metadata(rebuilt["metadata"]) == _norm_metadata(carried["metadata"])
