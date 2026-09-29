"""Tables deltaswamp writes, as Databricks reads them -- and back again.

Every case builds a table locally through a deltaswamp operation history
(appends through the kernel and delta-rs, transactional and distributed
writes, DML through each engine, DDL, OPTIMIZE, checkpoints, log compaction,
VACUUM, metadata cleanup), uploads the table directory to a Unity Catalog
volume, and checks it on a SQL warehouse:

(a) the rows agree (EXCEPT ALL both ways) at the latest version, at three
    earlier ones through VERSION AS OF, and after VACUUM;
(b) DESCRIBE DETAIL agrees on features, properties, protocol, partitioning
    and file count;
(c) Databricks UPDATE, DELETE, MERGE, INSERT, OPTIMIZE, VACUUM DRY RUN and
    REORG PURGE succeed on the table, deltaswamp reads every version
    Databricks wrote identically, and appends on top in a way Databricks
    reads back;
(d) table_changes() equals cdf() on change-data-feed tables;
(e) selective predicates on every column type -- decimal extremes, NaN,
    pre-1582 dates, pre-1900 timestamps, long strings, struct fields --
    return the same rows on Databricks as deltaswamp reads (data skipping on
    the log stats and the Parquet footers);
(f) the data files deltaswamp wrote name a Spark version in their footer, or
    hold no value Databricks' legacy calendar rebase would shift.

Opt-in on top of the live suite, because it runs a few hundred statements:

    DELTASWAMP_TEST_INTEROP=1 pytest tests/live/test_live_interop.py -v

It needs DELTASWAMP_TEST_WAREHOUSE_ID. DELTASWAMP_TEST_INTEROP_SHAPES=plain,dv
selects shapes; DELTASWAMP_TEST_INTEROP_WORKERS (default 6) sets how many
cases run at once. All state lives on one volume with a random name, dropped
at the end of the session whatever happens.

Open bugs are marked xfail(strict=True) per check, so the suite fails when
one is fixed until the mark comes off.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("deltalake")

from tests.live.conftest import LiveConfig
from tests.live.interop import (
    BY_ID,
    CASES,
    READ_KEYS,
    STATS_GROUPS,
    Case,
    Context,
    Diff,
    FooterAudit,
    Predicates,
    Refused,
    Roundtrip,
    Runner,
    Skip,
    Staging,
    Step,
    Warehouse,
    check_keys,
)

pytestmark = [pytest.mark.databricks, pytest.mark.filterwarnings("ignore::UserWarning")]


def _selected() -> list[Case]:
    wanted = {s.strip() for s in os.environ.get("DELTASWAMP_TEST_INTEROP_SHAPES", "").split(",")}
    wanted.discard("")
    if not wanted:
        return CASES
    return [c for c in CASES if c.shape in wanted or c.id in wanted]


SELECTED = _selected()


def _params(*keys: str) -> list[Any]:
    """(case id, key) for every selected case the key applies to, xfail marks attached."""
    out = []
    for case in SELECTED:
        for key in keys:
            if key not in check_keys(case):
                continue
            marks = []
            if key in case.known:
                marks.append(pytest.mark.xfail(strict=True, reason=case.known[key]))
            out.append(pytest.param(case.id, key, id=f"{case.id}-{key}", marks=marks))
    return out


# ------------------------------------------------------------------ fixtures


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


@pytest.fixture(scope="session")
def interop(
    live_config: LiveConfig, live_connection: Any, live_workspace: Any, tmp_path_factory: Any
) -> Any:
    """Run every selected case in the background, on a volume made for this session.

    The volume's name is random, so a crashed run never collides with the
    next, and it is dropped -- with everything uploaded to it, the tables
    Databricks wrote included -- even when a test fails or the run is
    interrupted.
    """
    if not _truthy("DELTASWAMP_TEST_INTEROP"):
        pytest.skip("the interop suite is opt-in: set DELTASWAMP_TEST_INTEROP=1")
    if not live_config.warehouse_id:
        pytest.skip("DELTASWAMP_TEST_WAREHOUSE_ID is required: the warehouse is the oracle")
    if not SELECTED:
        pytest.skip("DELTASWAMP_TEST_INTEROP_SHAPES selects no case")

    volume = f"{live_config.prefix}.dsi_{uuid.uuid4().hex[:10]}"
    try:
        live_connection.create_volume(volume)
    except Exception as exc:
        pytest.skip(f"cannot create a scratch volume in {live_config.prefix}: {exc}")
    runner: Runner | None = None
    try:
        ctx = Context(
            warehouse=Warehouse(live_workspace, live_config.warehouse_id),
            staging=Staging(live_connection.volume(volume)),
            local=tmp_path_factory.mktemp("interop"),
        )
        workers = int(os.environ.get("DELTASWAMP_TEST_INTEROP_WORKERS") or 6)
        runner = Runner(SELECTED, ctx, workers)
        yield runner
    finally:
        if runner is not None:
            runner.close()
        _drop_volume(live_connection, volume)


def _drop_volume(conn: Any, volume: str) -> None:
    for attempt in range(3):
        try:
            conn.drop_volume(volume)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)


def evidence(interop: Runner, case_id: str, key: str) -> Any:
    """What the case gathered for `key`, waiting for the case to finish."""
    result = interop.result(case_id)
    if key in result.errors:
        pytest.fail(f"gathering {key} failed:\n{result.errors[key]}", pytrace=False)
    if key not in result.evidence:
        pytest.fail(f"{key} was never gathered: an earlier stage of {case_id} failed")
    got = result.evidence[key]
    if isinstance(got, Skip):
        pytest.skip(got.reason)
    return got


def _assert_same(diff: Diff | Refused) -> None:
    if isinstance(diff, Refused):
        return  # both sides refuse the range: agreement
    assert diff.ok, diff.explain()


# ---------------------------------------------------------------- the build


@pytest.mark.parametrize(("case_id", "key"), _params("history"))
def test_history(interop: Runner, case_id: str, key: str) -> None:
    """Each step commits, or is refused with a typed error that writes nothing."""
    steps: list[Step] = evidence(interop, case_id, key)
    errors = [f"{s.name}: {s.detail}" for s in steps if s.outcome == "error"]
    assert not errors, "steps failed with an untyped error:\n" + "\n".join(errors)
    wrote = [s.name for s in steps if s.outcome == "refused" and s.before != s.after]
    assert not wrote, f"refused steps still committed: {wrote}"
    unexpected = [
        f"{s.name}: {s.outcome} {s.detail}" for s in steps if s.expect and s.outcome != s.expect
    ]
    assert not unexpected, "steps ended otherwise than the history requires:\n" + "\n".join(
        unexpected
    )


@pytest.mark.parametrize(("case_id", "key"), _params("footers"))
def test_footers(interop: Runner, case_id: str, key: str) -> None:
    """(f) Kernel files name a Spark version; unmarked delta-rs files hold no early value."""
    audit: FooterAudit = evidence(interop, case_id, key)
    assert audit.kernel + audit.deltars, "the audit found no data file deltaswamp wrote"
    assert not audit.kernel_unmarked, (
        f"kernel files without org.apache.spark.version: {audit.kernel_unmarked}"
    )
    assert not audit.early_in_unmarked, (
        "files without org.apache.spark.version hold dates before 1582-10-15 or timestamps "
        f"before 1900, which Databricks reads shifted: {audit.early_in_unmarked}"
    )


# -------------------------------------------------------------------- reads


@pytest.mark.parametrize(("case_id", "key"), _params(*READ_KEYS))
def test_reads_agree(interop: Runner, case_id: str, key: str) -> None:
    """(a) Databricks and deltaswamp see the same rows, now, back in time, after VACUUM."""
    _assert_same(evidence(interop, case_id, key))


@pytest.mark.parametrize(("case_id", "key"), _params("detail-pre", "detail-final"))
def test_describe_detail(interop: Runner, case_id: str, key: str) -> None:
    """(b) DESCRIBE DETAIL agrees with deltaswamp's features, properties, protocol, layout."""
    mismatches: dict[str, Any] = evidence(interop, case_id, key)
    assert not mismatches, mismatches


# --------------------------------------------------------------- change feed

_CDF = (
    "cdf-full",
    "cdf-window",
    "cdf-before-add-column",
    "cdf-kernel-dml",
    "cdf-databricks-writes",
)


@pytest.mark.parametrize(("case_id", "key"), _params(*_CDF))
def test_change_feed(interop: Runner, case_id: str, key: str) -> None:
    """(d) table_changes() and cdf() return the same changes over the same range."""
    _assert_same(evidence(interop, case_id, key))


# -------------------------------------------------------------- data skipping


@pytest.mark.parametrize(("case_id", "key"), _params(*[f"stats-{g}" for g in STATS_GROUPS]))
def test_stats_predicates(interop: Runner, case_id: str, key: str) -> None:
    """(e) Selective predicates return deltaswamp's row counts on Databricks."""
    got: Predicates = evidence(interop, case_id, key)
    assert got.checked, f"no column in the {got.group} group: the battery lost its coverage"
    wrong = "\n".join(f"  {p}: Databricks {n}, expected {e}" for p, n, e in got.wrong[:25])
    assert not got.wrong, f"{len(got.wrong)} of {got.checked} predicates disagree:\n{wrong}"


@pytest.mark.parametrize(("case_id", "key"), _params("stats-coverage"))
def test_stats_cover_every_group(interop: Runner, case_id: str, key: str) -> None:
    """The case declares every predicate group its table holds, so none goes untested."""
    declared, found = evidence(interop, case_id, key)
    assert declared == found


# --------------------------------------------------------- Databricks writes


@pytest.mark.parametrize(("case_id", "key"), _params("databricks-dml"))
def test_databricks_writes(interop: Runner, case_id: str, key: str) -> None:
    """(c) Databricks DML and maintenance run on the table deltaswamp wrote."""
    results: list[tuple[str, str | None]] = evidence(interop, case_id, key)
    failed = [f"{sql[:120]}: {err}" for sql, err in results if err]
    assert not failed, "\n".join(failed)


@pytest.mark.parametrize(("case_id", "key"), _params("roundtrip-read"))
def test_reads_what_databricks_wrote(interop: Runner, case_id: str, key: str) -> None:
    """(c) deltaswamp reads every version Databricks committed as Databricks does."""
    _assert_same(evidence(interop, case_id, key))


@pytest.mark.parametrize(("case_id", "key"), _params("roundtrip-append"))
def test_writes_on_what_databricks_wrote(interop: Runner, case_id: str, key: str) -> None:
    """(c) deltaswamp appends to Databricks' result, and Databricks reads it back.

    DELETE and a checkpoint are tried too; either may be refused (typed, with
    nothing written) where an engine cannot serve the table Databricks left.
    """
    got: Roundtrip = evidence(interop, case_id, key)
    bad = [f"{s.name}: {s.outcome} {s.detail}" for s in got.steps if s.outcome == "error"]
    bad += [
        f"{s.name}: required, {s.outcome} {s.detail}"
        for s in got.steps
        if s.expect and s.outcome != s.expect
    ]
    bad += [
        f"{s.name}: refused but committed"
        for s in got.steps
        if s.outcome == "refused" and s.before != s.after
    ]
    assert not bad, "\n".join(bad)
    assert got.diff is not None
    _assert_same(got.diff)


@pytest.mark.parametrize(("case_id", "key"), _params("variant-merge-null-element"))
def test_variant_merge_null_element(interop: Runner, case_id: str, key: str) -> None:
    """MERGE a row whose ARRAY<VARIANT> holds a NULL element into a Databricks-made table."""
    merged = evidence(interop, case_id, key)
    assert [r["id"] for r in merged] == [7003]
    assert len(merged[0]["av"]) == 3 and merged[0]["av"][1] is None


@pytest.mark.parametrize(("case_id", "key"), _params(*sorted({k for c in CASES for k in c.checks})))
def test_case_checks(interop: Runner, case_id: str, key: str) -> None:
    """What a case checks beyond the common pipeline (identity values unique
    across writers, domains kept, generated values right, ...)."""
    problems: list[str] = evidence(interop, case_id, key)
    assert not problems, "\n".join(problems)


def test_selection_is_known() -> None:
    """A typo in DELTASWAMP_TEST_INTEROP_SHAPES should not pass as an empty run."""
    wanted = {s.strip() for s in os.environ.get("DELTASWAMP_TEST_INTEROP_SHAPES", "").split(",")}
    wanted.discard("")
    known = {c.shape for c in CASES} | set(BY_ID)
    assert not wanted - known, f"unknown shapes {sorted(wanted - known)}; known: {sorted(known)}"
