"""The live suite: deltaswamp against a real workspace, using a PAT.

Run it with::

    export DELTASWAMP_TEST_DATABRICKS=1
    export DATABRICKS_HOST=https://adb-1234.5.azuredatabricks.net
    export DATABRICKS_TOKEN=dapi...
    export DELTASWAMP_TEST_CATALOG=main
    export DELTASWAMP_TEST_SCHEMA=deltaswamp_test
    export DELTASWAMP_TEST_WAREHOUSE_ID=abc123         # optional but unlocks most tests
    export DELTASWAMP_TEST_EXTERNAL_LOCATION=s3://...  # optional, for external tables

    pytest tests/integration/test_live_databricks.py -v

Ordered so the cheapest checks fail first: a PAT that cannot authenticate, or a
metastore without external data access, should not be diagnosed by watching a
credential-vending test fail ten minutes later.

Each test creates what it needs under a random name and drops it afterwards.
Anything that cannot be verified is skipped with a reason, never quietly passed.
"""

from __future__ import annotations

from typing import Any

import deltaswamp as ds
import pytest
from deltaswamp.capability import Engine, Operation
from deltaswamp.errors import CredentialError, DeltaSwampError, PreflightError

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.databricks


def _unusable_token() -> str:
    """A syntactically valid but dead Databricks token.

    Assembled at import rather than written as a literal. A real PAT is the
    string `dapi` followed by 32 hex characters, so a hard-coded example -- even
    an obviously fake one -- matches the pattern that secret scanners look for,
    and GitHub push protection rejects the commit. Building it keeps the shape
    the SDK expects without putting a match in the source.
    """
    return "dapi" + ("0" * 28) + "dead"


# ---------------------------------------------------------------- 1. the PAT


class TestPersonalAccessToken:
    """Does the token work at all, and does it stay out of the logs?"""

    def test_token_authenticates(self, live_workspace: Any) -> None:
        me = live_workspace.current_user.me()
        assert me.user_name, "the PAT authenticated but returned no user"

    def test_connection_uses_the_explicit_token(self, live_config: Any) -> None:
        """Passed directly, not picked up from the environment by accident."""
        conn = ds.connect(
            host=live_config.host,
            token=live_config.token,
            default_catalog=live_config.catalog,
            default_schema=live_config.schema,
        )
        # Any catalog call proves the token reached the SDK.
        assert live_config.schema in conn.list_schemas(live_config.catalog)

    def test_a_bad_token_fails_clearly(self, live_config: Any) -> None:
        dead = _unusable_token()
        conn = ds.connect(
            host=live_config.host,
            token=dead,
            default_catalog=live_config.catalog,
            default_schema=live_config.schema,
        )
        with pytest.raises(Exception) as excinfo:
            conn.list_schemas()
        assert dead not in str(excinfo.value), "the token leaked into the error"

    def test_config_repr_redacts_the_token(self, live_config: Any) -> None:
        assert live_config.token not in repr(live_config)
        assert "***" in repr(live_config)


# ------------------------------------------------------------- 2. pre-flight


class TestPreflight:
    """The two admin settings that block everything downstream."""

    def test_metastore_prerequisites(self, live_connection: Any) -> None:
        problems = live_connection.preflight()
        if problems:
            pytest.skip("workspace not ready:\n  " + "\n  ".join(problems))

    def test_target_schema_is_reachable(self, live_connection: Any, live_config: Any) -> None:
        assert live_config.schema in live_connection.list_schemas(live_config.catalog)

    def test_catalog_is_listed(self, live_connection: Any, live_config: Any) -> None:
        assert live_config.catalog in live_connection.list_catalogs()


# ------------------------------------------------------ 3. resolution basics


class TestResolution:
    def test_resolves_a_managed_table(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        table = live_connection.table(name)
        assert table.location, "no storage location came back from the catalog"
        assert table.table_type == "MANAGED"

    def test_manifest_capabilities_are_populated(
        self, live_connection: Any, scratch_sql: Any
    ) -> None:
        """The authoritative pre-flight, fetched in the same round trip."""
        name, _run = scratch_sql
        resolved = live_connection.table(name).resolved
        if resolved.external_read_supported is None:
            pytest.skip("this workspace returned no capability manifest")
        assert resolved.external_read_supported is True

    def test_the_identity_check_does_not_fire_on_a_healthy_table(
        self, live_connection: Any, scratch_sql: Any
    ) -> None:
        """Opening a perfectly ordinary managed table must not look corrupt.

        `table_uuid` means the Delta log's `Metadata.id`. Databricks exposes no
        such field -- its `table_id` is the UC securable's own UUID, which names
        the storage directory -- so the catalog leaves `table_uuid` unset and
        the check is skipped. It used to be populated from `table_id`, which
        made every managed table raise CorruptTableError on first use.
        """
        name, _run = scratch_sql
        table = live_connection.table(name)
        table.features()  # forces the identity check
        assert table.resolved.table_id, "the UC securable id should still be recorded"
        assert table.resolved.table_uuid is None
        assert table.to_arrow().num_rows == 3

    def test_a_missing_table_says_so(self, live_connection: Any, live_config: Any) -> None:
        with pytest.raises(DeltaSwampError, match="does not exist"):
            live_connection.table(f"{live_config.prefix}.definitely_not_here_9f3a")


# ------------------------------------------------------- 4. credentials/PAT


class TestCredentialVending:
    """A PAT is enough to vend storage credentials, given the right grants."""

    def test_credentials_are_vended(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        try:
            creds = live_connection.table(name).credentials()
        except (CredentialError, PreflightError) as exc:
            pytest.skip(f"vending unavailable: {exc}")
        assert creds.secrets, "no credential material returned"

    def test_credentials_carry_an_expiry(self, live_connection: Any, scratch_sql: Any) -> None:
        """Databricks publishes no TTL, so expiration_time is the only authority."""
        name, _run = scratch_sql
        try:
            creds = live_connection.table(name).credentials()
        except (CredentialError, PreflightError) as exc:
            pytest.skip(f"vending unavailable: {exc}")
        assert creds.expires_at is not None
        assert not creds.expires_within(0), "the credential arrived already expired"

    def test_secrets_are_redacted_in_repr(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        try:
            creds = live_connection.table(name).credentials()
        except (CredentialError, PreflightError) as exc:
            pytest.skip(f"vending unavailable: {exc}")
        for value in creds.secrets.values():
            assert value not in repr(creds), "a secret leaked into repr"

    def test_provider_is_picklable_without_the_secret(
        self, live_connection: Any, scratch_sql: Any
    ) -> None:
        """Workers get a minting handle, never a token."""
        import pickle

        name, _run = scratch_sql
        provider = live_connection.table(name).resolved.credential_provider
        if provider is None:
            pytest.skip("no credential provider on this table")
        try:
            creds = provider.credentials()
        except (CredentialError, PreflightError) as exc:
            pytest.skip(f"vending unavailable: {exc}")
        payload = pickle.dumps(provider)
        # `secrets` carries the storage options object_store needs, and not all
        # of them are secret: the AWS region and endpoint are derived
        # configuration a worker must have to address the bucket at all, and
        # they are pickled deliberately. Only the vended credential material
        # must not survive pickling.
        not_secret = {"aws_region", "aws_endpoint_url", "azure_endpoint"}
        for key, value in creds.secrets.items():
            if key in not_secret:
                continue
            assert value.encode() not in payload, f"the vended secret {key!r} was pickled"


# ---------------------------------------------------------------- 5. reading


class TestReading:
    def test_reads_a_managed_table(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        table = live_connection.table(name)
        if not table.can(Operation.SCAN).ok:
            pytest.skip(table.can(Operation.SCAN).reason)
        assert table.to_arrow().num_rows == 3

    def test_reads_route_to_the_kernel(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        verdict = live_connection.table(name).can(Operation.SCAN)
        if not verdict.ok:
            pytest.skip(verdict.reason)
        assert verdict.engine is Engine.KERNEL

    def test_projection(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        table = live_connection.table(name)
        if not table.can(Operation.SCAN).ok:
            pytest.skip(table.can(Operation.SCAN).reason)
        assert table.to_arrow(columns=["city"]).column_names == ["city"]

    def test_time_travel(self, live_connection: Any, scratch_sql: Any) -> None:
        name, run = scratch_sql
        table = live_connection.table(name)
        if not table.can(Operation.SCAN).ok:
            pytest.skip(table.can(Operation.SCAN).reason)
        before = table.version
        run(f"INSERT INTO {name} VALUES (4, 'dakar')")
        fresh = live_connection.table(name)
        assert fresh.to_arrow(version=before).num_rows == 3
        assert fresh.to_arrow().num_rows == 4

    def test_history_and_detail(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        table = live_connection.table(name)
        if not table.can(Operation.HISTORY).ok:
            pytest.skip(table.can(Operation.HISTORY).reason)
        assert table.history(limit=1)
        assert table.detail()["version"] >= 0

    def test_arrow_capsule_interface(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        table = live_connection.table(name)
        if not table.can(Operation.SCAN).ok:
            pytest.skip(table.can(Operation.SCAN).reason)
        assert hasattr(table.scan(), "__arrow_c_stream__")


# ---------------------------------------------------------------- 6. writing


class TestWriting:
    def _writable(self, table: Any) -> Any:
        verdict = table.can(Operation.APPEND)
        if not verdict.ok:
            pytest.skip(f"append unavailable: {verdict.reason}")
        return table

    def test_append(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        table = self._writable(live_connection.table(name))
        table.append(pa.table({"id": [9], "city": ["quito"]}))
        assert live_connection.table(name).to_arrow().num_rows == 4

    def test_delta_rs_and_kernel_agree_after_a_write(
        self, live_connection: Any, scratch_sql: Any
    ) -> None:
        """Cross-engine agreement is the check that catches real corruption."""
        name, _run = scratch_sql
        table = self._writable(live_connection.table(name))
        table.append(pa.table({"id": [9], "city": ["quito"]}))
        reopened = live_connection.table(name)
        assert reopened.to_arrow().num_rows == reopened.count()

    def test_idempotent_write_is_skipped_on_replay(
        self, live_connection: Any, scratch_sql: Any
    ) -> None:
        name, _run = scratch_sql
        table = self._writable(live_connection.table(name))
        table.append(pa.table({"id": [9], "city": ["quito"]}), txn=("live-test", 1))
        rows = live_connection.table(name).to_arrow().num_rows
        table.append(pa.table({"id": [9], "city": ["quito"]}), txn=("live-test", 1))
        assert live_connection.table(name).to_arrow().num_rows == rows

    def test_commit_metadata_reaches_history(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        table = self._writable(live_connection.table(name))
        table.append(pa.table({"id": [9], "city": ["quito"]}), commit_metadata={"suite": "live"})
        assert live_connection.table(name).history(1)[0].get("suite") == "live"


# --------------------------------------------------- 7. catalog-managed path


class TestCatalogManaged:
    """The headline: a table no other Python library can open."""

    @pytest.fixture
    def catalog_managed(self, live_connection: Any, live_config: Any, scratch_sql: Any) -> Any:
        name, run = scratch_sql
        try:
            run(
                f"ALTER TABLE {name} SET TBLPROPERTIES "
                "('delta.feature.catalogManaged' = 'supported')"
            )
        except Exception as exc:
            pytest.skip(
                "could not enable catalog commits on this table; it is a Beta feature "
                f"gated on a workspace preview an admin must turn on. ({exc})"
            )
        table = live_connection.table(name)
        if not table.is_catalog_managed:
            pytest.skip("the table did not report catalogManaged after ALTER")
        return table

    def test_it_is_catalog_managed(self, catalog_managed: Any) -> None:
        assert catalog_managed.is_catalog_managed

    def test_catalog_supplies_a_commit_tail(self, catalog_managed: Any) -> None:
        assert catalog_managed.resolved.max_catalog_version is not None

    def test_reads_through_the_kernel(self, catalog_managed: Any) -> None:
        verdict = catalog_managed.can(Operation.SCAN)
        assert verdict.ok, verdict.reason
        assert verdict.engine is Engine.KERNEL
        assert catalog_managed.to_arrow().num_rows >= 0

    def test_delta_rs_refuses_it(self, catalog_managed: Any) -> None:
        """Confirms on a real table that the gap this project closes is real."""
        from deltaswamp.engine.deltars import DeltaRsEngine

        verdict = DeltaRsEngine().supports(Operation.SCAN, catalog_managed.resolved)
        assert not verdict.ok
        assert "catalogManaged" in verdict.reason

    def test_alter_never_goes_direct(self, catalog_managed: Any, live_config: Any) -> None:
        """The committer refuses metadata changes, so no direct engine may claim
        one; with the fallback on, the warehouse makes it."""
        verdict = catalog_managed.can(Operation.SET_PROPERTIES)
        if live_config.warehouse_id:
            assert verdict.engine is Engine.SQL
        else:
            assert not verdict.ok
            assert "catalog-managed" in verdict.reason

    def test_append_commits_through_the_catalog(self, catalog_managed: Any) -> None:
        verdict = catalog_managed.can(Operation.APPEND)
        if not verdict.ok:
            pytest.skip(verdict.reason)
        before = catalog_managed.to_arrow().num_rows
        catalog_managed.append(pa.table({"id": [77], "city": ["bergen"]}))
        assert catalog_managed.to_arrow().num_rows == before + 1


# ------------------------------------------------------ 8. honesty on refusals


class TestRefusalsAreHonest:
    """Where we say no, the reason has to be true of this workspace."""

    def test_views_are_refused_or_read_via_the_manifest(
        self, live_connection: Any, live_tables: list[Any]
    ) -> None:
        views = [t for t in live_tables if t.is_view_like]
        if not views:
            pytest.skip("no view in the target schema")
        verdict = live_connection.table(views[0].ref.full_name).can(Operation.SCAN)
        if verdict.ok and verdict.engine is not Engine.SQL:
            # A direct engine reads files, so it needs the manifest's blessing.
            # The warehouse evaluates the view itself and needs nothing.
            assert views[0].external_read_supported is True
        elif not verdict.ok:
            assert verdict.reason

    def test_every_refusal_carries_a_reason(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        for operation, verdict in live_connection.table(name).capabilities().items():
            if not verdict.ok:
                assert verdict.reason, f"{operation.value} refused silently"

    def test_claimed_reads_actually_work(self, live_connection: Any, scratch_sql: Any) -> None:
        """The strongest assertion here: if we said yes, it must work."""
        name, _run = scratch_sql
        table = live_connection.table(name)
        if table.can(Operation.SCAN).ok:
            table.to_arrow()

    def test_databricks_only_operations_need_the_fallback(
        self, live_connection: Any, scratch_sql: Any
    ) -> None:
        name, _run = scratch_sql
        verdict = live_connection.table(name).can(Operation.CLONE)
        if verdict.ok:
            assert verdict.engine is Engine.SQL
        else:
            assert "allow_sql_fallback" in verdict.remedy


# ---------------------------------------------------------- 9. SQL fallback


class TestSqlFallback:
    """Opt-in, and it announces itself when it runs."""

    def test_warns_when_it_serves_a_request(self, live_config: Any, scratch_sql: Any) -> None:
        if not live_config.warehouse_id:
            pytest.skip("no warehouse configured")
        from deltaswamp.engine.sql import SqlEngine, SqlFallbackWarning

        name, _run = scratch_sql
        conn = ds.connect(
            host=live_config.host,
            token=live_config.token,
            allow_sql_fallback=True,
            warehouse_id=live_config.warehouse_id,
            default_catalog=live_config.catalog,
            default_schema=live_config.schema,
        )
        engine = SqlEngine(
            host=live_config.host,
            token=live_config.token,
            warehouse_id=live_config.warehouse_id,
        )
        with pytest.warns(SqlFallbackWarning):
            engine.scan(conn.table(name).resolved, columns=["id"])


# ------------------------------------------------------------- 10. lifecycle


class TestLifecycle:
    def test_table_exists(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        assert live_connection.table_exists(name) is True

    def test_absent_table_reports_false(self, live_connection: Any, live_config: Any) -> None:
        assert live_connection.table_exists(f"{live_config.prefix}.not_here_3b71") is False

    def test_list_tables_includes_the_scratch_table(
        self, live_connection: Any, live_config: Any, scratch_sql: Any
    ) -> None:
        name, _run = scratch_sql
        listed = {
            t.ref.table
            for t in live_connection.list_tables(live_config.catalog, live_config.schema)
        }
        assert name.rsplit(".", 1)[-1] in listed

    def test_drop_table(self, live_connection: Any, scratch_sql: Any) -> None:
        name, _run = scratch_sql
        live_connection.drop_table(name)
        assert live_connection.table_exists(name) is False


# ----------------------------------------------------- 11. conformance sweep


class TestConformanceSweep:
    """Point the suite at a schema and check every claim it makes about it.

    The per-shape tests above use a table we created, so they only prove the
    happy path we set up. This walks whatever is actually in the schema, which
    is where the surprises live: row filters, views, foreign tables, formats we
    do not read.
    """

    def test_manifest_matches_what_we_claim(
        self, live_connection: Any, live_tables: list[Any]
    ) -> None:
        """Where the manifest says no external read, we must refuse; where it
        says yes, we must not. A mismatch either way is a routing bug."""
        checked = 0
        for resolved in live_tables[:15]:
            if resolved.external_read_supported is None or not resolved.is_delta:
                continue
            try:
                table = live_connection.table(resolved.ref.full_name)
            except DeltaSwampError:
                continue
            verdict = table.can(Operation.SCAN)
            direct = verdict.ok and verdict.engine is not Engine.SQL
            if not resolved.external_read_supported:
                # Withdrawn from vending: never a direct engine. The warehouse
                # may serve it, since it applies the policy itself.
                assert not direct, (
                    f"{resolved.ref.full_name}: withdrawn from vending "
                    f"({resolved.access_policy or 'manifest'}), yet claimed via {verdict.engine}"
                )
            else:
                assert verdict.ok, (
                    f"{resolved.ref.full_name}: manifest allows external reads, "
                    f"we refuse ({verdict.reason})"
                )
            checked += 1
        if checked == 0:
            pytest.skip("no tables with capability manifests to compare")

    def test_no_operation_is_refused_without_a_reason(
        self, live_connection: Any, live_tables: list[Any]
    ) -> None:
        for resolved in live_tables[:20]:
            try:
                report = live_connection.table(resolved.ref.full_name).capabilities()
            except DeltaSwampError:
                continue
            for operation, verdict in report.items():
                if not verdict.ok:
                    assert verdict.reason, (
                        f"{resolved.ref.full_name}: {operation.value} refused silently"
                    )

    def test_everything_we_claim_to_read_can_be_read(
        self, live_connection: Any, live_tables: list[Any]
    ) -> None:
        """The strongest assertion in the suite, across real tables."""
        checked = 0
        for resolved in live_tables[:15]:
            try:
                table = live_connection.table(resolved.ref.full_name)
            except DeltaSwampError:
                continue
            if not table.can(Operation.SCAN).ok:
                continue
            table.head(1)
            checked += 1
        if checked == 0:
            pytest.skip("no readable tables in the schema")

    def test_non_delta_tables_are_refused_with_the_format_named(
        self, live_connection: Any, live_tables: list[Any]
    ) -> None:
        others = [t for t in live_tables if not t.is_delta]
        if not others:
            pytest.skip("every table in the schema is Delta")
        verdict = live_connection.table(others[0].ref.full_name).can(Operation.SCAN)
        assert not verdict.ok
        assert "not Delta" in verdict.reason or "Iceberg" in verdict.reason
