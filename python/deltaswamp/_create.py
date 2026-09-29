"""A distributed write that creates its table: `Connection.plan_write` of a new name.

Creating the table first and then planning the write left an empty table
behind whenever the job failed -- visible, registered in the catalog, and
refusing the retry as "already exists". Here nothing is visible until the
job's data can land:

* The driver composes the table's version 0 (protocol and metaData) and
  writes it as a *template* under ``<root>/_deltaswamp_pending/<plan>/``,
  where no reader looks. Workers resolve the table from it (the kernel takes
  it as the whole log) and write their files under the real root, in the
  real layout.
* The commit publishes version 0 and commits the files as version 1, then
  registers an external table in the catalog. A managed table's version 0 is
  written to its staging location at planning -- the catalog has not
  registered it, so nothing can see it -- and the commit finalizes the
  registration before committing the files through the catalog.
* A commit that certainly fails deletes the files and undoes the create:
  version 0 is removed while it is still the only version, and a managed
  table this plan registered is dropped while it is still empty.

Another writer may create the table while the job runs. Then ``error``
refuses, ``ignore`` keeps theirs, and ``append`` appends the files to it when
they fit its layout; each of the three deletes the files it does not commit.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .catalog.base import ResolvedTable
from .credentials.base import StaticCredentialProvider
from .errors import DeltaSwampError, InvalidArgumentError, UnreachableTableError
from .identity import RefKind, TableRef, parse_ref

if TYPE_CHECKING:
    from .connection import Connection
    from .distributed import WritePlan

log = logging.getLogger("deltaswamp")

#: The save modes a creating write takes (Spark's).
MODES = ("append", "overwrite", "error", "ignore")

_CATALOG_MANAGED = ("catalogManaged", "catalogOwned-preview")


class _PathCredentialProvider:
    """Path credentials for a table the catalog does not know yet, re-vended near expiry.

    An external table is written before it is registered, so there is no
    table to vend for. The driver keeps this provider for the commit, which
    may come an hour after planning; a worker receives the credential it
    vends when the plan is pickled (`distributed._for_workers`), never this.
    """

    def __init__(self, catalog: Any, location: str, portable: bool = False) -> None:
        self._catalog = catalog
        self._location = location
        self._credentials: Any = None
        self._portable = portable

    @property
    def table_id(self) -> str | None:
        return None

    @property
    def credential_key(self) -> str:
        """What a plan's `credential_source` is asked by, as a table has no id yet."""
        return f"path:{self._location}"

    def portable(self) -> _PathCredentialProvider:
        """A copy that pickles, catalog secrets included (`CredentialBroker.portable`)."""
        from .credentials.databricks import shipping

        return _PathCredentialProvider(shipping(self._catalog), self._location, portable=True)

    def credentials(self, operation: Any = None) -> Any:
        from .credentials.base import DEFAULT_REFRESH_MARGIN_SECONDS

        cached = self._credentials
        if cached is None or cached.expires_within(DEFAULT_REFRESH_MARGIN_SECONDS):
            self._credentials = self._catalog.path_credentials(self._location, "PATH_CREATE_TABLE")
        return self._credentials

    def invalidate(self) -> None:
        self._credentials = None

    def __reduce__(self) -> Any:
        if self._portable:
            return (_PathCredentialProvider, (self._catalog, self._location, True))
        # It holds the catalog, and with it the catalog's token.
        raise TypeError(
            "a creating write's path credential provider stays on the driver; "
            "workers get the storage credential it vends"
        )


def plan_create_write(
    conn: Connection,
    name: str,
    *,
    schema: Any,
    mode: str,
    location: str | None,
    partition_by: list[str] | None,
    cluster_by: list[str] | None,
    properties: dict[str, str] | None,
    comment: str | None,
    txn: tuple[str, int] | None,
    commit_metadata: dict[str, Any] | None,
    ship_catalog_auth: bool,
    supplies_defaults: bool,
    credential_source: Any = None,
    identity_tasks: int | None = None,
    identity_rows_per_task: int = 1 << 31,
    domain_metadata: dict[str, str] | None = None,
) -> WritePlan | None:
    """`Connection.plan_write` (see there)."""
    from .connection import _check_existing_layout

    if mode not in MODES:
        raise InvalidArgumentError(
            f"plan_write mode={mode!r}: the modes are 'append', 'overwrite', 'error' and 'ignore'"
        )
    planned: dict[str, Any] = {
        "txn": txn,
        "commit_metadata": commit_metadata,
        "ship_catalog_auth": ship_catalog_auth,
        "supplies_defaults": supplies_defaults,
        "credential_source": credential_source,
        "domain_metadata": domain_metadata,
    }
    identity: dict[str, Any] = {
        "identity_tasks": identity_tasks,
        "identity_rows_per_task": identity_rows_per_task,
    }
    if conn.table_exists(name):
        if mode == "error":
            raise UnreachableTableError(
                f"plan a write that creates {name}",
                "the table already exists",
                "plan with mode='append', 'overwrite' or 'ignore'",
            )
        if mode == "ignore":
            return None
        table = conn.table(name)
        _check_existing_layout(table, name, partition_by, properties)
        existing: WritePlan = table.plan_write(mode=mode, **planned, **identity)
        return existing
    if schema is None:
        raise InvalidArgumentError(
            f"{name} does not exist yet, so plan_write needs schema= to create it"
        )
    ref, schema, partition_by, cluster_by, properties = conn._create_args(
        name, schema, partition_by, cluster_by, properties
    )
    if ship_catalog_auth and ref.kind is RefKind.CATALOG:
        raise InvalidArgumentError(
            f"ship_catalog_auth=True cannot plan a write that creates {name}: until the "
            "commit registers it there is no table to vend credentials for. Workers get a "
            "storage credential vended on the driver; plan again with the default"
        )
    layout = {
        "partition_by": partition_by,
        "cluster_by": cluster_by,
        "properties": properties,
        "identity": (identity_tasks, identity_rows_per_task),
    }
    if ref.kind is RefKind.PATH:
        creator = _plan_path(conn, name, ref, schema, location, layout, comment)
    elif location is not None:
        creator = _plan_external(conn, name, ref, schema, location, layout, comment)
    else:
        creator = _plan_managed(conn, name, ref, schema, layout, comment)
    creator.mode = mode
    try:
        return creator.plan(conn, planned)
    except BaseException:
        creator.discard()
        raise


def _plan_path(
    conn: Connection,
    name: str,
    ref: TableRef,
    schema: Any,
    location: str | None,
    layout: dict[str, Any],
    comment: str | None,
) -> _Creator:
    from .catalog.filesystem import FilesystemCatalog
    from .connection import _check_local_create_path, _refuse_foreign_files

    if location is not None:
        raise InvalidArgumentError(
            f"{name!r} is a path, so the table is created there; location= applies only "
            "to a catalog name (an external table)"
        )
    _check_local_create_path(name)
    _refuse_foreign_files(name, conn.storage_options)
    resolved = FilesystemCatalog().resolve(ref)
    assert resolved.location is not None
    v0, reserved = _reserved_in_v0(_composed_v0(conn, schema, layout, comment), *layout["identity"])
    return _Creator(
        kind="path",
        name=name,
        ref=ref,
        location=resolved.location,
        provider=None,
        v0=v0,
        identity_reserved=reserved,
    )


def _plan_external(
    conn: Connection,
    name: str,
    ref: TableRef,
    schema: Any,
    location: str,
    layout: dict[str, Any],
    comment: str | None,
) -> _Creator:
    from ._storage import engine_options, store_options
    from .connection import _refuse_foreign_files

    catalog = conn._lifecycle_catalog(f"create the catalog table {ref}", ref)
    provider = _PathCredentialProvider(catalog, location)
    secrets = getattr(provider.credentials(), "secrets", None)
    _refuse_foreign_files(
        location, store_options(engine_options(conn.storage_options, secrets, location))
    )
    v0, reserved = _reserved_in_v0(_composed_v0(conn, schema, layout, comment), *layout["identity"])
    return _Creator(
        kind="external",
        name=name,
        ref=ref,
        location=location,
        provider=provider,
        v0=v0,
        catalog=catalog,
        comment=comment,
        identity_reserved=reserved,
    )


def _plan_managed(
    conn: Connection,
    name: str,
    ref: TableRef,
    schema: Any,
    layout: dict[str, Any],
    comment: str | None,
) -> _Creator:
    from . import _native
    from ._storage import engine_options, store_options
    from .catalog.ossuc import _cloud_for
    from .credentials.base import Credentials, Operation

    catalog = conn._lifecycle_catalog(f"create the catalog table {ref}", ref)
    try:
        staging = catalog.create_staging_table(ref)
    except DeltaSwampError as exc:
        raise UnreachableTableError(
            f"plan a write that creates the managed table {ref}",
            f"the catalog would not allocate it ({exc})",
            "create it first with conn.create_table(), which can fall back to a warehouse "
            "(allow_sql_fallback=True), then plan the write into it",
        ) from exc
    actions = conn._managed_v0(
        catalog,
        ref,
        staging,
        schema,
        layout["partition_by"],
        layout["cluster_by"],
        layout["properties"],
        comment,
    )
    actions, reserved = _reserved_in_v0(actions, *layout["identity"])
    options = store_options(
        engine_options(conn.storage_options, staging.storage_options, staging.location)
    )
    # Version 0 goes to the staging location now, with the staging credential
    # the catalog just vended: an hour from now, at commit, it may be dead.
    # The catalog has not registered the table, so nothing can see it yet.
    _native.commit_raw(staging.location, 0, actions, options=options or None)
    request = json.loads(
        _native.uc_create_table_request(staging.location, ref.table or "", options=options)
    )
    if comment:
        request["comment"] = comment
    credentials = Credentials(
        cloud=_cloud_for(staging.location),
        url=staging.location,
        expires_at=staging.expires_at,
        secrets=dict(staging.storage_options),
        table_id=staging.table_id,
        operation=Operation.READ_WRITE,
    )
    return _Creator(
        kind="managed",
        name=name,
        ref=ref,
        location=staging.location,
        provider=StaticCredentialProvider(credentials),
        v0=actions,
        catalog=catalog,
        comment=comment,
        finalize_request=request,
        table_id=staging.table_id,
        identity_reserved=reserved,
    )


def _composed_v0(
    conn: Connection, schema: Any, layout: dict[str, Any], comment: str | None
) -> list[str]:
    """Version 0 as `create_table` would write it, composed in a scratch directory.

    The engine `create_table` routes to writes it there -- with every
    property, feature and column-mapping id it would give the real table --
    and its log is read back. A comment the kernel sets in a second commit is
    folded into version 0's metaData.
    """
    import os
    import tempfile

    from .catalog.filesystem import FilesystemCatalog

    with tempfile.TemporaryDirectory(prefix="deltaswamp-create-") as scratch:
        root = os.path.join(scratch, "table")
        conn._create_at(
            FilesystemCatalog().resolve(parse_ref(root)),
            schema,
            partition_by=layout["partition_by"],
            cluster_by=layout["cluster_by"],
            mode="error",
            properties=layout["properties"],
            comment=comment,
        )
        log_dir = os.path.join(root, "_delta_log")
        commits = sorted(n for n in os.listdir(log_dir) if n.endswith(".json"))
        with open(os.path.join(log_dir, commits[0]), encoding="utf-8") as f:
            actions = [line for line in f.read().splitlines() if line.strip()]
        for later in commits[1:]:
            with open(os.path.join(log_dir, later), encoding="utf-8") as f:
                for line in f.read().splitlines():
                    if not line.strip():
                        continue
                    action = json.loads(line)
                    if "metaData" in action:
                        actions = [a for a in actions if "metaData" not in json.loads(a)]
                        actions.append(line)
                    elif "commitInfo" not in action:
                        raise UnreachableTableError(
                            "compose the new table's version 0",
                            f"creating it wrote {sorted(action)} after version 0",
                        )
    return actions


def _reserved_in_v0(v0: list[str], tasks: int | None, rows_per_task: int) -> tuple[list[str], Any]:
    """Version 0 with the job's identity values reserved in it, and the reservation.

    A table the write creates has generated nothing, so the job's block of
    each identity column starts at the column's start, and version 0 itself
    records the high-water mark past it: no commit of its own, as
    `Table.plan_write` needs on an existing table. The reservation is
    ``((0, blocks), slots)``, or ``((), 0)`` when the write reserves none.
    """
    from .engine import metadata as meta
    from .engine.values import _identity_spec, reserve_identity
    from .table import identity_reservation

    actions = [json.loads(line) for line in v0]
    metadata = next(a["metaData"] for a in actions if "metaData" in a)
    protocol = next(a["protocol"] for a in actions if "protocol" in a)
    fields = json.loads(metadata["schemaString"]).get("fields") or []
    identity = {
        f["name"]: _identity_spec(f["name"], f.get("metadata") or {})
        for f in fields
        if any(str(k).startswith("delta.identity.") for k in f.get("metadata") or {})
    }
    if not identity_reservation(identity, tasks, rows_per_task):
        return v0, ((), 0)
    assert tasks is not None
    state = meta.TableState(version=0, protocol=protocol, metadata=metadata)
    change, reserved = reserve_identity(state, dict.fromkeys(identity, tasks * rows_per_task))
    metadata["schemaString"] = change.metadata["schemaString"]
    blocks = tuple((n, b.first, b.step, b.count) for n, b in sorted(reserved.items()))
    lines = [json.dumps(a, separators=(",", ":")) for a in actions]
    return lines, ((0, blocks), tasks)


def _metadata_id(actions: list[str]) -> str:
    for line in actions:
        action = json.loads(line)
        if "metaData" in action:
            return str(action["metaData"]["id"])
    raise UnreachableTableError("compose the new table's version 0", "it has no metaData")


def _for_workers(actions: list[str]) -> list[str]:
    """Version 0 as workers resolve it: a catalog-managed table's without catalog management.

    Workers only write files, and a file depends on the schema, partitioning,
    column mapping and constraints (`kernel._write_layout`) -- none of which
    this changes. Resolving the table as catalog-managed would need the
    catalog's commit API on every worker for a table the catalog has not
    registered yet.
    """
    out = []
    for line in actions:
        action = json.loads(line)
        protocol = action.get("protocol")
        if protocol is not None:
            for key in ("readerFeatures", "writerFeatures"):
                if key in protocol:
                    protocol[key] = [f for f in protocol[key] if f not in _CATALOG_MANAGED]
            line = json.dumps(action, separators=(",", ":"))
        out.append(line)
    return out


@dataclass(eq=False)
class _Creator:
    """What creates a planned write's table at commit; lives on the driver's plan only."""

    kind: str  # "path", "external" or "managed"
    name: str
    ref: TableRef
    location: str
    provider: Any
    #: Version 0: published at commit (path, external), or already written to
    #: the staging location (managed).
    v0: list[str]
    catalog: Any = None
    comment: str | None = None
    #: A managed table's finalize request, built from version 0 at planning.
    finalize_request: dict[str, Any] | None = None
    #: A managed table's id, from staging.
    table_id: str | None = None
    mode: str = "append"
    plan_id: str = ""
    connection: Any = None
    #: The identity values version 0 reserves for the job (`_reserved_in_v0`).
    identity_reserved: Any = ((), 0)

    # ------------------------------------------------------------ planning

    def plan(self, conn: Connection, planned: dict[str, Any]) -> WritePlan:
        from . import _native

        self.connection = conn
        self.plan_id = uuid.uuid4().hex
        # Writing the template is also the check that this principal can write
        # there at all, before any worker runs.
        url, modified, size = _native.write_create_template(
            self.location, self.plan_id, _for_workers(self.v0), options=self._options()
        )
        pending = ResolvedTable(
            ref=parse_ref(self.location),
            location=self.location,
            credential_provider=self.provider,
            pending_commit=(url, int(modified), int(size)),
        )
        from .table import Table

        # Every check a write into the table would get -- features, defaults,
        # txn, commit_metadata -- on the table as it will be.
        plan: WritePlan = Table(conn, pending).plan_write(
            mode="append", _identity_reserved=self.identity_reserved, **planned
        )
        return replace(plan, version=0, table_identity=_metadata_id(self.v0), create=self)

    def discard(self) -> None:
        """Undo planning: the template goes (a managed staging location cannot be released)."""
        if not self.plan_id:
            return
        from . import _native

        try:
            _native.delete_create_template(self.location, self.plan_id, options=self._options())
        except Exception as exc:
            log.warning("could not delete the create template under %s: %s", self.location, exc)
            return
        _remove_empty_local_dirs(self.location, self.plan_id)

    def _options(self) -> dict[str, str] | None:
        from ._storage import engine_options, store_options

        secrets = None
        if self.provider is not None:
            from .credentials.base import Operation

            secrets = self.provider.credentials(Operation.READ_WRITE).as_storage_options()
        base = getattr(self.connection, "storage_options", None) if self.connection else None
        options = store_options(engine_options(base or {}, secrets, self.location))
        return options or None

    # -------------------------------------------------------------- commit

    def commit(
        self,
        plan: WritePlan,
        collected: list[bytes],
        *,
        operation: str,
        retries: int,
        abort_on_failure: bool,
    ) -> int:
        if self.kind != "managed":
            return self._created(
                plan,
                collected,
                operation=operation,
                retries=retries,
                abort_on_failure=abort_on_failure,
            )
        target = self._finalized(plan, collected, abort_on_failure)
        if isinstance(target, int):
            return target  # another writer's table, which this write did not join
        version = self._commit_data(
            target,
            collected,
            operation=operation,
            retries=retries,
            abort_on_failure=abort_on_failure,
        )
        self._forget_template(target)
        if self.kind == "external":
            self._register(target)
        return version

    def _created(
        self,
        plan: WritePlan,
        collected: list[bytes],
        *,
        operation: str,
        retries: int,
        abort_on_failure: bool,
    ) -> int:
        """Create a path or external table as one commit: version 0 with the job's files.

        No reader ever sees the table without its data. When version 0 is
        already there, its metaData id says whose: this plan's (an earlier
        attempt whose answer was lost, which committed the files with it) or
        another writer's, which the mode decides about (`_joined`).
        """
        from . import _native
        from .distributed import _unknown_outcome
        from .errors import DeltaSwampError, TransientCommitError

        def published() -> bool:
            return bool(_native.create_published(self.location, options=self._options()))

        # A table already there is settled first: the template's view of the
        # location would read that table's log as if it were its own.
        if not published():
            try:
                version: int = plan.engine.commit_files(
                    plan.table,
                    collected,
                    operation="CREATE TABLE AS SELECT",
                    txn=plan.txn,
                    commit_metadata=plan.commit_metadata,
                    table_identity=plan.table_identity,
                    domain_metadata=plan.domain_metadata,
                    create_template=self.v0,
                )
            except TransientCommitError as exc:
                # Unknown outcome: committing again finds version 0 and settles it.
                _unknown_outcome(exc)
                raise
            except DeltaSwampError:
                if not published():
                    # Nothing at version 0: a failure that committed nothing,
                    # and no table can reference the files.
                    if abort_on_failure:
                        self._abort_files(plan, collected)
                        self.discard()
                    raise
                # Version 0 appeared meanwhile: settled below, by whose it is.
            else:
                self._forget_template(plan)
                if self.kind == "external":
                    self._register(replace(plan, version=version))
                return version
        target = self._opened(plan)
        if target.table_identity == _metadata_id(self.v0):
            # This plan's own version 0, from an attempt whose answer was
            # lost: it was written with the files.
            self._forget_template(target)
            if self.kind == "external":
                self._register(target)
            return 0
        joined = self._joined(plan, target, collected, abort_on_failure)
        if isinstance(joined, int):
            return joined  # another writer's table, which this write did not join
        version = self._commit_data(
            joined,
            collected,
            operation=operation,
            retries=retries,
            abort_on_failure=abort_on_failure,
        )
        self._forget_template(joined)
        if self.kind == "external":
            self._register(joined)
        return version

    def _finalized(
        self, plan: WritePlan, collected: list[bytes], abort_on_failure: bool
    ) -> WritePlan | int:
        """Register a managed table in the catalog; the plan to commit the files with."""
        from .connection import _registered_anyway

        assert self.finalize_request is not None
        try:
            self.catalog.finalize_managed_table(self.ref, self.finalize_request)
        except Exception as exc:
            if not _registered_anyway(self.catalog, self.ref, exc):
                # Only when the catalog says the name is free: then no commit
                # of this plan can reference the files.
                if abort_on_failure and _certainly_absent(self.catalog, self.ref):
                    self._abort_files(plan, collected)
                    self.discard()
                else:
                    abort_on_failure = False
                raise UnreachableTableError(
                    f"create the managed table {self.ref}",
                    f"the catalog did not accept the registration ({exc}); the job's files "
                    + ("were deleted" if abort_on_failure else "are left unreferenced"),
                    "the staging location is not reclaimed automatically; plan the write "
                    "again, which allocates a fresh one",
                ) from exc
        resolved = self.catalog.resolve(self.ref)
        if resolved.table_id != self.table_id:
            # Another writer registered the name first: this plan's files
            # are in its own staging location, which that table cannot use.
            return self._lost_managed(plan, collected, resolved, abort_on_failure)
        from .table import Table

        table = Table(self.connection, resolved)
        return replace(
            plan,
            table=table._enrich(),
            catalog=self.connection._catalog_for(self.ref),
            create=None,
            version=0,
            mode="append",
            table_identity=_metadata_id(self.v0),
        )

    def _opened(self, plan: WritePlan) -> WritePlan:
        """`plan` retargeted at the table now at this location."""
        from .table import Table

        resolved = ResolvedTable(
            ref=parse_ref(self.location),
            location=self.location,
            credential_provider=self.provider,
        )
        table = Table(self.connection, resolved)
        enriched = table._enrich()
        snapshot = plan.engine.snapshot(enriched, write=True)
        return replace(
            plan,
            table=enriched,
            create=None,
            version=int(snapshot.version),
            mode="append",
            table_identity=str(snapshot.metadata_id),
        )

    def _joined(
        self,
        plan: WritePlan,
        theirs: WritePlan,
        collected: list[bytes],
        abort_on_failure: bool,
    ) -> WritePlan | int:
        """Another writer created the table while the job ran: the mode decides."""
        from .engine.kernel import _layout_still_fits, _write_layout

        current = _write_layout(plan.engine.snapshot(theirs.table, write=True))
        fits = current is not None and all(
            _layout_still_fits(written, current) for written in _fragment_layouts(collected)
        )
        if self.mode != "append" or not fits:
            if abort_on_failure or self.mode == "ignore":
                self._abort_files(theirs, collected)
            self._forget_template(theirs)
            if self.mode == "ignore":
                assert theirs.version is not None
                return theirs.version
            why = (
                "its layout differs from the one these files were written in"
                if self.mode == "append"
                else f"mode={self.mode!r} does not write into a table that exists"
            )
            raise UnreachableTableError(
                f"create {self.name}",
                f"another writer created the table while this job ran, and {why}; the job's "
                + ("files were deleted" if abort_on_failure else "files are left unreferenced"),
                "plan the write again against the table that exists now",
            )
        # The files fit their table: append them to it. They were stamped
        # with this plan's table id, which is not theirs.
        assert theirs.table_identity is not None
        collected[:] = [_restamped(f, theirs.table_identity) for f in collected]
        return theirs

    def _lost_managed(
        self, plan: WritePlan, collected: list[bytes], resolved: Any, abort_on_failure: bool
    ) -> int:
        if abort_on_failure or self.mode == "ignore":
            self._abort_files(plan, collected)
        if self.mode == "ignore":
            from .table import Table

            detail = plan.engine.detail(Table(self.connection, resolved)._enrich())
            return int(detail.get("version") or 0)
        raise UnreachableTableError(
            f"create the managed table {self.ref}",
            "another writer created a table of that name while this job ran; the job's files "
            "were written to this plan's own staging location, which that table cannot "
            "reference, so they "
            + ("were deleted" if abort_on_failure else "are left unreferenced"),
            "plan the write again, as an append into the table that exists now",
        )

    def _commit_data(
        self,
        target: WritePlan,
        collected: list[bytes],
        *,
        operation: str,
        retries: int,
        abort_on_failure: bool,
    ) -> int:
        """Commit the files into `target`; on a certain failure, delete them and undo the create."""
        from .distributed import _CertainFailure, _unknown_outcome
        from .errors import TransientCommitError

        try:
            return target._commit(
                collected,
                operation=operation,
                retries=retries,
                allow_concurrent_overwrite=False,
                allow_empty_overwrite=False,
            )
        except _CertainFailure as failure:
            if abort_on_failure:
                target._abort_after(collected, failure.error)
                if target.table_identity == _metadata_id(self.v0):
                    self._undo(target)
                self.discard()
            raise failure.error from failure.error.__cause__
        except TransientCommitError as exc:
            # Unknown outcome: commit() on the same plan again settles it --
            # version 0 is found to be this plan's and the files' landing
            # is checked before anything is committed.
            _unknown_outcome(exc)
            raise

    def _undo(self, target: WritePlan) -> None:
        """Remove the table this plan created, while it is still empty; best effort."""
        from . import _native

        try:
            if self.kind == "managed":
                detail = target.engine.detail(target.table)
                if int(detail.get("version") or 0) == 0:
                    self.connection.drop_table(self.name, if_exists=True)
            else:
                _native.rollback_create_table(
                    self.location, _metadata_id(self.v0), options=self._options()
                )
            self._forget_template(target)
        except Exception as exc:
            log.warning("could not undo the create of %s: %s", self.name, exc)

    def _forget_template(self, target: WritePlan) -> None:
        del target
        self.discard()

    def _register(self, target: WritePlan) -> None:
        """Register an external table once its data is in; idempotent."""
        try:
            self.connection.register_table(self.name, self.location, comment=self.comment)
        except Exception as exc:
            raise UnreachableTableError(
                f"register {self.ref}",
                f"the table was written at {self.location} (version {target.version} and "
                f"on), but registering it in the catalog failed: {exc}",
                f"conn.register_table({str(self.ref)!r}, {self.location!r}) once the cause "
                "is fixed; the data is committed",
            ) from exc

    # --------------------------------------------------------------- abort

    def abort(self, plan: WritePlan, collected: list[bytes]) -> int:
        """Delete the job's files and undo the create where nothing else landed."""
        target = self._current(plan)
        if (
            self.kind != "managed"
            and getattr(target.table, "pending_commit", None) is None
            and target.table_identity == _metadata_id(self.v0)
        ):
            # This plan's version 0 is there, and it was written with the
            # files: the write succeeded.
            raise UnreachableTableError(
                "abort these fragments",
                "their files were committed: they are in version 0 of the table this write created",
                "keep them; the write succeeded",
            )
        deleted = self._abort_files(target, collected, raising=True)
        if target is not plan and target.table_identity == _metadata_id(self.v0):
            self._undo(target)
        self.discard()
        return deleted

    def _current(self, plan: WritePlan) -> WritePlan:
        """The plan as it applies now: the created table's, or the template's before one."""
        from .errors import TableNotFoundError

        if self.kind == "managed":
            try:
                resolved = self.catalog.resolve(self.ref)
            except TableNotFoundError:
                return _uncreated(plan)
            if resolved.table_id != self.table_id:
                return _uncreated(plan)
            from .table import Table

            return replace(
                plan,
                table=Table(self.connection, resolved)._enrich(),
                catalog=self.connection._catalog_for(self.ref),
                create=None,
                version=0,
                mode="append",
                table_identity=_metadata_id(self.v0),
            )
        from . import _native

        # Whether version 0 is there decides which view is safe: the
        # template's deletes files unchecked, which only an absent table allows.
        if not _native.create_published(self.location, options=self._options()):
            return _uncreated(plan)
        return self._opened(plan)

    def _abort_files(
        self, target: WritePlan, collected: list[bytes], *, raising: bool = False
    ) -> int:
        """Delete the job's files, through `target` (the table they are checked against)."""
        try:
            if getattr(target.table, "pending_commit", None) is None:
                # A table exists: abort() refuses files any commit took.
                return int(replace(target, create=None).abort(collected))
            # No table yet, so no commit can reference them.
            from .distributed import _fragment_files

            paths, _ = _fragment_files(collected)
            if not paths:
                return 0
            failed = target.engine.delete_uncommitted(target.table, sorted(paths))
            if failed:
                log.warning(
                    "could not delete %d of %d uncommitted data file(s): %s",
                    len(failed),
                    len(paths),
                    "; ".join(failed[:10]),
                )
            return len(paths) - len(failed)
        except Exception as exc:
            if raising:
                raise
            log.warning("could not delete the job's files under %s: %s", self.location, exc)
            return 0


def _certainly_absent(catalog: Any, ref: TableRef) -> bool:
    """Whether the catalog says `ref` does not exist (False when it cannot say)."""
    try:
        return not bool(catalog.table_exists(ref))
    except Exception:
        return False


def _remove_empty_local_dirs(location: str, plan_id: str) -> None:
    """Remove the template's directories on a local filesystem, where they outlive their file."""
    import os
    from urllib.parse import unquote, urlparse

    parsed = urlparse(location)
    if parsed.scheme not in ("", "file"):
        return  # object stores have no directories
    root = unquote(parsed.path) if parsed.scheme else location
    pending = os.path.join(root, "_deltaswamp_pending")
    for directory in (
        os.path.join(pending, plan_id, "_delta_log"),
        os.path.join(pending, plan_id),
        pending,
    ):
        try:
            os.rmdir(directory)
        except OSError:
            return  # not empty (another plan's template), or gone


def _uncreated(plan: WritePlan) -> WritePlan:
    """`plan` as it stands before its table exists: resolved from the template."""
    return replace(plan, create=None)


def _fragment_layouts(collected: list[bytes]) -> list[str]:
    import pyarrow as pa

    from .engine.kernel import _FRAGMENT_LAYOUT

    out = []
    for fragment in collected:
        if not fragment:
            continue
        stamp = (pa.ipc.open_stream(fragment).schema.metadata or {}).get(_FRAGMENT_LAYOUT.encode())
        if stamp is not None:
            out.append(stamp.decode())
    return out


def _restamped(fragment: bytes, metadata_id: str) -> bytes:
    """`fragment` claimed for the table `metadata_id` names."""
    if not fragment:
        return fragment
    import pyarrow as pa

    reader = pa.ipc.open_stream(fragment)
    schema = reader.schema.with_metadata(
        {**(reader.schema.metadata or {}), b"deltaswamp.metadata_id": metadata_id.encode()}
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        for batch in reader:
            writer.write_batch(pa.RecordBatch.from_arrays(batch.columns, schema=schema))
    return bytes(sink.getvalue().to_pybytes())
