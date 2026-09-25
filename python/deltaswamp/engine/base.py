"""The engine abstraction.

Planning and execution are separate calls. `plan_scan` returns serializable
`ScanSplit`s and `execute_scan` consumes them, possibly in another process, so
every engine can be distributed. Writes mirror the kernel's `WriteState`:
workers produce add-file records and the driver commits them in one
transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..capability import ENGINE_METHODS, Capability, Operation
from ..capability import Engine as EngineKind
from ..catalog import ResolvedTable

__all__ = [
    "AddFile",
    "DeletionVectorDescriptor",
    "Engine",
    "ScanSplit",
    "missing_method",
]


def missing_method(engine: object, operation: Operation) -> Capability | None:
    """A refusal if `engine` has no method for `operation`, else None.

    Engines call this first in `supports()`, so an engine can never claim an
    operation and then fail with `AttributeError` when it is invoked.
    """
    name = ENGINE_METHODS.get(operation)
    if name is not None and callable(getattr(engine, name, None)):
        return None
    kind = getattr(engine, "kind", None)
    label = kind.value if kind is not None else type(engine).__name__
    return Capability(operation, ok=False, reason=f"{label} does not implement {operation.value}")


@dataclass(frozen=True, slots=True)
class DeletionVectorDescriptor:
    """A DV reference as it appears in an `add` action.

    `storage_type` is one of ``'u'`` (UUID, path relative to the table),
    ``'i'`` (inline) or ``'p'`` (absolute path). Inline DVs carry no `offset`,
    which is why it is optional rather than defaulted to zero.

    `cardinality` must be validated against the decoded bitmap on read. Nothing
    else cross-checks it, and a stale value yields wrong row counts silently
    rather than raising.
    """

    storage_type: str
    path_or_inline: str
    size_in_bytes: int
    cardinality: int
    offset: int | None = None

    @property
    def unique_id(self) -> str:
        """Distinguishes multiple DVs for one data file. Required for correct
        snapshot reconstruction."""
        if self.offset is None:
            return f"{self.storage_type}{self.path_or_inline}"
        return f"{self.storage_type}{self.path_or_inline}@{self.offset}"


@dataclass(frozen=True, slots=True)
class ScanSplit:
    """One unit of scan work. Must be serializable to a worker.

    Carries no credentials -- the worker gets a `CredentialProvider` and vends
    its own, so a long job cannot die on a token frozen at submission time.
    """

    path: str
    size: int
    partition_values: dict[str, str] = field(default_factory=dict)
    deletion_vector: DeletionVectorDescriptor | None = None
    # Opaque per-file physical->logical transform from kernel's scan metadata.
    transform: bytes | None = None
    # The commit version this file was observed at. Pass-through fields must be
    # decoded against *their own* commit's protocol, not the target snapshot's.
    commit_version: int | None = None


@dataclass(frozen=True, slots=True)
class AddFile:
    """A data file to add in a commit, produced by a (possibly remote) writer."""

    path: str
    size: int
    modification_time: int
    partition_values: dict[str, str] = field(default_factory=dict)
    stats: str | None = None
    # Row tracking: assigned by the kernel on append. On a *rewrite* the
    # connector must preserve these, and a commit retry must re-derive every
    # base_row_id against the new high-water mark -- re-sending the same values
    # hands out duplicate row IDs.
    base_row_id: int | None = None
    default_row_commit_version: int | None = None
    deletion_vector: DeletionVectorDescriptor | None = None


@runtime_checkable
class Engine(Protocol):
    """One way of actually talking to a table."""

    kind: EngineKind

    def supports(self, operation: Operation, table: ResolvedTable, **shape: Any) -> Capability:
        """Whether this engine can serve `operation` on this table.

        `shape` carries the arguments of the specific call -- `properties`,
        `schema_mode`, and so on -- so an engine can refuse a request it cannot
        honor rather than accepting it and failing partway through.

        Must never raise for an unsupported combination -- return a `Capability`
        with `ok=False` and a reason naming the blocker. The router aggregates
        these into `Table.capabilities()`.
        """
        ...

    def scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
        timestamp: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """Read the table, returning a PyCapsule-exporting Arrow stream.

        `limit` is a hint: an engine that streams may ignore it, because the
        caller stops consuming once it has enough. An engine that computes the
        whole result somewhere else first -- the SQL warehouse -- must push it
        down, or `head(3)` costs a full table scan on the warehouse.

        The in-process path. Implementations must apply deletion vectors without
        reordering rows within a file -- a DV is a positional keep-mask, so any
        repartitioning or pre-filtering ahead of it silently returns the right
        number of rows made of the wrong records.
        """
        ...

    # --- the distributed path, optional ---

    #: True if `plan_scan`/`execute_scan` are implemented.
    supports_distributed_scan: bool

    def plan_scan(
        self,
        table: ResolvedTable,
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
        version: int | None = None,
    ) -> list[ScanSplit]:
        """Resolve a snapshot and enumerate the files to read.

        Raises `NotImplementedError` when `supports_distributed_scan` is False.
        """
        ...

    def execute_scan(
        self,
        table: ResolvedTable,
        splits: list[ScanSplit],
        *,
        columns: list[str] | None = None,
        predicate: str | None = None,
    ) -> Any:
        """Read `splits` and return an Arrow stream (PyCapsule-exporting).

        Implementations MUST apply deletion vectors without reordering rows. A DV
        is a positional keep-mask in physical file order, so any repartitioning,
        limit pushdown or pre-filtering ahead of DV application silently returns
        the right number of rows made of the wrong records.
        """
        ...
