"""Distributed reads of tables only a SQL warehouse can read.

A table with a row filter or column mask, a view or a materialized view has
no files a worker may read: Unity Catalog vends no storage credential for it,
and only the warehouse can evaluate the policy. With the SQL fallback on,
`Table.plan_scan` runs the query once on the warehouse, with the projection,
predicate and time travel in the SQL, and plans the result instead: one split
per result chunk, which the warehouse serves as Arrow IPC behind a presigned
link (``EXTERNAL_LINKS`` disposition).

A worker fetches its chunks' links with nothing but the headers each link
carries -- never workspace credentials -- over https, to a public host (the
checks Delta Sharing's presigned URLs get). Links live about 15 minutes. By
default the plan carries every chunk's link, fetched on the driver when it
plans, so a job must start reading within that window. For longer jobs the
links are fetched fresh when a worker needs them, from:

- ``credential_source=``: the same picklable ``source(key, operation)`` a
  plan's storage credentials refresh through, reaching a driver-side
  `CredentialBroker` the plan was added to (``broker.add(plan)``; for a broker
  in another process, ``broker.add(CredentialBroker.portable(plan))``);
- ``ship_catalog_auth=True``: the plan carries the catalog's auth, and each
  worker asks the warehouse itself.
"""

from __future__ import annotations

import datetime as _dt
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse

from .errors import UnreachableTableError

__all__ = ["ChunkLink", "WarehouseDatasource", "WarehouseScanPlan", "WarehouseSplit"]

#: A link this close to its expiry is fetched afresh rather than used.
LINK_MARGIN_SECONDS = 30.0

#: How long one chunk download may take.
DOWNLOAD_TIMEOUT_SECONDS = 300.0

#: Statuses a store answers an expired presigned link with: S3 and Azure say
#: 403, GCS 400, some gateways 401.
_EXPIRED_STATUSES = frozenset({400, 401, 403})

#: The key prefix a broker serves a statement's result links under.
STATEMENT_KEY_PREFIX = "sql-statement:"


def _expiry(value: Any) -> float | None:
    """A link's ``expiration`` (ISO-8601 text) as epoch seconds, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        moment = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_dt.UTC)
    return moment.timestamp()


@dataclass(frozen=True)
class ChunkLink:
    """One presigned link to (part of) a result chunk."""

    url: str
    headers: tuple[tuple[str, str], ...] = ()
    expires_at: float | None = None

    @classmethod
    def of(cls, link: Any) -> ChunkLink:
        """From an SDK ``ExternalLink`` (or a dict a broker sent)."""
        if isinstance(link, dict):
            url = link.get("url") or link.get("external_link")
            headers = link.get("headers") or link.get("http_headers") or {}
            expires = link.get("expires_at", link.get("expiration"))
        else:
            url = getattr(link, "external_link", None)
            headers = getattr(link, "http_headers", None) or {}
            expires = getattr(link, "expiration", None)
        if not url:
            raise UnreachableTableError(
                "read a warehouse result chunk", "the warehouse returned a result link with no URL"
            )
        pairs = headers.items() if isinstance(headers, dict) else headers
        return cls(
            url=str(url),
            headers=tuple((str(k), str(v)) for k, v in pairs),
            expires_at=_expiry(expires),
        )

    def fresh(self, margin: float = LINK_MARGIN_SECONDS) -> bool:
        return self.expires_at is None or self.expires_at - time.time() > margin

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "headers": dict(self.headers), "expires_at": self.expires_at}

    def __repr__(self) -> str:
        # The query string is the signature: never print it.
        return f"ChunkLink(host={urlparse(self.url).hostname!r}, expires_at={self.expires_at!r})"


@dataclass(frozen=True)
class WarehouseSplit:
    """One result chunk: what a worker downloads."""

    statement_id: str
    chunk_index: int
    row_count: int
    #: The chunk's Arrow IPC bytes, as the manifest reports them.
    size: int
    links: tuple[ChunkLink, ...] = ()

    @property
    def path(self) -> str:
        # `distributed.balance` orders splits of equal size by path.
        return f"{self.statement_id}/{self.chunk_index:010d}"


class StatementLinks:
    """Fetches a statement's chunk links through a catalog's workspace client.

    Picklable as its catalog is: no literal secret unless the catalog was
    made to ship them (`credentials.databricks.shipping`), which
    ``ship_catalog_auth=True`` and `CredentialBroker.portable` do.
    """

    def __init__(self, catalog: Any, statement_id: str) -> None:
        self.catalog = catalog
        self.statement_id = statement_id

    def links(self, chunk_index: int) -> list[ChunkLink]:
        chunk = self.catalog.workspace.statement_execution.get_statement_result_chunk_n(
            self.statement_id, int(chunk_index)
        )
        return [ChunkLink.of(link) for link in getattr(chunk, "external_links", None) or []]


class _DriverLinks:
    """The driver's own fetcher: the backend that ran the statement. Never pickled."""

    def __init__(self, backend: Any, statement_id: str) -> None:
        self._backend = backend
        self.statement_id = statement_id

    def links(self, chunk_index: int) -> list[ChunkLink]:
        return [
            ChunkLink.of(link)
            for link in self._backend.chunk_links(self.statement_id, int(chunk_index))
        ]

    def __reduce__(self) -> Any:
        raise TypeError(
            "the driver's result-link fetcher holds its workspace client; give workers a "
            "credential_source reaching a CredentialBroker, or plan with ship_catalog_auth=True"
        )


@dataclass(frozen=True)
class WarehouseScanPlan:
    """A distributed read of a warehouse query's result: one split per chunk.

    Picklable. Ship the plan (or `partitions(n)` of it) to workers and call
    `read(splits)` or `stream(splits)` there, as with a `ScanPlan`.
    """

    splits: tuple[WarehouseSplit, ...]
    #: The result's Arrow schema, serialized (an empty result still has one).
    schema_ipc: bytes
    statement_id: str
    columns: tuple[str, ...] | None = None
    predicate: str | None = None
    #: The table's version the query read, when it pinned one.
    snapshot_version: int | None = None
    #: Column name -> year-month interval qualifier, for the text a direct
    #: read gives (the warehouse sends Arrow's month_interval).
    interval_qualifiers: tuple[tuple[str, str], ...] = ()
    #: How workers get fresh links: a picklable ``source(key, operation)``
    #: (`credential_source=`), or a `StatementLinks` that ships catalog auth.
    credential_source: Any = None
    shipped_links: Any = None
    warehouse: str | None = None
    #: The driver's fetcher, for `CredentialBroker.add(plan)`, and the catalog
    #: `CredentialBroker.portable(plan)` ships; neither is pickled.
    driver_links: Any = None
    catalog: Any = None

    #: Tells `Table.to_ray_dataset` and `DeltaSwampDatasource` which plan this is.
    is_warehouse_plan = True

    def __reduce__(self) -> tuple[Any, ...]:
        state = {
            f: getattr(self, f)
            for f in self.__dataclass_fields__
            if f not in ("driver_links", "catalog")
        }
        return (_rebuild, (state,))

    @property
    def link_key(self) -> str:
        """The key a `CredentialBroker` serves this plan's result links under."""
        return STATEMENT_KEY_PREFIX + self.statement_id

    @property
    def total_bytes(self) -> int:
        return sum(s.size for s in self.splits)

    @property
    def total_rows(self) -> int:
        return sum(s.row_count for s in self.splits)

    @property
    def links_expire_at(self) -> float | None:
        """When the earliest link the plan carries expires (epoch seconds), if known."""
        times = [link.expires_at for s in self.splits for link in s.links if link.expires_at]
        return min(times) if times else None

    @property
    def schema(self) -> Any:
        import pyarrow as pa

        return pa.ipc.read_schema(pa.py_buffer(self.schema_ipc))

    def partitions(self, n: int) -> list[tuple[Any, ...]]:
        """The splits in at most `n` groups of about equal bytes."""
        from .distributed import balance

        return balance(self.splits, n)

    def read(self, splits: Iterable[Any] | None = None) -> Any:
        """Read `splits` (default: all) into one Arrow table."""
        import pyarrow as pa

        reader = self.stream(splits)
        return pa.Table.from_batches(list(reader), schema=reader.schema)

    def stream(self, splits: Iterable[Any] | None = None) -> Any:
        """Read `splits` (default: all) as a `pyarrow.RecordBatchReader`, chunk by chunk."""
        import pyarrow as pa

        chosen = list(self.splits if splits is None else splits)
        # The first chunk's own schema, not the one derived from the
        # manifest's type names: the IPC stream is what the warehouse sent.
        first = list(self._chunk_batches(chosen[0])) if chosen else []
        schema = first[0].schema if first else self._output_schema()

        def batches() -> Iterator[Any]:
            yield from first
            for split in chosen[1:]:
                for batch in self._chunk_batches(split):
                    yield batch if batch.schema.equals(schema) else batch.cast(schema)

        return pa.RecordBatchReader.from_batches(schema, batches())

    # ------------------------------------------------------------- workers

    def _output_schema(self) -> Any:
        import pyarrow as pa

        schema = self.schema
        if not self.interval_qualifiers:
            return schema
        from .engine.intervals import month_interval_text

        return month_interval_text(pa, schema, [], dict(self.interval_qualifiers)).schema

    def _chunk_batches(self, split: WarehouseSplit) -> Iterator[Any]:
        import pyarrow as pa

        links = list(split.links)
        if not links or not all(link.fresh() for link in links):
            links = self._fresh_links(split, "its link expired" if links else "it has no link")
        batches: list[Any] = []
        schema = None
        for position, link in enumerate(links):
            try:
                data = _download(link)
            except _ExpiredLink:
                renewed = self._fresh_links(split, "the store refused its link as expired")
                if position >= len(renewed):
                    raise UnreachableTableError(
                        "read a warehouse result chunk",
                        f"chunk {split.chunk_index} came back with fewer links than before",
                    ) from None
                data = _download(renewed[position])
            reader = pa.ipc.open_stream(data)
            schema = schema or reader.schema
            batches.extend(reader)
        if self.interval_qualifiers and schema is not None:
            from .engine.intervals import month_interval_text

            table = month_interval_text(pa, schema, batches, dict(self.interval_qualifiers))
            batches = table.to_batches()
        rows = sum(b.num_rows for b in batches)
        if rows != split.row_count:
            raise UnreachableTableError(
                "read a warehouse result chunk",
                f"chunk {split.chunk_index} of statement {split.statement_id} held {rows} rows, "
                f"and the warehouse reported {split.row_count}",
            )
        yield from batches

    def _fresh_links(self, split: WarehouseSplit, why: str) -> list[ChunkLink]:
        if self.driver_links is not None:
            return list(self.driver_links.links(split.chunk_index))
        if self.shipped_links is not None:
            return list(self.shipped_links.links(split.chunk_index))
        if self.credential_source is not None:
            answer = self.credential_source(self.link_key, f"chunk:{split.chunk_index}")
            links = answer.get("links") if isinstance(answer, dict) else None
            if not links:
                raise UnreachableTableError(
                    "read a warehouse result chunk",
                    f"the credential source returned no links for chunk {split.chunk_index}",
                    "add the plan to the broker on the driver: broker.add(plan), or "
                    "broker.add(CredentialBroker.portable(plan)) for a broker in an actor",
                )
            return [ChunkLink.of(link) for link in links]
        raise UnreachableTableError(
            "read a warehouse result chunk",
            f"chunk {split.chunk_index} cannot be fetched: {why}, and the plan has no way to "
            "fetch a fresh one (a warehouse's result links live about 15 minutes)",
            "plan with credential_source= (a driver-side CredentialBroker the plan was added "
            "to) or ship_catalog_auth=True, or plan again and start reading sooner",
        )


def _rebuild(state: dict[str, Any]) -> WarehouseScanPlan:
    return WarehouseScanPlan(**state)


class _ExpiredLink(Exception):
    """The store refused a presigned link with a status an expired one gets."""


def _check_link(url: str) -> None:
    """Refuse a link this process must not fetch: plain http, or a private host.

    The warehouse chooses these URLs, and a worker fetches them from its own
    network position, so the rules are Delta Sharing's presigned-URL rules
    (`engine.sharing`), including its opt-in for a private network.
    """
    from .engine.sharing import ALLOW_PRIVATE_URLS_ENV, _private_urls_allowed, _public_address

    parts = urlparse(url)
    if parts.scheme.lower() != "https":
        raise UnreachableTableError(
            "read a warehouse result chunk",
            f"the warehouse returned a {parts.scheme or 'relative'} result link; result links "
            "are https presigned URLs",
        )
    if not parts.hostname:
        raise UnreachableTableError(
            "read a warehouse result chunk", "the warehouse returned a result link with no host"
        )
    if not _public_address(parts.hostname) and not _private_urls_allowed():
        raise UnreachableTableError(
            "read a warehouse result chunk",
            f"the warehouse returned a result link to {parts.hostname}, a private, loopback "
            "or link-local address, not to public object storage",
            f"set {ALLOW_PRIVATE_URLS_ENV}=1 for storage on a private network",
        )


def _open(request: urllib.request.Request, timeout: float) -> Any:
    """Open a checked link: every hop, and the address a name resolves to, is public."""
    from .engine.sharing import _CheckedRedirect, _GuardedHTTPHandler, _GuardedHTTPSHandler

    opener = urllib.request.build_opener(
        _GuardedHTTPHandler(), _GuardedHTTPSHandler(), _CheckedRedirect()
    )
    return opener.open(request, timeout=timeout)


#: How a link is opened: patched in tests.
opener: Callable[[urllib.request.Request, float], Any] = _open


def _download(link: ChunkLink) -> bytes:
    _check_link(link.url)
    # Only the headers the link carries: the URL is presigned, and workspace
    # credentials never go to a storage host.
    request = urllib.request.Request(link.url, headers=dict(link.headers), method="GET")
    try:
        with opener(request, DOWNLOAD_TIMEOUT_SECONDS) as response:
            data: bytes = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in _EXPIRED_STATUSES:
            raise _ExpiredLink(str(exc.code)) from exc
        raise UnreachableTableError(
            "read a warehouse result chunk",
            f"the store answered HTTP {exc.code} for {urlparse(link.url).hostname}",
        ) from exc
    except UnreachableTableError:
        raise
    except Exception as exc:
        raise UnreachableTableError(
            "read a warehouse result chunk",
            f"could not download it from {urlparse(link.url).hostname}: {type(exc).__name__}",
        ) from exc
    return data


def WarehouseDatasource(plan: WarehouseScanPlan) -> Any:
    """A Ray Data `Datasource` over a warehouse plan: one read task per group of chunks."""
    from ray.data import ReadTask
    from ray.data.block import BlockMetadata

    from .distributed import _datasource_base

    base = _datasource_base()

    class _Datasource(base):  # type: ignore[misc,valid-type]
        def __init__(self, plan: WarehouseScanPlan) -> None:
            self._plan = plan

        def get_name(self) -> str:
            return "deltaswamp-warehouse"

        def estimate_inmemory_data_size(self) -> int | None:
            return self._plan.total_bytes or None

        def get_read_tasks(
            self, parallelism: int, per_task_row_limit: int | None = None, **_: Any
        ) -> list[Any]:
            groups = self._plan.partitions(max(1, int(parallelism))) or [()]
            tasks = []
            for group in groups:
                plan = replace(self._plan, splits=tuple(group))

                def read(plan: WarehouseScanPlan = plan) -> Iterator[Any]:
                    import pyarrow as pa

                    reader = plan.stream()
                    produced = False
                    for batch in reader:
                        if batch.num_rows:
                            produced = True
                            yield pa.Table.from_batches([batch], schema=reader.schema)
                    if not produced:
                        yield reader.schema.empty_table()

                metadata = BlockMetadata(
                    num_rows=sum(s.row_count for s in group),
                    size_bytes=sum(s.size for s in group),
                    exec_stats=None,
                    input_files=None,
                )
                limit = (
                    {} if per_task_row_limit is None else {"per_task_row_limit": per_task_row_limit}
                )
                tasks.append(ReadTask(read, metadata, **limit))
            return tasks

    return _Datasource(plan)
