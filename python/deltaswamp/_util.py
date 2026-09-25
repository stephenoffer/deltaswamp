"""Small helpers shared across engines and catalogs."""

from __future__ import annotations

import datetime as dt
from typing import Any

from .errors import UnreachableTableError


def timestamp_ms(value: Any) -> int:
    """Epoch milliseconds from a datetime, a date, an ISO-8601 string or a number.

    A naive datetime or zoneless string is read as UTC.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
        try:
            value = dt.datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError as exc:
            raise UnreachableTableError(
                f"time travel to {value!r}", "not an ISO-8601 timestamp or epoch milliseconds"
            ) from exc
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = dt.datetime(value.year, value.month, value.day)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        # Exact integer arithmetic: float truncation rounded pre-1970 instants
        # toward zero, a millisecond late.
        epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
        return int((value - epoch) // dt.timedelta(milliseconds=1))
    raise UnreachableTableError(
        f"time travel to {value!r}", "expected a datetime, an ISO-8601 string or epoch millis"
    )


def enum_value(value: Any) -> str | None:
    """The string value of an SDK enum member, a plain string, or None."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def http_error_text(exc: BaseException) -> str:
    """An HTTP client error as ``HTTP <status>: <message>``."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    message = str(exc).strip()
    return f"HTTP {status}: {message}" if status is not None else message


def commit_backoff(attempt: int, *, base: float = 0.02, cap: float = 1.0) -> None:
    """Sleep before re-attempting a commit that lost its race.

    Writers that collide retry in lockstep without it: each re-reads the same
    snapshot at the same moment and they collide again, so eight threads
    appending at once used up five immediate retries and some still failed.
    Full jitter spreads them out; the cap keeps a long losing streak from
    stalling a writer for more than about a second at a time.
    """
    import random
    import time

    time.sleep(random.uniform(0, min(cap, base * (2**attempt))))
