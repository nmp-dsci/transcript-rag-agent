"""One ordered ring of Supadata API keys, shared by every client in a process.

The first key is always tried first. A key that answers 429 with
"Plan usage limit was exceeded" is out of credits for the rest of the month:
the ring marks it, moves on to the next key, and stays there for the life of
the process. A 429 that is only a per-second rate limit is retried on the same
key with a short backoff and never advances the ring, so a transient throttle
cannot start spending the fallback key. Every configured key exhausted raises
:class:`SupadataQuotaError` naming each key's reason.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

#: Substrings of Supadata's ``details`` field (docs.supadata.ai/errors/limit-exceeded).
QUOTA_MARKERS = ("usage limit",)
THROTTLE_MARKERS = ("rate limit",)


class SupadataQuotaError(RuntimeError):
    """Every configured key has reported its plan usage limit exceeded."""


@dataclass
class SupadataKeyRing:
    keys: tuple[str, ...]
    timeout_seconds: float = 120.0
    #: Attempts on one key for a rate-limit 429 before it is treated as exhausted.
    throttle_retries: int = 3
    active: int = 0
    #: Index → the ``details`` Supadata gave when that key was retired.
    exhausted: dict[int, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("SupadataKeyRing needs at least one key")

    @property
    def active_key(self) -> str:
        return self.keys[self.active]

    def status(self) -> dict[str, Any]:
        """One-based indexes only — never key material — for ``/api/health``."""
        return {
            "keys": len(self.keys),
            "active": self.active + 1,
            "exhausted": sorted(index + 1 for index in self.exhausted),
        }

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        key_index: int | None = None,
    ) -> tuple[httpx.Response, int]:
        """GET ``url`` with the active key; returns ``(response, key index used)``.

        ``key_index`` pins the request to one key (an async job must be
        polled by the key that started it); a pinned 429 is returned to the
        caller rather than failing over. A network error is retried like a
        throttle 429, then propagates as ``httpx.HTTPError`` if it persists.
        """
        if key_index is not None:
            return self._get_with_backoff(url, params, key_index), key_index
        while True:
            index = self.active
            response = self._get_with_backoff(url, params, index)
            if response.status_code != 429:
                return response, index
            self._mark_exhausted(index, _details(response))

    def _get_with_backoff(
        self, url: str, params: dict[str, Any] | None, index: int
    ) -> httpx.Response:
        headers = {"x-api-key": self.keys[index]}
        response: httpx.Response | None = None
        for attempt in range(self.throttle_retries):
            try:
                response = httpx.get(
                    url, params=params, headers=headers, timeout=self.timeout_seconds
                )
            except httpx.HTTPError:
                if attempt == self.throttle_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code != 429 or not _is_throttle(_details(response)):
                return response
            time.sleep(0.5 * (attempt + 1))
        assert response is not None
        return response

    def _mark_exhausted(self, index: int, reason: str) -> None:
        with self._lock:
            self.exhausted.setdefault(index, reason or "HTTP 429")
            remaining = [i for i in range(len(self.keys)) if i not in self.exhausted]
            if not remaining:
                detail = ", ".join(
                    f"key {i + 1}: {why}" for i, why in sorted(self.exhausted.items())
                )
                raise SupadataQuotaError(f"all {len(self.keys)} Supadata keys exhausted ({detail})")
            if self.active == index:
                self.active = remaining[0]
                log.warning(
                    "supadata key %d exhausted (%s); using key %d",
                    index + 1,
                    reason,
                    self.active + 1,
                )


def _details(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if not isinstance(body, dict):
        return ""
    return str(body.get("details") or body.get("message") or "")


def _is_throttle(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in THROTTLE_MARKERS) and not any(
        marker in lowered for marker in QUOTA_MARKERS
    )


_rings: dict[tuple[str, ...], SupadataKeyRing] = {}
_rings_lock = threading.Lock()


def ring_for(
    keys: str | Sequence[str] | SupadataKeyRing, timeout_seconds: float = 120.0
) -> SupadataKeyRing:
    """The one ring for this key set in this process.

    Concurrent ingestion workers each build their own fetcher; sharing the
    ring means key 1 is discovered exhausted once, not once per worker. A
    plain string (every pre-existing caller) is a one-key ring.
    """
    if isinstance(keys, SupadataKeyRing):
        return keys
    ordered = (keys,) if isinstance(keys, str) else tuple(key for key in keys if key)
    with _rings_lock:
        ring = _rings.get(ordered)
        if ring is None:
            ring = _rings[ordered] = SupadataKeyRing(ordered, timeout_seconds=timeout_seconds)
        return ring


def reset_rings() -> None:
    """Forget every shared ring (tests)."""
    with _rings_lock:
        _rings.clear()
