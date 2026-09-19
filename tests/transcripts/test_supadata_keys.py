from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.transcripts import supadata_keys
from src.transcripts.fetcher import SuperdataTranscriptFetcher, TranscriptFetchError
from src.transcripts.supadata_keys import SupadataKeyRing, SupadataQuotaError, ring_for

QUOTA = {
    "error": "limit-exceeded",
    "message": "Limit Exceeded",
    "details": "Plan usage limit was exceeded.",
}
THROTTLE = {
    "error": "limit-exceeded",
    "message": "Limit Exceeded",
    "details": "Request rate limit on current plan was exceeded.",
}
SEGMENTS = {
    "content": [{"text": "hello", "offset": 0, "duration": 1000, "lang": "en"}],
    "lang": "en",
}


class FakeSupadata:
    """Scripted responses per key: each key's list is consumed in order."""

    def __init__(self, script: dict[str, list[httpx.Response]]) -> None:
        self.script = {key: list(responses) for key, responses in script.items()}
        self.calls: list[tuple[str, str]] = []  # (key, url)

    def __call__(
        self, url: str, params: Any = None, headers: Any = None, timeout: Any = None
    ) -> httpx.Response:
        key = headers["x-api-key"]
        self.calls.append((key, url))
        queue = self.script.get(key)
        if not queue:
            raise AssertionError(f"unexpected request with key {key!r} to {url}")
        return queue.pop(0)

    def keys_used(self) -> list[str]:
        return [key for key, _ in self.calls]


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supadata_keys.time, "sleep", lambda _s: None)


def _install(
    monkeypatch: pytest.MonkeyPatch, script: dict[str, list[httpx.Response]]
) -> FakeSupadata:
    fake = FakeSupadata(script)
    monkeypatch.setattr(httpx, "get", fake)
    return fake


def test_quota_429_rolls_to_the_second_key_and_sticks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(
        monkeypatch,
        {
            "k1": [httpx.Response(429, json=QUOTA)],
            "k2": [httpx.Response(200, json={"ok": 1}), httpx.Response(200, json={"ok": 2})],
        },
    )
    ring = SupadataKeyRing(("k1", "k2"))

    first, used = ring.get("https://api.supadata.ai/v1/transcript", {"url": "u"})
    assert (first.json(), used) == ({"ok": 1}, 1)
    assert ring.status() == {"keys": 2, "active": 2, "exhausted": [1]}

    second, used = ring.get("https://api.supadata.ai/v1/transcript", {"url": "u"})
    assert (second.json(), used) == ({"ok": 2}, 1)
    # The dead key is not probed again for the life of the ring.
    assert fake.keys_used() == ["k1", "k2", "k2"]


def test_rate_limit_429_retries_the_same_key_without_advancing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install(
        monkeypatch,
        {"k1": [httpx.Response(429, json=THROTTLE), httpx.Response(200, json={"ok": 1})], "k2": []},
    )
    ring = SupadataKeyRing(("k1", "k2"))

    response, used = ring.get("https://api.supadata.ai/v1/metadata", {"url": "u"})

    assert (response.json(), used) == ({"ok": 1}, 0)
    assert fake.keys_used() == ["k1", "k1"]
    assert ring.status() == {"keys": 2, "active": 1, "exhausted": []}


def test_persistent_rate_limit_is_treated_as_exhaustion_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install(
        monkeypatch,
        {"k1": [httpx.Response(429, json=THROTTLE)] * 3, "k2": [httpx.Response(200, json={})]},
    )
    ring = SupadataKeyRing(("k1", "k2"))

    _, used = ring.get("https://api.supadata.ai/v1/metadata", {"url": "u"})

    assert used == 1
    assert fake.keys_used() == ["k1", "k1", "k1", "k2"]


def test_every_key_exhausted_names_each_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {"k1": [httpx.Response(429, json=QUOTA)], "k2": [httpx.Response(429, text="nope")]},
    )
    ring = SupadataKeyRing(("k1", "k2"))

    with pytest.raises(SupadataQuotaError) as excinfo:
        ring.get("https://api.supadata.ai/v1/transcript", {"url": "u"})

    message = str(excinfo.value)
    assert message.startswith("all 2 Supadata keys exhausted")
    assert "key 1: Plan usage limit was exceeded." in message
    assert "key 2: nope" in message
    assert ring.status()["exhausted"] == [1, 2]


def test_pinned_request_never_fails_over(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, {"k1": [httpx.Response(429, json=QUOTA)], "k2": []})
    ring = SupadataKeyRing(("k1", "k2"))

    response, used = ring.get("https://api.supadata.ai/v1/transcript/job", key_index=0)

    assert (response.status_code, used) == (429, 0)
    assert fake.keys_used() == ["k1"]
    assert ring.status()["exhausted"] == []


def test_other_errors_pass_through_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"k1": [httpx.Response(404, json={"error": "not-found"})], "k2": []})
    ring = SupadataKeyRing(("k1", "k2"))

    response, used = ring.get("https://api.supadata.ai/v1/transcript", {"url": "u"})

    assert (response.status_code, used) == (404, 0)


def test_ring_for_shares_one_ring_per_key_set() -> None:
    assert ring_for(["a", "b"]) is ring_for(("a", "b"))
    assert ring_for("a") is ring_for(["a"])
    assert ring_for("a") is not ring_for(["a", "b"])
    assert ring_for(["a", "", "b"]).keys == ("a", "b")


def test_ring_needs_a_key() -> None:
    with pytest.raises(ValueError):
        SupadataKeyRing(())


def test_network_error_retries_the_same_key_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def flaky(
        url: str, params: Any = None, headers: Any = None, timeout: Any = None
    ) -> httpx.Response:
        calls.append(headers["x-api-key"])
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json={"ok": 1})

    monkeypatch.setattr(httpx, "get", flaky)
    ring = SupadataKeyRing(("k1", "k2"))

    response, used = ring.get("https://api.supadata.ai/v1/transcript", {"url": "u"})

    assert (response.json(), used) == ({"ok": 1}, 0)
    assert calls == ["k1", "k1"]
    assert ring.status() == {"keys": 2, "active": 1, "exhausted": []}


def test_persistent_network_error_propagates_as_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fails(
        url: str, params: Any = None, headers: Any = None, timeout: Any = None
    ) -> httpx.Response:
        raise httpx.ConnectError("connection reset")

    monkeypatch.setattr(httpx, "get", always_fails)
    ring = SupadataKeyRing(("k1", "k2"))

    with pytest.raises(httpx.HTTPError):
        ring.get("https://api.supadata.ai/v1/transcript", {"url": "u"})


# --- the fetcher on top of the ring ------------------------------------------


def test_fetcher_falls_back_to_the_second_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(
        monkeypatch,
        {
            "k1": [httpx.Response(429, json=QUOTA)],
            "k2": [httpx.Response(200, json=SEGMENTS), httpx.Response(200, json={"title": "T"})],
        },
    )
    fetcher = SuperdataTranscriptFetcher(["k1", "k2"])

    transcript = fetcher.fetch("https://www.youtube.com/watch?v=3hk7nO_q0a8")

    assert transcript.raw_text == "hello"
    assert transcript.title == "T"
    # transcript on k1 (429) → k2; metadata goes straight to k2.
    assert fake.keys_used() == ["k1", "k2", "k2"]


def test_fetcher_polls_an_async_job_with_the_key_that_started_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install(
        monkeypatch,
        {
            "k1": [
                httpx.Response(202, json={"jobId": "j1"}),
                httpx.Response(200, json={"status": "completed", **SEGMENTS}),
            ],
            "k2": [httpx.Response(200, json={"title": "T"})],
        },
    )
    fetcher = SuperdataTranscriptFetcher(["k1", "k2"], poll_interval_seconds=0)
    ring = fetcher.ring
    scripted = fake.__call__

    def retire_k1_after_the_job_starts(url: str, **kwargs: Any) -> httpx.Response:
        response = scripted(url, **kwargs)
        if response.status_code == 202:
            # Another worker retires k1 between the 202 and the first poll.
            ring.exhausted[0] = "Plan usage limit was exceeded."
            ring.active = 1
        return response

    monkeypatch.setattr(httpx, "get", retire_k1_after_the_job_starts)

    transcript = fetcher.fetch("https://www.youtube.com/watch?v=3hk7nO_q0a8")

    assert transcript.raw_text == "hello"
    assert fake.calls[1] == ("k1", "https://api.supadata.ai/v1/transcript/j1")
    assert fake.calls[2] == ("k2", "https://api.supadata.ai/v1/metadata")


def test_fetcher_reports_exhaustion_as_a_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {"k1": [httpx.Response(429, json=QUOTA)], "k2": [httpx.Response(429, json=QUOTA)]},
    )
    fetcher = SuperdataTranscriptFetcher(["k1", "k2"])

    with pytest.raises(TranscriptFetchError, match="all 2 Supadata keys exhausted"):
        fetcher.fetch("https://www.youtube.com/watch?v=3hk7nO_q0a8")


def test_single_key_fetcher_keeps_the_old_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"only": [httpx.Response(429, json=QUOTA)]})
    fetcher = SuperdataTranscriptFetcher("only")

    with pytest.raises(TranscriptFetchError, match="all 1 Supadata keys exhausted"):
        fetcher.fetch("https://www.youtube.com/watch?v=3hk7nO_q0a8")
