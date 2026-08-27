"""The voice-to-text relay: /ws/stt and its health flag.

The properties that matter: the mic only exists when the server says so
(health ``stt``), demo mode refuses the socket *in the endpoint* (the demo
gate is HTTP middleware and never runs for WebSockets), and the relay
translates Deepgram's Results frames into this app's own message shape.
Deepgram itself is faked — no network, no key.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.main import create_app
from src.api.stt import deepgram_url, stt_available
from src.config import Settings


def forbidden_factory():
    raise AssertionError("the STT relay must never load the LLM/retrieval stack")


class FakeDeepgram:
    """Stands in for Deepgram's live socket.

    Every audio frame is answered with one Results message (first interim,
    then final), and CloseStream ends the stream — the same order the real
    service produces, minus the network.
    """

    def __init__(self) -> None:
        self.received: list[object] = []
        self.closed = False
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._audio_frames = 0

    def _results(self, text: str, is_final: bool) -> str:
        return json.dumps(
            {
                "type": "Results",
                "is_final": is_final,
                "speech_final": is_final,
                "channel": {"alternatives": [{"transcript": text}]},
            }
        )

    async def send(self, message: object) -> None:
        self.received.append(message)
        if isinstance(message, (bytes, bytearray)):
            self._audio_frames += 1
            if self._audio_frames == 1:
                await self._queue.put(self._results("what does the", False))
            else:
                await self._queue.put(self._results("What does the channel say?", True))
        elif isinstance(message, str):
            if json.loads(message).get("type") == "CloseStream":
                await self._queue.put(None)

    def __aiter__(self) -> "FakeDeepgram":
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        self.closed = True


def stt_client(
    settings: Settings, tmp_path: Path, *, connected: dict | None = None, **overrides
) -> TestClient:
    async def fake_connect(url: str, api_key: str):
        fake = FakeDeepgram()
        if connected is not None:
            connected.update({"fake": fake, "url": url, "api_key": api_key})
        return fake

    kwargs = dict(
        runner_factory=forbidden_factory,
        judge_factory=forbidden_factory,
        graph_store_factory=forbidden_factory,
        corpus_fn=lambda: {"videos": [], "channels": [], "totals": {}, "insights": []},
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        stt_connect_fn=fake_connect,
    )
    kwargs.update(overrides)
    return TestClient(create_app(settings, **kwargs))


def test_stt_available_requires_key_flag_and_full_mode():
    assert stt_available(api_key="k", enabled=True, demo_mode=False)
    assert not stt_available(api_key="", enabled=True, demo_mode=False)
    assert not stt_available(api_key="k", enabled=False, demo_mode=False)
    assert not stt_available(api_key="k", enabled=True, demo_mode=True)


def test_deepgram_url_pins_linear16_and_interims():
    url = deepgram_url("nova-3", 48_000)
    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    for fragment in (
        "model=nova-3",
        "encoding=linear16",
        "sample_rate=48000",
        "interim_results=true",
    ):
        assert fragment in url


def test_health_reports_stt_when_key_present(settings, tmp_path):
    client = stt_client(replace(settings, deepgram_api_key="dg-key"), tmp_path)
    assert client.get("/api/health").json()["stt"] is True


def test_health_reports_no_stt_without_key(settings, tmp_path):
    assert stt_client(settings, tmp_path).get("/api/health").json()["stt"] is False


def test_health_reports_no_stt_in_demo_mode(settings, tmp_path):
    client = stt_client(replace(settings, deepgram_api_key="dg-key", demo_mode=True), tmp_path)
    assert client.get("/api/health").json()["stt"] is False


def _expect_policy_close(client: TestClient) -> None:
    with client.websocket_connect("/ws/stt") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1008


def test_demo_mode_refuses_the_socket(settings, tmp_path):
    _expect_policy_close(
        stt_client(replace(settings, deepgram_api_key="dg-key", demo_mode=True), tmp_path)
    )


def test_missing_key_refuses_the_socket(settings, tmp_path):
    _expect_policy_close(stt_client(settings, tmp_path))


def test_absurd_sample_rate_refuses(settings, tmp_path):
    client = stt_client(replace(settings, deepgram_api_key="dg-key"), tmp_path)
    with client.websocket_connect("/ws/stt?sample_rate=1") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1008


def test_relay_streams_interim_then_final_and_closes_on_stop(settings, tmp_path):
    connected: dict = {}
    client = stt_client(
        replace(settings, deepgram_api_key="dg-key", stt_model="nova-3"),
        tmp_path,
        connected=connected,
    )

    with client.websocket_connect("/ws/stt?sample_rate=48000") as ws:
        ws.send_bytes(b"\x00\x01" * 128)
        interim = ws.receive_json()
        ws.send_bytes(b"\x00\x01" * 128)
        final = ws.receive_json()
        ws.send_text('{"type": "stop"}')
        # CloseStream ends the fake's stream, which closes the session.
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    assert interim == {
        "type": "transcript",
        "text": "what does the",
        "is_final": False,
        "speech_final": False,
    }
    assert final["is_final"] is True
    assert final["text"] == "What does the channel say?"

    assert connected["api_key"] == "dg-key"
    assert "sample_rate=48000" in connected["url"]
    fake = connected["fake"]
    audio = [m for m in fake.received if isinstance(m, (bytes, bytearray))]
    assert len(audio) == 2
    assert any(
        isinstance(m, str) and json.loads(m).get("type") == "CloseStream" for m in fake.received
    )
