"""Streaming voice-to-text: a WebSocket relay from the browser to Deepgram.

The browser captures raw PCM (linear16) with an AudioWorklet and streams it to
``/ws/stt``; this module forwards the audio to Deepgram's live endpoint and
streams ``{text, is_final, speech_final}`` JSON back, so interim words appear
in the composer as they are spoken. The relay exists so the Deepgram key never
reaches the browser and so the frontend codes against this app's own message
shape rather than Deepgram's — swapping the transcription vendor later is a
backend-only change.

Demo mode must be refused *here*: the app's demo gate is HTTP middleware,
which never runs for WebSocket connections.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

from fastapi import WebSocket

DEEPGRAM_LIVE_URL = "wss://api.deepgram.com/v1/listen"

#: Hard ceiling on one utterance session. Streaming bills by the minute, so an
#: abandoned open mic must not run unattended; the client also stops itself.
SESSION_CAP_SECONDS = 120.0

#: Deepgram drops a live connection after ~10s without audio; KeepAlive frames
#: hold it open across thinking pauses.
KEEPALIVE_INTERVAL_SECONDS = 5.0

# Anything a real browser AudioContext produces sits comfortably in this range.
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000

ConnectFn = Callable[[str, str], Awaitable[Any]]


def stt_available(*, api_key: str, enabled: bool, demo_mode: bool) -> bool:
    """Whether the mic should exist at all — mirrored into /api/health."""
    return bool(api_key) and enabled and not demo_mode


def deepgram_url(model: str, sample_rate: int) -> str:
    query = urlencode(
        {
            "model": model,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "interim_results": "true",
            "smart_format": "true",
            "endpointing": 300,
        }
    )
    return f"{DEEPGRAM_LIVE_URL}?{query}"


async def _default_connect(url: str, api_key: str) -> Any:
    import websockets

    return await websockets.connect(url, additional_headers={"Authorization": f"Token {api_key}"})


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


async def _pump_client_audio(websocket: WebSocket, upstream: Any) -> None:
    """Browser → Deepgram: binary frames are audio, ``{"type":"stop"}`` ends it.

    On stop this sends Deepgram's CloseStream and keeps draining the client
    socket rather than returning: Deepgram flushes trailing finals and then
    closes, which ends the transcript pump — the side that should decide when
    the session is over.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            with contextlib.suppress(Exception):
                await upstream.send('{"type": "CloseStream"}')
            return
        chunk = message.get("bytes")
        if chunk:
            await upstream.send(chunk)
            continue
        text = message.get("text")
        if text:
            try:
                stop = json.loads(text).get("type") == "stop"
            except (TypeError, ValueError):
                stop = False
            if stop:
                with contextlib.suppress(Exception):
                    await upstream.send('{"type": "CloseStream"}')


async def _pump_transcripts(upstream: Any, websocket: WebSocket) -> None:
    """Deepgram → browser, translated to this app's message shape."""
    async for raw in upstream:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if data.get("type") != "Results":
            continue
        alternatives = (data.get("channel") or {}).get("alternatives") or []
        text = str((alternatives[0] if alternatives else {}).get("transcript") or "")
        is_final = bool(data.get("is_final"))
        # Empty interims are noise; empty finals still matter — the client
        # clears its interim tail on them.
        if not text and not is_final:
            continue
        await _send_json(
            websocket,
            {
                "type": "transcript",
                "text": text,
                "is_final": is_final,
                "speech_final": bool(data.get("speech_final")),
            },
        )


async def _keepalive(upstream: Any) -> None:
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
        await upstream.send('{"type": "KeepAlive"}')


async def relay_stt(
    websocket: WebSocket,
    *,
    api_key: str,
    model: str,
    sample_rate: int,
    connect: ConnectFn | None = None,
    session_cap_seconds: float = SESSION_CAP_SECONDS,
) -> None:
    """Accept the browser socket and relay one utterance session."""
    connect = connect or _default_connect
    await websocket.accept()

    if not (MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE):
        await websocket.close(code=1008, reason="unsupported sample rate")
        return

    try:
        upstream = await connect(deepgram_url(model, sample_rate), api_key)
    except Exception:
        with contextlib.suppress(Exception):
            await _send_json(
                websocket,
                {"type": "error", "message": "transcription service unreachable"},
            )
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
        return

    tasks = {
        asyncio.create_task(_pump_client_audio(websocket, upstream)),
        asyncio.create_task(_pump_transcripts(upstream, websocket)),
        asyncio.create_task(_keepalive(upstream)),
    }
    try:
        # First side to finish ends the session: the transcript pump on
        # Deepgram's post-CloseStream flush, the audio pump on a client
        # disconnect, or neither within the hard cap.
        await asyncio.wait(tasks, timeout=session_cap_seconds, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await upstream.close()
        with contextlib.suppress(Exception):
            await websocket.close()
