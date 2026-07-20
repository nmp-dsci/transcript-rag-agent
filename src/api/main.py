"""FastAPI app serving the live transcript RAG chat.

The app wraps the same building blocks as the CLI ``chat`` session: questions
are answered by one or more selectable RAG setups via ``RagSetupRunner``, every
answered question is appended to the shared chat history, and the static
``chat.html`` viewer is regenerated so both surfaces stay in sync.

``POST /api/ask`` streams server-sent events so the browser can show progress
and render each setup's answer as soon as it completes.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.chat.frontend import (
    ANSWER_CSS,
    ANSWER_RENDER_JS,
    DEFAULT_CHAT_HTML_PATH,
    write_chat_html,
)
from src.chat.history import (
    DEFAULT_HISTORY_PATH,
    ChatAnswer,
    append_entry,
    build_entry,
    load_history,
)
from src.chat.setups import SETUP_KEYS, SETUP_SPECS, RagSetupRunner, setup_spec
from src.config import Settings, load_settings

INDEX_HTML_PATH = Path(__file__).parent / "static" / "index.html"

IndexFn = Callable[[list[str]], int]


def _default_index_fn(argv: list[str]) -> int:
    """Run an indexing CLI command, reusing the exact documented code path."""
    from src import cli

    return cli.main(argv)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    setups: list[str] = Field(default_factory=lambda: list(SETUP_KEYS), min_length=1)
    url: str | None = None


class IndexRequest(BaseModel):
    mode: Literal["video", "channel"]
    url: str | None = None
    channel: str | None = None
    latest: int = Field(default=5, ge=1, le=50)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None,
    *,
    runner_factory: Callable[[], RagSetupRunner] | None = None,
    history_path: Path = DEFAULT_HISTORY_PATH,
    chat_html_path: Path = DEFAULT_CHAT_HTML_PATH,
    index_fn: IndexFn = _default_index_fn,
) -> FastAPI:
    resolved = settings or load_settings(require_keys=True)
    factory = runner_factory or (lambda: RagSetupRunner.from_settings(resolved))

    app = FastAPI(title="Transcript RAG Chat", version="0.1.0")
    runner_lock = threading.Lock()
    runner_holder: dict[str, RagSetupRunner] = {}

    def runner_loaded() -> bool:
        return "runner" in runner_holder

    def get_runner() -> RagSetupRunner:
        # Building the retrieval stack loads embedding models, so do it once,
        # on the first question, never at startup.
        with runner_lock:
            if "runner" not in runner_holder:
                runner_holder["runner"] = factory()
            return runner_holder["runner"]

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> str:
        return INDEX_HTML_PATH.read_text(encoding="utf-8")

    @app.get("/assets/render.js")
    def render_js() -> PlainTextResponse:
        return PlainTextResponse(ANSWER_RENDER_JS, media_type="text/javascript")

    @app.get("/assets/answer.css")
    def answer_css() -> PlainTextResponse:
        return PlainTextResponse(ANSWER_CSS, media_type="text/css")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "runner_loaded": runner_loaded()}

    @app.get("/api/setups")
    def setups() -> dict:
        return {"setups": [asdict(spec) for spec in SETUP_SPECS]}

    @app.get("/api/history")
    def history() -> dict:
        return {
            "conversations": [entry.to_dict() for entry in load_history(history_path)]
        }

    @app.post("/api/ask")
    def ask(payload: AskRequest) -> StreamingResponse:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question must not be blank")
        unknown = [key for key in payload.setups if key not in SETUP_KEYS]
        if unknown:
            raise HTTPException(
                status_code=422, detail=f"Unknown setup(s): {', '.join(unknown)}"
            )
        keys = list(dict.fromkeys(payload.setups))
        url = payload.url.strip() if payload.url and payload.url.strip() else None

        def stream() -> Iterator[str]:
            # One failing setup is already captured as SetupResult.error by the
            # runner; this guard is for everything else (stack build, storage),
            # which must surface as an event rather than a dead stream.
            try:
                if not runner_loaded():
                    yield _sse(
                        "progress",
                        {"message": "Loading retrieval stack (first question only)..."},
                    )
                runner = get_runner()
                results = []
                for key in keys:
                    yield _sse(
                        "progress",
                        {"key": key, "message": f"Running {setup_spec(key).title} ..."},
                    )
                    result = runner.run(key, question, url=url)
                    results.append(result)
                    yield _sse("answer", asdict(ChatAnswer.from_result(result)))
                entry = build_entry(question, results, url=url)
                entries = append_entry(entry, history_path)
                write_chat_html(entries, chat_html_path)
                yield _sse("done", entry.to_dict())
            except Exception as exc:
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/index")
    def index_content(payload: IndexRequest) -> dict:
        if payload.mode == "video":
            if not payload.url:
                raise HTTPException(
                    status_code=422, detail="url is required when mode is 'video'"
                )
            argv = ["index-rag", payload.url]
            target = payload.url
        else:
            if not payload.channel:
                raise HTTPException(
                    status_code=422, detail="channel is required when mode is 'channel'"
                )
            argv = [
                "bulk-index",
                "channel",
                "--channel",
                payload.channel,
                "--latest",
                str(payload.latest),
            ]
            target = payload.channel
        exit_code = index_fn(argv)
        return {"ok": exit_code == 0, "exit_code": exit_code, "target": target}

    return app
