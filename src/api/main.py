"""FastAPI app serving the transcript RAG evaluation workbench.

The app wraps the same building blocks as the CLI ``chat`` session: questions
are answered by one or more selectable RAG setups via ``RagSetupRunner``, every
answered question is appended to the shared chat history, and the static
``chat.html`` viewer is regenerated so both surfaces stay in sync.

On top of asking, the workbench evaluates: ``POST /api/judge`` scores every
setup's answer to a question with the same RAGAS metrics (faithfulness, answer
relevancy, context precision), ``GET /api/scoreboard`` aggregates those scores
per retrieval method, and ``GET /api/corpus`` lists the indexed videos.

``POST /api/ask`` and ``POST /api/judge`` stream server-sent events so the
browser can show progress and render results as they complete.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.corpus import (
    list_chunks,
    list_corpus,
    load_chunk_corpus,
    load_chunk_embeddings,
)
from src.api.ranking import MODES, build_rankings
from src.api.scoreboard import build_scoreboard
from src.chat.frontend import (
    ANSWER_CSS,
    ANSWER_RENDER_JS,
    DEFAULT_CHAT_HTML_PATH,
    write_chat_html,
)
from src.chat.history import (
    DEFAULT_HISTORY_PATH,
    ChatAnswer,
    ChatEntry,
    append_entry,
    build_entry,
    load_history,
    update_entry,
)
from src.chat.setups import (
    SETUP_KEYS,
    SETUP_SPECS,
    AskScope,
    RagSetupRunner,
    setup_spec,
)
from src.config import Settings, load_settings
from src.evals.judge import RagasJudge, unjudgeable

INDEX_HTML_PATH = Path(__file__).parent / "static" / "index.html"
# Built React bundle (frontend/npm run build). Gitignored, so it may be absent.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

IndexFn = Callable[[list[str]], int]


def _default_index_fn(argv: list[str]) -> int:
    """Run an indexing CLI command, reusing the exact documented code path."""
    from src import cli

    return cli.main(argv)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    setups: list[str] = Field(default_factory=lambda: list(SETUP_KEYS), min_length=1)
    url: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    # Answer into an existing entry instead of creating a new one. Running more
    # setups on a question already asked has to land in the same entry, or the
    # scoreboard would never see them as competing answers.
    entry_id: str | None = None
    # Scope retrieval to one channel. Ignored when ``url`` pins a single video,
    # which is already narrower.
    channel_id: str | None = None
    retrieval_mode: Literal["semantic", "hybrid"] | None = None
    # Summary-first transcript filtering (the S4 path), previously CLI-only.
    filter_transcripts: bool = False
    # Prior turns to condense into a standalone retrieval query.
    history: list[str] = Field(default_factory=list)


class GraphRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    query: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class JudgeRequest(BaseModel):
    entry_id: str
    force: bool = False


class IndexRequest(BaseModel):
    mode: Literal["video", "channel"]
    url: str | None = None
    channel: str | None = None
    latest: int = Field(default=5, ge=1, le=50)


class RankRequest(BaseModel):
    query: str = Field(min_length=1)
    video_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    modes: list[Literal["semantic", "bm25"]] = Field(
        default_factory=lambda: list(MODES), min_length=1
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# Discovery and fetch are real, observable moments ahead of the single
# blocking CLI call that does everything else (chunking, embedding,
# summarizing), so only those two get their own stage.
_INDEX_STAGES = [
    ("discover", "Resolving target(s) ..."),
    ("fetch", "Fetching transcripts and building the index ..."),
]

# The CLI call is one indivisible unit of work with no internal progress
# reporting, so it gets a single ongoing stage rather than a fake progression
# through invented sub-steps. The heartbeat repeats that stage name at
# intervals so a reverse proxy or load balancer never sees an idle SSE
# connection during a multi-minute bulk index.
_INDEX_PROCESSING_STAGE = "processing"
_INDEX_PROCESSING_MESSAGE = "Chunking, embedding, and summarizing ..."
_INDEX_HEARTBEAT_MESSAGE = "Still indexing..."
_INDEX_HEARTBEAT_INTERVAL_SECONDS = 8.0


def _index_argv(payload: "IndexRequest") -> tuple[list[str], str]:
    """The CLI argv for an index request, and the human-readable target."""
    if payload.mode == "video":
        if not payload.url:
            raise HTTPException(
                status_code=422, detail="url is required when mode is 'video'"
            )
        return ["index-rag", payload.url], payload.url
    if not payload.channel:
        raise HTTPException(
            status_code=422, detail="channel is required when mode is 'channel'"
        )
    return (
        [
            "bulk-index",
            "channel",
            "--channel",
            payload.channel,
            "--latest",
            str(payload.latest),
        ],
        payload.channel,
    )


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _append_answers(
    entry_id: str, results: list, history_path: Path
) -> tuple[ChatEntry, list[ChatEntry]]:
    """Add newly-run setups to an existing entry, replacing same-key answers."""
    fresh = [ChatAnswer.from_result(result) for result in results]

    def mutate(entry: ChatEntry) -> None:
        replaced = {answer.key for answer in fresh}
        entry.answers = [
            answer for answer in entry.answers if answer.key not in replaced
        ] + fresh

    updated, entries = update_entry(entry_id, mutate, history_path)
    assert updated is not None  # existence is checked before the stream starts
    return updated, entries


def _run_setup_streaming(
    runner: Any,
    key: str,
    question: str,
    url: str | None,
    top_k: int | None,
    scope: Any = None,
) -> Iterator[tuple[str, Any]]:
    """Run one setup, yielding ``("step", event)`` then ``("result", result)``.

    The runner reports agent research steps through a synchronous callback, so
    the setup runs on a worker thread and its events are drained from a queue
    here — otherwise nothing could be sent until the whole answer was finished.
    """
    events: queue.Queue = queue.Queue()
    holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            holder["result"] = runner.run(
                key,
                question,
                url=url,
                top_k=top_k,
                on_agent_event=lambda event: events.put(("step", event)),
                scope=scope,
            )
        except Exception as exc:  # pragma: no cover - runner captures its own
            holder["error"] = exc
        finally:
            events.put(("finished", None))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        kind, value = events.get()
        if kind == "finished":
            break
        yield kind, value
    thread.join()
    if "error" in holder:
        raise holder["error"]
    yield "result", holder["result"]


def _run_index_streaming(
    index_fn: IndexFn, argv: list[str]
) -> Iterator[tuple[str, Any]]:
    """Run ``index_fn`` on a worker thread, yielding ``("heartbeat", None)`` while waiting.

    Same worker+queue shape as ``_run_setup_streaming``, except ``index_fn``
    reports no progress of its own, so the wait loop's ``queue.get`` is given a
    timeout: each time it elapses without the worker finishing, a heartbeat is
    yielded and the wait resumes. Yields ``("result", exit_code)`` once the
    worker completes.
    """
    events: queue.Queue = queue.Queue()
    holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            holder["exit_code"] = index_fn(argv)
        except Exception as exc:
            holder["error"] = exc
        finally:
            events.put(("finished", None))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        try:
            kind, _value = events.get(timeout=_INDEX_HEARTBEAT_INTERVAL_SECONDS)
        except queue.Empty:
            yield "heartbeat", None
            continue
        if kind == "finished":
            break
    thread.join()
    if "error" in holder:
        raise holder["error"]
    yield "result", holder["exit_code"]


def create_app(
    settings: Settings | None = None,
    *,
    runner_factory: Callable[[], RagSetupRunner] | None = None,
    judge_factory: Callable[[], RagasJudge] | None = None,
    corpus_fn: Callable[[], dict[str, Any]] | None = None,
    chunks_fn: Callable[[str], dict[str, Any]] | None = None,
    chunk_records_fn: Callable[[str | None], list[dict[str, Any]]] | None = None,
    graph_records_fn: Callable[[], list[dict[str, Any]]] | None = None,
    history_path: Path = DEFAULT_HISTORY_PATH,
    chat_html_path: Path = DEFAULT_CHAT_HTML_PATH,
    index_fn: IndexFn = _default_index_fn,
    frontend_dist: Path = FRONTEND_DIST,
) -> FastAPI:
    resolved = settings or load_settings(require_keys=True)
    runner_factory = runner_factory or (lambda: RagSetupRunner.from_settings(resolved))
    judge_factory = judge_factory or (lambda: RagasJudge.from_settings(resolved))
    corpus_fn = corpus_fn or (
        lambda: list_corpus(
            resolved.chroma_path,
            resolved.raw_transcript_collection,
            resolved.chunk_collection,
        )
    )
    chunks_fn = chunks_fn or (
        lambda video_id: list_chunks(
            resolved.chroma_path, video_id, resolved.chunk_collection
        )
    )
    chunk_records_fn = chunk_records_fn or (
        lambda video_id: load_chunk_corpus(
            resolved.chroma_path, resolved.chunk_collection, video_id
        )
    )
    graph_records_fn = graph_records_fn or (
        lambda: load_chunk_embeddings(resolved.chroma_path, resolved.chunk_collection)
    )
    judge_model_name = resolved.judge_model or resolved.deepseek_model

    app = FastAPI(title="Transcript RAG Evaluation Workbench", version="0.2.0")

    # Both stacks load models, so build each once, lazily, never at startup.
    locks = {"runner": threading.Lock(), "judge": threading.Lock()}
    holders: dict[str, Any] = {}

    def loaded(name: str) -> bool:
        return name in holders

    def get_runner() -> RagSetupRunner:
        with locks["runner"]:
            if "runner" not in holders:
                holders["runner"] = runner_factory()
            return holders["runner"]

    def get_judge() -> RagasJudge:
        with locks["judge"]:
            if "judge" not in holders:
                holders["judge"] = judge_factory()
            return holders["judge"]

    def bundle_index() -> Path:
        return frontend_dist / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> Any:
        # The React bundle wins when built; the legacy static page is the
        # fallback so `serve` keeps working without a Node toolchain present.
        if bundle_index().is_file():
            return FileResponse(bundle_index())
        return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))

    @app.get("/favicon.svg")
    def favicon() -> Any:
        # Vite emits this at the bundle root rather than under /assets, so it
        # needs its own route — the StaticFiles mount below would not reach it.
        icon = frontend_dist / "favicon.svg"
        if icon.is_file():
            return FileResponse(icon, media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="favicon not built")

    @app.get("/assets/render.js")
    def render_js() -> PlainTextResponse:
        return PlainTextResponse(ANSWER_RENDER_JS, media_type="text/javascript")

    @app.get("/assets/answer.css")
    def answer_css() -> PlainTextResponse:
        return PlainTextResponse(ANSWER_CSS, media_type="text/css")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "runner_loaded": loaded("runner"),
            "judge_loaded": loaded("judge"),
            "judge_model": judge_model_name,
            "answer_model": resolved.deepseek_model,
            "embedding_model": resolved.embedding_model,
            "ui": "react" if bundle_index().is_file() else "legacy",
        }

    @app.get("/api/setups")
    def setups() -> dict:
        return {"setups": [asdict(spec) for spec in SETUP_SPECS]}

    @app.get("/api/experiments")
    def experiments() -> dict:
        from src.api.experiments import load_experiments

        return load_experiments()

    @app.get("/api/history")
    def history() -> dict:
        return {
            "conversations": [entry.to_dict() for entry in load_history(history_path)]
        }

    @app.get("/api/corpus")
    def corpus() -> dict:
        return corpus_fn()

    @app.get("/api/corpus/{video_id}/chunks")
    def corpus_chunks(video_id: str) -> dict:
        return chunks_fn(video_id)

    @app.get("/api/scoreboard")
    def scoreboard(
        group_by: Literal["setup", "setup_model"] = "setup",
        judge_model: str | None = None,
    ) -> dict:
        board = build_scoreboard(
            load_history(history_path),
            group_by=group_by,
            judge_model=judge_model,
        )
        board["judge_model"] = judge_model_name
        return board

    @app.post("/api/rank")
    def rank(payload: RankRequest) -> dict:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="Query must not be blank")

        def semantic(text: str, top_k: int) -> list[dict[str, Any]]:
            provider = get_runner().provider
            if payload.video_id:
                chunks = provider.chunk_store.query_by_video_id(
                    payload.video_id, text, top_k
                )
            else:
                context = provider.get_context(
                    question=text,
                    source_url=None,
                    top_k=top_k,
                )
                chunks = context.retrieved_chunks or []
            return [
                {
                    "video_id": chunk.video_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "source_url": str(chunk.source_url) if chunk.source_url else None,
                    "score": chunk.score,
                }
                for chunk in chunks
            ]

        try:
            return build_rankings(
                query,
                modes=payload.modes,
                top_k=payload.top_k,
                semantic_fn=semantic,
                records_fn=lambda: chunk_records_fn(payload.video_id),
                video_id=payload.video_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
        if payload.entry_id:
            target_entry = next(
                (
                    entry
                    for entry in load_history(history_path)
                    if entry.id == payload.entry_id
                ),
                None,
            )
            if target_entry is None:
                raise HTTPException(
                    status_code=404, detail=f"Unknown entry: {payload.entry_id}"
                )
            if target_entry.question != question or (
                target_entry.url is not None and target_entry.url != url
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f"Question/url does not match entry {payload.entry_id}",
                )

        def stream() -> Iterator[str]:
            # One failing setup is already captured as SetupResult.error by the
            # runner; this guard is for everything else (stack build, storage),
            # which must surface as an event rather than a dead stream.
            try:
                if not loaded("runner"):
                    yield _sse(
                        "progress",
                        {"message": "Loading retrieval stack (first question only)..."},
                    )
                runner = get_runner()
                results = []
                scope = AskScope(
                    # A pinned video is already narrower than its channel, so
                    # sending both would silently widen the user's scope.
                    channel_id=None if url else payload.channel_id,
                    retrieval_mode=payload.retrieval_mode,
                    filter_transcripts=payload.filter_transcripts,
                    history=list(payload.history),
                )
                for key in keys:
                    yield _sse(
                        "progress",
                        {"key": key, "message": f"Running {setup_spec(key).title} ..."},
                    )
                    result = None
                    for kind, value in _run_setup_streaming(
                        runner, key, question, url, payload.top_k, scope
                    ):
                        if kind == "step":
                            yield _sse(
                                "agent_step",
                                {"key": key, **value.model_dump()},
                            )
                        else:
                            result = value
                    assert result is not None
                    results.append(result)
                    yield _sse("answer", asdict(ChatAnswer.from_result(result)))
                if payload.entry_id:
                    entry, entries = _append_answers(
                        payload.entry_id, results, history_path
                    )
                else:
                    entry = build_entry(question, results, url=url)
                    entries = append_entry(entry, history_path)
                write_chat_html(entries, chat_html_path)
                yield _sse("done", entry.to_dict())
            except Exception as exc:
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    @app.post("/api/judge")
    def judge(payload: JudgeRequest) -> StreamingResponse:
        entries = load_history(history_path)
        entry = next((e for e in entries if e.id == payload.entry_id), None)
        if entry is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown entry: {payload.entry_id}"
            )

        def stream() -> Iterator[str]:
            try:
                targets = [
                    answer
                    for answer in entry.answers
                    if payload.force or answer.evaluation is None
                ]
                scorable = [a for a in targets if not a.error and a.contexts]
                if scorable and not loaded("judge"):
                    yield _sse(
                        "progress",
                        {"message": "Loading RAGAS judge (first run only)..."},
                    )
                ragas_judge = get_judge() if scorable else None
                for answer in targets:
                    if answer.error:
                        answer.evaluation = unjudgeable(
                            "answer errored; not judged", judge_model_name
                        )
                    elif not answer.contexts:
                        answer.evaluation = unjudgeable(
                            "no stored retrieval contexts "
                            "(asked before context persistence)",
                            judge_model_name,
                        )
                    else:
                        yield _sse(
                            "progress",
                            {
                                "key": answer.key,
                                "message": f"Judging {answer.title} with RAGAS ...",
                            },
                        )
                        assert ragas_judge is not None
                        # Pass the model that wrote THIS answer, so self_graded
                        # reflects the actual pairing rather than the currently
                        # configured answering model.
                        answer.evaluation = ragas_judge.score(
                            entry.question,
                            answer.answer,
                            answer.contexts,
                            answer_model=answer.model,
                        )
                    yield _sse(
                        "scored",
                        {"key": answer.key, "evaluation": answer.evaluation},
                    )
                if targets:
                    evaluations = {answer.key: answer.evaluation for answer in targets}

                    def _apply_evaluations(fresh_entry: ChatEntry) -> None:
                        for fresh_answer in fresh_entry.answers:
                            if fresh_answer.key in evaluations:
                                fresh_answer.evaluation = evaluations[fresh_answer.key]

                    updated_entry, entries = update_entry(
                        payload.entry_id, _apply_evaluations, history_path
                    )
                    write_chat_html(entries, chat_html_path)
                    result_entry = updated_entry if updated_entry is not None else entry
                else:
                    result_entry = entry
                yield _sse("done", result_entry.to_dict())
            except Exception as exc:
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    @app.post("/api/chunk-graph")
    def chunk_graph(payload: GraphRequest) -> dict:
        """A kNN similarity graph over chunk embeddings, optionally highlighted.

        Structure is built from stored vectors alone, so drawing the graph never
        loads a model. Only the query overlay does, and only when asked for.
        """
        from src.rag.graph import build_chunk_graph_cached, nearest_chunks

        try:
            records = graph_records_fn()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read chunk embeddings: {exc}",
            ) from exc
        if not records:
            raise HTTPException(
                status_code=404,
                detail="No chunk embeddings found. Index a video first.",
            )
        try:
            graph = build_chunk_graph_cached(
                records, k=payload.k, min_similarity=payload.min_similarity
            )
        except ValueError as exc:
            # The corpus is too large to graph on demand, not a request/state
            # conflict, so 422 (oversized request given current data) fits
            # better than 409.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        query = (payload.query or "").strip()
        if query:
            embedding = get_runner().provider.chunk_store.embedding_model.embed_query(
                query
            )
            try:
                nearest = nearest_chunks(records, embedding, payload.top_k)
            except ValueError as exc:
                # Raised when the query vector's width differs from the stored
                # ones, which means the corpus was indexed with another
                # embedding model — a config problem, not a server fault.
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            graph = {**graph, "query": {"text": query, "nearest": nearest}}
        return graph

    @app.post("/api/index")
    def index_content(payload: IndexRequest) -> dict:
        argv, target = _index_argv(payload)
        exit_code = index_fn(argv)
        return {"ok": exit_code == 0, "exit_code": exit_code, "target": target}

    @app.post("/api/index/stream")
    def index_content_streaming(payload: IndexRequest) -> StreamingResponse:
        """Index with per-stage progress, and report what actually changed.

        The underlying CLI reports only at the end, so ``discover`` and
        ``fetch`` are announced before it starts, the whole blocking call runs
        under a single ``processing`` stage (with periodic heartbeats so the
        connection never looks idle), and completion is derived from corpus
        state either side of the run: a caller watching this stream sees
        exactly which videos and chunks were added.
        """
        argv, target = _index_argv(payload)

        def stream() -> Iterator[str]:
            try:
                before = corpus_fn()
                before_ids = {v["video_id"] for v in before.get("videos", [])}
                for stage, message in _INDEX_STAGES:
                    yield _sse("stage", {"stage": stage, "message": message})
                yield _sse(
                    "stage",
                    {
                        "stage": _INDEX_PROCESSING_STAGE,
                        "message": _INDEX_PROCESSING_MESSAGE,
                    },
                )
                exit_code = None
                for kind, value in _run_index_streaming(index_fn, argv):
                    if kind == "heartbeat":
                        yield _sse(
                            "stage",
                            {
                                "stage": _INDEX_PROCESSING_STAGE,
                                "message": _INDEX_HEARTBEAT_MESSAGE,
                            },
                        )
                    else:
                        exit_code = value
                assert exit_code is not None
                if exit_code != 0:
                    yield _sse(
                        "error",
                        {
                            "message": (
                                f"Indexing failed (exit {exit_code}). "
                                "Check the server log for the CLI output."
                            )
                        },
                    )
                    return
                after = corpus_fn()
                added = [
                    v
                    for v in after.get("videos", [])
                    if v["video_id"] not in before_ids
                ]
                yield _sse(
                    "done",
                    {
                        "ok": True,
                        "target": target,
                        "added_videos": added,
                        "added_video_count": len(added),
                        "added_chunk_count": (
                            after.get("totals", {}).get("chunks", 0)
                            - before.get("totals", {}).get("chunks", 0)
                        ),
                        "totals": after.get("totals", {}),
                        "insights": after.get("insights", []),
                        "channels": after.get("channels", []),
                    },
                )
            except Exception as exc:
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    # Mounted last so it can never shadow an /api route. Absent until the
    # frontend is built, which is why `/` falls back to the legacy page.
    if (frontend_dist / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=frontend_dist / "assets"),
            name="bundle-assets",
        )

    return app
