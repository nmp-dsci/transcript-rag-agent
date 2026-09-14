"""The one tool a guide agent gets beyond its files: ``retrieve_chunks``.

An in-process MCP server (no subprocess, no network) wrapping the same hybrid
retriever the Chat tab answers with. Whatever it returns is citeable as-is —
video id, chunk index, timestamps, text — so the agent never has to guess an
id. Scope defaults to the guide's own videos; the whole corpus is one flag
away for a comment that asks for evidence the guide's set does not hold.

The retrieval callable is injected so tests can drive the tool with a fake
and the server can be built without loading the embedding stack.
"""

from __future__ import annotations

import json
from typing import Any, Callable

#: ``retrieve(question, video_ids | None, top_k) -> list[chunk]``; each chunk has
#: ``video_id``, ``chunk_index``, ``text`` and optional ``title`` /
#: ``start_seconds`` / ``end_seconds`` / ``score`` as attributes or keys.
RetrieveChunksFn = Callable[[str, list[str] | None, int], list[Any]]

TOOL_NAME = "retrieve_chunks"
SERVER_NAME = "corpus"
#: The fully qualified name the SDK exposes the tool under, for ``allowed_tools``.
QUALIFIED_TOOL = f"mcp__{SERVER_NAME}__{TOOL_NAME}"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "What you want evidence for, phrased as a question.",
        },
        "video_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Restrict to these videos (default: the guide's own set). "
                "Pass an empty list to search the whole corpus."
            ),
        },
        "top_k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
    },
    "required": ["question"],
}


def _get(chunk: Any, name: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def _stamp(seconds: Any) -> str:
    if seconds is None:
        return "--:--"
    total = int(float(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_chunks(chunks: list[Any]) -> str:
    """One block per chunk, headed by the id the agent must cite."""
    if not chunks:
        return "No chunks matched. Try different words, or widen video_ids to []."
    parts = []
    for chunk in chunks:
        video_id = _get(chunk, "video_id", "")
        index = _get(chunk, "chunk_index", 0)
        title = _get(chunk, "title", None) or video_id
        score = _get(chunk, "score", None)
        head = f"### {video_id}@{index} · {title} · {_stamp(_get(chunk, 'start_seconds'))}"
        if score is not None:
            head += f" · score {float(score):.3f}"
        parts.append(f"{head}\n{str(_get(chunk, 'text', '') or '').strip()}")
    return "\n\n".join(parts)


class RetrievalLog:
    """Every call the agent made, for the run log and the stage stream."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, question: str, video_ids: list[str] | None, results: list[Any]) -> None:
        self.calls.append(
            {
                "question": question,
                "video_ids": list(video_ids) if video_ids else None,
                "results": len(results),
                "videos": sorted({str(_get(c, "video_id", "")) for c in results}),
            }
        )


def build_retrieval_tool(
    retrieve: RetrieveChunksFn,
    *,
    default_video_ids: list[str] | None,
    log: RetrievalLog | None = None,
    on_call: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """The ``retrieve_chunks`` SDK tool (``.handler`` is what the SDK invokes)."""
    from claude_agent_sdk import tool

    @tool(
        TOOL_NAME,
        "Search the transcript corpus with hybrid (semantic + keyword) retrieval. "
        "Returns chunks with the exact `video_id@chunk_index` to cite.",
        INPUT_SCHEMA,
    )
    async def retrieve_chunks(args: dict[str, Any]) -> dict[str, Any]:
        question = str(args.get("question") or "").strip()
        if not question:
            return {"content": [{"type": "text", "text": "question is required"}], "is_error": True}
        raw_ids = args.get("video_ids")
        if raw_ids is None:
            video_ids: list[str] | None = list(default_video_ids) if default_video_ids else None
        elif isinstance(raw_ids, list) and len(raw_ids) == 0:
            video_ids = None
        else:
            video_ids = [str(v) for v in raw_ids]
        top_k = max(1, min(30, int(args.get("top_k") or 10)))
        try:
            results = list(retrieve(question, video_ids, top_k))
        except Exception as exc:  # noqa: BLE001 - the agent gets the reason, the run continues
            return {
                "content": [{"type": "text", "text": f"retrieval failed: {exc}"}],
                "is_error": True,
            }
        if log is not None:
            log.record(question, video_ids, results)
        if on_call is not None:
            on_call(
                {
                    "tool": TOOL_NAME,
                    "question": question,
                    "video_ids": video_ids,
                    "results": len(results),
                }
            )
        return {"content": [{"type": "text", "text": format_chunks(results)}]}

    return retrieve_chunks


def build_retrieval_server(
    retrieve: RetrieveChunksFn,
    *,
    default_video_ids: list[str] | None,
    log: RetrievalLog | None = None,
    on_call: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """An in-process SDK MCP server config exposing ``retrieve_chunks``."""
    from claude_agent_sdk import create_sdk_mcp_server

    tool_def = build_retrieval_tool(
        retrieve, default_video_ids=default_video_ids, log=log, on_call=on_call
    )
    return create_sdk_mcp_server(SERVER_NAME, "1.0.0", tools=[tool_def])


def describe_tool_call(name: str, arguments: dict[str, Any], cwd: str | None = None) -> str:
    """One line for the activity log: what the agent did, without the payload.

    Paths are shown relative to the guide directory: the agent sometimes
    reads by absolute path, and the log is read by a person.
    """
    if name in ("Read", "Write", "Edit", "Glob", "Grep"):
        target = str(
            arguments.get("file_path") or arguments.get("pattern") or arguments.get("path") or ""
        )
        if cwd and target.startswith(cwd.rstrip("/") + "/"):
            target = target[len(cwd.rstrip("/")) + 1 :]
        return f"{name} {target}".strip()
    if name.endswith(TOOL_NAME):
        return f"retrieve_chunks {json.dumps(arguments.get('question', ''))}"
    return name
