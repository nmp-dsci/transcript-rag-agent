"""Persisted chat history for the interactive transcript RAG session.

One JSON file holds an ordered list of conversation entries. Each entry is a
question and the per-setup answers it produced. The frontend renders directly
from these records.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.chat.setups import SetupResult

DEFAULT_HISTORY_PATH = Path("dashboard/chat_history.json")

# Guards read-modify-write access to the history file so concurrent requests
# (e.g. /api/ask and /api/judge, or two browser tabs) cannot interleave and
# silently drop each other's writes.
_LOCK = threading.Lock()


@dataclass
class ChatAnswer:
    """One setup's answer within a conversation entry."""

    key: str
    title: str
    command: str
    answer: str
    references: list[dict[str, Any]] = field(default_factory=list)
    token_estimate: int = 0
    chunk_count: int = 0
    llm_calls: int | None = None
    iterations: int | None = None
    terminated_reason: str | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    # Retrieved chunk texts (judge input) and the RAGAS evaluation record.
    contexts: list[str] = field(default_factory=list)
    # What retrieval returned, in order — the input to recall metrics.
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    # Stack identity, recorded so the scoreboard can group by model instead of
    # averaging across them. Entries written before this existed keep ``None``.
    model: str | None = None
    embedding_model: str | None = None
    top_k: int | None = None
    # Retrieval scope and strategy for this answer. ``None`` on entries written
    # before scoping existed, which the scoreboard reports as pre-provenance.
    channel_id: str | None = None
    retrieval_mode: str | None = None
    # Follow-up questions the LLM proposed, offered to the user as next asks.
    followups: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: SetupResult) -> "ChatAnswer":
        return cls(
            key=result.key,
            title=result.title,
            command=result.command,
            answer=result.answer,
            references=[_reference_to_dict(ref) for ref in result.references],
            token_estimate=result.token_estimate,
            chunk_count=result.chunk_count,
            llm_calls=result.llm_calls,
            iterations=result.iterations,
            terminated_reason=result.terminated_reason,
            elapsed_seconds=result.elapsed_seconds,
            error=result.error,
            contexts=list(result.contexts),
            retrieved_chunk_ids=list(result.retrieved_chunk_ids),
            model=result.model,
            embedding_model=result.embedding_model,
            top_k=result.top_k,
            channel_id=result.channel_id,
            retrieval_mode=result.retrieval_mode,
            followups=list(result.followups),
        )


@dataclass
class ChatEntry:
    """A single asked question and all setup answers captured for it."""

    id: str
    question: str
    url: str | None
    asked_at: str
    answers: list[ChatAnswer] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatEntry":
        return cls(
            id=data["id"],
            question=data["question"],
            url=data.get("url"),
            asked_at=data["asked_at"],
            answers=[
                ChatAnswer(**_known_answer_fields(answer))
                for answer in data.get("answers", [])
            ],
        )


def _known_answer_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys so histories written by a newer build still load."""
    allowed = {f.name for f in fields(ChatAnswer)}
    return {key: value for key, value in data.items() if key in allowed}


def _reference_to_dict(reference: Any) -> dict[str, Any]:
    if hasattr(reference, "model_dump"):
        return reference.model_dump(mode="json")
    if isinstance(reference, dict):
        return reference
    return {"label": str(reference)}


def new_entry_id(now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return f"q-{moment.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def build_entry(
    question: str,
    results: list[SetupResult],
    *,
    url: str | None = None,
    now: datetime | None = None,
) -> ChatEntry:
    moment = now or datetime.now(timezone.utc)
    return ChatEntry(
        id=new_entry_id(moment),
        question=question,
        url=url,
        asked_at=moment.isoformat(),
        answers=[ChatAnswer.from_result(result) for result in results],
    )


def load_history(path: Path = DEFAULT_HISTORY_PATH) -> list[ChatEntry]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    conversations = data.get("conversations", []) if isinstance(data, dict) else data
    return [ChatEntry.from_dict(entry) for entry in conversations]


def save_history(entries: list[ChatEntry], path: Path = DEFAULT_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"conversations": [entry.to_dict() for entry in entries]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_entry(
    entry: ChatEntry, path: Path = DEFAULT_HISTORY_PATH
) -> list[ChatEntry]:
    with _LOCK:
        entries = load_history(path)
        entries.append(entry)
        save_history(entries, path)
        return entries


def update_entry(
    entry_id: str,
    mutate: Callable[[ChatEntry], None],
    path: Path = DEFAULT_HISTORY_PATH,
) -> tuple[ChatEntry | None, list[ChatEntry]]:
    """Reload the current history, mutate the target entry in place, and save.

    Reloading from disk immediately before saving (rather than reusing an
    in-memory snapshot taken before a long-running operation) avoids
    clobbering entries appended by other requests in the meantime.
    """
    with _LOCK:
        entries = load_history(path)
        entry = next((e for e in entries if e.id == entry_id), None)
        if entry is None:
            return None, entries
        mutate(entry)
        save_history(entries, path)
        return entry, entries
