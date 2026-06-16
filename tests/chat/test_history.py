from __future__ import annotations

from datetime import datetime, timezone

from src.agents.models import RagAnswerReference
from src.chat.history import (
    ChatEntry,
    append_entry,
    build_entry,
    load_history,
    new_entry_id,
    save_history,
)
from src.chat.setups import SetupResult


def _reference() -> RagAnswerReference:
    return RagAnswerReference(
        label="[1] 0:30",
        source_url="https://www.youtube.com/watch?v=abc",
        timestamp_url="https://www.youtube.com/watch?v=abc&t=30s",
        start_seconds=30.0,
        end_seconds=45.0,
        chunk_index=2,
        video_id="abc",
    )


def _result(**overrides) -> SetupResult:
    base = dict(
        key="rag_llm",
        title="rag_llm (single-hop)",
        command="uv run python -m src.cli rag-ask ...",
        answer="An answer.",
        references=[_reference()],
        token_estimate=120,
        chunk_count=3,
        llm_calls=1,
        elapsed_seconds=1.2,
    )
    base.update(overrides)
    return SetupResult(**base)


def test_new_entry_id_format() -> None:
    moment = datetime(2026, 6, 16, 14, 30, 2, tzinfo=timezone.utc)
    entry_id = new_entry_id(moment)
    assert entry_id.startswith("q-20260616-143002-")
    assert len(entry_id.split("-")[-1]) == 4


def test_build_entry_serializes_references() -> None:
    entry = build_entry("What is X?", [_result()], url=None)

    assert entry.question == "What is X?"
    assert len(entry.answers) == 1
    reference = entry.answers[0].references[0]
    assert isinstance(reference, dict)
    assert reference["label"] == "[1] 0:30"
    assert reference["video_id"] == "abc"


def test_history_roundtrip(tmp_path) -> None:
    path = tmp_path / "chat_history.json"
    entry = build_entry("Question one", [_result()])

    append_entry(entry, path)
    append_entry(build_entry("Question two", [_result(key="rag_agent")]), path)

    loaded = load_history(path)
    assert [e.question for e in loaded] == ["Question one", "Question two"]
    assert isinstance(loaded[0], ChatEntry)
    assert loaded[0].answers[0].references[0]["video_id"] == "abc"


def test_load_missing_history_returns_empty(tmp_path) -> None:
    assert load_history(tmp_path / "absent.json") == []


def test_save_history_writes_conversations_key(tmp_path) -> None:
    path = tmp_path / "nested" / "chat_history.json"
    save_history([build_entry("Q", [_result()])], path)
    text = path.read_text(encoding="utf-8")
    assert '"conversations"' in text
