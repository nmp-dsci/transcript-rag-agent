from __future__ import annotations

import json
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


def test_loads_entries_written_before_model_identity_existed(tmp_path) -> None:
    """Histories from earlier builds must keep loading, with model set to None."""
    path = tmp_path / "chat_history.json"
    path.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "id": "q-legacy",
                        "question": "Old question",
                        "url": None,
                        "asked_at": "2026-01-01T00:00:00+00:00",
                        "answers": [
                            {
                                "key": "rag_llm",
                                "title": "rag_llm (single-hop)",
                                "command": "rag-ask",
                                "answer": "An old answer.",
                                "references": [],
                                "token_estimate": 10,
                                "chunk_count": 1,
                                "elapsed_seconds": 1.0,
                                "contexts": ["ctx"],
                                "evaluation": None,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    [entry] = load_history(path)
    answer = entry.answers[0]
    assert answer.answer == "An old answer."
    assert answer.model is None
    assert answer.embedding_model is None
    assert answer.top_k is None


def test_ignores_unknown_answer_fields_from_newer_builds(tmp_path) -> None:
    path = tmp_path / "chat_history.json"
    save_history([build_entry("Q", [_result()])], path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["conversations"][0]["answers"][0]["invented_field"] = "from the future"
    path.write_text(json.dumps(data), encoding="utf-8")

    [entry] = load_history(path)
    assert entry.answers[0].key == "rag_llm"


def test_round_trips_model_identity(tmp_path) -> None:
    path = tmp_path / "chat_history.json"
    result = _result()
    result.model = "deepseek-v4"
    result.embedding_model = "all-MiniLM-L6-v2"
    result.top_k = 20
    save_history([build_entry("Q", [result])], path)

    answer = load_history(path)[0].answers[0]
    assert answer.model == "deepseek-v4"
    assert answer.embedding_model == "all-MiniLM-L6-v2"
    assert answer.top_k == 20


def test_followups_are_carried_from_result_to_history():
    """The LLM proposes follow-ups on every answer; the UI offers them as chips."""
    from src.chat.history import ChatAnswer
    from src.chat.setups import SetupResult

    result = SetupResult(
        key="rag_llm",
        title="t",
        command="c",
        answer="a",
        followups=[
            {
                "topic": "CGT discount",
                "rationale": "mentioned but not detailed",
                "followup_query": "how does the CGT discount change?",
                "confidence": 0.8,
            }
        ],
    )
    answer = ChatAnswer.from_result(result)
    assert answer.followups[0]["followup_query"] == "how does the CGT discount change?"


def test_followups_default_to_empty_for_older_answers():
    from src.chat.history import ChatAnswer
    from src.chat.setups import SetupResult

    answer = ChatAnswer.from_result(SetupResult(key="rag_llm", title="t", command="c", answer="a"))
    assert answer.followups == []


def test_trace_is_carried_from_result_and_round_trips(tmp_path):
    """The persisted trace is what lets a reload still show how an answer ran."""
    from src.chat.history import ChatAnswer, ChatEntry, load_history, save_history
    from src.chat.setups import SetupResult

    step = {
        "phase": "retrieve",
        "label": "Retrieve candidates",
        "detail": "semantic search — 10 candidates",
        "chunk_ids": ["chunk:vid00000001:0"],
        "model": None,
        "elapsed_ms": 12,
        "iteration": None,
    }
    answer = ChatAnswer.from_result(
        SetupResult(key="rag_llm", title="t", command="c", answer="a", trace=[step])
    )
    entry = ChatEntry(id="q-1", question="q?", url=None, asked_at="now", answers=[answer])

    path = tmp_path / "history.json"
    save_history([entry], path)
    loaded = load_history(path)

    assert loaded[0].answers[0].trace == [step]


def test_trace_defaults_to_empty_for_older_answers():
    from src.chat.history import ChatAnswer
    from src.chat.setups import SetupResult

    answer = ChatAnswer.from_result(SetupResult(key="rag_llm", title="t", command="c", answer="a"))
    assert answer.trace == []


def test_a_document_review_records_how_much_of_it_was_read(tmp_path):
    """The card must say "all 9 sections" after a reload, not guess it."""
    from src.chat.history import build_entry, load_history, save_history

    entry = build_entry(
        "review https://example.com/cv",
        [],
        document_id="doc:abc",
        document_detail="fetched — whole document — all 3 sections in context",
        document_sections_selected=[0, 1, 2],
    )

    path = tmp_path / "history.json"
    save_history([entry], path)
    loaded = load_history(path)[0]

    assert loaded.document_id == "doc:abc"
    assert loaded.document_detail == "fetched — whole document — all 3 sections in context"
    assert loaded.document_sections_selected == [0, 1, 2]


def test_the_selection_record_carries_no_document_text(tmp_path):
    """This file is committed; only facts *about* the document belong in it."""
    import json

    from src.chat.history import build_entry, save_history

    entry = build_entry(
        "review https://example.com/cv",
        [],
        document_id="doc:abc",
        document_detail="fetched — whole document — all 3 sections in context",
        document_sections_selected=[0, 1, 2],
    )

    path = tmp_path / "history.json"
    save_history([entry], path)

    written = json.loads(path.read_text(encoding="utf-8"))["conversations"][0]
    assert set(written) == {
        "id",
        "question",
        "url",
        "asked_at",
        "answers",
        "document_id",
        "document_detail",
        "document_sections_selected",
    }


def test_an_ordinary_question_records_no_selection(tmp_path):
    from src.chat.history import build_entry

    entry = build_entry("what changed?", [])

    assert entry.document_detail is None
    assert entry.document_sections_selected is None


def test_entries_written_before_the_selection_existed_still_load(tmp_path):
    """Back-compat: the committed history predates these fields."""
    import json

    from src.chat.history import load_history

    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "id": "q-1",
                        "question": "q?",
                        "url": None,
                        "asked_at": "now",
                        "answers": [],
                        "document_id": "doc:old",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_history(path)[0]

    assert loaded.document_id == "doc:old"
    assert loaded.document_detail is None
    assert loaded.document_sections_selected is None
