from __future__ import annotations

from src.agents.prompts import (
    SYSTEM_PROMPT,
    build_question_prompt,
    build_summary_prompt,
    build_transcript_context_prompt,
)


def test_context_prompt_includes_transcript() -> None:
    prompt = build_transcript_context_prompt("transcript text")

    assert "transcript text" in prompt


def test_summary_prompt_includes_output_contract_not_transcript() -> None:
    prompt = build_summary_prompt("Please summarize")

    assert "Please summarize" in prompt
    assert "top_findings" in prompt
    assert "Transcript:" not in prompt


def test_question_prompt_includes_question_transcript_and_video_id() -> None:
    prompt = build_question_prompt("What happened?", "3hk7nO_q0a8")

    assert "What happened?" in prompt
    assert "3hk7nO_q0a8" in prompt
    assert "Use only the transcript" not in prompt


def test_system_prompt_contains_grounding_rule() -> None:
    assert "Use only the transcript as evidence" in SYSTEM_PROMPT


def test_the_review_prompt_scopes_absence_claims_to_what_was_inspected() -> None:
    """A nav bar, a footer, or an unfetched linked page may hold the thing the
    review is about to call missing."""
    from src.agents.prompts import DOC_REVIEW_SYSTEM_PROMPT

    assert "Scope every absence to what you actually inspected" in DOC_REVIEW_SYSTEM_PROMPT
    assert "was not fetched" in DOC_REVIEW_SYSTEM_PROMPT


def test_the_review_user_prompt_does_not_tell_the_model_to_ignore_the_document() -> None:
    from src.agents.prompts import DOC_REVIEW_USER_PROMPT

    assert "only the retrieved" not in DOC_REVIEW_USER_PROMPT
    assert "[§N]" in DOC_REVIEW_USER_PROMPT
