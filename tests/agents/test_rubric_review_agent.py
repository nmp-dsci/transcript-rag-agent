"""The reviewer that applies every shipped pack to one document, one call each.

The behaviours under test are the ones a partial failure would otherwise hide: a
pack whose call fails leaves its rubrics unjudged and says why, and the review
still finishes with the other packs' verdicts in it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agents.rubric_review_agent import RubricReviewAgent
from src.documents.models import Document, DocumentSection
from src.documents.review import select_sections


def _evidence(video_id: str = "vid1", start: float = 60.0):
    return SimpleNamespace(
        video_id=video_id,
        chunk_id=f"chunk:{video_id}:2",
        quote="put the number in",
        model_quote="put the number in",
        start_seconds=start,
        quote_start_seconds=start + 3,
        channel_name="A Recruiter",
        title="How to write it",
        ratio=1.0,
        youtube_url=lambda: f"https://www.youtube.com/watch?v={video_id}&t={int(start + 1)}s",
    )


def _rubric(rubric_id: str):
    return SimpleNamespace(
        rubric_id=rubric_id,
        criterion=f"Criterion {rubric_id}.",
        check="Check it.",
        why="Because.",
        contested=False,
        unit_id="u",
        unit_kind="raptor",
        unit_title="Theme",
        creators=["A Recruiter"],
        evidence=[_evidence()],
    )


def _pack(topic: str, name: str, *ids: str):
    return SimpleNamespace(
        topic=topic,
        name=name,
        artifact="resume",
        rubrics=[_rubric(rubric_id) for rubric_id in ids],
    )


def _document() -> Document:
    return Document(
        id="doc:1",
        url="https://example.com",
        requested_url="https://example.com",
        title="Portfolio",
        sections=[
            DocumentSection(index=index, heading=f"Section {index}", text="some body text")
            for index in range(3)
        ],
    )


class FakeLlm:
    """Replies per call, or raises when the scripted reply is an exception."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(str(messages[-1].content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=json.dumps(reply))


def _agent(packs, replies) -> RubricReviewAgent:
    return RubricReviewAgent(FakeLlm(replies), packs, model_name="fake-model")


def test_one_call_per_pack_with_only_that_pack_s_rubrics() -> None:
    packs = [
        _pack("resume-design", "Resume design", "r0101"),
        _pack("job-search", "Job search", "r0201"),
    ]
    agent = _agent(
        packs,
        [
            {"verdicts": [{"rubric_id": "r0101", "verdict": "pass"}]},
            {"verdicts": [{"rubric_id": "r0201", "verdict": "n-a"}]},
        ],
    )
    document = _document()

    result = agent.review(document, select_sections(document, "review this"))

    assert result.llm_calls == 2
    # Asserted on the rubric listing, not the whole prompt: the JSON shape in
    # the instructions carries an illustrative id, and matching it anywhere in
    # the text would pass on the example rather than on the pack.
    first = agent.llm.prompts[0].split("RUBRICS")[1].split("Judge the document")[0]
    second = agent.llm.prompts[1].split("RUBRICS")[1].split("Judge the document")[0]
    assert "r0101" in first and "r0201" not in first
    assert "r0201" in second and "r0101" not in second
    assert [v.verdict for v in result.review.verdicts] == ["pass", "n-a"]


def test_a_failing_pack_leaves_its_rubrics_unjudged_and_the_review_finishes() -> None:
    packs = [
        _pack("resume-design", "Resume design", "r0101"),
        _pack("job-search", "Job search", "r0201", "r0202"),
    ]
    agent = _agent(
        packs,
        [
            RuntimeError("read timed out"),
            {
                "verdicts": [
                    {"rubric_id": "r0201", "verdict": "pass"},
                    {"rubric_id": "r0202", "verdict": "n-a"},
                ]
            },
        ],
    )
    document = _document()

    result = agent.review(document, select_sections(document, "review this"))

    assert result.review.stats["packs_failed"] == 1
    assert result.review.packs[0].error is not None
    assert "read timed out" in result.review.packs[0].error
    assert [v.verdict for v in result.review.verdicts] == ["unjudged", "pass", "n-a"]
    # The failed pack is a trace row, not a silence: three rubrics were asked
    # about and one pack's worth of them was never decided.
    assert any("failed" in step.label for step in result.trace)


def test_an_unparseable_reply_is_an_error_not_a_silent_skip() -> None:
    """`{}` here would render as "the reviewer considered them and declined"."""
    packs = [_pack("resume-design", "Resume design", "r0101")]
    agent = RubricReviewAgent(
        SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="sorry, I cannot")),
        packs,
    )
    document = _document()

    result = agent.review(document, select_sections(document, "review this"))

    assert result.review.packs[0].error is not None
    assert result.review.verdicts[0].verdict == "unjudged"


def test_the_document_is_in_the_prompt_with_its_section_markers() -> None:
    packs = [_pack("resume-design", "Resume design", "r0101")]
    agent = _agent(packs, [{"verdicts": [{"rubric_id": "r0101", "verdict": "pass"}]}])
    document = _document()

    agent.review(document, select_sections(document, "review this"))

    prompt = agent.llm.prompts[0]
    assert "[§1]" in prompt and "[§3]" in prompt
    assert "RUBRICS (1)" in prompt


def test_no_packs_built_is_a_clear_refusal() -> None:
    agent = _agent([], [])
    document = _document()

    with pytest.raises(ValueError, match="No rubric packs"):
        agent.review(document, select_sections(document, "review this"))


def test_progress_is_reported_per_pack() -> None:
    packs = [
        _pack("resume-design", "Resume design", "r0101"),
        _pack("job-search", "Job search", "r0201"),
    ]
    agent = _agent(
        packs,
        [
            {"verdicts": [{"rubric_id": "r0101", "verdict": "pass"}]},
            {"verdicts": [{"rubric_id": "r0201", "verdict": "pass"}]},
        ],
    )
    document = _document()
    seen: list[str] = []

    agent.review(document, select_sections(document, "review this"), on_progress=seen.append)

    assert len(seen) == 2
    assert "Resume design" in seen[0]
