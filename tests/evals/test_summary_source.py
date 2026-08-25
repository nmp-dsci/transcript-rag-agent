"""The summary-source ablation ranks videos, so it needs no judge.

Deliberately deterministic: the LLM this compares against is the one that ran
out of balance, so a comparison that needed it would be unrunnable exactly when
it matters most.
"""

from __future__ import annotations

from src.evals.golden import GoldenEntry
from src.evals.summary_source import build_arms, score_source
from src.rag.models import RawTranscriptDocument


class KeywordEmbedding:
    """Scores on shared vocabulary — enough to rank, with no model to load."""

    VOCAB = ("kubernetes", "tax", "property", "neural", "network")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if word in lowered else 0.0 for word in self.VOCAB]


def _doc(video_id: str, summary: str | None, description: str | None, title: str = "T"):
    return RawTranscriptDocument(
        transcript_id=f"t:{video_id}",
        video_id=video_id,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        fetched_at="2026-08-25T00:00:00Z",
        title=title,
        summary=summary,
        summary_source="llm" if summary else None,
        description=description,
    )


def _entry(question: str, video_id: str) -> GoldenEntry:
    return GoldenEntry(
        id=f"q:{video_id}",
        question=question,
        reference_answer="A reference answer.",
        expected_video_ids=[video_id],
        expected_chunk_ids=[f"chunk:{video_id}:0"],
        domain="corpus",
    )


LONG_KUBE = (
    "Running GenAI workloads on kubernetes with production patterns, covering "
    "operators, autoscaling and rollout strategy in detail across the talk."
)
LONG_TAX = (
    "The budget changed the tax position for property investors this year, and "
    "this session walks through what actually applies to a property portfolio."
)


class TestBuildArms:
    def test_only_videos_with_both_texts_are_comparable(self) -> None:
        docs = [
            _doc("a", "An LLM summary about kubernetes.", LONG_KUBE),
            _doc("b", "An LLM summary about tax.", None),           # no description
            _doc("c", None, LONG_TAX),                                # no LLM summary
        ]
        llm, description, _ = build_arms(docs)
        assert set(llm) == set(description) == {"a"}

    def test_thin_descriptions_are_counted_not_dropped_silently(self) -> None:
        docs = [
            _doc("a", "An LLM summary about kubernetes.", LONG_KUBE),
            _doc("b", "An LLM summary about tax.", "Work with me: ", title="X"),
        ]
        llm, _description, unroutable = build_arms(docs)
        assert unroutable == 1
        assert "b" not in llm

    def test_videos_already_on_the_description_source_are_skipped(self) -> None:
        """Nothing to compare — that arm is already the description."""
        already = _doc("a", "Some description text.", LONG_KUBE)
        already = already.model_copy(update={"summary_source": "description"})
        llm, description, _ = build_arms([already])
        assert llm == {} and description == {}


class TestScoreSource:
    def test_a_perfectly_routing_source_scores_recall_1(self) -> None:
        score = score_source(
            "perfect",
            {"kube": "kubernetes", "tax": "tax property"},
            KeywordEmbedding(),
            entries=[_entry("kubernetes question", "kube")],
        )
        assert score.recall_at_1 == 1.0
        assert score.mean_rank == 1.0
        assert score.questions == 1

    def test_questions_whose_video_is_absent_are_skipped_not_failed(self) -> None:
        """Both arms must be scored on exactly the same questions."""
        score = score_source(
            "partial",
            {"kube": "kubernetes"},
            KeywordEmbedding(),
            entries=[_entry("kubernetes question", "kube"), _entry("tax question", "missing")],
        )
        assert score.questions == 1

    def test_a_summary_that_misses_the_topic_ranks_below_one_that_hits_it(self) -> None:
        """The failure mode that matters: another video answers first.

        Every video is always ranked somewhere, so a bad summary does not
        vanish — it loses. recall@1 is what notices.
        """
        score = score_source(
            "useless",
            # The expected video's summary says nothing about the question;
            # the distractor's says exactly the right word.
            {"target": "unrelated words entirely", "distractor": "neural network"},
            KeywordEmbedding(),
            entries=[_entry("neural network question", "target")],
        )
        assert score.recall_at_1 == 0.0
        assert score.mean_rank == 2.0
