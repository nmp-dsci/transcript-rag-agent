"""Does a YouTube description route as well as an LLM summary?

The summary filter picks *which videos* a question is allowed to search, so
the summary's only job is to be findable by the right question. That makes the
comparison deterministic and judge-free: embed the question, score it against
every video's summary embedding, and ask whether the golden set's expected
video comes back.

Both variants are embedded in memory from text already on disk — the stored
LLM summary and the stored description — so the store is never mutated and the
two arms differ in exactly one thing: the summary text. No LLM is called, which
matters, because the LLM this compares against is the one that ran out of
balance.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evals.golden import load_golden
from src.rag.embeddings import cosine_similarity
from src.rag.summaries import DescriptionSummaryGenerator


@dataclass(frozen=True)
class RoutingScore:
    """How well one summary source routes the golden set."""

    label: str
    videos: int
    questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_rank: float | None
    unroutable: int

    def as_row(self) -> dict[str, object]:
        return {
            "label": self.label,
            "videos": self.videos,
            "questions": self.questions,
            "recall@1": round(self.recall_at_1, 3),
            "recall@3": round(self.recall_at_3, 3),
            "recall@5": round(self.recall_at_5, 3),
            "mean_rank": round(self.mean_rank, 2) if self.mean_rank is not None else None,
            "unroutable_videos": self.unroutable,
        }


def _rank_of_expected(
    question_embedding: list[float],
    summaries: dict[str, list[float]],
    expected: set[str],
) -> int | None:
    """1-based rank of the best-scoring expected video, or None if absent."""
    ranked = sorted(
        summaries.items(),
        key=lambda item: cosine_similarity(question_embedding, item[1]),
        reverse=True,
    )
    for position, (video_id, _embedding) in enumerate(ranked, start=1):
        if video_id in expected:
            return position
    return None


def score_source(
    label: str,
    summaries: dict[str, str],
    embedding_model,
    entries=None,
    unroutable: int = 0,
) -> RoutingScore:
    """Embed every summary once, then rank the golden questions against them."""
    entries = list(entries if entries is not None else load_golden())
    video_ids = sorted(summaries)
    vectors = embedding_model.embed_documents([summaries[v] for v in video_ids])
    embedded = dict(zip(video_ids, vectors))

    ranks: list[int | None] = []
    for entry in entries:
        expected = {v for v in (entry.expected_video_ids or []) if v in embedded}
        if not expected:
            # The golden answer's video is not in this arm's corpus, so the
            # question cannot discriminate between the arms. Skipping it keeps
            # both arms scored on exactly the same questions.
            continue
        ranks.append(
            _rank_of_expected(embedding_model.embed_query(entry.question), embedded, expected)
        )

    scored = [r for r in ranks if r is not None]
    total = len(ranks) or 1
    return RoutingScore(
        label=label,
        videos=len(summaries),
        questions=len(ranks),
        recall_at_1=sum(1 for r in scored if r <= 1) / total,
        recall_at_3=sum(1 for r in scored if r <= 3) / total,
        recall_at_5=sum(1 for r in scored if r <= 5) / total,
        mean_rank=(sum(scored) / len(scored)) if scored else None,
        unroutable=unroutable,
    )


def build_arms(documents) -> tuple[dict[str, str], dict[str, str], int]:
    """The two comparable arms, over videos that have *both* texts.

    Restricted to the overlap on purpose: an arm scored over more videos is
    solving an easier or harder ranking problem, and the comparison would stop
    being about the summary text.
    """
    generator = DescriptionSummaryGenerator()
    llm: dict[str, str] = {}
    description: dict[str, str] = {}
    unroutable = 0
    for document in documents:
        stored = (document.summary or "").strip()
        if not stored or document.summary_source == "description":
            continue
        try:
            description[document.video_id] = generator.summarize(document)
        except ValueError:
            # Too thin to route on — counted, not silently dropped.
            unroutable += 1
            continue
        llm[document.video_id] = stored
    shared = set(llm) & set(description)
    return (
        {v: llm[v] for v in shared},
        {v: description[v] for v in shared},
        unroutable,
    )
