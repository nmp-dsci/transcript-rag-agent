"""Reference-based evaluation against a curated golden dataset.

The RAGAS metrics in :mod:`src.evals.judge` are all reference-free: faithfulness
asks whether the answer follows from the chunks that were retrieved, answer
relevancy whether it addresses the question, and context precision whether the
retrieved chunks were useful. None of them can see what retrieval *missed*, and
none of them know what a correct answer looks like. A retriever that returns two
good chunks out of eight relevant ones scores well on all three.

This module adds the missing half by comparing against a hand-written reference
set (``golden_dataset.json``):

* :func:`context_recall` and :func:`video_recall` — deterministic, id-based
  coverage of the chunks a good retriever must surface. No LLM, no cost.
* :func:`answer_correctness_fns` — RAGAS' reference-based metrics, which do need
  an LLM and are built exactly like :meth:`src.evals.judge.RagasJudge.from_settings`
  builds its own.

:func:`evaluate_entry` combines both, taking injected score functions so tests
can run the whole path with fakes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from src.config import Settings

DEFAULT_DATASET_PATH = Path(__file__).with_name("golden_dataset.json")

#: Chunk identity as produced by :attr:`src.rag.models.TranscriptChunk.chunk_id`.
CHUNK_ID_PATTERN = re.compile(r"^chunk:(?P<video_id>[^:]+):(?P<chunk_index>\d+)$")

DOMAINS = ("property", "ai-coding")

#: Keys always present in an :func:`evaluate_entry` result.
METRIC_NAMES = [
    "context_recall",
    "video_recall",
    "answer_correctness",
    "answer_similarity",
    "llm_context_recall",
]

# (question, answer, reference_answer, contexts) -> score in [0, 1]. The extra
# reference argument is what separates these from judge.ScoreFn.
ReferenceScoreFn = Callable[[str, str, str, list[str]], float]


class GoldenDatasetError(ValueError):
    """Raised when a golden dataset file is missing, malformed, or invalid."""


class GoldenEntry(BaseModel):
    """One curated question with its grounded reference answer.

    ``expected_chunk_ids`` are the chunks a good retriever *must* surface for the
    reference answer to be reachable. They are chunking-dependent: re-indexing
    with different chunk sizes renumbers chunks and invalidates them.
    """

    id: str
    question: str
    reference_answer: str
    expected_video_ids: list[str]
    expected_chunk_ids: list[str]
    domain: str
    notes: str = ""

    @field_validator("id", "question", "reference_answer", "domain")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("domain")
    @classmethod
    def _known_domain(cls, value: str) -> str:
        if value not in DOMAINS:
            raise ValueError(f"must be one of {DOMAINS}, got {value!r}")
        return value

    @field_validator("expected_video_ids", "expected_chunk_ids")
    @classmethod
    def _non_empty_list(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must list at least one id")
        if len(set(value)) != len(value):
            raise ValueError("must not contain duplicates")
        return value

    @field_validator("expected_chunk_ids")
    @classmethod
    def _chunk_id_shape(cls, value: list[str]) -> list[str]:
        for chunk_id in value:
            if not CHUNK_ID_PATTERN.match(chunk_id):
                raise ValueError(
                    f"{chunk_id!r} is not a chunk id of the form chunk:<video_id>:<index>"
                )
        return value

    @model_validator(mode="after")
    def _videos_consistent(self) -> "GoldenEntry":
        """Every expected chunk must belong to a declared video, and vice versa.

        Catches the two ways this file drifts by hand: adding a chunk from a
        video nobody listed, and listing a video no chunk backs up.
        """
        from_chunks = {chunk_video_id(chunk_id) for chunk_id in self.expected_chunk_ids}
        declared = set(self.expected_video_ids)
        if from_chunks - declared:
            raise ValueError(
                "expected_chunk_ids reference videos missing from expected_video_ids: "
                f"{sorted(from_chunks - declared)}"
            )
        if declared - from_chunks:
            raise ValueError(
                "expected_video_ids lists videos with no expected chunk: "
                f"{sorted(declared - from_chunks)}"
            )
        return self


def chunk_video_id(chunk_id: str) -> str:
    """The video id embedded in a chunk id, or ``""`` if it is not a chunk id."""
    match = CHUNK_ID_PATTERN.match(chunk_id)
    return match.group("video_id") if match else ""


def load_golden(path: str | Path | None = None) -> list[GoldenEntry]:
    """Load and validate the golden dataset.

    Accepts either a bare JSON list of entries or an object with an ``entries``
    key (the shipped file uses the latter so it can carry corpus provenance
    alongside the entries).

    Raises :class:`GoldenDatasetError` with the offending entry's position and
    id on anything malformed — a silently half-loaded eval set is worse than no
    eval set.
    """
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenDatasetError(f"golden dataset not found: {dataset_path}") from exc
    except json.JSONDecodeError as exc:
        raise GoldenDatasetError(f"golden dataset {dataset_path} is not valid JSON: {exc}") from exc

    if isinstance(raw, dict):
        records = raw.get("entries")
        if records is None:
            raise GoldenDatasetError(f"golden dataset {dataset_path} has no 'entries' key")
    else:
        records = raw
    if not isinstance(records, list):
        raise GoldenDatasetError(f"golden dataset {dataset_path} entries must be a list")

    entries: list[GoldenEntry] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise GoldenDatasetError(
                f"golden dataset {dataset_path} entry {index} must be an object"
            )
        try:
            entries.append(GoldenEntry.model_validate(record))
        except ValidationError as exc:
            entry_id = record.get("id", "<no id>")
            raise GoldenDatasetError(
                f"golden dataset {dataset_path} entry {index} ({entry_id!r}) is invalid: {exc}"
            ) from exc

    ids = [entry.id for entry in entries]
    duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
    if duplicates:
        raise GoldenDatasetError(
            f"golden dataset {dataset_path} has duplicate entry ids: {duplicates}"
        )
    return entries


def _recall(retrieved: Iterable[str], expected: Iterable[str]) -> float:
    """Fraction of ``expected`` ids present in ``retrieved``.

    Both sides are deduplicated, so retrieving the same id twice never inflates
    the score and listing it twice in the reference never deflates it.

    Convention: recall of nothing is ``1.0``. With no expected ids there is
    nothing to miss, and scoring 0.0 would punish a retriever for a reference
    entry that asked for nothing. :class:`GoldenEntry` forbids empty expectations
    anyway, so this only applies to ad-hoc calls.
    """
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    return len(expected_set & set(retrieved)) / len(expected_set)


def context_recall(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    """Fraction of the expected chunks that retrieval actually surfaced.

    This is the non-LLM, id-based variant: deterministic, free, and reproducible.
    It is *not* RAGAS' ``context_recall``, which is LLM-judged — it asks a model
    whether each claim in a reference answer is attributable to the retrieved
    contexts, and so tolerates a different chunk carrying the same fact. This one
    scores identity, so re-chunking the corpus changes it even when retrieval
    quality is unchanged. Use both: this for a cheap regression signal on every
    run, RAGAS' for a semantic verdict. :func:`answer_correctness_fns` supplies
    the latter under the name ``llm_context_recall``.
    """
    return _recall(retrieved_chunk_ids, expected_chunk_ids)


def video_recall(retrieved_video_ids: list[str], expected_video_ids: list[str]) -> float:
    """Fraction of the expected videos that retrieval actually surfaced.

    The same measure one level up. Chunk boundaries move whenever chunk size or
    overlap changes, and neighbouring chunks often carry the same point, so
    chunk-level recall drops for reasons that have nothing to do with retrieval.
    Video-level recall is stable across re-chunking and answers the coarser
    question: did we even find the right source?
    """
    return _recall(retrieved_video_ids, expected_video_ids)


def answer_correctness_fns(settings: Settings) -> dict[str, ReferenceScoreFn]:
    """Build RAGAS' reference-based metrics as injectable score functions.

    Mirrors :meth:`src.evals.judge.RagasJudge.from_settings` exactly — same
    compat shim, same ``LangchainLLMWrapper(ChatOpenAI(...))`` at temperature 0,
    same ``HuggingFaceEmbeddings`` wrapper — so a correctness score is produced
    by the same judge stack as a faithfulness score and the two are comparable.

    Returns ``answer_correctness`` (factual overlap with the reference answer,
    weighted with semantic similarity), ``answer_similarity`` (embedding
    similarity to the reference alone), and ``llm_context_recall`` (RAGAS'
    LLM-judged context recall, which asks whether the reference answer's claims
    are attributable to the retrieved contexts).

    ragas and its model stack load slowly, so everything is imported lazily
    inside this function and never at module import.
    """
    from src.evals import _ragas_compat

    _ragas_compat.install()

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, AnswerSimilarity, LLMContextRecall

    model = settings.judge_model or settings.deepseek_model
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            api_key=settings.judge_api_key or settings.deepseek_api_key,
            base_url=settings.judge_base_url or settings.deepseek_base_url,
            temperature=0.0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    similarity = AnswerSimilarity(embeddings=embeddings)
    correctness = AnswerCorrectness(llm=llm, embeddings=embeddings, answer_similarity=similarity)
    recall = LLMContextRecall(llm=llm)

    def sample(question: str, answer: str, reference: str, contexts: list[str]) -> Any:
        return SingleTurnSample(
            user_input=question,
            response=answer,
            reference=reference,
            retrieved_contexts=list(contexts),
        )

    return {
        "answer_correctness": lambda q, a, r, c: float(
            correctness.single_turn_score(sample(q, a, r, c))
        ),
        "answer_similarity": lambda q, a, r, c: float(
            similarity.single_turn_score(sample(q, a, r, c))
        ),
        "llm_context_recall": lambda q, a, r, c: float(
            recall.single_turn_score(sample(q, a, r, c))
        ),
    }


def evaluate_entry(
    entry: GoldenEntry,
    answer: str,
    retrieved_chunk_ids: list[str],
    score_fns: dict[str, ReferenceScoreFn] | None = None,
    contexts: list[str] | None = None,
) -> dict[str, float | None]:
    """Score one answer against one golden entry.

    The two recall metrics are always computed; they are pure arithmetic over
    ids and cost nothing. Retrieved video ids are derived from the chunk ids, so
    callers only have to report what retrieval returned. Ids that are not valid
    chunk ids contribute nothing rather than raising — retrieval output is not
    the place to enforce the dataset's schema.

    Every other metric comes from ``score_fns``, keyed as in
    :func:`answer_correctness_fns`, and is ``None`` when its function was not
    supplied. ``contexts`` are the retrieved chunk *texts*, needed only by
    ``llm_context_recall``.

    Score functions are called directly and their exceptions propagate, unlike
    :meth:`src.evals.judge.RagasJudge.score`, which downgrades a failing metric
    to an error string. A reference metric that cannot run is a broken eval, not
    a low score, and a batch runner is the right place to decide whether to
    continue.
    """
    fns = score_fns or {}
    retrieved_video_ids = [
        video_id for video_id in map(chunk_video_id, retrieved_chunk_ids) if video_id
    ]
    scores: dict[str, float | None] = {
        "context_recall": round(context_recall(retrieved_chunk_ids, entry.expected_chunk_ids), 4),
        "video_recall": round(video_recall(retrieved_video_ids, entry.expected_video_ids), 4),
    }
    for name in ("answer_correctness", "answer_similarity", "llm_context_recall"):
        fn = fns.get(name)
        if fn is None:
            scores[name] = None
            continue
        scores[name] = round(
            float(fn(entry.question, answer, entry.reference_answer, list(contexts or []))), 4
        )
    return scores
