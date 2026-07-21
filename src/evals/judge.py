"""RAGAS answer judging for the web workbench.

Every setup's answer to a question is scored with the same three RAGAS
metrics — faithfulness (is the answer supported by the retrieved chunks?),
answer relevancy (does it address the question?), and context precision
(were the retrieved chunks useful?) — so all retrieval methods are graded
under one eval process.

The judge LLM defaults to the DeepSeek chat model already configured for
answering; override with ``YT_AGENT_JUDGE_MODEL`` / ``YT_AGENT_JUDGE_API_KEY``
/ ``YT_AGENT_JUDGE_BASE_URL`` to grade with an independent provider.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from src.config import Settings

RUBRIC_VERSION = "ragas-v1"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision"]


def ragas_version() -> str:
    """The installed ragas version, stamped onto every evaluation record.

    Metric implementations change between releases, so a score is only
    comparable to another score produced by the same version.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("ragas")
    except PackageNotFoundError:
        return "unknown"


# (question, answer, contexts) -> score in [0, 1]
ScoreFn = Callable[[str, str, list[str]], float]


@dataclass
class RagasJudge:
    """Scores answers via injected metric callables (real RAGAS or fakes)."""

    score_fns: dict[str, ScoreFn]
    judge_model: str
    embedding_model: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "RagasJudge":
        # ragas and its model stack load slowly; keep them out of module import.
        from src.evals import _ragas_compat

        _ragas_compat.install()

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
        )

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

        faithfulness = Faithfulness(llm=llm)
        # strictness controls how many synthetic questions are generated per
        # sample via a single n>1 chat completion; DeepSeek's OpenAI-compatible
        # endpoint rejects n>1, so keep it at 1.
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
        precision = LLMContextPrecisionWithoutReference(llm=llm)

        def sample(question: str, answer: str, contexts: list[str]) -> Any:
            return SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=list(contexts),
            )

        score_fns: dict[str, ScoreFn] = {
            "faithfulness": lambda q, a, c: float(
                faithfulness.single_turn_score(sample(q, a, c))
            ),
            "answer_relevancy": lambda q, a, c: float(
                relevancy.single_turn_score(sample(q, a, c))
            ),
            "context_precision": lambda q, a, c: float(
                precision.single_turn_score(sample(q, a, c))
            ),
        }
        return cls(
            score_fns=score_fns,
            judge_model=model,
            embedding_model=settings.embedding_model,
        )

    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        """Run every metric; a failing metric records an error, not a crash."""
        started = time.monotonic()
        scores: dict[str, float] = {}
        errors: list[str] = []
        for name, fn in self.score_fns.items():
            try:
                value = fn(question, answer, contexts)
                if value is None or math.isnan(value):
                    raise ValueError("metric returned no score")
                scores[name] = round(float(value), 4)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        composite = round(sum(scores.values()) / len(scores), 4) if scores else None
        return {
            "judge": "ragas",
            "judge_model": self.judge_model,
            "rubric_version": RUBRIC_VERSION,
            "ragas_version": ragas_version(),
            "embedding_model": self.embedding_model,
            "scores": scores,
            "composite": composite,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "error": "; ".join(errors) if errors else None,
        }


def unjudgeable(reason: str, judge_model: str = "") -> dict[str, Any]:
    """An evaluation record for answers that cannot be scored at all."""
    return {
        "judge": "ragas",
        "judge_model": judge_model,
        "rubric_version": RUBRIC_VERSION,
        "ragas_version": ragas_version(),
        "embedding_model": None,
        "scores": {},
        "composite": None,
        "elapsed_seconds": 0.0,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "error": reason,
    }
