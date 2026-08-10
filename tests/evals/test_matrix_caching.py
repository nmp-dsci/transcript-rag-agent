"""run_matrix's default-on caching: reuse across runs, invalidate on change."""

from __future__ import annotations

from dataclasses import dataclass

from src.chat.setups import SetupResult
from src.evals.golden import GoldenEntry
from src.evals.matrix import run_matrix


def entries() -> list[GoldenEntry]:
    return [
        GoldenEntry.model_validate(
            {
                "id": "g001",
                "question": "What changed in the budget?",
                "reference_answer": "Negative gearing was grandfathered.",
                "expected_video_ids": ["v1"],
                "expected_chunk_ids": ["chunk:v1:0"],
                "domain": "property",
            }
        )
    ]


@dataclass
class FakeSettings:
    deepseek_model: str = "deepseek-v4-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_mode: str = "hybrid"
    rag_top_k: int = 10
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    neighbor_span: int = 0
    neo4j_uri: str = "bolt://localhost:7687"
    judge_model: str | None = None
    judge_samples: int = 1


class FakeCorpus:
    """The corpus a fake runner retrieves from.

    ``run_matrix`` digests this into cache identity, because a cell scored
    before an ingestion is not a valid answer for the same question after one.
    A runner that exposes no corpus cannot be fingerprinted, so the fakes model
    the same contract the real ``RagSetupRunner`` does.
    """

    def __init__(self, video_ids=("v1", "v2"), chunks=2) -> None:
        self._metadatas = [{"video_id": video_ids[i % len(video_ids)]} for i in range(chunks)]

    def get(self, include=None):
        return {"metadatas": self._metadatas}


class FakeProvider:
    def __init__(self, corpus=None) -> None:
        self.chunk_store = type("Store", (), {"collection": corpus or FakeCorpus()})()


class CountingRunner:
    """A fake RagSetupRunner: records every call, answers deterministically."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.provider = FakeProvider()

    def run(self, key: str, question: str, *, top_k=None, scope=None) -> SetupResult:
        self.calls.append((key, question))
        return SetupResult(
            key=key,
            title=key,
            command=f"fake {key}",
            answer=f"Answer from {key}",
            token_estimate=10,
            chunk_count=1,
            elapsed_seconds=0.01,
            contexts=["some context"],
            retrieved_chunk_ids=["chunk:v1:0"],
            model="deepseek-v4-flash",
            embedding_model="all-MiniLM-L6-v2",
            top_k=top_k or 10,
        )


def test_a_second_run_with_identical_config_reuses_every_cell(tmp_path) -> None:
    runner = CountingRunner()
    settings = FakeSettings()

    first = run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    assert first["cache_misses"] == 1
    assert first["cache_hits"] == 0
    assert len(runner.calls) == 1

    second = run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    assert second["cache_hits"] == 1
    assert second["cache_misses"] == 0
    assert len(runner.calls) == 1  # no new answer call — the cell was reused


def test_adding_a_new_setup_only_scores_the_new_one(tmp_path) -> None:
    runner = CountingRunner()
    settings = FakeSettings()

    run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    assert runner.calls == [("rag_llm", "What changed in the budget?")]

    result = run_matrix(
        runner, settings, setups=["rag_llm", "graph_rag"], entries=entries(), cache_dir=tmp_path
    )
    assert result["cache_hits"] == 1  # rag_llm reused
    assert result["cache_misses"] == 1  # graph_rag freshly scored
    assert runner.calls == [
        ("rag_llm", "What changed in the budget?"),
        ("graph_rag", "What changed in the budget?"),
    ]
    assert set(result["runs"].keys()) == {"rag_llm", "graph_rag"}


def test_changing_the_answer_model_invalidates_the_cache(tmp_path) -> None:
    runner = CountingRunner()
    settings = FakeSettings()

    run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    settings.deepseek_model = "deepseek-v5"

    result = run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    assert result["cache_misses"] == 1
    assert result["cache_hits"] == 0
    assert len(runner.calls) == 2  # rescored under the new model


def test_editing_the_golden_question_invalidates_the_cache(tmp_path) -> None:
    runner = CountingRunner()
    settings = FakeSettings()

    run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)

    edited = entries()
    edited[0].question = "What changed in the budget this year?"
    result = run_matrix(runner, settings, setups=["rag_llm"], entries=edited, cache_dir=tmp_path)
    assert result["cache_misses"] == 1
    assert len(runner.calls) == 2


def test_refresh_bypasses_the_cache_even_when_nothing_changed(tmp_path) -> None:
    runner = CountingRunner()
    settings = FakeSettings()

    run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    result = run_matrix(
        runner,
        settings,
        setups=["rag_llm"],
        entries=entries(),
        cache_dir=tmp_path,
        refresh=True,
    )
    assert result["cache_misses"] == 1
    assert result["cache_hits"] == 0
    assert len(runner.calls) == 2


def test_a_failed_cell_is_not_cached_and_retries_next_run(tmp_path) -> None:
    class FlakyRunner:
        """Fails the first call, succeeds every call after — no shared
        ``calls`` bookkeeping with CountingRunner, so each attempt counts once."""

        def __init__(self) -> None:
            self.attempts = 0
            self.provider = FakeProvider()

        def run(self, key, question, *, top_k=None, scope=None):
            self.attempts += 1
            if self.attempts == 1:
                return SetupResult(key=key, title=key, command="x", answer="", error="boom")
            return SetupResult(
                key=key,
                title=key,
                command=f"fake {key}",
                answer=f"Answer from {key}",
                token_estimate=10,
                chunk_count=1,
                elapsed_seconds=0.01,
                contexts=["some context"],
                retrieved_chunk_ids=["chunk:v1:0"],
                model="deepseek-v4-flash",
                embedding_model="all-MiniLM-L6-v2",
                top_k=top_k or 10,
            )

    runner = FlakyRunner()
    settings = FakeSettings()

    first = run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    assert first["runs"]["rag_llm"]["entries"][0]["error"] == "boom"

    second = run_matrix(runner, settings, setups=["rag_llm"], entries=entries(), cache_dir=tmp_path)
    assert second["cache_misses"] == 1  # the errored cell was retried, not reused
    assert second["runs"]["rag_llm"]["entries"][0]["error"] is None
    assert runner.attempts == 2


# ── scoping a run to a sample of questions ───────────────────────────────────


def _two_entries() -> list[GoldenEntry]:
    first = entries()[0]
    second = first.model_copy(update={"id": "g007", "question": "What about tooling?"})
    return [first, second]


def test_a_run_records_which_questions_it_scored_not_just_how_many(tmp_path) -> None:
    """A sampled run is a different measurement from a whole-set one, and
    comparing them without the sample hides which questions were skipped."""
    result = run_matrix(
        CountingRunner(),
        FakeSettings(),
        setups=["rag_llm"],
        entries=_two_entries(),
        cache_dir=tmp_path,
    )

    assert result["entry_count"] == 2
    assert result["question_ids"] == ["g001", "g007"]


def test_scoping_to_a_sample_only_answers_that_sample(tmp_path) -> None:
    runner = CountingRunner()

    run_matrix(
        runner,
        FakeSettings(),
        setups=["rag_llm"],
        entries=_two_entries()[:1],
        cache_dir=tmp_path,
    )

    assert [question for _key, question in runner.calls] == ["What changed in the budget?"]
