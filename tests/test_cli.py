from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src import cli
from src.config import Settings
from src.transcripts.discovery import DiscoveredVideo
from src.transcripts.models import Transcript


class FakeStore:
    def __init__(self, path: Path, *args, **kwargs) -> None:
        self.transcript = None
        self.upserts = 0

    def get(self, video_id: str):
        return self.transcript

    def upsert(self, transcript):
        self.transcript = transcript
        self.upserts += 1


class FakeFetcher:
    calls = 0

    def __init__(self, api_key: str, *args, **kwargs) -> None:
        self.api_key = api_key

    def fetch(self, url: str) -> Transcript:
        FakeFetcher.calls += 1
        return Transcript(
            video_id="3hk7nO_q0a8",
            url=url,
            raw_text="cached transcript",
            fetched_at=datetime.now(timezone.utc),
        )


class FakeAgent:
    @classmethod
    def from_settings(cls, settings, context_provider=None):
        agent = cls()
        agent.context_provider = context_provider
        agent.last_context = None
        return agent

    def summarize(self, request):
        self.last_context = self.context_provider.get_transcript(
            request.video_id, request.source_url
        )
        from src.agents.models import TranscriptSummary

        return TranscriptSummary(summary="summary", top_findings=["a", "b", "c"])

    def answer(self, request):
        self.last_context = self.context_provider.get_transcript(
            request.video_id, request.source_url
        )
        from src.agents.models import TranscriptAnswer

        return TranscriptAnswer(
            question=request.question,
            answer="answer",
            source_video_id=request.video_id,
        )


class FakeRagAgent:
    last_request = None

    @classmethod
    def from_settings(cls, settings, context_provider=None):
        agent = cls()
        agent.context_provider = context_provider
        agent.last_context = None
        return agent

    def answer(self, request):
        FakeRagAgent.last_request = request
        from src.agents.models import RagTranscriptAnswer

        return RagTranscriptAnswer(question=request.question, answer="rag answer")


class FakeEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class FakeChunkStore:
    def __init__(self, *args, **kwargs) -> None:
        pass


class FakeIndexer:
    last_refresh = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def index(self, source_url: str, refresh: bool = False, refresh_summary: bool = False):
        FakeIndexer.last_refresh = (source_url, refresh, refresh_summary)

        class Result:
            raw_document = None
            chunks = [object(), object()]
            summary_status = "hit"
            removed_chunk_ids: list[str] = []

        return Result()


class FakeMultiProvider:
    def __init__(self, *args, **kwargs) -> None:
        pass


def test_cli_routes_summarize_with_cache_miss(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    FakeFetcher.calls = 0

    result = cli.main(["summarize", "https://www.youtube.com/watch?v=3hk7nO_q0a8"])

    assert result == 0
    assert FakeFetcher.calls == 1
    assert "Top 3 findings" in capsys.readouterr().out


def test_fetch_no_refresh_uses_cached_transcript(monkeypatch, tmp_path, capsys) -> None:
    transcript = Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        raw_text="already cached",
        fetched_at=datetime.now(timezone.utc),
    )

    class CachedStore(FakeStore):
        def __init__(self, path: Path, *args, **kwargs) -> None:
            super().__init__(path, *args, **kwargs)
            self.transcript = transcript

    _patch_cli(monkeypatch, tmp_path, store_cls=CachedStore)
    FakeFetcher.calls = 0

    result = cli.main(["fetch", "https://www.youtube.com/watch?v=3hk7nO_q0a8", "--no-refresh"])

    assert result == 0
    assert FakeFetcher.calls == 0
    assert "Cache status: hit" in capsys.readouterr().out


def test_summarize_uses_cached_transcript_without_fetch(monkeypatch, tmp_path, capsys) -> None:
    transcript = Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        raw_text="already cached",
        fetched_at=datetime.now(timezone.utc),
    )

    class CachedStore(FakeStore):
        def __init__(self, path: Path, *args, **kwargs) -> None:
            super().__init__(path, *args, **kwargs)
            self.transcript = transcript

    _patch_cli(monkeypatch, tmp_path, store_cls=CachedStore)
    FakeFetcher.calls = 0

    result = cli.main(["summarize", "https://www.youtube.com/watch?v=3hk7nO_q0a8"])

    assert result == 0
    assert FakeFetcher.calls == 0
    assert "summary" in capsys.readouterr().out


def test_rag_ask_uses_rag_only_agent(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)
    FakeRagAgent.last_request = None

    result = cli.main(["rag-ask", "question", "--top-k", "7"])

    assert result == 0
    assert FakeRagAgent.last_request is not None
    assert FakeRagAgent.last_request.question == "question"
    assert FakeRagAgent.last_request.source_url is None
    assert FakeRagAgent.last_request.top_k == 7
    assert "rag answer" in capsys.readouterr().out


def test_rag_ask_passes_transcript_filter_flags(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)
    FakeRagAgent.last_request = None

    result = cli.main(
        [
            "rag-ask",
            "question",
            "--filter-transcripts",
            "--transcript-filter-top-k",
            "3",
            "--transcript-filter-min-score",
            "0.4",
        ]
    )

    assert result == 0
    assert FakeRagAgent.last_request is not None
    assert FakeRagAgent.last_request.filter_transcripts is True
    assert FakeRagAgent.last_request.transcript_filter_top_k == 3
    assert FakeRagAgent.last_request.transcript_filter_min_score == 0.4
    assert "rag answer" in capsys.readouterr().out


def test_rag_ask_passes_recursive_flags(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)
    FakeRagAgent.last_request = None

    result = cli.main(
        [
            "rag-ask",
            "question",
            "--recursive",
            "--max-depth",
            "1",
            "--max-followups",
            "4",
            "--followup-top-k",
            "6",
            "--novelty-min-chunks",
            "1",
            "--max-total-followups",
            "5",
        ]
    )

    assert result == 0
    request = FakeRagAgent.last_request
    assert request is not None
    assert request.recursive is True
    assert request.recursion_options.max_depth == 1
    assert request.recursion_options.max_followups == 4
    assert request.recursion_options.followup_top_k == 6
    assert request.recursion_options.novelty_min_chunks == 1
    assert request.recursion_options.max_total_followups == 5
    assert "rag answer" in capsys.readouterr().out


def test_rag_ask_uses_recursive_env_default_and_opt_out(monkeypatch, tmp_path, capsys) -> None:
    settings = Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-cli",
        log_transcript_artifacts=False,
        rag_recursive_default=True,
    )
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda require_keys=True: settings)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", FakeMultiProvider)

    FakeRagAgent.last_request = None
    assert cli.main(["rag-ask", "question"]) == 0
    assert FakeRagAgent.last_request.recursive is True

    FakeRagAgent.last_request = None
    assert cli.main(["rag-ask", "question", "--no-recursive"]) == 0
    assert FakeRagAgent.last_request.recursive is False
    capsys.readouterr()


def test_index_rag_refreshes_pipeline_dashboard(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    calls = []
    monkeypatch.setattr(cli, "log_raw_transcript_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_summary_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_summary_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_refresh_rag_pipeline_dashboard",
        lambda settings: calls.append(settings) or tmp_path / "dashboard/rag_pipeline.html",
    )

    result = cli.main(["index-rag", "https://www.youtube.com/watch?v=3hk7nO_q0a8"])

    assert result == 0
    assert calls
    output = capsys.readouterr().out
    assert "RAG pipeline dashboard:" in output


def test_bulk_index_channel_dry_run_writes_run_record(monkeypatch, tmp_path, capsys) -> None:
    class BulkRawStore(FakeStore):
        def get_raw_document(self, video_id: str):
            return None

    class BulkChunkStore(FakeChunkStore):
        def has_chunks(self, video_id: str) -> bool:
            return False

        def count_chunks(self, video_id: str) -> int:
            return 0

    _patch_cli(monkeypatch, tmp_path, store_cls=BulkRawStore)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "TranscriptChunkStore", BulkChunkStore)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "_build_summary_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_summary_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "discover_latest_channel_videos",
        lambda channel, limit, client: [
            DiscoveredVideo(
                video_id="aaaaaaaaaaa",
                source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
                title="Video A",
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "_refresh_rag_pipeline_dashboard",
        lambda settings: tmp_path / "dashboard/rag_pipeline.html",
    )

    result = cli.main(
        [
            "bulk-index",
            "channel",
            "--channel",
            "@channel",
            "--latest",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Discovered: 1" in output
    assert "aaaaaaaaaaa discovered" in output
    run_files = list((tmp_path / "ingestion_runs").glob("*.json"))
    assert len(run_files) == 1


def test_refresh_rag_pipeline_dashboard_passes_default_filter_question(
    monkeypatch,
    tmp_path,
) -> None:
    settings = Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-cli",
        log_transcript_artifacts=False,
    )
    calls = {}
    monkeypatch.setattr(cli, "collect_pipeline_rows", lambda settings: [])
    monkeypatch.setattr(
        cli,
        "collect_filter_test_rows",
        lambda settings, rows, question: calls.setdefault("question", question) or [],
    )
    monkeypatch.setattr(cli, "write_dashboard", lambda **kwargs: calls.update(kwargs))

    output = cli._refresh_rag_pipeline_dashboard(settings)

    assert output == Path("dashboard/rag_pipeline.html")
    assert calls["question"] == cli.DEFAULT_FILTER_TEST_QUESTION
    assert calls["filter_test_question"] == cli.DEFAULT_FILTER_TEST_QUESTION


def _patch_cli(monkeypatch, tmp_path, store_cls=FakeStore) -> None:
    settings = Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-cli",
        log_transcript_artifacts=False,
    )
    monkeypatch.setattr(cli, "load_settings", lambda require_keys=True: settings)
    monkeypatch.setattr(cli, "RawTranscriptStore", store_cls)
    monkeypatch.setattr(cli, "SuperdataTranscriptFetcher", FakeFetcher)
    monkeypatch.setattr(cli, "TranscriptAgent", FakeAgent)
    monkeypatch.setattr(cli, "cli_run", _null_run)
    monkeypatch.setattr(cli, "log_transcript", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "log_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "log_answer", lambda *args, **kwargs: None)


class _null_run:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_eval_ablation_does_not_require_api_keys(monkeypatch, tmp_path, capsys) -> None:
    import src.evals.ablation as ablation
    import src.evals.regression as regression

    seen = {}

    def fake_load_settings(require_keys=True):
        seen["require_keys"] = require_keys
        return Settings(
            superdata_api_key="",
            deepseek_api_key="",
            deepseek_model="deepseek-v4",
            deepseek_base_url=None,
            chroma_path=tmp_path / "chroma",
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            mlflow_experiment_name="test-cli",
            log_transcript_artifacts=False,
        )

    monkeypatch.setattr(cli, "load_settings", fake_load_settings)
    result = {"run_id": "ablation-test", "entries": 0}
    monkeypatch.setattr(
        ablation,
        "run_default_ablation",
        lambda settings, top_k=None, sweep="default", on_progress=None: result,
    )
    monkeypatch.setattr(ablation, "format_table", lambda run: "table")
    monkeypatch.setattr(regression, "save_run", lambda run: tmp_path / "run.json")

    assert cli.main(["eval-ablation"]) == 0
    assert seen["require_keys"] is False
    capsys.readouterr()


# ── retrieval variants ────────────────────────────────────────────────────────


def _patch_rag_ask(monkeypatch, tmp_path) -> list:
    """Patch rag-ask's stack, capturing every provider it builds."""
    built: list = []

    class CapturingProvider:
        def __init__(self, *args, **kwargs) -> None:
            built.append(kwargs)

    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(cli, "RagIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "MultiTranscriptRagContextProvider", CapturingProvider)
    return built


def test_rag_ask_wires_the_requested_query_transform(monkeypatch, tmp_path) -> None:
    import src.rag.query_transform as query_transform

    built = _patch_rag_ask(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)
    asked: list = []
    monkeypatch.setattr(
        query_transform,
        "build_query_transform",
        lambda name, settings: asked.append(name) or object(),
    )

    assert cli.main(["rag-ask", "question", "--query-transform", "hyde"]) == 0
    assert asked == ["hyde"]
    assert built[0]["query_transform"] is not None


def test_rag_ask_without_a_transform_retrieves_on_the_question(monkeypatch, tmp_path) -> None:
    built = _patch_rag_ask(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "TranscriptChunkStore", FakeChunkStore)

    assert cli.main(["rag-ask", "question"]) == 0
    assert built[0]["query_transform"] is None


def test_rag_ask_contextual_reads_the_contextual_collection(monkeypatch, tmp_path) -> None:
    built = _patch_rag_ask(monkeypatch, tmp_path)
    opened: list[str] = []

    class RecordingChunkStore(FakeChunkStore):
        def __init__(self, path, embedding_model=None, collection_name=None) -> None:
            super().__init__()
            opened.append(collection_name)

    monkeypatch.setattr(cli, "TranscriptChunkStore", RecordingChunkStore)

    assert cli.main(["rag-ask", "question", "--contextual"]) == 0
    assert built[0]["query_transform"] is None
    # Retrieval reads the contextual collection; the indexer keeps writing to
    # the baseline one, since index-contextual owns the derived index.
    assert opened == ["transcript_chunks_contextual", "transcript_chunks"]


def test_index_contextual_refuses_an_empty_corpus(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "HuggingFaceEmbeddingModel", FakeEmbeddingModel)

    class EmptyChunkStore(FakeChunkStore):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def has_any_chunks(self) -> bool:
            return False

    monkeypatch.setattr(cli, "TranscriptChunkStore", EmptyChunkStore)

    assert cli.main(["index-contextual"]) == 1
    assert "index-rag" in capsys.readouterr().err


def test_eval_matrix_can_be_scoped_to_named_questions(monkeypatch, tmp_path) -> None:
    """Naming a sample is how coverage is traded for turnaround, so the run has
    to receive exactly those entries and nothing else."""
    import src.evals.matrix as matrix_module
    import src.evals.regression as regression

    _patch_cli(monkeypatch, tmp_path)
    seen: dict = {}

    def fake_run_matrix(runner, settings, **kwargs):
        seen["ids"] = [entry.id for entry in kwargs["entries"]]
        return {
            "run_id": "matrix-test",
            "setups": [],
            "comparison": {},
            "entry_count": len(seen["ids"]),
            "question_ids": list(seen["ids"]),
            "question_types": {"local": len(seen["ids"])},
            "cache_hits": 0,
            "cache_misses": 0,
            "runs": {},
        }

    monkeypatch.setattr(matrix_module, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(matrix_module, "format_matrix_table", lambda run: "table")
    monkeypatch.setattr(regression, "save_run", lambda run: tmp_path / "run.json")
    monkeypatch.setattr(cli, "RagTranscriptAgent", FakeRagAgent)

    from src.chat.setups import RagSetupRunner

    monkeypatch.setattr(RagSetupRunner, "from_settings", classmethod(lambda cls, s: object()))

    exit_code = cli.main(
        [
            "eval-matrix",
            "--setups",
            "rag_llm",
            "--questions",
            "g010,g001,g001",
            "--no-judge",
            "--no-reference-metrics",
        ]
    )

    assert exit_code == 0
    # Ordered by the request and de-duplicated, so one question cannot be
    # weighted twice in the averages.
    assert seen["ids"] == ["g010", "g001"]


def test_eval_matrix_rejects_an_unknown_question_id(monkeypatch, tmp_path, capsys) -> None:
    _patch_cli(monkeypatch, tmp_path)

    assert cli.main(["eval-matrix", "--questions", "g001,nope"]) == 2
    assert "nope" in capsys.readouterr().err


# ── re-scoring a committed critique run in place ──────────────────────────
#
# The gate that ungrades the retrieval arms landed after these runs were
# committed, so the files in ``evals/runs/`` carried numbers the scorer no
# longer certified while the app rendered them as measurements. ``--rescore
# --in-place`` is what closes that, and its whole value rests on moving exactly
# one thing.


def _stored_run() -> dict:
    return {
        "run_id": "critique-HELD-20260810-094922",
        "created_at": "2026-08-10T09:49:22+00:00",
        # A pack ablation is a critique run in a different envelope, and it is
        # read by a tab that keys off both of these.
        "kind": "pack-ablation",
        "topic": "resume-design",
        "verdicts": {"criteria_recall": {"leader": "merged"}},
        "config": {"matcher": "llm", "conflicts_available": 3},
        "cells": [
            {
                "setup": "rag_llm_filtered",
                "rubrics": 18,
                "scores": {"criteria_recall": 0.1579, "contested_coverage": 0.0},
                "conflicts": [{"conflict_id": "conflict:2"}],
                "conflicts_in_context": 2,
                "conflicts_named": 0,
            }
        ],
    }


def _rescored_run() -> dict:
    return {
        "run_id": "critique-HELD-20260811-101010",
        "created_at": "2026-08-11T10:10:10+00:00",
        "kind": "critique-eval",
        "grounding_gate": "retrieval_provenance",
        "ungraded_cells": ["rag_llm_filtered"],
        "rescored_from": "critique-HELD-20260810-094922",
        "config": {"matcher": "llm", "rescored_gate_from": "exclusive"},
        "cells": [
            {
                "setup": "rag_llm_filtered",
                "scores": {"criteria_recall": None, "contested_coverage": None},
                "gradable": False,
                "criteria_recall_ungated": 0.1579,
                "conflicts": [],
                "conflicts_in_context": 0,
                "conflicts_named": 0,
            }
        ],
    }


def test_a_rescore_keeps_the_contested_measurement_the_run_published() -> None:
    """The gate does not touch contested_coverage, so neither may the re-score.

    Its denominator is fixed by retrieval before the answering call, which is
    why it sits outside the gate. But the committed conflict corpus has been
    edited since these runs, so re-deriving it moves "0 of 2 disagreements in
    context" to "0 of 1" — a second difference, in the one file whose job is to
    make a single difference legible.
    """
    merged = cli._with_published_contested(_stored_run(), _rescored_run())

    cell = merged["cells"][0]
    assert cell["scores"]["contested_coverage"] == 0.0
    assert cell["conflicts_in_context"] == 2
    assert cell["conflicts"] == [{"conflict_id": "conflict:2"}]
    # And nothing else is restored: the gate's verdict stands.
    assert cell["scores"]["criteria_recall"] is None
    assert cell["gradable"] is False


def test_a_rescore_in_place_keeps_the_runs_identity_and_its_envelope() -> None:
    """Same run, same id, read under a different rule.

    The id is how the app expands a row and how the demo verdicts cite this
    measurement, and the envelope is how the packs tab finds it at all. A
    re-score that renamed either would be a new run — which is exactly the claim
    it must not make, because no answer was re-generated.
    """
    merged = cli._rescored_in_place(_stored_run(), _rescored_run())

    assert merged["run_id"] == "critique-HELD-20260810-094922"
    assert merged["created_at"] == "2026-08-10T09:49:22+00:00"
    assert merged["kind"] == "pack-ablation"
    assert merged["topic"] == "resume-design"
    assert merged["verdicts"] == {"criteria_recall": {"leader": "merged"}}
    # Envelope the scorer knows nothing about survives on the cell too.
    assert merged["cells"][0]["rubrics"] == 18
    # The gate's verdict is what actually changed.
    assert merged["grounding_gate"] == "retrieval_provenance"
    assert merged["ungraded_cells"] == ["rag_llm_filtered"]
    assert merged["cells"][0]["scores"]["criteria_recall"] is None
    # Nothing is lost: the published figure is on the cell, named as ungated.
    assert merged["cells"][0]["criteria_recall_ungated"] == 0.1579
    # And it does not claim to be derived from some other run.
    assert "rescored_from" not in merged
    assert merged["rescored_at"]
