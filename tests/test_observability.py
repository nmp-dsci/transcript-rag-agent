from __future__ import annotations

from pathlib import Path

import mlflow

from src.config import Settings
from src.observability import (
    cli_run,
    log_summary,
    log_transcript_filter_details,
    setup_mlflow,
)
from src.agents.models import TranscriptSummary
from src.rag.models import RetrievedChunk, RetrievedTranscriptSummary


def test_mlflow_setup_uses_configured_tracking_uri(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    setup_mlflow(settings)

    assert mlflow.get_tracking_uri() == f"file:{tmp_path / 'mlruns'}"


def test_cli_run_logs_summary_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with cli_run("summarize", settings, "3hk7nO_q0a8"):
        log_summary(TranscriptSummary(summary="s", top_findings=["a", "b", "c"]))

    runs = mlflow.search_runs(experiment_names=[settings.mlflow_experiment_name])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.command"] == "summarize"


def test_transcript_filter_logging_records_mlflow_details(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with cli_run("rag-ask", settings):
        log_transcript_filter_details(
            enabled=True,
            filter_top_k=3,
            min_score=0.4,
            selected_transcripts=[
                RetrievedTranscriptSummary(
                    transcript_id="raw_transcript:aaaaaaaaaaa",
                    video_id="aaaaaaaaaaa",
                    source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
                    summary="capital gains tax",
                    summary_model="deepseek-test",
                    summary_generated_at="2026-05-16T00:00:00+00:00",
                    summary_embedding=[1.0, 0.0, 1.0],
                    summary_embedding_model="fake",
                    summary_embedded_at="2026-05-16T00:01:00+00:00",
                    score=0.8,
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    transcript_id="raw_transcript:aaaaaaaaaaa",
                    video_id="aaaaaaaaaaa",
                    source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
                    chunk_index=0,
                    text="capital gains tax",
                    segment_count=1,
                    score=0.7,
                )
            ],
        )

    runs = mlflow.search_runs(experiment_names=[settings.mlflow_experiment_name])
    assert runs.iloc[0]["params.transcript_filter_enabled"] == "True"
    assert runs.iloc[0]["metrics.selected_transcript_count"] == 1
    assert runs.iloc[0]["tags.selected_video_ids"] == "aaaaaaaaaaa"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-exp",
        log_transcript_artifacts=False,
    )
