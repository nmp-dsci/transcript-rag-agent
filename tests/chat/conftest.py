from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-chat",
        log_transcript_artifacts=False,
    )
