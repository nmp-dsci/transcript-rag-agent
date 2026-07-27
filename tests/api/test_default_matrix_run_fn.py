"""The production wiring behind the Experiments tab's "Run eval matrix" button.

``create_app`` builds a default run function that assembles the answering
stack, runs the matrix and commits the result. Every other test injects a fake
``matrix_run_fn``, so without this one the real path would only ever be
exercised by a human pressing the button — and a typo in it would surface as a
failed background job rather than a failed test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="",
        deepseek_api_key="test-key",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri="",
        mlflow_experiment_name="test",
        log_transcript_artifacts=False,
    )


class Recorder:
    """Captures how the default run function calls into the eval stack."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.runner_built = 0
        self.judge_built = 0

    def run_matrix(self, runner: Any, settings: Settings, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        # Prove the callback reaches run_matrix by driving one cell through it.
        kwargs["on_cell"](
            {"setup": "rag_llm", "entry_id": "g001", "cached": True, "done": 1, "total": 1}
        )
        return {"run_id": "matrix-wired", "kind": "matrix", "cache_hits": 1, "cache_misses": 0}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()

    def fake_runner(cls: Any, settings: Settings) -> str:
        rec.runner_built += 1
        return "runner"

    def fake_judge(cls: Any, settings: Settings) -> str:
        rec.judge_built += 1
        return "judge"

    monkeypatch.setattr("src.evals.matrix.run_matrix", rec.run_matrix)
    monkeypatch.setattr(
        "src.evals.golden.answer_correctness_fns", lambda settings: {"answer_correctness": object()}
    )
    monkeypatch.setattr("src.chat.setups.RagSetupRunner.from_settings", classmethod(fake_runner))
    monkeypatch.setattr("src.evals.judge.RagasJudge.from_settings", classmethod(fake_judge))
    return rec


def wait_for_done(client: TestClient, timeout: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get("/api/eval/matrix").json()["job"]
        if job and job["status"] in {"done", "error"}:
            return job
        time.sleep(0.01)
    raise AssertionError("matrix run did not finish within timeout")


def build(settings: Settings, tmp_path: Path) -> tuple[TestClient, Path]:
    runs_dir = tmp_path / "runs"
    app = create_app(
        settings,
        corpus_fn=lambda: {"videos": [], "totals": {}},
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        index_fn=lambda argv: 0,
        frontend_dist=tmp_path / "no-bundle",
        runs_dir=runs_dir,
    )
    return TestClient(app), runs_dir


def test_button_press_runs_a_judged_reference_scored_matrix_and_commits_it(
    settings: Settings, tmp_path: Path, recorder: Recorder
) -> None:
    client, runs_dir = build(settings, tmp_path)

    client.post("/api/eval/matrix", json={"setups": ["rag_llm"]})
    job = wait_for_done(client)

    assert job["status"] == "done"
    assert job["run_id"] == "matrix-wired"
    assert job["cells_done"] == 1

    # Judged and reference-scored, or the Scoreboard would have no composite
    # and global/temporal questions no answer_correctness.
    assert recorder.kwargs["setups"] == ["rag_llm"]
    assert recorder.kwargs["judge"] is not None
    assert recorder.kwargs["reference_fns"] is not None
    assert recorder.runner_built == 1
    assert recorder.judge_built == 1

    committed = runs_dir / "matrix-wired.json"
    assert committed.exists(), "the run must be committed where both tabs read from"
    assert json.loads(committed.read_text())["run_id"] == "matrix-wired"


def test_a_committed_run_is_immediately_visible_to_both_tabs(
    settings: Settings, tmp_path: Path, recorder: Recorder
) -> None:
    client, _ = build(settings, tmp_path)
    assert client.get("/api/scoreboard").json()["runs"] == []

    client.post("/api/eval/matrix", json={"setups": ["rag_llm"]})
    wait_for_done(client)

    # The whole point of the pivot: one run, both surfaces, no extra step.
    board = client.get("/api/scoreboard").json()
    assert [run["run_id"] for run in board["runs"]] == ["matrix-wired"]
    experiments = client.get("/api/experiments").json()
    assert [run["run_id"] for run in experiments["matrix_runs"]] == ["matrix-wired"]


def test_a_failure_inside_the_run_surfaces_on_the_job(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("deepseek 402")

    monkeypatch.setattr("src.evals.matrix.run_matrix", boom)
    monkeypatch.setattr("src.evals.golden.answer_correctness_fns", lambda settings: {})
    monkeypatch.setattr(
        "src.chat.setups.RagSetupRunner.from_settings", classmethod(lambda cls, settings: None)
    )
    monkeypatch.setattr(
        "src.evals.judge.RagasJudge.from_settings", classmethod(lambda cls, settings: None)
    )

    client, runs_dir = build(settings, tmp_path)
    client.post("/api/eval/matrix", json={})
    job = wait_for_done(client)

    assert job["status"] == "error"
    assert "deepseek 402" in job["error"]
    # Nothing half-written: a failed sweep must not leave a run the tabs would rank.
    committed = list(runs_dir.glob("*.json")) if runs_dir.exists() else []
    assert committed == []
