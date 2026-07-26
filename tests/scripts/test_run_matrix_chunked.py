from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from src.config import Settings
from src.evals.golden import GoldenEntry

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_matrix_chunked.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_matrix_chunked", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-matrix-chunked",
        log_transcript_artifacts=False,
    )


def test_main_checkpoints_failed_cell_without_nameerror(monkeypatch, tmp_path) -> None:
    """``run`` is only bound inside the ``try`` block around ``run_golden_eval``;
    when a cell raises (the except branch), the checkpoint writer must read the
    always-bound ``record["config"]`` rather than the unbound ``run["config"]``,
    or every *failed* cell crashes the driver instead of being checkpointed as
    an error and moved past. Exercises the real nested ``score_setup`` closure
    inside ``main`` end to end, with ``run_golden_eval`` stubbed to raise.
    """
    module = _load_module()
    settings = _settings(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "CHECKPOINT", tmp_path / ".yt-agent" / "matrix_checkpoint.jsonl")
    monkeypatch.setattr(module, "load_settings", lambda: settings)

    entry = GoldenEntry(
        id="q1",
        question="What is discussed?",
        reference_answer="A reference answer.",
        expected_video_ids=["vid1"],
        expected_chunk_ids=["chunk:vid1:0"],
        domain="corpus",
    )
    monkeypatch.setattr(module, "load_golden", lambda: [entry])

    def failing_run_golden_eval(*a, **k):
        raise RuntimeError("DeepSeek 402 Insufficient Balance")

    monkeypatch.setattr(module, "run_golden_eval", failing_run_golden_eval)

    class FakeRunner:
        @classmethod
        def from_settings(cls, settings):
            return cls()

    class FakeJudge:
        @classmethod
        def from_settings(cls, settings):
            return cls()

    monkeypatch.setattr("src.chat.setups.RagSetupRunner", FakeRunner)
    monkeypatch.setattr("src.evals.judge.RagasJudge", FakeJudge)
    monkeypatch.setattr("src.evals.golden.answer_correctness_fns", lambda settings: {})

    saved: dict = {}
    monkeypatch.setattr(module, "save_run", lambda result: saved.setdefault("result", result) or Path("evals/runs/fake.json"))

    monkeypatch.setattr(sys, "argv", ["run_matrix_chunked.py", "--setups", "rag_llm"])

    exit_code = module.main()

    assert exit_code == 0
    checkpoint_lines = module.CHECKPOINT.read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_lines) == 1
    checkpointed = json.loads(checkpoint_lines[0])
    assert checkpointed["config"] == {}
    assert "402 Insufficient Balance" in checkpointed["entry"]["error"]
    # The failed cell still made it into the assembled matrix result rather
    # than crashing main() with an unbound-variable NameError.
    assert saved["result"]["runs"]["rag_llm"]["entries"][0]["error"]
