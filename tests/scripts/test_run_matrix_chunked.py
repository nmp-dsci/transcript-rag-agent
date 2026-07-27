from __future__ import annotations

import importlib.util
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


def test_main_records_a_failed_cell_and_keeps_going(monkeypatch, tmp_path) -> None:
    """One cell raising must not take the driver down with it.

    A sweep is long and expensive; a single provider error (a 402, a timeout)
    has to be recorded against that cell and the remaining queue still run,
    or one bad question throws away every result beside it. Exercises the real
    nested ``score_setup`` closure inside ``main`` end to end, with
    ``run_golden_eval`` stubbed to raise.

    Originally written against the checkpoint-file driver; the driver now
    records cells in the shared eval cache instead, but the guarantee under
    test is the same one.
    """
    module = _load_module()
    settings = _settings(tmp_path)
    cache_dir = tmp_path / "eval_cache"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "DEFAULT_CACHE_DIR", cache_dir)
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

    def fake_save_run(result):
        saved["result"] = result
        return Path("evals/runs/fake.json")

    monkeypatch.setattr(module, "save_run", fake_save_run)
    monkeypatch.setattr(sys, "argv", ["run_matrix_chunked.py", "--setups", "rag_llm"])

    exit_code = module.main()

    assert exit_code == 0
    # The failed cell reaches the assembled result rather than crashing main().
    failed = saved["result"]["runs"]["rag_llm"]["entries"][0]
    assert "402 Insufficient Balance" in failed["error"]
    # ...and is never cached, so the next run retries it instead of treating a
    # transient provider error as this cell's permanent score.
    assert not cache_dir.exists() or list(cache_dir.glob("*.json")) == []
