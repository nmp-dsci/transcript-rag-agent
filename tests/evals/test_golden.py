from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.config import Settings
from src.evals.golden import (
    CHUNK_ID_PATTERN,
    DEFAULT_DATASET_PATH,
    DOMAINS,
    METRIC_NAMES,
    GoldenDatasetError,
    GoldenEntry,
    chunk_video_id,
    context_recall,
    evaluate_entry,
    load_golden,
    video_recall,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _entry(**overrides: Any) -> dict[str, Any]:
    record = {
        "id": "g001",
        "question": "What changed?",
        "reference_answer": "Quite a lot changed.",
        "expected_video_ids": ["vid1"],
        "expected_chunk_ids": ["chunk:vid1:0", "chunk:vid1:1"],
        "domain": "property",
        "notes": "",
    }
    record.update(overrides)
    return record


def _write(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- the shipped dataset --------------------------------------------------


@pytest.fixture(scope="module")
def shipped() -> list[GoldenEntry]:
    return load_golden()


def test_shipped_dataset_loads(shipped: list[GoldenEntry]) -> None:
    assert DEFAULT_DATASET_PATH.exists()
    assert len(shipped) >= 7
    assert all(isinstance(entry, GoldenEntry) for entry in shipped)


def test_shipped_entry_ids_are_unique(shipped: list[GoldenEntry]) -> None:
    ids = [entry.id for entry in shipped]
    assert len(set(ids)) == len(ids)


def test_shipped_entries_are_well_formed(shipped: list[GoldenEntry]) -> None:
    """Re-assert the model's guarantees against the real file.

    ``load_golden`` would already have raised, but these are the properties a
    reader of this test needs to trust about the dataset, so state them.
    """
    for entry in shipped:
        assert entry.question.strip(), entry.id
        assert entry.reference_answer.strip(), entry.id
        assert entry.domain in DOMAINS, entry.id
        assert entry.expected_chunk_ids, entry.id
        assert entry.expected_video_ids, entry.id
        for chunk_id in entry.expected_chunk_ids:
            assert CHUNK_ID_PATTERN.match(chunk_id), f"{entry.id}: {chunk_id}"
        videos_from_chunks = {chunk_video_id(c) for c in entry.expected_chunk_ids}
        assert videos_from_chunks == set(entry.expected_video_ids), entry.id


def test_shipped_reference_answers_are_substantial(shipped: list[GoldenEntry]) -> None:
    """A one-line reference answer cannot discriminate between RAG setups."""
    for entry in shipped:
        assert len(entry.reference_answer) > 200, entry.id


def test_shipped_dataset_covers_both_domains(shipped: list[GoldenEntry]) -> None:
    assert {entry.domain for entry in shipped} == set(DOMAINS)


def test_shipped_entries_explain_themselves(shipped: list[GoldenEntry]) -> None:
    """Notes record where a question came from; without them entries rot silently."""
    for entry in shipped:
        assert entry.notes.strip(), entry.id


# --- loading and validation -----------------------------------------------


def test_load_accepts_a_bare_list(tmp_path: Path) -> None:
    entries = load_golden(_write(tmp_path, [_entry()]))
    assert [entry.id for entry in entries] == ["g001"]


def test_load_accepts_an_object_with_entries(tmp_path: Path) -> None:
    entries = load_golden(_write(tmp_path, {"corpus": {"videos": 10}, "entries": [_entry()]}))
    assert [entry.id for entry in entries] == ["g001"]


def test_load_missing_file_raises() -> None:
    with pytest.raises(GoldenDatasetError, match="not found"):
        load_golden(REPO_ROOT / "no" / "such" / "golden.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldenDatasetError, match="not valid JSON"):
        load_golden(path)


def test_load_object_without_entries_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldenDatasetError, match="no 'entries' key"):
        load_golden(_write(tmp_path, {"corpus": {}}))


def test_load_non_list_entries_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldenDatasetError, match="must be a list"):
        load_golden(_write(tmp_path, {"entries": {"id": "g001"}}))


def test_load_non_object_entry_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldenDatasetError, match="entry 0 must be an object"):
        load_golden(_write(tmp_path, ["g001"]))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"question": "   "}, id="blank-question"),
        pytest.param({"reference_answer": ""}, id="empty-reference"),
        pytest.param({"id": ""}, id="empty-id"),
        pytest.param({"domain": "finance"}, id="unknown-domain"),
        pytest.param({"expected_chunk_ids": []}, id="no-expected-chunks"),
        pytest.param({"expected_video_ids": []}, id="no-expected-videos"),
        pytest.param({"expected_chunk_ids": ["vid1:0"]}, id="chunk-id-missing-prefix"),
        pytest.param({"expected_chunk_ids": ["chunk:vid1:x"]}, id="chunk-id-non-numeric"),
        pytest.param({"expected_chunk_ids": ["chunk:vid1"]}, id="chunk-id-missing-index"),
        pytest.param(
            {"expected_chunk_ids": ["chunk:vid1:0", "chunk:vid1:0"]}, id="duplicate-chunk-ids"
        ),
    ],
)
def test_load_rejects_malformed_entries(tmp_path: Path, overrides: dict[str, Any]) -> None:
    with pytest.raises(GoldenDatasetError) as excinfo:
        load_golden(_write(tmp_path, [_entry(**overrides)]))
    # The message must name the position so a 9-entry file is debuggable.
    assert "entry 0" in str(excinfo.value)


def test_load_rejects_chunk_from_undeclared_video(tmp_path: Path) -> None:
    record = _entry(expected_chunk_ids=["chunk:vid1:0", "chunk:vid2:3"])
    with pytest.raises(GoldenDatasetError, match="missing from expected_video_ids"):
        load_golden(_write(tmp_path, [record]))


def test_load_rejects_video_with_no_expected_chunk(tmp_path: Path) -> None:
    record = _entry(expected_video_ids=["vid1", "vid2"])
    with pytest.raises(GoldenDatasetError, match="no expected chunk"):
        load_golden(_write(tmp_path, [record]))


def test_load_rejects_duplicate_entry_ids(tmp_path: Path) -> None:
    payload = [_entry(), _entry(question="Something else?")]
    with pytest.raises(GoldenDatasetError, match="duplicate entry ids"):
        load_golden(_write(tmp_path, payload))


def test_load_error_names_the_offending_entry(tmp_path: Path) -> None:
    payload = [_entry(), _entry(id="g002", domain="astrology")]
    with pytest.raises(GoldenDatasetError) as excinfo:
        load_golden(_write(tmp_path, payload))
    assert "entry 1" in str(excinfo.value)
    assert "g002" in str(excinfo.value)


def test_entry_strips_surrounding_whitespace() -> None:
    entry = GoldenEntry.model_validate(_entry(question="  What changed?  "))
    assert entry.question == "What changed?"


# --- chunk id parsing -----------------------------------------------------


@pytest.mark.parametrize(
    "chunk_id, expected",
    [
        ("chunk:vid1:0", "vid1"),
        ("chunk:5N-okeDdIuI:27", "5N-okeDdIuI"),
        ("chunk:vid1:x", ""),
        ("vid1:0", ""),
        ("", ""),
    ],
)
def test_chunk_video_id(chunk_id: str, expected: str) -> None:
    assert chunk_video_id(chunk_id) == expected


# --- recall math ----------------------------------------------------------


@pytest.mark.parametrize(
    "retrieved, expected, score",
    [
        pytest.param(["a", "b"], ["a", "b"], 1.0, id="perfect"),
        pytest.param(["a"], ["a", "b"], 0.5, id="half"),
        pytest.param(["a"], ["a", "b", "c", "d"], 0.25, id="quarter"),
        pytest.param(["z"], ["a", "b"], 0.0, id="zero"),
        pytest.param([], ["a", "b"], 0.0, id="retrieved-nothing"),
        pytest.param(["a", "b", "c"], ["a", "b"], 1.0, id="extra-retrieved-does-not-help"),
    ],
)
def test_context_recall_math(retrieved: list[str], expected: list[str], score: float) -> None:
    assert context_recall(retrieved, expected) == pytest.approx(score)


def test_context_recall_of_nothing_is_one() -> None:
    """Documented convention: with nothing expected, nothing was missed.

    Scoring 0.0 would punish a retriever for a reference entry that asked for
    nothing. ``GoldenEntry`` forbids empty expectations, so this only reaches
    ad-hoc callers.
    """
    assert context_recall(["a"], []) == 1.0
    assert context_recall([], []) == 1.0
    assert video_recall([], []) == 1.0


def test_recall_ignores_duplicates_on_both_sides() -> None:
    # Retrieving the same chunk twice must not inflate the score...
    assert context_recall(["a", "a"], ["a", "b"]) == 0.5
    # ...and a repeated expectation must not deflate it.
    assert context_recall(["a", "b"], ["a", "a", "b"]) == 1.0
    assert context_recall(["a", "a"], ["a", "a"]) == 1.0


def test_recall_is_order_independent() -> None:
    assert context_recall(["b", "a"], ["a", "b"]) == context_recall(["a", "b"], ["a", "b"])


def test_video_recall_math() -> None:
    assert video_recall(["v1", "v2"], ["v1", "v2"]) == 1.0
    assert video_recall(["v1"], ["v1", "v2"]) == 0.5
    assert video_recall(["v3"], ["v1", "v2"]) == 0.0
    assert video_recall(["v1", "v1", "v1"], ["v1"]) == 1.0


def test_video_recall_is_more_forgiving_than_chunk_recall() -> None:
    """Neighbouring chunks carry the same point; the video is the coarser truth."""
    retrieved_chunks = ["chunk:v1:4", "chunk:v1:5"]
    expected_chunks = ["chunk:v1:3", "chunk:v1:4"]
    assert context_recall(retrieved_chunks, expected_chunks) == 0.5
    assert video_recall([chunk_video_id(c) for c in retrieved_chunks], ["v1"]) == 1.0


# --- evaluate_entry -------------------------------------------------------


@pytest.fixture
def entry() -> GoldenEntry:
    return GoldenEntry.model_validate(
        _entry(
            expected_video_ids=["vid1", "vid2"],
            expected_chunk_ids=["chunk:vid1:0", "chunk:vid1:1", "chunk:vid2:7"],
        )
    )


def test_evaluate_entry_without_score_fns(entry: GoldenEntry) -> None:
    result = evaluate_entry(entry, "an answer", ["chunk:vid1:0", "chunk:vid1:1", "chunk:vid2:7"])

    assert set(result) == set(METRIC_NAMES)
    assert result["context_recall"] == 1.0
    assert result["video_recall"] == 1.0
    assert result["answer_correctness"] is None
    assert result["answer_similarity"] is None
    assert result["llm_context_recall"] is None


def test_evaluate_entry_partial_retrieval(entry: GoldenEntry) -> None:
    result = evaluate_entry(entry, "an answer", ["chunk:vid1:0", "chunk:vid9:2"])

    assert result["context_recall"] == pytest.approx(1 / 3, abs=1e-4)
    # vid1 found, vid2 missed: the noise chunk from vid9 neither helps nor hurts.
    assert result["video_recall"] == 0.5


def test_evaluate_entry_derives_video_recall_from_chunk_ids(entry: GoldenEntry) -> None:
    result = evaluate_entry(entry, "an answer", ["chunk:vid1:99", "chunk:vid2:99"])

    # None of the expected chunks, but both expected videos.
    assert result["context_recall"] == 0.0
    assert result["video_recall"] == 1.0


def test_evaluate_entry_ignores_unparseable_retrieved_ids(entry: GoldenEntry) -> None:
    """Retrieval output is not the place to enforce the dataset's schema."""
    result = evaluate_entry(entry, "an answer", ["not-a-chunk-id", "chunk:vid1:0"])

    assert result["context_recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["video_recall"] == 0.5


def test_evaluate_entry_with_injected_fakes(entry: GoldenEntry) -> None:
    seen: list[tuple[str, str, str, list[str]]] = []

    def record(score: float):
        def fn(question: str, answer: str, reference: str, contexts: list[str]) -> float:
            seen.append((question, answer, reference, contexts))
            return score

        return fn

    result = evaluate_entry(
        entry,
        "the answer",
        ["chunk:vid1:0"],
        score_fns={
            "answer_correctness": record(0.75),
            "answer_similarity": record(0.9),
            "llm_context_recall": record(0.6),
        },
        contexts=["ctx one", "ctx two"],
    )

    assert result["answer_correctness"] == 0.75
    assert result["answer_similarity"] == 0.9
    assert result["llm_context_recall"] == 0.6
    assert len(seen) == 3
    # Every reference metric sees the question, the candidate answer, the
    # reference answer, and the retrieved context texts — in that order.
    assert seen[0] == (entry.question, "the answer", entry.reference_answer, ["ctx one", "ctx two"])


def test_evaluate_entry_supplies_empty_contexts_by_default(entry: GoldenEntry) -> None:
    seen: list[list[str]] = []

    def fn(question: str, answer: str, reference: str, contexts: list[str]) -> float:
        seen.append(contexts)
        return 0.5

    evaluate_entry(entry, "a", ["chunk:vid1:0"], score_fns={"answer_correctness": fn})
    assert seen == [[]]


def test_evaluate_entry_rounds_scores(entry: GoldenEntry) -> None:
    result = evaluate_entry(
        entry,
        "a",
        ["chunk:vid1:0"],
        score_fns={"answer_correctness": lambda q, a, r, c: 0.123456789},
    )
    assert result["answer_correctness"] == 0.1235
    assert result["context_recall"] == 0.3333


def test_evaluate_entry_only_calls_supplied_fns(entry: GoldenEntry) -> None:
    called: list[str] = []
    result = evaluate_entry(
        entry,
        "a",
        ["chunk:vid1:0"],
        score_fns={"answer_similarity": lambda q, a, r, c: called.append("sim") or 0.4},
    )
    assert called == ["sim"]
    assert result["answer_similarity"] == 0.4
    assert result["answer_correctness"] is None
    assert result["llm_context_recall"] is None


def test_evaluate_entry_ignores_unknown_score_fns(entry: GoldenEntry) -> None:
    result = evaluate_entry(
        entry,
        "a",
        ["chunk:vid1:0"],
        score_fns={"faithfulness": lambda q, a, r, c: 1.0},
    )
    assert set(result) == set(METRIC_NAMES)


def test_evaluate_entry_propagates_score_fn_errors(entry: GoldenEntry) -> None:
    """A reference metric that cannot run is a broken eval, not a low score."""

    def broken(question: str, answer: str, reference: str, contexts: list[str]) -> float:
        raise RuntimeError("llm timeout")

    with pytest.raises(RuntimeError, match="llm timeout"):
        evaluate_entry(entry, "a", ["chunk:vid1:0"], score_fns={"answer_correctness": broken})


# --- answer_correctness_fns -----------------------------------------------


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "superdata_api_key": "sd-key",
        "deepseek_api_key": "ds-key",
        "deepseek_model": "deepseek-v4-flash",
        "deepseek_base_url": "https://api.deepseek.com",
        "chroma_path": Path("/tmp/chroma"),
        "mlflow_tracking_uri": "file:/tmp/mlruns",
        "mlflow_experiment_name": "test",
        "log_transcript_artifacts": False,
    }
    values.update(overrides)
    return Settings(**values)


class _FakeChatOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeEmbeddings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def offline_model_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    """Stub out the two constructors that would otherwise reach the network.

    ``ChatOpenAI`` would need a live endpoint and ``HuggingFaceEmbeddings``
    would download a sentence-transformers model. ragas itself is imported for
    real, so this still exercises the actual metric construction.
    """
    import langchain_huggingface
    import langchain_openai

    calls: dict[str, list[dict[str, Any]]] = {"llm": [], "embeddings": []}

    def fake_llm(**kwargs: Any) -> _FakeChatOpenAI:
        calls["llm"].append(kwargs)
        return _FakeChatOpenAI(**kwargs)

    def fake_embeddings(**kwargs: Any) -> _FakeEmbeddings:
        calls["embeddings"].append(kwargs)
        return _FakeEmbeddings(**kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_llm)
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", fake_embeddings)
    return calls


def test_answer_correctness_fns_returns_the_reference_metrics(
    offline_model_stack: dict[str, list[dict[str, Any]]],
) -> None:
    from src.evals.golden import answer_correctness_fns

    fns = answer_correctness_fns(_settings())

    assert set(fns) == {"answer_correctness", "answer_similarity", "llm_context_recall"}
    assert all(callable(fn) for fn in fns.values())


def test_answer_correctness_fns_mirrors_the_judge_stack(
    offline_model_stack: dict[str, list[dict[str, Any]]],
) -> None:
    """Correctness must come from the same judge as faithfulness, or the two
    numbers on a scoreboard row are not comparable."""
    from src.evals.golden import answer_correctness_fns

    answer_correctness_fns(_settings())

    (llm_kwargs,) = offline_model_stack["llm"]
    assert llm_kwargs["model"] == "deepseek-v4-flash"
    assert llm_kwargs["api_key"] == "ds-key"
    assert llm_kwargs["base_url"] == "https://api.deepseek.com"
    assert llm_kwargs["temperature"] == 0.0

    (embedding_kwargs,) = offline_model_stack["embeddings"]
    assert embedding_kwargs["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_answer_correctness_fns_prefers_the_independent_judge_settings(
    offline_model_stack: dict[str, list[dict[str, Any]]],
) -> None:
    from src.evals.golden import answer_correctness_fns

    answer_correctness_fns(
        _settings(
            judge_model="judge-model",
            judge_api_key="judge-key",
            judge_base_url="https://judge.example",
        )
    )

    (llm_kwargs,) = offline_model_stack["llm"]
    assert llm_kwargs["model"] == "judge-model"
    assert llm_kwargs["api_key"] == "judge-key"
    assert llm_kwargs["base_url"] == "https://judge.example"


def test_answer_correctness_fns_are_injectable_into_evaluate_entry(
    entry: GoldenEntry, offline_model_stack: dict[str, list[dict[str, Any]]]
) -> None:
    """The real factory's keys must line up with what evaluate_entry looks for."""
    from src.evals.golden import answer_correctness_fns

    fns = answer_correctness_fns(_settings())
    stubbed = {name: (lambda q, a, r, c: 0.5) for name in fns}

    result = evaluate_entry(entry, "a", ["chunk:vid1:0"], score_fns=stubbed)
    assert result["answer_correctness"] == 0.5
    assert result["answer_similarity"] == 0.5
    assert result["llm_context_recall"] == 0.5


def test_ragas_is_not_imported_at_module_import() -> None:
    """ragas and its model stack load slowly; importing golden must stay cheap."""
    code = "import sys; import src.evals.golden; assert 'ragas' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
