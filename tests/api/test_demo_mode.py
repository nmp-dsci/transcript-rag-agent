"""The demo gate: the public deployment's entire security boundary.

Demo mode has no login in front of it and no provider keys behind it, so the
properties tested here are the ones a visitor (or a script) could otherwise
abuse: every mutating/LLM route refuses, the two working GET streams refuse,
the read surface still answers, and the retrieval/judge stacks are never
loaded by anything a demo visitor can reach. The knowledge-graph routes are
also covered because demo serves them from the exported snapshot rather than
a live Neo4j.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings, load_settings

FAKE_CORPUS = {
    "videos": [{"video_id": "abc123", "title": "One video"}],
    "channels": [],
    "totals": {"videos": 1, "chunks": 2, "channels": 1},
    "insights": [],
}

CHUNK_RECORDS = [
    {
        "chunk_id": f"chunk:abc123:{index}",
        "video_id": "abc123",
        "chunk_index": index,
        "text": f"chunk {index}",
        "embedding": embedding,
    }
    for index, embedding in enumerate([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
]


def forbidden_factory():
    raise AssertionError("demo mode must never load the LLM/retrieval stack")


def demo_client(settings: Settings, tmp_path: Path, **overrides) -> TestClient:
    kwargs = dict(
        runner_factory=forbidden_factory,
        judge_factory=forbidden_factory,
        graph_store_factory=forbidden_factory,
        corpus_fn=lambda: FAKE_CORPUS,
        chunk_records_fn=lambda video_id: CHUNK_RECORDS,
        graph_records_fn=lambda: CHUNK_RECORDS,
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        runs_dir=tmp_path / "runs",
    )
    kwargs.update(overrides)
    return TestClient(create_app(replace(settings, demo_mode=True), **kwargs))


def test_health_reports_demo_mode(settings, tmp_path):
    assert demo_client(settings, tmp_path).get("/api/health").json()["mode"] == "demo"


def test_health_reports_full_mode_when_flag_off(settings, tmp_path):
    client = TestClient(
        create_app(
            settings,
            runner_factory=forbidden_factory,
            judge_factory=forbidden_factory,
            corpus_fn=lambda: FAKE_CORPUS,
            history_path=tmp_path / "history.json",
            chat_html_path=tmp_path / "chat.html",
        )
    )
    assert client.get("/api/health").json()["mode"] == "full"


@pytest.mark.parametrize(
    "path",
    [
        "/api/ask",
        "/api/judge",
        "/api/index",
        "/api/index/stream",
        "/api/index/queue",
        "/api/eval/matrix",
        "/api/rank",
        "/api/packs/resume-design/members/abc123",
    ],
)
def test_every_mutating_route_refuses(settings, tmp_path, path):
    response = demo_client(settings, tmp_path).post(path, json={})
    assert response.status_code == 403
    assert response.json() == {"detail": "demo"}


@pytest.mark.parametrize("path", ["/api/eval/matrix/stream", "/api/index/queue/stream"])
def test_the_two_working_get_streams_refuse(settings, tmp_path, path):
    response = demo_client(settings, tmp_path).get(path)
    assert response.status_code == 403
    assert response.json() == {"detail": "demo"}


def test_read_routes_still_answer(settings, tmp_path):
    client = demo_client(settings, tmp_path)
    assert client.get("/api/corpus").json()["totals"]["videos"] == 1
    assert client.get("/api/history").status_code == 200
    assert client.get("/api/setups").status_code == 200
    assert client.get("/").status_code == 200


def test_chunk_graph_structure_is_allowed_without_a_query(settings, tmp_path):
    response = demo_client(settings, tmp_path).post(
        "/api/chunk-graph", json={"k": 2, "min_similarity": 0.0}
    )
    assert response.status_code == 200
    assert response.json()["nodes"]


def test_chunk_graph_query_overlay_refuses(settings, tmp_path):
    # The structure build is model-free; the query overlay is not. In demo
    # the overlay must refuse rather than load the embedding stack.
    response = demo_client(settings, tmp_path).post(
        "/api/chunk-graph", json={"k": 2, "query": "what is a resume"}
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "demo"}


def snapshot_settings(settings: Settings, tmp_path: Path) -> Settings:
    snapshot = tmp_path / "graph_snapshot"
    snapshot.mkdir()
    (snapshot / "knowledge.json").write_text(
        json.dumps({"nodes": [{"id": "e1"}], "edges": [], "communities": []})
    )
    (snapshot / "entities.json").write_text(
        json.dumps({"e1": {"entity": {"id": "e1", "name": "Resume"}, "claims": []}})
    )
    (snapshot / "video_chunks.json").write_text(
        json.dumps({"abc123": {"chunks": {"0": {"entities": ["Resume"], "claims": []}}}})
    )
    return replace(settings, graph_snapshot_dir=snapshot)


def test_knowledge_graph_serves_the_snapshot_not_neo4j(settings, tmp_path):
    # graph_store_factory raises, so a pass proves the store was never touched.
    client = demo_client(snapshot_settings(settings, tmp_path), tmp_path)
    assert client.get("/api/graph/knowledge").json()["nodes"] == [{"id": "e1"}]
    assert (
        client.get("/api/graph/knowledge/entities/e1").json()["entity"]["name"] == "Resume"
    )
    assert client.get("/api/graph/knowledge/entities/nope").status_code == 404
    chunks = client.get("/api/graph/knowledge/videos/abc123/chunks").json()["chunks"]
    assert chunks["0"]["entities"] == ["Resume"]
    # An unenriched video answers the same empty shape the live store would.
    empty = client.get("/api/graph/knowledge/videos/other/chunks").json()
    assert empty == {"chunks": {}}


def test_snapshot_is_ignored_outside_demo_mode(settings, tmp_path):
    # Same snapshot on disk, flag off: the live store answers (and here the
    # factory raising surfaces as the 503 the endpoint maps failures to).
    with_snapshot = snapshot_settings(settings, tmp_path)
    client = TestClient(
        create_app(
            with_snapshot,
            runner_factory=forbidden_factory,
            judge_factory=forbidden_factory,
            graph_store_factory=forbidden_factory,
            corpus_fn=lambda: FAKE_CORPUS,
            history_path=tmp_path / "history.json",
            chat_html_path=tmp_path / "chat.html",
        )
    )
    assert client.get("/api/graph/knowledge").status_code == 503


def test_load_settings_needs_no_keys_and_no_env_file_in_demo(monkeypatch, tmp_path):
    for name in ("SUPERDATA_API_KEY", "SUPADATA_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(tmp_path / "does-not-exist.env"))
    monkeypatch.setenv("YT_AGENT_DEMO_MODE", "true")
    loaded = load_settings(require_keys=True)
    assert loaded.demo_mode is True
    assert loaded.deepseek_api_key == ""


def test_demo_flag_off_still_requires_keys(monkeypatch, tmp_path):
    for name in ("SUPERDATA_API_KEY", "SUPADATA_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(tmp_path / "does-not-exist.env"))
    monkeypatch.delenv("YT_AGENT_DEMO_MODE", raising=False)
    with pytest.raises(Exception):
        load_settings(require_keys=True)
