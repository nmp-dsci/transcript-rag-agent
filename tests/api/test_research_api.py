"""The build-report route, over a temporary ``experts/``.

Two behaviours matter here and neither is obvious from the handler.

A topic with no loop answers ``null`` with a 200, not a 404 — three of the four
declared packs will never have a loop run against them, and the panel decides to
render nothing on that answer rather than having to distinguish "not built" from
"route is broken".

A half-written report reads as "no loop has been run" rather than taking the
Experiments tab down with a 500, which is the same rule the pack reader follows.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings

CATALOG = {
    "version": 1,
    "packs": [
        {
            "topic": "resume-design",
            "name": "Resume design",
            "artifact": "resume",
            "routing_text": "how to write a technical resume",
        }
    ],
}

REPORT = {
    "kind": "deep-research",
    "topic": "resume-design",
    "gaps": [{"gap_id": "g01", "missing": "nothing about contact details", "probe": "contact"}],
    "rounds": [],
    "scores": None,
}


def _client(settings: Settings, tmp_path: Path) -> TestClient:
    packs_dir = tmp_path / "experts"
    packs_dir.mkdir()
    (packs_dir / "packs.json").write_text(json.dumps(CATALOG), encoding="utf-8")
    app = create_app(
        settings,
        corpus_fn=lambda: {"videos": [], "total": 0},
        packs_dir=packs_dir,
        frontend_dist=tmp_path / "no-bundle",
        runs_dir=tmp_path / "runs",
    )
    return TestClient(app)


def test_a_topic_with_no_loop_answers_null_rather_than_404(
    settings: Settings, tmp_path: Path
) -> None:
    response = _client(settings, tmp_path).get("/api/packs/resume-design/research")
    assert response.status_code == 200
    assert response.json() is None


def test_the_report_is_served_exactly_as_the_loop_wrote_it(
    settings: Settings, tmp_path: Path
) -> None:
    """Nothing is reshaped in the route: the file on disk is the artifact, and a
    reader has to be able to diff what the app renders against what is in git."""
    client = _client(settings, tmp_path)
    topic = tmp_path / "experts" / "resume-design"
    topic.mkdir()
    (topic / "research.json").write_text(json.dumps(REPORT), encoding="utf-8")
    body = client.get("/api/packs/resume-design/research").json()
    assert body == REPORT


def test_an_unreadable_report_reads_as_no_loop_rather_than_a_500(
    settings: Settings, tmp_path: Path
) -> None:
    client = _client(settings, tmp_path)
    topic = tmp_path / "experts" / "resume-design"
    topic.mkdir()
    (topic / "research.json").write_text("{not json", encoding="utf-8")
    response = client.get("/api/packs/resume-design/research")
    assert response.status_code == 200
    assert response.json() is None
