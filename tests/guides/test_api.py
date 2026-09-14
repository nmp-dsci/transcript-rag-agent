from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings
from src.guides.importer import import_lavish_html
from tests.guides.test_importer import SAMPLE


def forbidden():
    raise AssertionError("the guide routes must never load the LLM/retrieval stack")


def make_client(
    settings: Settings, tmp_path: Path, guides_dir: Path, demo: bool = False
) -> TestClient:
    return TestClient(
        create_app(
            replace(settings, demo_mode=demo),
            runner_factory=forbidden,
            judge_factory=forbidden,
            graph_store_factory=forbidden,
            corpus_fn=lambda: {"videos": [], "channels": [], "totals": {}, "insights": []},
            history_path=tmp_path / "history.json",
            chat_html_path=tmp_path / "chat.html",
            runs_dir=tmp_path / "runs",
            frontend_dist=tmp_path / "no-bundle",
            guides_dir=guides_dir,
        )
    )


def seeded(tmp_path: Path) -> Path:
    source = tmp_path / "page.html"
    source.write_text(SAMPLE, encoding="utf-8")
    guides_dir = tmp_path / "guides"
    import_lavish_html(source, slug="tiny-guide", guides_dir=guides_dir, known_videos={"abc123XYZ"})
    return guides_dir


def test_list_and_detail(settings: Settings, tmp_path: Path):
    client = make_client(settings, tmp_path, seeded(tmp_path))
    listing = client.get("/api/guides").json()
    assert [item["slug"] for item in listing["guides"]] == ["tiny-guide"]
    detail = client.get("/api/guides/tiny-guide").json()
    assert detail["title"] == "Tiny Guide"
    assert detail["claims"]["total"] == 4
    assert client.get("/api/guides/nope").status_code == 404
    assert client.get("/api/guides/..").status_code in (404, 405)


def test_static_mount_serves_page_stylesheet_and_bridge(settings: Settings, tmp_path: Path):
    guides_dir = seeded(tmp_path)
    (guides_dir / "guide-bridge.js").write_text("// bridge", encoding="utf-8")
    client = make_client(settings, tmp_path, guides_dir)
    page = client.get("/guides/tiny-guide/guide.html")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert client.get("/guides/guide.css").status_code == 200
    assert client.get("/guides/guide-bridge.js").status_code == 200
    assert client.get("/guides/tiny-guide/versions/v1.html").status_code == 200


def test_missing_guides_dir_is_an_empty_catalog(settings: Settings, tmp_path: Path):
    client = make_client(settings, tmp_path, tmp_path / "absent")
    assert client.get("/api/guides").json()["guides"] == []
    assert client.get("/guides/guide.css").status_code == 404


def test_demo_mode_serves_guides_read_only(settings: Settings, tmp_path: Path):
    client = make_client(settings, tmp_path, seeded(tmp_path), demo=True)
    assert client.get("/api/guides").status_code == 200
    assert client.get("/api/guides/tiny-guide").status_code == 200
    assert client.get("/guides/tiny-guide/guide.html").status_code == 200
    assert client.post("/api/guides", json={}).status_code == 403
