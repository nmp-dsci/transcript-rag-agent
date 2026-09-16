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


def test_job_routes_and_scope_start(settings: Settings, tmp_path: Path):
    """The write path end to end against a fake run function and fake provider."""
    import threading

    from types import SimpleNamespace

    from src.api.main import create_app

    calls: list = []
    done = threading.Event()

    def run_fn(job, on_event):
        calls.append(job)
        on_event(
            {
                "at": "t",
                "stage": "scope",
                "status": "done",
                "message": "ok",
                "video_ids": job.video_ids,
            }
        )
        on_event({"at": "t", "stage": "publish", "status": "done", "message": "v1", "version": 1})
        done.set()
        return {"version": 1, "cite_valid": 4, "cite_total": 4, "gaps": []}

    class FakeStore:
        def query_by_video_ids(self, ids, q, k):
            return []

    class FakeProvider:
        chunk_store = FakeStore()

        def get_context(self, question, top_k=10, **kw):
            return SimpleNamespace(retrieved_chunks=[SimpleNamespace(video_id="abc123XYZ")])

    corpus = {
        "videos": [
            {
                "video_id": "abc123XYZ",
                "title": "LLM judge basics",
                "channel_name": "A",
                "chunk_count": 12,
            },
            {"video_id": "zzz", "title": "Unrelated", "channel_name": "B", "chunk_count": 3},
        ],
        "channels": [],
        "totals": {},
        "insights": [],
    }
    sdk_state = {"problem": None}
    app = create_app(
        replace(settings, demo_mode=False),
        runner_factory=lambda: SimpleNamespace(provider=FakeProvider()),
        judge_factory=forbidden,
        graph_store_factory=forbidden,
        corpus_fn=lambda: corpus,
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        runs_dir=tmp_path / "runs",
        frontend_dist=tmp_path / "no-bundle",
        guides_dir=tmp_path / "guides",
        guide_run_fn=run_fn,
        guide_sdk_check=lambda: sdk_state["problem"],
    )
    client = TestClient(app)

    assert client.get("/api/guides/job").json() == {"job": None, "sdk": None}

    scoped = client.post("/api/guides/scope", json={"topic": "LLM judge", "limit": 5}).json()
    assert [c["video_id"] for c in scoped["candidates"]] == ["abc123XYZ"]
    assert scoped["candidates"][0]["title_match"] is True and len(scoped["probes"]) == 8
    assert client.post("/api/guides/scope", json={"topic": "   "}).status_code == 422

    # Validation before anything starts.
    assert client.post("/api/guides", json={"topic": "x", "video_ids": []}).status_code == 422
    assert client.post("/api/guides", json={"topic": "x", "video_ids": ["nope"]}).status_code == 422
    assert (
        client.post(
            "/api/guides", json={"topic": "x", "slug": "Bad Slug", "video_ids": ["abc123XYZ"]}
        ).status_code
        == 422
    )
    sdk_state["problem"] = "no token"
    assert (
        client.post("/api/guides", json={"topic": "x", "video_ids": ["abc123XYZ"]}).status_code
        == 503
    )
    sdk_state["problem"] = None

    started = client.post(
        "/api/guides",
        json={"topic": "LLM judge", "video_ids": ["abc123XYZ", "abc123XYZ"], "allow_web": True},
    )
    assert started.status_code == 202
    job = started.json()
    assert job["kind"] == "write" and job["slug"] == "llm-judge" and job["title"] == "Llm Judge"
    assert job["video_ids"] == ["abc123XYZ"] and job["allow_web"] is True
    assert done.wait(2)
    assert calls[0].slug == "llm-judge"
    for _ in range(50):
        snap = client.get("/api/guides/job").json()["job"]
        if snap["status"] == "done":
            break
        import time

        time.sleep(0.02)
    assert snap["status"] == "done" and snap["version"] == 1 and snap["cite_valid"] == 4
    assert "/api/guides/job" not in client.get("/api/guides/job").text  # not the 404 body


def test_ask_a_question_scopes_and_starts_at_once(settings: Settings, tmp_path: Path):
    """The ask path: a plain question, no video_ids — the server derives the
    topic, scopes the corpus and starts, with the question as the working
    title until the composer names the page."""
    import threading

    from types import SimpleNamespace

    from src.api.main import create_app

    done = threading.Event()
    seen: list = []

    def run_fn(job, on_event):
        seen.append(job)
        done.set()
        return {
            "version": 1,
            "cite_valid": 1,
            "cite_total": 1,
            "gaps": [],
            "title": "Calibrating LLM Judges",
        }

    class FakeStore:
        def query_by_video_ids(self, ids, q, k):
            return []

    class FakeProvider:
        chunk_store = FakeStore()

        def get_context(self, question, top_k=10, **kw):
            return SimpleNamespace(retrieved_chunks=[SimpleNamespace(video_id="abc123XYZ")])

    corpus = {
        "videos": [
            {
                "video_id": "abc123XYZ",
                "title": "LLM judge basics",
                "channel_name": "A",
                "chunk_count": 12,
            },
            {"video_id": "zzz", "title": "Unrelated", "channel_name": "B", "chunk_count": 3},
        ],
        "channels": [],
        "totals": {},
        "insights": [],
    }
    app = create_app(
        replace(settings, demo_mode=False),
        runner_factory=lambda: SimpleNamespace(provider=FakeProvider()),
        judge_factory=forbidden,
        graph_store_factory=forbidden,
        corpus_fn=lambda: corpus,
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        runs_dir=tmp_path / "runs",
        frontend_dist=tmp_path / "no-bundle",
        guides_dir=tmp_path / "guides",
        guide_run_fn=run_fn,
        guide_sdk_check=lambda: None,
    )
    client = TestClient(app)
    question = "How do teams calibrate an LLM judge against human labels, and when does it drift?"

    scoped = client.post("/api/guides/scope", json={"question": question}).json()
    assert scoped["topic"] == "calibrate an llm judge against human labels"
    assert scoped["question"] == question
    assert scoped["probes"][0] == question and len(scoped["probes"]) == 9
    assert [c["video_id"] for c in scoped["candidates"]] == ["abc123XYZ"]
    assert client.post("/api/guides/scope", json={}).status_code == 422

    started = client.post("/api/guides", json={"question": question})
    assert started.status_code == 202, started.text
    job = started.json()
    assert job["slug"] == "calibrate-an-llm-judge-against-human-labels"
    assert job["title"] == question and job["question"] == question
    assert job["topic"] == "calibrate an llm judge against human labels"
    assert job["video_ids"] == ["abc123XYZ"]
    assert done.wait(5)
    assert seen[0].question == question
    for _ in range(250):
        snap = client.get("/api/guides/job").json()["job"]
        if snap["status"] == "done":
            break
        import time

        time.sleep(0.02)
    # Once published, the job carries the composer's name for the rail.
    assert snap["status"] == "done" and snap["title"] == "Calibrating LLM Judges"


def test_ask_a_degenerate_question_is_rejected(settings: Settings, tmp_path: Path):
    """A question with no words in it (only punctuation/emoji) must not be
    allowed to derive a slug — two such questions would otherwise both
    degrade to the same 'guide' slug and silently collide."""
    from types import SimpleNamespace

    from src.api.main import create_app

    app = create_app(
        replace(settings, demo_mode=False),
        runner_factory=lambda: SimpleNamespace(provider=None),
        judge_factory=forbidden,
        graph_store_factory=forbidden,
        corpus_fn=lambda: {"videos": [], "channels": [], "totals": {}, "insights": []},
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        runs_dir=tmp_path / "runs",
        frontend_dist=tmp_path / "no-bundle",
        guides_dir=tmp_path / "guides",
        guide_run_fn=lambda job, on_event: {},
        guide_sdk_check=lambda: None,
    )
    client = TestClient(app)

    scoped = client.post("/api/guides/scope", json={"question": "???"})
    assert scoped.status_code == 422
    assert "words" in scoped.json()["detail"]

    started = client.post("/api/guides", json={"question": "\U0001f680\U0001f680\U0001f680"})
    assert started.status_code == 422
    assert "words" in started.json()["detail"]


def test_demo_blocks_starting_and_streaming_guide_jobs(settings: Settings, tmp_path: Path):
    client = make_client(settings, tmp_path, seeded(tmp_path), demo=True)
    assert client.post("/api/guides/scope", json={"topic": "x"}).status_code == 403
    assert client.post("/api/guides", json={"topic": "x", "video_ids": ["a"]}).status_code == 403
    assert client.get("/api/guides/job").status_code == 200
    assert client.get("/api/guides/job/stream").status_code == 403


def test_comment_and_revise_routes(settings: Settings, tmp_path: Path):
    from types import SimpleNamespace

    from src.api.main import create_app
    from src.guides.comments import open_comments
    from src.guides.catalog import GuidePaths

    jobs: list = []

    def run_fn(job, on_event):
        jobs.append(job)
        return {"version": 2, "cite_valid": 4, "cite_total": 4, "gaps": []}

    guides_dir = seeded(tmp_path)
    app = create_app(
        replace(settings, demo_mode=False),
        runner_factory=lambda: SimpleNamespace(provider=None),
        judge_factory=forbidden,
        graph_store_factory=forbidden,
        corpus_fn=lambda: {"videos": [], "channels": [], "totals": {}, "insights": []},
        history_path=tmp_path / "history.json",
        chat_html_path=tmp_path / "chat.html",
        runs_dir=tmp_path / "runs",
        frontend_dist=tmp_path / "no-bundle",
        guides_dir=guides_dir,
        guide_run_fn=run_fn,
        guide_sdk_check=lambda: None,
    )
    client = TestClient(app)
    assert client.post("/api/guides/nope/comments", json={"body": "x"}).status_code == 404
    assert client.post("/api/guides/tiny-guide/comments", json={"body": "   "}).status_code == 422
    created = client.post(
        "/api/guides/tiny-guide/comments",
        json={
            "body": "Cite the numbers",
            "anchor": "thesis-p2",
            "section_id": "thesis",
            "quote": "Body with",
        },
    )
    assert created.status_code == 201
    comment = created.json()
    assert comment["status"] == "open" and comment["anchor"] == "thesis-p2"
    detail = client.get("/api/guides/tiny-guide").json()
    assert [c["id"] for c in detail["comments"]] == [comment["id"]] and detail["comments_open"] == 1

    assert (
        client.post("/api/guides/tiny-guide/revise", json={"comment_ids": ["c-none"]}).status_code
        == 422
    )
    started = client.post("/api/guides/tiny-guide/revise", json={})
    assert started.status_code == 202
    job = started.json()
    assert (
        job["kind"] == "revise"
        and job["comment_ids"] == [comment["id"]]
        and job["title"] == "Tiny Guide"
    )
    for _ in range(50):
        if jobs:
            break
        import time

        time.sleep(0.02)
    assert jobs[0].comment_ids == [comment["id"]]
    assert (
        len(open_comments(GuidePaths(guides_dir, "tiny-guide"))) == 1
    )  # the fake run resolved nothing
