from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.guides.catalog import GuidePaths, read_manifest
from src.guides.prompts import composer_prompt, extractor_prompt, fix_prompt, load_skill
from src.guides.tools import QUALIFIED_TOOL, format_chunks
from src.guides.writer import (
    ALWAYS_DISALLOWED,
    WEB_TOOLS,
    AgentRequest,
    AgentResult,
    GuideWriter,
    WriterConfig,
    make_confinement_hook,
    sdk_env,
)

CHUNKS = {
    "v1": [{"chunk_index": 0, "text": "Judges drift when the rubric is vague."}],
    "v2": [{"chunk_index": 3, "text": "Pairwise comparison beats absolute scores."}],
    "v3": [{"chunk_index": 1, "text": "Calibrate the judge against human labels first."}],
}
VIDEOS = [
    {"video_id": "v1", "title": "Judge 101", "channel_name": "A", "chunk_count": 1},
    {"video_id": "v2", "title": "Pairwise", "channel_name": "A", "chunk_count": 1},
    {"video_id": "v3", "title": "Calibration", "channel_name": "B", "chunk_count": 1},
]


class Chunk:
    def __init__(self, video_id: str, index: int, text: str) -> None:
        self.video_id, self.chunk_index, self.text = video_id, index, text
        self.title = f"Title {video_id}"
        self.channel_name = "A" if video_id != "v3" else "B"
        self.start_seconds = 0.0
        self.end_seconds = 10.0
        self.source_url = f"https://www.youtube.com/watch?v={video_id}"


def chunks_for(video_ids: list[str]):
    return [
        Chunk(vid, r["chunk_index"], r["text"]) for vid in video_ids for r in CHUNKS.get(vid, [])
    ]


def lookup(video_id: str):
    return CHUNKS.get(video_id, [])


def retrieve(question: str, video_ids, top_k: int):
    return []


GOOD_PAGE = """<!doctype html><html lang="en"><head><title>T</title>
<link rel="stylesheet" href="/guides/guide.css"></head><body>
<section id="thesis"><p id="thesis-p1">Vague rubrics drift <cite data-video="v1" data-chunk="0" data-quote="rubric is vague">Judge 101</cite>.</p></section>
<section id="sources"><ol><li id="sources-li1"><cite data-video="v2" data-chunk="3">Pairwise</cite></li></ol></section>
<script src="/guides/guide-bridge.js"></script></body></html>
"""

BAD_PAGE = GOOD_PAGE.replace('data-chunk="0"', 'data-chunk="9"')


class FakeAgent:
    """Writes the files a real agent would, and records what it was asked."""

    def __init__(self, *, bad_first_compose: bool = False) -> None:
        self.requests: list[AgentRequest] = []
        self.bad_first_compose = bad_first_compose

    async def __call__(self, request: AgentRequest, retrieve, on_event) -> AgentResult:
        self.requests.append(request)
        on_event(
            {"type": "tool", "label": request.label, "name": "Read", "summary": "Read corpus/v1.md"}
        )
        if request.label.startswith("extract:"):
            name = request.label.split(":", 1)[1]
            cluster_ids = request.default_video_ids or []
            claims = [
                {
                    "theme": "t",
                    "claim": f"claim from {vid}",
                    "video_id": vid,
                    "chunk_index": CHUNKS[vid][0]["chunk_index"],
                    "quote": CHUNKS[vid][0]["text"][:12],
                    "kind": "principle",
                }
                for vid in cluster_ids
            ]
            claims.append(
                {
                    "theme": "t",
                    "claim": "made up",
                    "video_id": "v1",
                    "chunk_index": 42,
                    "quote": "nope",
                    "kind": "principle",
                }
            )
            (request.cwd / "evidence" / f"{name}.json").write_text(
                json.dumps(
                    {
                        "cluster": name,
                        "videos": cluster_ids,
                        "claims": claims,
                        "video_notes": [
                            {"video_id": vid, "contributed": f"note {vid}"} for vid in cluster_ids
                        ],
                        "gaps": ["cost per judged answer"],
                    }
                ),
                encoding="utf-8",
            )
        elif request.label == "compose":
            page = BAD_PAGE if self.bad_first_compose else GOOD_PAGE
            (request.cwd / "guide.html").write_text(page, encoding="utf-8")
            (request.cwd / "guide.md").write_text("---\ntitle: T\n---\n# T\n", encoding="utf-8")
        elif request.label.startswith("fix:"):
            (request.cwd / "guide.html").write_text(GOOD_PAGE, encoding="utf-8")
        return AgentResult(text="done", cost_usd=0.01, duration_ms=10, num_turns=3)


def make_writer(tmp_path: Path, agent: FakeAgent, **config) -> tuple[GuideWriter, list[dict]]:
    events: list[dict] = []
    paths = GuidePaths(tmp_path / "guides", "llm-as-a-judge")
    writer = GuideWriter(
        paths,
        chunks_for=chunks_for,
        retrieve=retrieve,
        known_videos={"v1", "v2", "v3"},
        chunk_lookup=lookup,
        config=WriterConfig(parallel_extract=False, **config),
        agent_fn=agent,
        on_event=events.append,
        skill_text="SKILL RULES",
    )
    return writer, events


def test_write_runs_every_stage_and_publishes(tmp_path: Path):
    agent = FakeAgent()
    writer, events = make_writer(tmp_path, agent)
    manifest = writer.write(
        topic="LLM-as-a-judge",
        title="LLM-as-a-Judge",
        video_ids=["v1", "v2", "v3"],
        videos_meta=VIDEOS,
        compiled_at="2026-09-14",
    )
    paths = writer.paths
    # Export: one file per video, clusters keep channels together.
    assert (paths.corpus / "v1.md").is_file() and (paths.corpus / "INDEX.md").is_file()
    # Extract: one request per cluster, corpus-only tools, cwd is the guide dir.
    extract = [r for r in agent.requests if r.label.startswith("extract:")]
    assert len(extract) == 1  # three videos → one cluster
    assert extract[0].model == "claude-sonnet-5" and extract[0].cwd == paths.dir
    assert QUALIFIED_TOOL in extract[0].allowed_tools and "Edit" not in extract[0].allowed_tools
    assert set(WEB_TOOLS) <= set(extract[0].disallowed_tools)
    assert set(ALWAYS_DISALLOWED) <= set(extract[0].disallowed_tools)
    assert "SKILL RULES" in extract[0].system_prompt and "no web access" in extract[0].system_prompt
    # Evidence was verified: the made-up claim is marked, the real ones count.
    evidence = json.loads((paths.evidence / "cluster-1.json").read_text())
    assert evidence["verified_claims"] == 3
    assert [c["verified"] for c in evidence["claims"]] == [True, True, True, False]
    # Compose: Opus, may edit, told the provenance numbers.
    compose = next(r for r in agent.requests if r.label == "compose")
    assert compose.model == "claude-opus-5" and "Edit" in compose.allowed_tools
    assert (
        "3 source videos · 3 transcript chunks read in full · 1 parallel extraction passes"
        in compose.prompt
    )
    # Verify + publish.
    assert manifest.cite_total == 2 and manifest.cite_valid == 2
    assert manifest.current_version == 1 and paths.version_html(1).is_file()
    assert manifest.chunk_count == 3 and manifest.cluster_count == 1
    assert manifest.gaps == ["cost per judged answer"]
    assert manifest.sources[0]["contributed"] == "note v1"
    assert manifest.model == {
        "extractor": "claude-sonnet-5",
        "composer": "claude-opus-5",
        "reviser": "claude-opus-5",
    }
    assert manifest.web_allowed is False
    assert manifest.provenance["costs"][0]["label"] == "extract:cluster-1"
    assert read_manifest(paths) is not None
    assert (paths.root / "index.json").is_file()
    # Events: every stage reported, tool calls logged, run log persisted.
    stages = [(e["stage"], e["status"]) for e in events]
    for expected in [
        ("scope", "done"),
        ("export", "done"),
        ("extract", "done"),
        ("compose", "done"),
        ("verify", "done"),
        ("publish", "done"),
    ]:
        assert expected in stages
    assert any(e["status"] == "tool" for e in events)
    assert writer.run_log is not None and writer.run_log.read_text().count("\n") == len(events)


def test_verify_failure_triggers_a_bounded_fix_pass(tmp_path: Path):
    agent = FakeAgent(bad_first_compose=True)
    writer, events = make_writer(tmp_path, agent)
    manifest = writer.write(topic="t", title="T", video_ids=["v1", "v2"], videos_meta=VIDEOS)
    labels = [r.label for r in agent.requests]
    assert labels == ["extract:cluster-1", "compose", "fix:1"]
    fix = agent.requests[-1]
    assert "chunk index does not exist" in fix.prompt and "#0" in fix.prompt
    assert manifest.cite_valid == 2
    assert (
        next(e for e in events if e["stage"] == "verify" and e["status"] == "done")["fix_passes"]
        == 1
    )


def test_verify_gives_up_after_max_fix_passes(tmp_path: Path):
    class NeverFixes(FakeAgent):
        async def __call__(self, request, retrieve, on_event):
            result = await super().__call__(request, retrieve, on_event)
            if request.label.startswith("fix:"):
                (request.cwd / "guide.html").write_text(BAD_PAGE, encoding="utf-8")
            return result

    agent = NeverFixes(bad_first_compose=True)
    writer, events = make_writer(tmp_path, agent, max_fix_passes=2)
    with pytest.raises(RuntimeError, match="do not resolve"):
        writer.write(topic="t", title="T", video_ids=["v1", "v2"], videos_meta=VIDEOS)
    assert [r.label for r in agent.requests] == ["extract:cluster-1", "compose", "fix:1", "fix:2"]
    assert events[-1]["stage"] == "run" and events[-1]["status"] == "error"
    assert read_manifest(writer.paths) is None  # nothing published


def test_resume_skips_completed_stages(tmp_path: Path):
    agent = FakeAgent()
    writer, _ = make_writer(tmp_path, agent)
    writer.write(topic="t", title="T", video_ids=["v1", "v2"], videos_meta=VIDEOS)
    again = FakeAgent()
    writer2, events = make_writer(tmp_path, again)
    manifest = writer2.write(topic="t", title="T", video_ids=["v1", "v2"], videos_meta=VIDEOS)
    # No agent ran: corpus, evidence and page were all on disk.
    assert again.requests == []
    assert [(e["stage"], e["status"]) for e in events if e["status"] == "skip"] == [
        ("export", "skip"),
        ("extract", "skip"),
        ("compose", "skip"),
    ]
    assert manifest.current_version == 2  # a re-publish snapshots a new version


def test_allow_web_adds_web_tools_and_records_it(tmp_path: Path):
    agent = FakeAgent()
    writer, _ = make_writer(tmp_path, agent, allow_web=True)
    manifest = writer.write(topic="t", title="T", video_ids=["v1"], videos_meta=VIDEOS)
    assert set(WEB_TOOLS) <= set(agent.requests[0].allowed_tools)
    assert not set(WEB_TOOLS) & set(agent.requests[0].disallowed_tools)
    assert "Web search is enabled" in agent.requests[0].system_prompt
    assert manifest.web_allowed is True


def test_agent_error_aborts_the_run(tmp_path: Path):
    class Fails(FakeAgent):
        async def __call__(self, request, retrieve, on_event):
            return AgentResult(error="billing_error")

    writer, events = make_writer(tmp_path, Fails())
    with pytest.raises(RuntimeError, match="extract:cluster-1: billing_error"):
        writer.write(topic="t", title="T", video_ids=["v1"], videos_meta=VIDEOS)


def test_prompts_bind_the_skill_and_files():
    skill = load_skill()
    assert skill.startswith("# Field guide writer") and "name: field-guide-writer" not in skill
    prompt = extractor_prompt(
        topic="x",
        cluster_name="cluster-2",
        files=["corpus/a.md"],
        video_ids=["a"],
        output_path="evidence/cluster-2.json",
    )
    assert "corpus/a.md" in prompt and "evidence/cluster-2.json" in prompt
    prompt = composer_prompt(
        topic="x",
        title="X",
        slug="x",
        evidence_files=["evidence/cluster-1.json"],
        corpus_files=["corpus/a.md"],
        video_count=1,
        chunk_count=2,
        cluster_count=1,
        compiled_at="d",
        sources=[{"video_id": "a", "title": "A"}],
        example_path="/ex.html",
    )
    assert "/ex.html" in prompt and "/guides/guide.css" in prompt and '"video_id": "a"' in prompt
    text = fix_prompt(
        structure_errors=["section without id"],
        invalid=[
            {
                "cite_index": 4,
                "section_id": "s",
                "video_id": "v",
                "chunk_index": 1,
                "quote": "q",
                "video_ok": True,
                "chunk_ok": True,
                "quote_ok": False,
            }
        ],
    )
    assert "section without id" in text and "data-quote not found" in text


def test_format_chunks_heads_each_block_with_the_cite_id():
    text = format_chunks(
        [
            {
                "video_id": "v1",
                "chunk_index": 7,
                "text": "hello",
                "title": "T",
                "start_seconds": 65,
                "score": 0.5,
            }
        ]
    )
    assert text.startswith("### v1@7 · T · 01:05 · score 0.500\nhello")
    assert "No chunks matched" in format_chunks([])


REVISED_PAGE = GOOD_PAGE.replace(
    '<p id="thesis-p1">Vague rubrics drift',
    '<p id="thesis-p1">Vague rubrics drift, calibrate first <cite data-video="v3" data-chunk="1" data-quote="Calibrate the judge">Calibration</cite>.',
)


class FakeReviser(FakeAgent):
    def __init__(self, receipt_items=None, write_receipt=True):
        super().__init__()
        self.receipt_items = receipt_items
        self.write_receipt = write_receipt

    async def __call__(self, request, retrieve, on_event):
        if request.label != "revise":
            return await super().__call__(request, retrieve, on_event)
        self.requests.append(request)
        (request.cwd / "guide.html").write_text(REVISED_PAGE, encoding="utf-8")
        if self.write_receipt:
            items = self.receipt_items
            if items is None:
                ids = [
                    line.split('"id": "')[1].split('"')[0]
                    for line in request.prompt.splitlines()
                    if '"id": "c-' in line
                ]
                items = [
                    {
                        "id": ids[0],
                        "outcome": "addressed",
                        "reason": "added calibration",
                        "sections": ["thesis-p1"],
                    }
                ] + [{"id": i, "outcome": "rejected", "reason": "not in corpus"} for i in ids[1:]]
            (request.cwd / "receipt.json").write_text(
                json.dumps({"summary": "s", "items": items, "changed_sections": ["thesis-p1"]})
            )
        return AgentResult(text="ok", num_turns=2)


def seeded_guide(tmp_path: Path):
    from src.guides.comments import add_comment

    writer, _ = make_writer(tmp_path, FakeAgent())
    writer.write(
        topic="t",
        title="T",
        video_ids=["v1", "v2", "v3"],
        videos_meta=VIDEOS,
        compiled_at="2026-09-14",
    )
    first = add_comment(
        writer.paths,
        body="mention calibration",
        anchor="thesis-p1",
        section_id="thesis",
        quote="drift",
    )
    second = add_comment(writer.paths, body="add pricing", anchor="thesis-p1")
    return writer.paths, first, second


def test_revise_applies_comments_as_a_tracked_batch(tmp_path: Path):
    from src.guides.comments import latest_comments

    paths, first, second = seeded_guide(tmp_path)
    agent = FakeReviser()
    writer, events = make_writer(tmp_path, agent)
    manifest = writer.revise()
    assert [r.label for r in agent.requests] == ["revise"]
    request = agent.requests[0]
    assert request.model == "claude-opus-5" and "Edit" in request.allowed_tools
    assert (
        first["id"] in request.prompt
        and "mention calibration" in request.prompt
        and "version 2" in request.prompt
    )
    assert (
        manifest.current_version == 2
        and paths.version_html(2).is_file()
        and paths.version_html(1).is_file()
    )
    assert manifest.cite_total == 3 and manifest.cite_valid == 3
    assert manifest.chunk_count == 3  # provenance preserved across a revision
    statuses = {c["id"]: c["status"] for c in latest_comments(paths)}
    assert statuses == {first["id"]: "addressed", second["id"]: "rejected"}
    assert json.loads(paths.receipt(2).read_text())["items"][1]["reason"] == "not in corpus"
    assert not (paths.dir / "receipt.json").exists()
    assert ("revise", "done") in [(e["stage"], e["status"]) for e in events]


def test_revise_fails_closed_on_an_incomplete_receipt(tmp_path: Path):
    from src.guides.comments import open_comments

    paths, first, _second = seeded_guide(tmp_path)
    agent = FakeReviser(receipt_items=[{"id": first["id"], "outcome": "addressed"}])
    writer, _ = make_writer(tmp_path, agent)
    with pytest.raises(RuntimeError, match="missing comment ids"):
        writer.revise()
    assert read_manifest(paths).current_version == 1
    assert not paths.version_html(2).exists()
    assert len(open_comments(paths)) == 2  # nothing resolved


def test_revise_needs_open_comments(tmp_path: Path):
    writer, _ = make_writer(tmp_path, FakeAgent())
    writer.write(topic="t", title="T", video_ids=["v1"], videos_meta=VIDEOS)
    with pytest.raises(RuntimeError, match="no open comments"):
        make_writer(tmp_path, FakeReviser())[0].revise()


def test_rate_limit_is_retried_once_after_a_pause(tmp_path: Path, monkeypatch):
    import asyncio as _asyncio

    class RateLimitedOnce(FakeAgent):
        def __init__(self):
            super().__init__()
            self.limited = False

        async def __call__(self, request, retrieve, on_event):
            if request.label == "compose" and not self.limited:
                self.limited = True
                self.requests.append(request)
                return AgentResult(error="rate_limit")
            return await super().__call__(request, retrieve, on_event)

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(_asyncio, "sleep", fake_sleep)
    agent = RateLimitedOnce()
    writer, events = make_writer(tmp_path, agent, rate_limit_pause_seconds=7)
    manifest = writer.write(topic="t", title="T", video_ids=["v1"], videos_meta=VIDEOS)
    assert slept == [7]
    assert [r.label for r in agent.requests] == ["extract:cluster-1", "compose", "compose"]
    assert manifest.current_version == 1
    assert any(e["status"] == "progress" and "rate limited" in e["message"] for e in events)


def test_stage_timeout_aborts_the_run(tmp_path: Path):
    import asyncio as _asyncio

    class Hangs(FakeAgent):
        async def __call__(self, request, retrieve, on_event):
            await _asyncio.sleep(5)
            return AgentResult()

    writer, _ = make_writer(tmp_path, Hangs(), stage_timeout_seconds=0.05)
    with pytest.raises(RuntimeError, match="no result after"):
        writer.write(topic="t", title="T", video_ids=["v1"], videos_meta=VIDEOS)


def test_confinement_hook_denies_a_glob_pattern_that_escapes_the_guide_dir(tmp_path: Path):
    import asyncio as _asyncio

    root = tmp_path / "guides" / "slug"
    root.mkdir(parents=True)
    hook = make_confinement_hook(root)

    async def run(tool_input):
        return await hook(
            {"tool_name": "Glob", "tool_input": tool_input}, "tool-use-1", None
        )

    escaping = _asyncio.run(run({"pattern": "/etc/*"}))
    assert escaping["hookSpecificOutput"]["permissionDecision"] == "deny"

    traversal = _asyncio.run(run({"pattern": "../../../secrets/**"}))
    assert traversal["hookSpecificOutput"]["permissionDecision"] == "deny"

    inside = _asyncio.run(run({"pattern": "corpus/*.md"}))
    assert inside == {}


def test_sdk_env_blanks_the_api_key_only_when_the_oauth_token_covers_the_run(monkeypatch, caplog):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live")
    assert sdk_env() == {"ANTHROPIC_API_KEY": ""}

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with caplog.at_level("WARNING"):
        env = sdk_env()
    assert env == {}
    assert any("bill" in record.message for record in caplog.records)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert sdk_env() == {}


def test_extract_regenerates_evidence_when_the_cluster_scope_changes(tmp_path: Path):
    agent = FakeAgent()
    writer, _ = make_writer(tmp_path, agent)
    writer.write(topic="t", title="T", video_ids=["v1", "v2"], videos_meta=VIDEOS)

    rescoped = FakeAgent()
    writer2, events = make_writer(tmp_path, rescoped)
    writer2.write(topic="t", title="T", video_ids=["v1", "v2", "v3"], videos_meta=VIDEOS)
    # The cluster now covers a third video, so the stale evidence is redone.
    assert [r.label for r in rescoped.requests if r.label.startswith("extract:")]
    assert ("extract", "skip") not in [(e["stage"], e["status"]) for e in events]


def test_publish_refuses_to_overwrite_an_existing_version(tmp_path: Path):
    agent = FakeAgent()
    writer, _ = make_writer(tmp_path, agent)
    writer.write(topic="t", title="T", video_ids=["v1"], videos_meta=VIDEOS)
    from src.guides.export import load_export
    from src.guides.catalog import read_manifest as _read_manifest

    manifest = _read_manifest(writer.paths)
    summary = load_export(writer.paths)
    with pytest.raises(RuntimeError, match="already exists"):
        writer.stage_publish(
            topic="t",
            title="T",
            summary=summary,
            report=type("R", (), {"total": 0, "valid": 0})(),
            compiled_at="2026-09-14",
            version=1,
            manifest=manifest,
        )
