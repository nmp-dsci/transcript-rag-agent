from __future__ import annotations

from pathlib import Path

from src.guides.catalog import GuidePaths
from src.guides.export import export_corpus, load_export, render_video_markdown
from src.guides.scope import (
    candidate_videos,
    cluster_videos,
    probe_questions,
    probes_for,
    title_matches,
    topic_from_question,
    topic_tokens,
)

VIDEOS = [
    {"video_id": "j1", "title": "LLM-as-a-Judge 101", "channel_name": "A", "chunk_count": 40},
    {
        "video_id": "j2",
        "title": "Judge the Judge: evaluators that work",
        "channel_name": "B",
        "chunk_count": 55,
    },
    {
        "video_id": "p1",
        "title": "Production GenAI playbook",
        "channel_name": "C",
        "chunk_count": 30,
    },
    {"video_id": "x1", "title": "Resume tips", "channel_name": "D", "chunk_count": 20},
]

HITS = {
    "j1": 6,  # surfaces for six of the eight probes
    "j2": 3,
    "p1": 1,
}


def retrieve(question: str, top_k: int):
    # Each video appears in the results of its first N probes.
    index = probe_questions("LLM-as-a-judge").index(question)
    return [{"video_id": vid} for vid, n in HITS.items() if index < n] * 2  # duplicates ignored


def test_tokens_probes_and_title_match():
    assert topic_tokens("LLM-as-a-judge for RAG") == ["llm-as-a-judge", "llm", "judge", "rag"]
    assert len(probe_questions("  LLM-as-a-judge ")) == 8
    assert probe_questions("x")[0] == "x"
    assert title_matches("LLM-as-a-judge", "LLM-as-a-Judge 101")
    assert (
        title_matches("production generative AI systems", "GenAI in production: RAG at scale")
        is False
    )
    assert title_matches(
        "production generative AI systems", "Scalable generative AI systems in production"
    )
    assert title_matches("LLM-as-a-judge", "LLM as a Judge Explained | Hands-On")
    assert title_matches("anything", None) is False


def test_topic_from_question_keeps_the_subject_phrase():
    cases = {
        "How do teams calibrate an LLM judge against human labels, and what do they do when it drifts?": "calibrate an llm judge against human labels",
        "What is the best way to do prompt caching?": "prompt caching",
        "prompt caching": "prompt caching",
        "Explain LLM-as-a-judge": "llm-as-a-judge",
        "How should I structure a RAG eval so the numbers mean something": "structure a rag eval so the numbers mean something",
        "What do recruiters actually look for on an AI engineer resume?": "recruiters actually look for on an ai engineer resume",
        "why is context engineering hard and why": "context engineering hard",
        "???": "???",
    }
    for question, topic in cases.items():
        assert topic_from_question(question) == topic, question


def test_probes_for_a_question_lead_with_the_question_itself():
    question = "How do teams calibrate an LLM judge against human labels?"
    probes = probes_for(question)
    assert probes[0] == question
    assert probes[1:] == probe_questions("calibrate an llm judge against human labels")
    # A bare topic is not duplicated as its own probe.
    assert probes_for("prompt caching") == probe_questions("prompt caching")


def test_candidates_ranked_by_probe_hits_and_title():
    ranked = candidate_videos("LLM-as-a-judge", VIDEOS, retrieve)
    assert [c.video_id for c in ranked] == ["j1", "j2", "p1"]
    first = ranked[0]
    assert first.probe_hits == 6 and first.title_match and first.score == 8
    assert len(first.probes) == 6
    assert ranked[2].score == 1 and ranked[2].title_match is False
    # A video no probe surfaced and whose title says nothing is not a candidate.
    assert all(c.video_id != "x1" for c in ranked)
    assert ranked[0].to_dict()["score"] == 8


def test_clusters_keep_channels_together_and_cap_count():
    videos = [
        {"video_id": f"v{i}", "channel_name": "chan-" + ("a" if i < 5 else "b"), "title": f"t{i}"}
        for i in range(10)
    ]
    clusters = cluster_videos(videos, max_clusters=6, min_per_cluster=3)
    assert len(clusters) == 4
    assert sum(len(c) for c in clusters) == 10
    assert clusters[0] == ["v0", "v1", "v2"]
    assert cluster_videos([]) == []
    assert cluster_videos(videos[:2]) == [["v0", "v1"]]


class Chunk:
    def __init__(self, video_id: str, index: int, text: str) -> None:
        self.video_id = video_id
        self.chunk_index = index
        self.text = text
        self.start_seconds = index * 60.0
        self.end_seconds = index * 60.0 + 59
        self.title = f"Title {video_id}"
        self.channel_name = "Chan"
        self.source_url = f"https://www.youtube.com/watch?v={video_id}"


def test_export_writes_one_file_per_video(tmp_path: Path):
    paths = GuidePaths(tmp_path, "g")
    chunks = [Chunk("a", 1, "second"), Chunk("a", 0, "first"), Chunk("b", 0, "only")]
    summary = export_corpus(
        paths, ["a", "b", "missing"], lambda ids: chunks, clusters=[["a"], ["b", "missing"]]
    )
    assert summary.chunk_count == 3
    text = (paths.corpus / "a.md").read_text()
    assert text.startswith("# Title a\n")
    assert "## a@0 · 00:00–00:59\n\nfirst\n" in text
    assert text.index("a@0") < text.index("a@1")
    assert (paths.corpus / "missing.md").read_text().startswith("# missing\n")
    index = (paths.corpus / "INDEX.md").read_text()
    assert "## Cluster 1" in index and "`corpus/a.md` — Title a (Chan, 2 chunks)" in index
    loaded = load_export(paths)
    assert loaded is not None and loaded.clusters == [["a"], ["b", "missing"]]
    assert [v.chunk_count for v in loaded.videos] == [2, 1, 0]
    assert render_video_markdown("z", []).startswith("# z\n")
