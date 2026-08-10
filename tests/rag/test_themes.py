"""RAPTOR level 2: clustering, domain tagging, excerpt choice and the store.

No embedding model and no LLM — the clustering runs on synthetic vectors and
the summarizer is a stub, because everything worth asserting here (do clusters
cross video boundaries, is the same corpus clustered the same way twice) is
settled before any model is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.themes import (
    Theme,
    ThemeClusterConfig,
    ThemeIndex,
    ThemeMember,
    ThemeStore,
    ThemeSummarizer,
    ThemeVideo,
    build_intruder_probe,
    build_themes,
    cluster_embeddings,
    format_excerpts,
    representative_excerpts,
    sort_records,
    theme_statistics,
    video_domain,
)


def _blob(centre: list[float], count: int, jitter: float, seed: int) -> list[list[float]]:
    """Points around one centre, deterministic and free of numpy in the test."""
    import random

    rng = random.Random(seed)
    return [[value + rng.uniform(-jitter, jitter) for value in centre] for _ in range(count)]


def _records(groups: list[tuple[str, list[list[float]]]]) -> list[dict]:
    records = []
    for video_id, vectors in groups:
        for index, vector in enumerate(vectors):
            records.append(
                {
                    "chunk_id": f"chunk:{video_id}:{index}",
                    "video_id": video_id,
                    "chunk_index": index,
                    "title": f"{video_id} title",
                    "channel_name": f"{video_id} channel",
                    "text": f"{video_id} chunk {index}",
                    "embedding": vector,
                }
            )
    return records


class StubSummarizer(ThemeSummarizer):
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.model_name = "stub-model"

    def summarize(self, excerpts):  # type: ignore[override]
        self.calls.append(excerpts)
        return f"Theme {len(self.calls)}", "A shared claim."


# ─── Clustering ──────────────────────────────────────────────────────────────


def _two_topic_corpus() -> list[dict]:
    """Two topics, each spoken about by two different videos.

    This is the shape the hypothesis is about: no single video holds a topic,
    so any cluster that recovers one has to cross a video boundary.
    """
    topic_a = [1.0, 0.0, 0.0, 0.0]
    topic_b = [0.0, 1.0, 0.0, 0.0]
    return _records(
        [
            ("vidA1", _blob(topic_a, 20, 0.05, 1)),
            ("vidA2", _blob(topic_a, 20, 0.05, 2)),
            ("vidB1", _blob(topic_b, 20, 0.05, 3)),
            ("vidB2", _blob(topic_b, 20, 0.05, 4)),
        ]
    )


def test_clusters_cross_video_boundaries_when_the_topic_does() -> None:
    records = sort_records(_two_topic_corpus())
    clusters = cluster_embeddings(
        [record["embedding"] for record in records],
        ThemeClusterConfig(global_dims=3, local_dims=2, target_cluster_size=40),
    )
    assert clusters
    spans = [len({records[row]["video_id"] for row, _ in cluster}) for cluster in clusters]
    assert max(spans) >= 2, "a topic split across two videos must cluster across them"


def test_clustering_is_reproducible_for_the_same_corpus() -> None:
    records = sort_records(_two_topic_corpus())
    vectors = [record["embedding"] for record in records]
    config = ThemeClusterConfig(global_dims=3, local_dims=2, target_cluster_size=40)
    first = cluster_embeddings(vectors, config)
    second = cluster_embeddings(vectors, config)
    assert first == second


def test_clustering_an_empty_corpus_returns_nothing() -> None:
    assert cluster_embeddings([]) == []


def test_sort_records_fixes_the_order_chroma_does_not_promise() -> None:
    records = [
        {"video_id": "b", "chunk_index": 1},
        {"video_id": "a", "chunk_index": 2},
        {"video_id": "a", "chunk_index": 0},
    ]
    assert [(record["video_id"], record["chunk_index"]) for record in sort_records(records)] == [
        ("a", 0),
        ("a", 2),
        ("b", 1),
    ]


# ─── Build ───────────────────────────────────────────────────────────────────


def test_build_themes_records_cross_video_reach_and_calls_the_llm_once_per_theme() -> None:
    summarizer = StubSummarizer()
    index = build_themes(
        _two_topic_corpus(),
        summarizer,
        config=ThemeClusterConfig(global_dims=3, local_dims=2, target_cluster_size=40),
        embedding_model="test-embed",
        chunk_collection="chunks",
    )
    assert index.themes
    assert len(summarizer.calls) == len(index.themes)
    assert index.summary_model == "stub-model"
    assert index.stats["cross_video_themes"] >= 1
    assert index.stats["chunks_clustered"] == 80
    cross = next(theme for theme in index.themes if theme.cross_video)
    assert cross.video_count >= 2
    assert sum(video.member_count for video in cross.videos) == cross.member_count


def test_build_themes_without_a_summarizer_still_produces_the_gate_numbers() -> None:
    index = build_themes(
        _two_topic_corpus(),
        None,
        config=ThemeClusterConfig(global_dims=3, local_dims=2, target_cluster_size=40),
    )
    assert index.themes
    assert index.summary_model == ""
    assert all(theme.summary == "" for theme in index.themes)
    assert index.stats["themes"] == len(index.themes)


def test_cross_video_themes_sort_ahead_of_single_video_ones() -> None:
    """A single-video cluster is a per-video summary in disguise, so it never
    heads the list even when it is the biggest thing in the corpus."""
    records = _records(
        [
            ("shared1", _blob([1.0, 0.0, 0.0, 0.0], 15, 0.05, 5)),
            ("shared2", _blob([1.0, 0.0, 0.0, 0.0], 15, 0.05, 6)),
            ("alone", _blob([0.0, 0.0, 1.0, 0.0], 40, 0.02, 7)),
        ]
    )
    index = build_themes(
        records,
        None,
        config=ThemeClusterConfig(global_dims=3, local_dims=2, target_cluster_size=40),
    )
    flags = [theme.cross_video for theme in index.themes]
    assert flags == sorted(flags, reverse=True)


# ─── Domain tagging and purity ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("title", "channel", "expected"),
    [
        ("The strategy to win after this budget", "Smart Property Investment", "property"),
        (
            "Ep. 55: What's Happening Across Australia's Property Markets?",
            "Suburb Data",
            "property",
        ),
        # "interview" alone would make these job-search videos; counting terms
        # rather than taking the first hit is what keeps them architectural.
        (
            "Sharding in System Design Interviews w/ Meta Staff Engineer",
            "Hello Interview",
            "system_design",
        ),
        (
            "System Design Interview - Distributed Message Queue",
            "System Design Interview",
            "system_design",
        ),
        (
            "This Simple RESUME got me 5 Machine Learning Interviews",
            "Boris Meinardus",
            "job_search",
        ),
        ("How to Write a Winning Tech Resume", "Anthony D. Mays", "job_search"),
        (
            "Exploring MLOps and LLMOps: Architectures and Best Practices",
            "Databricks",
            "ai_engineering",
        ),
        ("Building beautiful UI using AI (My design workflow)", "Ras Mic", "ui_design"),
        ("A talk about nothing in particular", "Somebody", "other"),
    ],
)
def test_video_domain(title: str, channel: str, expected: str) -> None:
    assert video_domain(title, channel) == expected


def test_theme_statistics_flags_a_job_search_theme_polluted_with_property() -> None:
    def theme(theme_id: str, domain: str, share: float) -> Theme:
        return Theme(
            theme_id=theme_id,
            title=theme_id,
            summary="",
            member_count=10,
            video_count=3,
            channel_count=3,
            cross_video=True,
            domain=domain,
            property_share=share,
        )

    stats = theme_statistics(
        [
            theme("theme:0", "job_search", 0.05),
            theme("theme:1", "job_search", 0.44),
            theme("theme:2", "property", 1.0),
        ]
    )
    assert stats["impure_job_search_themes"] == ["theme:1"]
    assert stats["max_property_share_in_job_search_theme"] == 0.44
    assert stats["cross_video_themes"] == 3
    assert stats["video_count_distribution"] == {"3": 3}


# ─── Excerpt selection ───────────────────────────────────────────────────────


def test_representative_excerpts_spread_across_videos_before_going_deep() -> None:
    """Handing the summarizer the top-probability chunks would hand it one
    video, and it would then write that video's outline."""
    records = [
        {"video_id": "loud", "chunk_id": f"chunk:loud:{i}", "chunk_index": i, "text": "x"}
        for i in range(10)
    ] + [
        {"video_id": "quiet", "chunk_id": "chunk:quiet:0", "chunk_index": 0, "text": "y"},
        {"video_id": "faint", "chunk_id": "chunk:faint:0", "chunk_index": 0, "text": "z"},
    ]
    members = [(index, 0.99 - index * 0.01) for index in range(10)]
    members += [(10, 0.4), (11, 0.3)]
    picked = representative_excerpts(members, records, limit=3)
    assert {record["video_id"] for record in picked} == {"loud", "quiet", "faint"}


def test_format_excerpts_labels_every_excerpt_with_its_creator() -> None:
    block = format_excerpts(
        [
            {"channel_name": "Creator One", "title": "Video A", "text": "first"},
            {"channel_name": "Creator Two", "title": "Video B", "text": "second"},
        ]
    )
    assert "[Creator One — Video A]" in block
    assert "[Creator Two — Video B]" in block


def test_format_excerpts_truncates_long_text() -> None:
    block = format_excerpts([{"text": "a" * 2000}], max_chars=50)
    assert len(block) < 200
    assert block.endswith("…")


# ─── Summarizer ──────────────────────────────────────────────────────────────


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content


class _Llm:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list = []

    def invoke(self, messages):
        self.messages = messages
        return _Reply(self.content)


def test_summarizer_parses_fenced_json() -> None:
    llm = _Llm('```json\n{"title":"A shared claim","summary":"Two creators agree."}\n```')
    title, summary = ThemeSummarizer(llm, "model").summarize([{"text": "hello"}])
    assert title == "A shared claim"
    assert summary == "Two creators agree."


def test_summarizer_rejects_an_empty_summary() -> None:
    with pytest.raises(ValueError):
        ThemeSummarizer(_Llm('{"title":"x","summary":""}'), "model").summarize([{"text": "a"}])


def test_summarizer_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        ThemeSummarizer(_Llm("no json here"), "model").summarize([{"text": "a"}])


# ─── Store ───────────────────────────────────────────────────────────────────


def _index() -> ThemeIndex:
    return ThemeIndex(
        generated_at="2026-01-01T00:00:00+00:00",
        embedding_model="embed",
        summary_model="model",
        chunk_collection="chunks",
        themes=[
            Theme(
                theme_id="theme:0",
                title="A claim several creators make",
                summary="Summary.",
                member_count=2,
                video_count=2,
                channel_count=2,
                cross_video=True,
                domain="job_search",
                videos=[
                    ThemeVideo(video_id="v1", member_count=1),
                    ThemeVideo(video_id="v2", member_count=1),
                ],
                members=[
                    ThemeMember(
                        chunk_id="chunk:v1:0", video_id="v1", chunk_index=0, probability=0.9
                    ),
                    ThemeMember(
                        chunk_id="chunk:v2:3", video_id="v2", chunk_index=3, probability=0.8
                    ),
                ],
            )
        ],
    )


def test_store_round_trips(tmp_path: Path) -> None:
    store = ThemeStore(tmp_path / "nested" / "themes.json")
    assert store.load() is None
    assert store.exists() is False
    store.save(_index())
    assert store.exists() is True
    loaded = store.load()
    assert loaded is not None
    assert loaded.themes[0].title == "A claim several creators make"
    assert loaded.themes[0].members[0].chunk_id == "chunk:v1:0"


def test_store_writes_readable_json(tmp_path: Path) -> None:
    path = ThemeStore(tmp_path / "themes.json").save(_index())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["themes"][0]["cross_video"] is True


# ─── Intruder probe ──────────────────────────────────────────────────────────


def test_intruder_probe_plants_exactly_one_outsider() -> None:
    index = build_themes(
        _two_topic_corpus(),
        None,
        config=ThemeClusterConfig(global_dims=3, local_dims=2, target_cluster_size=40),
    )
    probe = build_intruder_probe(index.themes, 0, options=5, seed=7)
    assert len(probe["chunk_ids"]) == 5
    assert len(set(probe["chunk_ids"])) == 5
    assert probe["intruder_chunk_id"] in probe["chunk_ids"]
    own = {member.chunk_id for member in index.themes[0].members}
    assert probe["intruder_chunk_id"] not in own
    assert sum(1 for chunk_id in probe["chunk_ids"] if chunk_id not in own) == 1


def test_intruder_probe_is_seeded() -> None:
    index = build_themes(
        _two_topic_corpus(),
        None,
        config=ThemeClusterConfig(global_dims=3, local_dims=2, target_cluster_size=40),
    )
    assert build_intruder_probe(index.themes, 0, seed=3) == build_intruder_probe(
        index.themes, 0, seed=3
    )


def test_intruder_probe_needs_another_theme_to_draw_from() -> None:
    single = _index().themes
    with pytest.raises(ValueError):
        build_intruder_probe(single, 0, options=5)
