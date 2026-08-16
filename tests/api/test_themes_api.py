"""/api/themes and /api/themes/{id}, plus the corpus-side readers behind them."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.corpus import list_themes, theme_detail
from src.api.main import create_app
from src.config import Settings
from src.rag.themes import (
    Theme,
    ThemeIndex,
    ThemeMember,
    ThemeStore,
    ThemeVideo,
)


def _index() -> ThemeIndex:
    return ThemeIndex(
        generated_at="2026-08-10T00:00:00+00:00",
        embedding_model="all-MiniLM-L6-v2",
        summary_model="deepseek-v4-flash",
        chunk_collection="transcript_chunks",
        stats={"themes": 1, "cross_video_themes": 1},
        themes=[
            Theme(
                theme_id="theme:0",
                title="Tailor resumes for ATS with quantified impact",
                summary="Eleven creators start from the posting, not the history.",
                member_count=3,
                video_count=2,
                channel_count=2,
                cross_video=True,
                domain="job_search",
                domain_mix={"job_search": 1.0},
                videos=[
                    ThemeVideo(
                        video_id="v1",
                        title="Winning tech resume",
                        channel_name="Anthony D. Mays",
                        member_count=2,
                        domain="job_search",
                    ),
                    ThemeVideo(
                        video_id="v2",
                        title="Dev resume",
                        channel_name="Anthony Sistilli",
                        member_count=1,
                        domain="job_search",
                    ),
                ],
                members=[
                    ThemeMember(
                        chunk_id="chunk:v1:1", video_id="v1", chunk_index=1, probability=0.9
                    ),
                    ThemeMember(
                        chunk_id="chunk:v1:0", video_id="v1", chunk_index=0, probability=0.8
                    ),
                    ThemeMember(
                        chunk_id="chunk:v2:5", video_id="v2", chunk_index=5, probability=0.7
                    ),
                ],
            )
        ],
    )


@pytest.fixture
def theme_path(tmp_path: Path) -> Path:
    path = tmp_path / "themes.json"
    ThemeStore(path).save(_index())
    return path


@pytest.fixture
def chunk_chroma(tmp_path: Path) -> Path:
    """A real Chroma collection, because hydration is the thing being tested."""
    import chromadb

    path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection("transcript_chunks")
    collection.upsert(
        ids=["chunk:v1:0", "chunk:v1:1", "chunk:v2:5"],
        documents=["first chunk text", "second chunk text", "other creator text"],
        embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        metadatas=[
            {
                "video_id": "v1",
                "chunk_index": 0,
                "start_seconds": 10.0,
                "end_seconds": 40.0,
                "source_url": "https://youtu.be/v1",
            },
            {
                "video_id": "v1",
                "chunk_index": 1,
                "start_seconds": 41.0,
                "end_seconds": 80.0,
                "source_url": "https://youtu.be/v1",
            },
            {
                "video_id": "v2",
                "chunk_index": 5,
                "start_seconds": 200.0,
                "end_seconds": 260.0,
                "source_url": "https://youtu.be/v2",
            },
        ],
    )
    return path


def test_list_themes_drops_members_but_keeps_the_counts(theme_path: Path) -> None:
    payload = list_themes(theme_path)
    assert payload["themes"][0]["members"] == []
    assert payload["themes"][0]["video_count"] == 2
    assert payload["themes"][0]["videos"][0]["channel_name"] == "Anthony D. Mays"
    assert payload["summary_model"] == "deepseek-v4-flash"


def test_list_themes_without_an_artifact_names_the_build_command(tmp_path: Path) -> None:
    payload = list_themes(tmp_path / "missing.json")
    assert payload["themes"] == []
    assert payload["build_command"] == "uv run python -m src.cli index-themes"


def test_theme_detail_groups_members_by_video_with_text_and_timestamps(
    theme_path: Path, chunk_chroma: Path
) -> None:
    detail = theme_detail(theme_path, chunk_chroma, "theme:0")
    assert detail is not None
    assert [group["video_id"] for group in detail["videos"]] == ["v1", "v2"]
    first = detail["videos"][0]["chunks"]
    # Chunk order inside a video is the transcript's, not the cluster's.
    assert [chunk["chunk_index"] for chunk in first] == [0, 1]
    assert first[0]["text"] == "first chunk text"
    assert first[0]["start_seconds"] == 10.0
    assert first[0]["source_url"] == "https://youtu.be/v1"
    assert detail["videos"][1]["chunks"][0]["text"] == "other creator text"


def test_theme_detail_is_none_for_an_unknown_theme(theme_path: Path, chunk_chroma: Path) -> None:
    assert theme_detail(theme_path, chunk_chroma, "theme:404") is None
    assert theme_detail(Path("/nowhere/themes.json"), chunk_chroma, "theme:0") is None


def test_theme_detail_survives_a_missing_chunk_collection(theme_path: Path, tmp_path: Path) -> None:
    """The artifact holds ids, the store holds text — a store that cannot be
    read leaves the structure intact rather than 500ing the whole view."""
    detail = theme_detail(theme_path, tmp_path / "empty-chroma", "theme:0")
    assert detail is not None
    assert detail["videos"][0]["chunks"][0]["text"] == ""


def _client(theme_path: Path, chroma: Path, tmp_path: Path) -> TestClient:
    settings = Settings(
        superdata_api_key="s",
        deepseek_api_key="d",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=chroma,
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-themes",
        log_transcript_artifacts=False,
        theme_path=theme_path,
    )
    return TestClient(create_app(settings))


def test_themes_endpoints_serve_the_artifact(
    theme_path: Path, chunk_chroma: Path, tmp_path: Path
) -> None:
    client = _client(theme_path, chunk_chroma, tmp_path)

    listing = client.get("/api/themes")
    assert listing.status_code == 200
    assert listing.json()["themes"][0]["theme_id"] == "theme:0"

    # Theme ids contain a colon, so the route has to accept it as a path.
    detail = client.get("/api/themes/theme:0")
    assert detail.status_code == 200
    assert detail.json()["theme"]["title"].startswith("Tailor resumes")
    assert len(detail.json()["videos"]) == 2

    assert client.get("/api/themes/theme:99").status_code == 404
