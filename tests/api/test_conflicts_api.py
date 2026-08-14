"""/api/conflicts and the corpus-side reader behind it."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.corpus import list_conflicts
from src.api.main import create_app
from src.config import Settings
from src.rag.conflicts import Conflict, ConflictIndex, ConflictStore, Side


def _side(video_id: str, creator: str, position: str, quote: str) -> Side:
    return Side(
        video_id=video_id,
        chunk_id=f"chunk:{video_id}:5",
        channel_name=creator,
        title=f"{creator} on resumes",
        start_seconds=310.6,
        end_seconds=382.7,
        position=position,
        quote=quote,
        quote_ratio=1.0,
        watch_url=f"https://www.youtube.com/watch?v={video_id}&t=310",
    )


@pytest.fixture()
def conflict_path(tmp_path: Path) -> Path:
    index = ConflictIndex(
        generated_at="2026-08-10T04:50:58+00:00",
        adjudicator_model="deepseek-v4-flash",
        embedding_model="all-MiniLM-L6-v2",
        stats={
            "conflicts": 1,
            "candidates_adjudicated": 478,
            "conflict_precision": 0.0021,
            "adjudicate_repeats": 3,
            "verdict_agreement": 0.9,
        },
        probes=[{"probe_id": "planted-flat", "expect": "conflict", "passed": True}],
        conflicts=[
            Conflict(
                conflict_id="conflict:0",
                axis="What font size should the body text of a resume be?",
                why_incompatible="11-12 point and 10 point are different sizes for the same text.",
                left=_side("vidA", "Jean Lee", "11 to 12 points", "around 11 to 12 points"),
                right=_side("vidB", "Greg Langstaff", "10 point", "going down to 10point font"),
                similarity=0.7,
                cross_channel=True,
                kind="axis",
                votes=3,
                repeats=3,
            )
        ],
    )
    path = tmp_path / "conflicts.json"
    ConflictStore(path).save(index)
    return path


def test_the_payload_carries_both_sides_and_no_verdict(conflict_path: Path) -> None:
    """Nothing in the wire format can express "the corpus says X"."""
    payload = list_conflicts(conflict_path)
    conflict = payload["conflicts"][0]
    assert conflict["left"]["channel_name"] == "Jean Lee"
    assert conflict["right"]["channel_name"] == "Greg Langstaff"
    assert "winner" not in conflict
    assert "verdict" not in conflict


def test_the_probes_travel_with_the_conflicts(conflict_path: Path) -> None:
    """A count is only worth reading beside the evidence its judge was calibrated."""
    payload = list_conflicts(conflict_path)
    assert payload["probes"][0]["probe_id"] == "planted-flat"
    assert payload["stats"]["candidates_adjudicated"] == 478
    # The vote reaches the wire: a 2/3 card and a 3/3 card are different claims
    # and the UI cannot distinguish them unless the payload does.
    assert payload["conflicts"][0]["votes"] == 3
    assert payload["conflicts"][0]["repeats"] == 3
    assert payload["stats"]["verdict_agreement"] == 0.9


def test_a_missing_artifact_names_the_build_command(tmp_path: Path) -> None:
    payload = list_conflicts(tmp_path / "missing.json")
    assert payload["conflicts"] == []
    assert payload["build_command"] == "uv run python -m src.cli index-conflicts"


def test_the_endpoint_serves_the_artifact(conflict_path: Path, tmp_path: Path) -> None:
    settings = Settings(
        superdata_api_key="s",
        deepseek_api_key="d",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-conflicts",
        log_transcript_artifacts=False,
        conflict_path=conflict_path,
    )
    response = TestClient(create_app(settings)).get("/api/conflicts")
    assert response.status_code == 200
    assert response.json()["conflicts"][0]["axis"].endswith("?")
