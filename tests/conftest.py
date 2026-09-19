from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.transcripts import supadata_keys
from src.transcripts.models import Transcript


@pytest.fixture(autouse=True)
def _fresh_supadata_rings() -> None:
    # Key rings are process-wide so ingestion workers share what they learn;
    # a key one test exhausts must not stay exhausted for the next.
    supadata_keys.reset_rings()


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        video_id="3hk7nO_q0a8",
        url="https://www.youtube.com/watch?v=3hk7nO_q0a8",
        title="Sample",
        language="en",
        provider="supadata",
        raw_text="This transcript explains three practical findings about agent systems.",
        fetched_at=datetime(2026, 5, 14, tzinfo=timezone.utc),
    )
