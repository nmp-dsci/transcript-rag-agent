"""The summary comes from the creator's description, not an LLM.

Supadata returns ``description`` in the same ``/v1/metadata`` call the fetch
step already makes, so this path costs nothing and — the point — has no
provider that can return 402 mid-index.
"""

from __future__ import annotations

import pytest

from src.rag.models import RawTranscriptDocument
from src.rag.summaries import DescriptionSummaryGenerator, clean_description


def _doc(description: str | None, title: str | None = "A fixture title") -> RawTranscriptDocument:
    return RawTranscriptDocument(
        transcript_id="t",
        video_id="vid",
        source_url="https://www.youtube.com/watch?v=vid",
        fetched_at="2026-08-25T00:00:00Z",
        title=title,
        description=description,
    )


class TestCleanDescription:
    def test_strips_links_timestamps_and_handles(self) -> None:
        raw = (
            "Support me: https://patreon.com/x and www.example.com\n"
            "0:00 Intro 12:34 The demo 1:02:03 Outro\n"
            "@mychannel #machinelearning\n"
            "Backpropagation explained from first principles."
        )
        cleaned = clean_description(raw)
        assert "patreon" not in cleaned
        assert "example.com" not in cleaned
        assert "12:34" not in cleaned
        assert "@mychannel" not in cleaned
        assert "Backpropagation explained from first principles." in cleaned

    def test_collapses_whitespace(self) -> None:
        assert clean_description("a\n\n  b\t c") == "a b c"

    def test_a_description_of_only_links_cleans_to_nothing(self) -> None:
        assert clean_description("https://a.com https://b.com @me") == ""


class TestDescriptionSummaryGenerator:
    def test_uses_the_description_and_leads_with_the_title(self) -> None:
        body = (
            "What the neurons are, why there are layers, and the math underlying "
            "the whole thing, built up from first principles."
        )
        summary = DescriptionSummaryGenerator().summarize(
            _doc(body, title="But what is a neural network?")
        )
        assert summary.startswith("But what is a neural network?")
        assert "why there are layers" in summary

    def test_makes_no_llm_call(self) -> None:
        """No client, no key, no network — the generator takes none."""
        generator = DescriptionSummaryGenerator()
        assert not hasattr(generator, "llm")
        assert generator.source == "description"
        assert generator.model_name == "youtube-description"

    def test_a_description_too_thin_to_route_on_raises(self) -> None:
        """The 'Work with me:' case — better recorded as failed than indexed."""
        with pytest.raises(ValueError, match="too thin to route on"):
            DescriptionSummaryGenerator().summarize(
                _doc("Work with me: https://example.com", title="Short")
            )

    def test_a_missing_description_raises_rather_than_returning_empty(self) -> None:
        with pytest.raises(ValueError, match="too thin to route on"):
            DescriptionSummaryGenerator().summarize(_doc(None, title="Short"))

    def test_the_threshold_is_tunable(self) -> None:
        doc = _doc("Short but real prose.", title="T")
        with pytest.raises(ValueError):
            DescriptionSummaryGenerator(min_chars=500).summarize(doc)
        assert DescriptionSummaryGenerator(min_chars=5).summarize(doc)
