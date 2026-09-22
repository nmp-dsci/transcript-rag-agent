from __future__ import annotations

import pytest

from src.channels.robots import RobotsPolicy

PERMISSIVE_ROBOTS = "User-agent: *\nAllow: /\n"

#: The shape one registered source actually serves: article paths open to
#: ordinary crawlers, closed to every named AI crawler.
AI_BLOCKING_ROBOTS = """User-agent: GPTBot
Disallow: /

User-agent: ClaudeBot
Disallow: /

User-agent: CCBot
Disallow: /

User-agent: *
Disallow: /subscribe
"""


@pytest.fixture
def permissive_robots() -> RobotsPolicy:
    """A policy that allows everything, with no crawl delay to wait out."""
    policy = RobotsPolicy(fetcher=lambda url: PERMISSIVE_ROBOTS)
    policy.wait = lambda url: None  # type: ignore[method-assign]
    return policy


@pytest.fixture
def ai_blocking_robots() -> RobotsPolicy:
    policy = RobotsPolicy(fetcher=lambda url: AI_BLOCKING_ROBOTS)
    policy.wait = lambda url: None  # type: ignore[method-assign]
    return policy


class FakeEmbeddings:
    """Deterministic embeddings, and a count of how often they were asked for.

    The count is the point: the corpus's re-ingest rule is "a chunk whose text
    has not changed keeps its vector", and the only way to show that is to
    prove the embedder was not called.
    """

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("agent" in lowered),
            float("eval" in lowered),
            float(len(lowered) % 7) / 7.0,
        ]

    @property
    def embedded_count(self) -> int:
        return sum(len(batch) for batch in self.document_calls)


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
