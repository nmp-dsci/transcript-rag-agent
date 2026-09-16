"""Turn a topic into the set of videos a guide should read in full.

No LLM here. The topic becomes a handful of templated probe questions, each
probe runs through the same hybrid retriever the Chat tab uses, and a video is
ranked by how many probes surfaced it — with a bonus when its title names the
topic outright. The result is a checklist the user confirms (or the CLI accepts
with ``--yes``), never a silent choice: which videos a guide rests on is the
one decision that shapes everything downstream.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

#: ``retrieve(question, top_k) -> chunks`` where each chunk has a ``video_id``
#: attribute or key. The hybrid provider's ``get_context(...).retrieved_chunks``
#: fits; so does a list of dicts in tests.
RetrieveFn = Callable[[str, int], Iterable[Any]]

PROBE_TEMPLATES = (
    "{topic}",
    "what is {topic} and why does it matter",
    "how to do {topic} step by step",
    "{topic} in production at scale",
    "common mistakes and failure modes in {topic}",
    "{topic} best practices and principles",
    "tools, frameworks and techniques for {topic}",
    "how to evaluate and measure {topic}",
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "in",
    "on",
    "to",
    "with",
    "as",
    "is",
    "are",
    "how",
    "what",
    "why",
    "into",
    "from",
    "by",
    "at",
    "vs",
    "versus",
}


def topic_tokens(topic: str) -> list[str]:
    """The words of a topic that carry meaning, lower-cased, de-duplicated."""
    words = re.findall(r"[a-z0-9][a-z0-9+#.-]*", topic.lower())
    seen: list[str] = []
    for word in words:
        word = word.strip(".-")
        # A hyphenated term counts whole *and* by its parts, so "LLM-as-a-judge"
        # meets "LLM as a Judge Explained" halfway.
        for part in [word, *(word.split("-") if "-" in word else [])]:
            if len(part) < 2 or part in STOPWORDS or part in seen:
                continue
            seen.append(part)
    return seen


def probe_questions(topic: str) -> list[str]:
    clean = " ".join(topic.split())
    return [template.format(topic=clean) for template in PROBE_TEMPLATES]


#: Leading question scaffolding that says nothing about the subject. Matched
#: as whole phrases at the start, longest first, so "what is the best way to
#: do X" leaves "do X" → "X" after the verb strip below.
_QUESTION_LEADS = (
    "what is the best way to",
    "what are the best ways to",
    "what's the best way to",
    "how do i",
    "how do you",
    "how do we",
    "how do teams",
    "how does one",
    "how should i",
    "how should we",
    "how can i",
    "how can we",
    "how to",
    "what is",
    "what are",
    "what's",
    "what does",
    "what do",
    "why do",
    "why does",
    "why is",
    "why are",
    "when should",
    "when do",
    "should i",
    "should we",
    "can you",
    "could you",
    "tell me about",
    "explain",
    "write a guide on",
    "write a guide about",
    "write me a guide on",
    "i want a guide on",
    "i want to know",
    "i want to understand",
    "give me",
    "make me",
)
_QUESTION_VERBS = ("do", "make", "build", "use", "get", "go about", "approach", "handle")
_QUESTION_TAILS = (" and why", " and how", " and when", " and where")


def topic_from_question(question: str) -> str:
    """The subject phrase of a plain-language question, for probes and the slug.

    No LLM: strip a leading "how do teams", a bare verb after it, a trailing
    "and why", and the question mark; then keep the content words in order,
    capped at ten. "How do teams calibrate an LLM judge against human
    labels, and what do they do when it drifts?" → "calibrate an llm judge
    against human labels".
    """
    text = " ".join(question.split()).strip().rstrip("?.!").strip().lower()
    for lead in sorted(_QUESTION_LEADS, key=len, reverse=True):
        if text.startswith(lead + " "):
            text = text[len(lead) + 1 :]
            break
    for verb in _QUESTION_VERBS:
        if text.startswith(verb + " "):
            text = text[len(verb) + 1 :]
            break
    # A second clause ("..., and what do they do when") is usually a follow-up
    # question; the first clause names the subject.
    text = re.split(r",\s*(?:and|or|but)\s+|;\s+", text, maxsplit=1)[0]
    for tail in _QUESTION_TAILS:
        if text.endswith(tail):
            text = text[: -len(tail)]
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9+#.'-]*", text)]
    # Trim leading/trailing stopwords but keep the ones inside the phrase so
    # the probes still read as English ("judge against human labels").
    while words and words[0] in STOPWORDS:
        words.pop(0)
    while words and words[-1] in STOPWORDS:
        words.pop()
    return " ".join(words[:10]) or " ".join(question.split())[:80]


def probes_for(question: str) -> list[str]:
    """The retrieval probes for a question: the question itself, then the
    topic templates over its subject phrase."""
    clean = " ".join(question.split())
    topic = topic_from_question(clean)
    probes = [clean] if clean.lower() != topic else []
    for probe in probe_questions(topic):
        if probe not in probes:
            probes.append(probe)
    return probes


def title_matches(topic: str, title: str | None) -> bool:
    """True when at least half of the topic's meaningful words appear in the title.

    "LLM-as-a-judge" against "LLM-as-a-Judge 101" is the case this exists for;
    the hyphenated form is one token, and so is the title's.
    """
    if not title:
        return False
    tokens = topic_tokens(topic)
    if not tokens:
        return False
    # Whole-word matching: "ai" must not match inside "genai".
    title_words = set(topic_tokens(title))
    hits = sum(1 for token in tokens if token in title_words)
    return hits / len(tokens) >= 0.5


def _video_id(chunk: Any) -> str | None:
    if isinstance(chunk, dict):
        return chunk.get("video_id")
    return getattr(chunk, "video_id", None)


@dataclass
class Candidate:
    video_id: str
    title: str
    channel_name: str
    chunk_count: int
    probe_hits: int = 0
    probes: list[str] = field(default_factory=list)
    title_match: bool = False

    @property
    def score(self) -> float:
        return self.probe_hits + (2.0 if self.title_match else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel_name": self.channel_name,
            "chunk_count": self.chunk_count,
            "probe_hits": self.probe_hits,
            "probes": list(self.probes),
            "title_match": self.title_match,
            "score": self.score,
        }


def candidate_videos(
    topic: str,
    videos: list[dict[str, Any]],
    retrieve: RetrieveFn,
    *,
    top_k: int = 30,
    probes: list[str] | None = None,
) -> list[Candidate]:
    """Rank the corpus for a topic. Only videos with some signal are returned."""
    questions = probes or probe_questions(topic)
    by_id: dict[str, Candidate] = {}
    for video in videos:
        video_id = str(video.get("video_id") or "")
        if not video_id:
            continue
        by_id[video_id] = Candidate(
            video_id=video_id,
            title=str(video.get("title") or video_id),
            channel_name=str(video.get("channel_name") or ""),
            chunk_count=int(video.get("chunk_count") or 0),
            title_match=title_matches(topic, video.get("title")),
        )
    for question in questions:
        hit: set[str] = set()
        for chunk in retrieve(question, top_k):
            found = _video_id(chunk)
            if found and found in by_id and found not in hit:
                hit.add(found)
                by_id[found].probe_hits += 1
                by_id[found].probes.append(question)
    ranked = [candidate for candidate in by_id.values() if candidate.score > 0]
    ranked.sort(key=lambda c: (-c.score, -c.chunk_count, c.title.lower()))
    return ranked


def cluster_videos(
    videos: list[dict[str, Any]], *, max_clusters: int = 6, min_per_cluster: int = 3
) -> list[list[str]]:
    """Split a video set into parallel extraction clusters.

    Same-channel videos stay together (a creator's vocabulary is consistent,
    so one extractor reads them best), and the cluster count is capped so a
    small set is not sliced into single-video passes.
    """
    if not videos:
        return []
    count = min(max_clusters, max(1, math.ceil(len(videos) / min_per_cluster)))
    ordered = sorted(
        videos,
        key=lambda v: (str(v.get("channel_name") or ""), str(v.get("title") or "")),
    )
    size = math.ceil(len(ordered) / count)
    clusters = [
        [str(v["video_id"]) for v in ordered[start : start + size]]
        for start in range(0, len(ordered), size)
    ]
    return [cluster for cluster in clusters if cluster]
