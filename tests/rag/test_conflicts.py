"""The disagreement layer: candidates, provenance, exclusivity and calibration.

No embedding model and no LLM. Candidate generation runs on hand-written
vectors and the adjudicator is a dict-backed stub, because everything this
module can get wrong is settled before a model is involved: which pairs are
eligible, whether a quote really came out of the stored transcript, and whether
one chunk can be spent twice.

The tests that matter most here are the negative ones. A conflict finder is
easy to make loud and the failure that would poison every downstream use of it
is a *manufactured* disagreement, so the complementary probes, the quote check
and the exclusivity rule each get their own case.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.conflicts import (
    DEFAULT_ADJUDICATE_REPEATS,
    AdjudicationCache,
    FIRM_VOTE_SHARE,
    PROBES,
    QUOTE_MATCH_RATIO,
    Adjudication,
    AdjudicationVote,
    CandidatePair,
    ClaimRef,
    Conflict,
    ConflictConfig,
    ConflictIndex,
    ConflictStore,
    build_conflicts,
    candidate_pairs,
    claims_from_chunks,
    corpus_fingerprint,
    states_a_position,
    conflict_statistics,
    format_pair,
    probe_pair,
    probes_passed,
    near_misses,
    resolve_conflicts,
    run_probes,
    stability_statistics,
    vote_ledger,
    verbatim_span,
)


class _Chunk:
    """The handful of attributes :func:`claims_from_chunks` reads off a chunk."""

    def __init__(self, chunk_id: str, video_id: str, text: str, **extra: object) -> None:
        self.chunk_id = chunk_id
        self.video_id = video_id
        self.text = text
        self.channel_name = extra.get("channel_name", "Creator")
        self.title = extra.get("title", "A talk")
        self.start_seconds = extra.get("start_seconds", 10.0)
        self.end_seconds = extra.get("end_seconds", 80.0)
        self.source_url = extra.get("source_url", "")


def _claim(
    claim_id: str,
    text: str,
    *,
    chunk_id: str,
    video_id: str,
    creator: str,
    chunk_text: str | None = None,
    start: float = 0.0,
) -> ClaimRef:
    return ClaimRef(
        claim_id=claim_id,
        text=text,
        chunk_id=chunk_id,
        chunk_text=chunk_text if chunk_text is not None else text,
        video_id=video_id,
        channel_name=creator,
        title=f"{creator} talk",
        start_seconds=start,
        end_seconds=start + 70.0,
    )


def _embedder(vectors: dict[str, list[float]]):
    """Embeds by exact claim text, so similarity is chosen rather than measured."""

    def embed(texts: list[str]) -> list[list[float]]:
        return [vectors[text] for text in texts]

    return embed


def _at(cosine: float, axis: int, dimensions: int = 20) -> list[float]:
    """A unit vector at ``cosine`` from ``e0``, tilted into its own dimension.

    Two of these built on different axes sit at the product of their cosines
    against each other, which is how a test picks a similarity band directly
    instead of nudging 2-D coordinates until the pair lands in one. Both bounds
    matter: below :attr:`ConflictConfig.similarity_floor` a pair is not about
    the same thing, and above the ceiling it is the same claim restated.
    """
    vector = [0.0] * dimensions
    vector[0] = cosine
    vector[axis] = (1.0 - cosine * cosine) ** 0.5
    return vector


# ── claims: identity comes from the chunk, never from the extraction ──────


def test_claims_take_every_identity_field_from_the_chunk() -> None:
    """The extraction supplies sentences; the chunk supplies where they are.

    This is the rule that exists because 35 of 39 model-supplied chunk indices
    measured in this corpus were wrong.
    """
    chunk = _Chunk(
        "chunk:vid1:3",
        "vid1",
        "the whole seventy seconds of speech",
        channel_name="Kevlin Henney",
        title="Modularity",
        start_seconds=402.4,
        end_seconds=471.9,
    )
    claims = claims_from_chunks([chunk], lambda _c: ["first sentence", "second sentence"])

    assert [claim.text for claim in claims] == ["first sentence", "second sentence"]
    for claim in claims:
        assert claim.video_id == "vid1"
        assert claim.chunk_id == "chunk:vid1:3"
        assert claim.channel_name == "Kevlin Henney"
        assert claim.start_seconds == 402.4
        # The claim sentence is for matching only; the chunk text is the evidence.
        assert claim.chunk_text == "the whole seventy seconds of speech"


def test_claims_skip_empty_chunks_and_blank_sentences() -> None:
    chunks = [_Chunk("chunk:a:0", "a", "   "), _Chunk("chunk:b:0", "b", "real text")]
    claims = claims_from_chunks(chunks, lambda _c: ["", "  ", "kept"])
    assert [claim.claim_id for claim in claims] == ["chunk:b:0#2"]


def test_watch_url_is_built_from_the_video_id_not_the_source_url() -> None:
    """A stored ``source_url`` may already carry ``t=``; two would fight.

    Eight videos in this corpus have a timestamp baked into their source url,
    and YouTube honours the first ``t`` it sees — so appending another silently
    lands the reader ~22s before the quote. The url is rebuilt from the id.
    """
    chunk = _Chunk(
        "chunk:vid1:0",
        "vid1",
        "text",
        start_seconds=402.9,
        source_url="https://www.youtube.com/watch?v=vid1&t=380s",
    )
    claim = claims_from_chunks([chunk], lambda _c: ["sentence"])[0]
    assert claim.watch_url == "https://www.youtube.com/watch?v=vid1&t=402"
    assert claim.watch_url.count("t=") == 1


# ── candidate generation: deterministic, and eligible pairs only ──────────


def test_candidates_exclude_same_video_same_creator_and_the_diagonal() -> None:
    vectors = {
        "a1": _at(1.0, 1),
        "a2": _at(0.8, 1),  # same video as a1
        "b1": _at(0.8, 2),  # same creator as a1, different video
        "c1": _at(0.8, 3),  # different creator — the only eligible partner
    }
    claims = [
        _claim("1", "a1", chunk_id="chunk:v1:0", video_id="v1", creator="Alice"),
        _claim("2", "a2", chunk_id="chunk:v1:1", video_id="v1", creator="Alice"),
        _claim("3", "b1", chunk_id="chunk:v2:0", video_id="v2", creator="Alice"),
        _claim("4", "c1", chunk_id="chunk:v3:0", video_id="v3", creator="Bob"),
    ]
    # Caps lifted: this test is about which pairs are *eligible*, and the
    # per-chunk cap would otherwise trim the eligible set to its best two.
    pairs = candidate_pairs(claims, _embedder(vectors), ConflictConfig(max_per_chunk=10))
    partners = {pair.key for pair in pairs}
    assert partners == {
        ("chunk:v1:0", "chunk:v3:0"),
        ("chunk:v1:1", "chunk:v3:0"),
        ("chunk:v2:0", "chunk:v3:0"),
    }
    assert all(pair.cross_channel for pair in pairs)


def test_within_creator_pairs_appear_only_when_asked_for() -> None:
    vectors = {"x": _at(1.0, 1), "y": _at(0.8, 1)}
    claims = [
        _claim("1", "x", chunk_id="chunk:v1:0", video_id="v1", creator="Alice"),
        _claim("2", "y", chunk_id="chunk:v2:0", video_id="v2", creator="Alice"),
    ]
    assert candidate_pairs(claims, _embedder(vectors)) == []
    opened = candidate_pairs(claims, _embedder(vectors), ConflictConfig(cross_channel_only=False))
    assert len(opened) == 1
    assert opened[0].cross_channel is False


def test_near_duplicates_above_the_ceiling_are_dropped_as_restatements() -> None:
    vectors = {"same": [1.0, 0.0], "identical": [1.0, 0.0]}
    claims = [
        _claim("1", "same", chunk_id="chunk:v1:0", video_id="v1", creator="Alice"),
        _claim("2", "identical", chunk_id="chunk:v2:0", video_id="v2", creator="Bob"),
    ]
    assert candidate_pairs(claims, _embedder(vectors)) == []


def test_pairs_below_the_floor_are_not_about_the_same_thing() -> None:
    vectors = {"left": [1.0, 0.0], "right": [0.0, 1.0]}
    claims = [
        _claim("1", "left", chunk_id="chunk:v1:0", video_id="v1", creator="Alice"),
        _claim("2", "right", chunk_id="chunk:v2:0", video_id="v2", creator="Bob"),
    ]
    assert candidate_pairs(claims, _embedder(vectors)) == []


def test_several_claims_from_the_same_two_chunks_collapse_to_one_pair() -> None:
    """Two chunks are one piece of evidence however many sentences they share."""
    vectors = {
        "a1": _at(1.0, 1),
        "a2": _at(0.9, 1),
        "b1": _at(0.8, 2),
        "b2": _at(0.75, 3),
    }
    claims = [
        _claim("1", "a1", chunk_id="chunk:v1:0", video_id="v1", creator="Alice"),
        _claim("2", "a2", chunk_id="chunk:v1:0", video_id="v1", creator="Alice"),
        _claim("3", "b1", chunk_id="chunk:v2:0", video_id="v2", creator="Bob"),
        _claim("4", "b2", chunk_id="chunk:v2:0", video_id="v2", creator="Bob"),
    ]
    pairs = candidate_pairs(claims, _embedder(vectors))
    assert len(pairs) == 1
    assert pairs[0].key == ("chunk:v1:0", "chunk:v2:0")


def test_one_chunk_cannot_take_the_whole_budget() -> None:
    """The cap that made the sweep worth paying for.

    A bi-encoder ranks paraphrase first and paraphrase is what agreement looks
    like, so one popular restatement drew twenty-four of the top forty pairs on
    the first sweep. Capping per chunk turns that cluster into two pairs and
    spends the rest on subjects the sweep had not reached.
    """
    vectors = {"hub": [1.0, 0.0]}
    claims = [_claim("0", "hub", chunk_id="chunk:v0:0", video_id="v0", creator="Hub")]
    for index in range(6):
        name = f"spoke{index}"
        vectors[name] = [0.9 - index * 0.01, 0.3]
        claims.append(
            _claim(
                str(index + 1),
                name,
                chunk_id=f"chunk:v{index + 1}:0",
                video_id=f"v{index + 1}",
                creator=f"Creator {index}",
            )
        )
    pairs = candidate_pairs(claims, _embedder(vectors), ConflictConfig(max_per_chunk=2))
    hub_pairs = [pair for pair in pairs if "chunk:v0:0" in pair.key]
    assert len(hub_pairs) == 2


def test_one_pair_of_videos_cannot_take_the_whole_budget() -> None:
    vectors: dict[str, list[float]] = {}
    claims: list[ClaimRef] = []
    for index in range(8):
        left, right = f"L{index}", f"R{index}"
        vectors[left] = _at(0.99, index + 1)
        vectors[right] = _at(0.8, index + 9)
        claims.append(
            _claim(f"l{index}", left, chunk_id=f"chunk:v1:{index}", video_id="v1", creator="Alice")
        )
        claims.append(
            _claim(f"r{index}", right, chunk_id=f"chunk:v2:{index}", video_id="v2", creator="Bob")
        )
    pairs = candidate_pairs(
        claims, _embedder(vectors), ConflictConfig(max_per_video_pair=3, max_per_chunk=8)
    )
    assert len(pairs) == 3


def test_candidate_generation_is_deterministic() -> None:
    vectors = {
        "a": [1.0, 0.0],
        "b": [0.95, 0.15],
        "c": [0.93, 0.2],
        "d": [0.9, 0.25],
    }
    claims = [
        _claim("1", "a", chunk_id="chunk:v1:0", video_id="v1", creator="Alice"),
        _claim("2", "b", chunk_id="chunk:v2:0", video_id="v2", creator="Bob"),
        _claim("3", "c", chunk_id="chunk:v3:0", video_id="v3", creator="Cara"),
        _claim("4", "d", chunk_id="chunk:v4:0", video_id="v4", creator="Dan"),
    ]
    first = candidate_pairs(claims, _embedder(vectors))
    second = candidate_pairs(list(reversed(claims)), _embedder(vectors))
    assert [pair.key for pair in first] == [pair.key for pair in second]


def test_the_adjudicator_is_shown_whole_chunks_not_claim_sentences() -> None:
    pair = CandidatePair(
        left=_claim(
            "1",
            "a claim sentence",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text="the full seventy seconds Alice actually said",
        ),
        right=_claim(
            "2",
            "another claim",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text="the full seventy seconds Bob actually said",
        ),
        similarity=0.8,
    )
    text = format_pair(pair)
    assert "the full seventy seconds Alice actually said" in text
    assert "a claim sentence" not in text


# ── provenance: the quote is cut out of the store, not typed by the model ──


def test_quote_is_returned_as_the_store_holds_it_including_asr_damage() -> None:
    """The Kleppmann case: the transcript says "right skew" every time.

    A model quoting from understanding rather than from the page silently fixes
    that, and the corrected quote then resolves against nothing. The span that
    ships is the store's own words.
    """
    stored = (
        "the problem you hit here is right skew, where two transactions read "
        "overlapping data and then both write, and neither sees the other."
    )
    match = verbatim_span(
        "the problem you hit here is write skew, where two transactions read overlapping data",
        stored,
    )
    assert match is not None
    assert "right skew" in match.text
    assert "write skew" not in match.text


def test_quote_the_speaker_never_said_is_rejected() -> None:
    stored = "keep your resume to a single page, no matter how long you have worked"
    assert (
        verbatim_span("two pages is the correct length for anybody with real experience", stored)
        is None
    )


def test_a_quote_too_short_to_prove_anything_is_rejected() -> None:
    stored = "well, it depends, and that is the honest answer to this one"
    assert verbatim_span("it depends", stored) is None


def test_the_match_ratio_is_the_share_of_the_quote_that_was_found() -> None:
    stored = "you should always put a cache in front of the database at scale"
    match = verbatim_span("you should always put a cache in front of the database", stored)
    assert match is not None
    assert match.ratio == 1.0
    assert QUOTE_MATCH_RATIO <= match.ratio


# ── resolution: four gates, each one counted ──────────────────────────────


def _pair(chunk_a: str, chunk_b: str, text_a: str, text_b: str, sim: float = 0.8) -> CandidatePair:
    return CandidatePair(
        left=_claim(
            "l",
            "left claim",
            chunk_id=chunk_a,
            video_id=chunk_a.split(":")[1],
            creator="Alice",
            chunk_text=text_a,
        ),
        right=_claim(
            "r",
            "right claim",
            chunk_id=chunk_b,
            video_id=chunk_b.split(":")[1],
            creator="Bob",
            chunk_text=text_b,
        ),
        similarity=sim,
    )


_LEFT_TEXT = "never put a cache in front of your database, it always serves stale data to users"
_RIGHT_TEXT = "always put a cache in front of your database, reads are what kill you at scale"


def _vote(*verdicts: Adjudication) -> AdjudicationVote:
    """The verdicts one pair drew, as the vote :func:`resolve_conflicts` reads."""
    return AdjudicationVote(verdicts=tuple(verdicts))


def _carries(**overrides: str) -> AdjudicationVote:
    """Three repeats that all say conflict — the shape that ships."""
    return _vote(_verdict(**overrides), _verdict(**overrides), _verdict(**overrides))


def _verdict(**overrides: str) -> Adjudication:
    base = dict(
        verdict="conflict",
        axis="Should you put a cache in front of your database?",
        why_incompatible="One says never and the other says always about the same decision.",
        position_a="Never cache.",
        position_b="Always cache.",
        quote_a="never put a cache in front of your database, it always serves stale data",
        quote_b="always put a cache in front of your database, reads are what kill you",
    )
    base.update(overrides)
    return Adjudication(**base)  # type: ignore[arg-type]


def test_a_conflict_keeps_both_sides_and_names_no_winner() -> None:
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    conflicts, tally = resolve_conflicts([pair], [_carries()])
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.axis.endswith("?")
    assert conflict.left.channel_name == "Alice"
    assert conflict.right.channel_name == "Bob"
    assert conflict.left.quote and conflict.right.quote
    assert tally == {
        "not_a_conflict": 0,
        "minority_verdict": 0,
        "unstated_axis": 0,
        "quote_not_in_transcript": 0,
        "quote_is_not_a_position": 0,
        "duplicate_evidence": 0,
    }
    assert conflict.votes == 3 and conflict.repeats == 3 and conflict.unanimous
    # The record cannot express a verdict even if something downstream wanted one.
    assert "winner" not in Conflict.model_fields
    assert "verdict" not in Conflict.model_fields


def test_complementary_and_unrelated_verdicts_produce_nothing() -> None:
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    conflicts, tally = resolve_conflicts(
        [pair, pair],
        [
            _vote(*[Adjudication(verdict="complementary")] * 3),
            _vote(*[Adjudication(verdict="unrelated")] * 3),
        ],
    )
    assert conflicts == []
    assert tally["not_a_conflict"] == 2


def test_a_conflict_nobody_can_state_is_not_reportable() -> None:
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    conflicts, tally = resolve_conflicts([pair], [_carries(axis="")])
    assert conflicts == []
    assert tally["unstated_axis"] == 1


def test_a_side_whose_quote_is_not_in_the_transcript_drops_the_conflict() -> None:
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    invented = _carries(quote_b="two pages is the right length for anybody with real experience")
    conflicts, tally = resolve_conflicts([pair], [invented])
    assert conflicts == []
    assert tally["quote_not_in_transcript"] == 1


def test_one_chunk_backs_at_most_one_conflict() -> None:
    """The anti-gaming rule: restating a disagreement does not make two.

    Both pairs rest on ``chunk:v1:0``, so the second is the same disagreement
    said twice and is dropped rather than counted. This is what bounds the count
    by how much distinct transcript exists rather than by how talkative the
    adjudicator is.
    """
    first = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.9)
    second = _pair("chunk:v1:0", "chunk:v3:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.7)
    conflicts, tally = resolve_conflicts([first, second], [_carries(), _carries()])
    assert len(conflicts) == 1
    assert tally["duplicate_evidence"] == 1


def test_the_highest_similarity_pair_wins_a_contested_chunk() -> None:
    high = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.9)
    low = _pair("chunk:v1:0", "chunk:v3:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.7)
    conflicts, _ = resolve_conflicts([high, low], [_carries(), _carries()])
    assert conflicts[0].right.video_id == "v2"


# ── the vote: a single draw is not a measurement ──────────────────────────


def test_a_minority_of_conflict_votes_does_not_carry():
    """The bug this module shipped with, as a test.

    The first sweep adjudicated every corpus pair exactly once. An independent
    re-run of the four cards it produced, three repeats each, drew 1/3, 2/3, 3/3
    and 1/3 — three of the four were coin flips. A pair that persuades one draw
    out of three is not a disagreement this layer should assert.
    """
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    vote = _vote(
        _verdict(),
        Adjudication(verdict="complementary"),
        Adjudication(verdict="complementary"),
    )
    conflicts, tally = resolve_conflicts([pair], [vote])
    assert conflicts == []
    # Counted apart from a flat rejection: this is the number that says how close
    # to the threshold the corpus sits, and a single-draw sweep cannot report it.
    assert tally["minority_verdict"] == 1
    assert tally["not_a_conflict"] == 0


def test_a_strict_majority_carries_and_the_tally_ships_with_it():
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    vote = _vote(_verdict(), Adjudication(verdict="complementary"), _verdict())
    conflicts, _ = resolve_conflicts([pair], [vote])
    assert len(conflicts) == 1
    # A reader has to be able to tell 2/3 from 3/3 on the card.
    assert conflicts[0].votes == 2
    assert conflicts[0].repeats == 3
    assert conflicts[0].unanimous is False


def test_an_even_split_falls_to_not_a_conflict():
    """Strict majority, so 1/2 fails — the conservative direction, as everywhere."""
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    vote = _vote(_verdict(), Adjudication(verdict="complementary"))
    conflicts, tally = resolve_conflicts([pair], [vote])
    assert conflicts == []
    assert tally["minority_verdict"] == 1


def test_every_pair_is_adjudicated_the_same_number_of_times_as_the_probes():
    """The asymmetry that caused this: probes repeated, corpus pairs drawn once."""
    calls: list[str] = []
    vectors = {"never cache": [1.0, 0.0], "always cache": [0.9, 0.3]}
    claims = [
        _claim(
            "1",
            "never cache",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "always cache",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]

    def adjudicate(pair: CandidatePair) -> Adjudication:
        calls.append(pair.left.chunk_id)
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        return _verdict()

    index = build_conflicts(
        claims,
        _embedder(vectors),
        adjudicate,
        config=ConflictConfig(adjudicate_repeats=3),
    )
    assert calls.count("chunk:v1:0") == 3
    assert index.stats["adjudicate_repeats"] == 3
    assert index.stats["adjudications"] == 3
    assert index.conflicts[0].votes == 3
    # The setting a count was produced under travels with the count.
    assert index.config["adjudicate_repeats"] == 3


def test_a_flapping_pair_is_published_as_flapping():
    """The artifact carries the error bar on every other number in it."""
    draws = {"n": 0}
    vectors = {"never cache": [1.0, 0.0], "always cache": [0.9, 0.3]}
    claims = [
        _claim(
            "1",
            "never cache",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "always cache",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]

    def adjudicate(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        draws["n"] += 1
        return _verdict() if draws["n"] == 1 else Adjudication(verdict="complementary")

    index = build_conflicts(
        claims,
        _embedder(vectors),
        adjudicate,
        config=ConflictConfig(adjudicate_repeats=3),
    )
    assert index.conflicts == []
    assert index.stats["pairs_with_split_verdicts"] == 1
    assert index.stats["verdict_agreement"] == 0.0
    assert index.stats["rejected"]["minority_verdict"] == 1


def test_near_misses_are_recorded_by_identity_not_only_counted():
    """ "Six pairs sat near the threshold" does not say which six.

    Without the list, a reviewer whose re-run produces a different set cannot
    tell whether the layer moved or the judge did.
    """
    carried = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.9)
    missed = _pair("chunk:v3:0", "chunk:v4:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.7)
    flat = _pair("chunk:v5:0", "chunk:v6:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.65)
    votes = [
        _carries(),
        _vote(_verdict(axis="Did they?"), Adjudication(verdict="complementary")),
        _vote(*[Adjudication(verdict="complementary")] * 3),
    ]
    rows = near_misses([carried, missed, flat], votes)
    # Only the one that persuaded someone and lost: a conflict is not a near
    # miss, and neither is a pair nobody ever called one.
    assert [row["chunk_ids"] for row in rows] == [["chunk:v3:0", "chunk:v4:0"]]
    assert rows[0]["votes"] == 1
    assert rows[0]["axis_claimed"] == "Did they?"


def test_near_misses_never_leak_into_the_conflicts(tmp_path: Path):
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    vote = _vote(
        _verdict(), Adjudication(verdict="complementary"), Adjudication(verdict="unrelated")
    )
    conflicts, _ = resolve_conflicts([pair], [vote])
    assert conflicts == []
    assert len(near_misses([pair], [vote])) == 1


# ── the quote has to be a position, not a question ────────────────────────


def test_a_rhetorical_question_is_not_a_position():
    """The card that shipped on ``"like what is a technology free domain model?"``.

    Eight words, resolving at 1.0 against the transcript, and stating nothing.
    The speaker does hold the position; that span does not express it, and the
    reader of a card sees the span rather than the surrounding minute.
    """
    assert states_a_position("like what is a technology free domain model?") is False
    assert states_a_position("Has anyone ever tried to do that") is False
    assert states_a_position("do you really think that is going to work") is False
    assert states_a_position("today I'll show you a refactor that removes framework deps") is True


def test_a_conflict_whose_side_is_a_question_is_dropped():
    left_text = "so what exactly is a technology free domain model supposed to look like here"
    right_text = "you should always separate the domain from the framework using ports and adapters"
    pair = _pair("chunk:v1:0", "chunk:v2:0", left_text, right_text)
    vote = _carries(
        quote_a="so what exactly is a technology free domain model supposed to look like",
        quote_b="you should always separate the domain from the framework using ports",
    )
    conflicts, tally = resolve_conflicts([pair], [vote])
    assert conflicts == []
    assert tally["quote_is_not_a_position"] == 1


def test_a_quote_too_short_to_be_a_position_is_dropped():
    """Ten words, not eight: eight is short enough for a fragment to clear it."""
    stored = "well it depends on the team you are working with here"
    assert verbatim_span("well it depends on the team", stored) is None


# ── axes and facts are different things ───────────────────────────────────


def test_a_factual_contradiction_is_labelled_as_one():
    """Two numbers for one quantity is not a matter of perspective.

    Even-handed framing is the honest rendering of "should you cache?" and a
    worse lie than picking a side for "what is the vacancy rate?". The layer
    still names no winner — it cannot check the world — but it says which kind
    of thing the reader is looking at.
    """
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    conflicts, _ = resolve_conflicts([pair], [_carries(kind="factual")])
    assert conflicts[0].kind == "factual"
    assert "winner" not in Conflict.model_fields

    conflicts, _ = resolve_conflicts([pair], [_carries()])
    assert conflicts[0].kind == "axis"


# ── statistics: a count with its denominator attached ─────────────────────


def test_precision_falls_when_more_candidates_are_proposed() -> None:
    """There is deliberately no metric here that rises by being noisier."""
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    conflicts, tally = resolve_conflicts([pair], [_carries()])
    tight = conflict_statistics(conflicts, candidates=10, tally=tally)
    wide = conflict_statistics(conflicts, candidates=200, tally=tally)
    assert tight["conflict_precision"] > wide["conflict_precision"]
    assert tight["conflicts"] == wide["conflicts"] == 1
    assert tight["candidates_adjudicated"] == 10


def test_statistics_report_zero_conflicts_without_pretending() -> None:
    stats = conflict_statistics([], candidates=478, tally={"not_a_conflict": 478})
    assert stats["conflicts"] == 0
    assert stats["conflict_precision"] == 0.0
    assert stats["channels"] == []
    assert stats["rejected"]["not_a_conflict"] == 478


def test_cross_and_within_channel_conflicts_are_counted_apart() -> None:
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    same_channel = CandidatePair(
        left=pair.left,
        right=_claim(
            "r",
            "right",
            chunk_id="chunk:v3:0",
            video_id="v3",
            creator="Alice",
            chunk_text=_RIGHT_TEXT,
        ),
        similarity=0.8,
    )
    conflicts, tally = resolve_conflicts([pair, same_channel], [_carries(), _carries()])
    stats = conflict_statistics(conflicts, candidates=2, tally=tally)
    assert stats["cross_channel_conflicts"] == 1
    assert stats["within_channel_conflicts"] == 0  # the second lost its chunk to the first


# ── calibration: both directions, or the count means nothing ──────────────


def _stub_adjudicator(verdict_by_probe: dict[str, str]):
    def adjudicate(pair: CandidatePair) -> Adjudication:
        probe_id = pair.left.chunk_id.split(":")[1]
        verdict = verdict_by_probe.get(probe_id, "complementary")
        if verdict != "conflict":
            return Adjudication(verdict=verdict)
        return Adjudication(
            verdict="conflict",
            axis="Do these two answer the same question differently?",
            why_incompatible="They cannot both be followed.",
            position_a="A's answer",
            position_b="B's answer",
            quote_a=pair.left.chunk_text[:60],
            quote_b=pair.right.chunk_text[:60],
        )

    return adjudicate


def test_probes_cover_both_directions() -> None:
    """Half the probes exist to catch an adjudicator that never says no."""
    expected = {probe.expect for probe in PROBES}
    assert expected == {"conflict", "complementary"}
    assert sum(1 for probe in PROBES if probe.expect == "complementary") >= 3


def test_an_adjudicator_that_calls_everything_a_conflict_fails_calibration() -> None:
    results = run_probes(_stub_adjudicator({probe.probe_id: "conflict" for probe in PROBES}))
    assert not probes_passed(results)
    failed = [row["probe_id"] for row in results if not row["passed"]]
    assert failed == [probe.probe_id for probe in PROBES if probe.expect == "complementary"]


def test_a_calibrated_adjudicator_passes_and_names_the_planted_axes() -> None:
    results = run_probes(
        _stub_adjudicator({p.probe_id: p.expect for p in PROBES}),
    )
    assert probes_passed(results)
    planted = [row for row in results if row["expect"] == "conflict"]
    # Surfaced as two named positions, not fused into one blended statement.
    assert all(row["axis"] and row["position_a"] and row["position_b"] for row in planted)


def test_a_flapping_adjudicator_is_recorded_as_flapping_not_rounded_to_a_tick() -> None:
    calls = {"n": 0}

    def flapping(pair: CandidatePair) -> Adjudication:
        calls["n"] += 1
        return Adjudication(verdict="conflict" if calls["n"] % 2 else "complementary")

    results = run_probes(flapping, probes=PROBES[:1], repeats=4)
    assert results[0]["unanimous"] is False
    assert results[0]["passed"] is False
    assert len(results[0]["verdicts"]) == 4


def test_probe_pairs_are_namespaced_so_they_can_never_look_like_corpus_chunks() -> None:
    for probe in PROBES:
        pair = probe_pair(probe)
        assert pair.left.chunk_id.startswith("probe:")
        assert pair.right.chunk_id.startswith("probe:")
        assert pair.cross_channel


# ── store and end-to-end build ────────────────────────────────────────────


def test_store_round_trips_and_writes_readable_json(tmp_path: Path) -> None:
    store = ConflictStore(tmp_path / "nested" / "conflicts.json")
    assert store.exists() is False
    assert store.load() is None
    index = ConflictIndex(generated_at="2026-08-10T00:00:00+00:00", stats={"conflicts": 0})
    path = store.save(index)
    assert path.is_file()
    assert json.loads(path.read_text())["stats"]["conflicts"] == 0
    assert store.load() is not None


def test_build_runs_the_probes_before_spending_the_budget_on_the_corpus() -> None:
    """A run whose adjudicator is broken should say so inside its own artifact."""
    vectors = {"never cache": [1.0, 0.0], "always cache": [0.9, 0.3]}
    claims = [
        _claim(
            "1",
            "never cache",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "always cache",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]

    def adjudicate(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        return _verdict()

    index = build_conflicts(
        claims,
        _embedder(vectors),
        adjudicate,
        embedding_model="test-embedder",
        adjudicator_model="test-adjudicator",
    )
    assert index.stats["probes_passed"] is True
    assert index.stats["conflicts"] == 1
    assert index.stats["candidates_adjudicated"] == 1
    assert index.stats["claims"] == 2
    assert len(index.probes) == len(PROBES)
    assert index.conflicts[0].cross_channel is True
    # Nothing planted ever reaches the reported conflicts.
    assert all(
        not side.chunk_id.startswith("probe:")
        for conflict in index.conflicts
        for side in (conflict.left, conflict.right)
    )


def test_a_build_that_finds_nothing_still_reports_what_it_looked_at() -> None:
    vectors = {"left": [1.0, 0.0], "right": [0.9, 0.3]}
    claims = [
        _claim(
            "1",
            "left",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "right",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]
    index = build_conflicts(
        claims,
        _embedder(vectors),
        _stub_adjudicator({p.probe_id: p.expect for p in PROBES}),
    )
    assert index.stats["conflicts"] == 0
    assert index.stats["candidates_adjudicated"] == 1
    assert index.stats["rejected"]["not_a_conflict"] == 1


# ── the vote is the only gate, and all four consumers are behind it ────────
#
# The first version of this layer let a pair through on a single draw, and the
# four places that report conflicts then disagreed with each other about what
# had shipped. These tests reach across layers on purpose — into the API payload
# the view reads and into the scorer's denominator — because "excluded from the
# conflicts list" is not the property that was broken. "Excluded everywhere" is.


_VOTED_OUT = "chunk:v3:0"


def _two_pair_corpus() -> list[ClaimRef]:
    """Four claims, two eligible pairs: v1/v2 and v3/v4, nothing across them.

    The vectors sit in two orthogonal planes, so the only pairs inside the
    similarity band are the two intended ones and the cross pairs score 0 —
    below :attr:`ConflictConfig.similarity_floor`, which is what "not about the
    same thing" looks like to the candidate generator.
    """
    return [
        _claim(
            "1",
            "carries",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "carries too",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
        _claim(
            "3", "loses", chunk_id=_VOTED_OUT, video_id="v3", creator="Carol", chunk_text=_LEFT_TEXT
        ),
        _claim(
            "4",
            "loses too",
            chunk_id="chunk:v4:0",
            video_id="v4",
            creator="Dave",
            chunk_text=_RIGHT_TEXT,
        ),
    ]


_TWO_PAIR_VECTORS = {
    "carries": [1.0, 0.0, 0.0, 0.0],
    "carries too": [0.9, 0.4358899, 0.0, 0.0],
    "loses": [0.0, 0.0, 1.0, 0.0],
    "loses too": [0.0, 0.0, 0.8, 0.6],
}


def _one_carries_one_flips():
    """An adjudicator that is certain about v1/v2 and flips on v3/v4.

    The v3/v4 pair draws conflict on its first look and complementary on the
    next two — 1/3, which is exactly what three of the four originally shipped
    cards scored when an independent evaluator re-ran them.
    """
    seen = {"n": 0}

    def adjudicate(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        if pair.left.chunk_id != _VOTED_OUT:
            return _verdict()
        seen["n"] += 1
        return _verdict() if seen["n"] == 1 else Adjudication(verdict="complementary")

    return adjudicate


def test_a_pair_that_loses_the_vote_reaches_none_of_the_four_consumers(tmp_path: Path) -> None:
    """The whole bug, end to end: the view, stats, precision, and the denominator.

    A 1/3 pair used to ship as a conflict in all four. Here it must appear in
    none of them — and the two it *does* appear in, ``near_misses`` and the
    ``minority_verdict`` tally, are both explicitly not conflicts.
    """
    from src.api.corpus import list_conflicts
    from src.evals.critique_run import contested_pairs

    index = build_conflicts(
        _two_pair_corpus(),
        _embedder(_TWO_PAIR_VECTORS),
        _one_carries_one_flips(),
        config=ConflictConfig(adjudicate_repeats=3),
    )
    path = ConflictStore(tmp_path / "conflicts.json").save(index)

    # 1. The view. The payload the frontend renders holds one card, and the
    #    voted-out pair's chunks are not anywhere in it.
    payload = list_conflicts(path)
    assert [conflict["conflict_id"] for conflict in payload["conflicts"]] == ["conflict:0"]
    assert _VOTED_OUT not in json.dumps(payload["conflicts"])
    assert payload["conflicts"][0]["votes"] == 3
    assert payload["conflicts"][0]["repeats"] == 3

    # 2. stats.conflicts.
    assert index.stats["conflicts"] == 1
    assert index.stats["cross_channel_conflicts"] == 1

    # 3. The conflict_precision *numerator* — the denominator still counts the
    #    losing pair, because it really was adjudicated and hiding that would
    #    make the precision flattering rather than honest.
    assert index.stats["candidates_adjudicated"] == 2
    assert index.stats["conflict_precision"] == 0.5
    assert index.stats["rejected"]["minority_verdict"] == 1

    # 4. The contested_coverage denominator, which the critique scorer builds
    #    from this same artifact.
    contested = contested_pairs(path)
    assert [pair.conflict_id for pair in contested] == ["conflict:0"]
    assert _VOTED_OUT not in {chunk for pair in contested for chunk in pair.chunk_ids}

    # Not silently dropped: it is on the record as a near miss, which is not a
    # conflict and is stored apart from them.
    assert [row["chunk_ids"] for row in index.near_misses] == [[_VOTED_OUT, "chunk:v4:0"]]
    assert index.near_misses[0]["votes"] == 1


def test_the_vote_tally_is_persisted_so_a_reader_can_see_a_weak_card(tmp_path: Path) -> None:
    """2/3 has to be distinguishable from 3/3 in the file, not just in memory."""
    pairs = [
        _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.9),
        _pair("chunk:v3:0", "chunk:v4:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.8),
    ]
    votes = [_carries(), _vote(_verdict(), Adjudication(verdict="complementary"), _verdict())]
    conflicts, _ = resolve_conflicts(pairs, votes)
    path = ConflictStore(tmp_path / "conflicts.json").save(
        ConflictIndex(generated_at="2026-08-10T00:00:00+00:00", conflicts=conflicts)
    )

    on_disk = json.loads(path.read_text(encoding="utf-8"))["conflicts"]
    assert [(row["votes"], row["repeats"]) for row in on_disk] == [(3, 3), (2, 3)]
    reloaded = ConflictStore(path).load()
    assert reloaded is not None
    assert [conflict.unanimous for conflict in reloaded.conflicts] == [True, False]


def test_the_configured_repeat_count_is_what_the_adjudicator_is_asked_for() -> None:
    """Five means five calls per pair, and one means the old single-draw sweep.

    One is still reachable — a cheap exploratory run is legitimate — but it
    lands in the artifact's own ``config``, so a count produced that way can
    never again be read as though it had been voted on.
    """
    calls: list[str] = []

    def adjudicate(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        calls.append(pair.left.chunk_id)
        return _verdict()

    claims = _two_pair_corpus()
    five = build_conflicts(
        claims,
        _embedder(_TWO_PAIR_VECTORS),
        adjudicate,
        config=ConflictConfig(adjudicate_repeats=5),
    )
    assert calls.count("chunk:v1:0") == 5
    assert five.config["adjudicate_repeats"] == 5
    assert five.stats["adjudicate_repeats"] == 5
    assert five.stats["adjudications"] == 10  # two pairs x five looks
    assert all(conflict.repeats == 5 for conflict in five.conflicts)

    calls.clear()
    once = build_conflicts(
        claims,
        _embedder(_TWO_PAIR_VECTORS),
        adjudicate,
        config=ConflictConfig(adjudicate_repeats=1),
    )
    assert calls.count("chunk:v1:0") == 1
    assert once.config["adjudicate_repeats"] == 1
    assert once.stats["adjudicate_repeats"] == 1
    assert [conflict.repeats for conflict in once.conflicts] == [1, 1]


def test_the_default_is_enough_looks_to_tell_a_certainty_from_a_coin_flip() -> None:
    """A caller who passes nothing gets the vote, and enough of it to read.

    Odd, so a strict majority always exists; a multiple of three, so a single
    run can report what three independent three-look builds would have said;
    and more than three, because at three looks the smallest majority *is* the
    firm band and the two cannot be told apart.
    """
    assert DEFAULT_ADJUDICATE_REPEATS % 2 == 1
    assert DEFAULT_ADJUDICATE_REPEATS % 3 == 0
    assert DEFAULT_ADJUDICATE_REPEATS > 3
    assert ConflictConfig().adjudicate_repeats == DEFAULT_ADJUDICATE_REPEATS


# ── the count ships with its spread ────────────────────────────────────────
#
# Two builds over an identical candidate set returned 4 and then 2. The count
# alone cannot express that, so these cover the numbers that can.


def test_the_vote_histogram_separates_a_sure_run_from_a_coin_flipping_one() -> None:
    """A pile of pairs at half the looks is what an untrustworthy count looks like."""
    pairs = [
        _pair(f"chunk:v{i}:0", f"chunk:w{i}:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.9 - i / 100)
        for i in range(3)
    ]
    sure = [_carries(), _carries(), _vote(*[Adjudication(verdict="complementary")] * 3)]
    stats = stability_statistics(pairs, sure)
    assert stats["vote_histogram"] == {"0": 1, "1": 0, "2": 0, "3": 2}
    assert stats["undecided_pairs"] == 0

    flipping = [
        _vote(_verdict(), _verdict(), Adjudication(verdict="complementary")),
        _vote(_verdict(), Adjudication(verdict="complementary"), _verdict()),
        _vote(_verdict(), Adjudication(verdict="complementary"), _verdict()),
    ]
    unsure = stability_statistics(pairs, flipping)
    assert unsure["vote_histogram"]["2"] == 3
    assert unsure["count_sd_estimate"] > stats["count_sd_estimate"]


def test_at_three_looks_the_firm_band_cannot_tell_a_coin_flip_from_a_certainty() -> None:
    """The argument for raising the repeat count, as an executable statement.

    ``FIRM_VOTE_SHARE`` is two thirds. At three looks the smallest majority *is*
    two thirds, so a pair the judge is evenly split on clears the firm band on
    half of all runs and is indistinguishable from one it is sure about. Nine
    looks is the first count that puts a rung between them: 5/9 carries the vote
    and is still undecided, 6/9 is firm.
    """
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    no = Adjudication(verdict="complementary")

    # Two thirds, and at three looks the smallest majority is exactly that —
    # the band and the threshold collapse onto each other.
    assert FIRM_VOTE_SHARE == 2 / 3
    assert 2 / 3 >= FIRM_VOTE_SHARE and 5 / 9 < FIRM_VOTE_SHARE

    at_three = stability_statistics([pair], [_vote(_verdict(), _verdict(), no)])
    assert at_three["undecided_pairs"] == 0  # 2/3 clears the band — the problem

    five_of_nine = _vote(*([_verdict()] * 5), *([no] * 4))
    six_of_nine = _vote(*([_verdict()] * 6), *([no] * 3))
    assert stability_statistics([pair], [five_of_nine])["undecided_pairs"] == 1
    assert stability_statistics([pair], [six_of_nine])["undecided_pairs"] == 0
    # Both carry the majority; only one of them is something to assert.
    assert len(resolve_conflicts([pair], [five_of_nine])[0]) == 1
    assert len(resolve_conflicts([pair], [six_of_nine])[0]) == 1


def test_a_pair_split_on_exactly_a_third_of_its_looks_counts_as_undecided() -> None:
    """The float that hid the commonest undecided pair there is.

    ``FIRM_VOTE_SHARE`` is ``2 / 3``, and the band's lower edge was written as
    ``1 - FIRM_VOTE_SHARE``. Neither two thirds nor one third is representable in
    binary and the subtraction rounds *up*: ``1 - 2 / 3`` is
    ``0.33333333333333337`` while ``1 / 3`` is ``0.3333333333333333``, so
    ``(1 - FIRM_VOTE_SHARE) <= votes / repeats`` was False for a pair that drew
    conflict on exactly a third of its looks — at every repeat count this module
    uses, and at three looks that is the *only* undecided tally there is.

    The shipped three-look artifact has eleven pairs at 1/3 and would have
    published ``undecided_pairs: 0`` next to its own ``minority_verdict: 11``,
    with the view rendering "0 pairs undecided" over a corpus whose whole error
    bar came from those eleven.
    """
    assert 1 - FIRM_VOTE_SHARE > 1 / 3  # the arithmetic that caused it
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    no = Adjudication(verdict="complementary")

    for repeats in (3, 6, 9, 15, 21):
        third = repeats // 3
        vote = _vote(*([_verdict()] * third), *([no] * (repeats - third)))
        stats = stability_statistics([pair], [vote])
        assert stats["undecided_pairs"] == 1, f"1/3 of {repeats} looks is undecided"
        # It loses the vote, so it is a near miss rather than a conflict — the
        # two counts describe the same pair and must not contradict each other.
        assert resolve_conflicts([pair], [vote])[1]["minority_verdict"] == 1

    # The mirror edge is unchanged: two thirds is firm, not undecided.
    firm = _vote(*([_verdict()] * 6), *([no] * 3))
    assert stability_statistics([pair], [firm])["undecided_pairs"] == 0


def test_the_spread_estimate_cannot_exceed_the_root_of_the_count_it_sits_beside() -> None:
    """The feasibility check on ``count_sd_estimate`` — see ``_ships_probability``.

    The count is a sum of independent indicators, so its variance is
    ``Σ q(1 - q)``, which is bounded by ``Σ q`` — the expected count itself. Any
    spread larger than the square root of the count the same model expects is
    arithmetically impossible, and this asserts the property rather than a
    number, so it holds for whatever histogram a future run produces.
    """
    from src.rag.conflicts import _ships_probability

    no = Adjudication(verdict="complementary")
    pairs = [
        _pair(f"chunk:v{i}:0", f"chunk:w{i}:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.9 - i / 100)
        for i in range(14)
    ]
    # The shipped artifact's shape, scaled down: mostly 0/3, a tail at 1/3, two
    # unanimous carriers.
    votes = [_vote(no, no, no)] * 3 + [_vote(_verdict(), no, no)] * 9 + [_carries(), _carries()]
    stats = stability_statistics(pairs, votes)
    expected_count = sum(_ships_probability(v.votes, v.repeats, 3) for v in votes)
    assert stats["count_sd_estimate"] <= expected_count**0.5

    # And the bias this bound exposes: the model expects far more conflicts than
    # the run it was fitted to actually produced.
    assert expected_count > 2 * len(resolve_conflicts(pairs, votes)[0])


def test_a_run_with_looks_to_spare_measures_its_own_three_look_spread() -> None:
    """The direct measurement: what independent 3-look builds would have said.

    The pair below is a deliberate coin flip — conflict on four of nine looks,
    arranged so the disjoint triples disagree with each other. Two of the three
    sub-runs would have shipped it and one would not, which is exactly the
    4-then-2 discrepancy that made the headline count unreadable, observed
    inside a single run instead of across two.
    """
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    no = Adjudication(verdict="complementary")
    vote = _vote(_verdict(), _verdict(), no, _verdict(), _verdict(), no, no, no, no)
    stats = stability_statistics([pair], [vote])
    assert stats["subsample_counts_at_3"] == [1, 1, 0]
    # It loses overall — 4 of 9 is not a majority — and is flagged as undecided
    # rather than quietly dropped.
    assert resolve_conflicts([pair], [vote])[0] == []
    assert stats["undecided_pairs"] == 1


def test_the_spread_is_not_reported_when_there_are_too_few_looks_to_measure_it() -> None:
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    stats = stability_statistics([pair], [_carries()])
    assert "subsample_counts_at_3" not in stats
    assert stats["vote_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 1}


def test_a_firm_conflict_is_counted_apart_from_one_that_barely_carried() -> None:
    """5/9 carries the vote and is still a coin flip. The gate reads the firm count."""
    vectors = {"never cache": [1.0, 0.0], "always cache": [0.9, 0.3]}
    claims = [
        _claim(
            "1",
            "never cache",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "always cache",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]
    draws = {"n": 0}

    def barely(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        draws["n"] += 1
        return _verdict() if draws["n"] <= 5 else Adjudication(verdict="complementary")

    index = build_conflicts(
        claims,
        _embedder(vectors),
        barely,
        config=ConflictConfig(adjudicate_repeats=9),
    )
    assert index.stats["conflicts"] == 1
    assert index.conflicts[0].votes == 5 and index.conflicts[0].repeats == 9
    # It ships — a majority is a majority — but it is not what the layer is
    # confident about, and the two numbers are reported apart.
    assert index.stats["firm_conflicts"] == 0
    assert index.stats["undecided_pairs"] == 1
    assert index.stats["count_sd_estimate"] > 0


# ── the population the count was taken over ───────────────────────────────


def test_the_corpus_a_count_was_taken_over_is_pinned_to_the_artifact() -> None:
    """Two counts are only comparable if they were taken over the same corpus.

    This repo's corpus grew 1372 -> 1460 -> 1736 chunks in one afternoon from
    ingests outside this layer, so "the count fell from 4 to 2" is unreadable
    without knowing whether the population moved underneath it.
    """
    chunks = [
        _Chunk("chunk:v1:0", "v1", "some words"),
        _Chunk("chunk:v1:1", "v1", "more words"),
        _Chunk("chunk:v2:0", "v2", "other words"),
    ]
    fingerprint = corpus_fingerprint(chunks)
    assert fingerprint["videos"] == 2
    assert fingerprint["chunks"] == 3
    # Order of the input must not change the identity of the corpus.
    assert corpus_fingerprint(list(reversed(chunks)))["digest"] == fingerprint["digest"]
    # Edited transcript is a different corpus even at the same counts.
    edited = [_Chunk("chunk:v1:0", "v1", "some words, re-transcribed"), *chunks[1:]]
    assert corpus_fingerprint(edited)["digest"] != fingerprint["digest"]
    assert corpus_fingerprint(edited)["chunks"] == 3


def test_the_blind_spot_is_stated_rather_than_left_to_be_inferred() -> None:
    """Chunks with no cached extraction reach nothing here, so the gap is named."""
    vectors = {"never cache": [1.0, 0.0], "always cache": [0.9, 0.3]}
    claims = [
        _claim(
            "1",
            "never cache",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "always cache",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]
    index = build_conflicts(
        claims,
        _embedder(vectors),
        _stub_adjudicator({p.probe_id: p.expect for p in PROBES}),
        corpus={"videos": 9, "chunks": 400, "digest": "abc123"},
    )
    # What was in the store, beside what actually reached the candidate stage.
    assert index.corpus["chunks"] == 400
    assert index.corpus["videos"] == 9
    assert index.corpus["digest"] == "abc123"
    assert index.corpus["chunks_with_claims"] == 2
    assert index.corpus["videos_with_claims"] == 2


# ── the rate set the variance claim rests on, on the record ───────────────


def test_every_adjudicated_pair_records_the_rate_it_drew() -> None:
    """A claim about spread has to be recomputable from the artifact.

    The whole population, not the interesting tail: the pairs that never drew a
    conflict verdict are most of the corpus and most of the evidence that a low
    count is the corpus rather than a filter.
    """
    carried = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.9)
    silent = _pair("chunk:v3:0", "chunk:v4:0", _LEFT_TEXT, _RIGHT_TEXT, sim=0.7)
    no = Adjudication(verdict="complementary")
    rows = vote_ledger([carried, silent], [_carries(), _vote(no, no, no)])

    assert [row["votes"] for row in rows] == [3, 0]
    assert all(row["repeats"] == 3 for row in rows)
    # The pair nobody ever called a conflict is present — it is the denominator.
    assert rows[1]["chunk_ids"] == ["chunk:v3:0", "chunk:v4:0"]
    assert rows[1]["video_ids"] == ["v3", "v4"]
    # Pair identity travels, so two runs can be matched pair by pair.
    assert rows[0]["chunk_ids"] == ["chunk:v1:0", "chunk:v2:0"]


def test_the_ledger_counts_errors_apart_from_complementary_verdicts() -> None:
    """A call that failed is not the judge saying "these agree".

    Worth separating because a provider outage mid-sweep would otherwise read as
    a corpus with no disagreements in it.
    """
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    vote = _vote(
        _verdict(),
        Adjudication(verdict="error", error="402"),
        Adjudication(verdict="complementary"),
    )
    row = vote_ledger([pair], [vote])[0]
    assert row["votes"] == 1
    assert row["errors"] == 1


def test_the_ledger_ships_inside_the_artifact_and_survives_a_round_trip(tmp_path: Path) -> None:
    vectors = {"never cache": [1.0, 0.0], "always cache": [0.9, 0.3]}
    claims = [
        _claim(
            "1",
            "never cache",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "always cache",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]
    index = build_conflicts(
        claims,
        _embedder(vectors),
        _stub_adjudicator({p.probe_id: p.expect for p in PROBES}),
        config=ConflictConfig(adjudicate_repeats=9),
    )
    assert len(index.vote_ledger) == index.stats["candidates_adjudicated"]
    # The histogram is a summary of the ledger, so the two must agree.
    from collections import Counter

    counted = Counter(str(row["votes"]) for row in index.vote_ledger)
    assert {k: v for k, v in index.stats["vote_histogram"].items() if v} == dict(counted)

    reloaded = ConflictStore(tmp_path / "conflicts.json")
    reloaded.save(index)
    assert reloaded.load().vote_ledger == index.vote_ledger


# ── resuming a killed sweep without destroying the vote ───────────────────


def _cache_claims() -> list[ClaimRef]:
    return [
        _claim(
            "1",
            "never cache",
            chunk_id="chunk:v1:0",
            video_id="v1",
            creator="Alice",
            chunk_text=_LEFT_TEXT,
        ),
        _claim(
            "2",
            "always cache",
            chunk_id="chunk:v2:0",
            video_id="v2",
            creator="Bob",
            chunk_text=_RIGHT_TEXT,
        ),
    ]


_CACHE_VECTORS = {"never cache": [1.0, 0.0], "always cache": [0.9, 0.3]}


def test_the_cache_keeps_nine_looks_as_nine_independent_draws(tmp_path: Path) -> None:
    """The trap: a cache keyed on the pair alone would collapse the vote.

    Every look would be served the first look's answer, every pair would come
    back unanimous, and the tally — the whole point of this layer — would become
    a lie that reads as maximum confidence.
    """
    draws = {"n": 0}

    def alternating(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        draws["n"] += 1
        return _verdict() if draws["n"] % 2 else Adjudication(verdict="complementary")

    cache = AdjudicationCache(tmp_path, "run-1")
    index = build_conflicts(
        _cache_claims(),
        _embedder(_CACHE_VECTORS),
        alternating,
        config=ConflictConfig(adjudicate_repeats=9),
        cache=cache,
    )
    ledger = index.vote_ledger[0]
    assert ledger["repeats"] == 9
    # Five of nine, not nine of nine: the looks stayed independent.
    assert 0 < ledger["votes"] < 9
    assert cache.writes == 9


def test_a_resumed_sweep_reuses_its_own_draws_and_stops_paying_for_them(
    tmp_path: Path,
) -> None:
    calls = {"n": 0}

    def counting(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        calls["n"] += 1
        return _verdict()

    first = build_conflicts(
        _cache_claims(),
        _embedder(_CACHE_VECTORS),
        counting,
        config=ConflictConfig(adjudicate_repeats=9),
        cache=AdjudicationCache(tmp_path, "run-1"),
    )
    corpus_calls = calls["n"]
    assert corpus_calls == 9

    resumed_cache = AdjudicationCache(tmp_path, "run-1")
    second = build_conflicts(
        _cache_claims(),
        _embedder(_CACHE_VECTORS),
        counting,
        config=ConflictConfig(adjudicate_repeats=9),
        cache=resumed_cache,
    )
    assert calls["n"] == corpus_calls  # not one more corpus call was paid for
    assert resumed_cache.hits == 9
    assert second.vote_ledger == first.vote_ledger


def test_a_second_measurement_needs_a_new_key_and_is_not_replayed(
    tmp_path: Path,
) -> None:
    """Reproducibility is what this layer is trying to establish.

    A cache that served run one's answers to run two would manufacture perfect
    agreement between two runs that never independently happened — so the run
    id scopes the whole cache and there is no default.
    """
    calls = {"n": 0}

    def counting(pair: CandidatePair) -> Adjudication:
        if pair.left.chunk_id.startswith("probe:"):
            probe_id = pair.left.chunk_id.split(":")[1]
            expect = next(p.expect for p in PROBES if p.probe_id == probe_id)
            return _stub_adjudicator({probe_id: expect})(pair)
        calls["n"] += 1
        return _verdict()

    for run_id in ("run-1", "run-2"):
        build_conflicts(
            _cache_claims(),
            _embedder(_CACHE_VECTORS),
            counting,
            config=ConflictConfig(adjudicate_repeats=9),
            cache=AdjudicationCache(tmp_path, run_id),
        )
    # Both runs paid in full: the second measurement is a real measurement.
    assert calls["n"] == 18


def test_a_failed_call_is_never_cached(tmp_path: Path) -> None:
    """A provider outage must not be pinned into the record as a verdict."""
    cache = AdjudicationCache(tmp_path, "run-1")
    pair = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    cache.put(pair, 0, Adjudication(verdict="error", error="402 Insufficient Balance"))
    assert cache.get(pair, 0) is None
    assert cache.writes == 0
    cache.put(pair, 0, _verdict())
    assert cache.get(pair, 0) is not None


def test_edited_transcript_invalidates_a_cached_verdict(tmp_path: Path) -> None:
    """A verdict about different words is not a verdict about these ones."""
    cache = AdjudicationCache(tmp_path, "run-1")
    original = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT)
    cache.put(original, 0, _verdict())
    rechunked = _pair("chunk:v1:0", "chunk:v2:0", _LEFT_TEXT, _RIGHT_TEXT + " and one more thing")
    assert cache.get(rechunked, 0) is None
