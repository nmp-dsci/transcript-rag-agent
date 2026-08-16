"""The held-out critique harness: matching, provenance, leak detection, run shape.

Everything here runs with fakes — a dict-backed transcript, a bag-of-words
"embedding" — because the point of the module under test is that it needs no
model and no API key to produce its numbers. The one test that does touch the
real dataset checks it against the real chunk store, and skips when the corpus
is not present.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.evals.critique import (
    CRITIQUE_METRICS,
    DEFAULT_GROUNDING_GATE,
    GATE_EXCLUSIVE,
    GATE_PROVENANCE,
    Citation,
    CitationCheck,
    ContestedPair,
    Criterion,
    CriterionMatch,
    CritiqueDataset,
    Finding,
    MatchResult,
    SetupCritique,
    attach_provenance,
    build_run,
    check_citation,
    gate_verdict,
    held_out_leaks,
    load_critique_dataset,
    consensus,
    embedding_matcher,
    enforce_one_to_one,
    ground_findings,
    repeated_matcher,
    match_criteria,
    parse_findings,
    quote_ratio,
    score_critique,
    verify_dataset,
)

TRANSCRIPT = {
    ("vidA", 100.0): [
        (
            "chunk:vidA:0",
            "you really want to keep it to one page because a recruiter will not read two",
        )
    ],
    ("vidB", 200.0): [
        ("chunk:vidB:0", "put numbers on every claim so the reader does not have to trust you")
    ],
    ("vidC", 300.0): [
        ("chunk:vidC:0", "a skills list is wasted space when the projects already show the skills")
    ],
}


def chunk_text(video_id: str, seconds: float) -> list[tuple[str, str]]:
    return TRANSCRIPT.get((video_id, seconds), [])


def fake_embed(texts):
    """A bag-of-words vector over a fixed vocabulary — no model, deterministic.

    Cosine over word overlap behaves like the real embedder for what these tests
    assert: restatements of a rule share vocabulary, unrelated rules do not.
    """
    vocab = sorted({word for text in texts for word in text.lower().split()})
    index = {word: i for i, word in enumerate(vocab)}
    vectors = []
    for text in texts:
        vector = [0.0] * len(vocab)
        for word in text.lower().split():
            vector[index[word]] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


def criterion(cid: str, text: str, *, video_id: str, seconds: float, kinds=("portfolio",)):
    return Criterion(
        id=cid,
        criterion=text,
        quote=TRANSCRIPT[(video_id, seconds)][0][1][:40],
        video_id=video_id,
        start_seconds=seconds,
        applies_to=tuple(kinds),
    )


def dataset(criteria):
    return CritiqueDataset(
        held_out_video_id="HELD",
        held_out_title="held out",
        artifact_id="doc:1",
        artifact_url="https://example.test/",
        artifact_kind="portfolio",
        criteria=tuple(criteria),
    )


# ── provenance ────────────────────────────────────────────────────────────


def test_quote_ratio_scores_the_share_of_the_quote_that_is_present():
    """A short quote inside a long chunk is a full match, not a partial one.

    The question a citation asks is "did they say this", not "is this everything
    they said" — a symmetric similarity would punish quoting eight words out of
    seventy, which is what an honest citation looks like.
    """
    assert quote_ratio("keep it to one page", TRANSCRIPT[("vidA", 100.0)][0][1]) == 1.0


def test_a_quote_nobody_said_does_not_resolve():
    check = check_citation(
        Citation(video_id="vidA", start_seconds=100.0, quote="always use two pages"),
        chunk_text,
    )
    assert check.resolved is False
    assert "quote not found" in check.reason


def test_a_citation_to_a_timestamp_with_no_chunk_does_not_resolve():
    check = check_citation(
        Citation(video_id="vidA", start_seconds=9999.0, quote="keep it to one page"),
        chunk_text,
    )
    assert check.resolved is False
    assert check.reason == "no chunk at this video and timestamp"


def test_a_verbatim_quote_at_its_own_timestamp_resolves():
    check = check_citation(
        Citation(video_id="vidB", start_seconds=200.0, quote="put numbers on every claim"),
        chunk_text,
    )
    assert check.resolved is True
    assert check.ratio == pytest.approx(1.0)


# ── criteria matching ─────────────────────────────────────────────────────


def test_a_reworded_rule_matches_the_criterion_it_restates():
    criteria = [criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0)]
    findings = [Finding(id="f1", criterion="the document should be one page", detail="")]
    matches = match_criteria(criteria, findings, fake_embed, threshold=0.5)
    assert matches[0].matched
    assert matches[0].finding_id == "f1"


def test_one_finding_cannot_cover_two_criteria():
    """Matching is one-to-one, so a system cannot buy recall by repeating itself.

    Both criteria here are close to the single finding; only the closer one may
    take it, and the other must be reported as a miss.
    """
    criteria = [
        criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
        criterion("c2", "keep the document to one page always", video_id="vidA", seconds=100.0),
    ]
    findings = [Finding(id="f1", criterion="keep the document to one page", detail="")]
    matches = match_criteria(criteria, findings, fake_embed, threshold=0.5)
    assert sum(m.matched for m in matches) == 1


def test_an_unmatched_criterion_still_reports_its_best_near_miss():
    """A miss has to be auditable, or an unmatched row means nothing.

    Without the near-miss score a reviewer cannot tell "the system never raised
    this" from "it raised it and the threshold was too tight".
    """
    criteria = [criterion("c1", "put numbers on claims", video_id="vidB", seconds=200.0)]
    findings = [Finding(id="f1", criterion="put numbers on some claims maybe", detail="")]
    matches = match_criteria(criteria, findings, fake_embed, threshold=0.99)
    assert matches[0].matched is False
    assert matches[0].finding_criterion == "put numbers on some claims maybe"
    assert matches[0].score > 0.0


def test_a_matcher_that_reuses_one_finding_has_the_reuse_taken_back():
    """The one-to-one rule is re-applied downstream, not left to the matcher.

    An LLM asked to pair eighteen criteria against ten findings will spend the
    same finding on several of them however firmly it is told not to, and that
    is exactly how a critique that made one broad point would score as having
    covered three separate rules.
    """
    rows = [
        CriterionMatch(
            criterion=criterion("c1", "one page", video_id="vidA", seconds=100.0),
            finding_id="f1",
            score=1.0,
        ),
        CriterionMatch(
            criterion=criterion("c2", "put numbers", video_id="vidB", seconds=200.0),
            finding_id="f1",
            score=0.4,
        ),
    ]
    kept = enforce_one_to_one(rows)
    assert [m.matched for m in kept] == [True, False]


def test_no_findings_leaves_every_criterion_unmatched():
    criteria = [criterion("c1", "one page", video_id="vidA", seconds=100.0)]
    matches = match_criteria(criteria, [], fake_embed)
    assert [m.matched for m in matches] == [False]


# ── leak detection ────────────────────────────────────────────────────────


def test_a_held_out_chunk_id_is_caught_by_prefix():
    """The proof must not depend on the filter it is proving.

    Chunk ids are ``chunk:{video_id}:{index}``, so this is a prefix test that
    would still fire with every ``$nin`` in the retrieval stack deleted.
    """
    leaks = held_out_leaks("HELD", chunk_ids=["chunk:HELD:3", "chunk:other:1"])
    assert leaks == ["chunk:HELD:3"]


def test_a_citation_to_the_held_out_video_is_a_leak_even_with_clean_retrieval():
    leaks = held_out_leaks(
        "HELD",
        chunk_ids=["chunk:other:1"],
        citations=[Citation(video_id="HELD", start_seconds=1.0, quote="x")],
    )
    assert leaks == ["citation:HELD@1.0"]


def test_a_clean_run_reports_no_leaks():
    assert (
        held_out_leaks(
            "HELD",
            chunk_ids=["chunk:vidA:0"],
            video_ids=["vidA"],
            citations=[Citation(video_id="vidA", start_seconds=100.0, quote="x")],
        )
        == []
    )


# ── scoring ───────────────────────────────────────────────────────────────


def scored(
    findings,
    criteria=None,
    conflicts=(),
    retrieved_chunk_ids=None,
    provenance=None,
    gate=DEFAULT_GROUNDING_GATE,
):
    """Score findings as one cell.

    ``provenance`` is what the engine says each finding's *own* reasoning
    retrieved — the thing :data:`GATE_PROVENANCE` grades against. It is written
    out per test rather than derived from the citations, because deriving it
    from the citations is exactly the vacuous gate this harness must not have.
    """
    data = dataset(
        criteria
        or [
            criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
            criterion("c2", "put numbers on every claim", video_id="vidB", seconds=200.0),
        ]
    )
    chunk_ids = (
        list(retrieved_chunk_ids)
        if retrieved_chunk_ids is not None
        else ["chunk:vidA:0", "chunk:vidB:0"]
    )
    critique = SetupCritique(
        setup="rag_llm_filtered",
        findings=attach_provenance(findings, provenance) if provenance else list(findings),
        retrieved_chunk_ids=chunk_ids,
        retrieved_video_ids=sorted({cid.split(":")[1] for cid in chunk_ids}),
    )
    return score_critique(
        critique,
        data,
        embedding_matcher(fake_embed, threshold=0.5),
        chunk_text,
        conflicts,
        gate=gate,
    )


#: One disagreement whose two sides are the two chunks every ``scored`` call
#: retrieves, so a test can put a conflict in context without restating it.
BOTH_SIDES = ContestedPair(
    conflict_id="conflict:0",
    axis="How long should the document be?",
    video_ids=("vidA", "vidB"),
    chunk_ids=("chunk:vidA:0", "chunk:vidB:0"),
)


def test_a_grounded_finding_counts_for_precision_and_provenance():
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="keep the document to one page",
                detail="this one is three pages",
                citations=(
                    Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),
                ),
            )
        ],
        provenance={"f1": ["chunk:vidA:0"]},
    )
    assert cell["scores"]["evidence_precision"] == 1.0
    assert cell["scores"]["provenance"] == 1.0
    assert cell["scores"]["criteria_recall"] == 0.5
    assert cell["gradable"] is True


def test_a_finding_with_an_invented_quote_is_not_grounded():
    """An unresolvable citation must cost both precision and provenance.

    This is the failure the harness exists to catch: a critique that reads as
    evidence-backed but whose evidence is not in the corpus.
    """
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="keep the document to one page",
                detail="",
                citations=(
                    Citation(video_id="vidA", start_seconds=100.0, quote="never exceed four pages"),
                ),
            )
        ],
        provenance={"f1": ["chunk:vidA:0"]},
    )
    assert cell["scores"]["evidence_precision"] == 0.0
    assert cell["scores"]["provenance"] == 0.0
    assert cell["findings"][0]["grounded"] is False


def test_a_fabricated_citation_is_named_at_the_top_of_the_cell():
    """The incident this harness exists to catch, kept where it can be seen.

    A baseline run cited a fluent, correctly-attributed sentence the speaker
    never said; it matched the transcript at 0.41 and a difflib ratio caught it
    with nobody reading anything. Buried as one `resolved: false` inside a
    nested array, an incident like that stops being visible.
    """
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="keep the document to one page",
                detail="",
                citations=(
                    Citation(
                        video_id="vidA",
                        start_seconds=100.0,
                        quote="never exceed four pages under any circumstances at all",
                    ),
                ),
            )
        ]
    )
    fabricated = cell["fabricated_citations"]
    assert len(fabricated) == 1
    # The model's own words, kept so the fabrication can be read and not only
    # counted — an invented sentence and a quote that dropped a filler word are
    # different failures with the same boolean.
    assert fabricated[0]["claimed_quote"].startswith("never exceed four pages")
    assert fabricated[0]["finding_id"] == "f1"
    assert fabricated[0]["ratio"] < 0.8


def test_a_clean_cell_names_no_fabrications():
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
            )
        ]
    )
    assert cell["fabricated_citations"] == []


def test_contested_coverage_is_none_when_no_disagreement_was_in_context():
    """ "Averaged a conflict away" and "was never shown one" are different failures.

    The metric this replaced reported 0.000 for both, which is how it managed to
    read 0.000 on every run ever measured while nothing was wrong with the model
    at all — its retrieval had simply never put two sides of anything in front of
    it.
    """
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
            )
        ]
    )
    assert cell["scores"]["contested_coverage"] is None
    assert cell["conflicts_in_context"] == 0


def test_a_disagreement_in_context_that_nobody_named_scores_zero():
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
            )
        ],
        conflicts=[BOTH_SIDES],
    )
    assert cell["scores"]["contested_coverage"] == 0.0
    assert cell["conflicts_in_context"] == 1
    assert cell["conflicts_named"] == 0


def test_naming_both_sides_of_a_disagreement_counts_it():
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                citations=(
                    Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),
                    Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),
                ),
            )
        ],
        conflicts=[BOTH_SIDES],
    )
    assert cell["scores"]["contested_coverage"] == 1.0
    assert cell["conflicts"][0]["named_by"] == ["f1"]


def test_the_self_declared_flag_neither_helps_nor_hurts():
    """The model's own boolean is recorded and ignored.

    A finding that asserts a conflict while resting on one video has not found
    one, and a finding that cites both sides has, whatever it ticked. Scoring the
    flag would measure willingness to claim rather than what was retrieved.
    """
    asserted = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                contested=True,
                citations=(
                    Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),
                    Citation(video_id="vidB", start_seconds=200.0, quote="on every claim"),
                ),
            )
        ],
        conflicts=[BOTH_SIDES],
    )
    assert asserted["scores"]["contested_coverage"] == 0.0
    assert asserted["self_declared_contested"] == 1

    unflagged = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                contested=False,
                citations=(
                    Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),
                    Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),
                ),
            )
        ],
        conflicts=[BOTH_SIDES],
    )
    assert unflagged["scores"]["contested_coverage"] == 1.0
    assert unflagged["self_declared_contested"] == 0


def test_restating_one_disagreement_does_not_make_it_two():
    """The denominator is fixed by retrieval, so verbosity cannot move it."""
    findings = [
        Finding(
            id=f"f{index}",
            criterion="put numbers on every claim",
            detail="",
            citations=(
                Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),
                Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),
            ),
        )
        for index in range(1, 6)
    ]
    cell = scored(findings, conflicts=[BOTH_SIDES])
    assert cell["scores"]["contested_coverage"] == 1.0
    assert cell["conflicts_named"] == 1
    assert cell["conflicts_in_context"] == 1


def test_a_disagreement_only_half_retrieved_is_not_available_to_score():
    """One side in context is not a disagreement the model could have named."""
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
            )
        ],
        conflicts=[BOTH_SIDES],
        retrieved_chunk_ids=["chunk:vidB:0"],
    )
    assert cell["conflicts_in_context"] == 0
    assert cell["scores"]["contested_coverage"] is None


def test_a_setup_that_produced_nothing_scores_none_not_zero():
    """Zero findings is unmeasured, not perfectly imprecise.

    Reporting 0.0 would rank a setup that crashed alongside one that made only
    ungrounded claims, which is the opposite of what the row is for.
    """
    cell = scored([])
    assert cell["scores"]["evidence_precision"] is None
    assert cell["scores"]["provenance"] is None
    assert cell["scores"]["criteria_recall"] == 0.0


def test_only_applicable_criteria_set_the_recall_denominator():
    """A resume-only rule cannot be reached by reviewing a website.

    Counting "keep it to one page" against a portfolio review would report a
    ceiling the system cannot hit for reasons that are not about retrieval, so
    the headline recall is over the applicable subset and the whole-list number
    is carried beside it.
    """
    criteria = [
        criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
        criterion(
            "c2", "put numbers on every claim", video_id="vidB", seconds=200.0, kinds=("resume",)
        ),
    ]
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="keep the document to one page",
                detail="",
                citations=(
                    Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),
                ),
            )
        ],
        criteria,
        provenance={"f1": ["chunk:vidA:0"]},
    )
    assert cell["criteria_applicable"] == 1
    assert cell["scores"]["criteria_recall"] == 1.0
    assert cell["criteria_recall_all"] == 0.5


def test_scoring_records_a_leak_it_finds_in_the_retrieved_ids():
    data = dataset([criterion("c1", "one page", video_id="vidA", seconds=100.0)])
    critique = SetupCritique(
        setup="rag_llm_filtered",
        findings=[],
        retrieved_chunk_ids=["chunk:HELD:2"],
        retrieved_video_ids=["HELD"],
    )
    cell = score_critique(critique, data, embedding_matcher(fake_embed), chunk_text)
    assert cell["held_out_leaks"] == 2
    assert "chunk:HELD:2" in cell["held_out_leak_ids"]


# ── parsing and the run file ──────────────────────────────────────────────


def test_parsing_drops_findings_and_citations_it_cannot_use():
    findings = parse_findings(
        {
            "findings": [
                {"criterion": "", "detail": "no rule at all"},
                {
                    "criterion": "cite your sources",
                    "detail": "d",
                    "citations": [
                        {"video_id": "vidA", "start_seconds": 100.0, "quote": "ok"},
                        {"video_id": "vidA", "quote": "no timestamp"},
                        {"start_seconds": 1.0, "quote": "no video"},
                    ],
                },
            ]
        }
    )
    assert len(findings) == 1
    assert len(findings[0].citations) == 1
    assert findings[0].id == "f02"


def test_the_run_id_names_the_held_out_video():
    """Two runs holding out different videos must not overwrite each other.

    ``save_run`` writes by run id with no collision guard, and a timestamp-only
    id is exactly the collision that would leave one experiment's scores filed
    under the other's held-out video while looking entirely healthy.
    """
    data = dataset([criterion("c1", "one page", video_id="vidA", seconds=100.0)])
    run = build_run(
        data,
        [{"setup": "rag_llm_filtered", "held_out_leaks": 0}],
        config={},
        baseline="rag_llm_filtered",
        now=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
    )
    assert run["run_id"] == "critique-HELD-20260810-120000"
    assert run["kind"] == "critique-eval"
    assert run["metrics"] == CRITIQUE_METRICS
    assert run["held_out_leaks"] == 0


def test_the_run_has_no_composite_metric():
    """The four metrics trade against each other and must not be blended.

    A weighted composite is the shape that lets a run look better while getting
    worse — precision bought by reporting fewer findings costs recall, and a
    blend hides the trade. Later slices have to beat the baseline on the metric
    they claim to move.
    """
    assert "composite" not in CRITIQUE_METRICS


# ── the committed dataset, against the real corpus ────────────────────────


def _corpus_available() -> bool:
    return Path(".yt-agent/chroma").exists()


@pytest.mark.skipif(not _corpus_available(), reason="no local corpus")
def test_every_committed_criterion_resolves_at_its_own_timestamp():
    """The ground truth has to pass the bar it sets for the system.

    A criterion whose quote is not in the transcript where it says it is would
    let the dataset's author put words in the held-out expert's mouth, and every
    recall number measured against it would be measuring that instead.
    """
    from src.config import load_settings
    from src.evals.critique_run import chunk_text_lookup

    data = load_critique_dataset()
    checks = verify_dataset(data, chunk_text_lookup(load_settings()))
    unresolved = [c for c in checks if not c.resolved]
    assert unresolved == [], [
        (c.citation.start_seconds, c.citation.quote, c.reason) for c in unresolved
    ]


# ── the padding attack, and the grounding gate that closes it ─────────────


def restating_findings(criteria, quote_video="vidA", quote_seconds=100.0):
    """One finding per criterion, all stapled to the same single real quote.

    The attack an independent reviewer used to sweep this scorer: recite the
    criteria as generic advice you already knew, staple one genuine citation to
    every line, and score 1.000 on all four.

    Each finding declares that one chunk as its own retrieval, so the
    retrieval-provenance gate has nothing to object to and the *exclusivity*
    gate is what has to kill this. Leaving provenance off would make the cell
    ungraded and prove nothing about the rule under test.
    """
    quote = TRANSCRIPT[(quote_video, quote_seconds)][0][1][:40]
    chunk_id = TRANSCRIPT[(quote_video, quote_seconds)][0][0]
    return [
        Finding(
            id=f"f{index:02d}",
            criterion=c.criterion,
            detail="",
            citations=(Citation(video_id=quote_video, start_seconds=quote_seconds, quote=quote),),
            retrieved_chunk_ids=(chunk_id,),
        )
        for index, c in enumerate(criteria, start=1)
    ]


def test_reciting_every_criterion_on_one_shared_quote_scores_nothing():
    """The headline regression test for this harness.

    Eighteen findings that restate the criteria, each carrying the *same* real
    quote, used to score criteria_recall 1.000 · evidence_precision 1.000 ·
    provenance 1.000. Provenance stays 1.000 — the quote really is in the
    transcript, and that metric only ever claimed the words are there — but no
    finding owns exclusive evidence, so nothing is grounded and neither recall
    nor precision credits any of it.
    """
    criteria = [
        criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
        criterion("c2", "put numbers on every claim", video_id="vidB", seconds=200.0),
        criterion("c3", "drop the redundant skills list", video_id="vidC", seconds=300.0),
    ]
    cell = scored(restating_findings(criteria), criteria)

    assert cell["scores"]["criteria_recall"] == 0.0
    assert cell["scores"]["evidence_precision"] == 0.0
    assert cell["scores"]["provenance"] == 1.0
    assert cell["findings_sharing_evidence"] == 3


def test_padding_the_output_cannot_buy_recall():
    """Recall must be invariant to how many findings are emitted.

    Same one grounded point, then nine more restating other criteria on the same
    quote. If extra findings were free, recall would climb; it must not move.
    """
    criteria = [
        criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
        criterion("c2", "put numbers on every claim", video_id="vidB", seconds=200.0),
    ]
    honest = Finding(
        id="f00",
        criterion="put numbers on every claim",
        detail="",
        citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
        retrieved_chunk_ids=("chunk:vidB:0",),
    )
    lean = scored([honest], criteria)
    padded = scored([honest, *restating_findings(criteria)], criteria)

    assert lean["scores"]["criteria_recall"] == padded["scores"]["criteria_recall"]
    # And padding is now actively expensive, which is the trade the docstring
    # claims: eleven findings resting on two pieces of evidence.
    assert padded["scores"]["evidence_precision"] < lean["scores"]["evidence_precision"]


def test_two_findings_sharing_their_only_evidence_ground_neither():
    """Symmetric on purpose — no tie-break may decide the score.

    Awarding the shared chunk to whichever finding came first would make recall
    depend on output order, and this scorer already had one source of
    run-to-run drift too many.
    """
    checks = {
        "f1": [
            CitationCheck(
                citation=Citation(video_id="vidA", start_seconds=100.0, quote="q"),
                resolved=True,
                reason="ok",
                chunk_id="chunk:vidA:0",
            )
        ],
        "f2": [
            CitationCheck(
                citation=Citation(video_id="vidA", start_seconds=100.0, quote="q"),
                resolved=True,
                reason="ok",
                chunk_id="chunk:vidA:0",
            )
        ],
    }
    findings = [
        Finding(id="f1", criterion="a", detail=""),
        Finding(id="f2", criterion="b", detail=""),
    ]
    exclusive = ground_findings(findings, checks)

    assert exclusive == {"f1": [], "f2": []}
    assert checks["f1"][0].shared is True


def test_a_matched_criterion_whose_finding_is_ungrounded_is_reported_not_hidden():
    """ "You made this point but the corpus did not pay for it" is its own state.

    Silently unmatching it would look identical to never having made the point,
    which is a different failure and a different fix.
    """
    criteria = [criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0)]
    cell = scored(
        [Finding(id="f1", criterion="keep the document to one page", detail="")], criteria
    )

    row = cell["matches"][0]
    assert row["matched"] is True
    assert row["ungrounded"] is True
    assert row["counted"] is False
    assert cell["criteria_matched_ungrounded"] == 1


# ── the second padding attack, and the retrieval-provenance gate ──────────
#
# The first attack gave every recited finding the *same* quote and exclusivity
# killed it. This is the one edit that survived that: give each recited finding
# its **own distinct** real chunk out of the pool the honest run retrieved.
# Measured on the committed baseline it scored evidence_precision 1.000 and a
# recall ceiling of 0.526 against the honest run's 0.556 and 0.158 — three times
# the recall for reciting advice the model already knew.


def recited_with_distinct_chunks(criteria, pool, *, declare_provenance=None):
    """The second attack: one recited finding per criterion, one chunk each.

    ``pool`` is ``(video_id, seconds)`` pairs — the honest run's own retrieved
    chunks, which is what makes this an attack rather than a fabrication: every
    quote is real, every timestamp is right, and every citation is exclusive.

    ``declare_provenance`` is what the attacking engine claims each finding
    retrieved. ``None`` is the honest state of a system that emitted all of them
    from one shared pool.
    """
    findings = []
    for index, (c, (video_id, seconds)) in enumerate(zip(criteria, pool), start=1):
        chunk_id, text = TRANSCRIPT[(video_id, seconds)][0]
        findings.append(
            Finding(
                id=f"a{index:02d}",
                criterion=c.criterion,
                detail="",
                citations=(Citation(video_id=video_id, start_seconds=seconds, quote=text[:40]),),
                retrieved_chunk_ids=(
                    None if declare_provenance is None else tuple(declare_provenance[index - 1])
                ),
            )
        )
    return findings


ATTACK_POOL = [("vidA", 100.0), ("vidB", 200.0), ("vidC", 300.0)]


def attack_criteria():
    return [
        criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
        criterion("c2", "put numbers on every claim", video_id="vidB", seconds=200.0),
        criterion("c3", "drop the redundant skills list", video_id="vidC", seconds=300.0),
    ]


def test_the_distinct_chunk_attack_sweeps_the_old_gate():
    """First, reproduce the defect — otherwise the fix proves nothing.

    Under the gate that shipped before 2026-08-11 this output is perfect on both
    scored metrics while having applied nothing to the document. That is the
    number ``KNOWN_GAP_attack2.md`` measured on the real corpus, at this
    fixture's scale.
    """
    criteria = attack_criteria()
    cell = scored(
        recited_with_distinct_chunks(criteria, ATTACK_POOL),
        criteria,
        retrieved_chunk_ids=["chunk:vidA:0", "chunk:vidB:0", "chunk:vidC:0"],
        gate=GATE_EXCLUSIVE,
    )

    assert cell["scores"]["evidence_precision"] == 1.0
    assert cell["scores"]["criteria_recall"] == 1.0
    assert cell["findings_sharing_evidence"] == 0


def test_the_distinct_chunk_attack_earns_nothing_under_the_provenance_gate():
    """The regression test for the second attack.

    The attacking engine emitted every finding from one shared retrieval, so it
    cannot say what any individual finding retrieved. Under the shipped gate
    that is **ungraded**, not passed: the two scores that claim the corpus
    reached a conclusion go to ``None``, the ungated pair is kept beside them so
    the sweep is still visible, and ``provenance`` stays 1.000 because the quotes
    really are in the transcript — that metric never claimed anything else.
    """
    criteria = attack_criteria()
    cell = scored(
        recited_with_distinct_chunks(criteria, ATTACK_POOL),
        criteria,
        retrieved_chunk_ids=["chunk:vidA:0", "chunk:vidB:0", "chunk:vidC:0"],
    )

    assert cell["scores"]["criteria_recall"] is None
    assert cell["scores"]["evidence_precision"] is None
    assert cell["scores"]["provenance"] == 1.0
    assert cell["gradable"] is False
    assert "cannot be graded" in cell["ungradable_reason"]
    # Kept, not hidden: this is what the old gate would have published.
    assert cell["criteria_recall_ungated"] == 1.0
    assert cell["evidence_precision_ungated"] == 1.0
    assert cell["criteria_recall_all"] is None
    assert cell["score_spread"]["criteria_recall_max"] is None


def test_a_finding_that_cites_a_chunk_its_own_retrieval_never_returned_grounds_nothing():
    """The gate's actual bite, on an engine that *can* answer the question.

    Each finding declares one chunk and cites a different one. Every quote
    resolves, every citation is exclusive, and none of it is that finding's own
    evidence — so nothing is grounded and both scores are zero rather than
    ungraded, because here the harness really did measure something.
    """
    criteria = attack_criteria()
    # a01 declares vidA and cites vidA (honest); a02 and a03 declare a chunk they
    # did not cite and cite one they did not retrieve.
    findings = recited_with_distinct_chunks(
        criteria,
        ATTACK_POOL,
        declare_provenance=[["chunk:vidA:0"], ["chunk:vidA:0"], ["chunk:vidB:0"]],
    )
    cell = scored(
        findings,
        criteria,
        retrieved_chunk_ids=["chunk:vidA:0", "chunk:vidB:0", "chunk:vidC:0"],
    )

    assert cell["gradable"] is True
    assert cell["scores"]["evidence_precision"] == 0.3333
    assert cell["scores"]["criteria_recall"] == 0.3333
    assert cell["citations_off_retrieval"] == 2
    assert cell["evidence_precision_ungated"] == 1.0
    # The diagnosis is on the citation, and it is not the fabrication one: the
    # quote is real and in the transcript, it is simply not this finding's.
    checks = {row["id"]: row["citation_checks"][0] for row in cell["findings"]}
    assert checks["a02"]["resolved"] is True
    assert checks["a02"]["off_retrieval"] is True
    assert checks["a01"]["off_retrieval"] is False
    assert cell["fabricated_citations"] == []


def test_a_chunk_grabbed_off_retrieval_does_not_spoil_the_finding_that_did_retrieve_it():
    """Order matters: the gate runs before exclusivity, not after.

    If ownership were computed over all resolving citations and *then* filtered,
    an attacker could delete an honest finding's grounding by citing its chunk —
    turning the gate into a weapon against the arm it is meant to protect.
    """
    honest = Finding(
        id="f1",
        criterion="keep the document to one page",
        detail="",
        citations=(Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),),
        retrieved_chunk_ids=("chunk:vidA:0",),
    )
    parasite = Finding(
        id="f2",
        criterion="put numbers on every claim",
        detail="",
        citations=(Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),),
        retrieved_chunk_ids=("chunk:vidB:0",),
    )
    cell = scored([honest, parasite])

    assert cell["findings"][0]["grounded"] is True
    assert cell["findings"][1]["grounded"] is False
    assert cell["scores"]["evidence_precision"] == 0.5
    # Under the old gate the two share a chunk and *neither* is grounded.
    assert cell["evidence_precision_ungated"] == 0.0


def test_an_engine_that_declares_nothing_is_ungraded_and_says_which_findings():
    findings = [
        Finding(id="f1", criterion="a", detail="", retrieved_chunk_ids=("chunk:vidA:0",)),
        Finding(id="f2", criterion="b", detail=""),
    ]
    gradable, reason = gate_verdict(findings, GATE_PROVENANCE)

    assert gradable is False
    assert "1 of 2 findings" in reason
    # The old gate has nothing to say about provenance and grades either way.
    assert gate_verdict(findings, GATE_EXCLUSIVE) == (True, None)


def test_an_empty_declaration_is_a_claim_and_grounds_nothing():
    """``()`` and ``None`` are different answers and must not collapse.

    ``None`` is "this engine cannot say" and takes the cell to ungraded. ``()``
    is "this finding retrieved nothing", which is a statement the engine made,
    and a finding that retrieved nothing has no evidence of its own however real
    its quote is.
    """
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="keep the document to one page",
                detail="",
                citations=(
                    Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),
                ),
                retrieved_chunk_ids=(),
            )
        ]
    )

    assert cell["gradable"] is True
    assert cell["scores"]["evidence_precision"] == 0.0
    assert cell["scores"]["criteria_recall"] == 0.0
    assert cell["scores"]["provenance"] == 1.0


def test_the_gate_in_force_is_recorded_on_every_cell_and_on_the_run():
    """A number whose gate cannot be named cannot be compared with anything."""
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="keep the document to one page",
                detail="",
                citations=(
                    Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one page"),
                ),
                retrieved_chunk_ids=("chunk:vidA:0",),
            )
        ]
    )
    assert cell["grounding_gate"] == GATE_PROVENANCE
    assert cell["grounding_gate_applied"] == GATE_PROVENANCE
    assert cell["findings_with_provenance"] == 1

    run = build_run(dataset([]), [cell], config={}, baseline="rag_llm_filtered")
    assert run["grounding_gate"] == GATE_PROVENANCE
    assert run["config"]["grounding_gate"] == GATE_PROVENANCE
    assert run["ungraded_cells"] == []


def test_the_shipped_default_is_the_provenance_gate():
    """The default is the fix, not the hole.

    A scorer that ships with the known hole open republishes the overstated
    number every time anybody runs it, and "there is a flag" is not a defence
    when the number is the claim the project rests on.
    """
    assert DEFAULT_GROUNDING_GATE == GATE_PROVENANCE


def test_an_unknown_gate_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown grounding gate"):
        scored([], gate="whatever_looks_best")


def test_a_model_cannot_declare_its_own_provenance():
    """The engine records provenance; the model is never asked for it.

    A model allowed to write this field would list the chunks it wanted to cite,
    which is the gate agreeing with the attack.
    """
    payload = {
        "findings": [
            {
                "criterion": "cite your sources",
                "detail": "d",
                "retrieved_chunk_ids": ["chunk:vidA:0"],
                "citations": [{"video_id": "vidA", "start_seconds": 100.0, "quote": "ok"}],
            }
        ]
    }
    assert parse_findings(payload)[0].retrieved_chunk_ids is None
    # A stored run is the engine's own record, so re-scoring reads it back.
    assert parse_findings(payload, trust_provenance=True)[0].retrieved_chunk_ids == (
        "chunk:vidA:0",
    )


def test_provenance_survives_the_round_trip_through_a_run_file():
    """Re-scoring a run committed under the gate must not silently ungrade it."""
    finding = Finding(
        id="f1",
        criterion="a",
        detail="",
        retrieved_chunk_ids=("chunk:vidA:0",),
    )
    assert finding.to_dict()["retrieved_chunk_ids"] == ["chunk:vidA:0"]
    restored = parse_findings({"findings": [finding.to_dict()]}, trust_provenance=True)
    assert restored[0].retrieved_chunk_ids == ("chunk:vidA:0",)
    assert parse_findings({"findings": [{"criterion": "a"}]})[0].retrieved_chunk_ids is None


def test_a_finding_the_provenance_map_does_not_name_stays_ungradable():
    """A missing key is "unknown", never "retrieved nothing".

    Guessing the second would let a build that lost track of one rubric's unit
    keep grading the whole arm.
    """
    findings = attach_provenance(
        [Finding(id="f1", criterion="a", detail=""), Finding(id="f2", criterion="b", detail="")],
        {"f1": ["chunk:vidA:0"]},
    )
    assert findings[0].retrieved_chunk_ids == ("chunk:vidA:0",)
    assert findings[1].retrieved_chunk_ids is None


# ── reproducibility of the matcher ────────────────────────────────────────


def test_a_disagreeing_matcher_is_resolved_by_majority_vote():
    """The blocking defect: one call's pairing moved recall by ±20% relative.

    Here three runs of five pair c1 with f1 and two leave it unmatched; the
    consensus is the majority, and the agreement is recorded so a 3/5 verdict
    does not render identically to a unanimous one.
    """
    criteria = [criterion("c1", "keep it to one page", video_id="vidA", seconds=100.0)]
    findings = [Finding(id="f1", criterion="one page only", detail="")]
    runs = [{"c1": "f1"}, {"c1": "f1"}, {"c1": None}, {"c1": "f1"}, {"c1": None}]

    rows = consensus(criteria, findings, runs)

    assert rows[0].finding_id == "f1"
    assert rows[0].agreement == pytest.approx(0.6)


def test_a_criterion_most_runs_left_alone_stays_unmatched():
    criteria = [criterion("c1", "keep it to one page", video_id="vidA", seconds=100.0)]
    findings = [Finding(id="f1", criterion="something else", detail="")]
    runs = [{"c1": None}, {"c1": "f1"}, {"c1": None}]

    assert consensus(criteria, findings, runs)[0].matched is False


def test_the_run_reports_the_spread_its_own_matcher_produced():
    """A recall of 0.5 that came out of 0.0-1.0 is not a measurement.

    A later slice has to beat the baseline by more than this range, so the range
    travels with the score instead of being discoverable only by re-running.
    """
    criteria = [
        criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
        criterion("c2", "put numbers on every claim", video_id="vidB", seconds=200.0),
    ]
    findings = [
        Finding(
            id="f1",
            criterion="one page",
            detail="",
            citations=(Citation(video_id="vidA", start_seconds=100.0, quote="keep it to one"),),
            retrieved_chunk_ids=("chunk:vidA:0",),
        ),
        Finding(
            id="f2",
            criterion="numbers",
            detail="",
            citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
            retrieved_chunk_ids=("chunk:vidB:0",),
        ),
    ]
    flaky_runs = [{"c1": "f1", "c2": "f2"}, {"c1": "f1", "c2": None}, {"c1": "f1", "c2": "f2"}]

    def matcher(cs, fs):
        return MatchResult(matches=consensus(cs, fs, flaky_runs), runs=flaky_runs)

    cell = score_critique(
        SetupCritique(setup="s", findings=findings), dataset(criteria), matcher, chunk_text
    )

    assert cell["score_spread"]["criteria_recall_min"] == pytest.approx(0.5)
    assert cell["score_spread"]["criteria_recall_max"] == pytest.approx(1.0)
    assert cell["match_repeats"] == 3


def test_repeating_a_matcher_records_every_run_that_voted():
    calls = {"n": 0}

    def flaky(cs, fs):
        calls["n"] += 1
        chosen = "f1" if calls["n"] % 2 else None
        rows = [CriterionMatch(criterion=c) for c in cs]
        if chosen:
            rows[0].finding_id = chosen
        return MatchResult(matches=rows, runs=[{cs[0].id: chosen}])

    criteria = [criterion("c1", "one page", video_id="vidA", seconds=100.0)]
    result = repeated_matcher(flaky, repeats=5)(
        criteria, [Finding(id="f1", criterion="x", detail="")]
    )

    assert len(result.runs) == 5
    assert result.matches[0].finding_id == "f1"  # 3 of 5


# ── grouped recall ────────────────────────────────────────────────────────


def test_near_duplicate_criteria_do_not_impose_a_recall_ceiling():
    """One-to-one matching means one correct point can satisfy only one of a pair.

    c12 and c24 are both "link to your work"; a system that makes that point
    once can never reach both, which is a ceiling unrelated to retrieval. The
    grouped score reports recall without it; the per-criterion score stays the
    conservative headline.
    """
    pair = [
        Criterion(
            id="c12",
            criterion="link to your profiles",
            quote=TRANSCRIPT[("vidA", 100.0)][0][1][:20],
            video_id="vidA",
            start_seconds=100.0,
            applies_to=("portfolio",),
            group="g_links",
        ),
        Criterion(
            id="c24",
            criterion="link each project to its code",
            quote=TRANSCRIPT[("vidA", 100.0)][0][1][:20],
            video_id="vidA",
            start_seconds=100.0,
            applies_to=("portfolio",),
            group="g_links",
        ),
    ]
    cell = scored(
        [
            Finding(
                id="f1",
                criterion="link to your profiles",
                detail="",
                citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
            )
        ],
        pair,
        provenance={"f1": ["chunk:vidB:0"]},
    )

    assert cell["scores"]["criteria_recall"] == 0.5
    assert cell["criteria_groups"] == 1
    assert cell["criteria_recall_grouped"] == 1.0


@pytest.mark.skipif(not _corpus_available(), reason="no local corpus")
def test_every_criterion_excluded_from_the_artifact_says_why():
    """``applies_to`` is the softest judgement in the dataset and it moves the score.

    An independent audit found all six original exclusions removed criteria the
    system had *missed* and none it had reached — worth +33% relative to the
    headline. That asymmetry is not proof of bias, but a single author excluding
    criteria in the same file that reports the score needs the reason written
    down where a reviewer reads it.
    """
    data = load_critique_dataset()
    silent = [c.id for c in data.criteria if not c.applies(data.artifact_kind) and not c.note]
    assert silent == []


@pytest.mark.skipif(not _corpus_available(), reason="no local corpus")
def test_the_committed_runs_citations_resolve_against_the_real_corpus():
    """Recompute the *run's* provenance, not just the dataset's own quotes.

    ``test_committed_runs`` reads the stored per-citation verdicts, which is the
    right check for CI with no Chroma; this one re-derives them from the
    transcript, so a run whose stored checks were fabricated fails here.
    """
    import json

    from src.config import load_settings
    from src.evals.critique import Citation, check_citation
    from src.evals.critique_run import chunk_text_lookup

    lookup = chunk_text_lookup(load_settings())
    runs = sorted(Path("evals/runs").glob("critique-*.json"))
    assert runs, "no committed critique run"
    for path in runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        for cell in run["cells"]:
            for finding in cell["findings"]:
                for stored in finding["citation_checks"]:
                    fresh = check_citation(
                        Citation(
                            video_id=stored["video_id"],
                            start_seconds=stored["start_seconds"],
                            quote=stored["quote"],
                        ),
                        lookup,
                    )
                    assert fresh.resolved == stored["resolved"], (finding["id"], stored)
                    assert fresh.chunk_id == stored["chunk_id"]


def test_a_cached_pairing_is_reused_and_returns_the_identical_score(tmp_path):
    """The other half of the reproducibility fix.

    Voting narrows the matcher's noise; caching means a re-score does not
    re-roll it at all, so a number that moves between runs is a number somebody
    changed something to move.
    """
    from src.evals.critique_run import cached_matcher

    calls = {"n": 0}

    def inner(cs, fs):
        calls["n"] += 1
        rows = [CriterionMatch(criterion=c) for c in cs]
        rows[0].finding_id = "f1"
        rows[0].why = "same rule"
        return MatchResult(matches=rows, runs=[{cs[0].id: "f1"}])

    criteria = [criterion("c1", "one page", video_id="vidA", seconds=100.0)]
    findings = [Finding(id="f1", criterion="keep it to one page", detail="")]
    match = cached_matcher(inner, cache_dir=tmp_path)

    first = match(criteria, findings)
    second = match(criteria, findings)

    assert calls["n"] == 1
    assert first.matches[0].finding_id == second.matches[0].finding_id == "f1"
    assert second.matches[0].why == "same rule"


def test_changing_what_is_matched_invalidates_the_cached_pairing(tmp_path):
    """A cache that answered for a *different* matrix would be worse than none."""
    from src.evals.critique_run import cached_matcher

    calls = {"n": 0}

    def inner(cs, fs):
        calls["n"] += 1
        return MatchResult(
            matches=[CriterionMatch(criterion=c) for c in cs],
            runs=[{cs[0].id: None}],
        )

    criteria = [criterion("c1", "one page", video_id="vidA", seconds=100.0)]
    match = cached_matcher(inner, cache_dir=tmp_path)
    match(criteria, [Finding(id="f1", criterion="keep it to one page", detail="")])
    match(criteria, [Finding(id="f1", criterion="a completely different rule", detail="")])

    assert calls["n"] == 2


def test_the_held_out_transcript_cannot_be_read_through_the_raw_store():
    """Unreachable on today's path, which is how a leak gets built in later.

    The raw store lives in src/rag and has no exclusion of its own, so the
    harness wraps it in something that fails loud rather than trusting that
    nothing will ever call it.
    """
    from src.evals.critique_run import _HeldOutBlocked

    class Inner:
        def get_raw_document(self, video_id):
            return f"full transcript of {video_id}"

        def join_raw_text(self, video_id):
            return "every word"

    guarded = _HeldOutBlocked(Inner(), ["HELD"])

    assert guarded.get_raw_document("vidA") == "full transcript of vidA"
    with pytest.raises(ValueError, match="held out"):
        guarded.get_raw_document("HELD")
    with pytest.raises(ValueError, match="held out"):
        guarded.join_raw_text("HELD")


# ── V6: the rubric-driven arm inside the same harness ─────────────────────
#
# The gate for V6 is a recall comparison published beside the chunk-dump
# baseline, so the thing that has to be tested here is the *wiring*: that the
# rubric reviewer reaches this harness as a scorable setup without losing the
# properties the baseline is held to. What a pack review decides is covered in
# ``tests/documents/test_rubric_review.py``; what follows is only about the
# translation into a cell.


def _stub_pack(topic: str, name: str, rubrics: list, chunk_ids: list[str], video_ids: list[str]):
    from types import SimpleNamespace

    return SimpleNamespace(
        topic=topic,
        name=name,
        artifact="resume",
        rubrics=rubrics,
        units=[
            SimpleNamespace(
                # The id every rubric's ``unit_id`` points back at — how the
                # provenance gate learns which chunks *this* rubric was
                # distilled from rather than which the whole build saw.
                unit_id="u1",
                chunk_ids=list(chunk_ids),
                video_ids=list(video_ids),
            )
        ],
    )


def _stub_rubric(rubric_id: str, criterion: str, video_id: str, start: float, quote: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        rubric_id=rubric_id,
        criterion=criterion,
        check="Check it.",
        why="Because.",
        unit_id="u1",
        unit_kind="raptor",
        unit_title="Theme",
        creators=["A Recruiter"],
        contested=False,
        evidence=[
            SimpleNamespace(
                video_id=video_id,
                chunk_id=f"chunk:{video_id}:1",
                quote=quote,
                start_seconds=start,
                channel_name="A Recruiter",
                title="How to write it",
                youtube_url=lambda: f"https://www.youtube.com/watch?v={video_id}",
            )
        ],
    )


def _stub_document():
    from src.documents.models import Document, DocumentSection

    return Document(
        id="doc:1",
        url="https://example.com",
        requested_url="https://example.com",
        title="Portfolio",
        sections=[
            DocumentSection(index=index, heading=f"Section {index}", text="some body text")
            for index in range(3)
        ],
    )


class _ScriptedLlm:
    """One JSON reply per pack, in order."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)

    def invoke(self, messages):
        import json
        from types import SimpleNamespace

        return SimpleNamespace(content=json.dumps(self.replies.pop(0)))


def test_the_rubric_arm_scores_only_the_rubrics_the_document_failed(monkeypatch):
    """A pack of rules is not a critique of a document.

    Every rubric could be turned into a finding with a real corpus citation
    attached, and that is exactly the padding attack ``KNOWN_GAP_attack2.md``
    documents — it beats the honest baseline while having applied nothing. So a
    rubric earns a finding only when the reviewer failed this document on it,
    and this asserts the harness inherits that rule rather than re-deriving one.
    """
    from src.documents import rubric_review as rubric_module
    from src.evals.critique_run import rubric_critique

    pack = _stub_pack(
        "resume-design",
        "Resume design",
        [
            _stub_rubric("r0101", "Quantify every outcome.", "vidA", 100.0, "say the number"),
            _stub_rubric("r0102", "Keep it to one page.", "vidB", 200.0, "one page only"),
            _stub_rubric("r0103", "Put dates on the right.", "vidC", 300.0, "dates on the right"),
        ],
        ["chunk:vidA:1", "chunk:vidB:1"],
        ["vidA", "vidB"],
    )
    monkeypatch.setattr(rubric_module, "load_review_packs", lambda _dir=None: [pack])

    dataset = CritiqueDataset(
        held_out_video_id="HELD",
        held_out_title="held out",
        artifact_id="doc:1",
        artifact_url="https://example.com",
        artifact_kind="resume",
        criteria=(),
    )
    result = rubric_critique(
        _ScriptedLlm(
            [
                {
                    "verdicts": [
                        {
                            "rubric_id": "r0101",
                            "verdict": "fail",
                            "severity": "major",
                            "finding": "no numbers anywhere",
                            "sections": [1],
                        },
                        {"rubric_id": "r0102", "verdict": "pass"},
                        {"rubric_id": "r0103", "verdict": "n-a"},
                    ]
                }
            ]
        ),
        dataset,
        _stub_document(),
        "review this",
    )

    assert result.error is None
    assert [f.id for f in result.findings] == ["resume-design:r0101"]
    assert result.findings[0].criterion == "Quantify every outcome."
    assert result.findings[0].detail == "no numbers anywhere"
    # The citation is the pack's, not the model's — the reviewer is never asked
    # for a video id or a timestamp at all.
    assert [(c.video_id, c.start_seconds) for c in result.findings[0].citations] == [
        ("vidA", 100.0)
    ]
    # And it carries the narrower thing the gate grades on: the chunks of the one
    # unit *this rubric* was distilled from, not everything the build saw.
    assert result.findings[0].retrieved_chunk_ids == ("chunk:vidA:1", "chunk:vidB:1")


def test_a_rubric_arm_is_graded_because_it_can_say_what_each_finding_retrieved():
    """The asymmetry the gate exposes, stated as a test.

    A rubric is distilled from exactly one unit, so "what this finding
    retrieved" has an answer. A chunk-dump critique emits every finding from one
    shared pool and has none — which is why one of these arms is graded and the
    other is not.
    """
    from src.evals.critique_run import pack_finding_provenance

    pack = _stub_pack(
        "resume-design",
        "Resume design",
        [
            _stub_rubric("r0101", "Quantify.", "vidA", 100.0, "say the number"),
            _stub_rubric("r0102", "One page.", "vidB", 200.0, "one page only"),
        ],
        ["chunk:vidA:1", "chunk:vidB:1"],
        ["vidA", "vidB"],
    )
    bare = pack_finding_provenance([pack])
    assert bare == {
        "r0101": ["chunk:vidA:1", "chunk:vidB:1"],
        "r0102": ["chunk:vidA:1", "chunk:vidB:1"],
    }
    # Qualified by topic for the reviewer arm, whose finding ids are, because
    # ``r0101`` exists in all four packs.
    assert set(pack_finding_provenance([pack], qualify=True)) == {
        "resume-design:r0101",
        "resume-design:r0102",
    }


def test_the_d2_arms_carry_per_finding_provenance_too():
    """The pack ablation goes through the same scorer, so it needs the same field.

    Without it every D2 arm would be ungraded on the metric the ablation exists
    to compare, for a reason that has nothing to do with the arms.
    """
    from src.evals.pack_ablation import pack_as_critique

    pack = _stub_pack(
        "resume-design",
        "Resume design",
        [_stub_rubric("r0101", "Quantify.", "vidA", 100.0, "say the number")],
        ["chunk:vidA:1", "chunk:vidB:1"],
        ["vidA", "vidB"],
    )
    pack.arm = "merged"

    critique = pack_as_critique(pack)
    assert critique.setup == "merged"
    assert critique.findings[0].retrieved_chunk_ids == ("chunk:vidA:1", "chunk:vidB:1")
    # The cell-wide exposure list is still the whole build, and still wider.
    assert critique.retrieved_chunk_ids == ["chunk:vidA:1", "chunk:vidB:1"]


def test_a_rubric_whose_unit_is_missing_is_left_out_rather_than_guessed():
    """A pack that cannot say where a rubric came from must not be waved through.

    Mapping it to an empty list would read as "retrieved nothing" and score it
    zero; leaving it out leaves it ``None``, which takes the whole arm to
    ungraded. The second is the honest answer to a broken build.
    """
    from src.evals.critique_run import pack_finding_provenance

    pack = _stub_pack(
        "resume-design",
        "Resume design",
        [_stub_rubric("r0101", "Quantify.", "vidA", 100.0, "say the number")],
        ["chunk:vidA:1"],
        ["vidA"],
    )
    pack.rubrics[0].unit_id = "a-unit-this-pack-does-not-have"

    assert pack_finding_provenance([pack]) == {}


def test_the_rubric_arm_reports_what_its_pack_build_was_exposed_to(monkeypatch):
    """The leak check has to see the build's chunks, not the shipped quotes.

    A held-out video that reached the pack-build prompt and merely went unquoted
    has still contaminated the experiment, so ``retrieved_chunk_ids`` carries
    every chunk the build showed a model — which is what
    :func:`src.evals.critique.held_out_leaks` then re-scans by prefix.
    """
    from src.documents import rubric_review as rubric_module
    from src.evals.critique import held_out_leaks
    from src.evals.critique_run import pack_exposure, rubric_critique

    packs = [
        _stub_pack(
            "resume-design",
            "Resume design",
            [_stub_rubric("r0101", "Quantify.", "vidA", 100.0, "say the number")],
            ["chunk:vidA:1", "chunk:HELD:4"],
            ["vidA", "HELD"],
        ),
        _stub_pack(
            "job-search",
            "Job search",
            [_stub_rubric("r0201", "Apply narrowly.", "vidB", 200.0, "pick ten roles")],
            ["chunk:vidA:1", "chunk:vidB:2"],
            ["vidA", "vidB"],
        ),
    ]
    monkeypatch.setattr(rubric_module, "load_review_packs", lambda _dir=None: packs)

    chunk_ids, video_ids = pack_exposure(packs)
    assert chunk_ids == ["chunk:HELD:4", "chunk:vidA:1", "chunk:vidB:2"]
    assert video_ids == ["HELD", "vidA", "vidB"]

    dataset = CritiqueDataset(
        held_out_video_id="HELD",
        held_out_title="held out",
        artifact_id="doc:1",
        artifact_url="https://example.com",
        artifact_kind="resume",
        criteria=(),
    )
    result = rubric_critique(
        _ScriptedLlm([{"verdicts": []}, {"verdicts": []}]),
        dataset,
        _stub_document(),
        "review this",
    )

    assert result.retrieved_chunk_ids == chunk_ids
    assert held_out_leaks(
        "HELD",
        chunk_ids=result.retrieved_chunk_ids,
        video_ids=result.retrieved_video_ids,
    ) == ["chunk:HELD:4", "video:HELD"]
    # A pack that never declared this video held out is a different experiment
    # from one that did, even when the leak count happens to agree, so the trace
    # says which of the two happened rather than leaving it to be inferred.
    assert "NOT HELD" in result.trace[0]["detail"]


def test_a_rubric_arm_with_no_packs_built_is_a_reported_cell_not_a_crash(monkeypatch):
    """One arm failing must not take the comparison down.

    The whole point of the run is the two rows beside each other; an exception
    here would delete the baseline's number as well as this one's.
    """
    from src.documents import rubric_review as rubric_module
    from src.evals.critique_run import rubric_critique

    monkeypatch.setattr(rubric_module, "load_review_packs", lambda _dir=None: [])

    dataset = CritiqueDataset(
        held_out_video_id="HELD",
        held_out_title="held out",
        artifact_id="doc:1",
        artifact_url="https://example.com",
        artifact_kind="resume",
        criteria=(),
    )
    result = rubric_critique(_ScriptedLlm([]), dataset, _stub_document(), "review this")

    assert result.findings == []
    assert result.error is not None and "packs" in result.error


# ── re-scoring a committed run under a new gate, with no model at all ──────


def _committed_run(setup: str, findings: list[dict], votes: list[dict]) -> dict:
    """The shape ``rescore_committed_run`` reads, cut to what it reads."""
    return {
        "run_id": "critique-HELD-20260810-000000",
        "baseline": setup,
        "topic": "resume-design",
        "config": {"match_repeats": len(votes)},
        "cells": [
            {
                "setup": setup,
                "findings": findings,
                "retrieved_chunk_ids": ["chunk:vidA:0", "chunk:vidB:0"],
                "retrieved_video_ids": ["vidA", "vidB"],
                "match_runs": votes,
            }
        ],
    }


def test_a_committed_run_is_rescored_from_its_own_stored_votes():
    """Re-applying an arithmetic rule must not re-roll the pairing.

    The matcher does not agree with itself, so asking it again would move the
    number for a reason that is not the rule under test — and cost a call per
    cell. The votes are already in the run file, so replaying them makes a
    rescore free and byte-for-byte reproducible.
    """
    from src.evals.critique_run import replay_matcher, rescore_committed_run

    criteria = [
        criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
        criterion("c2", "put numbers on every claim", video_id="vidB", seconds=200.0),
    ]
    stored = _committed_run(
        "rag_llm_filtered",
        [
            {
                "id": "f1",
                "criterion": "keep the document to one page",
                "detail": "",
                "citations": [
                    {
                        "video_id": "vidA",
                        "start_seconds": 100.0,
                        "quote": "keep it to one page",
                    }
                ],
            }
        ],
        [{"c1": "f1", "c2": None}] * 3,
    )
    rescored = rescore_committed_run(
        stored,
        dataset(criteria),
        replay_matcher(stored),
        chunk_text,
        gate=GATE_EXCLUSIVE,
    )

    cell = rescored["cells"][0]
    assert cell["scores"]["criteria_recall"] == 0.5
    assert cell["match_repeats"] == 3
    assert rescored["config"]["rescored_gate_from"] == GATE_EXCLUSIVE
    assert rescored["grounding_gate"] == GATE_EXCLUSIVE


def test_replaying_more_cells_than_a_run_stored_is_an_error_not_a_guess():
    """Another cell's votes are not this cell's pairing."""
    from src.evals.critique_run import replay_matcher

    stored = _committed_run("s", [], [{"c1": None}])
    match = replay_matcher(stored)
    criteria = [criterion("c1", "one page", video_id="vidA", seconds=100.0)]
    match(criteria, [])
    with pytest.raises(ValueError, match="re-score its cells once each"):
        match(criteria, [])


def test_a_run_committed_before_the_gate_is_ungraded_unless_provenance_is_supplied():
    """The deliverable this gate was written to produce.

    Every committed run predates the ``retrieved_chunk_ids`` field. A pack arm's
    answer is still on disk and can be handed back; a retrieval arm's never
    existed. So the same rescore call grades one and ungrades the other, and
    neither outcome is a choice anybody made per-arm.
    """
    from src.evals.critique_run import replay_matcher, rescore_committed_run

    criteria = [criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0)]
    findings = [
        {
            "id": "f1",
            "criterion": "keep the document to one page",
            "detail": "",
            "citations": [
                {"video_id": "vidA", "start_seconds": 100.0, "quote": "keep it to one page"}
            ],
        }
    ]
    votes = [{"c1": "f1"}]

    stored = _committed_run("rag_llm_filtered", findings, votes)
    bare = rescore_committed_run(stored, dataset(criteria), replay_matcher(stored), chunk_text)
    assert bare["cells"][0]["scores"]["criteria_recall"] is None
    assert bare["cells"][0]["criteria_recall_ungated"] == 1.0
    assert bare["ungraded_cells"] == ["rag_llm_filtered"]

    stored = _committed_run("merged", findings, votes)
    graded = rescore_committed_run(
        stored,
        dataset(criteria),
        replay_matcher(stored),
        chunk_text,
        provenance={"merged": {"f1": ["chunk:vidA:0"]}},
    )
    assert graded["cells"][0]["scores"]["criteria_recall"] == 1.0
    assert graded["ungraded_cells"] == []
