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
    Citation,
    CitationCheck,
    Criterion,
    CriterionMatch,
    CritiqueDataset,
    Finding,
    MatchResult,
    SetupCritique,
    build_run,
    check_citation,
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


def scored(findings, criteria=None):
    data = dataset(
        criteria
        or [
            criterion("c1", "keep the document to one page", video_id="vidA", seconds=100.0),
            criterion("c2", "put numbers on every claim", video_id="vidB", seconds=200.0),
        ]
    )
    critique = SetupCritique(
        setup="rag_llm_filtered",
        findings=findings,
        retrieved_chunk_ids=["chunk:vidA:0", "chunk:vidB:0"],
        retrieved_video_ids=["vidA", "vidB"],
    )
    return score_critique(critique, data, embedding_matcher(fake_embed, threshold=0.5), chunk_text)


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
        ]
    )
    assert cell["scores"]["evidence_precision"] == 1.0
    assert cell["scores"]["provenance"] == 1.0
    assert cell["scores"]["criteria_recall"] == 0.5


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
        ]
    )
    assert cell["scores"]["evidence_precision"] == 0.0
    assert cell["scores"]["provenance"] == 0.0
    assert cell["findings"][0]["grounded"] is False


def test_contested_needs_two_distinct_videos_that_both_resolve():
    """Flagging a conflict is not the same as having found one.

    A finding marked contested whose two citations are the same video is one
    source, and must not count — otherwise the metric rewards the flag rather
    than the behaviour.
    """
    same_video = scored(
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
        ]
    )
    assert same_video["scores"]["contested_rate"] == 0.0

    two_videos = scored(
        [
            Finding(
                id="f1",
                criterion="put numbers on every claim",
                detail="",
                contested=True,
                citations=(
                    Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),
                    Citation(video_id="vidC", start_seconds=300.0, quote="a skills list"),
                ),
            )
        ]
    )
    assert two_videos["scores"]["contested_rate"] == 1.0


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
    """
    quote = TRANSCRIPT[(quote_video, quote_seconds)][0][1][:40]
    return [
        Finding(
            id=f"f{index:02d}",
            criterion=c.criterion,
            detail="",
            citations=(Citation(video_id=quote_video, start_seconds=quote_seconds, quote=quote),),
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
        ),
        Finding(
            id="f2",
            criterion="numbers",
            detail="",
            citations=(Citation(video_id="vidB", start_seconds=200.0, quote="put numbers"),),
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
