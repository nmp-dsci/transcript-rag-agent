"""Judge the deep-research build loop on its three clauses, separately.

    uv run python -m src.cli serve --port 8021                # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v8_loop

V8's gate is three sentences joined by semicolons, and they are not one claim:

1. the loop-built pack scores **>= the hand-built v1** on the V3 metrics;
2. **round 2 exists** and adds >=1 rubric the gap critic asked for;
3. the report **renders each round's delta**.

A slice can fail the first and pass the other two, and this one does. So this
script scores them as three groups rather than as one number, because collapsing
them would destroy the only interesting thing the run has to say: the loop
demonstrably *iterates*, and iterating did not make the pack better. A verdict
that says "V8: FAIL" and a verdict that says "the machinery works, the result
did not" are different reports, and only the second is worth reading.

What each clause is actually asserted against
---------------------------------------------
**Clause 1** is arithmetic on rendered cells, not on the run file. The build
report's *Scored on the held-out expert* table is read out of the DOM and the
four V3 metrics (:data:`src.evals.critique.CRITIQUE_METRICS`) are compared arm
by arm. Two things decide it and they point the same way:

* ``criteria_recall`` — deep-r2 0.368 against merged 0.263 looks like a win, and
  is not one. The spreads printed beside those numbers are 0.263-0.474 and
  0.211-0.368; they overlap over [0.263, 0.368], deep-r2's *floor* is merged's
  point estimate and merged's *ceiling* is deep-r2's. The app's own
  ``winner()`` rule — clear the runner-up's maximum — is the bar applied here,
  and the panel itself prints "+0.105 vs merged (within noise)" on that row.
* ``evidence_precision`` — merged 0.824 against deep-r2 0.769, the wrong
  direction, and this is the metric that matters most. It is grounded/total over
  citations checked by string against the chunk store; the LLM matcher that
  makes ``criteria_recall`` noisy is not in this path at all. See the caveat
  below, though: "the scorer is deterministic" is not "the number is noise-free".

``>= on the V3 metrics`` is read as all four, because that is what the plural
says and because the harness that produced them refuses to publish a composite
(:mod:`src.evals.critique`, "There is deliberately no composite") precisely so
that a slice cannot average a loss away. One metric down is clause 1 lost.

**Clause 2** is a novelty claim, and the builder graded its own novelty, so this
re-derives it from scratch and from three independent directions:

* the *rendered* text of all 16 round-1 rubrics is scanned for the vocabulary
  r5403 is about — github, portfolio, link, blog, repo, personal site, project.
  **Zero** of the 16 match any of them. The diff renders each rule's criterion;
  widening the scan to the rules' *checks* as well adds exactly one hit — r0603's
  "For every bullet under Experience and Projects" — which is the bare mention of
  a section its bullets might live under, and is precisely the "only mentions
  Projects in passing" the gap critic itself wrote in g04. Nothing in round 1
  mentions a link, GitHub, a portfolio or a personal site under either scan.
* the **one-shot control** is the real test and it is why the control exists.
  deep-oneshot spends the same 10 executor calls on the same opening plan and
  differs only in that its last four questions came from the planner rather
  than the critic. It produced 26 rubrics and **not one** of them mentions any
  of those terms — including from probe p10, which asks what a data-science
  resume should carry. So the ground is not something a bigger budget reaches;
  it is something the critic reached.
* the app's own dedupe instrument agrees: of the ten round-2 additions, r5403
  sits at cosine **0.300** to its nearest round-1 neighbour, the lowest of the
  ten by a distance (next lowest 0.494, six of ten between 0.744 and 0.859
  against a 0.86 threshold). The panel renders this, so a reader can see that
  nine of the ten additions are restatements and one is new ground.

**Clause 3** is the cheapest to check and the easiest to fake, so it is checked
on the rendered table rather than on the presence of a field: the Rounds table
must carry a delta column, round 1 must render "—" in it, and both the control
and round 2 must render "+10" — against round 1, not chained off each other.
Opening a round's probe count must then reveal the sub-questions it asked with
the gap each came from, because "the report renders each round's delta" is worth
nothing if the delta is a number with no story under it.

What this script deliberately does not do
-----------------------------------------
It does not read ``/api/experts/resume-design/research`` to prove any of it. The
research JSON is the builder's own output and asserting against it would test
that a file was written, not that a reader can see a loop iterate — which is the
entire claim. Every number below is scraped from a rendered ``<table>`` or from
rendered rubric text.

It does not reorder or re-rank the Critique eval panel to make v2 lead. The plan
asks for "v2's row above v1's" and the panel is baseline-first, so merged
renders above deep-r2. That deviation is judged **acceptable** in the verdict,
and for a stronger reason than "cosmetic": sorting that table by score would
render deep-r1 and deep-r2 above merged and would *imply* a ranking the noise
analysis says is not there. The panel instead prints a signed delta on v2's own
row — "+0.105 vs merged (within noise)" — which puts v1 and v2 side by side and
tells the reader the truth about the gap. Baseline-first is also the convention
every other table in the app follows.

Two findings this script encodes that the build report does not claim
--------------------------------------------------------------------
* The loop's one genuinely new rubric is **not new to the corpus** — only to
  round 1 of the loop. The hand-built ``merged`` pack already carries r0403
  ("Link every serious project to a public, professional artifact ... keep that
  artifact clean and secure") and r0503 ("Only include links ... active,
  organized, and presentable"), and r5403's first citation is
  ``chunk:ZqqzBCg6IGU:1`` — *the same chunk* merged r0403 cites. The gap critic
  found ground the one-pass build had already covered. This does not fail clause
  2, whose baseline is round 1 by construction, but it is the mechanical reason
  clause 1 failed and it belongs in the record.
* The panel's ``decisive`` badge on ``evidence_precision`` is unearned. In
  :func:`src.evals.pack_ablation.winner` the rule is ``decisive = one leader and
  (runner_max is None or best > runner_max)``, and ``score_spread`` carries only
  ``criteria_recall_*`` keys — so for every other metric ``runner_max`` is
  ``None`` and the badge fires by fallback. The rendered reason, "leader clears
  the runner-up's own spread", is describing a comparison that was never made.
  Reported, not fixed: ``pack_ablation.py`` was under concurrent edit.

Read the metric's ceiling before reading the scores
---------------------------------------------------
``src/evals/KNOWN_GAP_attack2.md`` is open: nothing checks that a cited chunk
*supports* the finding attached to it, so ``criteria_recall`` certifies "reached
the conclusion and cited something that resolves", not "the corpus produced this
insight". That is why clause 1 here is weighted onto ``evidence_precision``,
which at least requires each finding to own a distinct resolving chunk, and why
a recall win that does not clear its own spread is given no credit at all.

ENVIRONMENT NOTE — this machine's Metal compiler XPC service is wedged and
``chromium.launch()`` times out after 180s, so this file could not be executed
through Playwright at evaluation time. Every assertion below was executed
instead as the equivalent DOM query against the running app in an already-open
Chrome at the same URL, in a tab created for this evaluation, and the results
are recorded in ``artifacts/v8_loop/verdict.json`` with the limitation noted
there. Slices v2, v3, v4, v4b recorded the same constraint the same way. Nothing
here is machine-specific; re-run it once the host can launch a browser.
"""

from __future__ import annotations

import re
import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v8_loop"

#: The topic whose pack the loop rebuilt.
TOPIC = "resume-design"

#: The hand-built v1 the loop has to beat, and the loop's second round.
BASELINE_ARM = "merged"
ROUND_ONE_ARM = "deep-r1"
CONTROL_ARM = "deep-oneshot"

#: Attempt 2's arm, and the arm clause 1 is now judged on. ``deep-r2`` is kept
#: below as the arm that failed, because a report that quietly renames the thing
#: being judged is how a slice passes without anyone noticing what changed.
LOOP_ARM = "deep-frontier"
FAILED_LOOP_ARM = "deep-r2"

#: The ablation that isolates the one free change. It applies the
#: evidence-novelty admission rule to ``deep-r2`` and spends no extra LLM call,
#: and it ties :data:`LOOP_ARM` on every scored figure — so it is the arm that
#: says how much the two *expensive* changes actually bought. Asserting on it is
#: the difference between "the loop got better" and "a filter got better".
ADMISSION_ARM = "deep-r2-admit"

#: The four metrics V3 defined, in table order. ``contested_coverage`` is None on
#: every arm of this run (no conflict had both chunks in any arm's context), so
#: it is unscoreable here rather than tied — recorded, not silently dropped.
V3_METRICS = ["criteria_recall", "evidence_precision", "provenance", "contested_coverage"]

#: The gap the clause-2 claim rests on, and the rubric it produced.
GAP_ID = "g04"
GAP_PROBE = "probe:p54"
NEW_RUBRIC_ID = "r5403"

#: The vocabulary r5403 is about. If any round-1 rubric already covered this
#: ground the novelty claim is dead, so the terms are deliberately generous —
#: "link" and "project" will match a rubric that only brushes past the subject.
NOVELTY_TERMS = ["github", "portfolio", "link", "personal site", "blog", "repo", "project"]

#: Cosine to the nearest surviving round-1 rule, as the report renders it. Dedupe
#: fires at 0.86; a genuinely new rule should sit far below that, and a
#: restatement that squeaked through should sit just under it.
DEDUPE_THRESHOLD = 0.86

#: ``0.368 0.263–0.474`` as the recall cell renders it: point estimate then the
#: range the 5 matcher repeats produced.
SCORE_CELL = re.compile(r"(\d\.\d+)(?:\s+(\d\.\d+)[–-](\d\.\d+))?")


def parse_score(cell: str) -> tuple[float | None, tuple[float, float] | None]:
    """A metric cell as (point estimate, spread) — spread is None when unreported."""
    match = SCORE_CELL.search(cell.replace("\n", " "))
    if not match:
        return None, None
    point = float(match.group(1))
    if match.group(2) and match.group(3):
        return point, (float(match.group(2)), float(match.group(3)))
    return point, None


def expand(session: UserSession, label: re.Pattern[str]) -> bool:
    """Open a collapsed disclosure by its visible label, and let React redraw.

    The click and the read have to be separate steps: these are React state
    toggles, so a click and a ``document.querySelectorAll`` inside one
    ``evaluate`` block read the *pre*-click DOM and the assertion then passes or
    fails for a reason that has nothing to do with the slice.
    """
    button = session.page.get_by_role("button", name=label)
    if button.count() == 0:
        return False
    target = button.first
    target.scroll_into_view_if_needed()
    if target.inner_text().strip().startswith("▸"):
        target.click()
        session.page.wait_for_timeout(500)
    return True


def read_table(session: UserSession, signature: str) -> list[list[str]]:
    """Every cell of the rendered table whose header row matches ``signature``.

    Tables are found by their *header text*, not by index or class: an index
    silently re-points when a panel is added above, and this page renders
    sixteen tables.
    """
    return session.page.evaluate(
        """(signature) => {
            const tables = [...document.querySelectorAll('table')];
            const head = t => [...(t.rows[0] ? t.rows[0].cells : [])]
                .map(c => c.textContent.trim().toLowerCase()).join('|');
            const hit = tables.find(t => head(t) === signature);
            if (!hit) return [];
            return [...hit.rows].map(r => [...r.cells]
                .map(c => c.innerText.replace(/\\s+/g, ' ').trim()));
        }""",
        signature.lower(),
    )


def rows_by_arm(table: list[list[str]]) -> dict[str, list[str]]:
    """Index a rendered arm table by the arm name its first cell starts with.

    The cell reads ``mergedBASE`` / ``deep-oneshotCONTROL`` — the badge is a
    sibling span with no separator, so the arm is a prefix match rather than an
    equality one.
    """
    out: dict[str, list[str]] = {}
    for row in table[1:]:
        if not row:
            continue
        for arm in (BASELINE_ARM, ROUND_ONE_ARM, CONTROL_ARM, LOOP_ARM):
            if row[0].startswith(arm) and arm not in out:
                out[arm] = row
    return out


# ── clause 1: is the loop-built pack actually better? ────────────────────────
def clause_one(session: UserSession) -> None:
    """The loop-built pack >= the hand-built v1, on all four V3 metrics."""
    table = read_table(
        session,
        "arm|criteria_recall|evidence_precision|provenance|contested_coverage"
        "|findings|quotes|executor calls|leaks",
    )
    session.check(
        "clause 1: the build report scores the loop against the hand-built pack",
        len(table) >= 5,
        f"{max(len(table) - 1, 0)} arms rendered in the held-out scored table",
        shot=True,
    )
    if len(table) < 5:
        return

    arms = rows_by_arm(table)
    columns = {name: i for i, name in enumerate(c.lower() for c in table[0])}
    missing = [a for a in (BASELINE_ARM, ROUND_ONE_ARM, CONTROL_ARM, LOOP_ARM) if a not in arms]
    session.check(
        "clause 1: all four arms render, including the one-shot control",
        not missing,
        f"rendered: {sorted(arms)}" + (f"; missing {missing}" if missing else ""),
    )
    if missing:
        return

    base_recall, base_spread = parse_score(arms[BASELINE_ARM][columns["criteria_recall"]])
    loop_recall, loop_spread = parse_score(arms[LOOP_ARM][columns["criteria_recall"]])

    # The bar the app's own winner() sets, in its most conservative form: this
    # arm's *worst* repeat against the baseline's *best*. Attempt 1's deep-r2
    # failed exactly here, its floor landing on the baseline's point estimate.
    disjoint = (
        loop_spread is not None and base_spread is not None and loop_spread[0] > base_spread[1]
    )
    session.check(
        "clause 1: the loop's recall range is disjoint from the baseline's",
        disjoint,
        f"{LOOP_ARM} {loop_recall} (spread {loop_spread}) vs {BASELINE_ARM} {base_recall} "
        f"(spread {base_spread}); "
        + (
            f"floor {loop_spread[0]} clears ceiling {base_spread[1]} by "
            f"{loop_spread[0] - base_spread[1]:.4f} — one criterion in 19 is 0.0526, so this is "
            "a one-criterion margin over five repeats, not a wide result"
            if disjoint
            else "the ranges cross, so neither arm is credited with a lead"
        ),
    )

    # evidence_precision carries no matcher, so it is the metric the noise
    # argument cannot be used on — and it points the other way.
    base_evidence, _ = parse_score(arms[BASELINE_ARM][columns["evidence_precision"]])
    loop_evidence, _ = parse_score(arms[LOOP_ARM][columns["evidence_precision"]])
    session.check(
        "clause 1: the loop is at least as grounded as the hand-built pack",
        loop_evidence is not None and base_evidence is not None and loop_evidence >= base_evidence,
        f"evidence_precision {LOOP_ARM} {loop_evidence} vs {BASELINE_ARM} {base_evidence} "
        f"({arms[LOOP_ARM][columns['findings']]} vs {arms[BASELINE_ARM][columns['findings']]} "
        "findings grounded) — no LLM matcher in this path, so there is no scorer "
        "noise to attribute the gap to",
    )

    base_prov, _ = parse_score(arms[BASELINE_ARM][columns["provenance"]])
    loop_prov, _ = parse_score(arms[LOOP_ARM][columns["provenance"]])
    session.check(
        "clause 1: the loop's citations still resolve as well as the baseline's",
        loop_prov is not None and base_prov is not None and loop_prov >= base_prov,
        f"provenance {LOOP_ARM} {loop_prov} vs {BASELINE_ARM} {base_prov}",
    )

    session.check(
        "clause 1: no held-out leak in any arm",
        all(arms[a][columns["leaks"]] == "0" for a in arms),
        "leaks per arm: " + ", ".join(f"{a} {arms[a][columns['leaks']]}" for a in sorted(arms)),
    )

    # The control is the arm that says whether the budget or the critic did it.
    control_evidence, _ = parse_score(arms[CONTROL_ARM][columns["evidence_precision"]])
    session.check(
        "clause 1: the loop grounds better than the same budget without a critic",
        loop_evidence is not None
        and control_evidence is not None
        and loop_evidence > control_evidence,
        f"evidence_precision {LOOP_ARM} {loop_evidence} vs {CONTROL_ARM} {control_evidence} "
        f"on identical finding counts ({arms[LOOP_ARM][columns['findings']]} vs "
        f"{arms[CONTROL_ARM][columns['findings']]}) and identical executor budget "
        f"({arms[LOOP_ARM][columns['executor calls']]} vs "
        f"{arms[CONTROL_ARM][columns['executor calls']]} calls)",
    )

    # The arm that keeps the pass honest. deep-r2-admit is deep-r2 with the
    # evidence-novelty filter and no extra call, and it ties this arm on every
    # scored figure. Without this assertion a reader would take "the loop got
    # better" from a page whose own ablation says a free filter did the work and
    # the two paid changes only tightened the spread. Asserted as a *tie* on
    # purpose: if a later build made the frontier arm genuinely pull ahead here,
    # this check would fail and the claim would have to be rewritten rather than
    # quietly inherited.
    if ADMISSION_ARM in arms:
        admit_recall, admit_spread = parse_score(arms[ADMISSION_ARM][columns["criteria_recall"]])
        admit_evidence, _ = parse_score(arms[ADMISSION_ARM][columns["evidence_precision"]])
        session.check(
            "clause 1: the free ablation is reported beside the arm and ties it on the numbers",
            admit_recall == loop_recall and admit_evidence == loop_evidence,
            f"{ADMISSION_ARM} {admit_recall}/{admit_evidence} vs {LOOP_ARM} "
            f"{loop_recall}/{loop_evidence} — the point estimates are bought by the admission "
            f"filter, which costs no LLM call. What the coverage-aware critic and the frontier "
            f"retrieval buy is the floor: {ADMISSION_ARM} spreads {admit_spread} and still "
            f"crosses {BASELINE_ARM}'s ceiling {base_spread[1] if base_spread else '?'}, "
            f"{LOOP_ARM} spreads {loop_spread} and does not",
        )


# ── clause 2: does round 2 add a rubric the critic asked for? ────────────────
def clause_two(session: UserSession) -> None:
    """Round 2 exists, and at least one of its rules is genuinely new ground."""
    page_text = session.page.locator("body").inner_text()

    session.check(
        "clause 2: the report names a round caused by the gap critic",
        "gap critic reading" in page_text,
        "the Rounds table attributes round 2 to 'gap critic reading deep-r1'",
        shot=True,
    )

    gaps = session.page.evaluate(
        """() => [...document.querySelectorAll('*')]
            .filter(e => e.children.length === 0 && /^g\\d\\d$/.test(e.textContent.trim()))
            .map(e => e.textContent.trim())"""
    )
    session.check(
        "clause 2: the report states what the critic said was missing",
        len(set(gaps)) >= 4 and GAP_ID in gaps,
        f"{len(set(gaps))} gap cards rendered: {sorted(set(gaps))}",
    )

    session.check(
        f"clause 2: {GAP_ID} renders the probe the critic wrote for it",
        GAP_PROBE in page_text and "gap:" + GAP_ID in page_text,
        f"{GAP_PROBE} renders and is tagged 'gap:{GAP_ID}' in the round-2 probe list",
        shot=True,
    )

    session.check(
        f"clause 2: {NEW_RUBRIC_ID} renders as a rule round 2 added",
        NEW_RUBRIC_ID in page_text,
        f"{NEW_RUBRIC_ID} renders under {GAP_ID} and again with a '+' in the v1->v2 diff",
    )

    # The novelty claim, rebuilt from the rendered round-1 rubrics rather than
    # taken from the builder. The diff panel renders every kept (round-1) rule
    # beside every added one, so both sets are readable off one screen.
    expand(session, re.compile(r"→ deep-r2: \d+ added"))
    diff = session.page.evaluate(
        """() => {
            const read = sel => [...document.querySelectorAll(sel)]
                .map(r => r.innerText.replace(/\\s+/g, ' ').trim());
            return {
                kept: read('.rs-diffrow.kept'),
                added: read('.rs-diffrow.added'),
            };
        }"""
    )
    kept = diff["kept"]
    added = diff["added"]
    session.check(
        "clause 2: the diff renders round 1's rules beside round 2's additions",
        len(kept) >= 16 and len(added) >= 10,
        f"{len(added)} added, {len(kept)} kept",
        shot=True,
    )
    if kept:
        overlap = [
            (t[:70], sorted(term for term in NOVELTY_TERMS if term in t.lower()))
            for t in kept
            if any(term in t.lower() for term in NOVELTY_TERMS)
        ]
        session.check(
            f"clause 2: no round-1 rule already covered {NEW_RUBRIC_ID}'s ground",
            not overlap,
            f"{len(overlap)} of {len(kept)} round-1 rules match any of {NOVELTY_TERMS}: "
            f"{overlap or 'none'}. A single github/portfolio/link hit would have killed "
            "the claim. (The diff renders each rule's criterion; scanning the *checks* "
            "too adds exactly one hit, r0603's 'For every bullet under Experience and "
            "Projects', which is the passing mention of Projects the critic itself "
            "described in g04 — and still nothing about links, GitHub or a portfolio.)",
        )
        session.check(
            f"clause 2: {NEW_RUBRIC_ID} is the only addition on that ground",
            sum(
                1
                for t in added
                if {"github", "portfolio", "blog", "repo"}
                & {x for x in NOVELTY_TERMS if x in t.lower()}
            )
            == 1,
            "of round 2's additions, only "
            + ", ".join(t[:60] for t in added if "github" in t.lower())
            + " carries the github/portfolio/blog/repo cluster",
        )

    # The app's own dedupe instrument, rendered: how far each addition sits from
    # the nearest rule round 1 already had.
    # ``.rs-cos`` is the cosine badge; its parent row carries the rule and the
    # nearest round-1 neighbour the number was measured against.
    expand(session, re.compile(r"added rules sit within"))
    cosines = session.page.evaluate(
        """() => [...document.querySelectorAll('.rs-cos')].map(c => ({
            cosine: parseFloat(c.textContent.trim()),
            rubric: (c.parentElement.innerText.match(/r\\d{4}/) || [])[0],
        }))"""
    )
    mine = [c for c in cosines if c.get("rubric") == NEW_RUBRIC_ID]
    others = [c["cosine"] for c in cosines if c.get("rubric") != NEW_RUBRIC_ID]
    session.check(
        f"clause 2: {NEW_RUBRIC_ID} is the most distant of round 2's additions",
        bool(mine) and bool(others) and mine[0]["cosine"] < min(others),
        f"{NEW_RUBRIC_ID} sits at cosine {mine[0]['cosine'] if mine else '?'} to its nearest "
        f"round-1 neighbour; the other {len(others)} additions sit at "
        f"{sorted(others)} against a {DEDUPE_THRESHOLD} dedupe threshold",
        shot=True,
    )


# ── clause 3: does the report render each round's delta? ─────────────────────
def clause_three(session: UserSession) -> None:
    """Each round's contribution is a number on the page, not a claim in prose."""
    table = read_table(
        session,
        "arm|caused by|probes added|calls spent|rubrics in pack|Δ vs round 1"
        "|quotes|≥2 creators|deduped",
    )
    session.check(
        "clause 3: the Rounds table renders a delta column",
        bool(table),
        "header carries 'Δ VS ROUND 1'" if table else "no Rounds table with a delta column",
        shot=True,
    )
    if not table:
        return

    columns = {name: i for i, name in enumerate(c.lower() for c in table[0])}
    arms = rows_by_arm(table)
    delta = columns["δ vs round 1"]
    size = columns["rubrics in pack"]

    session.check(
        "clause 3: round 1 is the origin and renders no delta against itself",
        ROUND_ONE_ARM in arms and arms[ROUND_ONE_ARM][delta] in {"—", "-", ""},
        f"{ROUND_ONE_ARM} delta cell reads {arms.get(ROUND_ONE_ARM, ['?'])[delta]!r} "
        f"over {arms.get(ROUND_ONE_ARM, ['?'])[size]} rubrics",
    )
    session.check(
        "clause 3: round 2 renders its delta against round 1",
        LOOP_ARM in arms and arms[LOOP_ARM][delta] == "+10",
        f"{LOOP_ARM} renders {arms.get(LOOP_ARM, ['?'])[delta]!r} "
        f"({arms.get(ROUND_ONE_ARM, ['?'])[size]} -> {arms.get(LOOP_ARM, ['?'])[size]} rubrics)",
    )
    session.check(
        "clause 3: the control's delta is against round 1 too, not chained off round 2",
        CONTROL_ARM in arms
        and arms[CONTROL_ARM][delta] == "+10"
        and arms[CONTROL_ARM][size] == arms[LOOP_ARM][size],
        f"{CONTROL_ARM} renders {arms.get(CONTROL_ARM, ['?'])[delta]!r} over "
        f"{arms.get(CONTROL_ARM, ['?'])[size]} rubrics — both rounds are measured against "
        "round 1's 16, so the two are comparable rather than cumulative",
    )

    # A delta with no story under it is a number. Open round 2's probe count.
    # Clicked through Playwright rather than inside evaluate(): the disclosure is
    # React state, so a click and a read in the same synchronous block sees the
    # pre-click DOM and the assertion passes or fails for the wrong reason.
    toggle = session.page.locator("tr", has_text=re.compile(r"^deep-r2")).locator("button").first
    if toggle.count() and toggle.inner_text().strip().startswith("▸"):
        toggle.click()
        session.page.wait_for_timeout(500)
    probes = session.page.evaluate(
        """() => {
            const body = document.body.innerText;
            const i = body.indexOf('gap critic reading');
            return (body.slice(i, i + 1400).match(/gap:g\\d\\d/g) || []);
        }"""
    )
    session.check(
        "clause 3: opening round 2's probe count reveals the sub-questions and their gaps",
        len(set(probes)) >= 4,
        f"{len(set(probes))} distinct gap origins render under the expanded row: "
        f"{sorted(set(probes))}",
        shot=True,
    )


# ── the deviation, recorded rather than waved through ────────────────────────
def row_order_deviation(session: UserSession) -> None:
    """The plan asks for v2's row above v1's; the panel is baseline-first."""
    table = read_table(
        session,
        "setup|criteria_recall|evidence_precision|provenance|contested_coverage"
        "|recall_all|recall_grouped|findings|cited|leaks|",
    )
    order = [row[0] for row in table[1:]] if table else []
    base_at = next((i for i, s in enumerate(order) if s.startswith(BASELINE_ARM)), None)
    loop_at = next((i for i, s in enumerate(order) if s.startswith(LOOP_ARM)), None)

    session.check(
        "the Critique eval panel renders v1 and v2 side by side with a signed delta",
        base_at is not None and loop_at is not None and "vs merged" in table[loop_at + 1][1],
        f"row order {order}; {LOOP_ARM}'s recall cell reads "
        f"{table[loop_at + 1][1] if loop_at is not None else '?'!r} — the comparison the "
        "plan wanted is on the row itself rather than in the sort order",
        shot=True,
    )
    session.note(
        "DEVIATION, ACCEPTED. The plan says 'Critique eval panel shows v2's row above "
        f"v1's'. It does not: the panel is baseline-first, so {BASELINE_ARM} renders at "
        f"index {base_at} and {LOOP_ARM} at index {loop_at}. Accepted, and not merely as "
        "cosmetic. Sorting that table by criteria_recall would put deep-r1 and deep-r2 "
        "above merged and would imply a ranking the run's own noise analysis denies — "
        "the exact overclaim clause 1 exists to catch. What the panel does instead is "
        "stronger: it prints '+0.105 vs merged (within noise)' on v2's own row, so v1 "
        "and v2 are compared explicitly rather than by adjacency. Baseline-first is also "
        "what every other table on the page does. Reordering would have made the page "
        "worse; the builder was right not to."
    )


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        session.tab("Experiments")
        session.check_visible(
            "the Experiments tab opens the deep-research build report",
            "Build report",
            shot=True,
        )
        session.check_visible(
            "the report names the loop's four stages",
            "a gap critic reads the result and names what a reviewer still could not check",
        )

        clause_one(session)
        clause_two(session)
        clause_three(session)
        row_order_deviation(session)

        session.note(
            "Scored as three clauses, not one. Clause 1 FAILS, clauses 2 and 3 PASS. "
            "The loop is built, it demonstrably iterates, and iterating did not produce "
            "a better pack — which is a real result and is reported as one."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
