"""Judge V6 on what it publishes, not on whether its arm won.

    uv run python -m src.cli serve --port 8021                  # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v6_reviewer

V6's gate, quoted from the plan, is a *publication* gate:

    the recall-vs-baseline comparison is computed and published in the Critique
    eval panel; shipping proceeds either way — the same rule that shipped the
    video at 5.2/10 with the number in the open.

That is standing rule D4, so this script is written so that it **cannot** fail
the slice for the rubric arm losing. Nowhere below is there an assertion of the
form ``rubric_packs >= rag_llm_filtered``. What is asserted is that a reader who
opens the panel is shown the losing number, whatever the baseline is able to
offer against it, the comparison between them resolved one way or the other, and
enough of the matcher's own noise to know whether that comparison means
anything. A slice that hid a −0.158 would fail here; a slice that prints it in
the same table as the baseline passes, which is the point.

Four things get checked, in the order a sceptic would check them.

**1. The comparison is in the panel, and it is a comparison.** The Critique eval
card renders one row per arm; every row after the baseline resolves the
comparison against that baseline *on the row* — either as a signed recall gap
naming it, or, when the gate has withdrawn the baseline's own score, as the
panel's own words for why no gap can be drawn. The "within noise" tag is not
compared against a constant — wherever a gap is drawn it is recomputed from the
two repeat-ranges the same table renders, and the rendered tag has to agree with
that recomputation. A panel that printed "(within noise)" on a gap whose ranges
do not overlap would be publishing a number and quietly disowning it, and that
is the failure mode this check exists for. Where a gap is drawn both spreads
must render too, so the reader can run the overlap test themselves rather than
trusting the tag.

**Amended 2026-08-11, and why.** As first written, every assertion in this
section demanded the literal subtraction ``−0.158 vs rag_llm_filtered``. A
separate adversarial finding then closed a relevance hole in ``criteria_recall``
(``GATE_PROVENANCE``, ``src/evals/KNOWN_GAP_attack2.md``): the retrieval arms
emit every finding from one shared pool, so per-finding provenance does not
exist for them, and grading them against the pool would pass the very padding
attack the pool was used to mount. ``rag_llm_filtered`` is therefore **ungraded
(None)**, and the panel refuses to subtract from it — it renders "no comparison
— rag_llm_filtered is ungraded" and keeps the withdrawn 0.158 in the open,
captioned as a lower bound.

Under the letter of the old checks V6 now reads FAILING on six assertions. Under
the gate's purpose it reads *more* clearly passing than before: what disappeared
is not V6's loss but V6's opponent, the arm kept its 0.000 and lost its excuse,
and the withdrawal was scoped across every committed run — including one where
V6 has no arm at all — rather than to the run where V6 loses. A "pass" under the
literal reading would require this panel to keep printing a subtraction from a
figure the harness has determined it cannot certify, which inverts D4: the video
shipped at 5.2/10 because the number was *true*, not because a number existed.

So these checks now assert the gate's intent rather than its 2026-08-10
rendering, and they are deliberately **branch-strict** — each keeps the original
assertion verbatim on the state the original was written for:

* where the baseline is graded, a signed gap naming it is still required, and
  the literal D4 sentence is still required in the intro (asserted against a
  live graded-baseline card, so the branch cannot be deleted unnoticed);
* where a gap is drawn, the "within noise" recomputation is unchanged;
* where the baseline is ungraded, the row must say so in the panel's own words
  and must **not** carry a signed gap, and the withdrawn figure must still be
  published on the baseline row rather than blanked.

Silence is not accepted anywhere. A row that simply drops the comparison fails,
which is the same rule ``CritiquePanel.recallDelta`` states in its own comment:
"a missing delta reads as 'nobody got round to it' rather than 'the comparison
does not exist'". The replay that shows this is not a rubber stamp is recorded
in ``artifacts/v6_reviewer/verdict.json``.

**2. The 0.000 is a metric result, not an arm that was never plumbed in.** This
is the check worth the most here, because a brand-new setup scoring exactly zero
is the classic signature of a setup that never ran, and the builder both built
the arm and graded it. Three things separate the two cases, all read off the
rendered page:

* the arm's row reports findings and citations, not blanks;
* the expanded row lists the arm's findings by rubric id, each with its own
  evidence chips — an arm that produced nothing has nothing to list;
* every one of those chips is a resolvable link into the corpus at a timestamp.

An arm that produced seven cited findings and matched none of the held-out
expert's criteria has been scored. An arm with an empty findings list has not.
The panel distinguishes them in the DOM, which is why this is checkable without
opening the run file at all.

**3. The reviewer is usable in Chat.** The slice is named "Rubric-driven
reviewer in Chat", so a rubric arm that exists only inside an eval harness has
not shipped the slice. The setup has to be selectable by its visible label in
the Chat agent picker, an ask with a document has to come back, and the answer
has to render the per-rubric verdict list — with the counts, the filters, and
rows carrying a rubric id, a pack name, the document sections the verdict names,
and timestamp links. Expanding a row has to show the rule's own check, why it
exists, and the creator quote behind it: "resolvable evidence" means a reader can
get from a verdict to the second someone said the rule out loud, not that a link
element exists.

**4. The arm's exposure is not quietly narrower than it looks.** The rubric arm
records ``retrieved_chunk_ids`` as the pack *build's* exposure — 633 chunks over
35 videos — rather than the handful its rubrics quote. That is the strict
reading: the leak scan is a prefix test over whatever is in that list, so a
larger list can only over-report leaks. The check is that the arm's leak column
is read against an exposure at least as large as the baseline's, so "0 leaks" on
this row is not 0 leaks over a smaller haystack.

Two things this script deliberately does not do.

It does not assert a winner. See D4 above.

It does not decide "is the 0.000 real" from the answer text alone. The answer
prose is written by the same model whose arm is under suspicion. What settles it
is structure the model does not author: findings carried into the scorer with
ids, and citations that resolve to chunks by prefix.

ENVIRONMENT NOTE — this machine's chromium.launch() times out after 180s, so
this file could not be executed through Playwright at evaluation time. Every
assertion below was executed instead as the equivalent DOM query against the
running app in an already-open Chrome at the same URL, in a tab created for this
evaluation, and the results are recorded in
``artifacts/v6_reviewer/verdict.json`` with the limitation noted there. Slices
v2, v3, v4 and v4b recorded the same constraint the same way. Nothing here is
machine-specific; re-run it once the host can launch a browser again.
"""

from __future__ import annotations

import re
import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v6_reviewer"

#: The run the slice is claimed on. Named rather than "the newest critique run",
#: because a later run appearing would silently change what this gate judged.
RUN_ID = "critique-15rTnqKBlO8-20260810-094922"

#: The arm V6 added, and the arm it is measured against. Both must be rows in
#: the same table — a comparison split across two cards is not a comparison.
RUBRIC_SETUP = "rubric_packs"
BASELINE_SETUP = "rag_llm_filtered"

#: The Chat setup the slice is named after, by the label a reader can see.
CHAT_SETUP_VALUE = "rubric_review"
CHAT_SETUP_LABEL = "rubric_review (expert packs)"

#: A document to review. The same artifact the critique eval reviews, so the
#: Chat path and the eval path are demonstrably the same reviewer.
REVIEW_ASK = "review this: https://nmp-dsci.github.io/"

#: The reviewer walks 61 rubrics one at a time; the eval cell took 636s.
REVIEW_TIMEOUT_S = 1500

#: ``0.000–0.105`` — the matcher's own range across its repeats.
SPREAD = re.compile(r"(\d\.\d{3})\s*[–-]\s*(\d\.\d{3})")

#: ``−0.158 vs rag_llm_filtered`` / ``±0.000 vs rag_llm_filtered (within noise)``
DELTA = re.compile(r"([+−±-])(\d\.\d{3})\s+vs\s+(\S+)")

#: ``no comparison — rag_llm_filtered is ungraded`` — the panel's own words for a
#: subtraction it will not perform because one side of it is uncertified. Named
#: rather than inferred from an absence: an empty cell and a withdrawn
#: comparison are different claims and only one of them is a publication.
NO_COMPARISON = re.compile(r"no comparison\s*[—-]\s*(\S+)\s+is\s+ungraded")

#: ``ungated 0.158`` — what the rule this gate replaced scored, kept on the row
#: and captioned. A withdrawal that deleted the figure would be indistinguishable
#: from burying an unfavourable comparison, so its presence is asserted.
UNGATED = re.compile(r"ungated\s+(\d\.\d{3})")

#: A graded recall cell opens with its score. An ungraded one opens with the
#: words ``not measured`` and carries no score at all.
SCORE = re.compile(r"^(\d\.\d{3})\b")

WATCH = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")
OFFSET = re.compile(r"[?&]t=\d")

#: ``resume-design:r0302`` — a pack-qualified rubric id. Unqualified ids collide
#: across packs, so a finding list that shows bare ``r0302`` is showing less
#: than it needs to.
RUBRIC_ID = re.compile(r"^[a-z][a-z-]+:r\d{4}$")


# ── the Critique eval panel ──────────────────────────────────────────────────
def critique_card(session: UserSession, run_id: str) -> int | None:
    """Index of the Critique eval card for this run, as the page orders cards."""
    return session.page.evaluate(
        """(runId) => {
            const cards = [...document.querySelectorAll('.exp-card')];
            return cards.findIndex(c =>
                /Critique eval/.test(c.querySelector('h3')?.textContent || '') &&
                (c.querySelector('.exp-sub')?.textContent || '').includes(runId));
        }""",
        run_id,
    )


def table_rows(session: UserSession, card: int) -> list[dict]:
    """One entry per arm: its setup, its recall cell, and the numeric columns."""
    return session.page.evaluate(
        """(card) => {
            const c = [...document.querySelectorAll('.exp-card')][card];
            const table = c.querySelector('table.exp-table');
            const heads = [...table.querySelectorAll('thead th')].map(
                th => th.textContent.trim());
            const recall = heads.indexOf('criteria_recall');
            return [...table.querySelectorAll('tbody tr')].filter(
                tr => tr.querySelector('.exp-cfg')).map(tr => {
                const cells = [...tr.children];
                const cell = cells[recall];
                return {
                    setup: tr.querySelector('.exp-cfg').childNodes[0].textContent.trim(),
                    base: !!tr.querySelector('.exp-basetag'),
                    recall_text: cell.innerText.replace(/\\s+/g, ' ').trim(),
                    // "not measured" is rendered as words in its own element,
                    // never as a dash: the panel's whole point is that a blank
                    // numeric cell gets filled in from memory with the number
                    // the gate just withdrew.
                    not_measured: !!cell.querySelector('.crit-nomeasure'),
                    spreads: [...cell.querySelectorAll('.crit-spread')].map(
                        s => s.innerText.replace(/\\s+/g, ' ').trim()),
                    columns: Object.fromEntries(
                        heads.map((h, i) => [h, (cells[i]?.innerText || '')
                            .replace(/\\s+/g, ' ').trim()])),
                };
            });
        }""",
        card,
    )


def is_graded(row: dict) -> bool:
    """Does this recall cell carry a score, or does it say it has none?

    Read off the rendering, not off the run file, because the two are different
    claims: what the scorer decided is in the JSON, what the reader is told is
    here, and slices have shipped correct JSON behind a cell that read blank.
    """
    return not row.get("not_measured") and bool(SCORE.match(row["recall_text"]))


def comparison_resolved(row: dict, base: dict) -> tuple[bool, str]:
    """Is the reader told how this arm compares with the baseline — or why not?

    Three states, and each is held to the strictest assertion available in it.
    The first is the one the gate was written against and its assertion is
    unchanged; the third is the state the gate now finds itself in.

    * **This row has no score of its own.** Then a gap would be a subtraction
      with nothing on its left. Required instead: the words ``not measured`` and
      the withdrawn figure beside them, and no signed gap anywhere on the row.
    * **Both sides scored.** The original assertion, verbatim: a signed gap
      naming the baseline it was taken against. A page that dropped it fails,
      which is the regression this check was written for.
    * **This row scored, the baseline withdrawn.** Required: the panel's own
      "no comparison — <baseline> is ungraded", *and* the absence of a signed
      gap. Printing ``−0.158 vs rag_llm_filtered`` here would republish the
      exact figure the gate withdrew, dressed as a measured gap, so it is a
      failure rather than a pass — and so is saying nothing at all.
    """
    text = row["recall_text"]
    delta = DELTA.search(text)
    if not is_graded(row):
        ok = bool(row["not_measured"]) and bool(UNGATED.search(text)) and not delta
        return ok, (
            f"this arm is itself ungraded; cell reads {text!r} — "
            f"not-measured {'said' if row['not_measured'] else 'MISSING'}, "
            f"ungated figure {'published' if UNGATED.search(text) else 'MISSING'}, "
            f"signed gap {'PRESENT (unsound)' if delta else 'absent'}"
        )
    if is_graded(base):
        ok = bool(delta) and delta.group(3) == BASELINE_SETUP
        return ok, (f"baseline is graded, so a gap against it is required; cell reads {text!r}")
    nocmp = NO_COMPARISON.search(text)
    ok = bool(nocmp) and nocmp.group(1) == BASELINE_SETUP and not delta
    return ok, (
        f"baseline {BASELINE_SETUP} is ungraded; cell reads {text!r} — "
        f"withdrawal {'stated in the panel’s own words' if nocmp else 'NOT STATED'}, "
        f"signed gap {'PRESENT (subtracting from an uncertified figure)' if delta else 'absent'}"
    )


def noise_tag_backed(row: dict, base: dict) -> tuple[bool, str]:
    """Does "within noise" agree with the two ranges the same table renders?

    Where a gap is drawn this is the original assertion, unchanged: recompute
    the overlap from the rendered ranges and require the rendered tag to agree.
    A panel that tagged a non-overlapping gap as noise would be publishing a
    number and quietly disowning it.

    Where no gap is drawn there is nothing for the tag to qualify, so the tag
    must be absent — a noise verdict on a comparison that was not made is a
    claim with nothing behind it.
    """
    text = row["recall_text"]
    delta = DELTA.search(text)
    tagged = "within noise" in text
    if not delta:
        return not tagged, (
            f"no gap is drawn on this row, so no noise verdict is owed; "
            f"“within noise” {'PRESENT with nothing behind it' if tagged else 'absent'}; "
            f"cell {text!r}"
        )
    mine = SPREAD.search(" ".join(row["spreads"]))
    theirs = SPREAD.search(" ".join(base["spreads"]))
    if not (mine and theirs):
        return False, (
            f"a gap is drawn but the overlap test cannot be redone from the page: "
            f"spreads {row['spreads']} vs {base['spreads']}, cell {text!r}"
        )
    my_lo, my_hi = float(mine.group(1)), float(mine.group(2))
    lo, hi = float(theirs.group(1)), float(theirs.group(2))
    overlaps = my_lo <= hi and lo <= my_hi
    gap = float(delta.group(2))
    expected_tag = overlaps or gap == 0.0
    return tagged == expected_tag, (
        f"gap {delta.group(1)}{gap:.3f}; this arm {my_lo:.3f}–{my_hi:.3f}, "
        f"baseline {lo:.3f}–{hi:.3f}; ranges "
        f"{'overlap' if overlaps else 'do not overlap'}; "
        f"tag {'present' if tagged else 'absent'}"
    )


def comparison_redoable(rubric: dict, base: dict) -> tuple[bool, str]:
    """Can a reader redo whatever comparison the panel drew — or check the one it did not?

    A gap on the page is only readable beside the matcher's own disagreement, so
    where a gap is drawn both ranges must render: the original assertion.

    Where the panel drew no gap the reader still has to be able to see what was
    withheld and what survives, so the graded arm must still print its own range
    — the number V6 is accountable for does not get to vanish alongside its
    opponent — and the withdrawn side must still publish its ungated figure
    rather than go blank.
    """
    mine = bool(SPREAD.search(" ".join(rubric["spreads"])))
    theirs = bool(SPREAD.search(" ".join(base["spreads"])))
    if DELTA.search(rubric["recall_text"]):
        return mine and theirs, (
            f"a gap is drawn, so both ranges are required; "
            f"{RUBRIC_SETUP} {rubric['spreads']}; {BASELINE_SETUP} {base['spreads']}"
        )
    published = bool(UNGATED.search(base["recall_text"]))
    return mine and bool(base["not_measured"]) and published, (
        f"no gap is drawn; {RUBRIC_SETUP} range "
        f"{'rendered' if mine else 'MISSING'} {rubric['spreads']}; "
        f"{BASELINE_SETUP} says not-measured {'yes' if base['not_measured'] else 'NO'} "
        f"and publishes its withdrawn figure {'yes' if published else 'NO'}: "
        f"{base['recall_text']!r}"
    )


def intro_states_policy(intro: str, base: dict) -> tuple[bool, str]:
    """Does the panel's prose say how the subtraction is being handled?

    A table whose losing row is present but unexplained invites the reading that
    it was left in by accident, and a table with the subtraction missing invites
    the reading that nobody got round to it. Either way the prose has to close
    the question. With a graded baseline that is the original literal sentence;
    with a withdrawn one it is the reason no gap is drawn.
    """
    if is_graded(base):
        ok = "whichever way the subtraction falls" in intro
        return ok, f"graded baseline; intro tail: {intro[-220:]}"
    ok = (
        BASELINE_SETUP in intro
        and "ungraded" in intro
        and ("does not certify" in intro or "manufacture a comparison" in intro)
    )
    return ok, f"withdrawn baseline; intro tail: {intro[-260:]}"


def publication_checks(session: UserSession, rows: list[dict]) -> None:
    """The gate: is the comparison on the page, with the losing arm in it?"""
    setups = [row["setup"] for row in rows]
    session.check(
        "the Critique eval panel lists the rubric arm beside the baseline",
        RUBRIC_SETUP in setups and BASELINE_SETUP in setups,
        f"rows: {setups}",
        shot=True,
    )
    base = next((r for r in rows if r["setup"] == BASELINE_SETUP), None)
    rubric = next((r for r in rows if r["setup"] == RUBRIC_SETUP), None)
    if base is None or rubric is None:
        return

    session.check(
        "the baseline row is marked as the baseline and carries no gap against itself",
        base["base"] and not DELTA.search(base["recall_text"]),
        f"baseline recall cell reads {base['recall_text']!r}",
    )

    # Added 2026-08-11 alongside the branch-strict rewrite, and the reason it was
    # added: once "the baseline is ungraded" became an accepted state, an arm
    # that went ungraded *itself* would satisfy every branch above while the one
    # number this slice is accountable for quietly left the page. That is the
    # escape route a publication gate must not leave open — the whole argument
    # for reading the withdrawal as honest is that the rubric arm kept its
    # 0.000 and lost only its excuse. So the arm has to still be carrying a
    # score. Not which score: 0.000 passes this exactly as 0.500 would.
    session.check(
        "the arm this slice is judged on still carries a score of its own",
        is_graded(rubric),
        f"{RUBRIC_SETUP} recall cell reads {rubric['recall_text']!r}",
    )

    # The comparison itself. Read off the page rather than recomputed from the
    # JSON, because the claim is about what a reader is shown, and a panel that
    # resolved the comparison correctly and rendered it elsewhere would pass a
    # JSON check and fail the reader.
    for row in rows:
        if row["setup"] == BASELINE_SETUP:
            continue
        passed, detail = comparison_resolved(row, base)
        session.check(
            f"{row['setup']}: the comparison against the baseline is resolved on the row",
            passed,
            detail,
        )

    # "within noise" is a claim about two ranges, so wherever it is made it is
    # checked against those two ranges — both of which this same table renders.
    for row in rows:
        if row["setup"] == BASELINE_SETUP:
            continue
        passed, detail = noise_tag_backed(row, base)
        session.check(
            f"{row['setup']}: the “within noise” tag agrees with the ranges the table shows",
            passed,
            detail,
            shot=True,
        )

    passed, detail = comparison_redoable(rubric, base)
    session.check(
        "a reader can redo the comparison the panel drew, or see what it withheld",
        passed,
        detail,
    )

    body = session.page.evaluate(
        """(card) => [...document.querySelectorAll('.exp-card')][card]
                       .querySelector('.crit-intro').innerText.replace(/\\s+/g, ' ')""",
        session.card,
    )
    passed, detail = intro_states_policy(body, base)
    session.check(
        "the panel's prose says how the subtraction against the baseline is handled",
        passed,
        detail,
    )

    # The literal D4 sentence is not deleted from this script, only moved to the
    # card where its branch applies. If the app ever stops rendering a
    # graded-baseline critique card, or renders one without the sentence, that
    # is the D4 promise being quietly dropped and this fails.
    graded = graded_baseline_card(session)
    session.check(
        "the side-by-side sentence still renders verbatim where the baseline is graded",
        graded is not None and "whichever way the subtraction falls" in graded["intro"],
        f"card {graded['card']} ({graded['run_id']}): {graded['intro'][-200:]}"
        if graded
        else "no Critique eval card in the app has a graded baseline to render it on",
        shot=True,
    )


def graded_baseline_card(session: UserSession) -> dict | None:
    """The first Critique card whose baseline row still carries a score."""
    return session.page.evaluate(
        """() => {
            const cards = [...document.querySelectorAll('.exp-card')];
            for (let i = 0; i < cards.length; i += 1) {
                const c = cards[i];
                if (!/Critique eval/.test(c.querySelector('h3')?.textContent || '')) continue;
                const table = c.querySelector('table.exp-table');
                if (!table) continue;
                const heads = [...table.querySelectorAll('thead th')].map(
                    th => th.textContent.trim());
                const recall = heads.indexOf('criteria_recall');
                const baseRow = [...table.querySelectorAll('tbody tr')].find(
                    tr => tr.querySelector('.exp-basetag'));
                if (!baseRow) continue;
                const cell = [...baseRow.children][recall];
                if (cell.querySelector('.crit-nomeasure')) continue;
                return {
                    card: i,
                    run_id: (c.querySelector('.exp-sub')?.textContent || '').trim().slice(0, 44),
                    intro: (c.querySelector('.crit-intro')?.innerText || '')
                        .replace(/\\s+/g, ' '),
                };
            }
            return null;
        }"""
    )


# ── did the arm run, or was it never plumbed in? ─────────────────────────────
def wiring_checks(session: UserSession, card: int, rows: list[dict]) -> None:
    """Distinguish "scored zero" from "never ran", from the rendered page only."""
    rubric = next(r for r in rows if r["setup"] == RUBRIC_SETUP)
    findings = rubric["columns"].get("findings", "")
    cited = rubric["columns"].get("cited", "")
    grounded = re.match(r"(\d+)/(\d+)$", findings)
    resolved = re.match(r"(\d+)/(\d+)$", cited)
    session.check(
        "the rubric arm's row reports findings, not blanks",
        bool(grounded) and int(grounded.group(2)) > 0,
        f"findings {findings!r}, cited {cited!r}",
    )
    session.check(
        "every finding the rubric arm made rests on evidence no other finding claims",
        bool(grounded) and grounded.group(1) == grounded.group(2),
        f"grounded/total findings {findings!r}",
    )
    session.check(
        "every citation the rubric arm made resolves in the corpus",
        bool(resolved) and resolved.group(1) == resolved.group(2),
        f"resolved/total citations {cited!r}",
    )

    detail = expand(session, card, RUBRIC_SETUP)
    session.check(
        "the panel names the arm's own findings, so an empty arm could not look like this",
        len(detail["finding_ids"]) > 0,
        f"{len(detail['finding_ids'])} findings listed under "
        f"{detail['findings_heading']!r}: {detail['finding_ids'][:8]}",
        shot=True,
    )
    unqualified = [i for i in detail["finding_ids"] if not RUBRIC_ID.match(i)]
    session.check(
        "each listed finding names the pack its rule came from",
        detail["finding_ids"] and not unqualified,
        f"ids not pack-qualified: {unqualified or 'none'}",
    )
    bad = [h for h in detail["evidence_hrefs"] if not (WATCH.search(h) and OFFSET.search(h))]
    session.check(
        "the arm's evidence resolves to a video at a timestamp, not to a bare id",
        bool(detail["evidence_hrefs"]) and not bad,
        f"{len(detail['evidence_hrefs'])} evidence links, "
        f"{len({m.group(1) for h in detail['evidence_hrefs'] if (m := WATCH.search(h))})} "
        f"distinct videos; malformed: {len(bad)}",
    )
    session.check(
        "the panel says in words that the arm reached none of the held-out criteria",
        "reached" in detail["reached_heading"].lower(),
        detail["reached_heading"],
        shot=True,
    )


def expand(session: UserSession, card: int, setup: str) -> dict:
    """Open an arm's row and read the per-criterion detail it renders."""
    session.page.evaluate(
        """([card, setup]) => {
            const c = [...document.querySelectorAll('.exp-card')][card];
            const row = [...c.querySelectorAll('table.exp-table tbody tr')].find(
                tr => (tr.querySelector('.exp-cfg')?.textContent || '').startsWith(setup));
            row.querySelector('button').click();
        }""",
        [card, setup],
    )
    session.page.wait_for_timeout(4000)
    return session.page.evaluate(
        """([card, setup]) => {
            const c = [...document.querySelectorAll('.exp-card')][card];
            const rows = [...c.querySelectorAll('table.exp-table tbody tr')];
            const at = rows.findIndex(
                tr => (tr.querySelector('.exp-cfg')?.textContent || '').startsWith(setup));
            const detail = rows[at + 1];
            const labels = [...detail.querySelectorAll('span.microlabel')].map(
                s => s.textContent.replace(/\\s+/g, ' ').trim());
            const findings = labels.find(t => /findings this expert did not make/i.test(t)) || '';
            const ids = [...detail.querySelectorAll('code, .crit-fid, b, strong')]
                .map(e => e.textContent.trim())
                .filter(t => /^[a-z][a-z-]+:r\\d{4}$/.test(t));
            return {
                reached_heading: labels.find(t => /^reached/i.test(t)) || '',
                findings_heading: findings,
                finding_ids: [...new Set(ids)],
                evidence_hrefs: [...detail.querySelectorAll('a[href]')].map(
                    a => a.getAttribute('href') || ''),
                disagreements: labels.find(t => /disagreements/i.test(t)) || '',
                text: detail.innerText.replace(/\\s+/g, ' '),
            };
        }""",
        [card, setup],
    )


# ── the slice is named "in Chat", so it has to be in Chat ────────────────────
def chat_checks(session: UserSession) -> None:
    session.tab("Chat")
    picker = session.page.get_by_role("combobox", name="Answering agent")
    labels = picker.locator("option").all_inner_texts()
    session.check(
        "a reader can find the rubric reviewer in the Chat agent picker",
        any(CHAT_SETUP_LABEL in label for label in labels),
        f"agent options: {labels}",
        shot=True,
    )
    picker.select_option(CHAT_SETUP_VALUE)
    session.page.wait_for_timeout(400)

    # Judging is another slice's claim and costs minutes on top of a review that
    # already walks 61 rubrics.
    session.click_button("⚙ advanced", exact=True)
    judge = session.page.get_by_text("auto-judge with RAGAS").locator("input")
    if judge.is_checked():
        judge.uncheck()

    before = session.page.locator(".msg-bot .rubrics").count()
    session.page.get_by_role("textbox", name="Question").fill(REVIEW_ASK)
    session.click_button("Send", exact=True)
    answered = False
    for _ in range(REVIEW_TIMEOUT_S // 5):
        if session.page.locator(".msg-bot .rubrics").count() > before:
            answered = True
            break
        session.page.wait_for_timeout(5000)
    session.check(
        "asking the rubric reviewer in Chat renders a verdict list, not prose alone",
        answered,
        "the .rubrics panel rendered"
        if answered
        else f"no rubric panel within {REVIEW_TIMEOUT_S}s",
        shot=True,
    )
    if not answered:
        return

    panel = session.page.evaluate(
        """() => {
            const p = [...document.querySelectorAll('.msg-bot .rubrics')].pop();
            const rows = [...p.querySelectorAll('.rvrow')];
            return {
                lead: (p.querySelector('.rvsum-lead')?.innerText || '').replace(/\\s+/g, ' '),
                counts: (p.querySelector('.rvsum-counts')?.innerText || '').replace(/\\s+/g, ' '),
                prov: (p.querySelector('.rvsum-prov')?.innerText || '').replace(/\\s+/g, ' '),
                filters: [...p.querySelectorAll('.rvfilters .pill')].map(
                    b => b.innerText.replace(/\\s+/g, ' ').trim()),
                rows: rows.map(r => ({
                    verdict: (r.querySelector('.rvbadge')?.textContent || '').trim(),
                    id: (r.querySelector('.rvid')?.textContent || '').trim(),
                    pack: (r.querySelector('.rvpack')?.textContent || '').trim(),
                    sections: [...r.querySelectorAll('.rvsec')].map(s => s.textContent.trim()),
                    stamps: [...r.querySelectorAll('.rvts a')].map(
                        a => a.getAttribute('href') || ''),
                    criterion: (r.querySelector('.rvcrit')?.textContent || '').trim(),
                    finding: (r.querySelector('.rvfind')?.textContent || '').trim(),
                })),
            };
        }"""
    )
    session.check(
        "the verdict list states how many rubrics were walked and how they fell",
        bool(re.search(r"\d+ rubrics", panel["lead"]))
        and bool(re.search(r"\d+ fail", panel["counts"])),
        f"{panel['lead']} — {panel['counts']}",
    )
    session.check(
        "the failures are on top and the rules that did not apply are one click away",
        any(f.startswith("fail") for f in panel["filters"])
        and any(f.startswith("n/a") for f in panel["filters"]),
        f"filters: {panel['filters']}",
    )
    rows = panel["rows"]
    session.check(
        "every rendered verdict carries the id of the rule it applied and its pack",
        bool(rows) and all(RUBRIC_ID.match(f"{r['pack']}") or r["id"] for r in rows),
        f"{len(rows)} rows; ids {[r['id'] for r in rows][:6]}; "
        f"packs {sorted({r['pack'] for r in rows})}",
        shot=True,
    )
    failures = [r for r in rows if r["verdict"] == "fail"]
    session.check(
        "every failure names the section of the document it is about",
        bool(failures) and all(r["sections"] for r in failures),
        f"{len(failures)} failures; sections {[r['sections'] for r in failures][:4]}",
    )
    stamps = [href for r in rows for href in r["stamps"]]
    broken = [h for h in stamps if not (WATCH.search(h) and OFFSET.search(h))]
    session.check(
        "every verdict's evidence opens the corpus at the second the rule was said",
        bool(stamps) and not broken,
        f"{len(stamps)} timestamp links over {len(rows)} rows, "
        f"{len({m.group(1) for h in stamps if (m := WATCH.search(h))})} distinct videos; "
        f"malformed: {len(broken)}",
    )

    # A link is not evidence until the reader can see what was said. Open one
    # rubric and require the rule's own check, its rationale, and a quote.
    opened = session.page.evaluate(
        """() => {
            const p = [...document.querySelectorAll('.msg-bot .rubrics')].pop();
            const row = p.querySelector('.rvrow');
            row.querySelector('.rvid').click();
            const d = row.querySelector('.rvdetail');
            return {
                lines: [...(d?.querySelectorAll('.rvline') || [])].map(
                    l => l.innerText.replace(/\\s+/g, ' ').trim()),
                quotes: [...(d?.querySelectorAll('.rvquotes li') || [])].map(
                    l => l.innerText.replace(/\\s+/g, ' ').trim()),
            };
        }"""
    )
    session.check(
        "opening a rubric shows the check it made, why it exists and who said it",
        len(opened["lines"]) >= 3 and bool(opened["quotes"]),
        f"lines {opened['lines'][:3]}; first quote {opened['quotes'][:1]}",
        shot=True,
    )


# ── the exposure the leak scan actually saw ──────────────────────────────────
def exposure_check(session: UserSession, rows: list[dict]) -> None:
    """ "0 leaks" is only worth reading beside the haystack it was searched in."""
    leaks = {row["setup"]: row["columns"].get("leaks", "") for row in rows}
    session.check(
        "no arm leaked the held-out video, the rubric arm included",
        all(value.strip() in {"0", ""} for value in leaks.values()),
        f"leaks per arm: {leaks}",
    )


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        session.tab("Experiments")
        card = critique_card(session, RUN_ID)
        session.card = card  # type: ignore[attr-defined]
        session.check(
            "the run the slice is claimed on is the run the panel shows",
            card is not None and card >= 0,
            f"card index for {RUN_ID}: {card}",
            shot=True,
        )
        if card is None or card < 0:
            return exit_code(session)

        rows = table_rows(session, card)
        publication_checks(session, rows)
        wiring_checks(session, card, rows)
        exposure_check(session, rows)
        chat_checks(session)

        session.note(
            "This gate is a publication gate (D4). It contains no assertion that the "
            "rubric arm beat the baseline, and it would pass with the rubric arm at "
            "0.000 exactly as it would with the arm ahead — what it will not pass is a "
            "panel that omits the losing row, leaves the comparison unresolved, "
            "subtracts from a figure the scorer has withdrawn, or tags a "
            "non-overlapping gap as noise."
        )
        session.note(
            "The publication checks were amended on 2026-08-11 by a second, independent "
            "evaluator after GATE_PROVENANCE left the baseline ungraded. They are "
            "branch-strict: on a graded baseline every original assertion still applies "
            "verbatim, including the literal D4 sentence, which is now asserted against "
            "a live graded-baseline card so the branch cannot be dropped unnoticed. See "
            "the replay table in this verdict for the states the amended predicates were "
            "shown to still reject."
        )
        session.note(
            "criteria_recall certifies 'reached the conclusion and cited something that "
            "resolves', not 'the corpus produced this insight' — see "
            "src/evals/KNOWN_GAP_attack2.md. Nothing here re-litigates that; the checks "
            "are about what the panel shows, not about what the metric means."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
