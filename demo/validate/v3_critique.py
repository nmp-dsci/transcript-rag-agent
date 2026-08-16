"""Replay the V3 held-out critique eval in the browser and judge what it shows.

    uv run python -m src.cli serve --port 8021                 # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v3_critique

V3 claims something no other panel in this app claims: that the distilled corpus
reaches a named expert's conclusions *without having seen that expert*. The
number that carries the claim is ``criteria_recall``, and a recall number on its
own is unfalsifiable — it is only meaningful if a reader can open the row and see
which of the applicable criteria were reached, which were not, and what the
expert actually said at the second they said it.

So this script asserts the chain a sceptic would walk, on rendered page text:

* the **Critique eval card** exists on Experiments and names the held-out video;
* the **``held-out absent · 0 leaks`` chip** is on the card — the exclusion claim
  has to be visible, not buried in a JSON file;
* the **baseline row** carries the ``base`` chip and three scored metrics as
  numbers. The fourth, ``contested_coverage``, is nullable by construction since
  V7 — it is a fraction over the disagreements *this run had both sides of in
  context*, and 0/0 is an honest answer — so what is asserted there is not that
  it is a number but that it never prints a bare em dash: "averaged a conflict
  away" (0.000) and "was never shown one" (—) are different failures and the
  reader must be able to tell them apart without opening the run file;
* **Expand** opens a reached/missed split in which every criterion carries a
  clickable timestamp linking into the held-out video, and the resume-only ones
  carry an ``n/a to a portfolio`` chip — because that chip is what the recall
  denominator rests on, and a denominator the reader cannot see is a denominator
  the reader has to take on trust;
* nothing in that detail is **clipped**. The panel sits inside ``.exp-table``,
  which ships ``white-space: nowrap``, and the builder escaped it with a
  ``colSpan`` row rather than the block-in-last-cell precedent used elsewhere on
  this tab. That deviation is exactly the kind of call that is right in review
  and wrong in the browser, so the criteria sentences, the verbatim quotes and
  the matcher's reasons are each measured against the box they were drawn in.

Nothing here reads ``/api/experiments`` to prove a UI claim. The API returning a
correct cell and the reader being able to audit it are different claims, and this
tab has shipped the first without the second before.

Why the run's own arithmetic is no longer hard-coded here
---------------------------------------------------------
This file used to assert ``18/24 criteria apply``, ``criteria_recall 0.278``,
five reached, nineteen missed and six n/a chips. Every one of those had gone
stale by the time it was next run — the committed run says 19/24, 0.158, three
reached, twenty-one missed and five chips — and none of the changes were bugs.
Pinning a validator to one run's numbers makes it report *rescoring* as failure,
which is the fastest way to get a validator ignored. So the split, the
denominator, the reached count and the chip count are now **read off the page and
checked against each other**: the card's "N/24 apply" must equal the detail's
"of N", the missed column must account for the other 24-N, every excluded
criterion must carry its own chip, and the reached heading must be the number the
recall column prints. Those are the properties that were ever worth asserting;
the literals were only ever a proxy for them.

ENVIRONMENT NOTE — this machine's Metal compiler XPC service is wedged and
``chromium.launch()`` times out at 0% CPU, so the Playwright half of this file
could not be executed at evaluation time. Every assertion below was executed
instead as the equivalent DOM query through an already-open Chrome at the same
URL, and the results are recorded in ``artifacts/v3_critique/verdict.json`` with
that limitation noted there — the same way v2 and v4 recorded it.
"""

from __future__ import annotations

import re
import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v3_critique"

#: The held-out expert, by the title a reader sees on the card.
HELD_OUT_TITLE = "You asked me to roast your AI resumes"
HELD_OUT_VIDEO = "15rTnqKBlO8"

#: The baseline setup, and the chip that marks it as the row later slices must beat.
BASELINE_SETUP = "rag_llm_filtered"

#: The four metrics, in the order the card's header claims. Asserted as a list
#: rather than a set: "there is no composite" is part of the slice's design, and
#: a fifth blended column appearing here is the failure that check exists for.
#:
#: ``contested_coverage`` replaced ``contested_rate`` in V7. The two are not the
#: same measurement renamed — the old one was contested findings over all
#: findings (a rate a verbose run could move), the new one is disagreements the
#: context held **both sides of** and the findings named, with the denominator
#: fixed by retrieval. The consequence for this script is that the column is now
#: legitimately nullable, which is why the score check below no longer demands
#: four numbers.
METRICS = ["criteria_recall", "evidence_precision", "provenance", "contested_coverage"]

#: The exclusion chip. The whole experiment is void if this reads otherwise, so
#: it is matched on its exact wording rather than on "0" appearing somewhere.
LEAK_CHIP = "held-out absent · 0 leaks"

#: How many criteria the held-out expert stated in total. This one *is* a fixed
#: literal: it is a property of a hand-extracted dataset, not of a run, and a
#: card that stops listing all 24 has lost the denominator this panel exists to
#: show. Everything downstream of it — how many apply to a portfolio, how many
#: were reached, how many carry an n/a chip — is read off the page and checked
#: for *agreement with itself*, because those numbers move with the dataset and
#: with every rescore, and a validator pinned to one run's arithmetic reports
#: staleness as failure. (It did: this file asserted 18/24 and a recall of
#: 0.278, both of which the committed run had already left behind.)
CRITERIA_TOTAL = 24

#: A timestamped link into the held-out video looks like this. The point of the
#: link is that the reader can watch the expert say the criterion, so the video
#: id in the href has to be the *held-out* one — a link to any other video would
#: render identically and prove nothing.
WATCH = re.compile(rf"youtube\.com/watch\?v={HELD_OUT_VIDEO}&t=(\d+)s")

#: ``0.158`` or ``0.158 0.158–0.210`` — a score cell carries its spread beside
#: it, and ``contested_coverage`` carries its ``N/M in context`` denominator, so
#: the number has to be cut out of the cell rather than compared to it whole.
SCORE = re.compile(r"\d\.\d{3}")

#: The applicable/total split as the card states it.
APPLICABLE = re.compile(rf"(\d+)/{CRITERIA_TOTAL} criteria apply to a portfolio")

#: "reached — 3 of 19 criteria that apply to a portfolio".
REACHED = re.compile(r"reached — (\d+) of (\d+) criteria that apply to a portfolio")

#: "missed — 21 criteria, 5 of which a portfolio cannot be judged on".
MISSED = re.compile(r"missed — (\d+) criteria, (\d+) of which a portfolio cannot be judged on")

#: The denominator the contested column prints beside its score.
IN_CONTEXT = re.compile(r"(\d+)/(\d+) in context")

#: ``critique-15rTnqKBlO8-20260810-051413`` — the run a card is showing. The
#: trailing timestamp sorts lexically, which is how the newest card is found.
RUN_ID = re.compile(r"critique-[A-Za-z0-9_-]+-\d{8}-\d{6}")


# ── measuring what the page actually laid out ────────────────────────────────
def overflow(locator) -> dict[str, int]:
    """The box a node was drawn in versus the content inside it.

    Numbers rather than a boolean, because "this criterion is clipped" and "this
    criterion is short" produce the same boolean and very different evidence.
    """
    return locator.evaluate(
        """node => ({
            scrollWidth: node.scrollWidth,
            clientWidth: node.clientWidth,
            scrollHeight: node.scrollHeight,
            clientHeight: node.clientHeight,
            nowrap: getComputedStyle(node).whiteSpace === 'nowrap' ? 1 : 0,
        })"""
    )


def clipped(box: dict[str, int]) -> bool:
    return (
        box["scrollWidth"] > box["clientWidth"] + 1
        or box["scrollHeight"] > box["clientHeight"] + 1
        or bool(box["nowrap"])
    )


def check_nothing_clipped(session: UserSession, name: str, selector: str, *, minimum: int) -> None:
    """Every node matching ``selector`` shows all of its text.

    ``minimum`` guards the check itself: a selector that stopped matching would
    otherwise pass vacuously, which is the failure mode of every "nothing is
    broken" assertion.
    """
    nodes = session.page.locator(selector)
    count = nodes.count()
    if count < minimum:
        session.check(name, False, f"only {count} {selector} nodes rendered, expected ≥{minimum}")
        return
    worst: list[str] = []
    for index in range(count):
        node = nodes.nth(index)
        box = overflow(node)
        if clipped(box):
            text = " ".join(node.inner_text().split())[:60]
            worst.append(
                f"{text!r} {box['clientWidth']}x{box['clientHeight']}px "
                f"content {box['scrollWidth']}x{box['scrollHeight']}px"
                f"{' nowrap' if box['nowrap'] else ''}"
            )
    session.check(
        name,
        not worst,
        f"{count} nodes laid out in full" if not worst else f"{len(worst)} clipped: {worst[:3]}",
    )


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        page = session.page

        # ── the card is on Experiments at all ─────────────────────────────────
        session.tab("Experiments")
        session.check_visible(
            "Experiments shows a Critique eval card",
            "Critique eval — held-out expert",
            shot=True,
        )
        cards = page.locator("section.exp-card").filter(has_text="Critique eval — held-out expert")
        # One card per committed critique run, newest first. This used to assert
        # exactly one; V7 committed a second run and there are now two, so what
        # is asserted instead is the property that makes two survivable — each
        # card names the run it is, so a reader can tell which numbers are
        # current. See the note at the end of this run: two cards for the same
        # held-out video, headed with two different fourth metrics, is a real
        # problem this panel has and this check does not fix.
        matches = [
            RUN_ID.search(" ".join(cards.nth(i).inner_text().split())) for i in range(cards.count())
        ]
        named = [match.group(0) for match in matches if match is not None]
        session.check(
            "every critique card says which run it is",
            cards.count() >= 1 and len(named) == cards.count() and len(set(named)) == len(named),
            f"{cards.count()} critique cards, identified as {named}",
        )
        if cards.count() == 0 or len(named) != cards.count():
            return exit_code(session)
        # The newest run is the one the slice's claim is about. Sorted by the
        # timestamp in the run id rather than by DOM position, so a change to
        # the panel's ordering cannot silently point this script at stale
        # numbers — which is exactly the failure the second card creates.
        newest = max(range(cards.count()), key=lambda i: named[i])
        card = cards.nth(newest)
        head = " ".join(card.inner_text().split())

        session.check(
            "the card names the held-out expert",
            HELD_OUT_TITLE in head,
            f"title {'present' if HELD_OUT_TITLE in head else 'MISSING'}",
        )
        applicable_match = APPLICABLE.search(head)
        session.check(
            "the card states the applicable/total criteria split",
            applicable_match is not None,
            head[:200],
        )
        if applicable_match is None:
            return exit_code(session)
        # Read once here and reused everywhere below. The recall denominator,
        # the missed column and the n/a chips all have to be this same number,
        # and a card whose three statements of it disagree is exactly the bug
        # this panel would otherwise hide behind a plausible-looking fraction.
        applicable = int(applicable_match.group(1))
        session.check(
            "fewer criteria apply to a portfolio than the expert stated in total",
            0 < applicable < CRITERIA_TOTAL,
            f"{applicable} of {CRITERIA_TOTAL} apply — the n/a chips below must account "
            f"for the other {CRITERIA_TOTAL - applicable}",
        )

        # ── the exclusion chip ────────────────────────────────────────────────
        chip = card.locator(".exp-tag")
        chip_text = " ".join(chip.first.inner_text().split()) if chip.count() else ""
        session.check(
            "the held-out-absent chip is on the card",
            chip_text == LEAK_CHIP,
            f"chip reads {chip_text!r}, expected {LEAK_CHIP!r}",
            shot=True,
        )
        session.check(
            "the chip is styled as a pass, not a warning",
            "ok" in (chip.first.get_attribute("class") or ""),
            chip.first.get_attribute("class") or "",
        )

        # ── the baseline row: four non-null scores ────────────────────────────
        # Compared case-folded throughout: this tab renders headings, chips and
        # microlabels through ``text-transform: uppercase``, so the string the
        # DOM hands back is not the string in the source. Matching the rendered
        # casing would make these checks fail on a pure styling change, and
        # matching the source casing would make them fail today.
        header = [
            " ".join(th.inner_text().split()).lower() for th in card.locator("thead th").all()
        ]
        session.check(
            "the table heads the four metrics and no composite",
            [h for h in header if h in METRICS] == METRICS and "composite" not in header,
            f"header {header}",
        )

        row = card.locator("tbody tr").filter(has_text=BASELINE_SETUP).first
        session.check(
            "the chunk-dump baseline row is present",
            row.count() > 0,
            f"looking for {BASELINE_SETUP!r}",
        )
        # Asserted on the chip element, not on the row's text. The chip renders
        # glued to the setup name with no separator ("rag_llm_filteredBASE"), so
        # a word-boundary test on the row string is a check that cannot fail
        # honestly — it would report the chip missing while it is on screen.
        base_tag = row.locator(".exp-basetag")
        session.check(
            "the baseline row carries the base chip",
            base_tag.count() == 1 and base_tag.first.inner_text().strip().lower() == "base",
            f"{base_tag.count()} chips reading {base_tag.first.inner_text().strip()!r}"
            if base_tag.count()
            else "no chip",
        )

        # The scores as the reader sees them. A cell is not just a number: recall
        # prints its spread across matcher repeats beside it and contested
        # coverage prints its denominator, so the score is cut out rather than
        # compared whole.
        cells = [" ".join(td.inner_text().split()) for td in row.locator("td.num").all()]
        scores = cells[: len(METRICS)]
        graded = dict(zip(METRICS, scores))
        deterministic = ["criteria_recall", "evidence_precision", "provenance"]
        session.check(
            "the three scored metrics render as numbers, none as —",
            all(SCORE.search(graded[m]) for m in deterministic),
            f"scores rendered as {scores}",
            shot=True,
        )

        # ── the nullable column, and why it is allowed to be null ─────────────
        # V7 made contested coverage a fraction over "disagreements this run had
        # both sides of in context", which can be 0/0. That is the one metric on
        # this card that may honestly print nothing, and the check is therefore
        # not "it is a number" but "it never prints a bare em dash": a reader who
        # cannot tell 'averaged a conflict away' from 'was never shown one' is
        # being told the wrong thing by the same three characters.
        contested = graded["contested_coverage"]
        in_context = IN_CONTEXT.search(contested)
        session.check(
            "the contested-coverage cell states its denominator, not just a score",
            in_context is not None,
            f"contested_coverage cell reads {contested!r}",
        )
        if in_context is not None:
            named, available = int(in_context.group(1)), int(in_context.group(2))
            has_score = SCORE.search(contested) is not None
            session.check(
                "an em dash there means no disagreement was in context, never 0.000",
                has_score == (available > 0) and named <= available,
                f"cell reads {contested!r} — {named} named of {available} in context, "
                f"{'scored' if has_score else 'unscored'}",
            )
        session.check(
            "the row shows the leak count as zero",
            cells[-1] == "0" if cells else False,
            f"trailing numeric cells {cells[-3:]}",
        )

        # ── expand: the per-criterion evidence ────────────────────────────────
        row.get_by_role("button", name="Expand").click()
        page.wait_for_timeout(1200)
        detail = card.locator(".crit-detail")
        session.check(
            "expanding the row opens the criteria detail",
            detail.count() == 1 and detail.first.is_visible(),
            f"{detail.count()} detail blocks",
            shot=True,
        )
        if detail.count() != 1:
            return exit_code(session)

        body = " ".join(detail.first.inner_text().split()).lower()
        reached_head = REACHED.search(body)
        missed_head = MISSED.search(body)
        session.check(
            "the detail heads the reached column with the applicable denominator",
            reached_head is not None and int(reached_head.group(2)) == applicable,
            f"{reached_head.group(0) if reached_head else body[:180]!r}; "
            f"the card above said {applicable} apply",
        )
        session.check(
            "the detail heads the missed column and says how many are n/a",
            missed_head is not None and int(missed_head.group(2)) == CRITERIA_TOTAL - applicable,
            f"{missed_head.group(0) if missed_head else body[:400]!r}; "
            f"{CRITERIA_TOTAL - applicable} of {CRITERIA_TOTAL} do not apply",
        )
        if reached_head is None or missed_head is None:
            return exit_code(session)
        reached_count = int(reached_head.group(1))

        hits = detail.first.locator("li.crit-item.hit")
        misses = detail.first.locator("li.crit-item.miss")
        session.check(
            "every reached criterion is listed individually",
            hits.count() == reached_count,
            f"{hits.count()} hit rows for a heading that claims {reached_count}",
        )
        session.check(
            "every criterion the expert stated is on screen, reached or missed",
            hits.count() + misses.count() == CRITERIA_TOTAL
            and misses.count() == int(missed_head.group(1)),
            f"{hits.count()} reached + {misses.count()} missed = "
            f"{hits.count() + misses.count()} of {CRITERIA_TOTAL}",
        )

        # Every reached criterion has to show the finding that reached it — a
        # bare tick beside a rule is the unfalsifiable version of this.
        found = detail.first.locator("li.crit-item.hit .crit-found")
        session.check(
            "every reached criterion names the finding that reached it",
            found.count() == hits.count(),
            f"{found.count()} findings shown for {hits.count()} matches",
        )
        # The matcher may also supply a reason. It currently supplies none — the
        # committed run carries `why: null` on every counted match — so this is
        # recorded rather than asserted. Asserting it would fail the panel for a
        # gap in the scorer upstream of it; ignoring it would let the panel quietly
        # lose the one field that says *why* two differently worded rules are the
        # same rule. See the note at the end of this run.
        reasons = detail.first.locator("li.crit-item.hit .crit-why")
        session.note(
            f"matcher reasons on screen: {reasons.count()} for {hits.count()} reached "
            "criteria. The run file has why=null on every counted match, so the "
            "'↳ finding — reason' line renders without its reason. The reader can "
            "still see which finding matched, but not the matcher's argument for it."
        )

        # ── the timestamps are clickable and point at the held-out video ──────
        links = detail.first.locator("a.crit-ts")
        hrefs = [links.nth(i).get_attribute("href") or "" for i in range(links.count())]
        into_held_out = [h for h in hrefs if WATCH.search(h)]
        session.check(
            "every criterion carries a timestamp linking into the held-out video",
            len(into_held_out) >= 24,
            f"{len(into_held_out)} of {len(hrefs)} crit-ts links resolve to {HELD_OUT_VIDEO}",
        )
        labels = [
            " ".join(links.nth(i).inner_text().split())
            for i in range(links.count())
            if WATCH.search(hrefs[i])
        ]
        session.check(
            "the timestamps are shown as readable clock times",
            all(re.fullmatch(r"\d+:\d{2}", label) for label in labels[:24]),
            f"first labels {labels[:5]}",
        )

        # ── the n/a chips that set the denominator ────────────────────────────
        # Every criterion the recall denominator excluded has to be excluded
        # *visibly*, on the row it excluded, with the reason. A denominator a
        # reader cannot count back to is a denominator they have to take on
        # trust, and 3/19 and 3/24 are very different claims about this system.
        na = detail.first.locator("span.crit-na")
        na_text = {" ".join(na.nth(i).inner_text().split()).lower() for i in range(na.count())}
        excluded = CRITERIA_TOTAL - applicable
        session.check(
            "every criterion outside the denominator carries an n/a chip",
            na.count() == excluded and na_text == {"n/a to a portfolio"},
            f"{na.count()} chips reading {na_text}, "
            f"for {excluded} criteria the card says do not apply",
            shot=True,
        )
        na_rows = [
            " ".join(misses.nth(i).inner_text().split())
            for i in range(misses.count())
            if misses.nth(i).locator("span.crit-na").count()
        ]
        chipped_ids = {row.split()[0] for row in na_rows}
        session.check(
            "the chipped criteria are named individually, not just counted",
            len(chipped_ids) == excluded
            and all(re.fullmatch(r"c\d{2}", cid) for cid in chipped_ids),
            f"chipped {sorted(chipped_ids)}",
        )

        # The heading and the score must be the same number. The panel builds
        # "reached — N of A" from *all* matched rows while ``criteria_recall``
        # counts only the applicable ones, so the two agree exactly as long as
        # no criterion outside the applicable subset is ever matched. That is
        # true of this run and is not enforced anywhere, which is precisely why
        # a reader-facing check belongs here: the day a resume-only criterion
        # matches, the card will print a fraction that is not the score beside it.
        recall_cell = SCORE.search(graded["criteria_recall"])
        session.check(
            "the reached heading is the same number as the recall score",
            recall_cell is not None
            and abs(hits.count() / applicable - float(recall_cell.group(0))) < 0.001,
            f"heading says {hits.count()}/{applicable} = {hits.count() / applicable:.3f}, "
            f"the row shows {graded['criteria_recall']!r}",
        )

        # ── nothing is cut off ────────────────────────────────────────────────
        session.check(
            "the detail row escaped the table's nowrap",
            not bool(overflow(detail.first)["nowrap"]),
            "white-space on .crit-detail",
        )
        check_nothing_clipped(
            session,
            "every criterion sentence renders in full",
            ".crit-detail .crit-rule",
            minimum=24,
        )
        check_nothing_clipped(
            session, "every verbatim quote renders in full", ".crit-detail .crit-quote", minimum=24
        )
        check_nothing_clipped(
            session,
            "every matched finding + reason renders in full",
            ".crit-detail .crit-found",
            minimum=5,
        )
        check_nothing_clipped(
            session, "both column headings render in full", ".crit-detail .microlabel", minimum=2
        )

        # The card itself must not force the page sideways: a colSpan row that
        # widens the table past the viewport is the clipping bug in its other
        # form, and .exp-scroll would hide it inside a scroller nobody scrolls.
        page_box = page.evaluate(
            "() => ({doc: document.documentElement.scrollWidth, win: window.innerWidth})"
        )
        session.check(
            "the expanded card does not push the page horizontally",
            page_box["doc"] <= page_box["win"] + 1,
            f"document {page_box['doc']}px in a {page_box['win']}px viewport",
        )

        # ── collapse puts it away again ───────────────────────────────────────
        row.get_by_role("button", name="Collapse").click()
        page.wait_for_timeout(600)
        session.check(
            "collapsing the row hides the criteria again",
            card.locator(".crit-detail").count() == 0,
            f"{card.locator('.crit-detail').count()} detail blocks after collapse",
        )

        session.note(
            f"{cards.count()} Critique eval cards are on this tab, one per committed "
            "run, for the same held-out video and the same artifact. Since V7 they "
            "no longer agree: the newest is headed CONTESTED_COVERAGE and the "
            "superseded one CONTESTED_RATE — two different measurements under two "
            "names in the same column position — and the same baseline setup prints "
            "a different evidence_precision on each. Nothing on the page marks one "
            "as current. This script reads the newest by run id; a reader scrolling "
            "the tab has no such rule."
        )
        session.note(
            f"criteria_recall {graded['criteria_recall']} is "
            f"{hits.count()}/{applicable} over the applicable subset; "
            f"criteria_recall_all ({hits.count()}/{CRITERIA_TOTAL}) is a separate "
            "column and is not the headline number."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
