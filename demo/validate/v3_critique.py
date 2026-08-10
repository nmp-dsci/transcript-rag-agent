"""Replay the V3 held-out critique eval in the browser and judge what it shows.

    uv run python -m src.cli serve --port 8021                 # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v3_critique

V3 claims something no other panel in this app claims: that the distilled corpus
reaches a named expert's conclusions *without having seen that expert*. The
number that carries the claim is ``criteria_recall``, and a recall number on its
own is unfalsifiable — 0.278 is only meaningful if a reader can open the row and
see which five of the eighteen were reached, which thirteen were not, and what
the expert actually said at the second they said it.

So this script asserts the chain a sceptic would walk, on rendered page text:

* the **Critique eval card** exists on Experiments and names the held-out video;
* the **``held-out absent · 0 leaks`` chip** is on the card — the exclusion claim
  has to be visible, not buried in a JSON file;
* the **baseline row** carries the ``base`` chip and four *non-null* scores, so a
  metric the run could not measure cannot masquerade as one it measured;
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
METRICS = ["criteria_recall", "evidence_precision", "provenance", "contested_rate"]

#: The exclusion chip. The whole experiment is void if this reads otherwise, so
#: it is matched on its exact wording rather than on "0" appearing somewhere.
LEAK_CHIP = "held-out absent · 0 leaks"

#: Criteria the dataset marks resume-only. Each must be visibly flagged in the
#: missed column, because these six are what turn a recall of 5/24 into 5/18.
RESUME_ONLY = ["c01", "c08", "c09", "c18", "c20", "c21"]

#: A timestamped link into the held-out video looks like this. The point of the
#: link is that the reader can watch the expert say the criterion, so the video
#: id in the href has to be the *held-out* one — a link to any other video would
#: render identically and prove nothing.
WATCH = re.compile(rf"youtube\.com/watch\?v={HELD_OUT_VIDEO}&t=(\d+)s")


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
        card = page.locator("section.exp-card").filter(has_text="Critique eval — held-out expert")
        session.check(
            "exactly one critique card is rendered",
            card.count() == 1,
            f"{card.count()} cards",
        )
        if card.count() != 1:
            return exit_code(session)
        card = card.first
        head = " ".join(card.inner_text().split())

        session.check(
            "the card names the held-out expert",
            HELD_OUT_TITLE in head,
            f"title {'present' if HELD_OUT_TITLE in head else 'MISSING'}",
        )
        session.check(
            "the card states the applicable/total criteria split",
            "18/24 criteria apply to a portfolio" in head,
            head[:200],
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

        # The scores as the reader sees them — an em dash is what a null renders
        # as, and a null in the baseline is a metric this harness cannot score.
        cells = [" ".join(td.inner_text().split()) for td in row.locator("td.num").all()]
        scores = cells[: len(METRICS)]
        numeric = [c for c in scores if re.fullmatch(r"\d\.\d{3}", c)]
        session.check(
            "all four baseline scores render as numbers, none as —",
            len(numeric) == len(METRICS),
            f"scores rendered as {scores}",
            shot=True,
        )
        session.check(
            "the recall shown is the applicable-subset one",
            scores[0] == "0.278",
            f"criteria_recall cell reads {scores[0]!r}",
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
        session.check(
            "the detail heads the reached column with the applicable denominator",
            "reached — 5 of 18 criteria that apply to a portfolio" in body,
            body[:180],
        )
        session.check(
            "the detail heads the missed column and says how many are n/a",
            "missed — 19 criteria, 6 of which a portfolio cannot be judged on" in body,
            body[:400],
        )

        hits = detail.first.locator("li.crit-item.hit")
        misses = detail.first.locator("li.crit-item.miss")
        session.check(
            "five reached criteria are listed individually",
            hits.count() == 5,
            f"{hits.count()} hit rows",
        )
        session.check(
            "nineteen missed criteria are listed individually",
            misses.count() == 19,
            f"{misses.count()} miss rows",
        )

        # Every reached criterion has to show the finding that reached it and
        # why — a bare tick beside a rule is the unfalsifiable version of this.
        reasons = detail.first.locator("li.crit-item.hit .crit-why")
        session.check(
            "every reached criterion shows the matcher's reason",
            reasons.count() == hits.count(),
            f"{reasons.count()} reasons for {hits.count()} matches",
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
        na = detail.first.locator("span.crit-na")
        na_text = {" ".join(na.nth(i).inner_text().split()).lower() for i in range(na.count())}
        session.check(
            "six criteria are chipped as n/a to a portfolio",
            na.count() == 6 and na_text == {"n/a to a portfolio"},
            f"{na.count()} chips reading {na_text}",
            shot=True,
        )
        na_rows = [
            " ".join(misses.nth(i).inner_text().split())
            for i in range(misses.count())
            if misses.nth(i).locator("span.crit-na").count()
        ]
        chipped_ids = {row.split()[0] for row in na_rows}
        session.check(
            "the chipped criteria are exactly the six resume-only ones",
            chipped_ids == set(RESUME_ONLY),
            f"chipped {sorted(chipped_ids)}, dataset says {RESUME_ONLY}",
        )

        # The heading and the score must be the same number. The panel builds
        # "reached — N of 18" from *all* matched rows while ``criteria_recall``
        # counts only the applicable ones, so the two agree exactly as long as
        # no criterion outside the applicable subset is ever matched. That is
        # true of this run and is not enforced anywhere, which is precisely why
        # a reader-facing check belongs here: the day a resume-only criterion
        # matches, the card will print a fraction that is not the score beside it.
        session.check(
            "the reached heading is the same number as the recall score",
            abs(hits.count() / 18 - float(scores[0])) < 0.001,
            f"heading says {hits.count()}/18 = {hits.count() / 18:.3f}, the row shows {scores[0]}",
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
            "criteria_recall 0.278 is 5/18 over the applicable subset; "
            "criteria_recall_all (5/24 = 0.208) is not shown on this card."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
