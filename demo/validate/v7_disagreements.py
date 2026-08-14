"""Walk the V7 disagreement layer as a reader who does not trust it, and judge.

    uv run python -m src.cli serve --port 8021                      # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v7_disagreements

V7 claims the one thing every other view in this app is built to prevent: that
when two creators contradict each other, the corpus is allowed to say so, with
both sides intact and no answer. That claim is only settled in the browser, and
it fails in two opposite directions that a passing unit test cannot tell apart.

It fails if the layer is **not there** — no sub-tab, no cards, an axis with one
side, a quote a reader cannot check. And it fails if the layer **picks a
winner**: a first position that reads as the right one, a wider column, a tick, a
"but actually", any of which turns "the corpus disagrees" into "the corpus says
X" while looking exactly the same in a snapshot test.

So this script asserts, on rendered page text only:

* **RAG Pipeline → Disagreements** exists as a sub-tab and opens a list of cards;
* every card leads with an **axis stated as a question**. A question has no
  answer in it; a declarative axis ("10pt is too small") is a verdict wearing an
  axis's clothes, and the difference is one character;
* every card has **exactly two sides, from two different creators**, each with a
  position, a quote of real length, and a timestamp;
* the two sides are **symmetric in the layout** — same label vocabulary, same
  rendered width to within a pixel or two. Asymmetry is how a page picks a side
  without saying so, and it is invisible to anything but a real browser;
* **no winner language** anywhere a card speaks in its own voice. Checked against
  the card's own prose only: creator names, quotes and video titles are the
  corpus's words, and one of the videos in this corpus is literally called "Write
  a Better Resume", so a naive scan of the whole card reports the corpus's
  vocabulary as the app's editorial;
* every timestamp link carries **exactly one ``t=`` and one ``v=``**. Eight
  videos in this corpus already ship a ``t=`` inside ``source_url``; a layer that
  appended a second would produce a link that renders identically, resolves, and
  opens at the wrong second — or at no second at all;
* the link's offset **matches the clock printed beside it**, so the timestamp a
  reader reads and the timestamp they land on are the same one;
* the **denominator is on screen**: how many pairs were adjudicated, the
  precision, and the rejection tally. "4 disagreements" with no denominator is
  the number this layer must never be reducible to;
* the count carries its **spread**. The judge does not agree with itself, so the
  headline is a measurement, and a measurement printed as a bare integer is read
  as exact. Either a ± beside the count or the observed spread of independent
  sub-runs settles it; neither is optional once the count is small enough that
  one look either way changes the story;
* the **calibration strip** shows both halves — planted contradictions that
  surfaced and complementary pairs that did not — because an adjudicator that
  says "conflict" to everything scores full marks on the first half alone;
* nothing is **clipped**, measured at four widths. This app has shipped
  ``nowrap`` truncation on unbounded LLM text before, and a truncated quote is
  worse here than anywhere else in the app: the whole promise of the card is that
  the reader can check the words against the video.

Nothing here reads ``/api/conflicts`` to prove a UI claim. That four conflicts
sit in a JSON file and that a reader can walk one to a video at the second it was
said are different claims, and this repo has shipped the first without the
second.

WHAT THIS SCRIPT DOES NOT SETTLE — and no browser script could. It checks that
each card is *shaped* like a disagreement and *presented* without a winner. It
cannot check that the two people actually disagree, and it cannot re-run the
adjudicator to find out; that would make this file a test of one LLM's mood.

What it *can* insist on is that the page does not hide how unsure the number is.
The layer was rebuilt to adjudicate each corpus pair several times and carry only
on a strict majority, which is the right fix and closed the asymmetry this script
first recorded (probes repeated, corpus pairs not). It did not make the count
stable: pairs the judge answers at even odds carry a majority half the time at
*every* repeat count, and the modelled spread only falls from about 1.1 at three
looks to 0.7 at twenty-one. So a small count is a legitimate result, and printing
it without its error bar is not.

ENVIRONMENT NOTE — this machine's Metal compiler XPC service is wedged and
``chromium.launch()`` times out at 0% CPU with no page and no navigation, so the
Playwright half of this file could not be executed at evaluation time. Every
assertion below was executed instead as the equivalent DOM query through an
already-open Chrome against the same running app, and the results are recorded in
``artifacts/v7_disagreements/verdict.json`` with that limitation noted there.
v2, v3 and v4 recorded the same constraint the same way; re-run this file on a
healthy machine to regenerate it with screenshots.
"""

from __future__ import annotations

import re
import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v7_disagreements"

#: The sub-tab label a reader clicks. Text, not a test id: a label that changed
#: to something a human could no longer find must fail this script.
SUBTAB = "Disagreements"

#: The gate: at least this many cross-creator disagreements, each with both sides
#: sourced. Three is the number the slice was commissioned against.
MIN_CONFLICTS = 3

#: A quote shorter than this proves nothing — a three-word "it depends" resolves
#: perfectly against any transcript. Matches ``MIN_QUOTE_WORDS`` in
#: :mod:`src.rag.conflicts`, so a quote that shipped from there must clear the
#: same bar when it is read off the page.
MIN_QUOTE_WORDS = 8

#: Words that decide something. Applied only to the text a **card writes in its
#: own voice** — the axis, the two one-line positions, the incompatibility note —
#: never to quotes, creator names or video titles, which are the corpus speaking.
#: "Write a Better Resume" and "answers to the wrong question" are real titles in
#: this corpus, and a scan of whole cards flags both.
WINNER = re.compile(
    r"\b(winner|wins?|won|correct|incorrect|right answer|mistaken|debunk\w*|"
    r"verdict|the truth|actually right|is wrong|proven wrong|outdated advice|"
    r"the better|the best|we recommend|you should instead|settles?)\b",
    re.I,
)

#: mm:ss or h:mm:ss as the provenance line prints it.
CLOCK = re.compile(r"\bat (\d+):(\d{2})\b")


# ── measuring what the page actually laid out ────────────────────────────────
def overflow(locator) -> dict[str, int]:
    """The box a node was drawn in versus the content inside it.

    Numbers rather than a boolean: "this quote is clipped" and "this quote is
    short" produce the same boolean and very different evidence.
    """
    return locator.evaluate(
        """node => ({
            scrollWidth: node.scrollWidth,
            clientWidth: node.clientWidth,
            scrollHeight: node.scrollHeight,
            clientHeight: node.clientHeight,
            nowrap: getComputedStyle(node).whiteSpace === 'nowrap' ? 1 : 0,
            clamped: getComputedStyle(node).webkitLineClamp !== 'none' ? 1 : 0,
        })"""
    )


def clipped(box: dict[str, int]) -> bool:
    return (
        box["scrollWidth"] > box["clientWidth"] + 1
        or box["scrollHeight"] > box["clientHeight"] + 1
        or bool(box["nowrap"])
        or bool(box["clamped"])
    )


def check_nothing_clipped(session: UserSession, name: str, selector: str, *, minimum: int) -> None:
    """Every node matching ``selector`` shows all of its text.

    ``minimum`` guards the check itself: a selector that stopped matching would
    pass vacuously, which is the failure mode of every "nothing is broken"
    assertion.
    """
    nodes = session.page.locator(selector)
    count = nodes.count()
    if count < minimum:
        session.check(name, False, f"only {count} {selector} nodes rendered, expected >={minimum}")
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
                f"{' line-clamped' if box['clamped'] else ''}"
            )
    session.check(
        name,
        not worst,
        f"{count} nodes laid out in full" if not worst else f"{len(worst)} clipped: {worst[:3]}",
    )


def words(text: str) -> int:
    return len(text.replace("“", "").replace("”", "").split())


def seconds_from_clock(label: str) -> int | None:
    match = CLOCK.search(label)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def main() -> int:  # noqa: PLR0915 - one linear click path, read top to bottom
    require_server()
    with UserSession(SLICE) as session:
        page = session.page

        # ── the layer is reachable at all ─────────────────────────────────────
        session.tab("RAG Pipeline")
        subtab = page.get_by_role("button", name=SUBTAB, exact=True)
        session.check(
            "RAG Pipeline offers a Disagreements sub-tab",
            subtab.count() >= 1,
            f"{subtab.count()} buttons labelled {SUBTAB!r}",
            shot=True,
        )
        if subtab.count() == 0:
            return exit_code(session)
        subtab.first.click()
        page.wait_for_timeout(1400)

        cards = page.locator("article.dis-card")
        session.check(
            "the tab lists disagreements as cards",
            cards.count() >= MIN_CONFLICTS,
            f"{cards.count()} conflict cards rendered, gate is >={MIN_CONFLICTS}",
            shot=True,
        )
        if cards.count() == 0:
            return exit_code(session)

        # ── the denominator, on screen, beside the count ──────────────────────
        # "4 disagreements" on its own is the number this layer must never be
        # reducible to: a run that looked at six pairs and a run that looked at
        # 478 print the same 4.
        statline = " ".join(page.locator(".dis-statline").inner_text().split())
        session.check(
            "the count is shown with the pairs it was drawn from",
            bool(re.search(r"\d+ disagreements", statline))
            and bool(re.search(r"from \d+ candidate pairs adjudicated", statline)),
            statline[:180],
        )
        session.check(
            "a precision is shown, so proposing more candidates cannot flatter the layer",
            bool(re.search(r"precision \d\.\d{3}", statline)),
            statline[:180],
        )
        # "channels", not "creators". This wording changed under the script and
        # the change is *correct*: the field behind it is ``channel_name``, the
        # uploading channel, and "two creators disagree" is a stronger claim
        # than that field can carry — two of the sides that shipped in the first
        # build were spoken by a guest and by an interviewee, not by the channel
        # owner. The script follows the honest word rather than holding the view
        # to the overclaiming one.
        session.check(
            "the channels and videos the layer touched are counted",
            bool(re.search(r"\d+ channels . \d+ videos", statline)),
            statline[:180],
        )
        # The count is a measurement taken with a judge that does not agree with
        # itself, and a measurement printed without its spread is read as exact.
        # This is the whole difference between "the corpus contains 2
        # disagreements" and "this run of this judge found 2, and another run of
        # the same judge over the same pairs would plausibly have found 1 or 3".
        #
        # Three renderings satisfy that, and the check takes any of them. The
        # first two are what a build that measured its own spread prints. The
        # third is what a build that did *not* prints, and it is deliberately
        # not a modelled decimal dressed up in the same slot and shape as a
        # measured one: a reader who cannot tell a computed number from a run's
        # own measurement has been given a worse page, not a better one. What it
        # must do instead is say the spread is missing, and still leave the
        # reader holding "about N" rather than "exactly N". A bare integer with
        # none of the three fails, which is the state this check was added for.
        subruns = " ".join(page.locator(".dis-rejected").all_inner_texts())
        caveat = " ".join(" ".join(page.locator(".dis-caveat").all_inner_texts()).split())
        measured = bool(re.search(r"\d+ disagreements\s*±\s*[\d.]+", statline))
        observed = bool(re.search(r"groups of three .* yields", " ".join(subruns.split())))
        stated = (
            bool(re.search(r"spread", caveat, re.I))
            and bool(re.search(r"±|give or take|roughly one|approximate", caveat, re.I))
            and bool(re.search(r"about \d+|nearer \d+|approximate", caveat + statline, re.I))
        )
        session.check(
            "the headline count carries its spread, not a bare integer",
            measured or observed or stated,
            f"measured±={measured} observed-subruns={observed} stated-caveat={stated}; "
            f"{statline[:120]}",
        )
        # A fallback that outlives the gap it covers is its own defect: a page
        # that prints a measured ± *and* a paragraph explaining that no ± was
        # recorded is contradicting itself, and the reader has no way to know
        # which half is stale. The two are mutually exclusive by construction, so
        # assert it — this is cheap on any single build and is the check that
        # would catch the caveat being left permanently wired in.
        session.check(
            "the missing-spread note retires once a build records its own spread",
            not (measured and caveat),
            "a measured ± and a 'no spread recorded' note are not both on the page"
            if not (measured and caveat)
            else f"both present: {statline[:80]!r} alongside {caveat[:80]!r}",
        )
        # Counted as "at least one node carries the tally" rather than "exactly
        # one node has this class": the sub-run line the check above is asking
        # for is rendered with the same ``.dis-rejected`` class, so an exact
        # count of 1 would start failing on the day the spread ships — a check
        # that breaks when the defect beside it is fixed is worse than no check.
        session.check(
            "the rejection tally is on the page, not only in the artifact",
            bool(re.search(r"\d+ not a conflict", " ".join(subruns.split()))),
            " ".join(subruns.split())[:160] if subruns else "no tally",
        )

        # ── the calibration strip: both halves ────────────────────────────────
        probehead = page.locator(".dis-probehead")
        session.check(
            "the calibration strip is on the same page as the findings",
            probehead.count() == 1,
            f"{probehead.count()} calibration headers",
        )
        if probehead.count() == 1:
            summary = " ".join(probehead.inner_text().split())
            planted = re.search(r"(\d+)/(\d+) planted contradictions surfaced", summary)
            complementary = re.search(
                r"(\d+)/(\d+) complementary pairs correctly rejected", summary
            )
            session.check(
                "calibration reports the planted contradictions it surfaced",
                planted is not None and planted.group(1) == planted.group(2),
                summary[:200],
            )
            # The half that carries the weight. An adjudicator answering
            # "conflict" every time passes every planted pair and fails all of
            # these, so a strip that reported only the first half would be
            # decoration.
            session.check(
                "calibration also reports the complementary pairs it rejected",
                complementary is not None
                and int(complementary.group(2)) >= 2
                and complementary.group(1) == complementary.group(2),
                summary[:200],
            )
            probehead.click()
            page.wait_for_timeout(500)
            probes = page.locator("li.dis-probe")
            session.check(
                "each calibration probe can be opened and read individually",
                probes.count() >= 4,
                f"{probes.count()} probes listed",
                shot=True,
            )
            # A planted pair that "passed" by being fused into one blended
            # statement would look identical in a pass count. The axis and the
            # two positions are what make it checkable.
            axes = page.locator(".dis-probeaxis")
            session.check(
                "a surfaced planted pair shows its axis and both positions, not a blend",
                axes.count() >= 1
                and all(
                    "against" in " ".join(axes.nth(i).inner_text().split())
                    for i in range(axes.count())
                ),
                f"{axes.count()} probes name an axis",
            )

        # ── every card: an axis, two sides, two creators ──────────────────────
        total = cards.count()
        bad_axis: list[str] = []
        one_sided: list[int] = []
        same_creator: list[int] = []
        short_quotes: list[str] = []
        for index in range(total):
            card = cards.nth(index)
            axis = " ".join(card.locator(".dis-axis").inner_text().split())
            # An axis is a question. A declarative axis is a verdict with the
            # question mark filed off, and this is the cheapest place to catch it.
            if not axis.endswith("?"):
                bad_axis.append(axis[:80])
            sides = card.locator(".dis-side")
            if sides.count() != 2:
                one_sided.append(index)
                continue
            creators = {
                " ".join(sides.nth(k).locator(".dis-creator").inner_text().split())
                for k in range(2)
            }
            if len(creators) != 2:
                same_creator.append(index)
            for k in range(2):
                quote = " ".join(sides.nth(k).locator(".dis-quote").inner_text().split())
                if words(quote) < MIN_QUOTE_WORDS:
                    short_quotes.append(f"card {index} side {k}: {quote[:60]!r}")

        session.check(
            "every axis is stated as a question, not as an answer",
            not bad_axis,
            f"all {total} axes end in '?'" if not bad_axis else f"declarative: {bad_axis}",
        )
        session.check(
            "every disagreement has exactly two sides",
            not one_sided,
            f"all {total} cards show two sides"
            if not one_sided
            else f"cards {one_sided} do not show two",
        )
        session.check(
            "both sides of every disagreement come from different creators",
            not same_creator,
            f"all {total} cards are cross-creator"
            if not same_creator
            else f"cards {same_creator} quote one creator twice",
        )
        session.check(
            f"every quote is at least {MIN_QUOTE_WORDS} words",
            not short_quotes,
            f"{2 * total} quotes all >= {MIN_QUOTE_WORDS} words"
            if not short_quotes
            else f"{len(short_quotes)} too short: {short_quotes[:3]}",
        )

        # ── the layout does not pick a side ───────────────────────────────────
        labels = page.locator(".dis-sidelabel")
        label_text = [
            " ".join(labels.nth(i).inner_text().split()).lower() for i in range(labels.count())
        ]
        session.check(
            "the two sides are labelled without ranking them",
            set(label_text) == {"one view", "the other"},
            f"side labels in use: {sorted(set(label_text))}",
        )
        # Same width to the pixel. A wider column is an editorial choice that no
        # snapshot test can see and every reader can.
        widths = page.locator(".dis-side").evaluate_all(
            "nodes => nodes.map(n => Math.round(n.getBoundingClientRect().width))"
        )
        pairs = [(widths[i], widths[i + 1]) for i in range(0, len(widths) - 1, 2)]
        lopsided = [p for p in pairs if abs(p[0] - p[1]) > 2]
        session.check(
            "the two sides are drawn the same width, so neither reads as the main one",
            not lopsided,
            f"{len(pairs)} side pairs, widest gap "
            f"{max((abs(a - b) for a, b in pairs), default=0)}px",
        )

        # ── no winner language in the app's own voice ─────────────────────────
        # Read from the card's own prose only. The quote, the creator name and
        # the video title are the corpus talking, and this corpus contains a
        # video called "Write a Better Resume" and another called "...answers to
        # the wrong question" — scanning whole cards reports those as editorial.
        editorial: list[str] = []
        for index in range(total):
            card = cards.nth(index)
            voice = " ".join(
                " ".join(card.locator(selector).all_inner_texts())
                for selector in (".dis-axis", ".dis-position", ".dis-why p")
            )
            hits = WINNER.findall(voice)
            if hits:
                editorial.append(f"card {index}: {sorted(set(hits))}")
        session.check(
            "no card declares a winner in its own voice",
            not editorial,
            f"{total} cards state an axis and two positions and nothing else"
            if not editorial
            else f"verdict language: {editorial}",
        )
        page_text = " ".join(page.locator(".dis-wrap").inner_text().split()).lower()
        session.check(
            "the page says outright that it is not answering",
            "never a verdict" in page_text or "no verdict" in page_text,
            page_text[:200],
        )
        session.check(
            "no card renders a resolution field",
            not re.search(r"\b(winner|resolution|the corpus says)\b", page_text),
            "no winner/resolution field in the rendered layer",
        )

        # ── provenance: one t=, one v=, and the clock a reader reads ──────────
        links = page.locator("a.dis-ts")
        hrefs = [links.nth(i).get_attribute("href") or "" for i in range(links.count())]
        session.check(
            "every side carries a timestamp link",
            links.count() == 2 * total,
            f"{links.count()} links for {total} cards",
        )
        # Eight videos in this corpus already carry a `t=` in `source_url`. A
        # link that appended a second one renders identically and lands wrong,
        # so this is counted rather than pattern-matched.
        double_t = [h for h in hrefs if len(re.findall(r"[?&]t=", h)) != 1]
        double_v = [h for h in hrefs if len(re.findall(r"[?&]v=", h)) != 1]
        session.check(
            "each link carries exactly one t= offset",
            not double_t,
            f"{len(hrefs)} links each with one t="
            if not double_t
            else f"{len(double_t)} links with a duplicated or missing t=",
        )
        session.check(
            "each link carries exactly one v= video id",
            not double_v,
            f"{len(hrefs)} links each with one v="
            if not double_v
            else f"{len(double_v)} links with a duplicated or missing v=",
        )
        session.check(
            "every link points at YouTube, not at a relative path",
            all(h.startswith("https://www.youtube.com/watch?") for h in hrefs),
            f"{sum(1 for h in hrefs if h.startswith('https://www.youtube.com/watch?'))}"
            f"/{len(hrefs)} links resolve to a youtube watch URL",
        )
        # The clock the reader reads and the second they land on must be the
        # same one; a link that opens 90 seconds away from the quote fails the
        # only promise the card makes.
        drift: list[str] = []
        for i, href in enumerate(hrefs):
            label = " ".join(links.nth(i).inner_text().split())
            shown = seconds_from_clock(label)
            offset = re.search(r"[?&]t=(\d+)", href)
            if shown is None or offset is None or abs(int(offset.group(1)) - shown) > 1:
                drift.append(f"{label[:50]!r} vs t={offset.group(1) if offset else '?'}")
        session.check(
            "the offset in every link matches the clock printed beside it",
            not drift,
            f"{len(hrefs)} timestamps agree with their links"
            if not drift
            else f"{len(drift)} disagree: {drift[:3]}",
        )
        ratios = page.locator(".dis-ratio")
        ratio_text = [" ".join(ratios.nth(i).inner_text().split()) for i in range(ratios.count())]
        session.check(
            "each quote states how much of it was found in the stored transcript",
            ratios.count() == 2 * total
            and all(re.search(r"quote \d+% in transcript", t) for t in ratio_text),
            f"{ratios.count()} provenance chips, e.g. {ratio_text[:1]}",
            shot=True,
        )

        # ── nothing a reader has to read is clipped, at four widths ───────────
        # A quote truncated with an ellipsis is worse here than anywhere else in
        # this app: the entire claim of the card is that these are the speaker's
        # words and the reader can check them.
        # The ``minimum`` guards below exist to stop a selector that has quietly
        # stopped matching from passing vacuously — not to re-assert the
        # conflict gate, which is already checked once above and fails loudly on
        # its own when it fails. Deriving them from the cards actually on screen
        # keeps the anti-vacuity protection exactly (a renamed ``.dis-quote``
        # still trips it) while keeping a low count from printing the same
        # failure twenty-four more times in different words.
        per_card = max(1, total)
        per_side = max(1, 2 * total)
        for width in (1440, 1100, 860, 700):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(350)
            check_nothing_clipped(
                session, f"every axis renders in full at {width}px", ".dis-axis", minimum=per_card
            )
            check_nothing_clipped(
                session, f"every quote renders in full at {width}px", ".dis-quote", minimum=per_side
            )
            check_nothing_clipped(
                session,
                f"every position renders in full at {width}px",
                ".dis-position",
                minimum=per_side,
            )
            check_nothing_clipped(
                session,
                f"every incompatibility note renders in full at {width}px",
                ".dis-why p",
                minimum=per_card,
            )
            check_nothing_clipped(
                session,
                f"every creator name renders in full at {width}px",
                ".dis-creator",
                minimum=per_side,
            )
            check_nothing_clipped(
                session,
                f"every timestamp line renders in full at {width}px",
                ".dis-ts",
                minimum=per_side,
            )
            check_nothing_clipped(
                session,
                f"every stat chip renders in full at {width}px",
                ".dis-statline .th-tag",
                minimum=4,
            )
            # The longest unbounded prose in the layer, and the piece carrying
            # the reader's only warning that the count is approximate. A build
            # that records its own spread renders none, so the guard is 0 rather
            # than 1: this asserts "if it is here, it is readable", which is the
            # only assertion available for an element that correctly disappears.
            check_nothing_clipped(
                session,
                f"the missing-spread note renders in full at {width}px",
                ".dis-caveat",
                minimum=0,
            )
            page_box = page.evaluate(
                "() => ({doc: document.documentElement.scrollWidth, win: window.innerWidth})"
            )
            session.check(
                f"the layer does not push the page sideways at {width}px",
                page_box["doc"] <= page_box["win"] + 1,
                f"document {page_box['doc']}px in a {page_box['win']}px viewport",
            )
        page.set_viewport_size({"width": 1440, "height": 900})

        session.note(
            "Shape and presentation only. This script cannot check that the two "
            "channels on a card actually disagree. The asymmetry it was first "
            "written against — corpus pairs adjudicated once, calibration probes "
            "repeated — is gone: both now run at the same repeat count and the "
            "tally is on every card. What replaced it is a smaller count against "
            "an unchanged gate, and a spread the shipped artifact cannot report."
        )
        session.note(
            "The 'no winner language' check reads only .dis-axis, .dis-position "
            "and .dis-why — the card's own voice. Quotes, creator names and video "
            "titles are excluded because they are the corpus's words: this corpus "
            "contains 'Write a Better Resume' and 'Why Hexagonal, Onion and Clean "
            "architecture are answers to the wrong question', and a scan of whole "
            "cards flags both as editorial."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
