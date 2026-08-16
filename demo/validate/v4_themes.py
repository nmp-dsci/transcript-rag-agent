"""Walk the V4 theme layer the way a sceptic would, and judge what the page says.

    uv run python -m src.cli serve --port 8021                # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v4_themes

V4 claims a level the per-video summaries cannot reach: a theme that several
creators state, not one video's outline renamed. That claim is only settled in
the browser if a reader can (a) find the layer at all, (b) see for each theme how
many videos and creators it draws on, and (c) open one and read the actual
transcript at the second it was said, in more than one video.

So this script asserts, on rendered page text only:

* **RAG Pipeline → Themes** exists as a sub-tab and opens a theme list;
* every row carries a **count a reader can audit** — videos, creators, chunks —
  and a single-video theme is *labelled* rather than silently mixed in, because
  that is the case where the layer added nothing;
* clicking a cross-video theme opens **member chunks grouped by video**, with at
  least two distinct video groups and two distinct creator names — the whole
  hypothesis in one screen;
* clicking a chunk reveals **transcript text and a timestamp link** whose href
  carries the video id of the group it sits under and a ``t=<seconds>s`` offset
  matching the clock the row displays. A link to the wrong video renders
  identically and proves nothing, so the id is compared, not just the shape;
* a **single-video theme** still opens and still says so;
* nothing is **clipped**, at four true viewport widths either side of the layout
  breakpoint. Theme titles and one-line summaries are LLM output of unbounded
  length and this app has shipped ``nowrap`` truncation on such a field before;
  jsdom has no layout, so only a real browser settles it.

WIDTH TECHNIQUE — READ THIS BEFORE CHANGING THE SWEEP. ``styles.ts:145`` is
``@media (max-width: 900px) { .th-layout { grid-template-columns: 1fr } }``: a
**viewport** query. There are no container queries and no ``container-type``
anywhere in that stylesheet, so a sample taken by constraining a container —
setting ``.app``'s width, which an earlier evaluation of this slice did — cannot
fire it. Such a sample measures wrapping inside a narrow box while the layout is
still two columns, and says nothing about the collapsed layout however narrow the
number in the note looks. The 2026-08-10 verdict's "820px" sample was invalid for
exactly this reason and has been withdrawn.

``resize_window`` is no better on this host: it reports success while leaving
``window.innerWidth`` unchanged. What works is Playwright's
``page.set_viewport_size`` (below) or, when the launcher is wedged, a
**same-origin iframe** sized to the target CSS pixel width with the view clicked
open inside it, so ``innerWidth`` really is the target. Either way,
``check_breakpoint_fired`` is the guard: it asserts the grid actually resolved to
two tracks above 900px and one below, so a technique that cannot move the
viewport fails the sweep instead of quietly passing it.

Nothing here reads ``/api/themes`` to prove a UI claim. That thirty themes exist
in a JSON file and that a reader can walk one to a timestamp are different
claims, and this repo has shipped the first without the second.

ENVIRONMENT NOTE — this machine's Metal compiler XPC service is wedged and
``chromium.launch()`` times out, so this file could not be executed through
Playwright at evaluation time. Every assertion below was executed instead as the
equivalent DOM query through an already-open Chrome at the same URL and viewport,
and the results are recorded in ``artifacts/v4_themes/verdict.json`` with this
same limitation noted there. Previous slices (v2, v3) recorded the same
constraint the same way.
"""

from __future__ import annotations

import re
import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v4_themes"

#: The sub-tab label a reader clicks. Text, not a test id: a label that changed
#: to something unfindable must fail this script.
SUBTAB = "Themes"

#: A theme that is the hypothesis working. Chosen for being the widest in the
#: layer *and* for not being dominated by one channel, so opening it is a real
#: test of "several creators" rather than of "one lecture with visitors".
CROSS_THEME_TITLE = "Tailor resumes for ATS with quantified impact"

#: A theme that is the hypothesis failing, kept visible on purpose. It has to
#: still be reachable and still be labelled.
SINGLE_THEME_TITLE = "Coupling is not bad"

#: The chip wording that tells a reader a theme adds nothing over level 1.
SINGLE_CHIP = "1 video only"

#: ``https://www.youtube.com/watch?v=<id>&t=<n>s`` — the id must be the group's.
WATCH = re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})(?:.*?)[?&]t=(\d+)s")

#: mm:ss as the chunk header renders it.
CLOCK = re.compile(r"(\d+):(\d{2})")

#: True CSS pixel viewport widths, two either side of ``BREAKPOINT_PX``. 860 and
#: 700 are the ones that matter: they are the only samples in this file that
#: exercise the collapsed one-column layout at all.
WIDTHS = (1440, 1100, 860, 700)

#: ``styles.ts:145`` — ``@media (max-width: 900px)`` on ``.th-layout``.
BREAKPOINT_PX = 900


# ── measuring what the page actually laid out ────────────────────────────────
def overflow(locator) -> dict[str, int]:
    """The box a node was drawn in versus the content inside it."""
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
    pass vacuously, which is how "nothing is broken" assertions rot.
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
            )
    session.check(
        name,
        not worst,
        f"{count} nodes laid out in full" if not worst else f"{len(worst)} clipped: {worst[:3]}",
    )


def layout_columns(session: UserSession) -> int:
    """How many grid tracks ``.th-layout`` resolved to, right now.

    Read off the computed style rather than off the declared CSS, because the
    whole question is whether the media query *fired* — which is a property of
    the viewport the page is being laid out in, not of the stylesheet.
    """
    return session.page.locator(".th-layout").evaluate(
        "node => getComputedStyle(node).gridTemplateColumns.trim().split(/\\s+/).length"
    )


def check_breakpoint_fired(session: UserSession, observed: dict[int, int]) -> None:
    """The sweep moved the viewport, not just a box inside it.

    This is the check that makes the rest of the sweep mean anything. Constrain
    ``.app`` to 820px instead of resizing the viewport and every clipping probe
    below still passes — while this one fails, because the grid is still two
    tracks wide. Any future driver that cannot really change ``innerWidth`` fails
    here rather than reporting a collapsed-layout result it never measured.
    """
    expected = {w: (1 if w <= BREAKPOINT_PX else 2) for w in observed}
    wrong = {w: (observed[w], expected[w]) for w in observed if observed[w] != expected[w]}
    below = [w for w in observed if w <= BREAKPOINT_PX]
    session.check(
        f"the {BREAKPOINT_PX}px breakpoint really fires: .th-layout collapses to one column",
        not wrong and bool(below),
        f"columns by viewport width {observed}"
        if not wrong
        else f"{wrong} — (observed, expected); the viewport did not actually change",
    )


def check_panels_do_not_overlap(session: UserSession, width: int) -> None:
    """The list and the detail sit beside each other or above each other.

    Collapsing a two-column grid is where panels land on top of one another, and
    text drawn under other text is unreadable without being ``scrollWidth``-
    clipped, so the probe above cannot see it.
    """
    boxes = session.page.evaluate(
        """() => {
            const pick = sel => {
                const node = document.querySelector(sel);
                if (!node) return null;
                const r = node.getBoundingClientRect();
                return {x: r.x, y: r.y, right: r.right, bottom: r.bottom};
            };
            return {list: pick('.th-list'), detail: pick('.th-detail')};
        }"""
    )
    first, second = boxes["list"], boxes["detail"]
    if not first or not second:
        session.check(
            f"the theme list and the theme detail do not overlap at {width}px",
            False,
            f"list={first is not None}, detail={second is not None}",
        )
        return
    overlapping = not (
        first["right"] <= second["x"] + 1
        or second["right"] <= first["x"] + 1
        or first["bottom"] <= second["y"] + 1
        or second["bottom"] <= first["y"] + 1
    )
    session.check(
        f"the theme list and the theme detail do not overlap at {width}px",
        not overlapping,
        f"list {round(first['x'])},{round(first['y'])}-{round(first['right'])},"
        f"{round(first['bottom'])} vs detail {round(second['x'])},{round(second['y'])}-"
        f"{round(second['right'])},{round(second['bottom'])}",
    )


def seconds(clock: str) -> int | None:
    match = CLOCK.search(clock)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def open_theme(session: UserSession, title_fragment: str) -> bool:
    """Click the theme row whose title contains ``title_fragment``."""
    row = session.page.locator("button.th-row").filter(has_text=title_fragment)
    if row.count() == 0:
        session.check(f"a theme row reads {title_fragment!r}", False, "no matching row")
        return False
    row.first.scroll_into_view_if_needed()
    row.first.click()
    session.page.wait_for_timeout(900)
    return True


def main() -> int:  # noqa: PLR0915 - one linear click path, read top to bottom
    require_server()
    with UserSession(SLICE) as session:
        page = session.page

        # ── the layer is reachable at all ─────────────────────────────────────
        session.tab("RAG Pipeline")
        subtab = page.get_by_role("button", name=SUBTAB, exact=True)
        session.check(
            "RAG Pipeline offers a Themes sub-tab",
            subtab.count() >= 1,
            f"{subtab.count()} buttons labelled {SUBTAB!r}",
            shot=True,
        )
        if subtab.count() == 0:
            return exit_code(session)
        subtab.first.click()
        page.wait_for_timeout(1200)

        rows = page.locator("button.th-row")
        session.check(
            "the Themes tab lists themes",
            rows.count() >= 5,
            f"{rows.count()} theme rows rendered",
            shot=True,
        )
        if rows.count() == 0:
            return exit_code(session)

        # ── every row is auditable: counts, not adjectives ────────────────────
        row_texts = [" ".join(rows.nth(i).inner_text().split()) for i in range(rows.count())]
        chunky = [t for t in row_texts if re.search(r"\d+ chunks", t)]
        session.check(
            "every theme row states its chunk count",
            len(chunky) == len(row_texts),
            f"{len(chunky)}/{len(row_texts)} rows show a chunk count",
        )
        cross_rows = [t for t in row_texts if re.search(r"\d+ videos . \d+ creators", t)]
        single_rows = [t for t in row_texts if SINGLE_CHIP in t]
        session.check(
            "cross-video rows state videos AND creators",
            len(cross_rows) >= 5,
            f"{len(cross_rows)} rows carry a 'N videos - N creators' chip",
        )
        session.check(
            "single-video themes are labelled, not hidden",
            len(single_rows) >= 1 and len(cross_rows) + len(single_rows) == len(row_texts),
            f"{len(single_rows)} labelled '{SINGLE_CHIP}', "
            f"{len(cross_rows)} cross-video, {len(row_texts)} rows total",
        )

        # The headline claim of the tab, stated on the page rather than in a file.
        head = " ".join(page.locator(".th-listhead").inner_text().split())
        session.check(
            "the list header states how many themes span 2+ videos",
            bool(re.search(r"\d+ of \d+ span 2\+ videos", head)),
            head[:160],
        )

        # ── the hypothesis, opened ────────────────────────────────────────────
        if not open_theme(session, CROSS_THEME_TITLE):
            return exit_code(session)

        detail = page.locator(".th-detail")
        session.check_visible(
            "the opened theme shows a one-line summary",
            "creators agree",
            shot=True,
        )
        groups = detail.locator(".th-group")
        session.check(
            "members are grouped by the video they came from",
            groups.count() >= 2,
            f"{groups.count()} video groups in this theme",
        )
        names = [
            " ".join(groups.nth(i).locator(".th-groupname").inner_text().split())
            for i in range(groups.count())
        ]
        creators = [
            " ".join(groups.nth(i).locator(".th-groupmeta").inner_text().split()).split(" · ")[0]
            for i in range(groups.count())
        ]
        session.check(
            "the groups are distinct videos, not the same video twice",
            len(set(names)) == len(names) and len(set(names)) >= 2,
            f"{len(set(names))} distinct video titles: {names[:3]}",
        )
        session.check(
            "the groups name at least two different creators",
            len(set(creators)) >= 2,
            f"{len(set(creators))} distinct creators: {sorted(set(creators))[:4]}",
        )
        session.check(
            "each group states how many chunks it contributed",
            all(
                re.search(r"\d+ chunk", c)
                for c in [
                    " ".join(groups.nth(i).locator(".th-groupmeta").inner_text().split())
                    for i in range(groups.count())
                ]
            ),
            "per-group chunk counts present",
        )

        # ── a chunk opens to transcript text and a timestamp that lands ───────
        # One chunk per group, not one chunk overall. The claim under test is
        # that a link lands in *the video whose group it sits under*, and that is
        # only visible across groups: a single group cannot distinguish "the
        # right video" from "some video".
        walked: list[dict[str, object]] = []
        for index in range(groups.count()):
            group = groups.nth(index)
            head = group.locator("button.th-chunkhead").first
            clock_text = " ".join(head.locator(".th-chunktime").inner_text().split())
            head.scroll_into_view_if_needed()
            head.click()
            page.wait_for_timeout(400)
            body = group.locator(".th-chunkbody").first
            transcript = " ".join(body.locator("p").inner_text().split()) if body.count() else ""
            link = body.locator("a") if body.count() else None
            href = link.first.get_attribute("href") if link is not None and link.count() else ""
            walked.append(
                {
                    "video": names[index],
                    "clock": clock_text,
                    "chars": len(transcript),
                    "match": WATCH.search(href or ""),
                    "href": href or "",
                }
            )
            head.click()  # the panel is an accordion; leave it as it was found
            page.wait_for_timeout(150)

        thin = [f"{w['video'][:34]!r} {w['chars']} chars" for w in walked if int(w["chars"]) <= 120]
        session.check(
            "clicking a chunk reveals its transcript text",
            bool(walked) and not thin,
            f"{len(walked)} chunks opened, "
            f"{min(int(w['chars']) for w in walked) if walked else 0}-"
            f"{max(int(w['chars']) for w in walked) if walked else 0} chars each"
            if not thin
            else f"{len(thin)} too short: {thin[:3]}",
            shot=True,
        )
        linkless = [w["video"] for w in walked if w["match"] is None]
        session.check(
            "the chunk carries a timestamp link into the source video",
            bool(walked) and not linkless,
            f"{len(walked)}/{len(walked)} chunks render a /watch?v=…&t=<n>s link"
            if not linkless
            else f"{len(linkless)} without one: {linkless[:3]}",
        )
        # The link must point at *this group's* video: a link to any other video
        # in the corpus renders identically and proves nothing. Without reading
        # the API the id cannot be checked against the group's title, so what is
        # asserted is the page's internal consistency — one id per group, and a
        # different id for every group. A group whose link had been crossed with
        # another group's video collapses the distinct count below the group
        # count and fails here. (This cannot catch every group being wrong in the
        # same way; nothing available from the DOM alone can.)
        ids = [m.group(1) for m in (w["match"] for w in walked) if m is not None]
        session.check(
            "the link's video id belongs to the group it sits under",
            len(ids) == groups.count() and len(set(ids)) == groups.count(),
            f"{len(set(ids))} distinct video ids over {groups.count()} groups: {ids[:4]}",
        )
        drift = [
            f"{w['video'][:28]!r} row {w['clock']} vs t={w['match'].group(2)}s"
            for w in walked
            if w["match"] is None
            or seconds(str(w["clock"])) is None
            or abs(int(w["match"].group(2)) - int(seconds(str(w["clock"])) or -9999)) > 1
        ]
        session.check(
            "the link's offset matches the clock the row displays",
            bool(walked) and not drift,
            f"{len(walked)} offsets agree with the clock beside them"
            if not drift
            else f"{len(drift)} disagree: {drift[:3]}",
        )

        # ── the failure case is still visible and still says so ───────────────
        if open_theme(session, SINGLE_THEME_TITLE):
            single_text = " ".join(detail.inner_text().split())
            session.check(
                "a single-video theme opens and admits it adds nothing",
                "came from one video" in single_text,
                single_text[:200],
                shot=True,
            )
            session.check(
                "the single-video theme still lists its member chunks",
                detail.locator(".th-group").count() == 1
                and detail.locator("button.th-chunkhead").count() >= 5,
                f"{detail.locator('.th-group').count()} group, "
                f"{detail.locator('button.th-chunkhead').count()} chunks",
            )

        # ── nothing the reader has to read is clipped, at four true widths ────
        # Read the WIDTH TECHNIQUE paragraph in the module docstring before
        # touching this loop. ``set_viewport_size`` is load-bearing: it is what
        # makes ``@media (max-width: 900px)`` evaluate, and the breakpoint guard
        # after the loop fails if whatever driver runs this could not move the
        # viewport for real.
        open_theme(session, CROSS_THEME_TITLE)
        page.locator(".th-detail button.th-chunkhead").first.click()
        page.wait_for_timeout(400)

        columns_seen: dict[int, int] = {}
        for width in WIDTHS:
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(400)
            columns_seen[width] = layout_columns(session)
            check_nothing_clipped(
                session,
                f"every theme row title renders in full at {width}px",
                ".th-row-title",
                minimum=20,
            )
            check_nothing_clipped(
                session,
                f"every row chip renders in full at {width}px",
                ".th-row .th-tag",
                minimum=40,
            )
            check_nothing_clipped(
                session, f"the theme title renders in full at {width}px", ".th-title", minimum=1
            )
            check_nothing_clipped(
                session, f"the theme summary renders in full at {width}px", ".th-summary", minimum=1
            )
            check_nothing_clipped(
                session,
                f"every video group heading renders in full at {width}px",
                ".th-groupname",
                minimum=5,
            )
            check_nothing_clipped(
                session,
                f"the expanded chunk text renders in full at {width}px",
                ".th-chunkbody p",
                minimum=1,
            )
            check_panels_do_not_overlap(session, width)
            page_box = page.evaluate(
                "() => ({doc: document.documentElement.scrollWidth, win: window.innerWidth})"
            )
            session.check(
                f"the Themes tab does not push the page sideways at {width}px",
                page_box["doc"] <= page_box["win"] + 1,
                f"document {page_box['doc']}px in a {page_box['win']}px viewport, "
                f"{columns_seen[width]} column(s)",
                shot=width <= BREAKPOINT_PX,
            )
        check_breakpoint_fired(session, columns_seen)
        page.set_viewport_size({"width": 1440, "height": 900})

        session.note(
            "'.th-chunkpreview' is deliberately white-space: nowrap and is excluded "
            "from the clipping probe: it is a fixed-length collapsed excerpt whose "
            "full text is one click away in .th-chunkbody, not a truncated label. It "
            "is the only nowrap node in the whole theme layer, which is what the "
            "comment at styles.ts:133-138 claims and what makes the exclusion narrow."
        )
        session.note(
            "The width sweep is a VIEWPORT sweep. Constraining a container instead "
            "(setting .app's width, as the 2026-08-10 run of this script did) cannot "
            "fire styles.ts:145's `@media (max-width: 900px)`, and there are no "
            "container queries in that stylesheet, so such a sample never exercises "
            "the collapsed layout at all. check_breakpoint_fired() is the guard "
            "against that: it fails when the grid is still two tracks wide below "
            "900px, which is exactly what a container-constrained run produces."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
