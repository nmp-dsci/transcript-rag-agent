"""Replay the V1 document-review path in the browser and judge what it shows.

    uv run python -m src.cli serve --port 8021              # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v1_document

V1 claims the app can *review your artifact against the corpus* — not merely
answer questions about it. That claim is only settled by what a reviewer sees
after clicking: a document card that states how much of the page the answer
read, an answer that names the document's own sections, and, in the same
answer, corpus timestamps that resolve to real videos. Feedback that could have
been written without opening the document would satisfy none of that.

So this script clicks the path a person clicks — Chat, the history rail, the
review conversation, the trace, the ``Sources (N)`` disclosure — and asserts on
rendered page text only. Nothing here calls ``/api/ask`` or ``/api/documents``:
an API that can produce a good review proves the server works, not that the
reviewer can see it, and this slice's whole risk lives in the gap between the
two.

Two reviews of one page are walked, not one. The first states its criteria in
the question; the second is ``review this: <url>``, the shortest thing a user
can type, which leaves the app to work out for itself what the page should be
judged against. That second path is where a document reviewer stops being a
demo, so a run that only exercised the first would be reporting on the easy
half. Walking both also puts the shared-document card under the only pressure
that ever broke it: two entries, one cached page, each needing its own
selection line.

The document-specific assertions are deliberately phrases that exist *only on
the reviewed page* ("deterministic policy gate", "AWS App Runner"). A review
that scored well on generic resume advice would still fail them, which is the
point: they are the cheapest available proxy for "this was written by something
that read the document".
"""

from __future__ import annotations

import re
import sys
from urllib.parse import parse_qs, urlparse

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v1_document"

#: The two committed reviews, searched for by their question text — that is how
#: a person finds them, and matching on an entry id would pass even if the rail
#: rendered nothing a human could recognise.
WORDED = "Review my portfolio site"
BARE_URL = "review this: https://nmp-dsci.github.io/"

#: The document card's own claim about how much of the page the answer read.
#: A review of part of a document that renders as a review of the document is
#: the specific failure this line exists to prevent, so it is asserted verbatim.
SELECTION_LINE = "whole document — all 9 sections in context"

#: Section markers the answer must render as citations. §2 carries the hero
#: claims, §5 a project the review praises, §9 the "Earlier work" group.
SECTION_MARKERS = ("§2", "§5", "§9")

#: Phrases that appear on the reviewed page and nowhere in generic advice about
#: resumes. Both reviews are held to the same list, because the claim under
#: test is about the feature, not about one lucky answer.
DOCUMENT_PHRASES = (
    "deterministic policy gate",
    "AWS App Runner",
    "Case study",
)

#: Words the *criteria* query must contain when the user supplied none. The
#: failure this catches is the one that made the bare-URL path worthless: a
#: query built from the document's own subject matter, which retrieves
#: transcripts about building the projects rather than about presenting them.
CRITERIA_WORDS = ("recruiters", "hiring managers")

#: A source row reads "[1] open at 5:46 ozwmlFencJI"; this pulls the parts back
#: out of the rendered text so they can be checked against the link.
SOURCE_ROW = re.compile(r"open at (\d+:\d{2}(?::\d{2})?)")


def watch_link(href: str) -> tuple[str, str] | None:
    """The (video id, start seconds) a citation's link opens, or None.

    Parsed rather than pattern-matched on the whole string: a source url keeps
    whatever query parameters it was indexed with (YouTube's ``pp=`` tracking
    blob, for one), and a citation that carries them is still a citation that
    opens the right video at the right second. Insisting on an exact url shape
    would fail those for a reason no reader would care about.
    """
    parsed = urlparse(href)
    if parsed.netloc not in ("www.youtube.com", "youtube.com") or parsed.path != "/watch":
        return None
    params = parse_qs(parsed.query)
    video = (params.get("v") or [""])[0]
    offset = (params.get("t") or [""])[0]
    if not video or not re.fullmatch(r"\d+s?", offset):
        return None
    return video, offset.rstrip("s")


def open_review_conversation(session: UserSession, question: str) -> bool:
    """Click a review thread in the history rail the way a person would."""
    rail = session.page.locator(".rail")
    entry = rail.get_by_text(question, exact=False).first
    if entry.count() == 0:
        return False
    entry.scroll_into_view_if_needed()
    entry.click()
    session.page.wait_for_timeout(1500)
    return True


def trace_query(session: UserSession) -> tuple[str, bool]:
    """The query the open answer says it searched the corpus with, and whether
    the page clipped it.

    Read from the expanded trace rather than from the API, because the claim is
    that a *reader* can see it. The clipping check is the point of the second
    return value: the retrieval step's own detail line is ``white-space: nowrap``
    with an ellipsis, and a query rendered the same way would be no more
    checkable there than it was when it lived inside that line. So the element
    is asked whether its content overflows the box it is drawn in.
    """
    summary = session.page.get_by_text("trace — ", exact=False).first
    if summary.count() == 0:
        return "", True
    summary.click()
    session.page.wait_for_timeout(400)
    line = session.page.locator(".trace-query")
    if line.count() == 0:
        return "", True
    text = line.first.locator("span").last
    clipped = bool(
        text.evaluate("node => node.scrollWidth > node.clientWidth + 1")
        or text.evaluate("node => node.scrollHeight > node.clientHeight + 1")
    )
    return text.inner_text().strip(), clipped


def review_checks(session: UserSession, question: str, tag: str) -> dict[str, object]:
    """Open one review conversation and assert everything visible about it.

    Returns the facts the *cross-conversation* checks need, because the most
    interesting failures at this point are not within one review but between
    two: a selection line frozen from whichever conversation was opened first,
    or two different questions that turn out to have searched the corpus with
    the same string.
    """
    opened = open_review_conversation(session, question)
    session.check(f"{tag}: conversation opens on click", opened, f"clicked: {opened}")
    if not opened:
        return {}

    # -- the document card, and what it says it read --------------------------
    card = session.page.locator(".doccard")
    session.check(
        f"{tag}: a document card renders above the answer",
        card.count() == 1,
        f"{card.count()} document card(s)",
        shot=True,
    )
    card_text = card.first.inner_text() if card.count() else ""
    session.check(
        f"{tag}: card states which sections were in context",
        SELECTION_LINE in card_text,
        f"selection line {'present' if SELECTION_LINE in card_text else 'missing'}: "
        f"{SELECTION_LINE!r}",
    )
    session.check(
        f"{tag}: card names the reviewed page",
        "Work · Nathan Phillips" in card_text and "nmp-dsci.github.io" in card_text,
        f"card head: {card_text.splitlines()[:3]}",
    )
    # The card's section numbering is what a [§N] citation points at, so the
    # two have to agree or the citation is unfollowable.
    numbered = session.page.locator(".docsec-num")
    session.check(
        f"{tag}: card numbers every section §1..§9",
        numbered.count() == 9 and numbered.first.inner_text().strip() == "§1",
        f"{numbered.count()} numbered sections, first={numbered.first.inner_text().strip()!r}",
    )

    # -- the query the corpus was actually searched with ----------------------
    query, clipped = trace_query(session)
    session.check(
        f"{tag}: the trace shows the query the corpus was searched with",
        len(query) > 60,
        f"{len(query)} characters of query rendered: {query[:110]!r}",
        shot=True,
    )
    session.check(
        f"{tag}: the query wraps rather than being clipped to one line",
        not clipped,
        "the element's content fits the box it is drawn in"
        if not clipped
        else "content overflows its box — the query is visually cut off",
    )
    session.check(
        f"{tag}: no URL was embedded in the corpus query",
        "http" not in query and "nmp-dsci" not in query,
        f"url tokens in query: {'http' in query or 'nmp-dsci' in query}",
    )

    # -- the answer names document sections -----------------------------------
    body = session.page.locator(".body").first.inner_text()
    missing_markers = [marker for marker in SECTION_MARKERS if marker not in body]
    session.check(
        f"{tag}: answer cites document sections by number",
        not missing_markers,
        f"missing markers: {missing_markers or 'none'}",
        shot=True,
    )
    rendered_markers = session.page.locator(".body .cite-doc")
    session.check(
        f"{tag}: section citations render as chips, not raw text",
        rendered_markers.count() >= 3 and "[§" not in body,
        f"{rendered_markers.count()} §-chips; unrendered '[§' in text: {'[§' in body}",
    )

    # -- the answer is about *this* document ----------------------------------
    missing_phrases = [phrase for phrase in DOCUMENT_PHRASES if phrase not in body]
    session.check(
        f"{tag}: answer quotes phrases that exist only on the reviewed page",
        not missing_phrases,
        f"missing: {missing_phrases or 'none'}",
    )

    # -- and cites the corpus in the same answer ------------------------------
    summary = session.page.get_by_text("Sources (", exact=False).first
    session.check(f"{tag}: answer offers a Sources disclosure", summary.count() > 0, "")
    if summary.count() == 0:
        return {"selection": card_text, "query": query}
    summary_label = summary.inner_text()
    summary.click()
    session.page.wait_for_timeout(400)

    rows = session.page.locator("details.refs li")
    session.check(
        f"{tag}: Sources expands to one row per citation",
        rows.count() >= 5,
        f"{summary_label!r} -> {rows.count()} rows",
        shot=True,
    )

    timestamps: list[str] = []
    videos: set[str] = set()
    bad_links: list[str] = []
    for index in range(rows.count()):
        row = rows.nth(index)
        row_text = row.inner_text()
        match = SOURCE_ROW.search(row_text)
        if not match:
            bad_links.append(f"row {index}: no timestamp in {row_text!r}")
            continue
        timestamps.append(match.group(1))
        href = row.locator("a").first.get_attribute("href") or ""
        link = watch_link(href)
        if link is None:
            bad_links.append(f"row {index}: {href!r} is not a timestamped video link")
            continue
        video, _ = link
        videos.add(video)
        # The id shown beside the link and the id the link opens must be the
        # same video, or the row cites one thing and links another.
        if video not in row_text:
            bad_links.append(f"row {index}: link video {video} not in {row_text!r}")

    session.check(
        f"{tag}: every corpus citation carries a timestamp",
        len(timestamps) == rows.count() and rows.count() > 0,
        f"{len(timestamps)} of {rows.count()} rows show 'open at <time>' — e.g. {timestamps[:3]}",
    )
    session.check(
        f"{tag}: every citation resolves to a video at that moment",
        not bad_links,
        "; ".join(bad_links) if bad_links else f"{rows.count()} rows link to watch?v=…&t=…s",
    )
    session.check(
        f"{tag}: citations span more than one video",
        len(videos) > 1,
        f"{len(videos)} distinct videos cited: {sorted(videos)}",
    )

    # -- the hypothesis, stated as one check ----------------------------------
    session.check(
        f"{tag}: one answer carries both document sections and corpus timestamps",
        not missing_markers and not bad_links and rows.count() > 0,
        f"{len(SECTION_MARKERS)} section markers checked, {rows.count()} corpus citations",
        shot=True,
    )
    return {"selection": card_text, "query": query, "videos": videos}


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        session.tab("Chat")

        rail_text = session.page.locator(".rail").inner_text()
        for label, question in (("worded", WORDED), ("bare URL", BARE_URL)):
            session.check(
                f"history rail lists the {label} review",
                question in rail_text,
                f"rail entries containing {question!r}: {rail_text.count(question)}",
                shot=label == "worded",
            )

        worded = review_checks(session, WORDED, "worded")
        # Opening the second review with the first still cached is the exact
        # condition under which the card used to show the first one's selection.
        bare = review_checks(session, BARE_URL, "bare URL")
        if not worded or not bare:
            return exit_code(session)

        # -- what the two reviews prove only together -------------------------
        session.check(
            "both reviews of the same page get their own card",
            SELECTION_LINE in str(worded["selection"]) and SELECTION_LINE in str(bare["selection"]),
            "each card states its own selection with the page cached from the other",
        )
        session.check(
            "a bare URL is not searched with the words the user typed",
            str(bare["query"]) != str(worded["query"]),
            f"worded and bare-URL queries differ: {str(bare['query'])[:90]!r}",
        )
        missing_criteria = [
            word for word in CRITERIA_WORDS if word not in str(bare["query"]).lower()
        ]
        session.check(
            "a bare URL searches for review criteria, not for the page's subject",
            not missing_criteria,
            f"criteria words missing from the query: {missing_criteria or 'none'}",
            shot=True,
        )

        session.note(
            "Section markers and corpus citations are visually distinct: §N chips carry no "
            "link because the section is on the page above; [n] chips link out to the video."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
