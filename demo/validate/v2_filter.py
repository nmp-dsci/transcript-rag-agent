"""Replay the V2 summary-filter path in the browser and judge what it shows.

    uv run python -m src.cli serve --port 8021              # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v2_filter

V2 claims that routing a question to whole videos by their per-video summary
*before* any chunk is searched removes off-topic material without costing
accuracy. Two halves of that claim are visible in the app and nowhere else:

* In **Chat**, the answer's trace must show the routing decision — not "5 videos
  matched", which is a count and settles nothing, but *which* five. A filter
  that kept five career videos and no property ones is the entire hypothesis,
  so the list is the evidence and a list clipped to its first two entries is no
  evidence at all. This script therefore measures the rendered element: it asks
  the browser whether the note's content overflows the box it is drawn in, the
  same probe the V1 validator used on the retrieval query after that line had
  shipped ``white-space: nowrap``.
* On the **Scoreboard**, the new setup has to be a column you can compare
  against ``rag_llm``. A retrieval variant that cannot be put beside the
  baseline in the surface that ranks setups has not been added to the lab; it
  has been added to one conversation.

Both are asserted on rendered page text. Nothing here reads ``/api/ask`` or
``/api/scoreboard``: this slice's whole risk lives in the gap between a server
that routes correctly and a reader who can see that it did.

The script also re-checks four fixes folded in from earlier slices, because
each one is a claim about what renders rather than about what is computed: the
273-character corpus-coverage caveat wrapping in full, the ``partial context``
badge on a *collapsed* Scoreboard row, the "not rescored" explanation on the
one cell that has no Expand button to hide behind, and the cap reason at the
0.5999 rounding edge that used to print as "faithfulness 0.60 below 0.6". The
last of those cannot be seen on any committed run — no cell sits at that edge —
so it is exercised against a fixture run served by a second, throwaway app
instance, exactly as ``v0_judge_negatives`` does for the two conditional
banners. A fixture is not a committed run and must not be written into
``evals/runs``.

**Provenance of the 2026-08-10 evaluation.** This script did not run to a
verdict on that machine: ``chromium.launch`` timed out on every installed
Playwright revision (1208/1223/1228/1234, headless and headed), each shell
wedging in an uninterruptible ``UEs`` state within 0.03s of exec — the same
fault that hung a bare ``import torch`` in any freshly spawned process. It is
an environment failure, not a slice failure, and it is recorded here rather
than worked around, because a validator that reports a verdict it did not
measure is worse than one that reports nothing. Every assertion below was
instead executed by hand against a live Chrome on the same two app instances,
and the measurements are in the evaluator's report. Re-run this script once the
host can launch a browser again; nothing about it is machine-specific.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v2_filter"

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"

#: The setup under test, by the label a reader sees rather than by its key.
FILTERED_TITLE = "rag_llm (summary-filtered)"
BASELINE_TITLE = "rag_llm (single-hop)"

#: The resume question the live run asks. Deliberately one of the golden
#: entries (g015) so its answer is comparable to the eval numbers, and
#: deliberately a *resume* question, because "resume answers stop containing
#: off-genre chunks" is how the hypothesis is worded.
RESUME_QUESTION = "How do I make my resume ATS-friendly?"

#: A committed conversation that already ran both arms of the same question.
#: Replayed as well as the live ask because it is the case the slice's own
#: evidence rests on — the interview question whose unfiltered answer pulls
#: chunks out of three system-design talks.
INTERVIEW_QUESTION = "How should I prepare for behavioural interview questions?"

#: The committed review whose trace carries the corpus-coverage caveat. It is
#: the only page in the history the classifier declines to name, which is the
#: only condition under which the warning is written at all.
COVERAGE_QUESTION = "review this: https://en.wikipedia.org/wiki/Sourdough"
COVERAGE_TEXT = (
    "This document does not match a kind the corpus has criteria for "
    "(resume, portfolio, professional profile, cover letter). The retrieved "
    "chunks are the closest the corpus holds, which may be advice about a "
    "different kind of document entirely — say so rather than applying it."
)

#: Video ids by content cluster, so "routed only to career videos" is checked
#: against ids on the page instead of against an impression of the titles.
PROPERTY_VIDEOS = {"gXf7fRvuaXA", "ZiEGOgTC56Y", "7m27Go3K1d0", "AdRL6tKu3Gk", "Bw58mV015z4"}
SYSTEM_DESIGN_VIDEOS = {
    "1NngTUYPdpI",
    "L521gizea4s",
    "iJLL-KPqBpM",
    "m4q7VkgDWrM",
    "5ZjhNTM8XU8",
    "wXvljefXyEo",
    "uFGJVQvR59A",
}
OFF_GENRE = PROPERTY_VIDEOS | SYSTEM_DESIGN_VIDEOS

#: The Scoreboard cells that carry each folded-in badge, found by the question
#: text a reader would filter on rather than by golden id.
PARTIAL_CONTEXT_QUESTION = "What are the main themes across this corpus of videos?"
NOT_RESCORED_QUESTION = "What is now the best property strategy for investors post budget?"

#: The fixture instance for the grounding-floor rounding check.
FIXTURE_RUN = "matrix-29990101-000003-floor"
FIXTURE_PORT = 8023

VIDEO_ID = re.compile(r"\(([0-9A-Za-z_-]{11})\)")


# ── measuring what the page actually laid out ────────────────────────────────
def overflow(locator) -> dict[str, int]:
    """The box a node was drawn in versus the content inside it.

    Returned as numbers rather than a boolean because "the list is clipped" and
    "the list happens to be short" produce the same boolean and very different
    evidence. A wrapping element has ``scrollHeight == clientHeight`` and more
    than one line of height; a ``nowrap`` one has ``scrollWidth > clientWidth``
    and exactly one.
    """
    return locator.evaluate(
        """node => ({
            scrollWidth: node.scrollWidth,
            clientWidth: node.clientWidth,
            scrollHeight: node.scrollHeight,
            clientHeight: node.clientHeight,
            lineHeight: Math.round(parseFloat(getComputedStyle(node).lineHeight) || 0),
            nowrap: getComputedStyle(node).whiteSpace === 'nowrap' ? 1 : 0,
        })"""
    )


def check_wraps_in_full(session: UserSession, name: str, locator, expected: str | None = None):
    """One element, asserted to show all of its text rather than a first line."""
    box = overflow(locator)
    rendered = locator.inner_text().strip()
    clipped = (
        box["scrollWidth"] > box["clientWidth"] + 1
        or box["scrollHeight"] > box["clientHeight"] + 1
        or bool(box["nowrap"])
    )
    detail = (
        f"{len(rendered)} chars laid out in {box['clientWidth']}x{box['clientHeight']}px "
        f"(content {box['scrollWidth']}x{box['scrollHeight']}px, "
        f"line-height {box['lineHeight']}px, white-space:"
        f"{'nowrap' if box['nowrap'] else 'normal'})"
    )
    session.check(name, not clipped, detail)
    if expected is not None:
        session.check(
            f"{name} — and the text is complete",
            expected in " ".join(rendered.split()),
            f"rendered {len(rendered)} of {len(expected)} expected characters",
        )
    return box, rendered


# ── the trace, as a reader opens it ──────────────────────────────────────────
def open_trace(session: UserSession) -> bool:
    trace = session.page.locator("details.trace").first
    if trace.count() == 0:
        return False
    trace.scroll_into_view_if_needed()
    trace.locator("summary").click()
    session.page.wait_for_timeout(400)
    return True


def summary_filter_note(session: UserSession):
    """The ``videos`` line under the Summary filter step, or None.

    Located by walking down from the step whose *label* reads "Summary filter",
    not by taking the first ``.trace-note`` on the page — a note under the wrong
    step would satisfy a looser selector while telling the reader nothing about
    routing.
    """
    steps = session.page.locator("details.trace .trace-line")
    for index in range(steps.count()):
        line = steps.nth(index)
        if "Summary filter" not in line.inner_text():
            continue
        block = line.locator("xpath=..")
        note = block.locator(".trace-note")
        return line, (note.first if note.count() else None)
    return None, None


def filter_trace_checks(session: UserSession, tag: str) -> set[str]:
    """Everything the routing step has to show, and the ids it named."""
    if not open_trace(session):
        session.check(f"{tag}: the answer offers a trace", False, "no details.trace on the page")
        return set()

    line, note = summary_filter_note(session)
    session.check(
        f"{tag}: the trace shows a Summary filter step",
        line is not None,
        "step labelled 'Summary filter' present" if line is not None else "no such step",
        shot=True,
    )
    if line is None:
        return set()

    phase = line.locator(".ph").first.inner_text().strip()
    session.check(
        f"{tag}: the routing step is marked as a filter phase",
        phase == "filter",
        f"phase chip reads {phase!r}",
    )
    session.check(
        f"{tag}: the step says how many videos it kept",
        "videos matched by per-video summary" in line.inner_text(),
        " ".join(line.inner_text().split())[:120],
    )

    session.check(
        f"{tag}: the matched videos are listed, not just counted",
        note is not None,
        "a wrapping note line follows the step" if note is not None else "no note line — "
        "the step reports a count and nothing a reader could check it against",
    )
    if note is None:
        return set()

    label = note.locator(".trace-query-label").inner_text().strip()
    session.check(
        f"{tag}: the list is labelled as the videos it routed to",
        label.lower() == "videos",
        f"label reads {label!r}",
    )

    text_span = note.locator("span").last
    box, rendered = check_wraps_in_full(
        session, f"{tag}: the matched-video list renders in full, not clipped", text_span
    )
    session.check(
        f"{tag}: the list wraps onto more than one line",
        box["lineHeight"] > 0 and box["clientHeight"] >= box["lineHeight"] * 2 - 1,
        f"{box['clientHeight']}px tall at a {box['lineHeight']}px line-height",
        shot=True,
    )

    ids = set(VIDEO_ID.findall(rendered))
    session.check(
        f"{tag}: every listed video carries its id and match score",
        len(ids) >= 2 and rendered.count("·") >= 1 and re.search(r"0\.\d\d", rendered) is not None,
        f"{len(ids)} video ids and {rendered.count('·') + 1} entries: {' '.join(rendered.split())[:180]}",
    )
    return ids


# ── the second app instance, for the one thing no committed run can show ─────
def free_port(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def serve_fixtures(port: int):
    """A throwaway app whose runs directory is ``fixtures/``.

    Its runner is built lazily and this instance is never asked a question, so
    it costs a process and no model load.
    """
    script = (
        "import uvicorn;"
        "from src.api.main import create_app;"
        "from src.config import load_settings;"
        "from pathlib import Path;"
        f"uvicorn.run(create_app(load_settings(), runs_dir=Path({str(FIXTURES)!r})),"
        f" host='127.0.0.1', port={port}, log_level='warning')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        if process.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return process
        except Exception:  # noqa: BLE001 - still starting
            time.sleep(1)
    process.terminate()
    return None


def floor_rounding_checks(session: UserSession) -> None:
    """The cap reason at the rounding edge, read off the page.

    No committed run has a cell whose faithfulness rounds to the floor, so this
    runs against a fixture built by pushing one cell to 0.5999 and recompositing
    it through the shipped rubric — the sentence on the page is the one the
    shipped code wrote, not one the fixture invented.
    """
    session.page.get_by_role("combobox", name="Matrix run").select_option(FIXTURE_RUN)
    session.page.wait_for_timeout(1200)
    session.click("Answers (", exact=False)
    session.page.get_by_role("combobox", name="Filter by question").select_option(
        label_containing(session, "Filter by question", "How do I set up Herder")
    )
    session.page.wait_for_timeout(600)
    expand = session.page.get_by_role("button", name="Expand").first
    if expand.count():
        expand.click()
        session.page.wait_for_timeout(400)
    page = session.text()
    session.check(
        "a capped cell states the faithfulness it was capped for",
        "below 0.6" in page,
        "cap reason rendered on the page",
        shot=True,
    )
    printed = re.findall(r"faithfulness (\d+\.\d+) below (\d+\.\d+)", page)
    contradictions = [pair for pair in printed if float(pair[0]) >= float(pair[1])]
    session.check(
        "no cap reason prints a number that is not below the floor it names",
        bool(printed) and not contradictions,
        f"reasons rendered: {printed}; self-contradicting: {contradictions or 'none'}",
    )
    session.check(
        "a 0.5999 faithfulness renders at the precision its claim needs",
        ("faithfulness 0.5999 below 0.6" in page) and ("faithfulness 0.60 below 0.6" not in page),
        "reads 'faithfulness 0.5999 below 0.6'"
        if "faithfulness 0.5999 below 0.6" in page
        else f"reads {printed}",
        shot=True,
    )


def label_containing(session: UserSession, combobox: str, needle: str) -> str:
    options = session.page.get_by_role("combobox", name=combobox).locator("option")
    for index in range(options.count()):
        text = options.nth(index).inner_text()
        if needle in text:
            return text
    return needle


# ── the walk ─────────────────────────────────────────────────────────────────
def live_resume_ask(session: UserSession) -> set[str]:
    """Ask a resume question as the filtered setup, right now, in the browser."""
    session.page.get_by_role("combobox", name="Answering agent").select_option(
        label=FILTERED_TITLE
    )
    session.page.wait_for_timeout(300)
    # Judging the answer is a different slice's claim and costs minutes, so the
    # run is asked for without it.
    session.click_button("⚙ advanced", exact=True)
    session.page.get_by_text("auto-judge with RAGAS").locator("input").uncheck()
    session.page.wait_for_timeout(200)

    # Counted rather than waited on by text: the thread may already hold a
    # trace from a conversation opened earlier, and "the text is on the page"
    # would then be true before this question was ever sent.
    before = session.page.locator("details.trace").count()
    session.page.get_by_role("textbox", name="Question").fill(RESUME_QUESTION)
    session.click_button("Send", exact=True)
    deadline = time.time() + 420
    arrived = False
    while time.time() < deadline:
        if session.page.locator("details.trace").count() > before:
            arrived = True
            break
        session.page.wait_for_timeout(2000)
    session.check(
        "a resume question answers under the summary-filtered setup",
        arrived,
        "the answer and its trace rendered" if arrived else "no answer within 7 minutes",
        shot=True,
    )
    if not arrived:
        return set()
    session.page.wait_for_timeout(1500)
    chip = session.page.locator(".setupchip").last.inner_text().strip()
    session.check(
        "the answer is labelled with the setup that produced it",
        chip == FILTERED_TITLE,
        f"setup chip reads {chip!r}",
    )
    return filter_trace_checks(session, "live resume ask")


def open_conversation(session: UserSession, question: str) -> bool:
    rail = session.page.locator(".rail")
    entry = rail.get_by_text(question, exact=False).first
    if entry.count() == 0:
        return False
    entry.scroll_into_view_if_needed()
    entry.click()
    session.page.wait_for_timeout(1500)
    return True


def committed_interview_conversation(session: UserSession) -> None:
    """The two arms of one question, as a reader compares them."""
    opened = open_conversation(session, INTERVIEW_QUESTION)
    session.check(
        "the history rail holds a question answered by both arms",
        opened,
        f"opened {INTERVIEW_QUESTION!r}: {opened}",
    )
    if not opened:
        return
    tabs = session.page.locator(".tab")
    titles = [tabs.nth(i).inner_text() for i in range(tabs.count())]
    session.check(
        "both arms of the question are offered as tabs on one answer",
        any(FILTERED_TITLE in t for t in titles) and any(BASELINE_TITLE in t for t in titles),
        f"tabs: {[' '.join(t.split()) for t in titles]}",
    )
    filtered_tab = session.page.get_by_text(FILTERED_TITLE, exact=False).first
    if filtered_tab.count():
        filtered_tab.click()
        session.page.wait_for_timeout(800)
    ids = filter_trace_checks(session, "committed interview question")
    off_genre = ids & OFF_GENRE
    session.check(
        "an interview question routes to no property or system-design video",
        bool(ids) and not off_genre,
        f"routed to {sorted(ids)}; off-genre among them: {sorted(off_genre) or 'none'}",
        shot=True,
    )


def coverage_warning_checks(session: UserSession) -> None:
    opened = open_conversation(session, COVERAGE_QUESTION)
    session.check(
        "the review of an unclassifiable page is in the history",
        opened,
        f"opened {COVERAGE_QUESTION!r}: {opened}",
    )
    if not opened or not open_trace(session):
        return
    notes = session.page.locator("details.trace .trace-note")
    target = None
    for index in range(notes.count()):
        if "does not match a kind the corpus has criteria for" in notes.nth(index).inner_text():
            target = notes.nth(index)
            break
    session.check(
        "the corpus-coverage caveat is on the page, not only in a tooltip",
        target is not None,
        "caveat rendered as trace text" if target is not None else "caveat missing",
    )
    if target is None:
        return
    label = target.locator(".trace-query-label").inner_text().strip()
    session.check(
        "the caveat is labelled as a warning rather than as more description",
        label.lower() == "warning",
        f"label reads {label!r}",
    )
    check_wraps_in_full(
        session,
        "the 273-character coverage caveat renders in full on a wrapping line",
        target.locator("span").last,
        expected=COVERAGE_TEXT,
    )
    session.shot("coverage_caveat")


def scoreboard_checks(session: UserSession) -> None:
    session.tab("Scoreboard")
    session.click("Answers (", exact=False)
    session.page.wait_for_timeout(600)

    setup_filter = session.page.get_by_role("combobox", name="Filter by setup")
    options = setup_filter.locator("option")
    labels = [options.nth(i).inner_text().strip() for i in range(options.count())]
    session.check(
        "the summary-filtered setup is selectable on the Scoreboard",
        FILTERED_TITLE in labels,
        f"setup filter offers {labels}",
        shot=True,
    )
    if FILTERED_TITLE in labels:
        setup_filter.select_option(label=FILTERED_TITLE)
        session.page.wait_for_timeout(500)
        rows = session.page.locator(".qpanel tbody tr")
        session.check(
            "selecting it shows its graded cells",
            rows.count() > 0,
            f"{rows.count()} rows under {FILTERED_TITLE}",
        )
        setup_filter.select_option(label="all setups")
        session.page.wait_for_timeout(500)

    # Beside, not merely present: one question, both setups, on the same screen.
    question_filter = session.page.get_by_role("combobox", name="Filter by question")
    question_filter.select_option(
        label=label_containing(session, "Filter by question", "ATS-friendly")
    )
    session.page.wait_for_timeout(600)
    body = session.page.locator(".qpanel tbody").inner_text()
    session.check(
        "one question shows the filtered setup's cell beside the baseline's",
        FILTERED_TITLE in body and BASELINE_TITLE in body,
        f"setups rendered for this question: "
        f"{sorted({line for line in body.splitlines() if line.startswith('rag_')})}",
        shot=True,
    )
    question_filter.select_option(label="all questions")
    session.page.wait_for_timeout(400)


def folded_in_scoreboard_fixes(session: UserSession) -> None:
    question_filter = session.page.get_by_role("combobox", name="Filter by question")

    # -- partial context, without expanding anything ------------------------
    question_filter.select_option(
        label=label_containing(session, "Filter by question", PARTIAL_CONTEXT_QUESTION[:40])
    )
    session.page.wait_for_timeout(600)
    badges = session.page.locator(".qpanel tbody .badge", has_text="partial context")
    session.check(
        "a partially-judged cell says so on its collapsed row",
        badges.count() > 0,
        f"{badges.count()} 'partial context' badge(s) visible with every row collapsed",
        shot=True,
    )
    if badges.count():
        title = badges.first.get_attribute("title") or ""
        session.check(
            "the badge states how much context the judge actually saw",
            re.search(r"judged on \d+ of \d+ retrieved chunks", title) is not None,
            f"badge title: {title!r}",
        )

    # -- the skip reason on a cell with no Expand button --------------------
    question_filter.select_option(
        label=label_containing(session, "Filter by question", NOT_RESCORED_QUESTION[:40])
    )
    session.page.wait_for_timeout(600)
    body = session.page.locator(".qpanel tbody")
    expanded = session.page.locator(".qpanel tbody .cellwhy")
    text = body.inner_text()
    session.check(
        "the un-rescored cell explains itself with nothing expanded",
        "could not be scored under this run" in text and expanded.count() == 0,
        f"explanation present: {'could not be scored under this run' in text}; "
        f"expanded panels open: {expanded.count()}",
        shot=True,
    )
    session.check(
        "and it names the reason it was skipped",
        "no answer to grade" in text,
        f"reason rendered: {'no answer to grade' in text}",
    )
    session.check(
        "the cell that carries it has no Expand button to hide behind",
        session.page.locator(".qpanel tbody").get_by_role("button", name="Expand").count()
        < session.page.locator(".qpanel tbody tr").count(),
        f"{session.page.locator('.qpanel tbody').get_by_role('button', name='Expand').count()} "
        f"Expand buttons across {session.page.locator('.qpanel tbody tr').count()} rows",
    )


def main() -> int:
    require_server()
    fixture_process = None
    with UserSession(SLICE) as session:
        # ── Chat: the routing decision, as a reader checks it ──────────────
        session.tab("Chat")
        # The live ask goes first, into an empty thread: everything below opens
        # a stored conversation, and a trace already on the page would make
        # "the answer rendered" true before the question was asked.
        live_ids = live_resume_ask(session)
        if live_ids:
            leaked = live_ids & OFF_GENRE
            session.check(
                "a resume question routes to no property or system-design video",
                not leaked,
                f"routed to {sorted(live_ids)}; off-genre among them: {sorted(leaked) or 'none'}",
                shot=True,
            )
        committed_interview_conversation(session)
        coverage_warning_checks(session)

        # ── Scoreboard: the new setup as a column beside the baseline ──────
        scoreboard_checks(session)
        folded_in_scoreboard_fixes(session)

        # ── the rounding edge, on a fixture run ───────────────────────────
        if not FIXTURES.joinpath(f"{FIXTURE_RUN}.json").exists():
            session.check(
                "the grounding-floor fixture exists",
                False,
                f"missing {FIXTURES / (FIXTURE_RUN + '.json')}",
            )
        elif not free_port(FIXTURE_PORT):
            session.check(
                "a fixture instance can be started for the rounding check",
                False,
                f"port {FIXTURE_PORT} is already in use",
            )
        else:
            fixture_process = serve_fixtures(FIXTURE_PORT)
            if fixture_process is None:
                session.check(
                    "a fixture instance can be started for the rounding check",
                    False,
                    "the second app instance did not come up within 90s",
                )
            else:
                try:
                    session.page.goto(
                        f"http://127.0.0.1:{FIXTURE_PORT}/", wait_until="networkidle"
                    )
                    session.page.wait_for_timeout(900)
                    session.tab("Scoreboard")
                    floor_rounding_checks(session)
                finally:
                    fixture_process.terminate()
                    fixture_process.wait(timeout=20)

        session.note(
            "The Chat checks are the hypothesis; the Scoreboard checks are whether the "
            "hypothesis was added to the lab. They can and did come apart."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
