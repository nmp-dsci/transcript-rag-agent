"""V0 — Judge v2 (depth-v2) on the Scoreboard: the reviewer's click path.

    uv run python -m src.cli serve --port 8021          # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v0_judge

The slice claims the judge can now see depth, and that seeing depth changes
which setup wins. That claim is only settled in the browser, so every assertion
below reads rendered page text. Nothing here calls the API: a correct
``/api/scoreboard`` payload behind a table that renders three columns would
satisfy the API and fail the slice.

The path is the one a reviewer takes with no knowledge of the implementation:

1. open Scoreboard and pick the ``depth-v2`` run out of the run selector;
2. read the Rubric panel — eight metrics, banded into grounding and depth, each
   carrying a visible weight, because the weights *are* the claim about what a
   good answer is;
3. open Answers, expand a capped cell, and read why it was capped as page text
   rather than as a tooltip;
4. switch back to the ``ragas-v1`` run the depth run was rejudged from and check
   it still renders its three metrics;
5. read the leaderboard order under both and compare them. If the two orders are
   identical, the five new dimensions moved nothing and the slice failed its own
   hypothesis, however well the table renders.
"""

from __future__ import annotations

import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v0_judge"

DEPTH_RUN = "matrix-20260809-071818-depth-v2"
#: The run ``DEPTH_RUN`` was rejudged from — same answers, same grounding
#: scores, older rubric. Comparing against any other run would confound the
#: rubric change with a change of answers.
BASELINE_RUN = "matrix-20260729-025133"

GROUNDING_METRICS = ["Faithful", "Precision", "Relevant"]
DEPTH_METRICS = ["Insight", "Specific", "Coverage", "Breadth", "Calibrated"]

#: A committed run with exactly five judged questions per row — the sample size
#: the low-n warning is supposed to catch, and the size the first depth-v2 run
#: happened to be when the threshold was still exclusive.
THIN_RUN = "matrix-20260729-061607"


def select_run(session: UserSession, run_id: str) -> str:
    """Pick a run in the visible run selector; returns the option's label."""
    picker = session.page.get_by_role("combobox", name="Matrix run")
    label = picker.locator(f'option[value="{run_id}"]').inner_text()
    picker.select_option(run_id)
    session.page.wait_for_timeout(1500)
    return label.strip()


def leaderboard_order(session: UserSession) -> list[str]:
    """The setup column of the leaderboard, top to bottom, as rendered."""
    table = session.page.get_by_role("table", name="Leaderboard")
    return [cell.inner_text().strip() for cell in table.locator("tbody tr td:first-child").all()]


def rubric_metric_rows(session: UserSession) -> list[str]:
    """The metric-name column of the Rubric panel, as rendered."""
    table = session.page.get_by_role("table", name="Rubric metrics")
    return [row.inner_text().strip() for row in table.locator("tr.metricrow td:first-child").all()]


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        session.tab("Scoreboard")

        # ── 1. the depth-v2 run is selectable ────────────────────────────
        options = session.page.get_by_role("combobox", name="Matrix run").inner_text()
        session.check(
            "depth-v2 run offered in the run selector",
            DEPTH_RUN in options and "depth-v2" in options,
            f"selector options: {options.strip()[:300]!r}",
        )
        depth_label = select_run(session, DEPTH_RUN)
        session.check(
            "run label names its rubric",
            "depth-v2" in depth_label,
            f"selected option reads {depth_label!r}",
            shot=True,
        )

        # ── 2. the metric table lists 8 rows, grouped and weighted ───────
        rows = rubric_metric_rows(session)
        session.check(
            "rubric table lists 8 metric rows",
            len(rows) == 8,
            f"{len(rows)} rows: {rows}",
        )
        missing = [m for m in GROUNDING_METRICS + DEPTH_METRICS if m not in rows]
        session.check(
            "3 grounding + 5 depth metrics named",
            not missing,
            f"missing from the rendered table: {missing}" if missing else "all 8 present",
        )
        page = session.text()
        session.check(
            "metrics banded into grounding and depth with band weights",
            "grounding" in page and "depth" in page and "40%" in page and "60%" in page,
            "grounding 40% / depth 60% bands visible" if "40%" in page else "band weights missing",
        )
        weights = session.page.get_by_role("table", name="Rubric metrics").inner_text()
        per_metric = [w for w in ("20%", "15%", "10%", "5%") if w in weights]
        session.check(
            "each metric carries its own weight",
            len(per_metric) == 4,
            f"weights rendered in the rubric table: {per_metric}",
            shot=True,
        )
        session.check_visible(
            "composite formula stated on the page",
            "capped at 0.5 when faithfulness < 0.6",
        )

        depth_order = leaderboard_order(session)
        session.check(
            "leaderboard renders a ranking under depth-v2",
            len(depth_order) >= 2,
            f"order: {depth_order}",
        )

        # ── 2b. the ranking discloses how it was produced ────────────────
        # A leaderboard the answering model also graded is self-assessment. It
        # is only safe to publish if the page says so where the ranking is
        # read, not in a field a reviewer would have to go looking for.
        board = session.text()
        session.check(
            "self-grading is disclosed above the leaderboard",
            "self-graded" in board and "also" in board and "graded them" in board,
            "self-graded banner present" if "self-graded" in board else "no banner",
            shot=True,
        )
        prov = session.page.locator(".provbar").inner_text()
        session.check(
            "both judges are named in provenance, not just the grounding one",
            "depth judge" in prov.lower() and "deepseek-v4-flash" in prov,
            f"provenance bar reads: {' '.join(prov.split())[:300]!r}",
        )

        # ── 2c. n=20 is not treated as a thin sample ─────────────────────
        leaderboard = session.page.get_by_role("table", name="Leaderboard").inner_text()
        session.check(
            "the n=20 run is not badged thin",
            "n=20" in leaderboard and "thin" not in leaderboard,
            f"'thin' present: {'thin' in leaderboard}; n values: "
            f"{[t for t in leaderboard.split() if t.startswith('n=')]}",
        )

        # ── 3. a capped cell explains itself in the Answers panel ────────
        session.page.get_by_text("Answers (", exact=False).first.click()
        session.page.wait_for_timeout(500)
        capped_cells = session.page.locator(
            "tr", has=session.page.get_by_text("capped", exact=True)
        )
        found = capped_cells.count()
        session.check(
            "a capped answer is visible in the Answers panel",
            found > 0,
            f"{found} row(s) carry a 'capped' badge",
        )
        if found:
            row = capped_cells.first
            row.scroll_into_view_if_needed()
            row.get_by_role("button", name="Expand").click()
            session.page.wait_for_timeout(400)
            why = row.inner_text()
            session.check(
                "cap reason is readable page text, not a tooltip",
                "depth cannot rescue an ungrounded answer" in why and "below 0.6" in why,
                f"expanded cell reads: {' '.join(why.split())[:400]!r}",
                shot=True,
            )
            session.check(
                "capped cell shows the score it was capped from",
                "Composite capped at" in why and "from" in why,
                f"cap line present: {'Composite capped at' in why}",
            )
            session.check(
                "depth metrics carry per-answer rationales",
                any(name in why for name in DEPTH_METRICS),
                f"rationale labels present: {[n for n in DEPTH_METRICS if n in why]}",
            )

        # ── 3b. an ungrounded answer the cap never touched ───────────────
        # The failure this catches: an answer with faithfulness 0.09 scores
        # below the cap on its own, so a "capped" badge alone would leave the
        # worst answer on the page looking like an ordinary low score.
        ungrounded_cells = session.page.locator(
            "tr", has=session.page.get_by_text("ungrounded", exact=True)
        )
        found_ug = ungrounded_cells.count()
        session.check(
            "an ungrounded-but-uncapped answer is marked as such",
            found_ug > 0,
            f"{found_ug} row(s) carry an 'ungrounded' badge",
        )
        if found_ug:
            row = ungrounded_cells.first
            row.scroll_into_view_if_needed()
            row.get_by_role("button", name="Expand").click()
            session.page.wait_for_timeout(400)
            why = row.inner_text()
            session.check(
                "the ungrounded reason is readable page text",
                "below 0.6" in why and "depth cannot rescue an ungrounded answer" in why,
                f"expanded cell reads: {' '.join(why.split())[:360]!r}",
                shot=True,
            )
            session.check(
                "'ungrounded' is distinguishable from 'capped' on the page",
                "The cap" in why and "changed nothing here" in why and "capped at" not in why,
                "the cell states the cap changed nothing"
                if "changed nothing here" in why
                else f"reads: {' '.join(why.split())[:240]!r}",
            )

        # ── 3c. a cell that could not be scored under this rubric ────────
        # Previously this cell kept its ragas-v1 composite and was averaged
        # into a depth-v2 mean. It must now be excluded *and* visible.
        session.check(
            "the un-rescored cell is shown rather than silently dropped",
            "not rescored" in session.text(),
            f"'not rescored' present: {'not rescored' in session.text()}",
        )
        skipped_rows = session.page.locator(
            "tr", has=session.page.get_by_text("not rescored", exact=True)
        )
        session.check(
            "exactly the one un-rescored cell is marked",
            skipped_rows.count() == 1,
            f"{skipped_rows.count()} row(s) marked 'not rescored'",
            shot=True,
        )

        # ── 3d. partial context is disclosed on the cell it affected ─────
        # Unlike capped / ungrounded / not-rescored, this disclosure exists only
        # inside the expanded cell — there is no badge on the collapsed row — so
        # it has to be navigated to by filter rather than found by scanning.
        collapsed_badge = session.page.locator(
            "tr", has=session.page.get_by_text("partial context", exact=True)
        ).count()
        session.page.get_by_role("combobox", name="Filter by question").select_option("g010")
        session.page.get_by_role("combobox", name="Filter by setup").select_option("graph_rag")
        session.page.wait_for_timeout(400)
        rows_left = session.page.locator("details.qpanel").nth(1).locator("tbody tr")
        session.check(
            "the partially-resolved cell can be filtered to",
            rows_left.count() == 1,
            f"{rows_left.count()} row(s) after filtering to g010 x graph_rag",
        )
        if rows_left.count() == 1:
            row = rows_left.first
            row.scroll_into_view_if_needed()
            row.get_by_role("button", name="Expand").click()
            session.page.wait_for_timeout(400)
            text = " ".join(row.inner_text().split())
            session.check(
                "the partial-context cell says how much context it was judged on",
                "Judged on 24 of 36 retrieved chunks" in text,
                f"reads: {text[text.find('partial context') :][:220]!r}"
                if "partial context" in text
                else f"reads: {text[:220]!r}",
                shot=True,
            )
            session.note(
                f"partial context has no collapsed-row badge ({collapsed_badge} found before "
                "expanding): unlike capped/ungrounded/not-rescored it is reachable only by "
                "expanding that exact cell."
            )
        session.page.get_by_role("combobox", name="Filter by question").select_option("")
        session.page.get_by_role("combobox", name="Filter by setup").select_option("")
        session.page.wait_for_timeout(300)

        # ── 3e. no mixed-rubric banner on a single-rubric run ────────────
        session.check(
            "a single-rubric run shows no mixed-rubric warning",
            "mixed rubrics" not in session.text(),
            "no mixed-rubric banner, as expected for a depth-v2-only run",
        )

        # ── 4. the older ragas-v1 run still renders ──────────────────────
        base_label = select_run(session, BASELINE_RUN)
        session.check(
            "the pre-depth run is still selectable and labelled ragas-v1",
            "ragas-v1" in base_label,
            f"selected option reads {base_label!r}",
        )
        base_rows = rubric_metric_rows(session)
        session.check(
            "ragas-v1 renders its 3 metrics",
            len(base_rows) == 3 and all(m in base_rows for m in GROUNDING_METRICS),
            f"{len(base_rows)} rows: {base_rows}",
            shot=True,
        )
        session.check(
            "no depth column leaks into the ragas-v1 view",
            not any(m in base_rows for m in DEPTH_METRICS),
            f"depth metrics present under ragas-v1: {[m for m in DEPTH_METRICS if m in base_rows]}",
        )
        base_order = leaderboard_order(session)

        # ── 5. the two rankings differ ───────────────────────────────────
        session.check(
            "the two rankings are read from the same setups",
            sorted(base_order) == sorted(depth_order),
            f"ragas-v1 setups {sorted(base_order)} vs depth-v2 {sorted(depth_order)}",
        )
        session.check(
            "depth-v2 ranks the setups differently from ragas-v1",
            base_order != depth_order and len(base_order) > 1,
            f"ragas-v1: {base_order} | depth-v2: {depth_order}",
            shot=True,
        )
        session.note(f"ragas-v1 ranking as rendered: {base_order}")
        session.note(f"depth-v2 ranking as rendered: {depth_order}")
        session.note(
            f"{DEPTH_RUN} covers 20 questions over 4 setups, rejudged from {BASELINE_RUN}; "
            "above the app's low-sample threshold, so the ranking is not marked thin."
        )

        # ── 6. a five-question run is still called thin ──────────────────
        # The threshold used to be exclusive, so a run of exactly LOW_N
        # questions — which is what the first depth-v2 run was — escaped the
        # warning it existed to trigger. Checked on a run that still has n=5.
        select_run(session, THIN_RUN)
        thin_table = session.page.get_by_role("table", name="Leaderboard").inner_text()
        session.check(
            "a run of exactly 5 judged questions is badged thin",
            "n=5" in thin_table and "thin" in thin_table,
            f"n values {[t for t in thin_table.split() if t.startswith('n=')]}; "
            f"'thin' present: {'thin' in thin_table}",
            shot=True,
        )
    return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
