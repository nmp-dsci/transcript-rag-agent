"""The two warnings that must *not* fire, checked against runs that trigger them.

    YT_AGENT_APP_URL=http://127.0.0.1:8022 \
      PYTHONPATH=. uv run --group demo python -m demo.validate.v0_judge_negatives

A warning banner is only evidence if it is conditional. "Self-graded" rendered
on every run would be indistinguishable from a hard-coded string, and the
mixed-rubric banner cannot be seen at all on the committed runs because all of
them are judged under a single rubric. So both are exercised here against
fixtures built for the purpose:

* ``matrix-29990101-000002-independent`` — the committed depth-v2 run with both
  judges renamed to a model the answers did not come from. The banner must be
  absent and provenance must name the other judge.
* ``matrix-29990101-000001-mixed`` — the same run with one setup's cells left at
  ``ragas-v1``. The banner must appear and the columns must fall back.

These fixtures are *not* committed runs and must not live in ``evals/runs``, so
they are served by a second app instance pointed at
``demo/validate/fixtures``. The committed depth-v2 run is served by the same
instance as a positive control: if the self-graded banner shows there and not on
the independent run, the banner is reading the data rather than the page.
"""

from __future__ import annotations

import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v0_judge_negatives"

SELF_GRADED_RUN = "matrix-20260809-071818-depth-v2"
INDEPENDENT_RUN = "matrix-29990101-000002-independent"
MIXED_RUN = "matrix-29990101-000001-mixed"


def select_run(session: UserSession, run_id: str) -> None:
    session.page.get_by_role("combobox", name="Matrix run").select_option(run_id)
    session.page.wait_for_timeout(1500)


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        session.tab("Scoreboard")

        # ── positive control on this same instance ───────────────────────
        select_run(session, SELF_GRADED_RUN)
        session.check(
            "control: the self-graded run does show the banner",
            "self-graded" in session.text(),
            "banner present on the run whose judge matches its answer model",
        )

        # ── the banner is conditional, not decoration ────────────────────
        select_run(session, INDEPENDENT_RUN)
        page = session.text()
        session.check(
            "an independently-judged run shows no self-graded banner",
            "self-graded" not in page,
            "'self-graded' absent" if "self-graded" not in page else "banner still rendered",
            shot=True,
        )
        prov = session.page.locator(".provbar").inner_text()
        session.check(
            "provenance names the independent judge instead",
            "gpt-5-mini" in prov and "deepseek" not in prov.split("embeddings")[0],
            f"provenance reads: {' '.join(prov.split())[:200]!r}",
        )
        session.check(
            "the leaderboard still ranks under the independent judge",
            session.page.get_by_role("table", name="Leaderboard").locator("tbody tr").count() == 4,
            "4 setup rows rendered",
        )

        # ── the mixed-rubric warning fires when rubrics actually mix ─────
        select_run(session, MIXED_RUN)
        mixed_page = session.text()
        session.check(
            "a mixed-rubric run warns that its composites are not on one scale",
            "mixed rubrics" in mixed_page,
            "'mixed rubrics' banner present" if "mixed rubrics" in mixed_page else "no banner",
            shot=True,
        )
        session.check(
            "the warning names both rubrics and says the ranking is unusable",
            "depth-v2" in mixed_page and "ragas-v1" in mixed_page and "unusable" in mixed_page,
            "banner names both rubrics and calls the ranking unusable",
        )
        rubric_rows = session.page.get_by_role("table", name="Rubric metrics").locator(
            "tr.metricrow"
        )
        session.check(
            "a mixed run falls back to the three metrics every record has",
            rubric_rows.count() == 3,
            f"{rubric_rows.count()} metric rows under a mixed run",
        )
    return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
