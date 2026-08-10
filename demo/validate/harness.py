"""Replay a user's click path against the running app and record what was visible.

    uv run --group demo python -m demo.validate.v0_judge

Every vertical slice in the build plan claims a hypothesis that is only true if
you can *see* it in the app. A passing unit test does not settle that: it proves
a function returns the right value, not that the reviewer opening
http://127.0.0.1:8021 finds the thing the slice promised. So each slice ships
with a script here that drives a real browser through the exact path a human
would take — click the tab, click the sub-tab, expand the row — and asserts on
what the page then says.

Two rules make these scripts trustworthy as evidence:

* **The script must not touch the API directly to prove a UI claim.** If the
  assertion is "the metric table lists 8 rows", it reads the rendered table. An
  API call proving the server *could* return 8 rows is a different, weaker claim,
  and slices have shipped broken UIs behind correct APIs before.
* **The author of a slice does not write its validator.** Generation and
  evaluation run as separate agents, so the script encodes what an outsider
  would look for rather than what the implementer knows they built.

Each run writes ``artifacts/<slice>/verdict.json`` (checks, pass/fail, timing)
plus one screenshot per step, so a failure is reviewable after the fact without
re-running anything.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout, sync_playwright

BASE_URL = os.environ.get("YT_AGENT_APP_URL", "http://127.0.0.1:8021")
ARTIFACTS = Path(__file__).parent / "artifacts"

# The app fetches corpus/health/setups on mount and renders panels as those
# settle; a fixed settle beat after navigation is cheaper and less flaky than
# per-panel network waits, which differ by tab.
SETTLE_MS = 700


@dataclass
class Check:
    """One thing the reviewer would look at, and whether it was there."""

    name: str
    passed: bool
    detail: str
    shot: str | None = None


@dataclass
class Verdict:
    slice_id: str
    passed: bool = True
    started_at: str = ""
    duration_s: float = 0.0
    url: str = BASE_URL
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "checks": [asdict(c) for c in self.checks]}


class UserSession:
    """A browser driven the way a person drives it: by clicking visible things.

    Selectors here are deliberately text- and role-based rather than CSS class
    names. A class-name selector keeps passing when a label changes to something
    a human could no longer find, which is exactly the failure these scripts
    exist to catch.
    """

    def __init__(self, slice_id: str, *, headless: bool = True) -> None:
        self.slice_id = slice_id
        self.headless = headless
        self.verdict = Verdict(slice_id=slice_id)
        self.dir = ARTIFACTS / slice_id
        self._shots = 0
        self._t0 = 0.0

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "UserSession":
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self.verdict.started_at = datetime.now(timezone.utc).isoformat()
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        # 1440x900 is the widest layout the app designs for; a narrower viewport
        # collapses panels and would fail checks for layout reasons rather than
        # for the reason the slice is being judged on.
        self._ctx = self._browser.new_context(viewport={"width": 1440, "height": 900})
        self.page: Page = self._ctx.new_page()
        self.console: list[str] = []
        self.page.on("console", lambda m: self.console.append(f"{m.type}: {m.text}"))
        self.page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))
        self.page.goto(BASE_URL, wait_until="networkidle")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # SystemExit/KeyboardInterrupt are the script ending, not the slice failing.
        control_flow = isinstance(exc, (SystemExit, KeyboardInterrupt))
        if exc is not None and not control_flow:
            self.check("script completed", False, f"{exc_type.__name__}: {exc}")
        errors = [line for line in self.console if line.startswith("pageerror")]
        if errors:
            self.check("no uncaught frontend errors", False, "; ".join(errors[:3]))
        self.verdict.duration_s = round(time.time() - self._t0, 2)
        self.shot("final")
        try:
            self._ctx.close()
            self._browser.close()
        finally:
            self._pw.stop()
        (self.dir / "verdict.json").write_text(json.dumps(self.verdict.as_dict(), indent=2) + "\n")
        self._print_summary()
        # The verdict file is the report, so a failed *check* never becomes a
        # traceback — but a deliberate exit propagates.
        return not control_flow

    def _print_summary(self) -> None:
        for c in self.verdict.checks:
            print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}")
        state = "PASS" if self.verdict.passed else "FAIL"
        print(f"{state}  {self.slice_id}  ({self.verdict.duration_s}s)  -> {self.dir}/verdict.json")

    # -- navigation --------------------------------------------------------
    def tab(self, label: str) -> None:
        """Click a top-nav tab by its visible label ("Scoreboard", "Chat", ...)."""
        self.page.get_by_role("navigation", name="Views").get_by_role(
            "button", name=label, exact=True
        ).click()
        self.page.wait_for_timeout(SETTLE_MS)

    def click(self, label: str, *, exact: bool = False, nth: int = 0) -> None:
        """Click the nth visible element whose text matches — a sub-tab, row, or link."""
        target = self.page.get_by_text(label, exact=exact).nth(nth)
        target.scroll_into_view_if_needed()
        target.click()
        self.page.wait_for_timeout(SETTLE_MS)

    def click_button(self, label: str, *, exact: bool = True, nth: int = 0) -> None:
        self.page.get_by_role("button", name=label, exact=exact).nth(nth).click()
        self.page.wait_for_timeout(SETTLE_MS)

    def wait_for_text(self, needle: str, *, timeout_ms: int = 15000) -> bool:
        """Wait for text to appear; returns whether it did rather than raising."""
        try:
            self.page.get_by_text(needle).first.wait_for(state="visible", timeout=timeout_ms)
            return True
        except PlaywrightTimeout:
            return False

    # -- reading the page --------------------------------------------------
    def text(self) -> str:
        return self.page.locator("body").inner_text()

    def count_text(self, needle: str) -> int:
        return self.page.get_by_text(needle).count()

    def shot(self, name: str) -> str:
        self._shots += 1
        path = self.dir / f"{self._shots:02d}_{name}.png"
        try:
            self.page.screenshot(path=str(path), full_page=True)
        except Exception:  # a closed page during teardown is not a slice failure
            return ""
        return path.name

    # -- assertions --------------------------------------------------------
    def check(self, name: str, passed: bool, detail: str = "", *, shot: bool = False) -> bool:
        shot_name = self.shot(name.replace(" ", "_")[:40]) if shot else None
        self.verdict.checks.append(
            Check(name=name, passed=bool(passed), detail=detail, shot=shot_name)
        )
        if not passed:
            self.verdict.passed = False
        return bool(passed)

    def check_visible(self, name: str, needle: str, **kw: Any) -> bool:
        """Assert text a user could actually read on the page right now."""
        present = self.count_text(needle) > 0
        return self.check(name, present, f"{'found' if present else 'missing'}: {needle!r}", **kw)

    def note(self, message: str) -> None:
        self.verdict.notes.append(message)


def exit_code(*sessions: UserSession) -> int:
    return 0 if all(s.verdict.passed for s in sessions) else 1


def require_server() -> None:
    """Fail loudly and early if the app is not up — an empty page fails every check."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"status {resp.status}")
    except Exception as exc:  # noqa: BLE001 - message matters more than the type
        print(
            f"app not reachable at {BASE_URL} ({exc}).\n"
            "start it with: uv run python -m src.cli serve --port 8021",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
