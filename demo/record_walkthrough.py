"""Record the narrated walkthrough from ``script.json``.

The scene list is data, not code, so the adversarial reviewer can patch it and
re-record deterministically — a take is reproducible from a committed artefact
rather than from a conversation. This module only knows how to *execute* a
scene; what the scenes are is entirely ``script.json``'s business.

    uv run --group demo python demo/record_walkthrough.py --out take_01

Scenes carrying a ``blocked`` note are skipped with a warning rather than
failing the run: a missing demo fixture should cost you that scene, not the
whole recording.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from narration import Narrator, take_webm  # noqa: E402

DEMO = Path(__file__).parent
ARTIFACTS = DEMO / "artifacts"


def load_script(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("scenes"):
        raise SystemExit(f"{path} has no scenes")
    return data


def _resolve(value: str, script: dict[str, Any]) -> str:
    """Expand ``{base_url}`` / ``{slides_url}`` / ``{demo_dir}`` placeholders.

    The title deck loads over ``file://`` rather than through a route on the
    app: it is a recording prop, and adding a demo-only endpoint to the
    production server to serve it would be infrastructure the product does not
    need. Playwright records it in the same context either way.
    """
    slides = script["slides_url"].format(demo_dir=DEMO.as_posix())
    return value.format(base_url=script["base_url"], slides_url=slides, demo_dir=DEMO.as_posix())


def run_action(page: Page, action: dict[str, Any], script: dict[str, Any]) -> None:
    kind = action["do"]
    if kind == "goto":
        page.goto(_resolve(action["url"], script), wait_until="domcontentloaded")
    elif kind == "wait":
        page.wait_for_timeout(int(action["ms"]))
    elif kind == "scroll_to":
        page.eval_on_selector(
            action["selector"],
            "el => el.scrollIntoView({behavior: 'smooth', block: 'start'})",
        )
    elif kind == "click":
        page.locator(action["selector"]).first.click()
    elif kind == "hide":
        # Dev-only chrome (the runner-warm diagnostic) is not part of the
        # product story and reads as a leaked internal on a portfolio recording.
        # Hidden rather than removed, so layout does not reflow mid-scene.
        page.eval_on_selector_all(
            action["selector"], "els => els.forEach(el => el.style.visibility = 'hidden')"
        )
    elif kind == "click_text":
        # Pick the entry whose text contains `text` — history labels are the
        # user's own questions, so an index would break the moment they ask
        # another one.
        page.locator(action["selector"], has_text=action["text"]).first.click()
    else:
        raise ValueError(f"unknown action: {kind}")


def record(script: dict[str, Any], stem: str, *, skip_blocked: bool = True) -> Path:
    scenes = [scene for scene in script["scenes"] if not (skip_blocked and scene.get("blocked"))]
    for scene in script["scenes"]:
        if scene.get("blocked") and skip_blocked:
            print(f"⚠ skipping scene '{scene['key']}': {scene['blocked']}")

    voiceover = {scene["key"]: scene["vo"] for scene in scenes}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    narrator = Narrator(ARTIFACTS, voiceover)
    narrator.synthesize()

    viewport = script["viewport"]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport=viewport,
            record_video_dir=str(ARTIFACTS),
            record_video_size=viewport,
        )
        page = context.new_page()
        # The clock has to start when the *video* starts, not when the first
        # scene does. Playwright begins capturing at context creation, so any
        # setup done before `narrator.start()` becomes video the SRT timeline
        # knows nothing about — which offsets both the subtitles and the mixed
        # audio from the visuals by exactly that much. Start first, then load,
        # and keep the pre-roll under the dead-air budget.
        narrator.start()
        page.goto(script["base_url"], wait_until="domcontentloaded")
        page.wait_for_timeout(1200)
        for scene in scenes:
            print(f"  ▸ {scene['key']}: {scene['shot']}")
            for action in scene.get("actions", []):
                run_action(page, action, script)
            narrator.caption(page, scene["key"])
            narrator.scene(page, scene["key"], int(scene.get("hold_ms", 8000)))

        page.wait_for_timeout(700)
        context.close()
        browser.close()

    webm = take_webm(ARTIFACTS, stem)
    narrator.finish(webm, stem)
    return ARTIFACTS / f"{stem}.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=DEMO / "script.json")
    parser.add_argument("--out", default="walkthrough", help="output stem")
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="attempt scenes marked blocked instead of skipping them",
    )
    args = parser.parse_args()

    script = load_script(args.script)
    mp4 = record(script, args.out, skip_blocked=not args.include_blocked)
    print(f"\n{mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
