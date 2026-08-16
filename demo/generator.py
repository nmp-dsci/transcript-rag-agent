"""The generator half of the loop: rewrite the script from the review.

``loop.apply_fixes`` could only overwrite narration text and hold durations, so
every structural finding — *crop this frame*, *open that panel*, *stop showing
two contradictory tables* — was logged "(manual)" and nothing changed. The loop
detected well and remade nothing.

This module closes it. The generator is an agent, not a field-patcher: it is
given the current script, the reviewer's verdict, and the frames the reviewer
was looking at, and it returns a **complete replacement script** in which the
actions themselves may change. Because it sees the same frames, it can act on
"the caption covers the row it describes" by scrolling differently, rather than
by rewording the caption to talk about something else.

Everything it returns is validated before it is written: an agent that emits an
unknown action or drops the corpus URL would otherwise produce a take that
fails for reasons unrelated to the critique it was answering.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# Anything the recorder can execute. The generator is told this is the whole
# vocabulary; a script using anything else is rejected rather than recorded.
# Kept just under review.CAPTION_CHAR_BUDGET so a rewrite cannot produce a take
# that is guaranteed to fail the deterministic pass and burn a round.
CAPTION_BUDGET = 185

ACTIONS = {
    "goto": "navigate — url may use {base_url} or {slides_url}",
    "wait": "pause — ms",
    "click": "click the first match — selector",
    "click_text": "click the first match containing text — selector, text",
    "scroll_to": "smooth-scroll an element into view — selector",
    "hide": "visibility:hidden on every match — selector (for dev-only chrome)",
}

SELECTOR_NOTES = """Selectors that exist in this app (verified):
  .rentry                    a conversation in the chat history rail
  details.refs > summary     the collapsed "Sources (N)" panel on an answer
  details.refs               the sources list once open — timestamped links
  .topstat                   header chip incl. the dev-only "stack cold" flag
  .panel                     a card on Experiments/Scoreboard
  .qpanel                    collapsed disclosure panels (Questions, Answers)
  table                      any data table
Tabs are hash-routed: {base_url}/#chat, /#pipeline, /#board, /#experiments, /#design
The title deck is {slides_url}#s1 and #s2 (one viewport per slide)."""

GENERATOR_SYSTEM = """You rewrite a screen-recording script so the next take \
answers a reviewer's criticisms.

You are not editing prose. Most findings are about what is ON SCREEN, and the \
fix is usually a different action — scroll further, open a panel, hide an \
element, hold longer — not a reworded sentence. Rewording a caption so it stops \
describing the problem is the failure mode to avoid: it games the score without \
improving the video.

Hard constraints (a script violating these is rejected and wastes a round):
- Every narration line MUST be <= 185 characters. The caption bar is two lines   at the recorded width; longer text wraps over the UI it describes.
- Narration must be sayable in roughly the scene's hold time (~2.5 words/sec).

Hard rules:
- NEVER introduce a number, score, run ID or proper noun into narration unless   you have SEEN it in that scene's frame. Adding specificity you cannot confirm   is the single worst thing you can do here: it manufactures the exact defect   the reviewer caps for. When unsure, describe the shot qualitatively.
- Never make a claim the frame cannot support. If a capability cannot be shown, \
  remove the claim rather than keep asserting it.
- A scene marked "blocked" must keep its blocked field and stays unrecorded.
- Keep scene "key" values stable so scores stay comparable across rounds.
- Output ONLY the complete JSON script. No prose, no fences."""


def build_prompt(script: dict[str, Any], verdict: dict[str, Any], frame_dir: Path | None) -> str:
    fixes = verdict.get("must_fix", []) or []
    lines = [
        "The previous take scored "
        f"{verdict.get('total')}/10. Per-dimension: {verdict.get('scores')}.",
    ]
    if verdict.get("capped"):
        lines.append(f"It was CAPPED: {verdict.get('cap_reason')}")
    lines += ["", "The reviewer requires these fixes:"]
    for fix in fixes:
        lines.append(
            f"  scene {fix.get('scene')}: {fix.get('problem')}\n"
            f"    evidence: {fix.get('evidence')}\n"
            f"    suggested: {fix.get('fix')}"
        )
    lines += [
        "",
        "Actions the recorder can execute (nothing else is valid):",
        *[f"  {name}: {desc}" for name, desc in ACTIONS.items()],
        "",
        SELECTOR_NOTES,
        "",
        "Current script:",
        json.dumps(script, indent=2),
    ]
    if frame_dir is not None:
        lines += [
            "",
            f"The frames the reviewer graded are in {frame_dir} "
            "(sceneNN.jpg, in scene order). Read them before rewriting: the fix "
            "for a framing problem is visible there and not in the script.",
        ]
    lines += ["", "Return the complete rewritten script as JSON."]
    return "\n".join(lines)


def extract_json(text: str) -> Any:
    """Pull the script object out of an agent reply.

    A greedy ``{.*}`` match spans from the first brace to the last, which
    swallows any commentary containing braces and yields a parse error blamed on
    the model. Prefer a fenced block, then scan for the first balanced object.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    # Every balanced object in the reply, not just the first: a preamble like
    # "I considered {this} but…" yields a well-formed object that is not the
    # script, and stopping there reports a parse failure for a valid reply.
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
        start = text.find("{", start + 1)
    # Prefer the object that actually looks like a script.
    candidates.sort(key=lambda text: 0 if '"scenes"' in text else 1)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"generator returned no parseable JSON:\n{text[:400]}")


def validate(candidate: Any, previous: dict[str, Any]) -> dict[str, Any]:
    """Reject a script the recorder could not run, before it wastes a round."""
    if not isinstance(candidate, dict):
        raise ValueError("generator did not return an object")
    scenes = candidate.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("rewritten script has no scenes")

    previous_keys = [scene["key"] for scene in previous["scenes"]]
    new_keys = [scene.get("key") for scene in scenes]
    # Adding a scene is a legitimate structural fix — the reviewer asked for one
    # and the first version of this validator rejected the whole rewrite for it.
    # Dropping or renaming an existing scene is not: scores stop being
    # comparable across rounds, which is the only reason the loop can converge.
    missing = [key for key in previous_keys if key not in new_keys]
    if missing:
        raise ValueError(f"rewrite dropped scenes: {missing}")
    if new_keys[: len(previous_keys)] != previous_keys:
        raise ValueError(f"existing scenes reordered: {new_keys} vs {previous_keys}")

    for scene in scenes:
        vo = scene.get("vo")
        if not vo:
            raise ValueError(f"scene {scene.get('key')} has no narration")
        if len(vo) > CAPTION_BUDGET:
            raise ValueError(
                f"scene {scene.get('key')}: narration is {len(vo)} chars "
                f"(budget {CAPTION_BUDGET}) — it would fail the caption check"
            )
        for action in scene.get("actions", []):
            kind = action.get("do")
            if kind not in ACTIONS:
                raise ValueError(f"unknown action {kind!r} in scene {scene.get('key')}")
            if kind in {"click", "click_text", "scroll_to", "hide"} and not action.get("selector"):
                raise ValueError(f"{kind} without a selector in scene {scene.get('key')}")
            if kind == "goto" and not action.get("url"):
                raise ValueError(f"goto without a url in scene {scene.get('key')}")

    # Settings the reviewer has no business changing.
    for field in ("base_url", "slides_url", "viewport"):
        candidate[field] = previous[field]
    for scene, old in zip(scenes, previous["scenes"]):
        if old.get("blocked"):
            scene["blocked"] = old["blocked"]
    return candidate


def regenerate(
    script: dict[str, Any],
    verdict: dict[str, Any],
    *,
    frame_dir: Path | None = None,
    model: str = "claude-sonnet-5",
    oauth_token: str = "",
) -> dict[str, Any]:
    """Ask the generator agent for a full rewrite; validate before returning."""
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, query

    if oauth_token:
        os.environ.setdefault("CLAUDE_CODE_OAUTH_TOKEN", oauth_token)

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=GENERATOR_SYSTEM,
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        cwd=str(frame_dir) if frame_dir else None,
    )
    prompt = build_prompt(script, verdict, frame_dir)

    async def run() -> str:
        chunks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            for block in getattr(message, "content", []) or []:
                if hasattr(block, "text"):
                    chunks.append(block.text)
        return "".join(chunks)

    raw = asyncio.run(run())
    return validate(extract_json(raw), script)
