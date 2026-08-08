"""Build -> score -> rebuild, until the take clears the bar or the rounds run out.

    uv run --group demo python demo/loop.py --threshold 8 --max-rounds 5

Each round records a take from ``script.json``, runs the deterministic checks,
then puts one keyframe per scene in front of an adversarial reviewer. Anything
under the threshold comes back as ``must_fix[]``, which is applied to
``script.json`` and re-recorded.

Two properties matter more than the loop itself:

* **The best take ships, not the last one.** A refinement loop that always keeps
  its final attempt can hand you a round-5 regression.
* **Every round's score is written to ``review_log.json`` and committed.** If the
  video ships at 7.4 after five rounds, that number is in the repo. A loop whose
  failures are invisible is just a slower one-shot.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from generator import CAPTION_BUDGET, regenerate  # noqa: E402
from record_walkthrough import ARTIFACTS, load_script, record  # noqa: E402
from review import (  # noqa: E402
    REVIEWER_SYSTEM,
    deterministic_checks,
    extract_frames,
    parse_srt,
    reviewer_prompt,
    weighted_total,
)

DEMO = Path(__file__).parent
REVIEW_LOG = ARTIFACTS / "review_log.json"
MISSION = """A portfolio piece for a senior/principal AI engineer. In 60-75 \
seconds it must show: (1) you can ask a corpus of expert YouTube transcripts a \
real question and get an answer whose every claim cites the exact second of the \
exact video; (2) you can point it at your own resume and have it critiqued \
against that corpus; (3) six retrieval architectures were measured head to head \
on a labelled set by one judge, including three that lost. The story is Itch -> \
Build -> Proof. The audience is a hiring engineer who has seen a hundred RAG \
demos and believes none of them."""


def _api_key(name: str) -> str:
    """Same resolution order the app uses — env first, then ``~/.env``."""
    key = os.environ.get(name, "")
    if key:
        return key
    env = Path(os.environ.get("YT_AGENT_ENV_PATH", "~/.env")).expanduser()
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


# The app answers with DeepSeek, which is text-only, so the reviewer has to come
# from elsewhere. All three read images; which one is available is a billing
# question, not a design one, so the backend is swappable rather than assumed.
#
# `claude-sdk` is the default: it runs through the Claude Agent SDK on the
# CLAUDE_CODE_OAUTH_TOKEN (a subscription credential, sk-ant-oat…), so a review
# round costs nothing per-token. The `anthropic` backend uses the pay-as-you-go
# ANTHROPIC_API_KEY (sk-ant-api…) and is a different account entirely — on this
# machine it has no credit.
PROVIDERS = {
    "claude-sdk": ("CLAUDE_CODE_OAUTH_TOKEN", "claude-sonnet-5"),
    "openai": ("OPENAI_API_KEY", "gpt-4o"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-5"),
}


def _review_claude_sdk(prompt: str, frames: list[Path], model: str) -> str:
    """Review through the Claude Agent SDK, billed to the subscription.

    The agent reads the frames off disk with its own Read tool rather than being
    handed base64 blobs — that is what the tool is for, and it keeps the prompt
    small no matter how many scenes the take has. Tools are restricted to Read
    so a reviewer cannot wander the repo and grade the source instead of the
    video it was asked about.
    """
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, query

    frame_dir = frames[0].parent
    listing = "\n".join(f"  {f.name}  -> scene {i + 1}" for i, f in enumerate(frames))
    task = (
        f"{prompt}\n\n"
        f"The frames are on disk in {frame_dir}. Read each one, in order:\n{listing}\n\n"
        "Read every frame before scoring. Output ONLY the JSON object."
    )
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=REVIEWER_SYSTEM,
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        cwd=str(frame_dir),
    )

    async def run() -> str:
        chunks: list[str] = []
        async for message in query(prompt=task, options=options):
            for block in getattr(message, "content", []) or []:
                if getattr(block, "type", None) == "text" or hasattr(block, "text"):
                    chunks.append(getattr(block, "text", ""))
        return "".join(chunks)

    os.environ.setdefault("CLAUDE_CODE_OAUTH_TOKEN", _api_key("CLAUDE_CODE_OAUTH_TOKEN"))
    return asyncio.run(run())


def _review_openai(prompt: str, frames: list[Path], model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_api_key("OPENAI_API_KEY"))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        encoded = base64.b64encode(frame.read_bytes()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )
    response = client.chat.completions.create(
        model=model,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content or ""


def _review_anthropic(prompt: str, frames: list[Path], model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key("ANTHROPIC_API_KEY"))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(frame.read_bytes()).decode(),
                },
            }
        )
    message = client.messages.create(
        model=model,
        max_tokens=4000,
        system=REVIEWER_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"reviewer returned no JSON:\n{text[:400]}")
    return json.loads(match.group(0))


def adversarial_review(
    frames: list[Path],
    captions: list,
    previous_fixes: list[str],
    model: str,
    provider: str = "claude-sdk",
) -> dict[str, Any]:
    """One cold review of one take. Vision model, JSON verdict.

    The reviewer is never told its own previous score — only what was changed —
    so a round cannot drift upward by anchoring on the last number it produced.
    The weighted total is recomputed here from the per-dimension scores rather
    than trusted from the model: the cap is a rule, not an opinion.
    """
    backends = {
        "claude-sdk": _review_claude_sdk,
        "openai": _review_openai,
        "anthropic": _review_anthropic,
    }
    prompt = reviewer_prompt(captions, MISSION, previous_fixes)
    raw = backends[provider](prompt, frames, model)
    verdict = _extract_json(raw)
    verdict["total"] = weighted_total(verdict.get("scores", {}), capped=bool(verdict.get("capped")))
    verdict["provider"] = provider
    verdict["model"] = model
    return verdict


def apply_fixes(script_path: Path, must_fix: list[dict[str, Any]]) -> list[str]:
    """Patch ``script.json`` from the reviewer's fixes.

    Only narration text and hold duration are machine-applicable; anything
    structural (a missing shot, a wrong selector) is returned for a human,
    because silently inventing a new scene would make the next round's score
    meaningless.
    """
    script = json.loads(script_path.read_text(encoding="utf-8"))
    by_index = {i + 1: scene for i, scene in enumerate(script["scenes"])}
    applied: list[str] = []
    for fix in must_fix:
        scene = by_index.get(int(fix.get("scene", 0)))
        if scene is None:
            continue
        note = fix.get("fix", "")
        if new_vo := fix.get("new_vo"):
            # The reviewer volunteers `new_vo` even though the schema does not
            # ask for it, and an unbounded one produced a 331-char caption that
            # failed the deterministic budget for two rounds running. A fix that
            # cannot be recorded is not a fix.
            if len(new_vo) > CAPTION_BUDGET:
                applied.append(
                    f"scene {fix['scene']}: rejected {len(new_vo)}-char narration "
                    f"(budget {CAPTION_BUDGET})"
                )
                continue
            scene["vo"] = new_vo
            applied.append(f"scene {fix['scene']}: narration rewritten")
        elif hold := fix.get("new_hold_ms"):
            scene["hold_ms"] = int(hold)
            applied.append(f"scene {fix['scene']}: hold -> {hold}ms")
        else:
            applied.append(f"scene {fix['scene']}: {note} (manual)")
    script_path.write_text(json.dumps(script, indent=2) + "\n", encoding="utf-8")
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=DEMO / "script.json")
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--provider", default="claude-sdk", choices=sorted(PROVIDERS))
    parser.add_argument("--model", default=None, help="defaults to the provider's vision model")
    parser.add_argument(
        "--review-only",
        type=Path,
        default=None,
        help="score an existing <stem>.mp4 without recording anything",
    )
    args = parser.parse_args()

    key_name, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    if not _api_key(key_name):
        print(f"✗ no {key_name} — the reviewer needs a vision model", file=sys.stderr)
        return 2

    script = load_script(args.script)
    live_scenes = [s for s in script["scenes"] if not s.get("blocked")]
    log: list[dict[str, Any]] = []
    best: tuple[float, str] | None = None
    fixes_applied: list[str] = []

    for round_no in range(1, args.max_rounds + 1):
        stem = f"take_{round_no:02d}"
        print(f"\n=== round {round_no}/{args.max_rounds} — recording {stem} ===")
        if args.review_only and round_no == 1:
            mp4 = args.review_only
            stem = mp4.stem
        else:
            mp4 = record(script, stem)

        captions = parse_srt(mp4.with_suffix(".srt"))
        checks = deterministic_checks(mp4, captions, expected_scenes=len(live_scenes))
        entry: dict[str, Any] = {
            "round": round_no,
            "take": stem,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "deterministic": {"passed": checks.passed, "failures": checks.failures},
        }
        if not checks.passed:
            # Objective defects never cost a review round: fix and re-record.
            print("✗ deterministic checks failed — rebuilding without a review call")
            for failure in checks.failures:
                print(f"   {failure}")
            entry["score"] = None
            log.append(entry)
            REVIEW_LOG.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
            continue

        frames = extract_frames(mp4, captions, ARTIFACTS / f"frames_{stem}")
        print(f"  reviewing {len(frames)} frames with {args.provider}/{model} ...")
        verdict = adversarial_review(frames, captions, fixes_applied, model, args.provider)
        score = float(verdict["total"])
        entry["score"] = score
        entry["verdict"] = verdict
        log.append(entry)
        REVIEW_LOG.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

        print(f"  score {score:.2f}/10" + ("  [CAPPED]" if verdict.get("capped") else ""))
        for fix in verdict.get("must_fix", []):
            print(f"   scene {fix.get('scene')}: {fix.get('problem')}")

        if best is None or score > best[0]:
            best = (score, stem)

        if score >= args.threshold:
            print(f"\n✓ cleared {args.threshold} at round {round_no}")
            break
        if round_no < args.max_rounds:
            # The generator sees the same frames the reviewer graded, so a
            # framing complaint can be answered by changing the shot rather
            # than by rewording the caption away from the problem.
            print("  regenerating the script from the review ...")
            try:
                rewritten = regenerate(
                    script,
                    verdict,
                    frame_dir=ARTIFACTS / f"frames_{stem}",
                    model=model if args.provider == "claude-sdk" else "claude-sonnet-5",
                    oauth_token=_api_key("CLAUDE_CODE_OAUTH_TOKEN"),
                )
            except Exception as exc:
                # A rejected rewrite must not end the run: fall back to the
                # narration-only patch so the round still makes some progress.
                print(f"  ! generator rejected ({exc}); falling back to text-only fixes")
                fixes_applied = apply_fixes(args.script, verdict.get("must_fix", []))
                script = load_script(args.script)
                continue
            args.script.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
            fixes_applied = [
                f"scene {f.get('scene')}: {f.get('problem')}" for f in verdict.get("must_fix", [])
            ]
            script = rewritten
            entry["regenerated"] = True

    if best is None:
        print("\n✗ no take was ever scored", file=sys.stderr)
        return 1

    score, stem = best
    for suffix in (".mp4", ".srt"):
        source = ARTIFACTS / f"{stem}{suffix}"
        if source.exists():
            shutil.copy(source, ARTIFACTS / f"walkthrough{suffix}")
    print(f"\nshipping {stem} at {score:.2f}/10 -> walkthrough.mp4")
    print(f"scorecard: {REVIEW_LOG}")
    if score < args.threshold:
        print(
            f"NOTE: shipped below the {args.threshold} bar. That number is in "
            f"{REVIEW_LOG.name} and belongs in the write-up."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
