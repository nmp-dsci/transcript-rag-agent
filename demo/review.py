"""Score a recorded take: deterministic checks first, then an adversarial read.

The reviewer never streams video. It receives **one keyframe per narrated
scene**, sampled at the midpoint of that scene's SRT caption, plus the caption
text and timings. Eleven scenes is eleven images — enough to judge composition,
legibility and whether a spoken number matches the pixels behind it, small
enough to fit one context.

Deterministic checks run first and cost nothing: a caption that overflows or a
three-second hole of silence is an objective defect, and spending a model call
to discover it is waste. A deterministic failure short-circuits to a rebuild
*without* consuming one of the five review rounds.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# A caption bar two lines deep at the recorded width holds roughly this much
# before it wraps to a third line and starts covering the UI it describes.
CAPTION_CHAR_BUDGET = 190
# Longer than this with nothing spoken reads as the video having stalled.
MAX_SILENT_GAP_S = 2.5
# Words per second for a comfortable listen; below this a caption is gone
# before it can be read.
MIN_READABLE_WPS = 1.2
MAX_COMFORTABLE_WPS = 4.2


@dataclass(frozen=True)
class Caption:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class CheckResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _secs(stamp: str) -> float:
    hours, minutes, rest = stamp.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_srt(path: Path) -> list[Caption]:
    blocks = re.findall(
        r"(\d+)\n([\d:,]+) --> ([\d:,]+)\n(.+?)(?=\n\n|\n*\Z)",
        path.read_text(encoding="utf-8"),
        re.S,
    )
    return [Caption(int(n), _secs(a), _secs(b), " ".join(text.split())) for n, a, b, text in blocks]


def probe_duration(media: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(media),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def extract_frames(video: Path, captions: list[Caption], out_dir: Path) -> list[Path]:
    """One frame per caption, at the caption's midpoint.

    Midpoint rather than start: at a caption's start the UI is often still
    settling from the previous scene's action, which would have the reviewer
    grading a transition instead of the shot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for caption in captions:
        midpoint = (caption.start + caption.end) / 2
        dst = out_dir / f"scene{caption.index:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{midpoint:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                # 1280-wide keeps every label readable while keeping five frames
                # inside the agent SDK's 1 MB message buffer, which a full-size
                # set overflowed mid-loop.
                "-vf",
                "scale=1280:-2",
                "-q:v",
                "6",
                str(dst),
            ],
            check=True,
        )
        frames.append(dst)
    return frames


def deterministic_checks(
    video: Path, captions: list[Caption], *, expected_scenes: int | None = None
) -> CheckResult:
    """Objective defects, found without a model call."""
    result = CheckResult(passed=True)
    if not captions:
        result.failures.append("no captions in the SRT — the take has no narration at all")
        result.passed = False
        return result

    if expected_scenes is not None and len(captions) != expected_scenes:
        result.failures.append(
            f"SRT has {len(captions)} captions but the script defines "
            f"{expected_scenes} scenes — a scene was dropped or duplicated"
        )

    for caption in captions:
        if len(caption.text) > CAPTION_CHAR_BUDGET:
            result.failures.append(
                f"scene {caption.index}: caption is {len(caption.text)} chars "
                f"(budget {CAPTION_CHAR_BUDGET}) — it will wrap over the UI"
            )
        words = len(caption.text.split())
        if caption.duration > 0:
            wps = words / caption.duration
            if wps > MAX_COMFORTABLE_WPS:
                result.failures.append(
                    f"scene {caption.index}: {wps:.1f} words/sec — too fast to follow"
                )
            elif wps < MIN_READABLE_WPS:
                result.warnings.append(f"scene {caption.index}: {wps:.1f} words/sec — drags")

    for previous, nxt in zip(captions, captions[1:]):
        gap = nxt.start - previous.end
        if gap > MAX_SILENT_GAP_S:
            result.failures.append(f"scenes {previous.index}->{nxt.index}: {gap:.1f}s of dead air")

    if captions[0].start > MAX_SILENT_GAP_S:
        result.failures.append(f"{captions[0].start:.1f}s of dead air before scene 1")

    duration = probe_duration(video)
    tail = duration - captions[-1].end
    if tail > MAX_SILENT_GAP_S:
        result.failures.append(f"{tail:.1f}s of dead air after the last caption")
    result.warnings.append(f"runtime {duration:.1f}s")

    result.passed = not result.failures
    return result


RUBRIC = [
    (
        "claim_integrity",
        3,
        "Does every spoken number, name and claim match what is visible in that "
        "scene's frame? An unverifiable claim is the worst defect available.",
    ),
    ("hook", 2, "By the end of scene 2, does a viewer know what this does FOR THEM?"),
    (
        "legibility",
        2,
        "Is every element the narration refers to actually on screen, readable, "
        "and unobscured by the caption bar?",
    ),
    ("pacing", 1, "Dead air, rushed cuts, or captions gone before they can be read."),
    ("arc", 1, "Does it land Itch -> Build -> Proof, or is it a feature tour?"),
    (
        "production",
        1,
        "Spinners, empty states, error text, dev artefacts, overlapping chrome, "
        "stale counts that disagree with the app header.",
    ),
]

REVIEWER_SYSTEM = """You are reviewing a portfolio walkthrough video for a \
senior/principal AI engineer. You are not the author's colleague; you are the \
hiring engineer looking for the strongest reason to dismiss this candidate.

Find that reason. Default to a LOWER score when uncertain. A video that is \
merely competent is a 6, not an 8.

Rules you must follow:
- Every criticism MUST cite a scene number and quote the caption text or \
  describe what is visible in that frame. A criticism you cannot ground in a \
  specific frame is inadmissible — drop it rather than assert it.
- CLAIM INTEGRITY IS ABSOLUTE. If any spoken number, count or name cannot be \
  confirmed against the pixels of its own frame, cap `total` at 6.0 no matter \
  how strong the rest is. Say which claim and which frame.
- Do not praise. The `must_fix` list is the deliverable.
"""


def reviewer_prompt(captions: list[Caption], mission: str, previous_fixes: list[str]) -> str:
    lines = [
        "MISSION — what this video has to accomplish:",
        mission,
        "",
        "RUBRIC — score each 0-10; total is the weighted mean:",
    ]
    lines += [f"  {name} (weight x{w}): {desc}" for name, w, desc in RUBRIC]
    lines += ["", "NARRATION, scene by scene (frames attached in order):"]
    lines += [f"  scene {c.index} [{c.start:.1f}-{c.end:.1f}s]: {c.text}" for c in captions]
    if previous_fixes:
        lines += [
            "",
            "Fixes applied since the last take (judge the result, not the effort;",
            "you are NOT told what the previous score was):",
        ]
        lines += [f"  - {fix}" for fix in previous_fixes]
    lines += [
        "",
        "Return JSON only:",
        '{"scores": {"claim_integrity": 0-10, "hook": 0-10, "legibility": 0-10,',
        '  "pacing": 0-10, "arc": 0-10, "production": 0-10},',
        ' "total": 0-10, "capped": bool, "cap_reason": str|null,',
        ' "must_fix": [{"scene": int, "problem": str, "evidence": str, "fix": str}],',
        ' "verdict": str}',
    ]
    return "\n".join(lines)


def weighted_total(scores: dict[str, float], *, capped: bool) -> float:
    total = sum(scores.get(name, 0) * weight for name, weight, _ in RUBRIC)
    total /= sum(weight for _, weight, _ in RUBRIC)
    return min(total, 6.0) if capped else round(total, 2)


def load_verdict(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
