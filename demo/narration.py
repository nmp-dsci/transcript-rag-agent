"""Shared narration pipeline: Deepgram Aura voice-over + burned captions + SRT.

The walkthrough recorder drives Playwright and calls into this module so TTS
synthesis, the caption bar, scene pacing, SRT export, and the audio mix are
defined once. One narration line per scene is spoken, shown as the caption, and
written to the SRT — a single source of truth, so the video and its subtitles
can never drift apart. Without DEEPGRAM_API_KEY the video is produced silent
(captions only), which keeps the recorder runnable in CI and for anyone cloning
the repo without a TTS key.

Ported from the sibling floor-plan-reviewer project; the caption bar is
recoloured to this app's palette (frontend/src/theme.css).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page

TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-hyperion-en")


class Narrator:
    def __init__(self, art: Path, voiceover: dict[str, str]) -> None:
        self.art = art
        self.vo = voiceover
        self.vo_dir = art / "vo"
        self.subs: list[tuple[float, str, str]] = []  # (start_s, key, text)
        self.durations: dict[str, float] = {}
        self.t0 = 0.0
        self.voiced = False

    # ---- TTS ----
    def _api_key(self) -> str:
        """Environment first, then the same env file the app itself loads.

        The sibling looked only in a repo-local ``.env``; this project keeps its
        keys in ``~/.env`` (overridable via ``YT_AGENT_ENV_PATH``, exactly as
        ``src/config.py`` resolves them), so looking only beside the source would
        silently fall back to a silent recording on a machine that has the key.
        The repo-local path is still checked last, for a clone that keeps one.
        """
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if key:
            return key
        candidates = [
            Path(os.environ.get("YT_AGENT_ENV_PATH", "~/.env")).expanduser(),
            Path(__file__).parents[1] / ".env",
        ]
        for env in candidates:
            if not env.exists():
                continue
            for line in env.read_text().splitlines():
                if line.startswith("DEEPGRAM_API_KEY="):
                    value = line.split("=", 1)[1].strip()
                    if value:
                        return value
        return ""

    def synthesize(self) -> bool:
        key = self._api_key()
        if not key:
            print("⚠ no DEEPGRAM_API_KEY — recording silent (captions only)")
            return False
        self.vo_dir.mkdir(parents=True, exist_ok=True)
        for name, text in self.vo.items():
            out = self.vo_dir / f"{name}.mp3"
            req = urllib.request.Request(
                f"https://api.deepgram.com/v1/speak?model={TTS_MODEL}&encoding=mp3",
                data=json.dumps({"text": text}).encode(),
                headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
                method="POST",
            )
            out.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    out,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.durations[name] = float(probe.stdout.strip())
            print(f"  🔊 {name}: {self.durations[name]:.1f}s")
        self.voiced = True
        return True

    # ---- timeline ----
    def start(self) -> None:
        self.t0 = time.monotonic()

    def caption(self, page: Page, key: str) -> None:
        text = self.vo[key]
        self.subs.append((time.monotonic() - self.t0, key, text))
        page.evaluate(
            """(text) => {
              let bar = document.getElementById('demo-caption');
              if (!bar) {
                bar = document.createElement('div');
                bar.id = 'demo-caption';
                Object.assign(bar.style, {
                  position: 'fixed', left: '0', right: '0', top: '80%', zIndex: '99999',
                  background: 'rgba(18,22,29,0.94)', color: '#e6edf3',
                  font: '600 16.5px/1.45 "Helvetica Neue", Arial, sans-serif',
                  padding: '13px 30px', textAlign: 'center',
                  borderTop: '3px solid #2f81f7', borderBottom: '3px solid #2f81f7',
                  pointerEvents: 'none',
                });
                document.body.appendChild(bar);
              }
              bar.textContent = text;
            }""",
            text,
        )

    def scene(self, page: Page, key: str, visual_ms: int) -> None:
        """Hold the scene at least as long as its spoken line (plus a beat)."""
        voiced_ms = int(self.durations.get(key, 0) * 1000) + 600
        page.wait_for_timeout(max(visual_ms, voiced_ms))

    # ---- outputs ----
    @staticmethod
    def _srt_time(s: float) -> str:
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    def _write_srt(self, total: float, stem: str) -> Path:
        out = self.art / f"{stem}.srt"
        blocks = []
        for i, (start, _key, text) in enumerate(self.subs):
            end = self.subs[i + 1][0] if i + 1 < len(self.subs) else total
            blocks.append(f"{i + 1}\n{self._srt_time(start)} --> {self._srt_time(end)}\n{text}\n")
        out.write_text("\n".join(blocks))
        return out

    def _mix(self, silent_mp4: Path, out_mp4: Path) -> None:
        cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent_mp4)]
        filters: list[str] = []
        for i, (start, key, _text) in enumerate(self.subs):
            cmd += ["-i", str(self.vo_dir / f"{key}.mp3")]
            filters.append(f"[{i + 1}:a]adelay={int(start * 1000)}:all=1[a{i}]")
        mixspec = "".join(f"[a{i}]" for i in range(len(self.subs)))
        filters.append(f"{mixspec}amix=inputs={len(self.subs)}:normalize=0[aout]")
        cmd += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)

    def finish(self, final_webm: Path, stem: str) -> None:
        # Playwright's recorded duration is not wall-clock: the encoder flushes
        # a variable tail after the context closes, so an identical script can
        # yield a 76s video one run and an 87s video the next. Trusting
        # `time.monotonic()` for the last caption's end therefore leaves a
        # phantom gap that reads as dead air. Measure the video instead, and
        # trim the tail so the SRT and the picture always agree.
        total = time.monotonic() - self.t0
        recorded = _probe_duration(final_webm)
        if recorded is not None and recorded > total + 0.5:
            trimmed = self.art / f"_{stem}_trimmed.webm"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(final_webm),
                    "-t",
                    f"{total:.3f}",
                    "-c",
                    "copy",
                    str(trimmed),
                ],
                check=True,
            )
            trimmed.replace(final_webm)
            print(f"  ✂ trimmed {recorded - total:.1f}s of encoder tail")
        srt = self._write_srt(total, stem)
        print(f"subtitles: {srt}")
        print(f"video: {final_webm} ({final_webm.stat().st_size // 1024} KB)")
        if not shutil.which("ffmpeg"):
            return
        mp4 = self.art / f"{stem}.mp4"
        # A voiced run with no captions would build an `amix=inputs=0` filter, so
        # the silent path also covers "narrated but nothing was ever captioned".
        if self.voiced and self.subs:
            silent = self.art / f"_{stem}_silent.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(final_webm), str(silent)],
                check=True,
            )
            self._mix(silent, mp4)
            silent.unlink()
            print(f"video: {mp4} ({mp4.stat().st_size // 1024} KB, narrated · {TTS_MODEL})")
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(final_webm), str(mp4)], check=True
            )
            print(f"video: {mp4} ({mp4.stat().st_size // 1024} KB, silent)")


def _probe_duration(media: Path) -> float | None:
    """Duration in seconds, or None when ffprobe cannot read the container."""
    try:
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
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def take_webm(art: Path, stem: str) -> Path:
    """Rename the most-recent Playwright .webm capture to <stem>.webm."""
    raw = sorted(art.glob("*.webm"), key=lambda p: p.stat().st_mtime)[-1]
    final = art / f"{stem}.webm"
    shutil.move(raw, final)
    return final
