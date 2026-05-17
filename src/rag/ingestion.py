from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.transcripts.discovery import DiscoveredVideo


@dataclass
class IngestionCandidateRecord:
    video_id: str
    source_url: str
    title: str | None = None
    channel_name: str | None = None
    published_at: str | None = None
    outcome: str | None = None
    error: str | None = None
    chunk_count: int | None = None
    duration_seconds: float | None = None


@dataclass
class IngestionRunRecord:
    run_id: str
    label: str | None
    mode: str
    query: str | None
    channel: str | None
    since: str | None
    until: str | None
    started_at: str
    completed_at: str | None = None
    status: str = "running"
    stage: str | None = None
    candidate_count: int = 0
    indexed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    candidates: list[IngestionCandidateRecord] = field(default_factory=list)
    error: str | None = None

    def complete(self) -> None:
        self.completed_at = _now_iso()
        self.candidate_count = len(self.candidates)
        self.indexed_count = sum(
            1
            for candidate in self.candidates
            if candidate.outcome in {"indexed", "summary_refreshed"}
        )
        self.skipped_count = sum(
            1 for candidate in self.candidates if candidate.outcome == "skipped_existing"
        )
        self.failed_count = sum(
            1 for candidate in self.candidates if candidate.outcome == "failed"
        )
        self.status = "failed" if self.error or self.failed_count else "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "mode": self.mode,
            "query": self.query,
            "channel": self.channel,
            "since": self.since,
            "until": self.until,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "stage": self.stage,
            "candidate_count": self.candidate_count,
            "indexed_count": self.indexed_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "error": self.error,
            "candidates": [candidate.__dict__ for candidate in self.candidates],
        }


def start_ingestion_run(
    *,
    mode: str,
    label: str | None = None,
    query: str | None = None,
    channel: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> IngestionRunRecord:
    started_at = _now_iso()
    basis = label or query or channel or mode
    run_id = f"{_filename_timestamp(started_at)}__{mode}__{_slug(basis)}"
    return IngestionRunRecord(
        run_id=run_id,
        label=label,
        mode=mode,
        query=query,
        channel=channel,
        since=since,
        until=until,
        started_at=started_at,
    )


def candidate_record(video: DiscoveredVideo) -> IngestionCandidateRecord:
    return IngestionCandidateRecord(
        video_id=video.video_id,
        source_url=str(video.source_url),
        title=video.title,
        channel_name=video.channel_name,
        published_at=video.published_at,
        duration_seconds=video.duration_seconds,
    )


def write_ingestion_run(record: IngestionRunRecord, directory: Path | str) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"{record.run_id}.json"
    output.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def load_ingestion_runs(directory: Path | str) -> list[dict[str, Any]]:
    path = Path(directory)
    if not path.exists():
        return []
    runs: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("*.json"), reverse=True):
        try:
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            runs.append(loaded)
    return sorted(runs, key=lambda item: str(item.get("started_at", "")), reverse=True)


def ingestion_runs_dir(chroma_path: Path) -> Path:
    return chroma_path.parent / "ingestion_runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _filename_timestamp(value: str) -> str:
    return value.replace(":", "-").replace("+00:00", "Z")


def _slug(value: str | None) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value or "run").strip("-").lower()
    return text[:48] or "run"
