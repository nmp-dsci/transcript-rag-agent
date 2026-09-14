"""Write the chunks a guide will read to ``guides/<slug>/corpus/``.

The extraction agents read files, not a database: one markdown file per
video, every chunk in order with its index and timestamps, so a citation the
agent writes (``video_id`` + ``chunk_index``) is something it copied from a
heading rather than something it inferred. The directory is gitignored; it
rebuilds from Chroma in seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from src.guides.catalog import GuidePaths

#: ``chunks_for(video_ids) -> chunks`` in (video_id, chunk_index) order; each
#: chunk has ``video_id``, ``chunk_index``, ``text`` and optional ``start_seconds``
#: / ``end_seconds`` / ``title`` / ``channel_name`` / ``source_url``, as
#: attributes or keys. ``TranscriptChunkStore.chunks_for_videos`` fits.
ChunksFn = Callable[[list[str]], Iterable[Any]]


def _get(chunk: Any, name: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def _stamp(seconds: Any) -> str:
    if seconds is None:
        return "--:--"
    total = int(float(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


@dataclass
class ExportedVideo:
    video_id: str
    title: str
    channel_name: str
    path: str
    chunk_count: int


@dataclass
class ExportSummary:
    videos: list[ExportedVideo] = field(default_factory=list)
    clusters: list[list[str]] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return sum(video.chunk_count for video in self.videos)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "videos": [vars(video) for video in self.videos],
            "clusters": [list(cluster) for cluster in self.clusters],
        }


def render_video_markdown(video_id: str, chunks: list[Any]) -> str:
    first = chunks[0] if chunks else None
    title = str(_get(first, "title", None) or video_id) if first else video_id
    channel = str(_get(first, "channel_name", None) or "") if first else ""
    url = str(_get(first, "source_url", None) or f"https://www.youtube.com/watch?v={video_id}")
    lines = [
        f"# {title}",
        "",
        f"- video_id: `{video_id}`",
        f"- channel: {channel or 'unknown'}",
        f"- url: {url}",
        f"- chunks: {len(chunks)}",
        "",
        "Cite a passage as `video_id@chunk_index` — both are in each heading below.",
        "",
    ]
    for chunk in chunks:
        index = int(_get(chunk, "chunk_index", 0))
        start = _stamp(_get(chunk, "start_seconds"))
        end = _stamp(_get(chunk, "end_seconds"))
        text = str(_get(chunk, "text", "") or "").strip()
        lines.append(f"## {video_id}@{index} · {start}–{end}")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_corpus(
    paths: GuidePaths,
    video_ids: list[str],
    chunks_for: ChunksFn,
    *,
    clusters: list[list[str]],
) -> ExportSummary:
    """Write one markdown file per video plus ``corpus/INDEX.md`` and ``corpus.json``."""
    paths.corpus.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[Any]] = {video_id: [] for video_id in video_ids}
    for chunk in chunks_for(list(video_ids)):
        video_id = str(_get(chunk, "video_id", ""))
        if video_id in by_video:
            by_video[video_id].append(chunk)
    summary = ExportSummary(clusters=[list(cluster) for cluster in clusters])
    for video_id in video_ids:
        chunks = sorted(by_video[video_id], key=lambda c: int(_get(c, "chunk_index", 0)))
        target = paths.corpus / f"{video_id}.md"
        target.write_text(render_video_markdown(video_id, chunks), encoding="utf-8")
        first = chunks[0] if chunks else None
        summary.videos.append(
            ExportedVideo(
                video_id=video_id,
                title=str(_get(first, "title", None) or video_id) if first else video_id,
                channel_name=str(_get(first, "channel_name", None) or "") if first else "",
                path=f"corpus/{video_id}.md",
                chunk_count=len(chunks),
            )
        )
    _write_index(paths, summary)
    return summary


def _write_index(paths: GuidePaths, summary: ExportSummary) -> None:
    lines = ["# Corpus for this guide", ""]
    for number, cluster in enumerate(summary.clusters, 1):
        lines.append(f"## Cluster {number}")
        lines.append("")
        for video_id in cluster:
            video = next((v for v in summary.videos if v.video_id == video_id), None)
            if video is None:
                continue
            lines.append(
                f"- `{video.path}` — {video.title} ({video.channel_name or 'unknown'}, "
                f"{video.chunk_count} chunks)"
            )
        lines.append("")
    (paths.corpus / "INDEX.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (paths.corpus / "corpus.json").write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_export(paths: GuidePaths) -> ExportSummary | None:
    try:
        data = json.loads((paths.corpus / "corpus.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    summary = ExportSummary(clusters=[list(c) for c in data.get("clusters", [])])
    for video in data.get("videos", []):
        summary.videos.append(ExportedVideo(**video))
    return summary


def corpus_files(paths: GuidePaths, video_ids: list[str]) -> list[Path]:
    return [paths.corpus / f"{video_id}.md" for video_id in video_ids]
