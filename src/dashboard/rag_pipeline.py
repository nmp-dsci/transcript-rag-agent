from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.config import ConfigError, Settings, load_settings
from src.dashboard.theme import dark_style_block
from src.rag.embeddings import HuggingFaceEmbeddingModel, cosine_similarity


DEFAULT_FILTER_TEST_QUESTION = (
    "can you summarise the impact of negative gearing, capital gains tax on the "
    "property market, how does impact the long terms trends of property prices "
    "and what type of properties are winners and losers"
)


@dataclass(frozen=True)
class TranscriptDashboardRow:
    transcript_id: str
    video_id: str
    source_url: str
    title: str | None
    description: str | None
    channel_name: str | None
    channel_id: str | None
    language: str | None
    provider: str
    upload_date: str | None
    duration_seconds: float | None
    view_count: int | None
    like_count: int | None
    thumbnail_url: str | None
    tags: list[str]
    transcript_languages: list[str]
    fetched_at: str
    segment_count: int
    transcript_chars: int
    summary: str | None
    summary_model: str | None
    summary_generated_at: str | None
    raw_embedding_dim: int
    raw_embedding_preview: str
    summary_embedding: list[float]
    summary_embedding_dim: int
    summary_embedding_preview: str
    summary_embedding_model: str | None
    summary_embedded_at: str | None
    chunk_count: int
    total_chunk_chars: int
    avg_chunk_chars: int
    first_start_seconds: float | None
    last_end_seconds: float | None
    transcript_text: str
    chunks: list[dict[str, Any]]


@dataclass(frozen=True)
class FilterTestRow:
    rank: int
    video_id: str
    title: str | None
    channel_name: str | None
    source_url: str
    cosine_similarity: float
    chroma_score: float | None
    passes_threshold: bool
    selected_by_chroma: bool
    summary: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-pipeline-dashboard",
        description="Render an HTML dashboard of indexed transcripts, summaries, embeddings, and chunks.",
    )
    parser.add_argument("--output", type=Path, default=Path("dashboard/rag_pipeline.html"))
    parser.add_argument("--filter-test-question", default=DEFAULT_FILTER_TEST_QUESTION)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(require_keys=False)
        rows = collect_pipeline_rows(settings)
        filter_test_rows = collect_filter_test_rows(
            settings,
            rows,
            args.filter_test_question,
        )
    except (ConfigError, Exception) as exc:
        parser.exit(1, f"Error: {exc}\n")

    write_dashboard(
        output=args.output,
        rows=rows,
        settings=settings,
        filter_test_question=args.filter_test_question,
        filter_test_rows=filter_test_rows,
    )
    print(f"Wrote {args.output}")
    return 0


def write_dashboard(
    output: Path,
    rows: list[TranscriptDashboardRow],
    settings: Settings,
    filter_test_question: str = DEFAULT_FILTER_TEST_QUESTION,
    filter_test_rows: list[FilterTestRow] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(rows, settings, filter_test_question, filter_test_rows),
        encoding="utf-8",
    )


def collect_pipeline_rows(settings: Settings) -> list[TranscriptDashboardRow]:
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    raw_collection = client.get_or_create_collection(settings.raw_transcript_collection)
    chunk_collection = client.get_or_create_collection(settings.chunk_collection)
    summary_collection = client.get_or_create_collection(
        settings.transcript_summary_collection
    )

    raw_result = raw_collection.get(include=["documents", "metadatas"])
    chunk_result = chunk_collection.get(include=["documents", "metadatas"])
    summary_result = summary_collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    chunks_by_video = _chunks_by_video(chunk_result)
    summaries_by_video = _summaries_by_video(summary_result)
    rows: list[TranscriptDashboardRow] = []

    documents = raw_result.get("documents") or []
    metadatas = raw_result.get("metadatas") or []
    for index, document_text in enumerate(documents):
        metadata = metadatas[index] or {}
        body = _json_loads(document_text)
        video_id = str(metadata.get("video_id", ""))
        segments = body.get("segments", [])
        transcript_text = " ".join(str(segment.get("text", "")) for segment in segments)
        raw_embedding = body.get("summary_embedding") or []
        summary = summaries_by_video.get(video_id, {})
        summary_embedding = summary.get("embedding") or []
        chunks = chunks_by_video.get(video_id, [])
        total_chunk_chars = sum(len(str(chunk.get("text", ""))) for chunk in chunks)
        start_values = [
            float(chunk["start_seconds"])
            for chunk in chunks
            if chunk.get("start_seconds") is not None
        ]
        end_values = [
            float(chunk["end_seconds"])
            for chunk in chunks
            if chunk.get("end_seconds") is not None
        ]
        rows.append(
            TranscriptDashboardRow(
                transcript_id=str(metadata.get("transcript_id", "")),
                video_id=video_id,
                source_url=str(metadata.get("source_url", "")),
                title=_none_if_empty(metadata.get("title")),
                description=body.get("description")
                or _none_if_empty(metadata.get("description")),
                channel_name=_none_if_empty(metadata.get("channel_name")),
                channel_id=_none_if_empty(metadata.get("channel_id")),
                language=_none_if_empty(metadata.get("language")),
                provider=str(metadata.get("provider", "")),
                upload_date=_none_if_empty(metadata.get("upload_date")),
                duration_seconds=_float_or_none(metadata.get("duration_seconds")),
                view_count=_int_or_none(metadata.get("view_count")),
                like_count=_int_or_none(metadata.get("like_count")),
                thumbnail_url=_none_if_empty(metadata.get("thumbnail_url")),
                tags=[str(tag) for tag in body.get("tags", [])],
                transcript_languages=[
                    str(lang) for lang in body.get("transcript_languages", [])
                ],
                fetched_at=str(metadata.get("fetched_at", "")),
                segment_count=int(metadata.get("segment_count", len(segments))),
                transcript_chars=len(transcript_text),
                summary=_none_if_empty(metadata.get("summary"))
                or _none_if_empty(summary.get("summary")),
                summary_model=_none_if_empty(metadata.get("summary_model"))
                or _none_if_empty(summary.get("summary_model")),
                summary_generated_at=_none_if_empty(
                    metadata.get("summary_generated_at")
                )
                or _none_if_empty(summary.get("summary_generated_at")),
                raw_embedding_dim=len(raw_embedding),
                raw_embedding_preview=_embedding_preview(raw_embedding),
                summary_embedding=[float(value) for value in summary_embedding],
                summary_embedding_dim=len(summary_embedding),
                summary_embedding_preview=_embedding_preview(summary_embedding),
                summary_embedding_model=_none_if_empty(
                    metadata.get("summary_embedding_model")
                )
                or _none_if_empty(summary.get("summary_embedding_model")),
                summary_embedded_at=_none_if_empty(metadata.get("summary_embedded_at"))
                or _none_if_empty(summary.get("summary_embedded_at")),
                chunk_count=len(chunks),
                total_chunk_chars=total_chunk_chars,
                avg_chunk_chars=(total_chunk_chars // len(chunks)) if chunks else 0,
                first_start_seconds=min(start_values) if start_values else None,
                last_end_seconds=max(end_values) if end_values else None,
                transcript_text=transcript_text,
                chunks=chunks,
            )
        )
    return sorted(rows, key=lambda row: row.video_id)


def collect_filter_test_rows(
    settings: Settings,
    rows: list[TranscriptDashboardRow],
    question: str,
) -> list[FilterTestRow]:
    if not rows:
        return []
    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    question_embedding = embedding_model.embed_query(question)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    summary_collection = client.get_or_create_collection(
        settings.transcript_summary_collection
    )
    chroma_result = summary_collection.query(
        query_embeddings=[question_embedding],
        n_results=max(len(rows), 1),
        include=["metadatas", "distances"],
    )
    chroma_scores: dict[str, float] = {}
    chroma_ids = (chroma_result.get("ids") or [[]])[0]
    chroma_metadatas = (chroma_result.get("metadatas") or [[]])[0]
    chroma_distances = (chroma_result.get("distances") or [[]])[0]
    for index, _summary_id in enumerate(chroma_ids):
        metadata = chroma_metadatas[index] or {}
        distance = chroma_distances[index] if index < len(chroma_distances) else None
        if distance is None:
            continue
        chroma_scores[str(metadata.get("video_id", ""))] = 1.0 - float(distance)

    scored: list[FilterTestRow] = []
    for row in rows:
        embedding = _embedding_from_preview_source(row)
        similarity = cosine_similarity(question_embedding, embedding)
        chroma_score = chroma_scores.get(row.video_id)
        scored.append(
            FilterTestRow(
                rank=0,
                video_id=row.video_id,
                title=row.title,
                channel_name=row.channel_name,
                source_url=row.source_url,
                cosine_similarity=similarity,
                chroma_score=chroma_score,
                passes_threshold=similarity >= settings.transcript_filter_min_score,
                selected_by_chroma=False,
                summary=row.summary,
            )
        )
    scored.sort(key=lambda item: item.cosine_similarity, reverse=True)
    ranked: list[FilterTestRow] = []
    for rank, row in enumerate(scored, 1):
        ranked.append(row.__class__(**{**row.__dict__, "rank": rank}))
    selected_video_ids = set()
    for row in sorted(
        ranked,
        key=lambda item: item.cosine_similarity,
        reverse=True,
    )[: settings.transcript_filter_top_k]:
        if row.passes_threshold:
            selected_video_ids.add(row.video_id)
    return [
        row.__class__(
            **{**row.__dict__, "selected_by_chroma": row.video_id in selected_video_ids}
        )
        for row in ranked
    ]


def render_html(
    rows: list[TranscriptDashboardRow],
    settings: Settings,
    filter_test_question: str = DEFAULT_FILTER_TEST_QUESTION,
    filter_test_rows: list[FilterTestRow] | None = None,
) -> str:
    filter_test_rows = filter_test_rows or []
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>RAG Pipeline Dashboard</title>",
            *dark_style_block(),
            "<script>",
            "function showTab(id){document.querySelectorAll('.tab,.panel').forEach(e=>e.classList.remove('active'));document.getElementById('tab-'+id).classList.add('active');document.getElementById('panel-'+id).classList.add('active')}",
            "</script>",
            "</head>",
            "<body>",
            "<header><h1>RAG Pipeline Dashboard</h1></header>",
            "<main>",
            _metrics(rows, settings),
            '<div class="tabs">',
            '<button class="tab active" id="tab-transcripts" onclick="showTab(\'transcripts\')">All Transcripts</button>',
            '<button class="tab" id="tab-filter-test" onclick="showTab(\'filter-test\')">Filter Test</button>',
            '<button class="tab" id="tab-chunks" onclick="showTab(\'chunks\')">Chunks</button>',
            '<button class="tab" id="tab-config" onclick="showTab(\'config\')">Config</button>',
            "</div>",
            '<section class="panel active" id="panel-transcripts">',
            _transcripts_table(rows),
            "</section>",
            '<section class="panel" id="panel-filter-test">',
            _filter_test_panel(filter_test_question, filter_test_rows, settings),
            "</section>",
            '<section class="panel" id="panel-chunks">',
            _chunks_table(rows),
            "</section>",
            '<section class="panel" id="panel-config">',
            _config_table(settings),
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _metrics(rows: list[TranscriptDashboardRow], settings: Settings) -> str:
    transcript_count = len(rows)
    chunk_count = sum(row.chunk_count for row in rows)
    summary_count = sum(1 for row in rows if row.summary)
    embedded_count = sum(1 for row in rows if row.summary_embedding_dim)
    return (
        '<div class="metric-grid">'
        f'<div class="metric"><span>Transcripts</span><strong>{transcript_count}</strong></div>'
        f'<div class="metric"><span>Chunks</span><strong>{chunk_count}</strong></div>'
        f'<div class="metric"><span>Summaries</span><strong>{summary_count}</strong></div>'
        f'<div class="metric"><span>Summary vectors</span><strong>{embedded_count}</strong></div>'
        "</div>"
        f"<p><strong>Chroma path:</strong> <code>{html.escape(str(settings.chroma_path))}</code></p>"
    )


def _transcripts_table(rows: list[TranscriptDashboardRow]) -> str:
    if not rows:
        return "<p>No raw transcripts found. Run <code>uv run python -m src.cli index-rag URL</code>.</p>"
    headers = [
        "Video",
        "Source",
        "Title",
        "Channel",
        "Metadata",
        "Summary",
        "Summary Model",
        "Summary Encoding",
        "Chunks",
        "Transcript",
    ]
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><code>{html.escape(row.video_id)}</code></td>"
            f'<td><a href="{html.escape(row.source_url)}">{html.escape(row.source_url)}</a></td>'
            f"<td>{html.escape(row.title or '')}</td>"
            f"<td>{html.escape(row.channel_name or '')}<br><code>{html.escape(row.channel_id or '')}</code></td>"
            "<td>"
            f"uploaded={html.escape(row.upload_date or '')}<br>"
            f"duration={_format_duration(row.duration_seconds)}<br>"
            f"views={_format_int(row.view_count)}<br>"
            f"likes={_format_int(row.like_count)}<br>"
            f"transcript langs={html.escape(', '.join(row.transcript_languages))}<br>"
            f"tags={html.escape(', '.join(row.tags[:8]))}"
            f"{_description_details(row.description)}"
            "</td>"
            f'<td class="summary">{html.escape(row.summary or "Missing summary")}</td>'
            f"<td>{html.escape(row.summary_model or '')}<br><code>{html.escape(row.summary_generated_at or '')}</code></td>"
            "<td>"
            f"raw dim={row.raw_embedding_dim}<br>"
            f"summary dim={row.summary_embedding_dim}<br>"
            f"model={html.escape(row.summary_embedding_model or '')}<br>"
            f"<code>{html.escape(row.summary_embedding_preview or row.raw_embedding_preview)}</code>"
            "</td>"
            f"<td>{row.chunk_count}<br>avg chars={row.avg_chunk_chars}<br>total chars={row.total_chunk_chars}</td>"
            "<td>"
            f"segments={row.segment_count}<br>chars={row.transcript_chars}"
            f"<details><summary>Review transcript</summary><pre>{html.escape(row.transcript_text)}</pre></details>"
            "</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _chunks_table(rows: list[TranscriptDashboardRow]) -> str:
    chunk_rows = []
    for row in rows:
        for chunk in row.chunks:
            chunk_rows.append(
                "<tr>"
                f"<td><code>{html.escape(row.video_id)}</code></td>"
                f"<td>{chunk.get('chunk_index', '')}</td>"
                f"<td>{html.escape(str(chunk.get('start_seconds', '')))}-{html.escape(str(chunk.get('end_seconds', '')))}</td>"
                f"<td>{len(str(chunk.get('text', '')))}</td>"
                f"<td><pre>{html.escape(str(chunk.get('text', '')))}</pre></td>"
                "</tr>"
            )
    if not chunk_rows:
        return "<p>No chunks found.</p>"
    return (
        "<table><thead><tr><th>Video</th><th>Chunk</th><th>Time</th>"
        "<th>Chars</th><th>Text</th></tr></thead><tbody>"
        + "".join(chunk_rows)
        + "</tbody></table>"
    )


def _filter_test_panel(
    question: str,
    rows: list[FilterTestRow],
    settings: Settings,
) -> str:
    if not rows:
        return "<p>No transcript summary vectors found for filter testing.</p>"
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{row.rank}</td>"
            f"<td><code>{html.escape(row.video_id)}</code></td>"
            f"<td>{html.escape(row.title or '')}</td>"
            f"<td>{html.escape(row.channel_name or '')}</td>"
            f"<td class=\"metric\">{row.cosine_similarity:.4f}</td>"
            f"<td class=\"metric\">{'' if row.chroma_score is None else f'{row.chroma_score:.4f}'}</td>"
            f"<td>{'yes' if row.passes_threshold else 'no'}</td>"
            f"<td>{'yes' if row.selected_by_chroma else 'no'}</td>"
            f"<td>{html.escape(row.summary or '')}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<h2>Transcript Summary Filter Test</h2>",
            f"<p><strong>Question:</strong> {html.escape(question)}</p>",
            f"<p><strong>Threshold:</strong> <code>{settings.transcript_filter_min_score}</code> "
            f"<strong>Filter top K:</strong> <code>{settings.transcript_filter_top_k}</code></p>",
            "<p>Cosine similarity is computed directly between the question embedding and each stored transcript summary embedding. The current retrieval filter now applies the threshold to this cosine score. Chroma score is shown separately as <code>1 - distance</code> for diagnosing distance-metric mismatches.</p>",
            "<table><thead><tr>"
            "<th>Rank</th><th>Video</th><th>Title</th><th>Channel</th>"
            "<th>Cosine similarity</th><th>Chroma score</th>"
            "<th>Passes threshold</th><th>Selected by current filter</th><th>Summary</th>"
            "</tr></thead><tbody>",
            "".join(table_rows),
            "</tbody></table>",
        ]
    )


def _config_table(settings: Settings) -> str:
    values = {
        "raw_transcript_collection": settings.raw_transcript_collection,
        "chunk_collection": settings.chunk_collection,
        "transcript_summary_collection": settings.transcript_summary_collection,
        "embedding_model": settings.embedding_model,
        "rag_top_k": settings.rag_top_k,
        "transcript_filter_top_k": settings.transcript_filter_top_k,
        "transcript_filter_min_score": settings.transcript_filter_min_score,
    }
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in values.items()
    )
    return f"<table>{rows}</table>"


def _format_duration(value: float | None) -> str:
    if value is None:
        return ""
    total = int(value)
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_int(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value:,}"


def _description_details(description: str | None) -> str:
    if not description:
        return ""
    return (
        "<details><summary>Description</summary>"
        f"<pre>{html.escape(description)}</pre>"
        "</details>"
    )


def _chunks_by_video(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    for index, text in enumerate(documents):
        metadata = dict(metadatas[index] or {})
        metadata["text"] = text
        video_id = str(metadata.get("video_id", ""))
        grouped[video_id].append(metadata)
    for chunks in grouped.values():
        chunks.sort(key=lambda chunk: int(chunk.get("chunk_index", 0)))
    return grouped


def _summaries_by_video(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    raw_embeddings = result.get("embeddings")
    embeddings = raw_embeddings if raw_embeddings is not None else []
    summaries: dict[str, dict[str, Any]] = {}
    for index, summary in enumerate(documents):
        metadata = dict(metadatas[index] or {})
        embedding = embeddings[index] if index < len(embeddings) else []
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        metadata["summary"] = summary
        metadata["embedding"] = list(embedding)
        summaries[str(metadata.get("video_id", ""))] = metadata
    return summaries


def _embedding_from_preview_source(row: TranscriptDashboardRow) -> list[float]:
    return row.summary_embedding


def _embedding_preview(embedding: list[float], limit: int = 8) -> str:
    if not embedding:
        return ""
    return ", ".join(f"{float(value):.4f}" for value in embedding[:limit])


def _json_loads(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
