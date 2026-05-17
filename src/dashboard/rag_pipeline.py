from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from src.config import ConfigError, Settings, load_settings
from src.dashboard.chunk_space import (
    fit_chunk_projection,
    nearest_chunks_for_question,
    projection_from_json,
    projection_to_json,
    transform_question,
    write_projection_artifact,
)
from src.dashboard.theme import dark_style_block
from src.rag.embeddings import HuggingFaceEmbeddingModel, cosine_similarity
from src.rag.ingestion import ingestion_runs_dir, load_ingestion_runs


DEFAULT_FILTER_TEST_QUESTION = (
    "can you summarise the impact of negative gearing, capital gains tax on the "
    "property market, how does impact the long terms trends of property prices "
    "and what type of properties are winners and losers"
)
DEFAULT_CHUNK_SPACE_QUESTION = (
    "what does this video say for capital gains tax, is it being grandfathered "
    "or every now under new rules, does that mean if I sell before 30 June 2027 "
    "I can still access 50% discount"
)
MAX_CHUNK_SPACE_NEAREST = 25


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
    parser.add_argument("--question", default=DEFAULT_CHUNK_SPACE_QUESTION)
    parser.add_argument("--refresh-projection", action="store_true")
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
        chunk_space = collect_chunk_space_data(
            settings,
            rows,
            question=args.question,
            output_dir=args.output.parent / "chunk_space",
            refresh_projection=args.refresh_projection,
        )
    except (ConfigError, Exception) as exc:
        parser.exit(1, f"Error: {exc}\n")

    write_dashboard(
        output=args.output,
        rows=rows,
        settings=settings,
        filter_test_question=args.filter_test_question,
        filter_test_rows=filter_test_rows,
        chunk_space=chunk_space,
    )
    print(f"Wrote {args.output}")
    return 0


def write_dashboard(
    output: Path,
    rows: list[TranscriptDashboardRow],
    settings: Settings,
    filter_test_question: str = DEFAULT_FILTER_TEST_QUESTION,
    filter_test_rows: list[FilterTestRow] | None = None,
    chunk_space: dict[str, Any] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = load_ingestion_runs(ingestion_runs_dir(settings.chroma_path))
    if chunk_space is None:
        chunk_space = collect_chunk_space_data(
            settings,
            rows,
            question=DEFAULT_CHUNK_SPACE_QUESTION,
            output_dir=output.parent / "chunk_space",
            refresh_projection=False,
        )
    output.write_text(
        render_html(
            rows,
            settings,
            filter_test_question,
            filter_test_rows,
            ingestion_runs=runs,
            chunk_space=chunk_space,
        ),
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


def collect_chunk_space_data(
    settings: Settings,
    rows: list[TranscriptDashboardRow],
    question: str,
    output_dir: Path,
    refresh_projection: bool = False,
) -> dict[str, Any]:
    chunks = _collect_chunk_embeddings(settings, rows)
    if len(chunks) < 2:
        return {"question": question, "chunks": [], "nearest": [], "message": "At least two embedded chunks are required for PCA."}

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    embeddings = np.asarray([chunk["embedding"] for chunk in chunks], dtype=float)
    projection_path = output_dir / "projection.json"
    projection = None
    if not refresh_projection and projection_path.exists():
        try:
            projection = projection_from_json(json.loads(projection_path.read_text(encoding="utf-8")))
            if projection.n_chunks != len(chunks):
                projection = None
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            projection = None
    if projection is None:
        projection = fit_chunk_projection(embeddings, chunk_ids, settings.embedding_model)
        write_projection_artifact(projection_path, projection)

    coords_by_id = {chunk_id: (x, y) for chunk_id, x, y in projection.chunk_coords}
    embedding_model = HuggingFaceEmbeddingModel(settings.embedding_model)
    question_embedding = np.asarray(embedding_model.embed_query(question), dtype=float)
    question_coords = transform_question(projection, question_embedding)
    nearest = nearest_chunks_for_question(
        question_embedding,
        embeddings,
        chunk_ids,
        top_k=min(MAX_CHUNK_SPACE_NEAREST, len(chunk_ids)),
    )
    nearest_by_id = {item.chunk_id: item.score for item in nearest}
    chunk_payload = []
    for chunk in chunks:
        x, y = coords_by_id.get(chunk["chunk_id"], (0.0, 0.0))
        chunk_payload.append(
            {
                **{key: value for key, value in chunk.items() if key != "embedding"},
                "x": x,
                "y": y,
                "score": nearest_by_id.get(chunk["chunk_id"]),
            }
        )
    nearest_payload = [
        {
            **next(item for item in chunk_payload if item["chunk_id"] == nearest_item.chunk_id),
            "score": nearest_item.score,
        }
        for nearest_item in nearest
    ]
    question_payload = {
        "question": question,
        "embedding": question_embedding.astype(float).tolist(),
        "x": question_coords[0],
        "y": question_coords[1],
        "nearest": nearest_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "question.json").write_text(
        json.dumps(question_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "question": question,
        "projection": projection_to_json(projection),
        "question_point": question_payload,
        "chunks": chunk_payload,
        "nearest": nearest_payload,
        "message": None,
    }


def render_html(
    rows: list[TranscriptDashboardRow],
    settings: Settings,
    filter_test_question: str = DEFAULT_FILTER_TEST_QUESTION,
    filter_test_rows: list[FilterTestRow] | None = None,
    ingestion_runs: list[dict[str, Any]] | None = None,
    chunk_space: dict[str, Any] | None = None,
) -> str:
    filter_test_rows = filter_test_rows or []
    ingestion_runs = ingestion_runs or []
    chunk_space = chunk_space or {"question": DEFAULT_CHUNK_SPACE_QUESTION, "chunks": [], "nearest": []}
    ingestion_tab = (
        ['<button class="tab" id="tab-ingestion" onclick="showTab(\'ingestion\')">Ingestion Runs</button>']
        if ingestion_runs
        else []
    )
    ingestion_panel = (
        [
            '<section class="panel" id="panel-ingestion">',
            _ingestion_runs_panel(ingestion_runs),
            "</section>",
        ]
        if ingestion_runs
        else []
    )
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
            _filter_script(),
            "</script>",
            "</head>",
            "<body>",
            "<header><h1>RAG Pipeline Dashboard</h1></header>",
            "<main>",
            _metrics(rows, settings),
            '<div class="tabs">',
            '<button class="tab active" id="tab-transcripts" onclick="showTab(\'transcripts\')">Transcripts</button>',
            *ingestion_tab,
            '<button class="tab" id="tab-chunk-space" onclick="showTab(\'chunk-space\')">Chunk Space</button>',
            "</div>",
            '<section class="panel active" id="panel-transcripts">',
            _transcripts_table(rows),
            _filter_test_panel(filter_test_question, filter_test_rows, settings),
            _chunks_table(rows),
            _config_table(settings),
            "</section>",
            *ingestion_panel,
            '<section class="panel" id="panel-chunk-space">',
            _chunk_space_panel(chunk_space),
            "</section>",
            "</main>",
            f'<script type="application/json" id="chunk-space-data">{_json_script_payload(chunk_space)}</script>',
            _chunk_space_script(),
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
    channels = sorted({row.channel_name for row in rows if row.channel_name})
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
            f"<td>{_link_or_details(row.source_url)}</td>"
            f"<td>{_details_if_long(row.title or '')}</td>"
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
            f'<td class="summary">{_details_if_long(row.summary or "Missing summary")}</td>'
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
    channel_options = "".join(
        f"<option>{html.escape(channel)}</option>" for channel in channels
    )
    return "\n".join(
        [
            '<div class="filters">',
            '<label>Title <input data-filter-table="transcripts-table" data-filter-col="2" oninput="filterTable(\'transcripts-table\')"></label>',
            '<label>Channel <select data-filter-table="transcripts-table" data-filter-col="3" onchange="filterTable(\'transcripts-table\')"><option value="">all</option>',
            channel_options,
            "</select></label>",
            "</div>",
            '<table id="transcripts-table" class="transcripts-table">',
            "<colgroup><col><col><col><col><col><col><col><col><col><col></colgroup>",
            f"<thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>",
            f"<tbody>{''.join(body)}</tbody></table>",
        ]
    )


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


def _ingestion_runs_panel(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "<h2>Ingestion Runs</h2><p>No ingestion run records found.</p>"
    rows = []
    for run in runs:
        candidates = run.get("candidates") if isinstance(run.get("candidates"), list) else []
        counts = (
            f"{run.get('candidate_count', 0)} / {run.get('indexed_count', 0)} / "
            f"{run.get('skipped_count', 0)} / {run.get('failed_count', 0)}"
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(run.get('run_id', '')))}</code><br>{html.escape(str(run.get('label') or ''))}</td>"
            f"<td>{html.escape(str(run.get('mode', '')))}</td>"
            f"<td>{html.escape(str(run.get('query') or run.get('channel') or ''))}<br>{html.escape(str(run.get('since') or ''))} {html.escape(str(run.get('until') or ''))}</td>"
            f"<td>{html.escape(str(run.get('started_at', '')))}</td>"
            f"<td class=\"num\">{_duration_between(run.get('started_at'), run.get('completed_at'))}</td>"
            f"<td>{html.escape(str(run.get('status', '')))}</td>"
            f"<td class=\"num\">{counts}</td>"
            f"<td>{_candidate_details(candidates)}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<h2>Ingestion Runs</h2>",
            '<div class="filters"><label>Text <input data-filter-table="runs" data-filter-col="0" oninput="filterTable(\'runs\')"></label><label>Mode <select data-filter-table="runs" data-filter-col="1" onchange="filterTable(\'runs\')"><option value="">all</option><option>channel</option><option>search</option><option>single</option></select></label><label>Status <select data-filter-table="runs" data-filter-col="5" onchange="filterTable(\'runs\')"><option value="">all</option><option>completed</option><option>failed</option></select></label></div>',
            '<table id="runs"><thead><tr><th>Run ID / label</th><th>Mode</th><th>Query or channel</th><th>Started at</th><th>Duration</th><th>Status</th><th>Discovered / indexed / skipped / failed</th><th>Candidates</th></tr></thead><tbody>',
            "".join(rows),
            "</tbody></table>",
        ]
    )


def _candidate_details(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""
    body = []
    for candidate in candidates:
        failed = " class=\"failed\"" if candidate.get("outcome") == "failed" else ""
        body.append(
            f"<tr{failed}>"
            f"<td><code>{html.escape(str(candidate.get('video_id', '')))}</code></td>"
            f"<td>{html.escape(str(candidate.get('outcome') or ''))}</td>"
            f"<td>{html.escape(str(candidate.get('title') or ''))}</td>"
            f"<td>{html.escape(str(candidate.get('channel_name') or ''))}</td>"
            f"<td>{html.escape(str(candidate.get('published_at') or ''))}</td>"
            f"<td>{html.escape(str(candidate.get('error') or ''))}</td>"
            "</tr>"
        )
    return (
        "<details><summary>Review candidates</summary><table><thead><tr>"
        "<th>Video</th><th>Outcome</th><th>Title</th><th>Channel</th><th>Published</th><th>Error</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></details>"
    )


def _chunk_space_panel(data: dict[str, Any]) -> str:
    if data.get("message"):
        return f"<h2>Chunk Space</h2><p>{html.escape(str(data['message']))}</p>"
    chunks = data.get("chunks") or []
    nearest = data.get("nearest") or []
    if not chunks:
        return "<h2>Chunk Space</h2><p>No chunk projection data found.</p>"
    projection = data.get("projection") or {}
    variance = projection.get("explained_variance") or [0, 0]
    return "\n".join(
        [
            "<h2>Chunk Space</h2>",
            "<p>Chunk embeddings reduced to 2 dimensions via PCA fit over the full chunk corpus. The example question is overlaid as a cross at its projected position. Highlighted points are the top-k nearest chunks to the question computed in the original embedding space, not in the 2-D projection.</p>",
            f"<p><strong>Question:</strong> {html.escape(str(data.get('question', '')))}</p>",
            f"<p><strong>Chunks:</strong> <code>{len(chunks)}</code> <strong>PCA variance:</strong> <code>{float(variance[0]):.3f}, {float(variance[1]):.3f}</code></p>",
            '<div class="filters"><label>Top K <input id="chunkTopK" type="range" min="1" max="25" value="10" oninput="renderChunkSpace()"></label><span id="chunkTopKValue">10</span><label>Colour by <select id="chunkColorBy" onchange="renderChunkSpace()"><option value="video_id">video_id</option><option value="none">none</option></select></label></div>',
            '<svg class="scatter" id="chunkScatter" viewBox="0 0 900 560" role="img"></svg>',
            '<h3>Nearest Chunks</h3><table><thead><tr><th>Video</th><th>Timestamp</th><th>Similarity</th><th>Text preview</th></tr></thead><tbody id="nearestChunksBody">',
            "".join(_nearest_row(item, index) for index, item in enumerate(nearest[:10])),
            "</tbody></table>",
        ]
    )


def _nearest_row(item: dict[str, Any], index: int) -> str:
    return (
        f'<tr data-chunk-id="{html.escape(str(item.get("chunk_id", "")))}" onclick="highlightChunkPoint(\'{html.escape(str(item.get("chunk_id", "")))}\')">'
        f"<td><code>{html.escape(str(item.get('video_id', '')))}</code></td>"
        f"<td>{_timestamp(item.get('start_seconds'))}</td>"
        f"<td class=\"num\">{float(item.get('score') or 0):.4f}</td>"
        f"<td>{html.escape(_preview(str(item.get('text', ''))))}</td>"
        "</tr>"
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


def _filter_script() -> str:
    return (
        "function filterTable(id){const table=document.getElementById(id);"
        "if(!table)return;const filters=[...document.querySelectorAll('[data-filter-table=\"'+id+'\"]')];"
        "[...table.tBodies[0].rows].forEach(row=>{let show=true;filters.forEach(f=>{const v=f.value.toLowerCase();"
        "if(!v)return;const c=Number(f.dataset.filterCol);const text=(row.cells[c]?.innerText||'').toLowerCase();"
        "if(!text.includes(v))show=false});row.style.display=show?'':'none'})}"
    )


def _json_script_payload(value: dict[str, Any]) -> str:
    return json.dumps(value).replace("</", "<\\/")


def _chunk_space_script() -> str:
    return """
<script>
function chunkData(){return JSON.parse(document.getElementById('chunk-space-data').textContent)}
function scale(values,minOut,maxOut){const min=Math.min(...values),max=Math.max(...values);return v=>max===min?(minOut+maxOut)/2:minOut+(v-min)*(maxOut-minOut)/(max-min)}
function colorFor(value){if(!value)return '#8cc8ff';let h=0;for(let i=0;i<value.length;i++)h=(h*31+value.charCodeAt(i))%360;return `hsl(${h} 70% 62%)`}
function renderChunkSpace(){const data=chunkData();const svg=document.getElementById('chunkScatter');if(!svg||!data.chunks?.length)return;const k=Number(document.getElementById('chunkTopK').value);document.getElementById('chunkTopKValue').textContent=k;const colorBy=document.getElementById('chunkColorBy').value;const nearest=data.nearest.slice(0,k);const nearestIds=new Set(nearest.map(c=>c.chunk_id));const chunks=data.chunks;const xs=chunks.map(c=>c.x).concat([data.question_point.x]);const ys=chunks.map(c=>c.y).concat([data.question_point.y]);const sx=scale(xs,40,860),sy=scale(ys,520,40);svg.innerHTML='';chunks.slice(0,5000).forEach(c=>{const el=document.createElementNS('http://www.w3.org/2000/svg','circle');el.setAttribute('cx',sx(c.x));el.setAttribute('cy',sy(c.y));el.setAttribute('r',nearestIds.has(c.chunk_id)?5:2.5);el.setAttribute('fill',nearestIds.has(c.chunk_id)?'#ffcc66':(colorBy==='video_id'?colorFor(c.video_id):'#8cc8ff'));el.setAttribute('opacity',nearestIds.has(c.chunk_id)?'1':'0.65');el.setAttribute('data-chunk-id',c.chunk_id);const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=`${c.title||c.video_id} ${c.start_seconds||''}s similarity=${c.score==null?'':c.score.toFixed(4)} ${String(c.text||'').slice(0,140)}`;el.appendChild(title);svg.appendChild(el)});const q=document.createElementNS('http://www.w3.org/2000/svg','text');q.setAttribute('x',sx(data.question_point.x));q.setAttribute('y',sy(data.question_point.y));q.setAttribute('fill','#ff6b6b');q.setAttribute('font-size','24');q.setAttribute('text-anchor','middle');q.textContent='+';svg.appendChild(q);const body=document.getElementById('nearestChunksBody');body.innerHTML=nearest.map(c=>`<tr data-chunk-id="${c.chunk_id}" onclick="highlightChunkPoint('${c.chunk_id}')"><td><code>${c.video_id}</code></td><td>${Math.floor((c.start_seconds||0)/60)}:${String(Math.floor((c.start_seconds||0)%60)).padStart(2,'0')}</td><td class="num">${c.score.toFixed(4)}</td><td>${String(c.text||'').slice(0,140).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}</td></tr>`).join('')}
function highlightChunkPoint(id){document.querySelectorAll('[data-chunk-id]').forEach(e=>e.style.outline='');document.querySelectorAll(`[data-chunk-id="${id}"]`).forEach(e=>e.style.outline='2px solid #ff6b6b')}
document.addEventListener('DOMContentLoaded',renderChunkSpace);
</script>
"""


def _duration_between(started_at: object, completed_at: object) -> str:
    start = _parse_datetime(started_at)
    end = _parse_datetime(completed_at)
    if start is None or end is None:
        return ""
    return f"{(end - start).total_seconds():.1f}s"


def _timestamp(value: object) -> str:
    seconds = _float_or_none(value)
    if seconds is None:
        return ""
    total = int(seconds)
    minutes, second = divmod(total, 60)
    return f"{minutes}:{second:02d}"


def _preview(value: str, limit: int = 140) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _parse_datetime(value: object):
    if not value:
        return None
    from datetime import datetime, timezone

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


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
    return _details_if_long(description, summary="Description")


def _details_if_long(value: str, summary: str | None = None, limit: int = 100) -> str:
    text = value or ""
    if len(text) <= limit:
        return html.escape(text)
    preview = _preview(text, limit)
    label = summary or preview
    return (
        f"<details><summary>{html.escape(label)}</summary>"
        f"<pre>{html.escape(text)}</pre>"
        "</details>"
    )


def _link_or_details(url: str, limit: int = 100) -> str:
    escaped_url = html.escape(url)
    link = f'<a href="{escaped_url}">{escaped_url}</a>'
    if len(url) <= limit:
        return link
    return (
        f'<details><summary><a href="{escaped_url}">Open source</a></summary>'
        f"<pre>{escaped_url}</pre></details>"
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


def _collect_chunk_embeddings(
    settings: Settings,
    rows: list[TranscriptDashboardRow],
) -> list[dict[str, Any]]:
    row_by_video = {row.video_id: row for row in rows}
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(settings.chunk_collection)
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    raw_embeddings = result.get("embeddings")
    embeddings = raw_embeddings if raw_embeddings is not None else []
    chunks: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        if index >= len(embeddings):
            continue
        embedding = embeddings[index]
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        metadata = metadatas[index] or {}
        video_id = str(metadata.get("video_id", ""))
        row = row_by_video.get(video_id)
        chunks.append(
            {
                "chunk_id": str(chunk_id),
                "video_id": video_id,
                "title": row.title if row else None,
                "source_url": str(metadata.get("source_url", "")),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "start_seconds": _float_or_none(metadata.get("start_seconds")),
                "end_seconds": _float_or_none(metadata.get("end_seconds")),
                "text": documents[index] if index < len(documents) else "",
                "embedding": [float(value) for value in embedding],
            }
        )
    return chunks


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
