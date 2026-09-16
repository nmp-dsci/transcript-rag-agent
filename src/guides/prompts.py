"""Prompt builders for the guide agents. The rules live in the skill file;
these only bind them to one guide's files and numbers.

Kept apart from :mod:`src.guides.writer` so a prompt change is a one-file diff
that reviews on its own, and so the tests can assert on prompt content without
running an agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "field-guide-writer" / "SKILL.md"


def load_skill(path: Path | None = None) -> str:
    """The versioned writer skill, frontmatter stripped."""
    text = (path or SKILL_PATH).read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    return text.strip() + "\n"


CORPUS_ONLY_RULE = (
    "Evidence comes only from the files under corpus/ and the retrieve_chunks tool. "
    "If the corpus does not support a claim, record it as a gap rather than filling it "
    "from memory. You have no web access on this run."
)

WEB_ALLOWED_RULE = (
    "Evidence comes first from the files under corpus/ and the retrieve_chunks tool. "
    "Web search is enabled for this run: use it only for facts the corpus lacks, and "
    "mark every web-sourced fact with an <a href> to its URL — never a <cite>."
)


def extractor_system_prompt(skill: str, *, allow_web: bool) -> str:
    rule = WEB_ALLOWED_RULE if allow_web else CORPUS_ONLY_RULE
    return (
        "You are an extraction pass for a field guide. You read transcript chunks in full "
        "and record specific, citeable claims as JSON. You write nothing but the evidence "
        f"file you are asked for.\n\n{rule}\n\n--- SKILL ---\n{skill}"
    )


def extractor_prompt(
    *,
    topic: str,
    cluster_name: str,
    files: list[str],
    video_ids: list[str],
    output_path: str,
) -> str:
    file_list = "\n".join(f"- {name}" for name in files)
    return (
        f"Topic of the guide: {topic}\n\n"
        f"Your cluster is `{cluster_name}` — these videos: {', '.join(video_ids)}.\n"
        f"Read every one of these files completely, start to finish:\n{file_list}\n\n"
        "Then write the evidence file exactly at "
        f"`{output_path}` following Stage A of the skill: 25–60 concrete claims, each "
        "with the `video_id` and `chunk_index` copied from the chunk heading "
        "(`VIDEO_ID@N`) and a verbatim 3–12 word `quote` from that chunk. Include "
        "one `video_notes` entry per video, any `disagreements` between the talks, and "
        "the `gaps` a reader would still have.\n\n"
        "Use retrieve_chunks only to double-check a passage you cannot find again. "
        "Do not write any other file. Finish by writing the JSON file; no summary needed."
    )


def composer_system_prompt(skill: str, *, allow_web: bool) -> str:
    rule = WEB_ALLOWED_RULE if allow_web else CORPUS_ONLY_RULE
    return (
        "You are the composer of a field guide: a senior practitioner and editor who "
        "turns verified evidence into a page a reader acts on. You write guide.html and "
        f"guide.md and nothing else.\n\n{rule}\n\n--- SKILL ---\n{skill}"
    )


def composer_prompt(
    *,
    topic: str,
    title: str,
    slug: str,
    evidence_files: list[str],
    corpus_files: list[str],
    video_count: int,
    chunk_count: int,
    cluster_count: int,
    compiled_at: str,
    sources: list[dict[str, Any]],
    example_path: str | None,
    question: str | None = None,
) -> str:
    sources_json = json.dumps(sources, indent=1, ensure_ascii=False)
    example = (
        f"The reference example is at `{example_path}` — read it first for structure, "
        "class names and density; do not copy its content.\n\n"
        if example_path
        else ""
    )
    if question:
        opening = (
            f"Write the field guide (slug `{slug}`) that answers this question from the "
            f"corpus: **{question}**\n\n"
            "Name the guide yourself: a short library title of two to five words that says "
            "what it covers (not the question restated). Put it in `<title>` as "
            "`Name — a field guide from the transcript corpus`, in the nav brand, and in the "
            "`guide.md` frontmatter `title`. The hero `<h1>` is the editorial headline and may "
            "differ. Open the thesis by answering the question directly.\n\n"
        )
    else:
        opening = f"Write the field guide **{title}** (slug `{slug}`) on the topic: {topic}.\n\n"
    return (
        opening + f"{example}"
        f"Evidence files (read all of them):\n"
        + "\n".join(f"- {name}" for name in evidence_files)
        + "\n\nCorpus files, for re-reading a passage in context:\n"
        + "\n".join(f"- {name}" for name in corpus_files)
        + "\n\nProvenance to state in the hero and footer: "
        f"{video_count} source videos · {chunk_count} transcript chunks read in full · "
        f"{cluster_count} parallel extraction passes · compiled {compiled_at}.\n\n"
        f"Sources (video_id → title), for the Sources section:\n{sources_json}\n\n"
        "Output, following Stage B of the skill:\n"
        '1. `guide.html` — a full HTML document: `<!doctype html>`, `<html lang="en">`, '
        "a `<head>` with `<meta charset>`, viewport, `<title>`, the Google Fonts link for "
        "Instrument Serif + Geist + Geist Mono, and "
        '`<link rel="stylesheet" href="/guides/guide.css">`; a `<body>` that ends with '
        '`<script src="/guides/guide-bridge.js"></script>`. Every `<section>` has an id; '
        "every h2/h3/p/li/blockquote/pre/figure has a document-order id `{section}-{tag}{n}`. "
        "Every sourced claim carries `<cite data-video data-chunk data-quote>`.\n"
        "2. `guide.md` — the same guide as markdown with YAML frontmatter and `[VIDEO_ID@N]` cites.\n\n"
        "Write both files with the Write tool. Do not stop to ask questions; make the "
        "editorial calls yourself and finish."
    )


def fix_prompt(*, structure_errors: list[str], invalid: list[dict[str, Any]]) -> str:
    lines = ["The verifier rejected guide.html. Fix it in place with Edit; keep every existing id."]
    if structure_errors:
        lines.append("\nStructure errors:")
        lines.extend(f"- {error}" for error in structure_errors)
    if invalid:
        lines.append("\nCites that do not resolve (cite_index is document order):")
        for claim in invalid:
            why = []
            if not claim.get("video_ok"):
                why.append("video not in corpus")
            if claim.get("chunk_ok") is False:
                why.append("chunk index does not exist for that video")
            if claim.get("quote_ok") is False:
                why.append("data-quote not found in that chunk's text")
            lines.append(
                f"- #{claim.get('cite_index')} in section {claim.get('section_id')}: "
                f"video={claim.get('video_id')} chunk={claim.get('chunk_index')} "
                f"quote={json.dumps(claim.get('quote'))} — {', '.join(why) or 'invalid'}"
            )
    lines.append(
        "\nFor each bad cite: re-read the chunk in corpus/ (or call retrieve_chunks), then "
        "correct the video_id / chunk_index / data-quote so it resolves — or, if the claim "
        "is not in the corpus, remove the claim. Also apply the same fix to guide.md. "
        "Do not rewrite unrelated content."
    )
    return "\n".join(lines)


def reviser_system_prompt(skill: str, *, allow_web: bool) -> str:
    rule = WEB_ALLOWED_RULE if allow_web else CORPUS_ONLY_RULE
    return (
        "You are revising a published field guide from reader commentary. You edit "
        "guide.html and guide.md in place and write receipt.json, nothing else.\n\n"
        f"{rule}\n\n--- SKILL ---\n{skill}"
    )


def reviser_prompt(
    *,
    title: str,
    comments: list[dict[str, Any]],
    evidence_files: list[str],
    corpus_files: list[str],
    next_version: int,
) -> str:
    items = json.dumps(
        [
            {
                "id": c.get("id"),
                "anchor": c.get("anchor"),
                "section_id": c.get("section_id"),
                "quote": c.get("quote"),
                "body": c.get("body"),
            }
            for c in comments
        ],
        indent=1,
        ensure_ascii=False,
    )
    return (
        f"Revise **{title}** to version {next_version} from these reader comments:\n{items}\n\n"
        "Read guide.html and claims.json first. Evidence files:\n"
        + "\n".join(f"- {name}" for name in evidence_files)
        + "\nCorpus files:\n"
        + "\n".join(f"- {name}" for name in corpus_files)
        + "\n\nFollow Stage C of the skill: edit guide.html and guide.md in place with Edit "
        "(keep every existing id; new elements get the next id in their section), cite "
        "anything new, and write `receipt.json` with exactly one outcome per comment id. "
        "Reject a comment plainly when the corpus cannot support it. Do not ask questions; finish."
    )


def markdown_prompt(*, title: str) -> str:
    """Backfill ``guide.md`` from an existing page without recomposing it."""
    return (
        f"Read `guide.html` (the published page for **{title}**) and `manifest.json`, then "
        "write `guide.md`: the same guide as markdown, per the skill's guide.md rules — YAML "
        "frontmatter (`title`, `topic`, `compiled_at`, `videos`, `chunks`, `sources` as a list "
        "of `{video_id, title}` taken from the manifest), the same section headings in the "
        'same order, every claim kept, and every `<cite data-video="ID" data-chunk="N">` '
        "rendered as `[ID@N]` after the claim (a cite with no chunk renders as `[ID]`). Drop "
        "navigation, copy buttons and decoration; keep prompts as fenced code blocks. Do not "
        "change guide.html. Write only guide.md, then stop."
    )
