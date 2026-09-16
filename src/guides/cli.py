"""``python -m src.cli guides …`` — the terminal path to the Field Guides tab.

Kept out of :mod:`src.cli` so the guide commands read as one unit and the
2,600-line parser gains two hooks rather than a page. Every subcommand here
writes only under ``guides/`` and reads the corpus through the same functions
the API uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import Settings
from src.guides.catalog import (
    DEFAULT_GUIDES_DIR,
    guide_paths,
    list_guides,
    read_manifest,
    slugify,
    write_index,
    write_manifest,
)


def add_guides_parser(subparsers: Any) -> None:
    guides = subparsers.add_parser(
        "guides",
        help=(
            "Field guides: import, write, verify, comment on and revise the "
            "long-form guides under guides/ (the Field Guides tab)."
        ),
    )
    guides.add_argument(
        "--guides-dir",
        type=Path,
        default=None,
        help=f"Guide directory (default: {DEFAULT_GUIDES_DIR})",
    )
    sub = guides.add_subparsers(dest="guides_command", required=True)

    listing = sub.add_parser("list", help="List the guides and their cite rates")
    listing.add_argument("--json", action="store_true", help="Print the catalog as JSON")

    imp = sub.add_parser(
        "import",
        help=(
            "Bring a hand-made HTML page (e.g. a .lavish artifact) under guides/ as "
            "v1: shared stylesheet, stable ids, <cite> markup from its YouTube links."
        ),
    )
    imp.add_argument("source", type=Path, help="Path to the HTML page")
    imp.add_argument("--slug", default=None, help="Guide slug (default: from the title)")
    imp.add_argument("--title", default=None, help="Override the page title")
    imp.add_argument("--topic", default=None, help="The topic string (default: the title)")
    imp.add_argument("--compiled-at", default=None, help="Date the page was compiled (YYYY-MM-DD)")
    imp.add_argument(
        "--no-css",
        action="store_true",
        help="Do not (re)write guides/guide.css from the page's stylesheet",
    )
    imp.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an already-imported guides/<slug>/versions/v1.html",
    )

    verify = sub.add_parser("verify", help="Re-run the citation verifier on a guide")
    verify.add_argument("slug")

    markdown = sub.add_parser(
        "markdown",
        help="Backfill guide.md (the agent-facing copy) from a guide's published page",
    )
    markdown.add_argument("slug")
    markdown.add_argument("--model", default=None, help="Default: claude-opus-5")

    comment = sub.add_parser("comment", help="Add a reader comment to a guide")
    comment.add_argument("slug")
    comment.add_argument("body", nargs="?", default=None, help="The comment text")
    comment.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help=(
            "Import comments from a JSON list of {body, anchor?, section_id?, quote?} — "
            "e.g. annotations exported from a Lavish review"
        ),
    )
    comment.add_argument("--anchor", default=None, help="Element id in the page (e.g. thesis-p2)")
    comment.add_argument("--section", default=None, help="Section id (e.g. thesis)")
    comment.add_argument("--quote", default="", help="The selected text the comment is about")

    comments = sub.add_parser("comments", help="List a guide's comments and their status")
    comments.add_argument("slug")
    comments.add_argument("--json", action="store_true")

    revise = sub.add_parser(
        "revise",
        help=(
            "Apply a guide's open comments as one tracked batch with the Agent SDK, "
            "verify, and publish the next version with a per-comment receipt."
        ),
    )
    revise.add_argument("slug")
    revise.add_argument(
        "--only", default=None, help="Comma-separated comment ids (default: all open)"
    )
    revise.add_argument("--allow-web", action="store_true")
    revise.add_argument("--model", default=None, help="Default: claude-opus-5")

    scope = sub.add_parser(
        "scope", help="Rank the corpus for a topic — the videos a guide would read"
    )
    scope.add_argument("--topic", default=None)
    scope.add_argument(
        "--question",
        default=None,
        help="A plain-language question; the topic and probes are derived",
    )
    scope.add_argument("--limit", type=int, default=25)
    scope.add_argument("--json", action="store_true")

    write = sub.add_parser(
        "write",
        help=(
            "Write a guide with the Agent SDK: scope → export every chunk of the "
            "confirmed videos → parallel extraction passes → compose → verify → publish. "
            "Resumes from the last completed stage if the guide directory already has one."
        ),
    )
    write.add_argument("--topic", default=None, help="What the guide is about")
    write.add_argument(
        "--question",
        default=None,
        help="Ask instead: the topic is derived and the composer names the guide",
    )
    write.add_argument("--slug", default=None, help="Guide slug (default: from the title)")
    write.add_argument("--title", default=None, help="Page title (default: the topic, title-cased)")
    write.add_argument(
        "--videos",
        default=None,
        help="Comma-separated video ids to use instead of scoping (skips the checklist)",
    )
    write.add_argument("--limit", type=int, default=20, help="Max candidates to offer (default 20)")
    write.add_argument(
        "--min-score", type=float, default=2.0, help="Drop candidates below this scope score"
    )
    write.add_argument(
        "--yes", action="store_true", help="Accept the candidate list without asking"
    )
    write.add_argument(
        "--allow-web", action="store_true", help="Enable WebSearch/WebFetch this run"
    )
    write.add_argument("--extract-model", default=None, help="Default: claude-sonnet-5")
    write.add_argument("--compose-model", default=None, help="Default: claude-opus-5")
    write.add_argument("--max-clusters", type=int, default=6)
    write.add_argument("--serial", action="store_true", help="Run extraction passes one at a time")
    write.add_argument("--compiled-at", default=None)


def _corpus_lookup(settings: Settings) -> tuple[set[str], Any, dict[str, int]]:
    from src.api.corpus import list_corpus, load_chunk_corpus

    corpus = list_corpus(
        settings.chroma_path, settings.raw_transcript_collection, settings.chunk_collection
    )
    videos = corpus.get("videos", [])
    known = {video["video_id"] for video in videos}
    counts = {video["video_id"]: int(video.get("chunk_count") or 0) for video in videos}

    def chunk_lookup(video_id: str) -> list[dict[str, Any]]:
        return load_chunk_corpus(settings.chroma_path, settings.chunk_collection, video_id)

    return known, chunk_lookup, counts


def run_guides(args: argparse.Namespace, settings: Settings) -> int:
    guides_dir: Path = args.guides_dir or DEFAULT_GUIDES_DIR
    command = args.guides_command

    if command == "list":
        listing = list_guides(guides_dir)
        if args.json:
            print(json.dumps(listing, indent=2))
            return 0
        if not listing["guides"]:
            print(f"No guides under {guides_dir}")
            return 0
        for guide in listing["guides"]:
            rate = f"{guide['cite_valid']}/{guide['cite_total']}" if guide["cite_total"] else "—"
            print(
                f"{guide['slug']:<32} v{guide['current_version']}  "
                f"{guide['videos']:>3} videos  cites {rate:<8} {guide['title']}"
            )
        return 0

    if command == "import":
        from src.guides.importer import import_lavish_html, rewrite

        source: Path = args.source
        if not source.is_file():
            print(f"Not a file: {source}")
            return 2
        _, _, found_title = rewrite(source.read_text(encoding="utf-8"))
        slug = args.slug or slugify(args.title or found_title or source.stem)
        known, chunk_lookup, counts = _corpus_lookup(settings)
        manifest = import_lavish_html(
            source,
            slug=slug,
            guides_dir=guides_dir,
            topic=args.topic,
            title=args.title,
            known_videos=known,
            chunk_lookup=chunk_lookup,
            chunk_counts=counts,
            compiled_at=args.compiled_at,
            write_css=not args.no_css,
            force=args.force,
        )
        print(
            f"Imported {manifest.title!r} as guides/{slug} v1: "
            f"{len(manifest.video_ids)} sources, cites {manifest.cite_valid}/{manifest.cite_total}"
        )
        missing = sorted(set(manifest.video_ids) - known)
        if missing:
            print(f"  not in corpus: {', '.join(missing)}")
        return 0

    if command == "verify":
        from src.guides.verify import verify_html, write_claims

        paths = guide_paths(args.slug, guides_dir)
        found = read_manifest(paths)
        if found is None:
            print(f"Unknown guide: {args.slug}")
            return 2
        manifest = found
        known, chunk_lookup, _counts = _corpus_lookup(settings)
        report = verify_html(
            paths.html.read_text(encoding="utf-8"), known_videos=known, chunk_lookup=chunk_lookup
        )
        write_claims(paths.claims, report)
        manifest.cite_total = report.total
        manifest.cite_valid = report.valid
        write_manifest(paths, manifest)
        write_index(guides_dir)
        print(f"{args.slug}: cites {report.valid}/{report.total} ({report.pass_rate:.1%})")
        for error in report.structure_errors:
            print(f"  structure: {error}")
        for claim in report.invalid:
            print(
                f"  invalid: cite #{claim.cite_index} video={claim.video_id} chunk={claim.chunk_index}"
            )
        return 0 if report.ok else 1

    if command == "scope":
        known, _lookup, _counts, videos = _corpus_videos(settings)
        provider = _provider(settings)
        from src.guides.scope import candidate_videos, probes_for, topic_from_question, topic_has_words

        question = " ".join((args.question or "").split())
        explicit_topic = " ".join((args.topic or "").split())
        topic = explicit_topic or (topic_from_question(question) if question else "")
        if not topic:
            print("Give --question or --topic.")
            return 2
        if question and not explicit_topic and not topic_has_words(topic):
            print("Ask a question with some words in it.")
            return 2
        if question:
            print(f"topic: {topic}")
        ranked = candidate_videos(
            topic,
            videos,
            lambda q, k: provider.get_context(q, top_k=k).retrieved_chunks,
            probes=probes_for(question) if question else None,
        )[: args.limit]
        if args.json:
            print(json.dumps([c.to_dict() for c in ranked], indent=2))
            return 0
        _print_candidates(ranked)
        return 0

    if command == "write":
        return _run_write(args, settings, guides_dir)

    if command == "markdown":
        return _run_markdown(args, settings, guides_dir)

    if command == "comment":
        from src.guides.comments import add_comment

        paths = guide_paths(args.slug, guides_dir)
        if read_manifest(paths) is None:
            print(f"Unknown guide: {args.slug}")
            return 2
        if args.from_json is not None:
            try:
                items = json.loads(args.from_json.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"Cannot read {args.from_json}: {exc}")
                return 2
            if not isinstance(items, list):
                print("The JSON file must hold a list of comments.")
                return 2
            added = 0
            for item in items:
                if not isinstance(item, dict) or not str(item.get("body") or "").strip():
                    continue
                record = add_comment(
                    paths,
                    body=str(item["body"]),
                    anchor=item.get("anchor"),
                    section_id=item.get("section_id"),
                    quote=str(item.get("quote") or ""),
                    author=str(item.get("author") or "lavish"),
                )
                added += 1
                print(f"{record['id']}  {record['anchor'] or '—':<16} {record['body'][:80]}")
            print(f"{added} comments added to guides/{args.slug}")
            return 0
        if not args.body:
            print("A comment body (or --from-json) is required.")
            return 2
        record = add_comment(
            paths,
            body=args.body,
            anchor=args.anchor,
            section_id=args.section,
            quote=args.quote,
            author="cli",
        )
        print(f"{record['id']} added to guides/{args.slug} (anchor {record['anchor'] or '—'})")
        return 0

    if command == "comments":
        from src.guides.comments import latest_comments

        records = latest_comments(guide_paths(args.slug, guides_dir))
        if args.json:
            print(json.dumps(records, indent=2))
            return 0
        if not records:
            print("No comments.")
            return 0
        for record in records:
            resolved = (
                f" (v{record['resolved_in_version']})" if record.get("resolved_in_version") else ""
            )
            print(
                f"{record['id']}  {record.get('status', 'open'):<9}{resolved:<6} "
                f"{record.get('anchor') or '—':<16} {record['body'][:80]}"
            )
            if record.get("reason"):
                print(f"{'':>18}↳ {record['reason'][:100]}")
        return 0

    if command == "revise":
        return _run_revise(args, settings, guides_dir)

    print(f"Unknown guides command: {command}")
    return 2


def _corpus_videos(
    settings: Settings,
) -> tuple[set[str], Any, dict[str, int], list[dict[str, Any]]]:
    from src.api.corpus import list_corpus

    corpus = list_corpus(
        settings.chroma_path, settings.raw_transcript_collection, settings.chunk_collection
    )
    videos = list(corpus.get("videos", []))
    known, lookup, counts = _corpus_lookup(settings)
    return known, lookup, counts, videos


def _provider(settings: Settings) -> Any:
    """The app's hybrid retrieval provider (loads the embedding stack once)."""
    from src.chat.setups import RagSetupRunner

    return RagSetupRunner.from_settings(settings).provider


def _print_candidates(ranked: list[Any]) -> None:
    print(f"{'#':>3}  {'score':>5}  {'hits':>4}  {'chunks':>6}  video_id      title")
    for number, candidate in enumerate(ranked, 1):
        flag = "*" if candidate.title_match else " "
        print(
            f"{number:>3}  {candidate.score:>5.1f}  {candidate.probe_hits:>4}  "
            f"{candidate.chunk_count:>6}  {candidate.video_id:<12} {flag} {candidate.title[:70]}"
        )


def _confirm_selection(ranked: list[Any]) -> list[str]:
    """Interactive checklist: drop numbers, add ids, or accept as shown."""
    _print_candidates(ranked)
    print(
        "\nEnter to accept all; or numbers to drop (e.g. `3 7`), `+VIDEO_ID` to add, `q` to abort:"
    )
    selected = [c.video_id for c in ranked]
    while True:
        try:
            answer = input("> ").strip()
        except EOFError:
            answer = ""
        if answer == "":
            return selected
        if answer.lower() == "q":
            return []
        for token in answer.split():
            if token.startswith("+"):
                if token[1:] and token[1:] not in selected:
                    selected.append(token[1:])
            elif token.isdigit():
                index = int(token) - 1
                if 0 <= index < len(ranked) and ranked[index].video_id in selected:
                    selected.remove(ranked[index].video_id)
        print(f"{len(selected)} selected: {', '.join(selected)}")
        print("Enter to accept, or keep editing:")


def _sdk_ready() -> str | None:
    from src.guides.service import sdk_problem

    return sdk_problem()


def _print_event(event: dict[str, Any]) -> None:
    stamp = event.get("at", "")[11:19]
    stage = event.get("stage", "")
    status = event.get("status", "")
    message = event.get("message", "")
    if status == "tool":
        print(f"  {stamp}  {stage:<18} · {message}", flush=True)
    else:
        print(f"{stamp}  [{stage} {status}] {message}", flush=True)


def _run_markdown(args: argparse.Namespace, settings: Settings, guides_dir: Path) -> int:
    problem = _sdk_ready()
    if problem:
        print(problem)
        return 2
    from src.guides.writer import GuideWriter, WriterConfig

    paths = guide_paths(args.slug, guides_dir)
    manifest = read_manifest(paths)
    if manifest is None:
        print(f"Unknown guide: {args.slug}")
        return 2
    known, lookup, _counts = _corpus_lookup(settings)
    writer = GuideWriter(
        paths,
        chunks_for=lambda ids: [],
        retrieve=lambda q, ids, k: [],
        known_videos=known,
        chunk_lookup=lookup,
        config=WriterConfig(compose_model=args.model or WriterConfig.compose_model),
        on_event=_print_event,
    )
    try:
        writer.write_markdown(title=manifest.title, video_ids=list(manifest.video_ids))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed: {exc}")
        return 1
    write_index(guides_dir)
    print(f"Wrote guides/{args.slug}/guide.md")
    return 0


def _run_revise(args: argparse.Namespace, settings: Settings, guides_dir: Path) -> int:
    problem = _sdk_ready()
    if problem:
        print(problem)
        return 2
    from src.guides.comments import open_comments
    from src.guides.service import build_writer
    from src.guides.writer import WriterConfig

    paths = guide_paths(args.slug, guides_dir)
    manifest = read_manifest(paths)
    if manifest is None:
        print(f"Unknown guide: {args.slug}")
        return 2
    pending = open_comments(paths)
    only = [c.strip() for c in (args.only or "").split(",") if c.strip()]
    if only:
        pending = [c for c in pending if c["id"] in set(only)]
    if not pending:
        print("No open comments to apply.")
        return 1
    print(f"Revising guides/{args.slug} with {len(pending)} comments:")
    for record in pending:
        print(f"  {record['id']}  {record.get('anchor') or '—':<16} {record['body'][:80]}")
    config = WriterConfig(
        compose_model=args.model or WriterConfig.compose_model, allow_web=args.allow_web
    )
    writer = build_writer(
        settings, _provider(settings), paths, config=config, on_event=_print_event
    )
    try:
        published = writer.revise(comment_ids=[c["id"] for c in pending])
    except Exception as exc:  # noqa: BLE001
        print(f"Failed: {exc}")
        if writer.run_log:
            print(f"Run log: {writer.run_log}")
        return 1
    receipt = json.loads(paths.receipt(published.current_version).read_text(encoding="utf-8"))
    print(f"Published v{published.current_version}: {receipt.get('summary', '')}")
    for item in receipt.get("items", []):
        print(f"  {item['id']}  {item['outcome']:<9} {item.get('reason', '')[:90]}")
    return 0


def _run_write(args: argparse.Namespace, settings: Settings, guides_dir: Path) -> int:
    problem = _sdk_ready()
    if problem:
        print(problem)
        return 2

    from src.guides.scope import candidate_videos, probes_for, topic_from_question, topic_has_words
    from src.guides.service import build_writer, retrieval_fns
    from src.guides.writer import WriterConfig

    question = " ".join((args.question or "").split())
    explicit_topic = " ".join((args.topic or "").split())
    topic = explicit_topic or (topic_from_question(question) if question else "")
    if not topic:
        print("Give --question or --topic.")
        return 2
    if question and not explicit_topic and not topic_has_words(topic):
        print("Ask a question with some words in it.")
        return 2
    # Asked: the question is the working title until the composer names the page.
    title = args.title or question or topic.title()
    slug = args.slug or slugify(topic if question else title)
    paths = guide_paths(slug, guides_dir)
    known, _lookup, _counts, videos = _corpus_videos(settings)
    provider = _provider(settings)
    retrieve_whole, _retrieve = retrieval_fns(provider)

    if args.videos:
        video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        ranked = [
            c
            for c in candidate_videos(
                topic, videos, retrieve_whole, probes=probes_for(question) if question else None
            )
            if c.score >= args.min_score
        ][: args.limit]
        if not ranked:
            print("No candidate videos for that topic. Try --videos or a broader topic.")
            return 1
        if args.yes:
            _print_candidates(ranked)
            video_ids = [c.video_id for c in ranked]
        else:
            video_ids = _confirm_selection(ranked)
            if not video_ids:
                print("Aborted.")
                return 1
    unknown = [v for v in video_ids if v not in known]
    if unknown:
        print(f"Not in the corpus: {', '.join(unknown)}")
        return 1

    config = WriterConfig(
        extract_model=args.extract_model or WriterConfig.extract_model,
        compose_model=args.compose_model or WriterConfig.compose_model,
        allow_web=args.allow_web,
        max_clusters=args.max_clusters,
        parallel_extract=not args.serial,
    )
    writer = build_writer(settings, provider, paths, config=config, on_event=_print_event)
    print(f"Writing guides/{slug} — {title!r} on {topic!r} from {len(video_ids)} videos")
    try:
        manifest = writer.write(
            topic=topic,
            title=title,
            video_ids=video_ids,
            videos_meta=videos,
            compiled_at=args.compiled_at,
            question=question or None,
        )
    except Exception as exc:  # noqa: BLE001 - one line, then the run log has the rest
        print(f"Failed: {exc}")
        if writer.run_log:
            print(f"Run log: {writer.run_log}")
        return 1
    print(
        f"Published guides/{slug} v{manifest.current_version}: {len(manifest.video_ids)} videos, "
        f"{manifest.chunk_count} chunks, cites {manifest.cite_valid}/{manifest.cite_total}, "
        f"{len(manifest.gaps)} gaps"
    )
    return 0
