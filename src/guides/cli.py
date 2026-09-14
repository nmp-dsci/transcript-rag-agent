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
from src.guides.catalog import DEFAULT_GUIDES_DIR, guide_paths, list_guides, slugify


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

    verify = sub.add_parser("verify", help="Re-run the citation verifier on a guide")
    verify.add_argument("slug")


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
        from src.guides.catalog import read_manifest, write_index, write_manifest
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

    print(f"Unknown guides command: {command}")
    return 2
