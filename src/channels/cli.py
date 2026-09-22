"""``python -m src.cli channels …`` — the operator surface for watched sources.

Four verbs, and the boundary between them is the thing worth keeping straight:

``list``     what is configured, and what each channel has accumulated.
``add``      append a channel to ``channels.yaml``, comments preserved.
``poll``     ask channels what is new, and ingest it. ``--dry-run`` prints the
             candidates and writes nothing — the same code path, minus its
             consumer, rather than a second implementation that can drift.
``refresh``  re-visit sources already stored, to notice what changed under them.

``poll`` and ``refresh`` are deliberately separate commands rather than one
``sync``. They answer different questions, cost different amounts, and fail in
different ways; a single command would hide which half was responsible for an
afternoon of requests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.channels.ingest import WebIngestor
from src.channels.models import ChannelConfig, ChannelConfigError
from src.channels.pollers import PollContext, poll_channel
from src.channels.refresh import due_sources, refresh_sources
from src.channels.registry import (
    ChannelStateStore,
    append_channel,
    load_registry,
    state_path,
)
from src.channels.robots import RobotsPolicy
from src.channels.schedule import is_due, record_poll, reset
from src.config import Settings


def add_channel_parsers(subparsers: argparse._SubParsersAction) -> None:
    channels = subparsers.add_parser("channels", help="Watched text sources for the corpus")
    verbs = channels.add_subparsers(dest="channels_command", required=True)

    listing = verbs.add_parser("list", help="Show configured channels and their state")
    listing.add_argument("--json", action="store_true", help="Machine-readable output")

    add = verbs.add_parser("add", help="Append a channel to channels.yaml")
    add.add_argument("--id", required=True, help="Lowercase kebab-case identifier")
    add.add_argument(
        "--kind", required=True, choices=["rss", "atom", "sitemap", "github_docs", "url_list"]
    )
    add.add_argument("--url", help="Feed, sitemap or repository URL")
    add.add_argument("--label", default="", help="Human-readable name")
    add.add_argument("--urls", nargs="*", default=[], help="url_list: the URLs to watch")
    add.add_argument("--paths", nargs="*", default=[], help="github_docs: repo-relative files")
    add.add_argument("--include", help="Keep only candidate URLs matching this regex")
    add.add_argument("--include-title", help="Keep only candidates whose title matches")
    add.add_argument(
        "--body-in-feed",
        action="store_true",
        help="The feed carries the article, so the page is never fetched",
    )
    add.add_argument("--disabled", action="store_true", help="Add it switched off")

    poll = verbs.add_parser("poll", help="Find and ingest what is new at each channel")
    poll.add_argument(
        "--id", action="append", default=[], help="Poll only this channel; repeatable"
    )
    poll.add_argument("--dry-run", action="store_true", help="Print candidates, write nothing")
    poll.add_argument("--force", action="store_true", help="Poll even when not yet due")
    poll.add_argument(
        "--reset", action="store_true", help="Clear state first, re-enabling failures"
    )

    refresh = verbs.add_parser("refresh", help="Re-visit stored sources to notice changes")
    refresh.add_argument("--id", help="Only sources from this channel")
    refresh.add_argument("--limit", type=int, help="Stop after this many sources")
    refresh.add_argument(
        "--force", action="store_true", help="Re-visit everything, ignoring cadence"
    )
    refresh.add_argument("--dry-run", action="store_true", help="List what is due, fetch nothing")
    refresh.add_argument(
        "--reextract",
        action="store_true",
        help="Re-derive chunks even when the page is unchanged, after an extractor change",
    )


def _channels_file(settings: Settings) -> Path | None:
    return settings.channels_file


def _stores(settings: Settings) -> tuple[WebIngestor, RobotsPolicy]:
    from src.rag.embeddings import HuggingFaceEmbeddingModel
    from src.rag.web_store import WebChunkStore, WebSourceStore

    embeddings = HuggingFaceEmbeddingModel(settings.embedding_model, settings.embedding_device)
    robots = RobotsPolicy()
    ingestor = WebIngestor(
        source_store=WebSourceStore(
            settings.chroma_path, embeddings, settings.web_source_collection
        ),
        chunk_store=WebChunkStore(settings.chroma_path, embeddings, settings.web_chunk_collection),
        robots=robots,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    return ingestor, robots


def _run_list(args: argparse.Namespace, settings: Settings) -> int:
    registry = load_registry(_channels_file(settings))
    states = ChannelStateStore(state_path(settings.chroma_path))
    if not len(registry):
        print("No channels configured. Add one with: channels add --id … --kind rss --url …")
        return 0
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "channels": [
                        {**channel.to_dict(), "state": states.get(channel.id).to_dict()}
                        for channel in registry
                    ]
                },
                indent=2,
            )
        )
        return 0
    print(f"{'id':<32} {'kind':<12} {'on':<3} {'last polled':<20} {'every':>7}  note")
    for channel in registry:
        state = states.get(channel.id)
        note = state.disabled_reason or state.last_error or f"{len(state.seen_ids)} seen"
        print(
            f"{channel.id:<32} {channel.kind:<12} {'yes' if channel.enabled else 'no ':<3} "
            f"{(state.last_polled_at or 'never')[:19]:<20} {state.interval_hours:>6.0f}h  {note[:44]}"
        )
    print(f"\n{len(registry)} channels, {len(registry.enabled())} enabled")
    return 0


def _run_add(args: argparse.Namespace, settings: Settings) -> int:
    channel = ChannelConfig(
        id=args.id,
        kind=args.kind,
        url=args.url,
        label=args.label,
        enabled=not args.disabled,
        urls=tuple(args.urls),
        paths=tuple(args.paths),
        include=args.include,
        include_title=args.include_title,
        body_in_feed=args.body_in_feed,
    )
    target = append_channel(channel, _channels_file(settings))
    print(f"Added channel {channel.id!r} ({channel.kind}) to {target}")
    print("Nothing is fetched until you run: channels poll --id " + channel.id)
    return 0


def _run_poll(args: argparse.Namespace, settings: Settings) -> int:
    registry = load_registry(_channels_file(settings))
    states = ChannelStateStore(state_path(settings.chroma_path))
    wanted = set(args.id or [])
    selected = (
        [c for c in registry if not wanted or c.id in wanted]
        if wanted
        else list(registry.enabled())
    )
    if not selected:
        print("No matching channels." if wanted else "No enabled channels.")
        return 1

    context = PollContext()
    ingestor = None if args.dry_run else _stores(settings)[0]
    if ingestor is not None:
        context = PollContext(robots=ingestor.robots)

    total_new = total_chunks = total_embedded = 0
    for channel in selected:
        state = states.get(channel.id)
        if args.reset:
            reset(state, channel)
        if not is_due(channel, state, force=args.force):
            # Distinguish the three reasons, because they need different
            # actions: switch it on in the file, reset it, or just wait.
            if not channel.enabled:
                reason = "disabled in channels.yaml — set enabled: true to poll it"
            elif state.disabled_reason:
                reason = f"{state.disabled_reason} (use --reset to try again)"
            else:
                reason = f"not due until {(state.next_due_at or '?')[:19]} (use --force)"
            print(f"{channel.id:<32} skipped: {reason[:76]}")
            continue
        result = poll_channel(channel, state, context)
        record_poll(channel, state, result)
        if result.error:
            states.put(state)
            print(f"{channel.id:<32} FAILED: {result.error[:70]}")
            continue
        filtered = f", {result.filtered_out} filtered out" if result.filtered_out else ""
        print(f"{channel.id:<32} {len(result.candidates)} new{filtered}")
        for candidate in result.candidates:
            if ingestor is None:
                print(f"    would ingest  {candidate.reader_url}")
                continue
            outcome = ingestor.ingest(candidate, channel)
            if outcome.source is not None:
                state.remember([candidate.external_id])
            total_chunks += outcome.chunk_count
            total_embedded += outcome.embedded_count
            total_new += 1 if outcome.changed else 0
            detail = outcome.reason or f"{outcome.chunk_count} chunks, {outcome.words} words"
            print(f"    {outcome.outcome:<10} {candidate.reader_url[-62:]:<62} {detail[:60]}")
        states.put(state)
    if not args.dry_run:
        states.save()
        print(
            f"\n{total_new} sources indexed or updated, {total_chunks} chunks, "
            f"{total_embedded} embedded"
        )
    else:
        print("\nDry run: nothing was fetched into the corpus and no state was written.")
    return 0


def _run_refresh(args: argparse.Namespace, settings: Settings) -> int:
    registry = load_registry(_channels_file(settings))
    ingestor, _robots = _stores(settings)
    sources = due_sources(
        ingestor.source_store, channel_id=args.id, force=args.force, limit=args.limit
    )
    if not sources:
        print("Nothing due. Use --force to re-visit anyway.")
        return 0
    if args.dry_run:
        print(f"{len(sources)} sources due:")
        for source in sources:
            print(
                f"  {source.state:<10} last fetched {source.last_fetched_at[:19]}  "
                f"{source.reader_url[-70:]}"
            )
        return 0
    channels = {channel.id: channel for channel in registry}
    report = refresh_sources(ingestor, sources, channels, reextract=args.reextract)
    for outcome in report.outcomes:
        detail = outcome.reason or f"{outcome.embedded_count} re-embedded of {outcome.chunk_count}"
        print(f"  {outcome.outcome:<10} {outcome.url[-64:]:<64} {detail[:52]}")
    print(f"\n{report.summary()}")
    return 0


def run_channels(args: argparse.Namespace, settings: Settings) -> int:
    """Dispatch a ``channels`` subcommand."""
    handlers = {
        "list": _run_list,
        "add": _run_add,
        "poll": _run_poll,
        "refresh": _run_refresh,
    }
    handler = handlers.get(args.channels_command)
    if handler is None:  # pragma: no cover - argparse requires a known verb
        print(f"Unknown channels command: {args.channels_command}")
        return 2
    try:
        return handler(args, settings)
    except ChannelConfigError as exc:
        print(f"channels.yaml: {exc}")
        return 1
