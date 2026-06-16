"""Interactive REPL for the transcript RAG agent.

The session offers two actions from a top-level menu: ask a question (answered
by one or more selectable RAG setups) or fetch/index a new URL (single video or
bulk channel). Every answered question is appended to the chat history and the
WhatsApp-style ``chat.html`` view is regenerated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.chat.frontend import DEFAULT_CHAT_HTML_PATH, write_chat_html
from src.chat.history import (
    DEFAULT_HISTORY_PATH,
    build_entry,
    load_history,
)
from src.chat.setups import (
    SETUP_SPECS,
    RagSetupRunner,
    select_setups,
)
from src.config import Settings

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
IndexFn = Callable[[list[str]], int]


def _default_index_fn(argv: list[str]) -> int:
    """Run an indexing CLI command, reusing the exact documented code path."""
    from src import cli

    return cli.main(argv)


def run_session(
    settings: Settings,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    history_path: Path = DEFAULT_HISTORY_PATH,
    chat_html_path: Path = DEFAULT_CHAT_HTML_PATH,
    runner_factory: Callable[[], RagSetupRunner] | None = None,
    index_fn: IndexFn = _default_index_fn,
) -> int:
    runner_factory = runner_factory or (lambda: RagSetupRunner.from_settings(settings))
    runner_holder: dict[str, RagSetupRunner] = {}

    def get_runner() -> RagSetupRunner:
        if "runner" not in runner_holder:
            output("Loading retrieval stack (first question only)...")
            runner_holder["runner"] = runner_factory()
        return runner_holder["runner"]

    output("YouTube Transcript RAG — interactive chat")
    output(f"History: {history_path}   View: {chat_html_path}")

    while True:
        output("")
        output("Main menu:")
        output("  [1] Ask a question")
        output("  [2] Fetch / index a new URL")
        output("  [q] Quit")
        try:
            choice = input_fn("Choose: ").strip().lower()
        except EOFError:
            output("")
            break
        if choice in {"q", "quit", "exit"}:
            break
        if choice == "1":
            try:
                _ask_flow(
                    settings,
                    get_runner,
                    input_fn,
                    output,
                    history_path,
                    chat_html_path,
                )
            except EOFError:
                output("")
                break
        elif choice == "2":
            try:
                _fetch_flow(input_fn, output, index_fn)
            except EOFError:
                output("")
                break
        else:
            output(f"Unknown choice: {choice!r}")

    output("Goodbye.")
    return 0


def _ask_flow(
    settings: Settings,
    get_runner: Callable[[], RagSetupRunner],
    input_fn: InputFn,
    output: OutputFn,
    history_path: Path,
    chat_html_path: Path,
) -> None:
    question = input_fn("Question: ").strip()
    if not question:
        output("No question entered; back to menu.")
        return
    url = (
        input_fn("Restrict to a single video URL (optional, blank for all): ").strip()
        or None
    )

    output("")
    output("RAG setups:")
    for index, spec in enumerate(SETUP_SPECS, 1):
        output(f"  [{index}] {spec.title} — {spec.description}")
    output("  [a] all (compare every setup)")

    keys: list[str] | None = None
    while keys is None:
        raw = input_fn("Choose setup(s) (e.g. 1,3 or a; blank to cancel): ").strip()
        if not raw:
            output("Cancelled; back to menu.")
            return
        try:
            keys = select_setups(raw)
        except ValueError as exc:
            output(f"  {exc}. Try again.")

    output("")
    results = get_runner().run_many(
        keys, question, url=url, on_progress=lambda message: output(f"  {message}")
    )

    entry = build_entry(question, results, url=url)
    entries = load_history(history_path)
    entries.append(entry)
    from src.chat.history import save_history

    save_history(entries, history_path)
    write_chat_html(entries, chat_html_path)

    output("")
    output(f"Captured {len(results)} answer(s) for: {entry.id}")
    for result in results:
        status = "error" if result.error else f"{len(result.answer)} chars"
        output(f"  - {result.title}: {status} ({result.elapsed_seconds}s)")
    output(f"Updated {chat_html_path} — open it to read the conversation.")


def _fetch_flow(
    input_fn: InputFn,
    output: OutputFn,
    index_fn: IndexFn,
) -> None:
    output("")
    output("Fetch a new URL:")
    output("  [1] Single video URL")
    output("  [2] Bulk (whole channel)")
    mode = input_fn("Choose: ").strip().lower()

    if mode == "1":
        url = input_fn("Video URL: ").strip()
        if not url:
            output("No URL entered; back to menu.")
            return
        output(f"Indexing {url} ...")
        code = index_fn(["index-rag", url])
        output("Done." if code == 0 else f"index-rag exited with code {code}.")
    elif mode == "2":
        channel = input_fn("Channel (URL or @handle): ").strip()
        if not channel:
            output("No channel entered; back to menu.")
            return
        latest_raw = input_fn("How many latest videos? [5]: ").strip() or "5"
        if not latest_raw.isdigit():
            output(f"Invalid number: {latest_raw!r}; back to menu.")
            return
        output(f"Bulk indexing latest {latest_raw} from {channel} ...")
        code = index_fn(
            ["bulk-index", "channel", "--channel", channel, "--latest", latest_raw]
        )
        output("Done." if code == 0 else f"bulk-index exited with code {code}.")
    else:
        output(f"Unknown choice: {mode!r}")
