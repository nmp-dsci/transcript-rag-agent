"""A channel for reporting which indexing stage is actually running.

The queue drives indexing through ``cli.main(argv)``, which returns an exit
code and nothing else. Before this module the queue filled that silence by
announcing "discover", "fetch" and "processing" in three consecutive
statements *before* doing any work, then blocking on one opaque call for the
entire run. A progress bar built on that jumped straight to 3/6 and froze —
worse than no progress bar, because it looked informative.

Rather than reshape the CLI boundary to carry progress, a stage reporter is
installed for the duration of a run and the indexer calls it as each stage
genuinely begins.

Scoped with a :class:`~contextvars.ContextVar` rather than a global so
concurrent ingestion workers cannot report into each other's jobs. Note that a
new thread does *not* inherit the caller's context, so the reporter must be
installed on the thread that actually runs the index — see
``IngestionQueue._process``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

#: The stages of a core index, in order. Enrichment is deliberately absent:
#: summaries come from the video description and are written during ``embed``,
#: and graph extraction is a separate job. These four are the whole progress
#: bar, and every one of them can fail only on local work or Supadata.
CORE_STAGES: tuple[str, ...] = ("discover", "fetch", "chunk", "embed")

StageReporter = Callable[[str], None]

_reporter: ContextVar[StageReporter | None] = ContextVar("stage_reporter", default=None)


@contextmanager
def stage_reporter(reporter: StageReporter | None) -> Iterator[None]:
    """Install ``reporter`` for this context, restoring the previous one after."""
    token = _reporter.set(reporter)
    try:
        yield
    finally:
        _reporter.reset(token)


def report_stage(stage: str) -> None:
    """Announce that ``stage`` has begun. A no-op when nothing is listening.

    Never raises: progress reporting must not be able to fail an index. That
    is the same rule the enrichment steps follow, for the same reason.
    """
    reporter = _reporter.get()
    if reporter is None:
        return
    try:
        reporter(stage)
    except Exception:  # pragma: no cover - defensive
        pass


def stage_index(stage: str) -> int | None:
    """1-based position of ``stage`` in :data:`CORE_STAGES`, or None."""
    try:
        return CORE_STAGES.index(stage) + 1
    except ValueError:
        return None
