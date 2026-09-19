"""A pool of ingestion workers, with every job's progress broadcast to subscribers.

``POST /api/index/stream`` blocks the request for the whole job and the
frontend locks its form while that stream is open, so a second video or
channel cannot be queued until the first finishes. This module decouples
submission from execution: :meth:`IngestionQueue.enqueue` returns immediately
with a job id, background workers drain jobs in submission order, and every
job's progress is broadcast to every subscriber — so the queue view stays live
across multiple browser tabs, not just the one that submitted a job.

Several workers rather than one, because indexing a video is almost entirely
network wait — a Supadata fetch, then local chunking and embedding — and one
worker leaves that latency completely unhidden. Three by default: enough to
cover the waiting, few enough that the CPU-bound embedding step (pinned to CPU
deliberately; MPS wedges the server) does not thrash. The measured precedent
in this repo is ``cli.py``'s eval matrix, where 4 workers ran cleanly and 6-8
collapsed.

Concurrency costs something the single-worker version got for free: a job used
to be able to diff the whole corpus before and after itself and call the
difference "what I added". With overlapping jobs that difference also contains
the other job's videos. See :meth:`_attribute_added`.
"""

from __future__ import annotations

import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.rag.progress import CORE_STAGES, stage_index, stage_reporter

JobStatus = str  # "queued" | "running" | "done" | "error"

#: How often a still-running job re-broadcasts its "processing" message, so a
#: subscribing SSE connection is never silent long enough to look dead.
DEFAULT_HEARTBEAT_SECONDS = 8.0

#: Concurrent ingestion jobs. Indexing is network-bound with a CPU-bound
#: embedding tail, so this hides latency without saturating the cores.
DEFAULT_MAX_WORKERS = 3

#: What each stage is doing, in the user's words rather than the function's.
STAGE_MESSAGES: dict[str, str] = {
    "discover": "Resolving target(s) …",
    "fetch": "Fetching the transcript from Supadata …",
    "chunk": "Chunking on transcript timings …",
    "embed": "Embedding chunks and writing the summary …",
}


@dataclass
class IngestionJob:
    """One queued ingestion request and its live progress."""

    id: str
    mode: str  # "video" | "channel" | "enrichment"
    target: str
    argv: list[str]
    latest: int | None = None
    status: JobStatus = "queued"
    stage: str | None = None
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    #: 1-based position of :attr:`stage` among the core indexing stages, so
    #: the UI can render "3 / 4" without hardcoding the stage list.
    stage_index: int | None = None
    stage_total: int = len(CORE_STAGES)
    #: Set only on ``mode="enrichment"`` jobs: the videos to catch the graph
    #: up on. An ingest leaves this empty and discovers its own added videos.
    enrich_video_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "target": self.target,
            "latest": self.latest,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "stage_index": self.stage_index,
            "stage_total": self.stage_total,
            "enrich_video_ids": self.enrich_video_ids,
        }


class IngestionQueue:
    """FIFO ingestion queue with a single worker and pub-sub progress events.

    ``index_fn``/``corpus_fn`` are injected exactly like ``create_app``'s own
    parameters, so tests can fake both without touching the filesystem or a
    live Chroma store.
    """

    def __init__(
        self,
        index_fn: Callable[[list[str]], int],
        corpus_fn: Callable[[], dict[str, Any]],
        graph_fn: Callable[[list[str]], dict[str, Any]] | None = None,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        failure_hint: Callable[[], str | None] | None = None,
    ) -> None:
        self._index_fn = index_fn
        self._corpus_fn = corpus_fn
        # Optional: a plain-language reason for a non-zero exit that the CLI
        # only printed to its own stderr — e.g. every Supadata key is out of
        # credits — so the job says that instead of "check the server log".
        self._failure_hint = failure_hint
        # Optional: extracts entities/claims for newly added videos once the
        # vector index succeeds. Failures here are enrichment-only — a broken
        # graph extraction must not fail a job whose vector index is already
        # good, so _process reports it in job.result["graph"] instead.
        self._graph_fn = graph_fn
        self._heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestionJob] = {}
        self._order: list[str] = []
        self._pending: queue.Queue[str] = queue.Queue()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        #: Videos already reported as "added" by some finished job, so two
        #: overlapping jobs never both claim the same one.
        self._attributed: set[str] = set()
        self.max_workers = max(1, max_workers)
        self._workers = [
            threading.Thread(target=self._run, daemon=True, name=f"ingestion-queue-{index}")
            for index in range(self.max_workers)
        ]
        for worker in self._workers:
            worker.start()

    def enqueue(
        self,
        *,
        mode: str,
        target: str,
        argv: list[str],
        latest: int | None = None,
    ) -> IngestionJob:
        job = IngestionJob(
            id=uuid.uuid4().hex[:12], mode=mode, target=target, argv=argv, latest=latest
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._pending.put(job.id)
        self._broadcast_job(job)
        return job

    def enqueue_enrichment(self, video_ids: list[str]) -> IngestionJob:
        """Queue a graph catch-up for videos already in the corpus.

        Runs through the same worker and the same progress broadcasting as an
        ingest, because it fails the same ways and the user watches it in the
        same place. ``argv`` is empty: this job indexes nothing, it only
        enriches what indexing already wrote.
        """
        job = IngestionJob(
            id=uuid.uuid4().hex[:12],
            mode="enrichment",
            target=f"{len(video_ids)} video(s)",
            argv=[],
            enrich_video_ids=list(video_ids),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._pending.put(job.id)
        self._broadcast_job(job)
        return job

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job, queued first, in submission order."""
        with self._lock:
            return [self._jobs[job_id].to_dict() for job_id in self._order]

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        """A per-connection event queue, seeded with the current snapshot.

        The seed means a client that connects mid-run still sees every
        already-queued and in-progress job immediately, not just future
        updates.
        """
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        subscriber.put({"type": "snapshot", "jobs": self.snapshot()})
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _broadcast_job(self, job: IngestionJob) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        event = {"type": "job", "job": job.to_dict()}
        for subscriber in subscribers:
            subscriber.put(event)

    def _run(self) -> None:  # pragma: no cover - exercised via enqueue in tests
        while True:
            job_id = self._pending.get()
            with self._lock:
                job = self._jobs.get(job_id)
            if job is not None:
                self._process(job)

    def _process(self, job: IngestionJob) -> None:
        if job.mode == "enrichment":
            self._process_enrichment(job)
            return
        job.status = "running"
        job.stage = "discover"
        job.stage_index = 1
        job.message = STAGE_MESSAGES["discover"]
        self._broadcast_job(job)
        try:
            before = self._corpus_fn()
            before_ids = {v["video_id"] for v in before.get("videos", [])}

            def on_stage(stage: str) -> None:
                job.stage = stage
                job.stage_index = stage_index(stage)
                job.message = STAGE_MESSAGES.get(stage)
                self._broadcast_job(job)

            # Installed *inside* the callable because ``_run_blocking`` runs it
            # on a fresh thread, and a new thread does not inherit the
            # caller's context.
            def run_index() -> int:
                with stage_reporter(on_stage):
                    return self._index_fn(job.argv)

            exit_code = self._run_blocking(job, run_index, "Still indexing...")

            if exit_code != 0:
                job.status = "error"
                hint = self._failure_hint() if self._failure_hint else None
                job.error = hint or (
                    f"Indexing failed (exit {exit_code}). Check the server log for the CLI output."
                )
                self._broadcast_job(job)
                return

            after = self._corpus_fn()
            added = self._attribute_added(job, before_ids, after.get("videos", []))
            job.result = {
                "ok": True,
                "target": job.target,
                "added_videos": added,
                "added_video_count": len(added),
                "added_chunk_count": sum(int(v.get("chunk_count") or 0) for v in added),
                "totals": after.get("totals", {}),
                "insights": after.get("insights", []),
                "channels": after.get("channels", []),
            }

            if self._graph_fn is not None and added:
                job.stage = "graph"
                added_ids = [v["video_id"] for v in added]
                job.message = f"Extracting entities & claims for {len(added_ids)} new video(s) ..."
                self._broadcast_job(job)
                try:
                    job.result["graph"] = self._run_blocking(
                        job, lambda: self._graph_fn(added_ids), "Still extracting the graph..."
                    )
                except Exception as exc:
                    # Enrichment-only: the vector index already succeeded, so a
                    # graph extraction failure must not flip the whole job to
                    # "error" — it just means these videos stay
                    # vector-RAG-only until index-graph is run again.
                    job.result["graph"] = {"ok": False, "error": str(exc)}

            job.status = "done"
            job.stage = "done"
            job.stage_index = job.stage_total
            job.message = None
            self._broadcast_job(job)
        except Exception as exc:  # a broken job must not stop the worker
            job.status = "error"
            job.error = str(exc)
            self._broadcast_job(job)

    def _attribute_added(
        self,
        job: IngestionJob,
        before_ids: set[str],
        after_videos: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Which of the new videos belong to *this* job.

        With one worker, "everything that appeared while I ran" was exact.
        With several, a job's before/after window overlaps its neighbours', so
        the naive difference happily reports another job's video as its own.

        Two rules, in order:

        * A single-video job knows precisely which video it was for — its own
          target URL — so it claims that one and nothing else.
        * A channel job cannot know without re-reading the CLI's run record,
          so it claims whatever is new and unclaimed. First finisher wins.
          That can misattribute a video *between* two concurrent channel runs,
          but it can never double-count one, and the corpus totals stay right.
        """
        fresh = [v for v in after_videos if v["video_id"] not in before_ids]
        with self._lock:
            target = _video_id_in(job.target) if job.mode == "video" else None
            unclaimed = [v for v in fresh if v["video_id"] not in self._attributed]
            if target is not None:
                claimed = [v for v in unclaimed if v["video_id"] == target]
            else:
                # Either a channel job, or a single-video target this could not
                # parse. Falling back to "everything unclaimed" keeps a job
                # that cannot identify itself reporting *something* true,
                # rather than silently reporting nothing.
                claimed = unclaimed
            self._attributed.update(v["video_id"] for v in claimed)
        return claimed

    def _process_enrichment(self, job: IngestionJob) -> None:
        """Catch the graph up, with the same failure isolation as an ingest.

        Enrichment failing is expected and survivable — it is the step that
        still needs a paid provider — so it reports rather than raises, and
        the corpus it was enriching is untouched either way.
        """
        job.status = "running"
        job.stage = "graph"
        job.message = f"Extracting entities & claims for {len(job.enrich_video_ids)} video(s) ..."
        self._broadcast_job(job)
        if self._graph_fn is None:
            job.status = "error"
            job.error = "No graph extractor is configured on this server."
            self._broadcast_job(job)
            return
        try:
            result = self._run_blocking(
                job,
                lambda: self._graph_fn(job.enrich_video_ids),
                "Still extracting the graph...",
            )
            # For an *ingest*, a failed graph is a footnote — the vector index
            # already succeeded and the video is retrievable. For an
            # enrichment job the graph is the entire job, so the same result
            # is a failure and has to read as one.
            ok = bool((result or {}).get("ok", True))
            job.result = {"ok": ok, "target": job.target, "graph": result}
            if ok:
                job.status = "done"
                job.stage = "done"
                job.message = None
            else:
                failed = (result or {}).get("failed")
                job.status = "error"
                job.error = (
                    f"Graph extraction failed for {failed} chunk(s). "
                    "The corpus is unchanged and still retrievable — "
                    "re-run the pass once the provider is available."
                )
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
        self._broadcast_job(job)

    def _run_blocking(
        self, job: IngestionJob, fn: Callable[[], Any], heartbeat_message: str
    ) -> Any:
        """Run ``fn`` on a worker thread, heartbeating ``job.message`` while it runs.

        Same shape as ``_run_index_streaming`` in ``main.py`` — a timed
        ``queue.get`` re-broadcasts the job's current message on every
        timeout so a long-running step never looks stalled to a subscriber.
        """
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        holder: dict[str, Any] = {}

        def worker() -> None:
            try:
                holder["result"] = fn()
            except Exception as exc:
                holder["error"] = exc
            finally:
                events.put(("finished", None))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            try:
                kind, _value = events.get(timeout=self._heartbeat_seconds)
            except queue.Empty:
                job.message = heartbeat_message
                self._broadcast_job(job)
                continue
            if kind == "finished":
                break
        thread.join()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]


def _video_id_in(target: str) -> str | None:
    """The YouTube video id inside a job target, if it has one.

    Deliberately local and forgiving rather than reusing the transcripts
    package's strict extractor: this is used to *attribute* a result, so a
    target it cannot parse should mean "claim nothing", not raise.
    """
    text = (target or "").strip()
    match = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", text)
    if match:
        return match.group(1)
    # A bare id is also a legitimate target — the CLI accepts one.
    return text if re.fullmatch(r"[A-Za-z0-9_-]{11}", text) else None
