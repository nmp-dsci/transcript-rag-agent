"""A queue of ingestion jobs, processed one at a time, broadcast to subscribers.

``POST /api/index/stream`` blocks the request for the whole job and the
frontend locks its form while that stream is open, so a second video or
channel cannot be queued until the first finishes. This module decouples
submission from execution: :meth:`IngestionQueue.enqueue` returns immediately
with a job id, a single background worker thread drains jobs in submission
order (``bulk-index`` only supports concurrency 1 anyway, so serial execution
loses nothing), and every job's progress is broadcast to every subscriber —
so the queue view stays live across multiple browser tabs, not just the one
that submitted a job.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

JobStatus = str  # "queued" | "running" | "done" | "error"

#: How often a still-running job re-broadcasts its "processing" message, so a
#: subscribing SSE connection is never silent long enough to look dead.
DEFAULT_HEARTBEAT_SECONDS = 8.0


@dataclass
class IngestionJob:
    """One queued ingestion request and its live progress."""

    id: str
    mode: str  # "video" | "channel"
    target: str
    argv: list[str]
    latest: int | None = None
    status: JobStatus = "queued"
    stage: str | None = None
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
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
    ) -> None:
        self._index_fn = index_fn
        self._corpus_fn = corpus_fn
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
        self._worker = threading.Thread(target=self._run, daemon=True, name="ingestion-queue")
        self._worker.start()

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
        job.status = "running"
        job.stage = "discover"
        job.message = "Resolving target(s) ..."
        self._broadcast_job(job)
        try:
            before = self._corpus_fn()
            before_ids = {v["video_id"] for v in before.get("videos", [])}

            job.stage = "fetch"
            job.message = "Fetching transcripts and building the index ..."
            self._broadcast_job(job)

            job.stage = "processing"
            job.message = "Chunking, embedding, and summarizing ..."
            self._broadcast_job(job)
            exit_code = self._run_blocking(
                job, lambda: self._index_fn(job.argv), "Still indexing..."
            )

            if exit_code != 0:
                job.status = "error"
                job.error = (
                    f"Indexing failed (exit {exit_code}). Check the server log for the CLI output."
                )
                self._broadcast_job(job)
                return

            after = self._corpus_fn()
            added = [v for v in after.get("videos", []) if v["video_id"] not in before_ids]
            job.result = {
                "ok": True,
                "target": job.target,
                "added_videos": added,
                "added_video_count": len(added),
                "added_chunk_count": (
                    after.get("totals", {}).get("chunks", 0)
                    - before.get("totals", {}).get("chunks", 0)
                ),
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
            job.message = None
            self._broadcast_job(job)
        except Exception as exc:  # a broken job must not stop the worker
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
