"""Write and revise field guides from the workbench, with live progress.

The same shape as :mod:`src.api.matrix_runner` — one job at a time, a daemon
thread, per-connection subscriber queues, a heartbeat so an SSE connection
is never silent — because the constraints are the same: a guide run is ten
to twenty-five minutes of agent sessions against one Chroma path, and two at
once would double the load for no useful meaning of "queue another".

What differs is what a job carries. A guide run is a sequence of named
stages (scope → export → extract → compose → verify → publish, or the
revision's compose → verify → publish), each with counters the Research map
draws, plus a bounded activity log of what the agent did: every file read,
every ``retrieve_chunks`` question, every gap it flagged. All of it is
derived from the writer's event stream, so the runner adds no state the run
log on disk does not already hold.
"""

from __future__ import annotations

import datetime as dt
import queue
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_HEARTBEAT_SECONDS = 8.0
ACTIVITY_LIMIT = 200

#: ``run_fn(job, on_event) -> result dict`` — the writer, behind a seam so the
#: routes are testable without an agent. ``on_event`` takes the writer's
#: stage/tool events exactly as :class:`src.guides.writer.GuideWriter` emits them.
RunFn = Callable[["GuideJob", Callable[[dict[str, Any]], None]], dict[str, Any]]

WRITE_STAGES = ["scope", "export", "extract", "compose", "verify", "publish"]
REVISE_STAGES = ["revise", "verify", "publish"]
MARKDOWN_STAGES = ["markdown"]


@dataclass
class GuideJob:
    """One guide run and its live progress."""

    id: str
    kind: str  # "write" | "revise" | "markdown"
    slug: str
    topic: str = ""
    title: str = ""
    video_ids: list[str] = field(default_factory=list)
    allow_web: bool = False
    comment_ids: list[str] = field(default_factory=list)
    status: str = "running"  # "running" | "done" | "error"
    stage: str | None = None
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    message: str | None = None
    counters: dict[str, int] = field(default_factory=dict)
    clusters: dict[str, dict[str, Any]] = field(default_factory=dict)
    activity: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=ACTIVITY_LIMIT))
    gaps: list[str] = field(default_factory=list)
    version: int | None = None
    cite_valid: int | None = None
    cite_total: int | None = None
    error: str | None = None
    started_at: str = ""
    finished_at: str | None = None

    def stage_order(self) -> list[str]:
        if self.kind == "revise":
            return list(REVISE_STAGES)
        if self.kind == "markdown":
            return list(MARKDOWN_STAGES)
        return list(WRITE_STAGES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "slug": self.slug,
            "topic": self.topic,
            "title": self.title,
            "video_ids": list(self.video_ids),
            "allow_web": self.allow_web,
            "comment_ids": list(self.comment_ids),
            "status": self.status,
            "stage": self.stage,
            "stage_order": self.stage_order(),
            "stages": {name: dict(value) for name, value in self.stages.items()},
            "message": self.message,
            "counters": dict(self.counters),
            "clusters": {name: dict(value) for name, value in self.clusters.items()},
            "activity": list(self.activity),
            "gaps": list(self.gaps),
            "version": self.version,
            "cite_valid": self.cite_valid,
            "cite_total": self.cite_total,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class GuideRunner:
    """Runs one guide job at a time and broadcasts progress to subscribers."""

    def __init__(self, run_fn: RunFn, heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS) -> None:
        self._run_fn = run_fn
        self._heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._job: GuideJob | None = None
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def is_running(self) -> bool:
        with self._lock:
            return self._job is not None and self._job.status == "running"

    def current(self) -> GuideJob | None:
        with self._lock:
            return self._job

    def start(
        self,
        *,
        kind: str,
        slug: str,
        topic: str = "",
        title: str = "",
        video_ids: list[str] | None = None,
        allow_web: bool = False,
        comment_ids: list[str] | None = None,
    ) -> tuple[GuideJob, bool]:
        """Begin a job. Returns ``(job, started)``; ``started`` is False when
        one is already running, in which case that job is returned."""
        with self._lock:
            if self._job is not None and self._job.status == "running":
                return self._job, False
            job = GuideJob(
                id=uuid.uuid4().hex[:12],
                kind=kind,
                slug=slug,
                topic=topic,
                title=title,
                video_ids=list(video_ids or []),
                allow_web=allow_web,
                comment_ids=list(comment_ids or []),
                message=f"Starting {kind} for {slug} ...",
                started_at=_now(),
            )
            self._job = job
        self._broadcast(job)
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"guide-{kind}").start()
        return job, True

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return self._job.to_dict() if self._job is not None else None

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        subscriber.put({"type": "snapshot", "job": self.snapshot()})
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _broadcast(self, job: GuideJob, activity: dict[str, Any] | None = None) -> None:
        with self._lock:
            event = (
                {"type": "activity", "job_id": job.id, "event": activity}
                if activity is not None
                else {"type": "job", "job": job.to_dict()}
            )
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

    # -- event folding -------------------------------------------------------
    def _on_event(self, job: GuideJob, event: dict[str, Any]) -> None:
        """Fold one writer event into the job; tool events go to the activity log."""
        status = str(event.get("status", ""))
        stage = str(event.get("stage", ""))
        if status == "tool":
            entry = {
                "at": event.get("at", _now()),
                "label": stage,
                "name": event.get("name") or event.get("tool") or "",
                "message": event.get("message", ""),
                "question": event.get("question"),
                "results": event.get("results"),
            }
            with self._lock:
                job.activity.append(entry)
                job.counters["tool_calls"] = job.counters.get("tool_calls", 0) + 1
                name = str(entry["name"])
                if name.endswith("retrieve_chunks") and entry.get("question") is not None:
                    job.counters["retrieval_queries"] = job.counters.get("retrieval_queries", 0) + 1
                elif name in ("WebSearch", "WebFetch"):
                    job.counters["web_fetches"] = job.counters.get("web_fetches", 0) + 1
                elif name == "Read":
                    job.counters["files_read"] = job.counters.get("files_read", 0) + 1
            self._broadcast(job, activity=entry)
            return

        with self._lock:
            record = job.stages.setdefault(stage, {"status": "pending"})
            record["status"] = status
            record["message"] = event.get("message", "")
            record["at"] = event.get("at", _now())
            if status in ("start", "progress"):
                job.stage = stage
            if stage != "run":
                job.message = event.get("message") or job.message
            data = {k: v for k, v in event.items() if k not in ("at", "stage", "status", "message")}
            if data:
                record["data"] = {**record.get("data", {}), **data}
            if stage == "export" and "chunk_count" in event:
                job.counters["chunk_count"] = int(event["chunk_count"])
                if "clusters" in event:
                    job.counters["clusters"] = int(event["clusters"])
                for video in event.get("videos") or []:
                    if isinstance(video, dict):
                        job.clusters.setdefault("videos", {})[str(video.get("video_id"))] = video
            if stage == "extract":
                cluster = event.get("cluster")
                if cluster:
                    entry = job.clusters.setdefault(str(cluster), {"status": "pending", "claims": 0})
                    entry["status"] = "done" if "claims" in event else ("skipped" if status == "skip" else "reading")
                    if "claims" in event:
                        entry["claims"] = int(event["claims"])
                        job.counters["clusters_done"] = job.counters.get("clusters_done", 0) + 1
                if status == "start":
                    job.counters["clusters"] = int(event.get("total", job.counters.get("clusters", 0)))
                if status == "done":
                    job.counters["claims"] = int(event.get("claims", 0))
            if stage == "verify" and status == "done":
                job.cite_valid = int(event.get("valid", 0))
                job.cite_total = int(event.get("total", 0))
            if stage == "publish" and status == "done":
                job.version = int(event.get("version", 0)) or None
            if status == "error" and stage != "run":
                job.error = event.get("message") or job.error
        self._broadcast(job)

    def _run(self, job: GuideJob) -> None:
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        holder: dict[str, Any] = {}

        def worker() -> None:
            try:
                holder["result"] = self._run_fn(job, lambda event: self._on_event(job, event))
            except Exception as exc:  # noqa: BLE001 - surfaced on the job
                holder["error"] = exc
            finally:
                events.put(("finished", None))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            try:
                kind, _value = events.get(timeout=self._heartbeat_seconds)
            except queue.Empty:
                self._broadcast(job)
                continue
            if kind == "finished":
                break
        thread.join()

        with self._lock:
            job.finished_at = _now()
            if "error" in holder:
                job.status = "error"
                job.error = str(holder["error"])
                job.message = None
            else:
                result = holder.get("result") or {}
                if result.get("version") is not None:
                    job.version = int(result["version"])
                if result.get("cite_valid") is not None:
                    job.cite_valid = int(result["cite_valid"])
                    job.cite_total = int(result.get("cite_total") or 0)
                job.gaps = list(result.get("gaps") or job.gaps)
                job.message = None
                job.status = "done"
        self._broadcast(job)
