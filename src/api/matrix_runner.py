"""Run the judged eval matrix from the workbench, with live progress.

``eval-matrix`` was CLI-only, which made the Scoreboard's data something you
had to leave the app to produce. This runs the same
:func:`~src.evals.matrix.run_matrix` in the background and broadcasts its
progress, so the Experiments tab can start a run and watch it finish.

Single-slot rather than a queue, unlike :mod:`src.api.ingestion_queue`: two
concurrent matrix runs would contend on the same cell cache and build a second
set of agent/judge stacks against one Chroma path, and there is no useful
meaning to "queue another full sweep" — starting one while another is running
returns the run already in flight instead.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable

#: How often a running job re-broadcasts its current message, so a subscribed
#: SSE connection is never silent long enough to look dead. A single cell can
#: take a minute or more to answer and judge.
DEFAULT_HEARTBEAT_SECONDS = 8.0

#: ``(setups, on_cell) -> the committed run dict``. Injected so tests drive the
#: whole path without an LLM, exactly like ``create_app``'s other seams.
RunFn = Callable[[list[str], Callable[[dict[str, Any]], None]], dict[str, Any]]


@dataclass
class MatrixJob:
    """One eval-matrix run and its live progress."""

    id: str
    setups: list[str]
    status: str = "running"  # "running" | "done" | "error"
    message: str | None = None
    cells_done: int = 0
    cells_total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    run_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MatrixRunner:
    """Runs one eval matrix at a time and broadcasts progress to subscribers."""

    def __init__(
        self,
        run_fn: RunFn,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self._run_fn = run_fn
        self._heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._job: MatrixJob | None = None
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def is_running(self) -> bool:
        with self._lock:
            return self._job is not None and self._job.status == "running"

    def start(self, setups: list[str]) -> MatrixJob:
        """Begin a run, or return the one already in flight.

        Returning the running job rather than raising keeps the button
        idempotent: a double click, or two browser tabs pressing it at once,
        both end up watching the same run.
        """
        with self._lock:
            if self._job is not None and self._job.status == "running":
                return self._job
            job = MatrixJob(
                id=uuid.uuid4().hex[:12],
                setups=list(setups),
                message="Starting eval matrix ...",
            )
            self._job = job
        self._broadcast(job)
        threading.Thread(target=self._run, args=(job,), daemon=True, name="eval-matrix").start()
        return job

    def snapshot(self) -> dict[str, Any] | None:
        """The current or most recent job, or ``None`` if none has ever run."""
        with self._lock:
            return self._job.to_dict() if self._job is not None else None

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        """A per-connection event queue, seeded with the current job.

        The seed means a client that opens the stream mid-run sees the run
        immediately rather than waiting for the next progress event.
        """
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        subscriber.put({"type": "snapshot", "job": self.snapshot()})
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _broadcast(self, job: MatrixJob) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        event = {"type": "job", "job": job.to_dict()}
        for subscriber in subscribers:
            subscriber.put(event)

    def _on_cell(self, job: MatrixJob, cell: dict[str, Any]) -> None:
        job.cells_done = int(cell.get("done", job.cells_done))
        job.cells_total = int(cell.get("total", job.cells_total))
        if cell.get("cached"):
            job.cache_hits += 1
        else:
            job.cache_misses += 1
        job.message = (
            f"{cell.get('setup')} × {cell.get('entry_id')} — "
            f"{'cached' if cell.get('cached') else 'scored'}"
        )
        self._broadcast(job)

    def _run(self, job: MatrixJob) -> None:
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        holder: dict[str, Any] = {}

        def worker() -> None:
            try:
                holder["result"] = self._run_fn(job.setups, lambda cell: self._on_cell(job, cell))
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
                # Re-broadcast the current state so a proxy never sees an idle
                # SSE connection while one cell is being answered and judged.
                self._broadcast(job)
                continue
            if kind == "finished":
                break
        thread.join()

        if "error" in holder:
            job.status = "error"
            job.error = str(holder["error"])
            job.message = None
        else:
            result = holder.get("result") or {}
            job.status = "done"
            job.run_id = result.get("run_id")
            job.message = None
            # Prefer the run's own tallies: they count every cell, including any
            # the runner's callback never saw.
            job.cache_hits = int(result.get("cache_hits", job.cache_hits))
            job.cache_misses = int(result.get("cache_misses", job.cache_misses))
        self._broadcast(job)
