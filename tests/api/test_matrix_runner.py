"""MatrixRunner: one run at a time, progress broadcast, failures contained."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from src.api.matrix_runner import MatrixRunner


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def instant_run(result: dict[str, Any] | None = None) -> Callable:
    def run_fn(setups: list[str], on_cell: Callable[[dict[str, Any]], None]) -> dict:
        return result or {"run_id": "matrix-test", "cache_hits": 0, "cache_misses": 0}

    return run_fn


def test_start_runs_and_records_the_committed_run_id() -> None:
    runner = MatrixRunner(
        run_fn=instant_run({"run_id": "matrix-abc", "cache_hits": 3, "cache_misses": 1})
    )
    runner.start(["rag_llm"])

    wait_until(lambda: runner.snapshot()["status"] == "done")
    job = runner.snapshot()
    assert job["run_id"] == "matrix-abc"
    assert job["cache_hits"] == 3
    assert job["cache_misses"] == 1
    assert job["error"] is None


def test_progress_cells_are_counted_and_broadcast() -> None:
    def run_fn(setups: list[str], on_cell) -> dict:
        on_cell({"setup": "rag_llm", "entry_id": "g001", "cached": True, "done": 1, "total": 2})
        on_cell({"setup": "rag_llm", "entry_id": "g002", "cached": False, "done": 2, "total": 2})
        return {"run_id": "matrix-abc"}

    runner = MatrixRunner(run_fn=run_fn)
    subscriber = runner.subscribe()
    assert subscriber.get(timeout=1)["type"] == "snapshot"

    runner.start(["rag_llm"])
    wait_until(lambda: runner.snapshot()["status"] == "done")

    job = runner.snapshot()
    assert job["cells_done"] == 2
    assert job["cells_total"] == 2

    statuses = []
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and (not statuses or statuses[-1] != "done"):
        statuses.append(subscriber.get(timeout=1)["job"]["status"])
    assert statuses[-1] == "done"


def test_starting_while_running_returns_the_run_in_flight() -> None:
    release = threading.Event()

    def slow_run(setups: list[str], on_cell) -> dict:
        release.wait(timeout=2)
        return {"run_id": "matrix-abc"}

    runner = MatrixRunner(run_fn=slow_run, heartbeat_seconds=0.05)
    try:
        first = runner.start(["rag_llm"])
        wait_until(lambda: runner.is_running())
        second = runner.start(["rag_agent"])
        # A second press must not launch a second sweep against the same cache.
        assert second.id == first.id
        assert second.setups == ["rag_llm"]
    finally:
        release.set()


def test_a_failed_run_is_reported_without_killing_the_runner() -> None:
    def boom(setups: list[str], on_cell) -> dict:
        raise RuntimeError("deepseek 402")

    runner = MatrixRunner(run_fn=boom)
    runner.start(["rag_llm"])
    wait_until(lambda: runner.snapshot()["status"] == "error")
    assert "deepseek 402" in runner.snapshot()["error"]

    # The runner is still usable afterwards.
    ok = MatrixRunner(run_fn=instant_run())
    ok.start(["rag_llm"])
    wait_until(lambda: ok.snapshot()["status"] == "done")


def test_snapshot_is_none_before_any_run() -> None:
    assert MatrixRunner(run_fn=instant_run()).snapshot() is None


def test_subscriber_is_seeded_then_unsubscribed_cleanly() -> None:
    runner = MatrixRunner(run_fn=instant_run())
    subscriber = runner.subscribe()
    assert subscriber.get(timeout=1) == {"type": "snapshot", "job": None}

    runner.unsubscribe(subscriber)
    runner.start(["rag_llm"])
    wait_until(lambda: runner.snapshot()["status"] == "done")
    assert subscriber.empty()


def test_heartbeat_rebroadcasts_while_a_cell_is_slow() -> None:
    release = threading.Event()

    def slow_run(setups: list[str], on_cell) -> dict:
        release.wait(timeout=2)
        return {"run_id": "matrix-abc"}

    runner = MatrixRunner(run_fn=slow_run, heartbeat_seconds=0.05)
    subscriber = runner.subscribe()
    subscriber.get(timeout=1)  # seed
    try:
        runner.start(["rag_llm"])
        # Without the heartbeat the stream would be silent for the whole cell.
        first = subscriber.get(timeout=1)
        second = subscriber.get(timeout=1)
        assert first["job"]["status"] == "running"
        assert second["job"]["status"] == "running"
    finally:
        release.set()
