"""IngestionQueue: enqueue never blocks, jobs run in order, progress broadcasts."""

from __future__ import annotations

import threading
import time

from src.api.ingestion_queue import IngestionQueue


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def test_enqueue_returns_immediately_even_while_a_job_is_running() -> None:
    release = threading.Event()
    calls: list[list[str]] = []

    def slow_index_fn(argv: list[str]) -> int:
        calls.append(argv)
        release.wait(timeout=2)
        return 0

    queue_ = IngestionQueue(
        index_fn=slow_index_fn,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
        heartbeat_seconds=0.05,
    )
    try:
        started = time.monotonic()
        first = queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])
        second = queue_.enqueue(mode="video", target="b", argv=["index-rag", "b"])
        elapsed = time.monotonic() - started

        # Both enqueue calls return well before the slow job finishes — this
        # is the whole point: submission never blocks on execution. The bound
        # is half the job's 2s block rather than something tight: it only has
        # to catch an enqueue that *joins* the worker, and a tighter figure
        # fails on a loaded machine for reasons that say nothing about the
        # queue. The real proof is below — the job is still running and only
        # the first has reached index_fn.
        assert elapsed < 1.0
        # The worker thread races with this assertion: it may already have
        # picked the first job off the queue and flipped it to "running" by
        # the time we check, so either status is consistent with enqueue not
        # blocking. What must not happen is the second job starting early.
        assert first.status in ("queued", "running")
        assert second.status == "queued"

        wait_until(lambda: queue_.snapshot()[0]["status"] == "running")
        # The second job must not start until the first releases.
        assert queue_.snapshot()[1]["status"] == "queued"
        assert len(calls) == 1
    finally:
        release.set()


def test_jobs_process_in_submission_order() -> None:
    order: list[str] = []

    def index_fn(argv: list[str]) -> int:
        order.append(argv[-1])
        return 0

    queue_ = IngestionQueue(
        index_fn=index_fn,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
    )
    queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])
    queue_.enqueue(mode="video", target="b", argv=["index-rag", "b"])
    queue_.enqueue(mode="video", target="c", argv=["index-rag", "c"])

    wait_until(lambda: order == ["a", "b", "c"])
    wait_until(lambda: all(j["status"] == "done" for j in queue_.snapshot()))


def test_completed_job_reports_added_videos_and_totals() -> None:
    before = {"videos": [], "totals": {"videos": 0, "chunks": 0}}
    after = {
        "videos": [{"video_id": "abc123"}],
        "totals": {"videos": 1, "chunks": 5},
        "insights": ["insight"],
        "channels": ["chan"],
    }
    calls = {"n": 0}

    def corpus_fn() -> dict:
        calls["n"] += 1
        return before if calls["n"] == 1 else after

    queue_ = IngestionQueue(index_fn=lambda argv: 0, corpus_fn=corpus_fn)
    job = queue_.enqueue(mode="video", target="abc123", argv=["index-rag", "abc123"])

    wait_until(lambda: queue_.snapshot()[0]["status"] == "done")
    done = queue_.snapshot()[0]
    assert done["id"] == job.id
    assert done["result"]["added_video_count"] == 1
    assert done["result"]["added_chunk_count"] == 5
    assert done["result"]["insights"] == ["insight"]
    assert done["result"]["channels"] == ["chan"]


def test_nonzero_exit_code_marks_job_errored_without_stopping_the_worker() -> None:
    exit_codes = iter([1, 0])

    def index_fn(argv: list[str]) -> int:
        return next(exit_codes)

    queue_ = IngestionQueue(
        index_fn=index_fn,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
    )
    queue_.enqueue(mode="video", target="bad", argv=["index-rag", "bad"])
    queue_.enqueue(mode="video", target="good", argv=["index-rag", "good"])

    wait_until(lambda: len(queue_.snapshot()) == 2 and queue_.snapshot()[1]["status"] == "done")
    failed, succeeded = queue_.snapshot()
    assert failed["status"] == "error"
    assert "exit 1" in failed["error"]
    assert succeeded["status"] == "done"


def test_exception_in_index_fn_marks_job_errored() -> None:
    def index_fn(argv: list[str]) -> int:
        raise RuntimeError("boom")

    queue_ = IngestionQueue(
        index_fn=index_fn,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
    )
    queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])

    wait_until(lambda: queue_.snapshot()[0]["status"] == "error")
    assert "boom" in queue_.snapshot()[0]["error"]


def test_subscriber_is_seeded_with_the_current_snapshot() -> None:
    queue_ = IngestionQueue(
        index_fn=lambda argv: 0,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
    )
    queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])
    wait_until(lambda: queue_.snapshot()[0]["status"] == "done")

    subscriber = queue_.subscribe()
    seed = subscriber.get(timeout=1)
    assert seed["type"] == "snapshot"
    assert seed["jobs"][0]["target"] == "a"


def test_subscriber_receives_progress_events_for_new_jobs() -> None:
    queue_ = IngestionQueue(
        index_fn=lambda argv: 0,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
    )
    subscriber = queue_.subscribe()
    assert subscriber.get(timeout=1)["type"] == "snapshot"  # initial seed

    queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])

    seen_statuses = []
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and "done" not in seen_statuses:
        event = subscriber.get(timeout=1)
        assert event["type"] == "job"
        seen_statuses.append(event["job"]["status"])
    assert "queued" in seen_statuses
    assert "running" in seen_statuses
    assert seen_statuses[-1] == "done"


def test_graph_fn_runs_after_a_successful_job_with_added_videos() -> None:
    before = {"videos": [], "totals": {"videos": 0, "chunks": 0}}
    after = {"videos": [{"video_id": "abc123"}], "totals": {"videos": 1, "chunks": 5}}
    calls = {"n": 0}
    graph_calls: list[list[str]] = []

    def corpus_fn() -> dict:
        calls["n"] += 1
        return before if calls["n"] == 1 else after

    def graph_fn(video_ids: list[str]) -> dict:
        graph_calls.append(video_ids)
        return {"ok": True, "extracted": 3}

    queue_ = IngestionQueue(index_fn=lambda argv: 0, corpus_fn=corpus_fn, graph_fn=graph_fn)
    queue_.enqueue(mode="video", target="abc123", argv=["index-rag", "abc123"])

    wait_until(lambda: queue_.snapshot()[0]["status"] == "done")
    done = queue_.snapshot()[0]
    assert graph_calls == [["abc123"]]
    assert done["result"]["graph"] == {"ok": True, "extracted": 3}


def test_graph_fn_is_skipped_when_nothing_new_was_added() -> None:
    same = {"videos": [{"video_id": "existing"}], "totals": {"videos": 1, "chunks": 5}}
    graph_calls: list[list[str]] = []

    queue_ = IngestionQueue(
        index_fn=lambda argv: 0,
        corpus_fn=lambda: same,
        graph_fn=lambda video_ids: graph_calls.append(video_ids) or {"ok": True},
    )
    queue_.enqueue(mode="video", target="existing", argv=["index-rag", "existing"])

    wait_until(lambda: queue_.snapshot()[0]["status"] == "done")
    assert graph_calls == []
    assert "graph" not in queue_.snapshot()[0]["result"]


def test_graph_fn_failure_does_not_fail_the_job() -> None:
    before = {"videos": [], "totals": {"videos": 0, "chunks": 0}}
    after = {"videos": [{"video_id": "abc123"}], "totals": {"videos": 1, "chunks": 5}}
    calls = {"n": 0}

    def corpus_fn() -> dict:
        calls["n"] += 1
        return before if calls["n"] == 1 else after

    def broken_graph_fn(video_ids: list[str]) -> dict:
        raise RuntimeError("neo4j is down")

    queue_ = IngestionQueue(index_fn=lambda argv: 0, corpus_fn=corpus_fn, graph_fn=broken_graph_fn)
    queue_.enqueue(mode="video", target="abc123", argv=["index-rag", "abc123"])

    wait_until(lambda: queue_.snapshot()[0]["status"] == "done")
    done = queue_.snapshot()[0]
    # The vector index already succeeded — a broken graph extraction is
    # enrichment-only and must not flip a good job to "error".
    assert done["status"] == "done"
    assert done["result"]["graph"]["ok"] is False
    assert "neo4j is down" in done["result"]["graph"]["error"]


def test_no_graph_fn_configured_leaves_result_unchanged() -> None:
    before = {"videos": [], "totals": {"videos": 0, "chunks": 0}}
    after = {"videos": [{"video_id": "abc123"}], "totals": {"videos": 1, "chunks": 5}}
    calls = {"n": 0}

    def corpus_fn() -> dict:
        calls["n"] += 1
        return before if calls["n"] == 1 else after

    queue_ = IngestionQueue(index_fn=lambda argv: 0, corpus_fn=corpus_fn)
    queue_.enqueue(mode="video", target="abc123", argv=["index-rag", "abc123"])

    wait_until(lambda: queue_.snapshot()[0]["status"] == "done")
    assert "graph" not in queue_.snapshot()[0]["result"]


def test_unsubscribe_stops_further_events() -> None:
    queue_ = IngestionQueue(
        index_fn=lambda argv: 0,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
    )
    subscriber = queue_.subscribe()
    subscriber.get(timeout=1)  # seed
    queue_.unsubscribe(subscriber)

    queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])
    wait_until(lambda: queue_.snapshot()[0]["status"] == "done")
    assert subscriber.empty()
