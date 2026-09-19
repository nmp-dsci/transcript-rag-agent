"""IngestionQueue: enqueue never blocks, jobs run in order, progress broadcasts."""

from __future__ import annotations

import threading
import time

from src.api.ingestion_queue import IngestionJob, IngestionQueue


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
        # Pinned to one worker: this test is about submission never blocking
        # on execution, and it proves that by checking the second job is still
        # queued. With the production pool a second worker would legitimately
        # start it, which would break the observation without breaking the
        # property. Overlap is covered by TestMultipleWorkers below.
        max_workers=1,
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


def test_a_single_worker_processes_in_submission_order() -> None:
    order: list[str] = []

    def index_fn(argv: list[str]) -> int:
        order.append(argv[-1])
        return 0

    queue_ = IngestionQueue(
        index_fn=index_fn,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
        max_workers=1,
    )
    queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])
    queue_.enqueue(mode="video", target="b", argv=["index-rag", "b"])
    queue_.enqueue(mode="video", target="c", argv=["index-rag", "c"])

    wait_until(lambda: order == ["a", "b", "c"])
    wait_until(lambda: all(j["status"] == "done" for j in queue_.snapshot()))


def test_a_worker_pool_runs_every_job_exactly_once() -> None:
    """Completion *order* is not a promise a pool can keep — and should not be.

    Jobs are still dequeued FIFO, but three workers finish three jobs in
    whatever order they finish. What must hold is that none is dropped and
    none runs twice.
    """
    order: list[str] = []
    guard = threading.Lock()

    def index_fn(argv: list[str]) -> int:
        with guard:
            order.append(argv[-1])
        return 0

    queue_ = IngestionQueue(
        index_fn=index_fn,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
        max_workers=3,
    )
    for name in ("a", "b", "c"):
        queue_.enqueue(mode="video", target=name, argv=["index-rag", name])

    wait_until(lambda: all(j["status"] == "done" for j in queue_.snapshot()))
    assert sorted(order) == ["a", "b", "c"]


def test_completed_job_reports_added_videos_and_totals() -> None:
    before = {"videos": [], "totals": {"videos": 0, "chunks": 0}}
    after = {
        "videos": [{"video_id": "abc123", "chunk_count": 5}],
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
        # One worker, so the exit codes map to the jobs deterministically.
        # With a pool both workers race on `next(exit_codes)` and which job
        # fails becomes a coin toss — which would be testing the fixture, not
        # the queue.
        max_workers=1,
    )
    queue_.enqueue(mode="video", target="bad", argv=["index-rag", "bad"])
    queue_.enqueue(mode="video", target="good", argv=["index-rag", "good"])

    wait_until(lambda: len(queue_.snapshot()) == 2 and queue_.snapshot()[1]["status"] == "done")
    failed, succeeded = queue_.snapshot()
    assert failed["status"] == "error"
    assert "exit 1" in failed["error"]
    assert succeeded["status"] == "done"


def test_failure_hint_replaces_the_generic_exit_message() -> None:
    hints = iter(
        ["Out of Supadata credits — all 2 configured keys reported their plan usage limit", None]
    )

    queue_ = IngestionQueue(
        index_fn=lambda argv: 1,
        corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
        max_workers=1,
        failure_hint=lambda: lambda: next(hints),
    )
    queue_.enqueue(mode="video", target="quota", argv=["index-rag", "quota"])
    queue_.enqueue(mode="video", target="other", argv=["index-rag", "other"])

    wait_until(lambda: len(queue_.snapshot()) == 2 and queue_.snapshot()[1]["status"] == "error")
    quota, other = queue_.snapshot()
    assert quota["error"].startswith("Out of Supadata credits")
    assert "exit 1" in other["error"]  # no hint → the generic message as before


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


class TestEnrichmentJobs:
    """Graph catch-up runs through the same queue as an ingest.

    Same worker, same broadcasting, same failure isolation — rather than a
    second execution path beside it that fails in its own untested ways.
    """

    def test_an_enrichment_job_extracts_only_the_videos_it_was_given(self) -> None:
        seen: list[list[str]] = []

        def graph_fn(video_ids: list[str]) -> dict:
            seen.append(list(video_ids))
            return {"ok": True, "extracted": len(video_ids), "failed": 0}

        q = IngestionQueue(
            index_fn=lambda argv: 0,
            corpus_fn=lambda: {"videos": [], "totals": {}},
            graph_fn=graph_fn,
            heartbeat_seconds=0.01,
        )
        job = q.enqueue_enrichment(["a", "b"])
        _wait_for(q, job.id, "done")

        assert seen == [["a", "b"]]
        assert q_job(q, job.id)["result"]["graph"]["extracted"] == 2

    def test_it_indexes_nothing(self) -> None:
        """Enrichment enriches what indexing already wrote — it never re-indexes."""
        indexed: list[list[str]] = []

        q = IngestionQueue(
            index_fn=lambda argv: indexed.append(argv) or 0,
            corpus_fn=lambda: {"videos": [], "totals": {}},
            graph_fn=lambda ids: {"ok": True, "failed": 0},
            heartbeat_seconds=0.01,
        )
        job = q.enqueue_enrichment(["a"])
        _wait_for(q, job.id, "done")

        assert indexed == []

    def test_a_failing_extractor_errors_the_job_without_killing_the_worker(self) -> None:
        calls = {"n": 0}

        def graph_fn(video_ids: list[str]) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Error code: 402 - Insufficient Balance")
            return {"ok": True, "failed": 0}

        q = IngestionQueue(
            index_fn=lambda argv: 0,
            corpus_fn=lambda: {"videos": [], "totals": {}},
            graph_fn=graph_fn,
            heartbeat_seconds=0.01,
        )
        first = q.enqueue_enrichment(["a"])
        _wait_for(q, first.id, "error")
        assert "402" in q_job(q, first.id)["error"]

        # The worker survived: the next job still runs.
        second = q.enqueue_enrichment(["b"])
        _wait_for(q, second.id, "done")

    def test_no_configured_extractor_is_reported_not_crashed(self) -> None:
        q = IngestionQueue(
            index_fn=lambda argv: 0,
            corpus_fn=lambda: {"videos": [], "totals": {}},
            graph_fn=None,
            heartbeat_seconds=0.01,
        )
        job = q.enqueue_enrichment(["a"])
        _wait_for(q, job.id, "error")
        assert "graph extractor" in q_job(q, job.id)["error"]


def q_job(q: IngestionQueue, job_id: str) -> dict:
    return next(j for j in q.snapshot() if j["id"] == job_id)


def _wait_for(q: IngestionQueue, job_id: str, status: str, timeout: float = 5.0) -> None:
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if q_job(q, job_id)["status"] == status:
            return
        _time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status}: {q_job(q, job_id)}")


def test_enrichment_reports_failure_even_when_the_extractor_returns_normally() -> None:
    """`ok: False` is a failed enrichment job, not a quiet success.

    For an ingest a failed graph is a footnote — the vector index already
    succeeded. For an enrichment job the graph *is* the job, so the same
    result has to read as a failure.
    """
    q = IngestionQueue(
        index_fn=lambda argv: 0,
        corpus_fn=lambda: {"videos": [], "totals": {}},
        graph_fn=lambda ids: {"ok": False, "extracted": 0, "failed": 26},
        heartbeat_seconds=0.01,
    )
    job = q.enqueue_enrichment(["a"])
    _wait_for(q, job.id, "error")

    error = q_job(q, job.id)["error"]
    assert "26 chunk(s)" in error
    # And it says the corpus is fine, because it is.
    assert "still retrievable" in error


class TestMultipleWorkers:
    """Indexing is mostly network wait, so one worker leaves it fully exposed."""

    def test_jobs_actually_overlap(self) -> None:
        started = threading.Semaphore(0)
        release = threading.Event()
        concurrent: list[int] = []
        live = {"n": 0}
        guard = threading.Lock()

        def slow_index_fn(argv: list[str]) -> int:
            with guard:
                live["n"] += 1
                concurrent.append(live["n"])
            started.release()
            release.wait(timeout=3)
            with guard:
                live["n"] -= 1
            return 0

        queue_ = IngestionQueue(
            index_fn=slow_index_fn,
            corpus_fn=lambda: {"videos": [], "totals": {"videos": 0, "chunks": 0}},
            heartbeat_seconds=0.05,
            max_workers=3,
        )
        try:
            for name in ("a", "b", "c"):
                queue_.enqueue(mode="video", target=name, argv=["index-rag", name])
            for _ in range(3):
                assert started.acquire(timeout=3), "a worker never picked up its job"
            assert max(concurrent) == 3
        finally:
            release.set()

    def test_the_pool_size_is_configurable(self) -> None:
        queue_ = IngestionQueue(
            index_fn=lambda argv: 0,
            corpus_fn=lambda: {"videos": [], "totals": {}},
            max_workers=5,
        )
        assert queue_.max_workers == 5

    def test_a_zero_or_negative_pool_still_gets_one_worker(self) -> None:
        """A misconfigured worker count must not silently stop all ingestion."""
        queue_ = IngestionQueue(
            index_fn=lambda argv: 0,
            corpus_fn=lambda: {"videos": [], "totals": {}},
            max_workers=0,
        )
        assert queue_.max_workers == 1
        job = queue_.enqueue(mode="video", target="a", argv=["index-rag", "a"])
        _wait_for(queue_, job.id, "done")


class TestAddedVideoAttribution:
    """With one worker, "everything new while I ran" was exact. Not any more.

    Driven directly rather than through the worker pool: the behaviour under
    test is how overlapping before/after windows are resolved, and forcing a
    real race to reproduce a specific interleaving tests the fixture's timing
    rather than the rule.
    """

    def _queue(self) -> IngestionQueue:
        return IngestionQueue(
            index_fn=lambda argv: 0,
            corpus_fn=lambda: {"videos": [], "totals": {}},
            max_workers=1,
        )

    def _job(self, mode: str, target: str) -> IngestionJob:
        return IngestionJob(id="j", mode=mode, target=target, argv=[])

    def test_a_single_video_job_claims_only_its_own_video(self) -> None:
        """Its neighbour's video appeared in the same window. Not its."""
        queue_ = self._queue()
        claimed = queue_._attribute_added(
            self._job("video", "https://www.youtube.com/watch?v=aaaaaaaaaaa"),
            set(),
            [{"video_id": "aaaaaaaaaaa"}, {"video_id": "bbbbbbbbbbb"}],
        )
        assert [v["video_id"] for v in claimed] == ["aaaaaaaaaaa"]

    def test_two_jobs_never_claim_the_same_video(self) -> None:
        """The failure this prevents: both jobs reporting "+2 videos"."""
        queue_ = self._queue()
        after = [{"video_id": "aaaaaaaaaaa"}, {"video_id": "bbbbbbbbbbb"}]
        first = queue_._attribute_added(
            self._job("video", "https://youtu.be/aaaaaaaaaaa"), set(), after
        )
        second = queue_._attribute_added(
            self._job("video", "https://youtu.be/bbbbbbbbbbb"), set(), after
        )
        claimed = [v["video_id"] for v in first + second]
        assert sorted(claimed) == ["aaaaaaaaaaa", "bbbbbbbbbbb"]

    def test_a_second_job_cannot_re_claim_a_video(self) -> None:
        queue_ = self._queue()
        after = [{"video_id": "aaaaaaaaaaa"}]
        job = self._job("video", "https://youtu.be/aaaaaaaaaaa")
        assert len(queue_._attribute_added(job, set(), after)) == 1
        assert queue_._attribute_added(job, set(), after) == []

    def test_a_channel_job_claims_everything_unclaimed(self) -> None:
        """It cannot know its own video ids, so first finisher wins."""
        queue_ = self._queue()
        after = [{"video_id": "aaaaaaaaaaa"}, {"video_id": "bbbbbbbbbbb"}]
        claimed = queue_._attribute_added(self._job("channel", "@someone"), set(), after)
        assert len(claimed) == 2

    def test_an_unparseable_target_falls_back_rather_than_claiming_nothing(self) -> None:
        """A job that cannot identify itself should still report something true."""
        queue_ = self._queue()
        claimed = queue_._attribute_added(
            self._job("video", "not-a-url"), set(), [{"video_id": "aaaaaaaaaaa"}]
        )
        assert [v["video_id"] for v in claimed] == ["aaaaaaaaaaa"]

    def test_videos_present_before_the_job_are_never_claimed(self) -> None:
        queue_ = self._queue()
        claimed = queue_._attribute_added(
            self._job("video", "https://youtu.be/aaaaaaaaaaa"),
            {"aaaaaaaaaaa"},
            [{"video_id": "aaaaaaaaaaa"}],
        )
        assert claimed == []

    def test_added_chunk_count_sums_only_claimed_videos(self) -> None:
        """The failure this prevents: a job reporting another job's chunks."""
        queue_ = self._queue()
        after = [
            {"video_id": "aaaaaaaaaaa", "chunk_count": 5},
            {"video_id": "bbbbbbbbbbb", "chunk_count": 9},
        ]
        claimed = queue_._attribute_added(
            self._job("video", "https://youtu.be/aaaaaaaaaaa"), set(), after
        )
        added_chunk_count = sum(int(v.get("chunk_count") or 0) for v in claimed)
        assert added_chunk_count == 5

    def test_added_chunk_count_is_zero_when_nothing_is_claimed(self) -> None:
        """Re-indexing an existing video claims nothing: "+0 chunks", not a totals delta."""
        queue_ = self._queue()
        claimed = queue_._attribute_added(
            self._job("video", "https://youtu.be/aaaaaaaaaaa"),
            {"aaaaaaaaaaa"},
            [{"video_id": "aaaaaaaaaaa", "chunk_count": 5}],
        )
        added_chunk_count = sum(int(v.get("chunk_count") or 0) for v in claimed)
        assert added_chunk_count == 0
