from __future__ import annotations

import threading
import time

from src.api.guide_runner import ACTIVITY_LIMIT, GuideRunner


def drain(subscriber, until=lambda e: False, timeout=2.0):
    events = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            event = subscriber.get(timeout=0.05)
        except Exception:
            continue
        events.append(event)
        if until(event):
            break
    return events


def test_folds_writer_events_into_stages_counters_and_activity():
    release = threading.Event()

    def run_fn(job, on_event):
        on_event(
            {
                "at": "t",
                "stage": "scope",
                "status": "done",
                "message": "3 videos",
                "video_ids": ["a", "b", "c"],
            }
        )
        on_event({"at": "t", "stage": "export", "status": "start", "message": "exporting"})
        on_event(
            {
                "at": "t",
                "stage": "export",
                "status": "done",
                "message": "12 chunks",
                "chunk_count": 12,
                "clusters": 2,
                "videos": [{"video_id": "a", "title": "A", "chunk_count": 5}],
            }
        )
        on_event(
            {
                "at": "t",
                "stage": "extract",
                "status": "start",
                "message": "2 passes",
                "total": 2,
                "pending": 2,
            }
        )
        on_event(
            {
                "at": "t",
                "stage": "extract:cluster-1",
                "status": "tool",
                "message": "Read corpus/a.md",
                "name": "Read",
            }
        )
        on_event(
            {
                "at": "t",
                "stage": "extract:cluster-1",
                "status": "tool",
                "message": "q",
                "name": "mcp__corpus__retrieve_chunks",
                "question": "why?",
                "results": 3,
            }
        )
        on_event(
            {
                "at": "t",
                "stage": "extract",
                "status": "progress",
                "message": "cluster-1: reading",
                "cluster": "cluster-1",
            }
        )
        on_event(
            {
                "at": "t",
                "stage": "extract",
                "status": "progress",
                "message": "cluster-1: 20 verified claims",
                "cluster": "cluster-1",
                "claims": 20,
            }
        )
        on_event(
            {"at": "t", "stage": "extract", "status": "done", "message": "20 claims", "claims": 20}
        )
        on_event(
            {
                "at": "t",
                "stage": "verify",
                "status": "done",
                "message": "ok",
                "valid": 9,
                "total": 10,
            }
        )
        release.wait(2)
        on_event({"at": "t", "stage": "publish", "status": "done", "message": "v1", "version": 1})
        return {"version": 1, "cite_valid": 9, "cite_total": 10, "gaps": ["g1"]}

    runner = GuideRunner(run_fn, heartbeat_seconds=0.05)
    subscriber = runner.subscribe()
    job, started = runner.start(
        kind="write", slug="g", topic="t", title="T", video_ids=["a", "b", "c"]
    )
    assert started and job.kind == "write"
    again, started_again = runner.start(kind="write", slug="other")
    assert started_again is False and again.id == job.id  # one at a time
    events = drain(
        subscriber,
        until=lambda e: (
            e["type"] == "job" and e["job"]["stages"].get("verify", {}).get("status") == "done"
        ),
    )
    release.set()
    final = drain(subscriber, until=lambda e: e["type"] == "job" and e["job"]["status"] == "done")
    assert final[-1]["job"]["status"] == "done"
    snap = runner.snapshot()
    assert snap["stage_order"] == ["scope", "export", "extract", "compose", "verify", "publish"]
    assert snap["counters"] == {
        "chunk_count": 12,
        "clusters": 2,
        "tool_calls": 2,
        "files_read": 1,
        "retrieval_queries": 1,
        "clusters_done": 1,
        "claims": 20,
    }
    assert snap["clusters"]["cluster-1"] == {"status": "done", "claims": 20}
    assert snap["clusters"]["videos"]["a"]["title"] == "A"
    assert [a["name"] for a in snap["activity"]] == ["Read", "mcp__corpus__retrieve_chunks"]
    assert snap["cite_valid"] == 9 and snap["cite_total"] == 10 and snap["version"] == 1
    assert snap["gaps"] == ["g1"] and snap["finished_at"]
    activity = [e for e in events if e["type"] == "activity"]
    assert len(activity) == 2 and activity[1]["event"]["question"] == "why?"
    assert events[0]["type"] == "snapshot"


def test_error_in_run_fn_marks_the_job():
    def run_fn(job, on_event):
        on_event({"at": "t", "stage": "compose", "status": "start", "message": "composing"})
        raise RuntimeError("extract:cluster-2: rate_limit")

    runner = GuideRunner(run_fn, heartbeat_seconds=0.05)
    subscriber = runner.subscribe()
    runner.start(kind="write", slug="g")
    events = drain(
        subscriber, until=lambda e: e["type"] == "job" and e["job"]["status"] != "running"
    )
    job = events[-1]["job"]
    assert job["status"] == "error" and "rate_limit" in job["error"] and job["stage"] == "compose"
    # A new job may start once the failed one has finished.
    _, started = runner.start(kind="revise", slug="g")
    assert started


def test_activity_log_is_bounded():
    def run_fn(job, on_event):
        for index in range(ACTIVITY_LIMIT + 50):
            on_event(
                {
                    "at": "t",
                    "stage": "compose",
                    "status": "tool",
                    "message": f"Read {index}",
                    "name": "Read",
                }
            )
        return {}

    runner = GuideRunner(run_fn, heartbeat_seconds=0.05)
    subscriber = runner.subscribe()
    runner.start(kind="markdown", slug="g")
    drain(
        subscriber, until=lambda e: e["type"] == "job" and e["job"]["status"] == "done", timeout=5
    )
    snap = runner.snapshot()
    assert len(snap["activity"]) == ACTIVITY_LIMIT
    assert snap["counters"]["tool_calls"] == ACTIVITY_LIMIT + 50
    assert snap["stage_order"] == ["markdown"]
