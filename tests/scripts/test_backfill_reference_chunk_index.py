from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backfill_reference_chunk_index.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("backfill_reference_chunk_index", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()

#: Two videos with real chunk boundaries: 70-second chunks from zero.
STARTS = {
    "abc": {index: float(index * 70) for index in range(30)},
    "xyz": {index: float(index * 70) for index in range(5)},
}


def _reference(label: str, video_id: str, start: float, chunk_index: int | None) -> dict:
    return {
        "label": label,
        "video_id": video_id,
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "timestamp_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
        "start_seconds": start,
        "end_seconds": start + 70.0,
        "chunk_index": chunk_index,
    }


def _history(references: list[dict], key: str = "rag_llm", retrieved: list[str] | None = None):
    return {
        "conversations": [
            {
                "id": "q-1",
                "question": "q",
                "url": None,
                "asked_at": "2026-07-01T00:00:00+00:00",
                "answers": [
                    {
                        "key": key,
                        "title": "t",
                        "command": "c",
                        "answer": "a",
                        "references": references,
                        "retrieved_chunk_ids": retrieved or [],
                    }
                ],
            }
        ]
    }


def test_a_label_shaped_chunk_index_is_replaced_by_the_real_one() -> None:
    """The exact historical failure: the model wrote the label it could see."""
    reference = _reference("[3]", "abc", 1400.0, chunk_index=3)

    assert MODULE.resolve(reference, STARTS) == 20


def test_a_second_of_clock_rounding_still_resolves() -> None:
    """The model was shown mm:ss, so a faithful echo is the truncated second."""
    starts = {"abc": {4: 593.36}}
    reference = _reference("[1]", "abc", 593.0, chunk_index=1)

    assert MODULE.resolve(reference, starts) == 4


def test_a_timestamp_that_matches_nothing_is_left_alone_not_snapped() -> None:
    reference = _reference("[1]", "abc", 12345.0, chunk_index=1)

    assert MODULE.resolve(reference, STARTS) is None


def test_a_reference_to_an_unknown_video_is_left_alone() -> None:
    reference = _reference("[1]", "nope", 70.0, chunk_index=1)

    assert MODULE.resolve(reference, STARTS) is None


def test_apply_rewrites_only_the_wrong_ones() -> None:
    history = _history(
        [
            _reference("[1]", "abc", 0.0, chunk_index=0),  # already right
            _reference("[2]", "abc", 1400.0, chunk_index=2),  # label, not index
            _reference("[3]", "abc", 99999.0, chunk_index=3),  # unresolvable
            _reference("[4]", "abc", 210.0, chunk_index=None),  # never had one
        ]
    )

    changed = MODULE.apply_rewrites(history["conversations"], STARTS)
    indices = [
        reference["chunk_index"]
        for reference in history["conversations"][0]["answers"][0]["references"]
    ]

    assert changed == 1
    assert indices == [0, 20, 3, None]


def test_the_audit_counts_every_reference_exactly_once() -> None:
    history = _history(
        [
            _reference("[1]", "abc", 0.0, chunk_index=0),
            _reference("[2]", "abc", 1400.0, chunk_index=2),
            _reference("[3]", "abc", 99999.0, chunk_index=3),
            _reference("[4]", "abc", 210.0, chunk_index=None),
        ]
    )

    result = MODULE.audit(history["conversations"], STARTS)

    assert result["already_right"] == 1
    assert len(result["rewrites"]) == 1
    assert len(result["unresolvable"]) == 1
    assert result["no_index"] == 1


def test_the_positional_cross_check_flags_a_disagreement() -> None:
    """Two independent routes must agree, or the repair is a guess."""
    history = _history(
        [_reference("[1]", "abc", 1400.0, chunk_index=1)],
        retrieved=["chunk:abc:7"],  # retrieval order says 7, the timestamp says 20
    )

    result = MODULE.audit(history["conversations"], STARTS)

    assert result["cross_checked"] == 1
    assert result["cross_check_failures"] == [
        ("q-1", "rag_llm", "[1]", "chunk:abc:7", "chunk:abc:20")
    ]


def test_the_positional_cross_check_passes_when_the_routes_agree() -> None:
    history = _history(
        [_reference("[1]", "abc", 1400.0, chunk_index=1)],
        retrieved=["chunk:abc:20"],
    )

    result = MODULE.audit(history["conversations"], STARTS)

    assert result["cross_checked"] == 1
    assert result["cross_check_failures"] == []


def test_graph_rag_answers_are_exempt_from_the_positional_cross_check() -> None:
    """Its stored chunk ids are graph evidence, never in citation order."""
    history = _history(
        [_reference("[1]", "abc", 1400.0, chunk_index=20)],
        key="graph_rag",
        retrieved=["chunk:abc:7"],
    )

    result = MODULE.audit(history["conversations"], STARTS)

    assert result["cross_checked"] == 0
    assert result["cross_check_failures"] == []


def test_the_write_guard_accepts_a_chunk_index_only_change() -> None:
    before = _history([_reference("[1]", "abc", 1400.0, chunk_index=1)])
    after = json.loads(json.dumps(before))
    after["conversations"][0]["answers"][0]["references"][0]["chunk_index"] = 20

    MODULE.assert_only_chunk_index_changed(before, after)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda h: h["conversations"][0]["answers"][0]["references"][0].__setitem__(
            "start_seconds", 1.0
        ),
        lambda h: h["conversations"][0]["answers"][0]["references"].pop(),
        lambda h: h["conversations"][0]["answers"][0].__setitem__("answer", "rewritten"),
        lambda h: h["conversations"][0].__setitem__("question", "different"),
        lambda h: h["conversations"][0]["answers"][0].__setitem__("chunk_index", 3),
    ],
)
def test_the_write_guard_rejects_anything_else(mutate) -> None:
    """A future edit that started touching other fields fails here, not in review."""
    before = _history([_reference("[1]", "abc", 1400.0, chunk_index=1)])
    after = json.loads(json.dumps(before))
    mutate(after)

    with pytest.raises(ValueError):
        MODULE.assert_only_chunk_index_changed(before, after)


def test_rewriting_is_idempotent() -> None:
    history = _history([_reference("[2]", "abc", 1400.0, chunk_index=2)])

    assert MODULE.apply_rewrites(history["conversations"], STARTS) == 1
    assert MODULE.apply_rewrites(history["conversations"], STARTS) == 0
    assert MODULE.audit(history["conversations"], STARTS)["rewrites"] == []
