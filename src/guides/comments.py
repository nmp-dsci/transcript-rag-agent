"""Reader commentary on a guide, and the receipts a revision answers it with.

``comments.jsonl`` is append-only: one JSON object per line, status changes
appended as new lines with the same id and the latest wins. That keeps the
file a faithful record of what was asked, and lets the CLI and the app write
it without a lock beyond the append itself.

A receipt is the reviser's per-comment outcome. It is validated before a
revision publishes: every open comment id must appear exactly once, each with
one of three outcomes — the same rule Lavish applies to its tracked batches,
because a revision that silently drops a comment is worse than one that
rejects it out loud.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.guides.catalog import GuidePaths, read_comments

OUTCOMES = ("addressed", "deferred", "rejected")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def add_comment(
    paths: GuidePaths,
    *,
    body: str,
    anchor: str | None = None,
    section_id: str | None = None,
    quote: str = "",
    author: str = "reader",
) -> dict[str, Any]:
    body = " ".join(body.split())
    if not body:
        raise ValueError("a comment needs a body")
    record = {
        "id": f"c-{uuid.uuid4().hex[:6]}",
        "created_at": _now(),
        "author": author,
        "anchor": anchor or None,
        "section_id": section_id or None,
        "quote": " ".join(quote.split())[:600],
        "body": body[:4000],
        "status": "open",
        "resolved_in_version": None,
        "reason": None,
    }
    _append(paths.comments, record)
    return record


def latest_comments(paths: GuidePaths) -> list[dict[str, Any]]:
    """One record per id — the last line for that id wins — in first-seen order."""
    return read_comments(paths)


def open_comments(paths: GuidePaths) -> list[dict[str, Any]]:
    return [c for c in latest_comments(paths) if c.get("status", "open") == "open"]


def resolve_comments(paths: GuidePaths, receipt: "Receipt") -> list[dict[str, Any]]:
    """Append the receipt's outcomes as status lines. Returns the updated records."""
    current = {c["id"]: c for c in latest_comments(paths)}
    updated = []
    for item in receipt.items:
        base = current.get(item.id)
        if base is None:
            continue
        record = {
            **base,
            "status": item.outcome,
            "reason": item.reason,
            "resolved_in_version": receipt.version,
            "resolved_at": receipt.created_at,
        }
        _append(paths.comments, record)
        updated.append(record)
    return updated


@dataclass
class ReceiptItem:
    id: str
    outcome: str
    reason: str = ""
    sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "outcome": self.outcome,
            "reason": self.reason,
            "sections": list(self.sections),
        }


@dataclass
class Receipt:
    version: int
    summary: str = ""
    items: list[ReceiptItem] = field(default_factory=list)
    changed_sections: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
            "changed_sections": list(self.changed_sections),
        }


class ReceiptError(ValueError):
    """The reviser's receipt does not account for every comment exactly once."""


def parse_receipt(data: Any, *, expected_ids: list[str], version: int) -> Receipt:
    """Validate the reviser's ``receipt.json`` against the comments it was given."""
    if not isinstance(data, dict):
        raise ReceiptError("receipt.json must be a JSON object")
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ReceiptError("receipt.json needs an `items` list")
    items: list[ReceiptItem] = []
    seen: list[str] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ReceiptError("every receipt item must be an object")
        comment_id = str(raw.get("id") or "")
        outcome = str(raw.get("outcome") or "").strip().lower()
        if outcome not in OUTCOMES:
            raise ReceiptError(f"{comment_id or '?'}: outcome must be one of {', '.join(OUTCOMES)}")
        if comment_id in seen:
            raise ReceiptError(f"{comment_id} appears more than once")
        seen.append(comment_id)
        sections = raw.get("sections") or []
        items.append(
            ReceiptItem(
                id=comment_id,
                outcome=outcome,
                reason=" ".join(str(raw.get("reason") or "").split()),
                sections=[str(s) for s in sections if isinstance(s, (str, int))],
            )
        )
    missing = [cid for cid in expected_ids if cid not in seen]
    extra = [cid for cid in seen if cid not in expected_ids]
    if missing:
        raise ReceiptError(f"receipt is missing comment ids: {', '.join(missing)}")
    if extra:
        raise ReceiptError(f"receipt names unknown comment ids: {', '.join(extra)}")
    changed = data.get("changed_sections") or []
    return Receipt(
        version=version,
        summary=" ".join(str(data.get("summary") or "").split()),
        items=items,
        changed_sections=[str(s) for s in changed if isinstance(s, (str, int))],
    )


def write_receipt(paths: GuidePaths, receipt: Receipt) -> Path:
    paths.receipts.mkdir(parents=True, exist_ok=True)
    target = paths.receipt(receipt.version)
    target.write_text(
        json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target
