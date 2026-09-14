"""Read the committed guides under ``guides/`` for the Field Guides tab.

Same contract as :mod:`src.api.packs` and :mod:`src.api.matrix_runs`: what the
app renders is exactly what a reviewer can open in the repo and diff. The
catalog (``guides/index.json``) is derived from the per-guide manifests rather
than hand-kept, so a guide directory that is present is a guide that is listed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: ``guides/`` at the repo root, independent of the server's working directory.
DEFAULT_GUIDES_DIR = Path(__file__).resolve().parents[2] / "guides"

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,80}$")

MANIFEST = "manifest.json"
GUIDE_HTML = "guide.html"
GUIDE_MD = "guide.md"
CLAIMS = "claims.json"
COMMENTS = "comments.jsonl"
VERSIONS_DIR = "versions"
RECEIPTS_DIR = "receipts"
EVIDENCE_DIR = "evidence"
CORPUS_DIR = "corpus"
RUNS_DIR = "runs"


def is_slug(value: str) -> bool:
    return bool(SLUG_RE.match(value))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "guide"


@dataclass
class GuidePaths:
    root: Path
    slug: str

    @property
    def dir(self) -> Path:
        return self.root / self.slug

    @property
    def manifest(self) -> Path:
        return self.dir / MANIFEST

    @property
    def html(self) -> Path:
        return self.dir / GUIDE_HTML

    @property
    def markdown(self) -> Path:
        return self.dir / GUIDE_MD

    @property
    def claims(self) -> Path:
        return self.dir / CLAIMS

    @property
    def comments(self) -> Path:
        return self.dir / COMMENTS

    @property
    def versions(self) -> Path:
        return self.dir / VERSIONS_DIR

    @property
    def receipts(self) -> Path:
        return self.dir / RECEIPTS_DIR

    @property
    def evidence(self) -> Path:
        return self.dir / EVIDENCE_DIR

    @property
    def corpus(self) -> Path:
        return self.dir / CORPUS_DIR

    @property
    def runs(self) -> Path:
        return self.dir / RUNS_DIR

    def version_html(self, version: int) -> Path:
        return self.versions / f"v{version}.html"

    def receipt(self, version: int) -> Path:
        return self.receipts / f"v{version}.json"


def guide_paths(slug: str, guides_dir: Path | None = None) -> GuidePaths:
    if not is_slug(slug):
        raise ValueError(f"not a guide slug: {slug!r}")
    return GuidePaths(guides_dir or DEFAULT_GUIDES_DIR, slug)


@dataclass
class Manifest:
    """What one guide is and where it came from. Serialised as ``manifest.json``."""

    slug: str
    title: str
    topic: str
    subtitle: str = ""
    status: str = "published"  # published | draft | error
    current_version: int = 1
    compiled_at: str = ""
    model: dict[str, str] = field(default_factory=dict)
    video_ids: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    chunk_count: int = 0
    cluster_count: int = 0
    cite_total: int = 0
    cite_valid: int = 0
    gaps: list[str] = field(default_factory=list)
    web_allowed: bool = False
    web_urls: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "topic": self.topic,
            "subtitle": self.subtitle,
            "status": self.status,
            "current_version": self.current_version,
            "compiled_at": self.compiled_at,
            "model": dict(self.model),
            "video_ids": list(self.video_ids),
            "sources": list(self.sources),
            "chunk_count": self.chunk_count,
            "cluster_count": self.cluster_count,
            "cite_total": self.cite_total,
            "cite_valid": self.cite_valid,
            "gaps": list(self.gaps),
            "web_allowed": self.web_allowed,
            "web_urls": list(self.web_urls),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


def read_manifest(paths: GuidePaths) -> Manifest | None:
    """The manifest, or ``None`` when absent *or* unreadable.

    A half-written guide must not take the tab down with a 500; it reads as
    "not built", which is what a manifest the app cannot parse effectively is.
    """
    try:
        data = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("slug"):
        return None
    try:
        return Manifest.from_dict(data)
    except TypeError:
        return None


def write_manifest(paths: GuidePaths, manifest: Manifest) -> None:
    paths.dir.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def list_versions(paths: GuidePaths) -> list[int]:
    if not paths.versions.is_dir():
        return []
    found = []
    for candidate in paths.versions.glob("v*.html"):
        match = re.fullmatch(r"v(\d+)\.html", candidate.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def read_comments(paths: GuidePaths) -> list[dict[str, Any]]:
    """Every comment line, oldest first. Malformed lines are skipped, not fatal."""
    if not paths.comments.is_file():
        return []
    comments: list[dict[str, Any]] = []
    for line in paths.comments.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("id"):
            comments.append(record)
    return comments


def read_claims(paths: GuidePaths) -> dict[str, Any] | None:
    try:
        data = json.loads(paths.claims.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _summary(manifest: Manifest, paths: GuidePaths) -> dict[str, Any]:
    comments = read_comments(paths)
    open_comments = sum(1 for comment in comments if comment.get("status", "open") == "open")
    return {
        **manifest.to_dict(),
        "videos": len(manifest.video_ids),
        "versions": list_versions(paths) or [manifest.current_version],
        "comments_open": open_comments,
        "comments_total": len(comments),
        "html_url": f"/guides/{manifest.slug}/{GUIDE_HTML}",
        "markdown_url": f"/guides/{manifest.slug}/{GUIDE_MD}" if paths.markdown.is_file() else None,
    }


def list_guides(guides_dir: Path | None = None) -> dict[str, Any]:
    """Every guide directory with a readable manifest, newest compile first."""
    root = guides_dir or DEFAULT_GUIDES_DIR
    guides: list[dict[str, Any]] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not is_slug(child.name):
                continue
            paths = GuidePaths(root, child.name)
            manifest = read_manifest(paths)
            if manifest is None:
                continue
            guides.append(_summary(manifest, paths))
    guides.sort(key=lambda item: item.get("compiled_at") or "", reverse=True)
    return {"guides": guides, "write_command": "python -m src.cli guides write --topic ..."}


def guide_detail(slug: str, guides_dir: Path | None = None) -> dict[str, Any] | None:
    """One guide: manifest, versions, comments and claims. ``None`` when unknown."""
    if not is_slug(slug):
        return None
    paths = GuidePaths(guides_dir or DEFAULT_GUIDES_DIR, slug)
    manifest = read_manifest(paths)
    if manifest is None:
        return None
    detail = _summary(manifest, paths)
    detail["comments"] = read_comments(paths)
    detail["claims"] = read_claims(paths)
    detail["version_urls"] = {
        str(version): f"/guides/{slug}/{VERSIONS_DIR}/v{version}.html"
        for version in list_versions(paths)
    }
    receipts: dict[str, Any] = {}
    if paths.receipts.is_dir():
        for candidate in sorted(paths.receipts.glob("v*.json")):
            try:
                receipts[candidate.stem] = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    detail["receipts"] = receipts
    return detail


def write_index(guides_dir: Path | None = None) -> Path:
    """Regenerate ``guides/index.json`` from the manifests. Returns its path."""
    root = guides_dir or DEFAULT_GUIDES_DIR
    root.mkdir(parents=True, exist_ok=True)
    listing = list_guides(root)
    index = {
        "guides": [
            {
                key: item[key]
                for key in (
                    "slug",
                    "title",
                    "topic",
                    "subtitle",
                    "status",
                    "current_version",
                    "compiled_at",
                    "videos",
                    "chunk_count",
                    "cite_total",
                    "cite_valid",
                    "versions",
                )
            }
            for item in listing["guides"]
        ]
    }
    target = root / "index.json"
    target.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
