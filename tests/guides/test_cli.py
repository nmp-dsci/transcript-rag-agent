from __future__ import annotations

import json
from pathlib import Path

from src.guides.catalog import GuidePaths
from src.guides.cli import run_guides
from src.guides.comments import latest_comments
from src.guides.importer import import_lavish_html
from tests.guides.test_importer import SAMPLE
import argparse


def ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_list_comment_and_import_from_json(tmp_path: Path, settings, capsys):
    source = tmp_path / "page.html"
    source.write_text(SAMPLE, encoding="utf-8")
    guides_dir = tmp_path / "guides"
    import_lavish_html(source, slug="tiny-guide", guides_dir=guides_dir, known_videos={"abc123XYZ"})

    assert run_guides(ns(guides_dir=guides_dir, guides_command="list", json=False), settings) == 0
    assert "tiny-guide" in capsys.readouterr().out

    assert (
        run_guides(
            ns(
                guides_dir=guides_dir,
                guides_command="comment",
                slug="tiny-guide",
                body="Tighten the lede",
                anchor="thesis-p1",
                section="thesis",
                quote="Lede",
                from_json=None,
            ),
            settings,
        )
        == 0
    )
    batch = tmp_path / "annotations.json"
    batch.write_text(
        json.dumps(
            [
                {"body": "Cite the numbers", "anchor": "thesis-p2", "quote": "quoted video"},
                {"body": "   "},
                {"body": "Whole-guide note"},
            ]
        )
    )
    assert (
        run_guides(
            ns(
                guides_dir=guides_dir,
                guides_command="comment",
                slug="tiny-guide",
                body=None,
                anchor=None,
                section=None,
                quote="",
                from_json=batch,
            ),
            settings,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "2 comments added" in out
    comments = latest_comments(GuidePaths(guides_dir, "tiny-guide"))
    assert [c["body"] for c in comments] == [
        "Tighten the lede",
        "Cite the numbers",
        "Whole-guide note",
    ]
    assert comments[1]["author"] == "lavish" and comments[0]["author"] == "cli"

    assert (
        run_guides(
            ns(guides_dir=guides_dir, guides_command="comments", slug="tiny-guide", json=False),
            settings,
        )
        == 0
    )
    assert "Tighten the lede" in capsys.readouterr().out
    assert (
        run_guides(
            ns(
                guides_dir=guides_dir,
                guides_command="comment",
                slug="nope",
                body="x",
                anchor=None,
                section=None,
                quote="",
                from_json=None,
            ),
            settings,
        )
        == 2
    )
