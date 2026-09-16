"""Deterministic clean-up of a composed page before it is published.

The composer is asked for the page skeleton the shared stylesheet expects,
but a model that gets the content right and the chrome slightly wrong should
not fail the run for it. Three fixes, all mechanical: the hero and main get
the ``wrap`` width class when they lack it, a ``nav.site`` is built from the
section headings when the page has none, and the reader bridge script is
appended when it is missing. Nothing here touches prose or cites.
"""

from __future__ import annotations

import html as html_lib
import re

BRIDGE_TAG = '<script src="/guides/guide-bridge.js"></script>'

_SECTION = re.compile(r'<section\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</section>', re.S)
_H2 = re.compile(r"<h2\b[^>]*>(.*?)</h2>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _add_class(tag: str, cls: str) -> str:
    match = re.search(r'\bclass="([^"]*)"', tag)
    if match:
        classes = match.group(1).split()
        if cls in classes:
            return tag
        return tag[: match.start(1)] + " ".join([*classes, cls]) + tag[match.end(1) :]
    return tag[:-1].rstrip() + f' class="{cls}">'


def section_links(page: str) -> list[tuple[str, str]]:
    """``(id, label)`` per top-level section with an ``h2``, in page order."""
    links = []
    for section_id, body in _SECTION.findall(page):
        heading = _H2.search(body)
        if not heading:
            continue
        label = " ".join(html_lib.unescape(_TAG.sub("", heading.group(1))).split())
        # Short nav labels: the part before a colon or dash, cut on a word.
        label = re.split(r"\s[—:–-]\s|:\s", label)[0]
        if len(label) > 24:
            label = label[:24].rsplit(" ", 1)[0]
        links.append((section_id, label))
    return links


def build_nav(title: str, links: list[tuple[str, str]]) -> str:
    items = "\n".join(
        f'      <a href="#{sid}">{html_lib.escape(label)}</a>' for sid, label in links
    )
    return (
        '<nav class="site" aria-label="Sections">\n  <div class="bar">\n'
        f'    <a class="brand" href="#top"><span class="mark" aria-hidden="true"></span>{html_lib.escape(title)}</a>\n'
        f'    <div class="links">\n{items}\n    </div>\n  </div>\n</nav>\n'
    )


def normalize_page(page: str, *, title: str) -> str:
    """Apply the three mechanical fixes. Idempotent."""
    out = page
    out = re.sub(
        r"<header\b[^>]*\bclass=\"[^\"]*\bhero\b[^\"]*\"[^>]*>",
        lambda m: _add_class(m.group(0), "wrap"),
        out,
        count=1,
    )
    out = re.sub(r"<main\b[^>]*>", lambda m: _add_class(m.group(0), "wrap"), out, count=1)
    if not re.search(r'<nav\b[^>]*\bclass="[^"]*\bsite\b', out):
        links = section_links(out)
        if links:
            nav = build_nav(title, links)
            hero = re.search(r"<header\b[^>]*\bclass=\"[^\"]*\bhero\b", out)
            if hero:
                out = out[: hero.start()] + nav + out[hero.start() :]
            else:
                out = re.sub(r"<body\b[^>]*>", lambda m: m.group(0) + "\n" + nav, out, count=1)
    if BRIDGE_TAG not in out:
        if "</body>" in out:
            out = out.replace("</body>", f"{BRIDGE_TAG}\n</body>", 1)
        else:
            out = out.rstrip() + f"\n{BRIDGE_TAG}\n"
    # The brand link needs somewhere to land.
    if 'id="top"' not in out:
        out = re.sub(
            r"<header\b[^>]*\bclass=\"[^\"]*\bhero\b[^\"]*\"[^>]*>",
            lambda m: m.group(0)[:-1] + ' id="top">' if " id=" not in m.group(0) else m.group(0),
            out,
            count=1,
        )
    return out
