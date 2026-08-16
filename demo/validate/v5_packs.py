"""Judge whether the corpus was really distilled into expert packs a reader can see.

    uv run python -m src.cli serve --port 8021                 # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v5_packs

V5's deliverable is a claim about *knowledge*: that many creators' transcripts
were reduced to four packs of checkable review criteria, each rule carrying the
words a named creator actually said. Every part of that claim is cheap to fake
and expensive to check. A directory of JSON under ``experts/``, a
``quote_resolution: 1.0`` field, a ``multi_creator_share`` of 0.76, a
``stats.creators`` of 11 — none of those is the claim. The claim is that a
reader who opens the Experiments tab finds four packs, reads a rule, clicks its
quote, and lands on a real person saying those words at that second in a
transcript this corpus holds.

This repo has already shipped a pack whose citation pointed at a quote nobody
said. So the central assertion here is deliberately the expensive one: **every
rendered quote is matched against the transcript text the app itself renders in
the RAG Pipeline corpus tree** — DOM against DOM, with the pack's own
``quote_resolution`` field treated as an unverified advertisement rather than
evidence. 105 citations, 31 videos, no sampling: a resolution rate is only
worth something if the denominator is all of them.

Six things are checked, in the order a sceptic would reach for them.

* **Render (D3, "the app renders them").** All four declared topics appear as
  selectable packs; each renders a rubric list whose count matches
  ``experts/<topic>/pack.json``; each rule opens to its evidence. A pack that is
  built and invisible fails D3 no matter what is on disk. The counts are read
  from the rendered list, not from the panel's own heading, because a heading
  that says "17 criteria" over 12 rows is the exact failure worth catching.

* **Resolution.** For every rendered citation: the video is one a reader can
  open in the corpus tree; the quote text appears verbatim inside the rendered
  transcript of the chunk the citation names; the deep link's ``t=`` offset
  falls inside that chunk's rendered time window; the creator printed beside the
  quote is the channel the corpus gives that video. Four separate ways a
  citation can be decorative rather than true, and a quote that resolves to
  *some* chunk of the right video is still recorded as a failure if it is not
  the chunk cited — a citation that is off by one chunk is a citation nobody
  checked.

* **Multi-creator.** The "N creators" badge is compared against the creator
  names actually rendered under the rule, and a rule whose two quotes come from
  two videos of one channel must not be badged as two creators. That is the
  precise overstatement the corpus audit found once already: a theme can span
  four videos and still be one podcast talking.

* **D2 as a comparison, not a decision.** The three arms must be on screen
  against each other with their numbers — raptor, communities, merged — not
  summarised as "merged won". The held-out harness scores one artifact, so only
  the resume pack can carry the scored table; the other three must still show
  the three arms compared on the deterministic counts, and must say why they
  have no score rather than silently omitting the row.

* **D5.** None of the five Australian property/tax videos may appear as a member
  or a citation of any rendered pack, and the surface must say the exclusion
  exists rather than leaving its absence to be taken on trust.

* **D3's other half — reviewable *and* overridable.** The membership table is
  read for the score and provenance of each routed video, and an override is
  actually clicked: the check is not that a button exists but that pressing it
  is recorded, acknowledged on screen, and honest about when it takes effect.

Two things this script asserts that the brief did not ask for.

It checks that the packs are **honest about their own age**. Every pack records
``corpus e1b17b35cb7190cf (1372 chunks, 56 videos)`` and the live corpus is
larger — material was ingested after the packs were built. Stating the corpus
you were built from is good practice and the panel does it; what a reader also
needs is to be told that the number in the pack header and the number in the app
header are not the same number. This is asserted as its own check so that a
future rebuild flips it rather than leaving it as prose in a note.

It checks that **V8's loop-built diff row has not broken V5's rendering**. That
section belongs to another slice and is not judged here, but it was added to
``PackPanel.tsx`` minutes before this evaluation and shares the panel V5 ships.

ENVIRONMENT NOTE — this machine's Metal compiler XPC service is wedged and
``chromium.launch()`` times out after 180s, so this file could not be executed
through Playwright at evaluation time. Every assertion below was executed
instead as the equivalent DOM query against the running app in an already-open
Chrome at the same URL, in a tab created for this evaluation, and the results
are recorded in ``artifacts/v5_packs/verdict.json`` with the limitation noted
there. Slices v2, v3, v4 and v4b recorded the same constraint the same way.
Nothing here is machine-specific; re-run it once the host can launch a browser.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v5_packs"

REPO = Path(__file__).resolve().parents[2]
EXPERTS = REPO / "experts"

#: D1 puts all four topics in scope, so all four are asserted rather than
#: whichever ones happen to have been built.
TOPICS = ["resume-design", "job-search", "system-design", "app-architecture"]

#: D5's exclusions: off-mission Australian property/tax material that stays
#: indexed for Q&A and must not reach a pack.
PROPERTY_VIDEOS = {"7m27Go3K1d0", "AdRL6tKu3Gk", "Bw58mV015z4", "ZiEGOgTC56Y", "gXf7fRvuaXA"}

#: The D2 numbers committed in ``experts/ablation.json``. Asserted as rendered
#: text so that "the ablation ran" and "a reader can see what it found" stay two
#: different claims.
D2_EXPECTED = {
    "raptor": ("0.263", "0.667"),
    "communities": ("0.105", "0.778"),
    "merged": ("0.263", "0.824"),
}

WATCH_ID = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")
WATCH_T = re.compile(r"[?&]t=(\d+)s")
#: ``chunk:<video>:<index> — opens at M:SS`` — the citation link's title, and the
#: only place the rendered pack names the chunk it is quoting.
CHUNK_TITLE = re.compile(r"(chunk:([A-Za-z0-9_-]{11}):(\d+))\s*—\s*opens at\s*(\d+:\d{2})")
THUMB_ID = re.compile(r"/vi(?:_webp)?/([A-Za-z0-9_-]{11})/")
CLOCK = re.compile(r"(\d+):(\d{2})")


def clock(text: str) -> int | None:
    match = CLOCK.search(text or "")
    return int(match.group(1)) * 60 + int(match.group(2)) if match else None


def normalise(text: str) -> str:
    """Compare quotes the way a reader compares them, not byte for byte.

    Curly quotes, dashes and stray punctuation differ between the pack JSON, the
    blockquote the panel renders and the transcript paragraph — none of which is
    the failure this is looking for. Words and digits are what must match.
    """
    lowered = (text or "").lower()
    for pair in ("‘'", "’'", "“\"", "”\"", "–-", "—-"):
        lowered = lowered.replace(pair[0], pair[1])
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", lowered)).strip()


def pack_file(topic: str) -> dict:
    """The artifact on disk — the thing D3 says the app must render."""
    return json.loads((EXPERTS / topic / "pack.json").read_text(encoding="utf-8"))


# ── the pack panel, as a reader drives it ────────────────────────────────────
def select_topic(session: UserSession, topic: str) -> None:
    """Click a pack's tab and wait for its detail, not just its heading.

    The panel clears ``detail`` before refetching, so the heading changes a beat
    before the body does. Asserting on the gap between the two is how a scrape
    ends up reporting the previous pack's rubrics under the next pack's name.
    """
    session.page.locator('.pk-card .exp-seg[aria-label="Expert pack"] button').filter(
        has_text=topic
    ).first.click()
    session.page.wait_for_function(
        """(topic) => {
            const card = document.querySelector('.pk-card');
            if (!card) return false;
            return card.querySelector('h3').innerText.includes(topic)
                && /corpus\\s/.test(card.querySelector('.exp-sub').innerText)
                && card.querySelector('.pk-rublist')
                && [...card.querySelectorAll('.pk-section h4')].some(h => /Membership/.test(h.innerText));
        }""",
        arg=topic,
        timeout=30000,
    )
    session.page.wait_for_timeout(400)


def read_pack(session: UserSession, topic: str) -> dict:
    """Open every rule in the pack and read back what a reader would see.

    Rules are expanded one at a time because the panel keeps a single open
    rubric; the evidence for a closed rule is not in the DOM at all, which is
    exactly why a scrape that only reads the collapsed list would report a pack
    full of criteria and no quotes and never notice.
    """
    select_topic(session, topic)
    heads = session.page.locator(".pk-rublist > .pk-rubric > .pk-rubhead")
    rubrics: list[dict] = []
    for index in range(heads.count()):
        head = heads.nth(index)
        head.click()
        session.page.wait_for_selector(".pk-rubric.open .pk-rubbody", timeout=5000)
        rubrics.append(
            session.page.evaluate(
                """() => {
                    const li = document.querySelector('.pk-rubric.open');
                    const head = li.querySelector('.pk-rubhead');
                    const body = li.querySelector('.pk-rubbody');
                    return {
                        rubric_id: head.querySelector('.pk-rubid').innerText.trim(),
                        criterion: head.querySelector('.pk-rubtext').innerText.trim(),
                        badges: [...head.querySelectorAll('.pk-badge')].map(
                            b => ({text: b.innerText.trim(), title: b.getAttribute('title')})),
                        unit: (body.querySelector('.pk-unit') || {innerText: ''}).innerText.trim(),
                        evidence: [...body.querySelectorAll('.pk-evlist > .pk-ev')].map(ev => ({
                            href: ev.querySelector('.pk-ts').getAttribute('href'),
                            title: ev.querySelector('.pk-ts').getAttribute('title'),
                            label: ev.querySelector('.pk-ts').innerText.trim(),
                            unresolved: !!ev.querySelector('.pk-ts.bad'),
                            creator: (ev.querySelector('.pk-creator') || {innerText: ''}).innerText.trim(),
                            quote: (ev.querySelector('.pk-quote') || {innerText: ''}).innerText
                                     .replace(/^[“"]/, '').replace(/[”"]$/, '').trim(),
                        })),
                    };
                }"""
            )
        )
        head.click()
    sections = session.page.evaluate(
        """() => {
            const card = document.querySelector('.pk-card');
            const table = el => el ? {
                headers: [...el.querySelectorAll('thead th')].map(t => t.innerText.trim()),
                rows: [...el.querySelectorAll('tbody tr')].map(
                    tr => [...tr.querySelectorAll('td')].map(td => td.innerText.replace(/\\s+/g, ' ').trim())),
            } : null;
            const sec = re => [...card.querySelectorAll('.pk-section')].find(
                s => re.test((s.querySelector('h4') || {innerText: ''}).innerText));
            const member = sec(/Membership/), arms = sec(/Arms as built/), d2 = sec(/^D2/);
            return {
                sub: card.querySelector('.exp-sub').innerText.replace(/\\s+/g, ' ').trim(),
                intro: card.querySelector('.pk-intro').innerText.replace(/\\s+/g, ' ').trim(),
                checks: [...card.querySelectorAll('.pk-checks .pk-check-item')].map(
                    x => x.innerText.replace(/\\s+/g, ' ').trim()),
                heading: card.querySelector('.pk-section h4').innerText.replace(/\\s+/g, ' ').trim(),
                membership: table(member && member.querySelector('table')),
                membership_head: member ? member.querySelector('h4').innerText.replace(/\\s+/g, ' ').trim() : null,
                pins: member ? [...member.querySelectorAll('tbody tr td.pk-pin')].map(
                    td => [...td.querySelectorAll('button')].map(b => b.innerText.trim())) : [],
                arms: table(arms && arms.querySelector('table')),
                d2: table(d2 && d2.querySelector('table')),
                d2_head: d2 ? d2.querySelector('h4').innerText.replace(/\\s+/g, ' ').trim() : null,
                d2_verdicts: d2 ? [...d2.querySelectorAll('.pk-verdict')].map(
                    v => v.innerText.replace(/\\s+/g, ' ').trim()) : [],
                no_d2: card.querySelector('.pk-nod2') ? card.querySelector('.pk-nod2').innerText.trim() : null,
                research: [...card.querySelectorAll('.pk-linkbtn')].map(b => b.innerText.replace(/\\s+/g, ' ').trim()),
                text: card.innerText.replace(/\\s+/g, ' '),
            };
        }"""
    )
    return {"topic": topic, "rubrics": rubrics, **sections}


def citations(packs: dict[str, dict]) -> list[dict]:
    """Every rendered quote, flattened, with its link taken apart."""
    rows: list[dict] = []
    for topic, pack in packs.items():
        for rubric in pack["rubrics"]:
            for item in rubric["evidence"]:
                video = WATCH_ID.search(item["href"] or "")
                offset = WATCH_T.search(item["href"] or "")
                chunk = CHUNK_TITLE.search(item["title"] or "")
                rows.append(
                    {
                        "topic": topic,
                        "rubric": rubric["rubric_id"],
                        "video": video.group(1) if video else None,
                        "t": int(offset.group(1)) if offset else None,
                        "chunk_video": chunk.group(2) if chunk else None,
                        "chunk_index": int(chunk.group(3)) if chunk else None,
                        "label_seconds": clock(item["label"]),
                        "creator": item["creator"],
                        "quote": item["quote"],
                        "unresolved": item["unresolved"],
                    }
                )
    return rows


# ── the corpus tree: the only ground truth this script will accept ───────────
def read_transcript(session: UserSession, video_id: str) -> dict | None:
    """Open one video in the corpus tree and read its rendered chunks.

    The row is found by the id in its thumbnail src rather than by title,
    because the corpus holds two videos with the same title from the same
    creator and a title match would silently read the wrong one.
    """
    summary = session.page.locator("nav.tree details > summary").filter(
        has=session.page.locator(f'img.thumb[src*="/{video_id}/"]')
    )
    if summary.count() == 0:
        return None
    summary.first.scroll_into_view_if_needed()
    summary.first.click()
    try:
        session.page.wait_for_function(
            """(id) => {
                const a = document.querySelector('.detail .chunkcard .cbody a');
                return a && (a.getAttribute('href') || '').includes('v=' + id);
            }""",
            arg=video_id,
            timeout=20000,
        )
    except Exception:  # noqa: BLE001 - a row that will not open is the finding
        return None
    session.page.wait_for_timeout(200)
    return session.page.evaluate(
        """() => ({
            title: document.querySelector('.detail .vhead .t').innerText.trim(),
            meta: document.querySelector('.detail .vhead .m').innerText.replace(/\\s+/g, ' ').trim(),
            cards: [...document.querySelectorAll('.detail .chunkcard')].map(card => {
                const head = card.querySelector('.cbody .h').innerText.replace(/\\s+/g, ' ');
                const index = /#c(\\d+)/.exec(head);
                const span = /(\\d+:\\d{2})\\s*[–-]\\s*(\\d+:\\d{2})/.exec(head);
                return {
                    index: index ? parseInt(index[1], 10) : null,
                    from: span ? span[1] : null,
                    to: span ? span[2] : null,
                    text: (card.querySelector('.cbody p') || {innerText: ''}).innerText,
                };
            }),
        })"""
    )


# ── checks ───────────────────────────────────────────────────────────────────
def render_checks(session: UserSession, packs: dict[str, dict]) -> None:
    """D3's first half: the packs exist on disk *and* on the screen."""
    tabs = session.page.evaluate(
        """() => [...document.querySelectorAll('.pk-card .exp-seg[aria-label="Expert pack"] button')]
                   .map(b => ({label: b.innerText.trim().split('\\n')[0],
                               unbuilt: !!b.querySelector('.pk-unbuilt')}))"""
    )
    labels = [tab["label"] for tab in tabs]
    session.check(
        "all four declared packs are offered in the app",
        all(topic in labels for topic in TOPICS),
        f"pack tabs rendered: {labels}; unbuilt: "
        f"{[t['label'] for t in tabs if t['unbuilt']] or 'none'}",
        shot=True,
    )
    for topic, pack in packs.items():
        stored = pack_file(topic)
        rendered_ids = [r["rubric_id"] for r in pack["rubrics"]]
        stored_ids = [r["rubric_id"] for r in stored["rubrics"]]
        session.check(
            f"{topic}: the rules on screen are the rules in the pack file",
            rendered_ids == stored_ids,
            f"{len(rendered_ids)} rendered, {len(stored_ids)} in experts/{topic}/pack.json; "
            f"heading reads {pack['heading'].split('click')[0].strip()!r}"
            + (
                ""
                if rendered_ids == stored_ids
                else f"; only on screen: {sorted(set(rendered_ids) - set(stored_ids))}; "
                f"only in the file: {sorted(set(stored_ids) - set(rendered_ids))}"
            ),
            shot=True,
        )
        rendered_quotes = sum(len(r["evidence"]) for r in pack["rubrics"])
        stored_quotes = sum(len(r["evidence"]) for r in stored["rubrics"])
        empty = [r["rubric_id"] for r in pack["rubrics"] if not r["evidence"]]
        session.check(
            f"{topic}: every rule opens to the quotes behind it",
            rendered_quotes == stored_quotes and not empty,
            f"{rendered_quotes} quotes rendered across {len(rendered_ids)} rules "
            f"({stored_quotes} in the pack file); rules with no evidence: {empty or 'none'}",
        )


def resolution_checks(session: UserSession, cites: list[dict]) -> dict[str, dict]:
    """The expensive one: does the evidence lead anywhere real?"""
    session.tab("RAG Pipeline")
    session.page.wait_for_selector("nav.tree img.thumb", timeout=20000)
    tree_ids = {
        m.group(1)
        for src in session.page.evaluate(
            "() => [...document.querySelectorAll('nav.tree img.thumb')].map(i => i.getAttribute('src'))"
        )
        if (m := THUMB_ID.search(src or ""))
    }
    wanted = sorted({c["video"] for c in cites if c["video"]})
    missing = [v for v in wanted if v not in tree_ids]
    session.check(
        "every cited video is one a reader can open in the corpus",
        not missing and not any(c["video"] is None for c in cites),
        f"{len(cites)} citations over {len(wanted)} videos; "
        f"citations with no video id: {sum(1 for c in cites if not c['video'])}; "
        f"cited but absent from the corpus tree: {missing or 'none'}",
        shot=True,
    )

    transcripts = {video: read_transcript(session, video) for video in wanted}
    unreadable = [v for v, t in transcripts.items() if t is None]

    wrong_chunk: list[str] = []
    nowhere: list[str] = []
    outside_window: list[str] = []
    wrong_creator: list[str] = []
    for cite in cites:
        transcript = transcripts.get(cite["video"])
        if transcript is None:
            continue
        where = f"{cite['topic']}/{cite['rubric']} {cite['video']}#c{cite['chunk_index']}"
        card = next((c for c in transcript["cards"] if c["index"] == cite["chunk_index"]), None)
        quote = normalise(cite["quote"])
        if card is None or quote not in normalise(card["text"]):
            elsewhere = next(
                (c["index"] for c in transcript["cards"] if quote in normalise(c["text"])), None
            )
            if elsewhere is None:
                nowhere.append(f"{where}: {cite['quote'][:60]!r}")
            else:
                wrong_chunk.append(f"{where}: the words are in #c{elsewhere}")
            continue
        start, end = clock(card["from"] or ""), clock(card["to"] or "")
        if start is not None and not (start - 5 <= (cite["t"] or -1) <= (end or start) + 5):
            outside_window.append(f"{where}: link at {cite['t']}s, chunk {card['from']}-{card['to']}")
        channel = (transcript["meta"] or "").split("·")[0].strip()
        if normalise(channel) != normalise(cite["creator"]):
            wrong_creator.append(f"{where}: labelled {cite['creator']!r}, corpus says {channel!r}")

    session.check(
        "every rendered quote is in the transcript chunk its citation names",
        not nowhere and not wrong_chunk and not unreadable,
        f"{len(cites) - len(nowhere) - len(wrong_chunk)} of {len(cites)} citations matched "
        f"the rendered transcript; not found anywhere in the video: {nowhere or 'none'}; "
        f"found in a different chunk: {wrong_chunk or 'none'}; "
        f"videos that would not open: {unreadable or 'none'}",
        shot=True,
    )
    session.check(
        "every citation's link lands inside the chunk it quotes",
        not outside_window,
        f"{len(cites) - len(outside_window)} of {len(cites)} link offsets inside the rendered "
        f"chunk window; outside: {outside_window or 'none'}",
    )
    session.check(
        "the creator printed beside a quote is that video's channel",
        not wrong_creator,
        f"{len(cites) - len(wrong_creator)} of {len(cites)} agree with the corpus header; "
        f"mismatched: {wrong_creator or 'none'}",
    )
    session.check(
        "no citation is rendered as unresolved",
        not any(c["unresolved"] for c in cites),
        f"{sum(1 for c in cites if c['unresolved'])} citations carry the unresolved style",
    )
    return transcripts


def creator_checks(session: UserSession, packs: dict[str, dict]) -> None:
    """The claim that is easiest to inflate: how many voices back a rule."""
    seen: dict[str, list[str]] = {}
    for topic, pack in packs.items():
        badge_wrong: list[str] = []
        double_counted: list[str] = []
        creators: set[str] = set()
        for rubric in pack["rubrics"]:
            badge = next((b for b in rubric["badges"] if "creator" in b["text"]), None)
            claimed = int(badge["text"].split()[0]) if badge else -1
            names = list(dict.fromkeys(e["creator"] for e in rubric["evidence"]))
            videos = {WATCH_ID.search(e["href"]).group(1) for e in rubric["evidence"]}
            creators.update(names)
            if claimed != len(names) or sorted((badge or {}).get("title", "").split(" · ")) != sorted(names):
                badge_wrong.append(
                    f"{rubric['rubric_id']}: badge says {claimed}, "
                    f"the rows name {len(names)} ({names})"
                )
            if claimed >= 2 and (len(set(names)) < 2 or len(videos) < 2):
                double_counted.append(
                    f"{rubric['rubric_id']}: {claimed} creators from {names} over {sorted(videos)}"
                )
        session.check(
            f"{topic}: the creator badge counts the creators actually shown",
            not badge_wrong,
            "; ".join(badge_wrong) if badge_wrong else f"{len(pack['rubrics'])} rules agree",
        )
        session.check(
            f"{topic}: no rule reaches two creators by counting one twice",
            not double_counted,
            "; ".join(double_counted) if double_counted else "no rule badged >=2 rests on one voice",
        )
        header = next((c for c in pack["checks"] if "DISTINCT CREATORS" in c), "")
        stated = int(re.search(r"(\d+)", header.split("CREATORS")[-1]).group(1)) if header else -1
        session.check(
            f"{topic}: the pack header's creator count is the one on the page",
            stated == len(creators),
            f"header says {stated}, the quotes name {len(creators)}: {sorted(creators)}",
        )
        for name in creators:
            seen.setdefault(name, []).append(topic)

    duplicates: dict[str, list[str]] = {}
    for topic, pack in packs.items():
        for rubric in pack["rubrics"]:
            key = normalise(rubric["criterion"])
            duplicates.setdefault(key, []).append(f"{topic}/{rubric['rubric_id']}")
    repeated = {k: v for k, v in duplicates.items() if len(v) > 1}
    session.check(
        "the four packs are four different sets of rules",
        not repeated,
        f"{len(duplicates)} distinct criteria across "
        f"{sum(len(p['rubrics']) for p in packs.values())} rules"
        + (f"; repeated: {list(repeated.values())[:3]}" if repeated else ""),
    )


def d2_checks(session: UserSession, packs: dict[str, dict]) -> None:
    """D2 must be *shown as a comparison*, not reported as a conclusion."""
    scored = {t: p for t, p in packs.items() if p["d2"]}
    session.check(
        "the D2 ablation is rendered as a three-way table",
        bool(scored) and all(len(p["d2"]["rows"]) == 3 for p in scored.values()),
        f"scored packs: {sorted(scored)}; "
        + "; ".join(f"{t}: {len(p['d2']['rows'])} arms" for t, p in scored.items()),
        shot=True,
    )
    for topic, pack in scored.items():
        rows = {row[0].replace("BASE", "").strip(): row for row in pack["d2"]["rows"]}
        missing = [arm for arm in D2_EXPECTED if arm not in rows]
        wrong = []
        for arm, (recall, precision) in D2_EXPECTED.items():
            row = rows.get(arm)
            if not row:
                continue
            if recall not in row[1] or precision not in row[2]:
                wrong.append(f"{arm}: recall {row[1]!r}, precision {row[2]!r}")
        session.check(
            f"{topic}: the committed D2 numbers are the ones on screen",
            not missing and not wrong,
            f"arms {sorted(rows)}; "
            + (
                "; ".join(wrong)
                if wrong
                else "raptor/communities/merged render "
                + ", ".join(f"{a} {r}/{p}" for a, (r, p) in D2_EXPECTED.items())
            ),
        )
        session.check(
            f"{topic}: the reader is told which lead survives the scorer's noise",
            len(pack["d2_verdicts"]) >= 2
            and any("decisive" in v for v in pack["d2_verdicts"]),
            " | ".join(pack["d2_verdicts"]),
        )
    unscored = [t for t in TOPICS if t not in scored]
    session.check(
        "packs with no D2 run still compare the three arms and say why they have no score",
        all(packs[t]["arms"] and len(packs[t]["arms"]["rows"]) == 3 for t in unscored)
        and all(packs[t]["no_d2"] for t in unscored),
        "; ".join(
            f"{t}: {len(packs[t]['arms']['rows'])} arms, "
            f"{'explains the gap' if packs[t]['no_d2'] else 'says nothing'}"
            for t in unscored
        ),
        shot=True,
    )


def d5_checks(session: UserSession, packs: dict[str, dict], cites: list[dict]) -> None:
    """The property videos stay out of the packs and the app says so."""
    leaked = sorted({c["video"] for c in cites if c["video"] in PROPERTY_VIDEOS})
    session.check(
        "no D5-excluded property video is cited by any rendered rule",
        not leaked,
        f"{len(cites)} citations over {len({c['video'] for c in cites})} videos; "
        f"property ids cited: {leaked or 'none'}",
    )
    titles = {
        m.group(1): row
        for row in session.page.evaluate(
            """() => [...document.querySelectorAll('nav.tree details > summary')].map(s => {
                const img = s.querySelector('img.thumb');
                return {src: img ? img.getAttribute('src') : '',
                        label: (s.querySelector('.label') || {textContent: ''}).textContent.trim()};
            })"""
        )
        if (m := THUMB_ID.search(row["src"] or "")) and m.group(1) in PROPERTY_VIDEOS
        for row in [row]
    }
    members = {
        topic: [r[1] for r in pack["membership"]["rows"]] for topic, pack in packs.items()
    }
    hits = {
        topic: [
            vid
            for vid, row in titles.items()
            if any(normalise(row["label"])[:30] in normalise(m) for m in rows)
        ]
        for topic, rows in members.items()
    }
    session.check(
        "no D5-excluded property video is a member of any rendered pack",
        not any(hits.values()),
        f"membership rows checked: {sum(len(r) for r in members.values())}; "
        f"property videos found: {ii if (ii := {k: v for k, v in hits.items() if v}) else 'none'}",
    )
    stated = {
        topic: any("5 videos blocked" in c for c in pack["checks"]) for topic, pack in packs.items()
    }
    session.check(
        "the exclusion is stated on the pack surface, not left to be assumed",
        all(stated.values()),
        f"packs whose header names the blocked videos: {sorted(t for t, ok in stated.items() if ok)}",
    )


def override_checks(session: UserSession, packs: dict[str, dict]) -> None:
    """D3's second half: reviewable, and actually overridable.

    The override is *pressed*, not merely found. A button that renders and 500s
    is the same slice failure as no button at all, and the only way to tell them
    apart is to click one and read what the panel then says.
    """
    reviewable = {
        topic: (len(pack["membership"]["rows"]), pack["membership"]["headers"])
        for topic, pack in packs.items()
    }
    session.check(
        "membership is reviewable: every routed video shows its score and where it came from",
        all(
            n > 0 and {"score", "video", "creator", "chunks", "source"} <= {h.lower() for h in heads}
            for n, heads in reviewable.values()
        ),
        "; ".join(f"{t}: {n} rows {h}" for t, (n, h) in reviewable.items()),
        shot=True,
    )
    pins = {topic: pack["pins"] for topic, pack in packs.items()}
    session.check(
        "every membership row offers an override",
        all(rows and all(r == ["in", "out", "auto"] for r in rows) for rows in pins.values()),
        "; ".join(f"{t}: {len(rows)} rows offering {rows[0] if rows else 'nothing'}" for t, rows in pins.items()),
    )

    # Pressed on job-search because that pack's manifest is not yet tracked, so
    # the round trip leaves no diff on committed state once it is undone.
    select_topic(session, "job-search")
    table = session.page.locator(".pk-card table.pk-table tbody tr")
    row = table.nth(table.count() - 1)
    row.locator(".pk-pin button", has_text="out").click()
    session.page.wait_for_selector(".pk-card .pk-notice", timeout=10000)
    notice = session.page.locator(".pk-card .pk-notice").inner_text().strip()
    session.page.wait_for_timeout(1200)
    pinned = session.page.locator(".pk-card table.pk-table tbody tr").nth(table.count() - 1)
    session.check(
        "an override is recorded, acknowledged, and honest about when it applies",
        "recorded" in notice and "next build" in notice and "pinned out" in pinned.inner_text(),
        f"notice reads {notice!r}; the row now shows "
        f"{' '.join(pinned.locator('.pk-badge').all_inner_texts())!r}",
        shot=True,
    )
    row.locator(".pk-pin button", has_text="auto").click()
    session.page.wait_for_timeout(1500)
    restored = session.page.locator(".pk-card table.pk-table tbody tr").nth(table.count() - 1)
    session.check(
        "the override can be handed back to the router",
        "pinned" not in restored.inner_text(),
        f"after 'auto' the row shows {' '.join(restored.locator('.pk-badge').all_inner_texts())!r}",
    )


def honesty_checks(session: UserSession, packs: dict[str, dict]) -> None:
    """Does the panel admit that the pack is older than the corpus it describes?"""
    live = session.page.evaluate(
        """() => {
            const m = document.body.innerText.match(/(\\d+) videos\\s*[··]\\s*(\\d+) chunks/);
            return m ? {videos: parseInt(m[1], 10), chunks: parseInt(m[2], 10)} : null;
        }"""
    )
    sub = packs[TOPICS[0]]["sub"]
    stated = re.search(r"\((\d+) chunks, (\d+) videos\)", sub)
    session.check(
        "each pack states the corpus it was built from",
        bool(stated) and all("corpus" in p["sub"] for p in packs.values()),
        f"header reads {sub!r}",
    )
    if not (stated and live):
        return
    pack_chunks, pack_videos = int(stated.group(1)), int(stated.group(2))
    drifted = (pack_chunks, pack_videos) != (live["chunks"], live["videos"])
    admits = any(
        word in packs[TOPICS[0]]["text"].lower()
        for word in ("stale", "out of date", "outdated", "rebuild", "no longer current")
    )
    session.check(
        "a reader is told when the pack is behind the corpus they are browsing",
        not drifted or admits,
        f"the pack says {pack_chunks} chunks / {pack_videos} videos; the app's own header says "
        f"{live['chunks']} chunks / {live['videos']} videos; the panel "
        f"{'names the drift' if admits else 'says nothing about the difference'}",
        shot=True,
    )


def v8_regression_check(session: UserSession, packs: dict[str, dict]) -> None:
    """V8's diff row shares this panel. It is not judged here — only tolerated."""
    with_loop = {t: [r for r in p["research"] if "Loop-built" in r] for t, p in packs.items()}
    intact = all(len(p["rubrics"]) > 0 and p["membership"]["rows"] for p in packs.values())
    session.check(
        "V8's loop-built section has not displaced V5's own rendering",
        intact,
        "; ".join(
            f"{t}: {len(p['rubrics'])} rules, {len(p['membership']['rows'])} members, "
            f"loop row {'present' if with_loop[t] else 'absent'}"
            for t, p in packs.items()
        ),
    )


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        session.tab("Experiments")
        session.page.wait_for_selector(".pk-card .pk-rublist", timeout=30000)

        packs = {topic: read_pack(session, topic) for topic in TOPICS}
        cites = citations(packs)

        render_checks(session, packs)
        creator_checks(session, packs)
        d2_checks(session, packs)
        honesty_checks(session, packs)
        v8_regression_check(session, packs)
        resolution_checks(session, cites)
        d5_checks(session, packs, cites)

        session.tab("Experiments")
        session.page.wait_for_selector(".pk-card .pk-rublist", timeout=30000)
        override_checks(session, packs)

        session.note(
            "Quote resolution is asserted against the transcript the RAG Pipeline renders, "
            "not against the pack's own quote_resolution field. A pack that scores itself is "
            "the category of evidence this directory exists to distrust, and this repo has "
            "already shipped a citation pointing at a quote nobody said."
        )
        session.note(
            "The override is pressed on job-search because that pack's manifest is untracked, "
            "so the in/auto round trip leaves committed state untouched. It writes a new "
            "updated_at into experts/job-search/manifest.json and restores overrides to {}."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
