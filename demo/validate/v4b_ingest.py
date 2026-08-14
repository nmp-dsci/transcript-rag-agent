"""Judge whether the two empty topics were really ingested, from the app's own surfaces.

    uv run python -m src.cli serve --port 8021                 # in one terminal
    PYTHONPATH=. uv run --group demo python -m demo.validate.v4b_ingest

V4b is the one slice in s11 whose deliverable is *content* rather than a
component: two of the user's four intended topics — system design and app
architecture — were nearly empty, D1 put both in scope, and the slice promises
at least five videos each. Content slices are the easiest kind to fake and the
hardest kind to check. A manifest listing five video ids, a corpus counter that
went from 35 to 56, a directory of built packs — none of those is the claim. The
claim is that a reader who opens this app now finds two topics that were not
there before, can read the transcripts behind them, and gets answers drawn from
them when they ask.

So this script asserts four things, in the order a sceptic would check them, and
each one only against rendered page text:

* **Presence.** The RAG Pipeline corpus tree lists at least five videos for each
  topic, under at least three distinct channels. Five videos from one creator is
  a playlist, not coverage, and the D1 wording ("both topics in scope") is not
  met by one lecturer's back catalogue. Video ids are taken from the rendered
  ``img.thumb`` src — the only place the corpus tree puts an id in the DOM — so
  the assertion names ids rather than titles a reader could not disambiguate.

* **Substance.** Opening each new video renders one ``.chunkcard`` per chunk the
  tree and the header claim, every card carries transcript prose, and no card
  carries an HTML entity. That last probe is not hypothetical: this repo has
  shipped chunks whose stored text was still escaped after a ``--refresh``, and
  a count of 63 chunks is equally true of 63 chunks of ``&amp;#39;``. The rendered
  character total is divided by the rendered duration, because a stub video and
  a real one both report a plausible chunk count while only one of them holds
  roughly a speaker's worth of words per minute.

* **Reach.** "Ingested" and "retrievable" are different claims and this repo has
  shipped the first without the second. So the Retrieval Lab is asked a question
  only the new material can answer, and every ranked row — semantic *and* BM25 —
  must come from a newly ingested video of that topic; then row 1 is clicked and
  the detail pane must resolve the truncated ``wXvl·`` prefix to the full id.
  Chat is then asked the same question end to end, and the Sources list under
  the answer must name new videos and nothing else. The Retrieval Lab settles
  whether the index holds the material; Chat settles whether an answer reaches
  it. Both are needed, because an index the answering agent never queries is a
  file, not a corpus. One headline query per topic ranks three or four videos
  and stops, so a further query is aimed at each video it did not rank: five
  videos of which two never surface is not five videos, and the headline probe
  alone cannot tell those two cases apart.

* **Discipline.** No channel contributes more than two videos to the new topic
  material (the cap the candidate list worked under), and none of the five
  property/tax videos D5 excluded appears among the new material or among the
  chunks either topic answer cites. The summary-gap insight chip is opened as
  well: a video with no per-video summary can never be chosen by the V2 summary
  pre-filter, so a new topic video sitting in that set would be half-ingested —
  present in the tree, unreachable through the routed path.

Two things this script deliberately does not do.

It does not read ``experts/system-design/manifest.json`` or the Expert-pack
panel to decide which video belongs to which topic. The panel was moving under
concurrent edits at evaluation time, and a manifest is a JSON file — the same
category of evidence this whole directory exists to distrust. Topic membership
comes from the V4b candidate list, which is the ingestion *input*, and the
retrieval probes are what actually tie a topic to its content.

It does not check that the topic packs exclude the property videos at manifest
level. That is a pack claim, not an ingestion claim, and the pack surface is out
of bounds here. What is checked is the reader-visible consequence: a topic
question returns and cites no property chunk.

ENVIRONMENT NOTE — this machine's Metal compiler XPC service is wedged and
``chromium.launch()`` times out after 180s, so this file could not be executed
through Playwright at evaluation time. Every assertion below was executed
instead as the equivalent DOM query against the running app in an already-open
Chrome at the same URL, and the results are recorded in
``artifacts/v4b_ingest/verdict.json`` with the limitation noted there. Slices
v2, v3 and v4 recorded the same constraint the same way. Nothing here is
machine-specific; re-run it once the host can launch a browser again.
"""

from __future__ import annotations

import re
import sys

from demo.validate.harness import UserSession, exit_code, require_server

SLICE = "v4b_ingest"

#: The corpus as it stood before V4b, taken from the last committed chunk-space
#: projection that predates the ingestion (commit 2f58ea1, 35 videos / 715
#: chunks — the figures the slice brief quotes). Anything in the tree and not in
#: here is new, which is how "the new material" is defined without asking the
#: server what it thinks is new.
PRE_V4B = {
    "012UjJeZY2k",
    "15rTnqKBlO8",
    "1jvxxa7tdjw",
    "3pFRqPqzBCM",
    "5N-okeDdIuI",
    "5gLVxMKeSGM",
    "5kxPMauR4fs",
    "7m27Go3K1d0",
    "8mMH6Pq8qnE",
    "AdRL6tKu3Gk",
    "Bw58mV015z4",
    "FpW8aiJPvts",
    "KuXPeGH_Vag",
    "MGjXraYMbhQ",
    "MXLF8b15GhQ",
    "QPUmFKboiqY",
    "RDjwaXnToes",
    "UxVM1xDBdt4",
    "ZiEGOgTC56Y",
    "ZqqzBCg6IGU",
    "_MT4SgfQ8QY",
    "by8wrrXW3So",
    "cvPEiPt7HXo",
    "dQ6RNltrXro",
    "eGmZZFJ-8PY",
    "fD0E57QYSPk",
    "gXf7fRvuaXA",
    "hNzpEeU3a4I",
    "hgIonrdRTSE",
    "iQyg-KypKAA",
    "owIP4Qr39BY",
    "ozwmlFencJI",
    "qfL4K_afFRE",
    "uC-FmLvw1u0",
    "vpU9lZY69O4",
}

#: Topic membership, from the V4b candidate list the ingestion worked from.
#: Listed in full — including the candidates that were never ingested — so a
#: shortfall reads as "3 of 9 arrived" rather than as a silently smaller target.
CANDIDATES: dict[str, set[str]] = {
    "system-design": {
        "1NngTUYPdpI",
        "L521gizea4s",
        "wXvljefXyEo",
        "uFGJVQvR59A",
        "m4q7VkgDWrM",
        "iJLL-KPqBpM",
        "YPorP8BsF_c",
        "5ZjhNTM8XU8",
        "e2iK8pUP9Vs",
    },
    "app-architecture": {
        "5OjqD-ow8GE",
        "fc6_NtD9soI",
        "4qfsmE11Ejo",
        "FXwBWS4qDAA",
        "eiDyK_ofPPM",
        "dnhshUdRW70",
        "KTy4rqgPOjg",
        "EZ05e7EMOLM",
        "co3acmgP2Ng",
        "zzMLg3Ys5vI",
        "msjnfdeDCmo",
    },
}

#: D5's exclusions. Off-mission Australian property/tax material that must not
#: turn up as new topic material or in a topic answer's citations.
PROPERTY_VIDEOS = {"7m27Go3K1d0", "AdRL6tKu3Gk", "Bw58mV015z4", "ZiEGOgTC56Y", "gXf7fRvuaXA"}

#: The bar the slice sets, and the bar it does not state but has to clear:
#: five videos from a single channel is a playlist, not a topic.
MIN_VIDEOS = 5
MIN_CHANNELS = 3
CHANNEL_CAP = 2

#: Questions the pre-V4b corpus — resumes, job search, AI agents, property —
#: cannot answer. If these route anywhere old, the probe is measuring nothing.
PROBES = {
    "system-design": (
        "When should you shard a database rather than add a read replica, "
        "and how do you pick a partition key?"
    ),
    "app-architecture": (
        "Is a modular monolith a better default than microservices, "
        "and how should I find module boundaries in a codebase?"
    ),
}

#: One further query per remaining video, so that "the new material is
#: retrievable" is not settled by the three videos the headline probe happened
#: to rank. Five videos of which two can never be reached is not five videos,
#: and the headline probe alone cannot tell the two cases apart.
COVERAGE_PROBES = {
    "system-design": [
        "cache invalidation, eviction policy and cache aside for a read heavy service",
        "REST API design: resources, verbs, idempotency and status codes",
        "strong versus eventual consistency, serializable isolation and write skew",
    ],
    "app-architecture": [
        "hexagonal ports and adapters: keeping domain logic out of the web framework",
        "how do you measure coupling strength, volatility and distance between components",
    ],
}

#: A transcript's characters per minute of runtime. Conversational English runs
#: ~700–1400; below the floor means a truncated or partially fetched transcript,
#: above the ceiling means the field holds something other than speech.
CHARS_PER_MINUTE = (700.0, 1500.0)

#: Anything left over from a stale escape pass.
ENTITY = re.compile(r"&(?:amp|lt|gt|quot|#39|#x27|nbsp);")

#: ``https://i.ytimg.com/vi/<id>/…`` or the ``vi_webp`` variant — the corpus
#: tree's only rendered carrier of a video id.
THUMB_ID = re.compile(r"/vi(?:_webp)?/([A-Za-z0-9_-]{11})/")

WATCH_ID = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")

#: ``Channel · 63:59 · 2026-02-05… · 63 chunks · 41,075 views``
META_CHUNKS = re.compile(r"(\d+) chunks")
META_CLOCK = re.compile(r"\b(\d+):(\d{2})\b")


def minutes(meta: str) -> float | None:
    match = META_CLOCK.search(meta)
    if not match:
        return None
    return int(match.group(1)) + int(match.group(2)) / 60.0


# ── the corpus tree, as a reader reads it ────────────────────────────────────
def tree_rows(session: UserSession) -> list[dict]:
    """Every video row in the rendered tree: channel, id, title, chunk count."""
    return session.page.evaluate(
        """() => {
            const root = document.querySelector('nav.tree > details');
            root.open = true;
            const rows = [];
            for (const chan of root.querySelectorAll(':scope > .lvl > details')) {
                chan.open = true;
                const channel = chan.querySelector(':scope > summary .label').textContent.trim();
                for (const vid of chan.querySelectorAll(':scope > .lvl > details')) {
                    const img = vid.querySelector(':scope > summary img.thumb');
                    rows.push({
                        channel,
                        thumb: img ? img.getAttribute('src') : '',
                        title: vid.querySelector(':scope > summary .label').textContent.trim(),
                        chunks: parseInt(
                            vid.querySelector(':scope > summary .cnt').textContent.trim(), 10),
                    });
                }
            }
            return rows;
        }"""
    )


def open_video(session: UserSession, title: str) -> dict | None:
    """Click a video row and read back what its detail pane renders."""
    summary = session.page.locator("nav.tree summary").filter(has_text=title[:40]).first
    if summary.count() == 0:
        return None
    summary.scroll_into_view_if_needed()
    summary.click()
    session.page.wait_for_selector(".detail .chunkcard", timeout=20000)
    session.page.wait_for_timeout(600)
    return session.page.evaluate(
        """() => {
            const detail = document.querySelector('.detail');
            const cards = [...detail.querySelectorAll('.chunkcard')];
            const texts = cards.map(c => (c.querySelector('.cbody p') || {textContent: ''}).textContent);
            const link = cards[0].querySelector('.cbody a');
            return {
                title: detail.querySelector('.vhead .t').textContent.trim(),
                meta: detail.querySelector('.vhead .m').textContent.replace(/\\s+/g, ' ').trim(),
                cards: cards.length,
                href: link ? link.getAttribute('href') : '',
                chars: texts.reduce((total, t) => total + t.length, 0),
                shortest: Math.min(...texts.map(t => t.trim().length)),
                sample: texts[0].slice(0, 200),
                entities: texts.filter(t => /&(amp|lt|gt|quot|#39|#x27|nbsp);/.test(t)).length,
            };
        }"""
    )


def presence_checks(session: UserSession, rows: list[dict]) -> dict[str, list[dict]]:
    """Which of the two topics a reader can actually find in the tree."""
    for row in rows:
        match = THUMB_ID.search(row["thumb"] or "")
        row["id"] = match.group(1) if match else None

    unidentified = [r for r in rows if not r["id"]]
    session.check(
        "every corpus row renders something a reader could pin to one video",
        not unidentified,
        f"{len(rows)} rows, {len(unidentified)} with no id in the DOM"
        + (f": {[r['title'][:40] for r in unidentified[:3]]}" if unidentified else ""),
        shot=True,
    )

    new = [r for r in rows if r["id"] and r["id"] not in PRE_V4B]
    session.check(
        "the corpus grew beyond the 35 videos that predate V4b",
        len(new) >= 10,
        f"{len(rows)} videos rendered, {len(new)} of them new since the pre-V4b projection",
    )

    found: dict[str, list[dict]] = {}
    for topic, candidates in CANDIDATES.items():
        mine = [r for r in new if r["id"] in candidates]
        found[topic] = mine
        session.check(
            f"{topic}: at least {MIN_VIDEOS} videos are listed in the corpus tree",
            len(mine) >= MIN_VIDEOS,
            f"{len(mine)} of {len(candidates)} candidates ingested: "
            f"{sorted(r['id'] for r in mine)}",
            shot=True,
        )
        channels = sorted({r["channel"] for r in mine})
        session.check(
            f"{topic}: the videos come from at least {MIN_CHANNELS} different channels",
            len(channels) >= MIN_CHANNELS,
            f"{len(channels)} channels: {channels}",
        )
        over = {c: len([r for r in mine if r["channel"] == c]) for c in channels}
        breached = {c: n for c, n in over.items() if n > CHANNEL_CAP}
        session.check(
            f"{topic}: no channel contributes more than {CHANNEL_CAP} videos",
            not breached,
            f"per-channel counts {over}" + (f"; over cap: {breached}" if breached else ""),
        )
        leaked = [r["id"] for r in mine if r["id"] in PROPERTY_VIDEOS]
        session.check(
            f"{topic}: no D5-excluded property video is among its material",
            not leaked,
            f"property ids present: {leaked or 'none'}",
        )
    return found


def substance_checks(session: UserSession, found: dict[str, list[dict]]) -> None:
    """Open each new video and judge the transcript, not the counter."""
    for topic, rows in found.items():
        mismatched: list[str] = []
        escaped: list[str] = []
        thin: list[str] = []
        rates: list[str] = []
        for row in rows:
            detail = open_video(session, row["title"])
            if detail is None:
                session.check(
                    f"{topic}: {row['id']} opens to a detail pane", False, "row not clickable"
                )
                continue
            claimed = META_CHUNKS.search(detail["meta"])
            claimed_n = int(claimed.group(1)) if claimed else -1
            if not (detail["cards"] == row["chunks"] == claimed_n):
                mismatched.append(
                    f"{row['id']}: tree {row['chunks']}, header {claimed_n}, "
                    f"{detail['cards']} cards"
                )
            if detail["entities"] or ENTITY.search(detail["sample"]):
                escaped.append(f"{row['id']}: {detail['entities']} cards carry HTML entities")
            if detail["shortest"] < 40 or detail["cards"] == 0:
                thin.append(f"{row['id']}: shortest card {detail['shortest']} chars")
            span = minutes(detail["meta"])
            if span:
                rate = detail["chars"] / span
                rates.append(f"{row['id']} {rate:.0f}")
                if not CHARS_PER_MINUTE[0] <= rate <= CHARS_PER_MINUTE[1]:
                    thin.append(
                        f"{row['id']}: {detail['chars']} chars over {span:.1f} min "
                        f"= {rate:.0f}/min, outside {CHARS_PER_MINUTE}"
                    )
            link_id = WATCH_ID.search(detail["href"] or "")
            if link_id and link_id.group(1) != row["id"]:
                mismatched.append(f"{row['id']}: first chunk links to {link_id.group(1)}")

        session.check(
            f"{topic}: the chunk count in the tree, the header and the pane agree",
            not mismatched,
            "; ".join(mismatched) if mismatched else f"{len(rows)} videos consistent",
        )
        session.check(
            f"{topic}: no chunk renders escaped or empty text",
            not escaped and not thin,
            "; ".join(escaped + thin) if (escaped or thin) else f"{len(rows)} videos clean",
            shot=True,
        )
        session.check(
            f"{topic}: each transcript holds a speaker's worth of words per minute",
            not any("outside" in t for t in thin),
            f"chars/min: {', '.join(rates)}",
        )


# ── the Retrieval Lab: is it in the index a reader can query? ────────────────
def rank(session: UserSession, question: str) -> dict[str, list[str]]:
    lab = session.page.locator(".lab")
    lab.get_by_role("textbox").first.fill(question)
    session.click_button("Rank", exact=True)
    session.page.wait_for_selector(".lab .rankcol .rrow", timeout=60000)
    session.page.wait_for_timeout(600)
    return session.page.evaluate(
        """() => Object.fromEntries([...document.querySelectorAll('.lab .rankcol')].map(col => [
            col.querySelector('.h').textContent.trim().split(/\\s+/)[0],
            [...col.querySelectorAll('.rrow')].map(r => r.querySelector('.cid').textContent.trim()),
        ]))"""
    )


def resolve_top_row(session: UserSession) -> str | None:
    """Click ranked row 1 and let the app say which video it belongs to.

    Needed twice over. The lab prints only a four-character prefix, and when
    every result comes from one video it prints no prefix at all — so a
    single-video ranking is exactly the case where the rendered list names
    nothing, and exactly the case worth resolving.
    """
    session.page.locator(".lab .rankcol .rrow").first.click()
    session.page.wait_for_selector(".detail .chunkcard", timeout=20000)
    session.page.wait_for_timeout(600)
    href = session.page.locator(".detail .chunkcard .cbody a").first.get_attribute("href") or ""
    match = WATCH_ID.search(href)
    return match.group(1) if match else None


def reach_checks(
    session: UserSession, found: dict[str, list[dict]], reached: dict[str, set[str]]
) -> None:
    for topic, question in PROBES.items():
        ids = {row["id"] for row in found[topic]}
        # The lab scopes to whatever video is selected, so start from a clean
        # mount: a scoped ranking would only prove the video it was scoped to.
        session.page.goto(session.verdict.url + "/#pipeline", wait_until="networkidle")
        session.page.wait_for_timeout(1200)
        scope = session.page.locator(".lab .chipselect").inner_text().strip()
        session.check(
            f"{topic}: the ranking is run over the whole corpus",
            "Whole corpus" in scope,
            f"scope reads {scope!r}",
        )
        columns = rank(session, question)
        for mode, cells in columns.items():
            # ``wXvl·#c11`` — the lab truncates the id to four characters once
            # results span more than one video.
            prefixes = {cell.split("·")[0] for cell in cells if "·" in cell}
            stale = {p for p in prefixes if not any(i.startswith(p) for i in ids)}
            session.check(
                f"{topic}: every {mode} row comes from a newly ingested video",
                bool(cells) and not stale,
                f"{len(cells)} rows from {sorted(prefixes)}"
                + (f"; not new topic material: {sorted(stale)}" if stale else ""),
                shot=True,
            )
            property_prefixes = {
                p for p in prefixes if any(v.startswith(p) for v in PROPERTY_VIDEOS)
            }
            session.check(
                f"{topic}: no {mode} row comes from a D5-excluded property video",
                not property_prefixes,
                f"property prefixes among results: {sorted(property_prefixes) or 'none'}",
            )

            reached[topic] |= {i for i in ids if any(i.startswith(p) for p in prefixes)}

        # The four-character prefix is not an id. Click through and make the app
        # resolve it, or "wXvl" could be any of eleven-character-space.
        resolved = resolve_top_row(session)
        session.check(
            f"{topic}: the top-ranked row opens the newly ingested video it claims",
            resolved is not None and resolved in ids,
            f"row 1 resolves to {resolved or 'no id'}; new {topic} ids are {sorted(ids)}",
            shot=True,
        )
        if resolved:
            reached[topic].add(resolved)


def coverage_checks(
    session: UserSession, found: dict[str, list[dict]], reached: dict[str, set[str]]
) -> None:
    """Every ingested video of a topic, reached by some whole-corpus query.

    The headline probe in ``reach_checks`` ranks three or four videos and stops.
    A topic where the other three are in the store but never surface is a topic
    with three usable videos, and the slice promised five — so each remaining
    video gets a query aimed at its own subject and the union has to cover them
    all.
    """
    for topic, queries in COVERAGE_PROBES.items():
        ids = {row["id"] for row in found[topic]}
        for query in queries:
            session.page.goto(session.verdict.url + "/#pipeline", wait_until="networkidle")
            session.page.wait_for_timeout(1200)
            columns = rank(session, query)
            cells = [cell for rows in columns.values() for cell in rows]
            prefixes = {cell.split("·")[0] for cell in cells if "·" in cell}
            reached[topic] |= {i for i in ids if any(i.startswith(p) for p in prefixes)}
            if not prefixes:
                # Every row from one video: the list names no id, so resolve it.
                single = resolve_top_row(session)
                if single:
                    reached[topic].add(single)
        missing = sorted(ids - reached[topic])
        session.check(
            f"{topic}: every ingested video surfaces for some whole-corpus query",
            not missing,
            f"{len(reached[topic])} of {len(ids)} reachable: {sorted(reached[topic])}"
            + (f"; never surfaced: {missing}" if missing else ""),
            shot=True,
        )


# ── Chat: does an answer actually reach the new material? ────────────────────
def ask(session: UserSession, question: str) -> bool:
    before = session.page.locator(".msg-bot").count()
    session.page.get_by_role("textbox", name="Question").fill(question)
    session.click_button("Send", exact=True)
    for _ in range(150):
        if session.page.locator(".msg-bot").count() > before:
            session.page.wait_for_timeout(1500)
            return True
        session.page.wait_for_timeout(2000)
    return False


def chat_checks(session: UserSession, found: dict[str, list[dict]]) -> None:
    session.tab("Chat")
    session.page.get_by_role("combobox", name="Answering agent").select_option("rag_llm")
    session.page.wait_for_timeout(300)
    # Judging is a different slice's claim and costs minutes.
    session.click_button("⚙ advanced", exact=True)
    judge = session.page.get_by_text("auto-judge with RAGAS").locator("input")
    if judge.is_checked():
        judge.uncheck()
    session.page.wait_for_timeout(200)

    for topic, question in PROBES.items():
        ids = {row["id"] for row in found[topic]}
        answered = ask(session, question)
        session.check(
            f"{topic}: the question is answered in Chat",
            answered,
            "answer and trace rendered" if answered else "no answer within 5 minutes",
            shot=True,
        )
        if not answered:
            continue
        payload = session.page.evaluate(
            """() => {
                const bot = [...document.querySelectorAll('.msg-bot')].pop();
                const trace = bot.querySelector('details.trace');
                if (trace) trace.open = true;
                const src = [...bot.querySelectorAll('details')].find(
                    d => /Sources/.test(d.querySelector('summary').textContent));
                if (src) src.open = true;
                const prose = [...bot.querySelectorAll('p')]
                    .map(p => p.innerText.replace(/\\s+/g, ' ').trim())
                    .filter(t => t.length > 60).join(' ');
                return {
                    sources: src ? src.innerText : '',
                    cited: [...bot.querySelectorAll('a[href*="watch"]')].map(
                        a => a.getAttribute('href')),
                    words: prose.split(' ').length,
                };
            }"""
        )
        cited = {m.group(1) for h in payload["cited"] if (m := WATCH_ID.search(h or ""))}
        session.check(
            f"{topic}: the answer cites chunks a reader can open",
            len(cited) > 0 and payload["words"] > 60,
            f"{len(payload['cited'])} citation links over {payload['words']} words "
            f"of answer, resolving to {sorted(cited)}",
        )
        session.check(
            f"{topic}: every cited video is one of the newly ingested ones",
            bool(cited) and cited <= ids,
            f"cited {sorted(cited)}; new {topic} ids {sorted(ids)}; "
            f"outside: {sorted(cited - ids) or 'none'}",
            shot=True,
        )
        session.check(
            f"{topic}: no D5-excluded property video is cited",
            not (cited & PROPERTY_VIDEOS),
            f"property ids cited: {sorted(cited & PROPERTY_VIDEOS) or 'none'}",
        )
        session.check(
            f"{topic}: the Sources list names the videos rather than only numbering them",
            all(video_id in payload["sources"] for video_id in sorted(cited)[:2]),
            " ".join(payload["sources"].split())[:180],
        )


# ── half-ingested is a real state, and it is invisible until you look ────────
def summary_gap_check(session: UserSession, found: dict[str, list[dict]]) -> None:
    """A video with no summary can never be picked by the V2 summary pre-filter."""
    session.tab("RAG Pipeline")
    chip = session.page.locator(".pipe-insights .pipe-chip").filter(
        has_text="no transcript summary"
    )
    if chip.count() == 0:
        session.check(
            "no new topic video is missing its per-video summary",
            True,
            "the corpus-health strip reports no missing summaries at all",
        )
        return
    chip.first.click()
    session.page.wait_for_timeout(1200)
    thumbs = session.page.evaluate(
        """() => [...document.querySelectorAll('nav.tree img.thumb')]
                   .map(i => i.getAttribute('src'))"""
    )
    gap = {m.group(1) for src in thumbs if (m := THUMB_ID.search(src or ""))}
    topic_ids = {row["id"] for rows in found.values() for row in rows}
    stranded = gap & topic_ids
    session.check(
        "no new topic video is stranded without a summary the filter can route on",
        not stranded,
        f"{len(gap)} videos have no summary ({sorted(gap)}); "
        f"of the new topic material: {sorted(stranded) or 'none'}",
        shot=True,
    )
    clear = session.page.locator(".pipe-clear")
    if clear.count():
        clear.first.click()
        session.page.wait_for_timeout(400)


def main() -> int:
    require_server()
    with UserSession(SLICE) as session:
        session.tab("RAG Pipeline")
        stats = " · ".join(
            " ".join(session.page.locator(".pipe-stat").nth(i).inner_text().split())
            for i in range(session.page.locator(".pipe-stat").count())
        )
        session.check(
            "the corpus surface states how much is in the corpus",
            bool(re.search(r"\d+ videos", stats)) and bool(re.search(r"\d+ chunks", stats)),
            stats,
            shot=True,
        )

        rows = tree_rows(session)
        found = presence_checks(session, rows)
        substance_checks(session, found)
        reached: dict[str, set[str]] = {topic: set() for topic in CANDIDATES}
        reach_checks(session, found, reached)
        coverage_checks(session, found, reached)
        chat_checks(session, found)
        summary_gap_check(session, found)

        session.note(
            "Topic membership comes from the V4b candidate list, not from "
            "experts/<topic>/manifest.json or the Expert-pack panel: the panel was under "
            "concurrent edit, and a manifest is the kind of evidence this directory exists "
            "to distrust. The retrieval probes are what tie a topic to its content."
        )
        session.note(
            "The corpus tree has no topic dimension — it groups by channel — so 'five videos "
            "for this topic' is asserted as five named ids rendered in the tree, not as a "
            "topic heading a reader could count under."
        )
        return exit_code(session)


if __name__ == "__main__":
    sys.exit(main())
