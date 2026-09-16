---
name: field-guide-writer
description: Write a long-form, fully cited field guide from an exported transcript corpus (guides/<slug>/corpus/*.md) — extraction passes produce evidence JSON, the composer writes guide.html + guide.md under the guide contract. Used by `python -m src.cli guides write` and the Field Guides tab; also readable by any agent that wants to write one by hand.
version: 1
---

# Field guide writer

A field guide is a reference a practitioner acts on, written *from* a corpus
of talks rather than from general knowledge. Every claim traces to a chunk a
reader can open. Ship Like a Studio (`guides/ship-like-a-studio/guide.html`)
is the reference example: read it before writing one.

## Non-negotiables

1. **Evidence comes only from `corpus/*.md` and the `retrieve_chunks` tool.**
   If the corpus does not support a claim, say so in `gaps` — never fill the
   gap from memory. Web tools are off unless the run says otherwise, and a
   web-sourced fact goes in an `<a>` with its URL, never in a `<cite>`.
2. **Every sourced claim carries a cite** —
   `<cite data-video="VIDEO_ID" data-chunk="N" data-quote="verbatim words">Short label</cite>` —
   where `VIDEO_ID@N` is copied from a chunk heading and `data-quote` is a
   short verbatim run (3–12 words) from that chunk. The verifier resolves
   every cite; one that does not resolve fails the guide.
3. **Every `<section>` has an `id`**, and headings, paragraphs and list items
   get stable ids (`{section}-p3`, `{section}-li12`) so reader comments can
   anchor to them. Ids are document-order and never reused.
4. **No external scripts, no inline event handlers.** The page links
   `/guides/guide.css`, may add its own `<style>`, may include its own inline
   `<script>` (copy buttons, reveals), and ends with
   `<script src="/guides/guide-bridge.js"></script>`.
5. **State the provenance on the page**: source video count, chunks read in
   full, extraction passes, compile date — in the hero, like the example.

## Stage A — extraction (one pass per cluster, Sonnet)

Input: the cluster's files under `corpus/`, read **in full**, top to bottom.
Output: `evidence/<cluster>.json`:

```json
{
  "cluster": "cluster-1",
  "videos": ["VIDEO_ID", "..."],
  "video_notes": [{"video_id": "VIDEO_ID", "title": "...", "contributed": "one line: what this talk uniquely adds"}],
  "claims": [
    {
      "theme": "short theme label (3–6 words)",
      "claim": "one specific, actionable statement in your own words",
      "video_id": "VIDEO_ID",
      "chunk_index": 17,
      "quote": "verbatim 3–12 words copied from that chunk",
      "kind": "principle | technique | metric | pitfall | tool | quote-worthy"
    }
  ],
  "disagreements": [{"topic": "...", "positions": [{"video_id": "...", "chunk_index": 0, "position": "..."}]}],
  "gaps": ["questions a reader would ask that these videos do not answer"]
}
```

Aim for 25–60 claims per cluster, concrete over general: numbers, named
tools, step orders, thresholds, the exact failure someone hit. Prefer a
claim with a number to one without. `quote` must be copied, not paraphrased —
it is checked against the chunk text.

## Stage B — composition (one pass, Opus)

Input: every `evidence/*.json`, plus `corpus/*.md` for re-reading where a
claim needs its context, plus `retrieve_chunks` for anything the evidence
files missed. Output: `guide.html` and `guide.md`.

**Skeleton** (exact — the shared stylesheet keys off these classes):

```html
<nav class="site" aria-label="Sections"><div class="bar">
  <a class="brand" href="#top"><span class="mark" aria-hidden="true"></span>TITLE</a>
  <div class="links"><a href="#thesis">Thesis</a> …one per section… </div>
</div></nav>
<header class="hero wrap" id="top"> …eyebrow, h1, p.sub, div.provenance… </header>
<main class="wrap"> …sections… </main>
<footer class="site"><div class="wrap caption">…</div></footer>
```

**Shape** (adapt the sections to the topic; keep the arc):

1. `header.hero` — eyebrow "A field guide from the transcript corpus", a
   serif `h1` with one italic `em`, a `p.sub` that says what the guide is and
   which creators it distils, and `div.provenance` with the counts.
2. **Thesis** section — the operating principle the corpus converges on, an
   `.epigraph` with the one quote-worthy line (cited), a `.triptych` of the
   three moves.
3. **The pipeline / the method** — numbered `.stage` blocks (`div.no` + `h3`
   + `p.why` + `ul`), each bullet cited.
4. Two or three **checklist** sections — `.levers` grids of `.lever` cards, or
   a `.ban` table of *delete on sight* → *replace with*.
5. **Prompts / templates / checklists to copy** — `figure.prompt` with
   `div.label` (title + `<button type="button" data-copy>Copy</button>`),
   `pre`, `div.note`.
6. **Sources** section (`id="sources"`) — `.cluster` groups, each an `ol` of
   `<li><a href="https://www.youtube.com/watch?v=ID" target="_blank" rel="noopener"><cite data-video="ID">Title</cite></a> <span class="tk">— what it contributed</span></li>`.
7. **Gaps** — a short list of what the corpus does not cover (from the
   evidence files). Honest gaps are part of the product.
8. `footer.site`.

**Voice**: confident, specific, second person where it helps ("demand
screenshots"), no hedging filler, no "in this article we will". Numbers
beat adjectives. Every paragraph earns its place.

**Cite density**: every bullet in a stage or lever, every number, every named
technique. A paragraph with no cite is either a transition or a mistake.

**guide.md**: the same sections as markdown with YAML frontmatter
(`title`, `topic`, `compiled_at`, `videos`, `chunks`, `sources: [{video_id,
title}]`), cites written as `[VIDEO_ID@N]` after the claim. This is the copy an
agent reads, so keep the headings identical to the page and drop the
decoration.

## Stage C — revision (one pass, Opus)

Input: the current `guide.html`, `claims.json`, every `evidence/*.json`, and
the open comments (id, anchor id, quoted text, body). Edit **in place**: keep
every id that exists, add new ids in sequence, cite anything new. Output the
edited `guide.html`, `guide.md`, and `receipt.json`:

```json
{
  "summary": "one paragraph on what changed",
  "items": [{"id": "c-07", "outcome": "addressed | deferred | rejected", "reason": "...", "sections": ["thesis-p2"]}],
  "changed_sections": ["thesis-p2", "pipeline-li4"]
}
```

Every comment id appears exactly once. "Rejected" is a legitimate outcome
when the corpus cannot support what was asked — say so plainly.
