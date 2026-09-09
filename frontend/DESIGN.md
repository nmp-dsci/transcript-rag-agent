# DESIGN.md — the workbench's visual rules

Read this before changing anything in `frontend/src/theme.css` or adding a
view. It records what the design system is, what must not change, and the
audit that keeps it honest.

`src/theme.css` is the single source of truth. Both themes define the same
token set, so **no component ever branches on the theme** — `index.html`
stamps `data-theme` on `<html>` before first paint and everything reads
tokens from there.

---

## 1. Who this is for

Someone evaluating a RAG system, not someone being sold one. They visit these
pages dozens of times. That single fact decides most of what follows: density
is fine, repetition is not, and standing prose that must be scrolled past on
every visit stops being read at all.

The landing page is the exception — it is the conversion surface, and it is
the only page allowed a hero or an orchestrated reveal.

---

## 2. Type

| Role | Token | Where |
|---|---|---|
| Display | `--display` (Instrument Sans) | `h1–h4`, `.brand`, panel headings |
| Keyword | `--serif` (Instrument Serif italic) | exactly one word: the *lab* in transcript·lab |
| Body / UI | `--sans` (system stack) | everything a reader parses |
| Data | `--mono` | numbers, ids, labels, code, chips |

Both faces are self-hosted in `public/fonts` (48 KB total). No third-party
font request on any page load. They are preloaded in `index.html` because the
brand is in the first paint.

Sizes come from the scale — `--t-3` (9.5px) through `--t3` (18px). **There are
no literal `font-size` values in `theme.css` and there must not be new ones.**
The scale exists because thirteen ad-hoc sizes had accumulated, several doing
the same job half a pixel apart in different panels.

The system stack leads with `system-ui`, not `'Segoe UI'`. Leading with a
Windows face meant macOS silently rendered Helvetica.

---

## 3. Colour

Four ink steps (`--text`, `--text2`, `--muted`, `--dim`), three surfaces
(`--bg`, `--panel`, `--panel2`/`--panel3`), one accent. The accent is reserved
for selection, links and primary actions — it is not decoration.

### Never change without re-deriving both themes together

- **`--good` / `--warn` / `--bad`** carry meaning, not style.
- **The six setup identity colours** (`theme.css`, `[data-key=...] .sw`) are
  pinned *identically across themes on purpose*. They label which agent
  produced an answer, and that label appears in stored history entries and in
  committed screenshots. A palette change must not touch them.

### Contrast is checked, not assumed

```sh
python3 scripts/contrast_audit.py
```

26 declared foreground/background pairs, both themes, WCAG AA (4.5:1 text,
3:1 large/non-text). It exits non-zero on any failure. Run it after touching
a colour token. It has already caught one real regression and one
pre-existing failure.

---

## 4. Motion

One token: `--dur` (200ms) with `--ease`. 200ms rather than the 300ms a
marketing page wants — a workbench is operated, not admired, and a control
that lags its own click reads as slow software.

- Transitions belong on the shared interactive classes only: `.nav button`,
  `.btn`, `.pill`, `.tab`, `.rentry`, table rows, `.themetoggle`.
- **`prefers-reduced-motion` is honoured**: `--dur` collapses to `0.01ms` and
  the two infinite animations (the streaming pulse, the idle mic glow) stop.
  Any new animation must be covered by that block.
- **Never animate a property that can hide content.** Reveals animate
  `opacity` and `transform` only, with `animation-fill-mode: backwards`, so
  an animation that never fires leaves the finished page rather than an
  invisible one.
- **Never animate data.** No count-ups on scores, no bar-grow on eval metrics,
  no row reveals on tables. Numbers that move read as salesmanship in an app
  whose whole claim is "RAG you can audit".

---

## 5. The never-list

1. No hero, tagline or reveal on a workbench tab. The landing only.
2. No animated numbers, anywhere.
3. No parallax, scroll-jacking, autoplay, 3D or shader effects.
4. No CSS framework or component library. Hand-rolled tokens that two themes
   share without branching are the asset, not the debt.
5. No lowering density on the data surfaces. Chunk trees, ranked columns and
   matrix tables are correctly dense for their users.
6. No new literal `font-size`, colour or duration — use a token or add one.
7. No colour as the only status signal; pair it with a glyph or a word.
8. No standing explanation in the default view. Use a `<details class="note-more">`
   whose summary is written as the question it answers, so a first-time reader
   still knows to open it.
9. No fact stated twice on the same screen. Pick the surface it belongs to.
10. No purple-to-blue gradient, no Inter, no ghost primary button, no
    wide-tracked display type — and no reverting to GitHub's palette, which is
    where this app started and what made it look like everyone else's devtool.

---

## 6. Concision

Every fact has one home:

- Corpus totals → the topbar (visible on every tab).
- Answer cost → the provenance row under the answer, once.
- Starter questions → `src/questions.ts`, shared by the landing and chat.
- Methodology → behind a labelled disclosure, never in the default view.

---

## 7. Checks before shipping a visual change

```sh
cd frontend && npx tsc --noEmit && npm test -- --run && npm run build
python3 scripts/contrast_audit.py
```

Then look at it: six pages × both themes × a wide and a narrow viewport.
Confirm zero horizontal overflow (`documentElement.scrollWidth ===
clientWidth`) and no clipped controls.

**Measure before claiming a defect.** Two of the four "defects" in the s21
plan turned out to be artifacts of reading a downscaled screenshot whose edge
cut the image — the pipeline header and the composer were both fine, proven by
measuring `scrollWidth` against `clientWidth` at six widths. Screenshots show
you where to look; the DOM tells you whether you are right.

Note that a browser extension such as Dark Reader will repaint the page and
make any colour judgement from a screenshot worthless. Read the computed token
values instead.
