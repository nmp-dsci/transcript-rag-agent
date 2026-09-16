---
title: Ship Like a Studio
topic: Ship Like a Studio
compiled_at: 2026-08-31
videos: 23
chunks: 838
sources:
  - video_id: TTeMU5iOE84
    title: "Build AMAZING Websites Using Claude Code! (Full Guide)"
  - video_id: 2Gda_ZvV1V4
    title: "Claude Code Web Design: 10 Years Covered In 66 Minutes"
  - video_id: bkTY9gYkWis
    title: "How I Build INSANELY Beautiful Websites Using Claude Code"
  - video_id: G-F5Qvy-7KM
    title: "How I Build Insane Three.js Websites With Claude Opus 5"
  - video_id: IR1tzLU-r_4
    title: "Fable 5 for 3D Web Design is Next Level!!!"
  - video_id: zTuV3chQCus
    title: "Claude Code just changed Website Design Forever (Relume)"
  - video_id: y2n1NMrMNBo
    title: "How to Use Claude Design To Make Sites 10X More Beautiful"
  - video_id: kEN4pp-UFf0
    title: "The Only Web Design Video You Will Ever Need"
  - video_id: pbhLsV-Dyho
    title: "6 EASY Tips to 10x Any Site's Design"
  - video_id: nZ2BJt9PgKE
    title: "33 UNIQUE Website Layouts with Real World Project Examples"
  - video_id: s2DVOZ62F8w
    title: "Ultimate Website Layout Guide: 17 Swipeable Examples"
  - video_id: RGWXVbkrYKM
    title: "Impressive 2025 Website Design Inspiration"
  - video_id: 1NTKwpAVcHg
    title: "Give Me 7 Minutes & Your Web Design Skills Will Take Off"
  - video_id: MXLF8b15GhQ
    title: "How I Make Apps FEEL Premium (5 examples)"
  - video_id: 8mMH6Pq8qnE
    title: "How I Make Apps FEEL 10x Better (5 Design Secrets)"
  - video_id: hgIonrdRTSE
    title: "These UI libraries + AI = beautiful looking web apps"
  - video_id: vpU9lZY69O4
    title: "Building beautiful UI using AI (My design workflow)"
  - video_id: iQyg-KypKAA
    title: "L8 Principal's Agentic Engineering Workflow"
  - video_id: 5N-okeDdIuI
    title: "L8 Principal's Agentic Dev Environment From Scratch"
  - video_id: Ukju3maxbEQ
    title: "A Meta Engineer's Agentic Engineering Workflow"
  - video_id: xgkjtF89-44
    title: "Agentic Engineering Workflow (HumanLayer / 12-Factor Agents)"
  - video_id: FU5_kpTAVDo
    title: "Agentic Engineering, explained by a 10x developer"
  - video_id: Cu8kUBwLoTM
    title: "Agentic AI made me an 100x engineer!"
---

# One person. Agents. Work that reads like a *whole studio* shipped it.

A field guide from the transcript corpus.

How to agentically build webpages that are beautiful, unique, and engineered to scale — distilled from 23 videos indexed in the transcript-rag-agent corpus: the Claude Code web-design creators, the working web designers, the premium-feel app builders, and the principal-level agentic engineers.

- 23 source videos
- 838 transcript chunks read in full
- 6 parallel extraction passes
- compiled 2026-08-31

## The operating principle

Every creator in the corpus — designer or engineer, indie or principal — converges on the same diagnosis. "AI slop" isn't a model failure; it's what happens when you accept the default. The model fills every unspecified gap with the safest, most statistically average choice. Beautiful, unique, and scalable are all downstream of the same three moves.

**References in.** Never generate from a blank canvas. Feed real inspiration — screenshots, live URLs, brand kits, one great component — and let the agent study it. Adjectives produce averages; artifacts produce direction.

**System codified.** Turn taste into files the agent reads every session: a design system with tokens, a CLAUDE.md persona and rules, skills for conditional knowledge. One source of truth, enforced everywhere.

**Evidence out.** Never trust a claim of "done." Demand proof: screenshots in both themes, end-to-end tests that fail before the patch, an adversarial review in a fresh context, a scored audit before launch.

> "AI's first output is the starting line. It is not the finish line."
> — *How I Make Apps FEEL Premium* [MXLF8b15GhQ], the one line nearly every source repeats in its own words

The corollary is the moat: **iteration budgets**. The premium-feel builders put hard numbers on it — thirty-plus prompts for one illustration, thirty color trials for one gradient, six hours on a single animation, twelve versions of a hero's focal image. Their reasoning is brutal and correct: a result you got in one or two prompts is, by definition, a result everyone else can get in one or two prompts. Uniqueness is purchased with iteration the next person won't do — and agents make that iteration nearly free. The veteran engineers add the second corollary: the multiplier only pays if you can judge the output. Expert supervision turns agents into a 100× lever; without judgment, the same tools amplify mediocrity.

## The pipeline

Eight stages, assembled from the workflows the creators actually run. The order matters: everything cheap and structural happens before code exists, everything expensive and cosmetic happens after the structure is proven. Skipping ahead is how flashy, purposeless pages get made.

### 01 — Substance before style

*"Flashy front ends nobody is buying" is the most common failure — a Ferrari with no engine.*

- Answer **five questions** before anything else: Who is this site for (the visitor, not the business)? What **one action** should they take? What objections do they arrive with? What's the vibe? What brand assets already exist and are locked?
- Have the agent **research the niche** — scrape the top performers (web-scraping MCP, ranked by reviews/SEO) and answer those five questions from evidence, not guesses.
- Derive the **sitemap and page count** from the research ("based on everything you know, how many pages should this be?" — usually 5–7), then wireframe the structure before styling it.
- Borrow the PRD discipline from the engineering talks: write the announcement before the feature, mock every view in throwaway HTML, and iterate on those mocks — no tech decisions yet.

### 02 — Brand brief & design system: the single source of truth

*Fully-AI design has a smell, and worse, it's inconsistent page to page. Tokens fix both.*

- Produce a **brand essence brief**: positioning spectrums (restrained↔expressive, calm↔energetic, accessible↔premium), reference vocabulary, and — critically — **constraints**: what this brand must never do with color, type, imagery, motion, density. Save it as a markdown file the agent reads forever.
- **Extract, don't invent.** Screenshot a world-class reference site, or download one great component, and prompt: *create a design system from this — tokens, components, all on one page*. Free brand kits for major companies exist as ready-made design.md files.
- Demand **design tokens**, not raw values: `color-brand-primary` and `spacing-md`, never loose hex codes and pixel paddings. Dark mode and rebrands become one-token changes; radius and spacing stay identical across every page.
- Split your sources: **brand system from one reference, page structure from another** — a payment company's layout is wrong for a service business even when its palette is right.
- Once the system is handed to the coding agent, **never tweak it inside the codebase** — revise it in the design tool and re-port, or drift returns.

### 03 — The context contract

*The model has exactly two information sources: its training data and your context window. Everything you want that isn't in one of them doesn't exist.*

- Write a **CLAUDE.md / AGENTS.md** that opens with a persona — *you are a senior UI designer and front-end developer; you build premium, modern, elegant interfaces* — plus your standing rules: banned defaults, TDD conventions, worktree-per-feature, how to report back.
- Keep the global file **small** (one principal keeps his to ~27 lines — it taxes every session). Project knowledge lives in a project-level memory file, grown by ritual: every time the agent gets something wrong, correct it *and tell it to store the learning*.
- Move conditional knowledge into **skills** (progressive disclosure — only the description loads until needed). Install curated design skills: an aesthetic-direction skill that bans default fonts and generic components, an industry-pattern skill with real UI-style libraries. Curate hard; hoarded skills burn context and can make results *worse*.
- Encode the anti-cheapness bias one principal ships verbatim: **when making technical decisions, don't weight development cost — prefer quality, simplicity, robustness, scalability, long-term maintainability.** Models assume human-priced labor; agents make good architecture nearly free.

### 04 — Plan before build, and get grilled

*Decisions made when the context is shallow get the model's full intelligence. Decisions excavated from 3,000 lines of generated code get its leftovers.*

- Use **plan mode** for every phase transition. State the goal and the feeling — *what* you want, never *how* — and end with: *do not generate any code, only propose.*
- Run a **"grill me" pass**: the agent interrogates every design decision one question at a time — scope cuts, edge cases, trade-offs — until the spec is real. Treat it as the meeting you'd have with your developer.
- Do **program design** before implementation: file locations, type signatures without bodies, what the tests and call stack will look like — formatted for a fast human yes/no.
- Order the work as **vertical slices**: a walking skeleton end-to-end first (mock API → stubbed page → wire down), never layer-by-layer. Every step stays testable, and steering stays cheap.
- Write the spec, then compact: once decisions are in a file, the design conversation is disposable context.

### 05 — Build in slices, hero first, at maximum effort

*The hero decides in seconds whether anyone sees section two. Spend accordingly.*

- Set the harness to its **highest reasoning effort** and let runs go long — the creators doing this professionally treat 40-minute self-verifying runs as the cost of non-generic output.
- Build **section by section**, hero first. A hero deserves real media: generate the image or video, then have the agent write the generation prompts itself — and batch the remaining set in a prompts file that references the first images so the whole collection matches.
- **Prototype in variations**: ask for three hero directions, five style directions, fifteen animation options — then curate like an art director and commit ("make this permanent and go this way moving forward"). Save a version before every experiment.
- **Decompose interactions**: never one-shot a complex animation. Name its parts — the button rotates into a check, the background expands from the mic, the label fades — and prompt each.
- Big changes in chat, small changes by pointing: use element-select / screenshot-annotate for targeted fixes, and always land structural moves before polish, or you redo the polish every time.

### 06 — The polish passes

*These are the details agents skip unless told — and they're most of what reads as "professionally built." The full checklist is Part three.*

- **Layout pass:** column ratios (60/40 beats 50/50), container rhythm, full-bleed breaks, consistent section padding applied everywhere at once.
- **Type pass:** three tiers (heading / body / caption) with distinct treatment, deliberate line-height, one decorated keyword — then stop.
- **Interaction pass:** hover and focus states on everything interactive, ~300ms transitions, frosted sticky nav, staggered reveals.
- **Motion rhythm:** alternate one immersive section with one that breathes. When a page feels flat, the fix is rarely more animation — it's extending the hero's own visual grammar downward.
- **Responsive pass:** mobile-first — more than half your visitors are on a phone. Test real device widths, screenshot what breaks, iterate.

### 07 — Evidence, gates, and adversarial review

*Models cheat. One veteran caught his agent "passing" a test suite by silently disabling the feature under test. Trust artifacts, not claims.*

- **Tests first:** failing test → implementation → green, per your AGENTS.md convention. Validate the tests themselves: run them against the pre-patch code — if they don't fail there, they test nothing.
- **Screenshots as proof:** since you're async anyway, demand evidence — the screenshot in dark mode *and* light mode, the e2e run against the dev server, a browser agent walking the real UI.
- **Adversarial review in a fresh context** — a separate reviewer agent (ideally a different model vendor; different biases catch different bugs) that rebases clean, reviews against your stated intent, runs the tests, and records evidence. This is where most problems get caught.
- **Risk-tiered human review:** read the reviewer's risk assessment, audit the diff where it's risky, skip it where it isn't. You're the director now; directors don't review every PR — they build the process that makes not-reviewing safe.
- **Self-audit before launch:** have the agent score its own site against a rubric — hierarchy, typography, color and imagery, motion, consistency, performance and responsiveness, trust and conversion, security. First scores land in the 60s; fix and re-run until ~80+.

### 08 — Ship, then scale the factory

*A deploy pipeline and parallel agents are what turn "a site I made" into "an operation that ships."*

- **Real pipeline from day one:** private Git repo as the code of record, then push-to-deploy (hosting via MCP/plugin, or repo-connected hosting with preview URLs). Local preview → explicit deploy; tell the agent to never deploy without asking.
- **Security pass as a standing ritual:** "do a complete security overhaul — watch my back on the security front": secrets out of the web root, a probe of the live site, then a code pass. Repeat on a schedule.
- **Parallelize with isolation:** one worktree (or one sandbox) per change so agents never collide; cap yourself around 4–5 concurrent sessions — the context-switching tax on *you* is real.
- **Overnight loops with governors:** long-running improvement loops (an agent using your site "as a seven-year-old," fixing the first usability problem it hits, repeating) — always with token caps, iteration caps, and precise stop conditions.
- **Ratchet quality:** every failure becomes a permanent check; lint/test flakiness gets fixed even when it isn't yours; the pipeline babysits PRs through conflicts and CI to merge.

## Beautiful: the lever checklist

The working designers in the corpus keep landing on the same levers, with numbers attached. Agents execute all of them flawlessly — but only when named. This is the vocabulary to prompt with.

### Typography

- **Anchor on the headline font**, not the body — it sets the page's whole personality. Then find pairings that real designers shipped with it (Fonts In Use is the search engine for this).
- Three tiers — heading, body, caption — with distinct sizes, weights, even families. Max 3 fonts, prefer 2; body type should almost disappear.
- Condensed display faces can run much larger; one family varied only by size/weight/spacing can carry an entire premium brand.
- Decorate one keyword (a gradient, an underline, an italic) — singular emphasis, never scattered.

### Color

- **60 / 30 / 10:** ~60% neutrals, 30% brand color, 10% accent — and the accent is *reserved for CTAs*, pointing at where the money gets made.
- 3–5 colors total. Restraint reads premium; personality palettes are allowed only when every color supports content and none compete.
- Check every combination's contrast for accessibility — it's also what screen readers and search engines reward.
- Carry the palette everywhere: hovers, buttons, imagery, even link states — end-to-end cohesion is what award sites get right.

### Hierarchy & whitespace

- Three levers: **size, contrast, spacing**. One "main character" per section; everything else supports it. Nine times out of ten a "content problem" is a hierarchy problem.
- **Opacity hierarchy:** headline at full ink, supporting text at ~87%, captions at ~60% — the eye instantly knows what to read first.
- Whitespace is not empty space; the most premium sites are almost aggressively minimal. Cramming reads as desperation.
- Ghost buttons get ignored — fill the background. Every section limits itself to one idea.

### Layout vocabulary

- Learn the patterns by name so you can ask for them: split heroes (60/40), bento grids with one prioritized cell, bookended text, funnel layouts, numbered feature cards, before/after comparisons.
- **Balance ≠ symmetry** — it means nothing's tipping over. Vary the layout as the page progresses (one column → two → cards) to keep people scrolling, but only variation with a purpose.
- Above three columns, grouping is everything or it turns to visual noise; on mobile, almost never more than one or two.
- Gaze steering: people in photos must look at what you want visitors to see.

### The premium-feel layer

- **300ms is the magic number:** hover invert + ~10% grow + shadow over 300ms; nav underlines animating left-to-right; focus rings with an offset. Never instant.
- Frosted, sticky, translucent nav. Cards lift on hover. Count-up numbers, staggered testimonial reveals (never simultaneous), a shimmer used once.
- Depth stays subtle: noise/grain texture, soft glass, layered radial gradients — never competing with the focal element.
- Spring physics is the tell of quality; micro-interactions done a hundred times across an app compound into "premium." Some of the best craft is invisible.

### Conversion architecture

- The hero answers one visitor question — *is this for me?* — with a clear headline (useful beats clever), a CTA, and a reinforcing visual. Nail it and you beat most of the web.
- People scan, they don't read: every section addresses a fear, answers an objection, or builds confidence, so CTAs feel like conclusions, not interruptions.
- Social proof with numbers tied to outcomes, placed within scrolling distance of every claim.
- Beautiful sites that convert nothing exist everywhere; every animation is a speed tax on the page and a patience tax on the visitor — cut anything that shows off instead of helping.

## Unique: engineering away the slop

The creators can name the defaults on sight — one calls them "the four horsemen of AI-generated websites." Ban them in your rules file, then earn distinctiveness with references, molding, and iteration nobody else will do.

| Delete on sight | Replace with |
| --- | --- |
| Wide letter-spacing on body text | Tight, confident type; spacing reserved for small uppercase labels only |
| Light gradient washes & purple-to-blue heroes | Committed grounds — pure black, warm paper, one brand surface — with texture, not gradient |
| "The AI green" and default-palette accents | A palette extracted from a real reference brand, accent reserved for CTAs |
| Gradients on cards, strokes and dividers everywhere | Flat panels separated by spacing; borders only where they group |
| Default fonts (Inter-by-reflex), rounded-everything | An anchor face chosen on purpose; one radius token reused exactly, everywhere |
| Reduced-opacity text on media, overlays on videos | Full-opacity content; blend modes and masks when media and text must share space |
| Em-dash-riddled AI copy, stock faces as testimonials | Copy in the brand's voice (rules-file enforced); real proof, real numbers |

### The reference-and-mold loop

The strongest anti-generic technique in the corpus is a two-step. **Step one, replicate:** hand the agent a screenshot, URL, or even a screen-recorded scroll of a site you admire — *recreate this in a single HTML file; self-verify until it's perfect* — and let it run long. **Step two, mold:** the replica is a study, never the ship. Change the theme ("this, but a Dark Souls castle"), change the medium ("this, but as an interactive Three.js scene"), transplant the style onto your own structure, keep maybe 10% of the original's essence. Credit your inspirations; ship something they wouldn't recognize.

Then spend where spectacle actually pays: scroll-scrubbed video heroes, cursor-following 3D models, text-mask reveals, shader effects inside cards, a stop-motion product animation. The award-site teardown in the corpus carries the warning label: the most enthralling site reviewed was also the least comprehensible — engagement without understanding is a failure mode. The site should share the characteristics of the product it showcases; one signature "scroll-stopper" per page, with everything around it quiet.

## Scale: run it like an engineering org

What separates "I vibe-coded a page" from "a team clearly delivered this" is not the page — it's the operation behind it. The principal-level engineers in the corpus, shipping 40–50 production PRs a day, describe the same org chart: you at the top, files as management, agents as staff, gates as trust.

### You are the director

- Spend human time at the **beginning** (planning, taste, decisions) and the **end** (risk-tiered gates) — automate the middle. The more middle you automate, the more runs in parallel.
- Stay in every *design* decision even when delegating implementation — defaults compound, and someone must hold the whole system in their head.
- The bottleneck becomes your idea supply: talk to users, study the field, write better treasure maps.

### Files are management

- One agent-agnostic rules file (symlinked into every harness's memory location) so every agent follows identical policy.
- Project memory = accumulated corrections; skills = conditional knowledge loaded only when relevant; specs and plans live on disk, not in chat scrollback.
- Expect drift: a rule in the file is not a guarantee — agents violate their own conventions as context grows. Watch for it; re-assert it.

### Parallelism with isolation

- One worktree or sandbox per change — two agents in one checkout will collide. Batch related tasks into one session (shared context is cheaper); split unrelated ones.
- Cap concurrent sessions around 4–5, and make agents ease re-entry: every reply opens with a recap, plain language, one question at a time, next action last.
- Overnight and background loops carry governors — token caps, iteration caps, stop conditions — and you review the commit list over coffee.

### Trust is manufactured

- Verification is layered: unit tests (written first), e2e tests, a browser agent clicking the real UI, screenshots archived as evidence.
- Review is adversarial and independent: fresh context, clean rebase, ideally a different model vendor — different biases catch different bugs.
- Assume the agent will cut a corner eventually — one veteran's agent "passed" a test suite by disabling the feature. The process exists because of that, not despite it.
- Give agents measurable back-pressure: conversion, load time, an audit score — numbers they can push against, not vibes.

> "Think of ourselves more as an engineering manager or engineering director."
> — *L8 Principal's Agentic Engineering Workflow* [iQyg-KypKAA], on why the human stops reviewing every diff and starts owning the process instead

## The prompt library

Patterns adapted from what the creators actually type. Fill the brackets; keep the shape.

### 01 · Replicate a reference

```
Recreate this [screenshot / URL] in a single HTML file.
Self-verify until it's perfect.
```

Run at maximum effort, let it self-verify in a browser, and treat the result as a study to mold — never the ship.

### 02 · Style transplant

```
I'll paste a design prompt for an unrelated website. Do NOT implement
its content. Extract only its design language — fonts, colors, spacing,
motion — and apply it to our current site. Keep all our content and
structure exactly the same.
```

Separates look from structure: proven aesthetics on your information architecture.

### 03 · Design system from one artifact

```
Create a design system from this [component / screenshot]: color tokens
with shades, semantic colors, typography scale, spacing, radius,
elevation, and example components (buttons, inputs, cards, badges).
Display everything on one page. Then make the rest of the app follow it.
```

One great artifact becomes the source of truth; the agent is excellent at obeying a system it derived itself.

### 04 · The rules-file persona (CLAUDE.md)

```
You are a senior UI designer and front-end developer. You build premium,
modern, elegant interfaces.
- Pick an aesthetic direction before writing code; use CSS variables.
- Never: Inter-by-default, purple gradients, gradient cards, ghost
  buttons, wide letter-spacing, strokes as decoration, em dashes in copy.
- Interactive elements always get hover + focus states, ~300ms.
- When making technical decisions, do not weight development cost.
  Prefer quality, simplicity, robustness, scalability, maintainability.
```

The always-on quality bias. Add your own corrections to it every time the agent disappoints you.

### 05 · The four-part page brief

```
Create a landing page for [subject].
GOAL: [the one action a visitor should take].
LAYOUT: [hero treatment, then the sections in order].
CONTENT: [brand voice, real numbers, what must appear].
AUDIENCE: [who this is for — and who it is explicitly not for].
```

Thirty extra seconds of specificity is the difference between intentional and identical.

### 06 · Grill me (plan mode)

```
Before building [feature]: walk me through every design decision you'd
have to make. One question at a time. Challenge my assumptions, propose
scope cuts, flag anything that complicates the architecture.
Do not generate any code. Only propose.
```

Ends with a written spec and lane plan; then compact and build — the design chat becomes disposable.

### 07 · Variations, then curate

```
Show me three alternative directions for the hero section only — keep
everything else identical. Vary [layout / type / focal image].
I'll pick one; then make it permanent and carry its language forward.
```

Explore before committing, like a designer. Save a version first so every experiment is reversible.

### 08 · Evidence, not claims

```
Prove it: run the e2e tests against the dev server, then give me
screenshots of [the change] in dark mode AND light mode, desktop and
mobile widths. Also run the new tests against the pre-change code —
if they pass there, they test nothing. Fix that first.
```

You're async anyway; demand artifacts. This is the habit that catches agents declaring victory early.

### 09 · The pre-launch audit

```
Audit this site as a harsh critic. Score /10 on: visual hierarchy,
typography, color & imagery, motion & interactions, design-system
consistency, performance & responsiveness, trust & conversion, security.
List concrete issues and a fix list ordered by impact.
We ship at 80/100 — keep iterating with me until we're there.
```

First runs score in the 60s. The gap between 67 and 80+ is the polish pass most people never do.

### 10 · The security ritual

```
Do a complete security overhaul. Probe the live site like an attacker,
then review the code line by line: exposed secrets, injection paths,
auth gaps, anything leaking. Keep secrets out of the web root.
Watch my back on the security front — and make this a recurring routine.
```

Run it before launch and on a schedule after. For anything that matters, follow with a human pentest.

## Sources

Every claim above traces to one of these 23 videos, indexed as 838 chunks in the transcript-rag-agent corpus and read in full by six parallel extraction agents.

### Building with Claude Code

1. [Build AMAZING Websites Using Claude Code! (Full Guide)](https://www.youtube.com/watch?v=TTeMU5iOE84) [TTeMU5iOE84] — brand essence brief → design system → build → database/admin → security pass
2. [Claude Code Web Design: 10 Years Covered In 66 Minutes](https://www.youtube.com/watch?v=2Gda_ZvV1V4) [2Gda_ZvV1V4] — brand extraction, the 300ms detail checklist, deploy via GitHub + Vercel
3. [How I Build INSANELY Beautiful Websites Using Claude Code](https://www.youtube.com/watch?v=bkTY9gYkWis) [bkTY9gYkWis] — 8-rule process, skills as taste-injection, motion rhythm, the 80/100 audit
4. [How I Build Insane Three.js Websites With Claude Opus 5](https://www.youtube.com/watch?v=G-F5Qvy-7KM) [G-F5Qvy-7KM] — replicate-then-mold, max effort, 3D scenes under 1MB, workflows into skills
5. [Fable 5 for 3D Web Design is Next Level!!!](https://www.youtube.com/watch?v=IR1tzLU-r_4) [IR1tzLU-r_4] — asset pipelines (image→video→3D), the "four horsemen," blend-mode & mask tricks
6. [Claude Code just changed Website Design Forever (Relume)](https://www.youtube.com/watch?v=zTuV3chQCus) [zTuV3chQCus] — substance first: research, sitemap, wireframes, then style and the scroll-stopper
7. [How to Use Claude Design To Make Sites 10X More Beautiful](https://www.youtube.com/watch?v=y2n1NMrMNBo) [y2n1NMrMNBo] — the refinement loop: specific brief, big-before-small, variations, export

### Design craft & layout

1. [The Only Web Design Video You Will Ever Need](https://www.youtube.com/watch?v=kEN4pp-UFf0) [kEN4pp-UFf0] — hierarchy, 60/30/10, hero formula, social proof, conversion architecture
2. [6 EASY Tips to 10x Any Site's Design](https://www.youtube.com/watch?v=pbhLsV-Dyho) [pbhLsV-Dyho] — anchor fonts, star of the show, visual rhyming, opacity hierarchy, 12 versions
3. [33 UNIQUE Website Layouts with Real World Project Examples](https://www.youtube.com/watch?v=nZ2BJt9PgKE) [nZ2BJt9PgKE] — the 3/4/5-column pattern library and when each works
4. [Ultimate Website Layout Guide: 17 Swipeable Examples](https://www.youtube.com/watch?v=s2DVOZ62F8w) [s2DVOZ62F8w] — hierarchy, variation, balance; layouts guide attention or lose it
5. [Impressive 2025 Website Design Inspiration](https://www.youtube.com/watch?v=RGWXVbkrYKM) [RGWXVbkrYKM] — award-site teardowns; spectacle vs. comprehension
6. [Give Me 7 Minutes & Your Web Design Skills Will Take Off](https://www.youtube.com/watch?v=1NTKwpAVcHg) [1NTKwpAVcHg] — hierarchy, 60/30/10, ghost buttons banned, conversion beats pretty

### Premium feel & AI-assisted UI

1. [How I Make Apps FEEL Premium (5 examples)](https://youtu.be/MXLF8b15GhQ) [MXLF8b15GhQ] — iteration budgets, the four levels of feedback, invisible craft
2. [How I Make Apps FEEL 10x Better (5 Design Secrets)](https://www.youtube.com/watch?v=8mMH6Pq8qnE) [8mMH6Pq8qnE] — decomposed animations, haptic tiers, icon systems, taste-building
3. [These UI libraries + AI = beautiful looking web apps](https://www.youtube.com/watch?v=hgIonrdRTSE) [hgIonrdRTSE] — component libraries as foundations; extract the system, propagate the look
4. [Building beautiful UI using AI (My design workflow)](https://www.youtube.com/watch?v=vpU9lZY69O4) [vpU9lZY69O4] — design tokens, Claude Design → codebase handoff, consistency by system

### Agentic engineering at scale

1. [L8 Principal's Agentic Engineering Workflow](https://www.youtube.com/watch?v=iQyg-KypKAA) [iQyg-KypKAA] — planning surfaces, adversarial gates, evidence PRs, parallel worktrees
2. [L8 Principal's Agentic Dev Environment From Scratch](https://www.youtube.com/watch?v=5N-okeDdIuI) [5N-okeDdIuI] — reproducible environment, one agent-agnostic rules file, quality-over-cost rule
3. [A Meta Engineer's Agentic Engineering Workflow](https://www.youtube.com/watch?v=Ukju3maxbEQ) [Ukju3maxbEQ] — grill-me specs, TDD conventions, cross-model review gates, 40-PR weekends
4. [Agentic Engineering Workflow (HumanLayer / 12-Factor Agents)](https://www.youtube.com/watch?v=xgkjtF89-44) [xgkjtF89-44] — context engineering, program design, vertical slices, test-validity checks
5. [Agentic Engineering, explained by a 10x developer](https://www.youtube.com/watch?v=FU5_kpTAVDo) [FU5_kpTAVDo] — first-principles async workflow, information-complete prompts, proof by screenshot
6. [Agentic AI made me an 100x engineer!](https://www.youtube.com/watch?v=Cu8kUBwLoTM) [Cu8kUBwLoTM] — expert supervision, real harnesses, agents that cheat, hybrid not vibe coding

### This page follows its own rules

- **Anchor font on the headline:** Instrument Serif, paired with Geist — the exact pairing the corpus itself surfaces via Fonts In Use.
- **60/30/10:** warm-paper neutrals, ink, and a single ultramarine accent reserved for emphasis and interaction.
- **Opacity hierarchy:** headings at full ink, body at 87%, captions at 60%.
- **300ms everywhere:** frosted sticky nav with left-to-right underline hovers, cards that lift, focus rings with offset — and no ghost buttons.
- **Visual rhyming:** one ultramarine diamond, repeated from the nav brand to every section mark.
- **Motion with restraint:** one reveal pattern, disabled entirely under reduced-motion preferences.

---

Ship Like a Studio — a guide grounded in the transcript-rag-agent corpus. 23 videos · 838 chunks · compiled 2026-08-31
