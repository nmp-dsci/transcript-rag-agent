---
title: The Interface Scorecard
topic: Front-end UI design — a scored rubric for reviewing and rebuilding app front ends
compiled_at: 2026-09-16
videos: 20
chunks: 1307
sources:
  - video_id: XZf5A0wcruE
    title: Amateur vs Pro UI Design | with examples
  - video_id: B-ytMSuwbf8
    title: Web Design for Beginners | FREE COURSE
  - video_id: wLJ40GV2XEc
    title: Top UI/UX Design Tips - How to Design a Great Bottom Mobile Navigation Bar
  - video_id: vYvzOyTA2z4
    title: How to Be A Web Designer in 2026 (Free Resources & My Best Advice)
  - video_id: lsgVRCJ4fko
    title: "Learn Design Systems: Figma Variables, Components, AI, and More"
  - video_id: r-MPiu0E4W4
    title: Design with AI - Full Guide (Tools, Workflows, Design Systems)
  - video_id: vpU9lZY69O4
    title: Building beautiful UI using AI (My design workflow)
  - video_id: 23lJ2YAnlC8
    title: Figma Website Design Tutorial 2026
  - video_id: hgIonrdRTSE
    title: These UI libraries + AI = beautiful looking web apps
  - video_id: YO7R0rYWDl8
    title: 14 Front End System Design Concepts | Explained in 10 Minutes
  - video_id: gXwF0VbTKvs
    title: Real Senior Frontend System Design Interview 2026 (AI Coding Included)
  - video_id: qF5il_9IwME
    title: Build a Full-Stack GenAI Project in 4 Hours (FastAPI, React, Supabase)
  - video_id: NovNcsKX8AU
    title: The Never Ending Lore of Harness | Vivek Trivedy (Product Lead, Langchain)
  - video_id: bkTY9gYkWis
    title: How I Build INSANELY Beautiful Websites Using Claude Code
  - video_id: Oo-5AWdQAt8
    title: Use AI Like a Senior Designer (3 Workflows)
  - video_id: G-F5Qvy-7KM
    title: How I Build Insane Three.js Websites With Claude Opus 5
  - video_id: TTeMU5iOE84
    title: Build AMAZING Websites Using Claude Code! (Full Guide)
  - video_id: omU3zR3K7-U
    title: The Best AI Automation Stack to Learn in 2026
  - video_id: yAOzupIW87E
    title: Data Scientist to AI Engineer Roadmap for 2026
  - video_id: a3SMraZWNNs
    title: "How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)"
---

# The Interface Scorecard

Twelve dimensions, each with a written Low / Medium / High descriptor, drawn from twenty talks by working designers, design-system leads, front-end system-design interviewers and AI-build practitioners. Use it to grade an app you already own, then to specify the version that grades higher.

20 source videos · 1,307 transcript chunks read in full · 5 parallel extraction passes · compiled 2026-09-16.

Cites are written `[VIDEO_ID@CHUNK]` and resolve to a chunk heading in the corpus.

---

## The answer: score the surface, but fix the system

Yes — front-end quality is rubric-able, and this guide is the rubric. But the scoring only works if you separate two things the corpus keeps separating. What a reviewer *sees* is colour, hierarchy, type, spacing, motion. What is actually *wrong* is almost always one layer down: a token that was never given a role, a component that was never given a state, an accent colour that was never given a budget. Score the surface. Then write the remediation against the system, because with a real design system one fix propagates everywhere [vpU9lZY69O4@19], and without one you get a dashboard button and a chat button with different corner radii inside the same app [vpU9lZY69O4@1].

The corpus is unusually consistent about what separates junior work from senior work, and it is not decoration. It is restraint: one accent colour with a job, one primary action per screen, one content focal point, and a spacing system that is arithmetic rather than opinion. The top-tier login screen in the corpus wins on colour, hierarchy, spacing, layout and imagery [XZf5A0wcruE@5] — five nouns, no novelty. The senior dashboard wins by removing things, having stripped away a lot of the brand colour so the data can speak [XZf5A0wcruE@9].

> "The old adage is that if everything is on fire then nothing is on fire, and the firefighter doesn't know which area to address." [XZf5A0wcruE@1]

**Score cold, twelve times.** Run all twelve dimensions before proposing anything. Low = 0, Medium = 1, High = 2, out of 24. The corpus's own self-audit habit is to score out of 100 with a deliberately strict prompt, treat 67/100 as a real first build [bkTY9gYkWis@32], and treat anything above 80 as launch-ready [bkTY9gYkWis@32]. That maps to 19/24 here.

**Name the tier tell.** Every Low has a signature. Heavy drop shadows everywhere are not a style choice; they are what happens when someone can't create separation with contrast [XZf5A0wcruE@15]. Name the tell, not the vibe.

**Propose at token level.** A remediation that says "make the buttons nicer" is unscoreable. One that says "introduce `border-focus` at 2px, retire the three component-specific greys" is. Tokens exist precisely so that nobody hunts hex codes by hand [vpU9lZY69O4@2].

---

## The twelve-dimension scorecard

Score each dimension Low (0), Medium (1) or High (2). The dimensions are equally weighted — that is an editorial call, not a corpus claim; no source in this corpus weights design dimensions against each other. **Total 0–24 · <10 rebuild · 10–18 remediate · 19+ ship.**

Read the Low column as a detector, not an insult. Most of these failures are one variable away from their High.

### 1 · Colour discipline

*Does the palette have a budget, and does the accent have a job?*

- **Low (0)** — The accent colour appears on icons, labels, input outlines, headers and tertiary links at once, so nothing reads as primary [XZf5A0wcruE@1]. Brand colour is painted onto every chart and box regardless of value [XZf5A0wcruE@8]. Body text is pure `#000` [B-ytMSuwbf8@47].
- **Medium (1)** — Accent is reserved for CTAs [XZf5A0wcruE@0] and roughly follows a 60/30/10 split of neutral, brand and accent [vYvzOyTA2z4@8], but data visualisation still borrows the brand palette and greys are picked ad hoc.
- **High (2)** — One purposeful accent threaded through primary, secondary and tertiary actions [XZf5A0wcruE@17]; a tonal "black" built by mixing the brand hue into grey with reusable 75/50/25/10/5% tints [B-ytMSuwbf8@48]; data viz has its own palette; inside the product, brand colour is largely stripped out [XZf5A0wcruE@9]. A black-and-white interface with controlled contrast scores High here [XZf5A0wcruE@3].

### 2 · Hierarchy and emphasis

*Can a first-time user tell, in one second, what this screen is for and what to click?*

- **Low (0)** — Two equally weighted, equally sized CTAs force the user to stop and guess [XZf5A0wcruE@2]. Text links are indistinguishable from placeholder text [XZf5A0wcruE@2]. In a dashboard, the logo and a giant top bar are the focal point [XZf5A0wcruE@6].
- **Medium (1)** — One clear primary action, but secondary and tertiary levels are still muddy; heading and label treatments are consistent within a screen but not across screens.
- **High (2)** — A readable ladder: primary action, secondary action, tertiary text link, each with its own treatment (case, opacity, underline, weight). The interface is treated as a tool, so it is allowed to be plain — simple, easy, straightforward [XZf5A0wcruE@7] — with the main thing kept the main thing [XZf5A0wcruE@10]. Search and filters sit where their scope implies: global controls in the global nav, section filters inside the section [XZf5A0wcruE@18].

### 3 · Typography

*Is the type scale generated, or guessed?*

- **Low (0)** — Four or more font families; sizes chosen per component; body copy below 16px. Base paragraph size should always be 16 [lsgVRCJ4fko@61], because at 14 legibility becomes font-and-weight dependent.
- **Medium (1)** — Two families with intentional contrast — three at the absolute max [vYvzOyTA2z4@10] — and named heading styles that decrease at each increment [23lJ2YAnlC8@4], but the increments are eyeballed.
- **High (2)** — Scale generated from a ratio: a major third (1.25) or an augmented fourth at 1.414 [B-ytMSuwbf8@67], with line-height at 1.2× the size and every value rounded to the nearest multiple of four [lsgVRCJ4fko@63]. Roles are assigned deliberately: the face with greatest small-size readability carries body, buttons, menus and captions [B-ytMSuwbf8@60]; a serif headline is a luxury signal, not a default [B-ytMSuwbf8@58].

### 4 · Spacing and alignment

*Is whitespace arithmetic, or vibes?*

- **Low (0)** — Arbitrary pixel values; ambiguous proximity, where a logo lockup looks half-attached to its heading and grouping reads accidental [XZf5A0wcruE@4].
- **Medium (1)** — An eight-point system is applied to components — every dimension a multiple of eight [B-ytMSuwbf8@71], e.g. a button at 16px vertical and 32px horizontal padding [B-ytMSuwbf8@71] — but section rhythm is inconsistent.
- **High (2)** — Component spacing and section rhythm both come from the scale, and the rhythm is re-scaled per breakpoint rather than inherited: 256px between major sections on large screens [B-ytMSuwbf8@188], stepping down to 168 on tablet and 96 on phone. A formal grid system is optional and can add unnecessary complexity [B-ytMSuwbf8@74].

### 5 · Tokens and design system

*Is there one source of truth, and does changing it change the product?*

- **Low (0)** — Hex codes and pixel values inline in components. No shared vocabulary, so the same button drifts between screens [vpU9lZY69O4@1].
- **Medium (1)** — Variables exist and are named — `padding: spacing-medium`, `color: brand-primary` [vpU9lZY69O4@2] — but some are component-specific, which is always a bad idea because it breaks propagation [lsgVRCJ4fko@36].
- **High (2)** — A tiered architecture (primitive/semantic, or brand/alias/mapped for large orgs) [lsgVRCJ4fko@10] where every mapped variable falls into surface, border, icon or text, the four categories that cover every possible component [lsgVRCJ4fko@38]. Colour scales are derived by opacity-blending the base against white and black so the step size earns the numeric name [lsgVRCJ4fko@17]. Every token carries a written description of when to use it — the most important part [r-MPiu0E4W4@84], and the thing that lets an agent tell a default surface from a secondary one. Dark mode is a token swap [vpU9lZY69O4@2].

### 6 · Components and states

*Does every interactive thing have a full state set, and is anything reused twice a component?*

- **Low (0)** — Buttons with a default state only. Copy-pasted markup instead of components. Icons dropped in as raw vectors of different sizes.
- **Medium (1)** — Default, hover, active and disabled are designed, plus a subdued secondary style [B-ytMSuwbf8@115], and variants are merged into one switchable component rather than five near-duplicates [23lJ2YAnlC8@10].
- **High (2)** — All of Medium, plus a visible focus state on its own `border-focus` token, treated as an accessibility requirement rather than a cosmetic variant [lsgVRCJ4fko@86]. Icons are components with a consistent bounding box so every icon occupies the same container size [lsgVRCJ4fko@79], and share one style — all outline or all solid [B-ytMSuwbf8@79]. Complex components decompose into atoms: a table has cell and heading atoms, not one rigid table baked around today's columns [r-MPiu0E4W4@103]. The rule for promotion is blunt: used more than once, it's a component [lsgVRCJ4fko@210].

### 7 · Responsive and touch

*Does the layout survive the phone, and can a thumb hit it?*

- **Low (0)** — Desktop layout squeezed down. Tap targets sized for a cursor, though a finger is much larger [B-ytMSuwbf8@114]. Mobile nav overlaps the home indicator [wLJ40GV2XEc@6].
- **Medium (1)** — Breakpoints exist — e.g. laptop above 1200px, tablet 1200–600, phone under 600px [B-ytMSuwbf8@184] — and type scale shifts with them, but spacing and component behaviour are hand-patched per screen.
- **High (2)** — A single system with per-breakpoint layout variables, so one semantic value like page margin resolves differently on desktop and mobile without duplicating tokens [lsgVRCJ4fko@205]; you do not maintain a completely different mobile design system [lsgVRCJ4fko@204], only mobile-optimised variants on components that genuinely change. Bottom nav holds three to five tabs, six at most [wLJ40GV2XEc@7], carrying only the most essential destinations [wLJ40GV2XEc@2], each with at least a 44×44px tap area [wLJ40GV2XEc@8] and roughly 24px icons [wLJ40GV2XEc@5].

### 8 · Accessibility

*Does it pass contrast, focus and reading order — the three things this corpus actually measures?*

- **Low (0)** — No contrast pass. Muted text picked by eye. Focus ring removed. Multiple H1s per page — an H1 is the one heading element you never duplicate [vYvzOyTA2z4@23].
- **Medium (1)** — Text hits roughly 4.5:1 for small text and 3:1 for large [vYvzOyTA2z4@17], checked in DevTools, and every palette tint was contrast-checked before the palette was locked [B-ytMSuwbf8@51].
- **High (2)** — All of Medium, plus graphical elements such as inactive nav icons clearing the 3:1 minimum for graphical elements [wLJ40GV2XEc@20] via controlled opacity rather than eyeballed washes; a visible focus state tied to a token [lsgVRCJ4fko@86]; and an explicit reading order in handoff, because the designer's default left-to-right-then-down is not always the logical order [Oo-5AWdQAt8@11]. Run an automated accessibility review as a gate — one such pass on a single frame surfaced 22 issues: 3 critical, 13 major, 6 minor [r-MPiu0E4W4@74].

### 9 · Motion and feedback

*Does motion carry information, or is it decoration applied everywhere?*

- **Low (0)** — Either no motion at all, or wall-to-wall animation and scroll effects, which makes a site playful and visually exhausting [bkTY9gYkWis@26]. Long waits shown as a bare spinner. A signature treatment smeared across the whole UI instead of reserved for a few unique moments [hgIonrdRTSE@12].
- **Medium (1)** — Hover and tap feedback exist on primary controls; transitions are present but unmodulated across the page.
- **High (2)** — Motion has rhythm: high-motion sections alternate with calm static ones, and purposeful motion is what turns a layout into an experience [bkTY9gYkWis@2]. State changes are legible — an active tab gets at least two simultaneous visual changes [wLJ40GV2XEc@11], plus a sliding underline between tabs [wLJ40GV2XEc@21]. Waiting is narrated, not spun: a per-phase text label with a gradient wipe [qF5il_9IwME@145] rather than dots, or intentional loaders that make dead time feel deliberate [hgIonrdRTSE@8]. Stylised shadows are allowed when they are thick, bold and on purpose [XZf5A0wcruE@12] and confined to elements that deserve attention.

### 10 · Performance and data

*Are the non-functional requirements quantified, or asserted?*

- **Low (0)** — "It feels fast." Uncompressed PNG/JPEG heroes. Every keystroke fires an API call. Live widgets poll on a timer, which at scale means clients DDoS your own back end [gXwF0VbTKvs@25].
- **Medium (1)** — Some measurement: First Contentful Paint and Time to Interactive tracked in DevTools before optimising [YO7R0rYWDl8@2], plus compressed WebP assets that take a multi-megabyte image down to about 100KB [vYvzOyTA2z4@25].
- **High (2)** — Goals stated as Core Web Vitals plus real-user monitoring [gXwF0VbTKvs@3] and broken into the three named dimensions: loading speed, input responsiveness, layout stability [gXwF0VbTKvs@21]. A deliberate rendering strategy chosen from the five available [YO7R0rYWDl8@0]. Re-render discipline via memoised components [YO7R0rYWDl8@3], lazy loading so heavy charts load on demand [YO7R0rYWDl8@4], caching by read/write ratio [gXwF0VbTKvs@29], and search inputs debounced by 500ms [YO7R0rYWDl8@11]. Visual richness is not an excuse for weight: a fully 3D landing page in the corpus shipped at under 1MB [G-F5Qvy-7KM@0].

### 11 · Content, trust and conversion

*Is the page honest, and does it have one job?*

- **Low (0)** — Fabricated testimonials [B-ytMSuwbf8@138]. Generic "people at work" stock [B-ytMSuwbf8@81]. A ten-field contact form. Deleted URLs that 404 instead of redirecting [vYvzOyTA2z4@26].
- **Medium (1)** — Real copy and real photography; a hero that carries the strongest message and a CTA in the prime above-the-fold real estate [B-ytMSuwbf8@100]; a minimal contact form of name, email, message [B-ytMSuwbf8@166].
- **High (2)** — All of Medium, plus imagery that looks shot by one photographer in one session [vYvzOyTA2z4@11], a hero carrying a real image or video because text-only heroes look plain [bkTY9gYkWis@16], and instrumented outcomes: analytics key events fired on purchase or form fill [vYvzOyTA2z4@35] so a redesign can be proven rather than argued.

### 12 · Distinctiveness

*Could you tell this product from the other forty built the same week?*

- **Low (0)** — The default generated look — same purple gradients, same Inter font [bkTY9gYkWis@5] — recognisable on sight as fully AI-designed [vpU9lZY69O4@0]. Produced by a vague brief: ask for something too vague and you get something extremely standard [G-F5Qvy-7KM@5].
- **Medium (1)** — A stated aesthetic direction chosen before code, with real typography pairing and a cohesive colour system rather than default components [bkTY9gYkWis@7].
- **High (2)** — A specific reference was curated, then moulded rather than cloned — the skill is growing enough vocabulary and taste to turn a copy into something of your own [G-F5Qvy-7KM@15] — and the builder's own judgement is visibly in it, because with no sauce or input from you [vpU9lZY69O4@11] the product looks like everybody else's. One or two signature moments are held in reserve [hgIonrdRTSE@1].

---

## Delete on sight

Findings that need no debate. If the reviewing agent sees the left column, it writes the right column into the remediation plan without asking.

| Delete on sight | Replace with |
| --- | --- |
| Accent colour on icons, labels, borders and links simultaneously [XZf5A0wcruE@1] | Accent reserved for primary CTAs only [XZf5A0wcruE@0] |
| A second CTA with equal size and weight [XZf5A0wcruE@2] | A text link: "Don't have an account? Register" |
| Pure `#000` body text [B-ytMSuwbf8@47] | A brand-tinted grey with derived 75/50/25/10/5% tints [B-ytMSuwbf8@48] |
| Drop shadows and line separators on everything [XZf5A0wcruE@15] | Separation by colour and contrast; shadows only where attention belongs [XZf5A0wcruE@12] |
| Brand colour applied to every chart series [XZf5A0wcruE@8] | A data-visualisation palette where value drives hue |
| Component-specific tokens [lsgVRCJ4fko@36] | Mapped tokens in surface / border / icon / text [lsgVRCJ4fko@38] |
| Body text under 16px [lsgVRCJ4fko@61] | 16px base with a ratio-derived scale rounded to multiples of four [lsgVRCJ4fko@63] |
| Removed focus rings [lsgVRCJ4fko@86] | A 2px focus border on its own `border-focus` token |
| Mixed outline and solid icons at varying sizes [lsgVRCJ4fko@79] | One icon set, one bounding box, built as components |
| Help, log out and legal in the bottom nav [wLJ40GV2XEc@3] | Three to five essential destinations; everything else in profile or a side menu [wLJ40GV2XEc@7] |
| Per-keystroke API calls and interval polling [gXwF0VbTKvs@25] | 500ms debounce [YO7R0rYWDl8@11] and Server-Sent Events for one-way live data [gXwF0VbTKvs@27] |
| Inline citations or detail links that open a new browser tab [qF5il_9IwME@147] | A slide-over panel that reveals the source in place [qF5il_9IwME@123] |
| Fake testimonials [B-ytMSuwbf8@138] | Nothing, until a real client says something real |

---

## Moving the surface dimensions up a tier

Dimensions 1–4 are where a reviewer forms their verdict in the first second. Each one has a small number of levers that reliably move a Low to a High.

### Colour

- Give the accent a budget before you pick it. It is for primary calls to action and primary areas of concern [XZf5A0wcruE@0].
- Start from three core colours — primary, secondary, background — before any accent exists [23lJ2YAnlC8@5], and store each as a variable, not a filled shape [23lJ2YAnlC8@6].
- Allocate roughly 60% neutral, 30% brand, 10% accent [vYvzOyTA2z4@8].
- Build tints, shades and tones deliberately — white, black and grey mixed into the base [B-ytMSuwbf8@25] — and work in RGB/HSB for screens, with hex as the exchange format [B-ytMSuwbf8@33].
- In practice only three harmonies get used: analogous, monochromatic, complementary [B-ytMSuwbf8@39]. Blue is the internet's most-used colour because it reads as trust [B-ytMSuwbf8@36] — an argument for choosing it deliberately, not by default.
- Internal and enterprise tools earn a restricted palette. One team deliberately shipped black, white and grey because simplicity read as more proper for the use case [qF5il_9IwME@131].

### Hierarchy

- One primary action per screen. A second equally weighted CTA is the single most common junior tell [XZf5A0wcruE@2].
- Differentiate every text role: label, placeholder, body, link. Uppercase, opacity, underline — pick a treatment, but pick one [XZf5A0wcruE@2].
- In dense tools, cut the chrome. Shrink the top bar and sidebar until the charts, not the logo, are the focal point [XZf5A0wcruE@6]. Once the user is inside the product, it's not a branding moment anymore [XZf5A0wcruE@11].
- Place controls by scope: global search in the global nav, section filters in the section, so users can predict what a control will affect [XZf5A0wcruE@18].
- Choose the right disclosure pattern: tabs show exactly one panel; accordions can show several and suit simple text, which is why FAQs use them [B-ytMSuwbf8@158].

### Type

- Anchor the scale at a 16px base, then generate upward with a ratio and 1.2× line-height, rounding to multiples of four [lsgVRCJ4fko@63].
- If you need airier display type, raise the base deliberately — one worked example uses a 21px base [B-ytMSuwbf8@66] with an augmented fourth ratio.
- Every size becomes a named style, never a one-off. Build H1 through H5 as reusable styles so a change is one edit [23lJ2YAnlC8@4], starting around 36px for H1 [23lJ2YAnlC8@3].
- Pair by contrast and by role. Brandon Grotesque over Proxima Nova works because Proxima stays legible at 16–18px where the display face does not [B-ytMSuwbf8@64].
- Source from Google Fonts or Adobe Fonts; the paid library buys you filtering by weight, width and x-height [B-ytMSuwbf8@55].

### Space

- Adopt eight-point spacing and apply it to margins, padding, widths and heights alike — the payoff is consistency and an easier developer handoff [B-ytMSuwbf8@71].
- Fix grouping with layout, not nudging. Putting a lockup inside an auto layout with consistent spacing makes the grouping read as intentional [XZf5A0wcruE@4]; auto layout behaves like flexbox [23lJ2YAnlC8@12], and text blocks set to hug their container re-space themselves as copy changes [23lJ2YAnlC8@18].
- Re-scale section rhythm per breakpoint rather than letting desktop whitespace ride down to the phone: 256 → 168 → 96 [B-ytMSuwbf8@188].
- A formal grid is optional; rulers and guides are often enough and easier to keep honest [B-ytMSuwbf8@74].

---

## The system underneath the score

Dimensions 5 and 6 are where remediation should be written, because they are the only place a fix becomes permanent. A design system is a single source of truth for all things design [vpU9lZY69O4@1]: components, guidelines and brand identity in one place. A token is not a hex code — it is a hex code that has been given a role and assigned to components [lsgVRCJ4fko@1].

### Build the token layer

- Pick an architecture and stick to it: two-tier primitive/semantic, or three-tier brand/alias/mapped when many brand scales must coexist [lsgVRCJ4fko@10].
- Derive colour scales by blending the base against white and black at 80/60/40/20/10% so the numeric names (100, 200, 950) are earned by the step size rather than guessed [lsgVRCJ4fko@17].
- Keep the small token families genuinely small. You rarely need more than three or four border widths or radii [lsgVRCJ4fko@69].
- Write the usage description for every variable. Without it neither a human nor an agent can tell a default surface from a secondary one — the description is the most important part [r-MPiu0E4W4@84].
- Never let one component own a token [lsgVRCJ4fko@36]. The whole return on tokens is that a brand change propagates and nobody search-and-replaces hex codes [vpU9lZY69O4@2].

### Build the component layer

- Componentise on the second use [lsgVRCJ4fko@210] — but don't push single-use flow elements into the shared library.
- Design the state set before the styling: default, hover, active, disabled, plus a subdued secondary [B-ytMSuwbf8@115].
- Merge near-duplicates into one component with switchable variants rather than shipping a component per variant [23lJ2YAnlC8@10].
- Decompose complex components into atoms — cell atoms, heading atoms — so a table isn't welded to one column set [r-MPiu0E4W4@103].
- For mobile, add a mobile-optimised variant to the components that actually change; don't fork the system [lsgVRCJ4fko@204].
- Name every layer and group as you create it. Waiting leaves you with dozens of indistinguishable groups [23lJ2YAnlC8@8].

### Keep it in sync

- If the dev side adopts a framework like Radix, rename every Figma variant and property to match its documentation exactly, or the dev side suffers [lsgVRCJ4fko@229] and you accrue technical debt from two conflicting sources of truth [lsgVRCJ4fko@232].
- Do not clone a public design system wholesale. Carbon and Polaris encode years of another organisation's debt and history — absolutely do not do that [lsgVRCJ4fko@235].
- Budget for the tooling reality: publishing a Figma library requires a paid workspace [lsgVRCJ4fko@200], and dev mode is a separate seat you may need to negotiate temporary access to [lsgVRCJ4fko@226].
- Keep design-system CSS separable from app code at the build level: it changes far less often, so it caches independently [gXwF0VbTKvs@29]. CSS-in-JS injects style tags into the document and defeats that.

Two paths into the system exist in the corpus and both are legitimate. You can hand-build it: a dedicated style-guide page split into typography, colour and illustration frames on a 1440×1024 canvas [23lJ2YAnlC8@1], before any real page layout. Or you can seed it: download the one component whose look you want and tell the agent to create a design system based on this component [hgIonrdRTSE@11], or feed a full-page screenshot with "please create a design system with the attached page" [vpU9lZY69O4@5]. If you have no reference at all, Vercel publishes its own design.md in light and dark as a starting base [vpU9lZY69O4@11]. What the corpus disputes is how far the seeded version gets you — see **Where the corpus disagrees**.

---

## The non-functional half

Dimensions 7, 8 and 10 fail silently. A designer reviewing screenshots will never catch them, which is exactly why they belong in a rubric. Separate what the UI must do from how well it does it at the very start — the difference between whether a car takes you from A to B [gXwF0VbTKvs@2] and how efficiently it does so.

### Rendering and bundle

- Choose deliberately among static generation, incremental static regeneration, server-side rendering, client-side rendering and partial pre-rendering [YO7R0rYWDl8@0].
- Measure before optimising: First Contentful Paint and Time to Interactive in the DevTools performance tab [YO7R0rYWDl8@2].
- Stop cascade re-renders with memoised components [YO7R0rYWDl8@3], memoised callbacks [YO7R0rYWDl8@3] and cached computations [YO7R0rYWDl8@3].
- Lazy-load heavy analytics and chart components on demand [YO7R0rYWDl8@4], and shrink what's left with tree shaking and bundle splitting [YO7R0rYWDl8@4].
- Prefer Vite for new builds; reach for a Webpack-compatible Rust bundler only when speeding up an existing Webpack setup [gXwF0VbTKvs@18]. When a front end grows past safe deploys, split it into independently deployable micro-frontends [YO7R0rYWDl8@1] loaded via Module Federation [YO7R0rYWDl8@1].

### Data and state

- Keep state shapes simple. When two values are always fetched and used together, merge them — simple beats complex [gXwF0VbTKvs@12].
- Don't write custom fetch hooks. Plain `useState` for component state, a library like TanStack for fetching and caching [gXwF0VbTKvs@15].
- Cache server responses in client state so repeat reads don't re-fetch [YO7R0rYWDl8@5], but set an expiry, because other users mutate the same data [YO7R0rYWDl8@7] — in React Query that's `staleTime` [YO7R0rYWDl8@8].
- Cut overfetching at the API shape: request only the fields the UI renders [YO7R0rYWDl8@9], with in-memory caching on repeat queries [YO7R0rYWDl8@9].
- Paginate with a cursor for feeds: fetch N+1 so the extra row tells you whether more exists, accepting that you lose arbitrary page jumps [YO7R0rYWDl8@13].
- Insert a backend-for-frontend when the upstream API shape is volatile or you don't own it, so the UI depends on a stable interface [gXwF0VbTKvs@24] — which also narrows the search space when an agent does root-cause analysis.

### Live data and weight

- Never poll from the client for live widgets. Ten graphs on one screen at a two-second interval is ten requests per client per tick — they will DDoS your back end [gXwF0VbTKvs@25].
- For server-to-client-only updates, Server-Sent Events are the native fit; WebSockets are overkill unless the client also pushes [gXwF0VbTKvs@27].
- Decide what to cache by read/write ratio: yesterday's time-series point never changes, so cache it hard; only the newest point can't be [gXwF0VbTKvs@29].
- Compress and convert images; the same asset drops from megabytes to around 100KB with no visible loss [vYvzOyTA2z4@25]. Rich does not mean heavy: an interactive 3D landing page in this corpus shipped under 1MB [G-F5Qvy-7KM@0].
- Size the deployment for the traffic. A chat-style app for around 40 concurrent users was sized at 2 vCPUs and 4GB RAM [qF5il_9IwME@182], with 1GB called out as too low.

### Accessibility and handoff

- Check contrast before the palette is locked, not after — several tints in a tonal scale will fail [B-ytMSuwbf8@51].
- Targets: roughly 4.5:1 for small text [vYvzOyTA2z4@17], 3:1 for large text, and 3:1 for graphical elements including inactive nav icons [wLJ40GV2XEc@20].
- One H1 per page, stating the page's topic — the one heading you never duplicate [vYvzOyTA2z4@23].
- Annotate reading order in handoff files. Free stamps exist for it [Oo-5AWdQAt8@13] — but place the numbers yourself, because you never forward raw AI annotations [Oo-5AWdQAt8@14].
- Prototype the journey before handoff by linking screens so navigation problems surface early [B-ytMSuwbf8@204]. Test mobile on a real device; a preview that looks fine on your monitor can feel wrong under a thumb [wLJ40GV2XEc@6].

---

## Dimension 12, in detail: what AI slop actually is

Distinctiveness is the dimension most reviewers skip and most users notice. The corpus is specific about the tell: the default output is the same purple gradients, the same Inter font [bkTY9gYkWis@5], and a fully AI-designed app is visually identifiable by its smell [vpU9lZY69O4@0] — odd borders, generic components, no point of view. The root cause is upstream of the model: ask something too vague and you get something extremely standard [G-F5Qvy-7KM@5].

There is also an honest cost. Getting non-slop output is not the cheap path — high-fidelity AI design costs meaningfully more in tokens, money and time [G-F5Qvy-7KM@14]. In one controlled dashboard-generation test, one model returned great quality in 2 minutes and 47,000 tokens [r-MPiu0E4W4@21] while a canvas-native tool burned 155,400 tokens on a single dashboard [r-MPiu0E4W4@28]. Budget for it or accept the score.

### Raise the floor

- Install a design skill that forces a decision before code: pick a real aesthetic direction, require genuine typography pairing, block generic components, enforce CSS variables [bkTY9gYkWis@7].
- Hand-pick two or three skills. Installing everything from a repository burns context and confuses the agent [bkTY9gYkWis@6].
- Run design work at high reasoning effort. Extra-high is described as the bare minimum [G-F5Qvy-7KM@13] for non-slop output — though see **Where the corpus disagrees** on the cost of maxing it everywhere.
- Once a technique works, capture it as a reusable skill instead of re-prompting it [G-F5Qvy-7KM@23]; directories exist for finding them [G-F5Qvy-7KM@24].

### Raise the ceiling

- Seed with real references — three to five screenshots from sites you admire, so the model infers taste instead of guessing from adjectives [bkTY9gYkWis@13].
- Or let the agent fetch them: an inspiration MCP server can be asked to find the top UI for a mobile travel app [G-F5Qvy-7KM@9], and then to break a visual direction down into key themes [r-MPiu0E4W4@37].
- Clone, then mould. Pasting a reference URL with "recreate this in a single HTML file, self-verify until it's perfect" [G-F5Qvy-7KM@11] is a starting point, not a deliverable.
- Generate options before committing. Five style directions from one brief, then converge by grafting across them [TTeMU5iOE84@16], telling the tool to go deeper on one [TTeMU5iOE84@17] while borrowing nav and type from the others.
- Start at the widget, not the page. Senior practice is to generate eight distinctively different versions [Oo-5AWdQAt8@3] of the single most important widget first [Oo-5AWdQAt8@1], then build the surrounding page yourself — and only repeat the exploration on widgets you're stuck on [Oo-5AWdQAt8@5].
- Build section by section, hero first, rather than generating the whole site at once [bkTY9gYkWis@16]. Brief on feeling and objective and let the model propose the how [bkTY9gYkWis@11].

### Know the ceiling

- No current coding model has strong visual judgement despite vendor claims [gXwF0VbTKvs@33] — so anchor them with stable artefacts (a design system, a state diagram) and let them interpolate between checkpoints.
- No tool one-prompts a world-class flow; viral demos hide significant manual polish [r-MPiu0E4W4@5].
- AI has largely commoditised front-end *code*, not front-end *judgement*: there is a real gap between creating the front end and doing the design and UX [omU3zR3K7-U@9].
- Always inject your own taste on top of the generated base, or the product looks like everybody else's [vpU9lZY69O4@11].

---

## The review pass

Six stages. Run them in order; the order is what stops an agent from proposing a redesign before it has a diagnosis.

### 1 · Inventory the surface

*You cannot score what you haven't enumerated.*

- List every distinct screen, every colour actually used, every font family, every spacing value, every component that appears more than once.
- Build a flow home base — one page linking every flow you want reviewed — so an agent can navigate between them and back [Oo-5AWdQAt8@6].
- Capture screenshots with numbered annotations; numbering produces far more precise, itemised findings than prose [qF5il_9IwME@144].

### 2 · Score cold, before any fix

*Scoring and solving in the same breath produces flattery.*

- Assign Low / Medium / High to all twelve dimensions with a one-sentence justification and a screen reference each. No recommendations yet.
- Make the rubric deliberately strict and treat the number as relative: above 50% is not bad, above 80% is launch-ready [bkTY9gYkWis@32].
- Expect the first honest score on a working product to land mid-range — the corpus's own first self-audit came back at 67/100 [bkTY9gYkWis@32].

### 3 · Locate the tier tell

*Every Low has a named cause in this corpus. Use the name.*

- Colour Low is almost always accent overuse [XZf5A0wcruE@1]; hierarchy Low is usually two equal CTAs or undifferentiated link text [XZf5A0wcruE@2].
- Separator-and-shadow clutter is a contrast-skill tell, not a style [XZf5A0wcruE@15].
- Cross-screen inconsistency is a missing-system tell, not a carelessness tell [vpU9lZY69O4@1].

### 4 · Test the flows, not just the frames

*Dimensions 2, 7 and 11 only fail in motion.*

- Run a think-aloud test with someone unconnected to the project: have them scroll the homepage and think out loud [vYvzOyTA2z4@15], then attempt one specific task while you watch for silent assumptions.
- If no user is available, run a persona-driven agent pass: give it a role, experience level and goal, and ask it to narrate expectations, hesitations and confidence on a 1–5 scale [Oo-5AWdQAt8@8] per step, then compare two flows with reasons [Oo-5AWdQAt8@9].
- Treat that verdict as a lens, not a truth — never take an AI suggestion 100% [Oo-5AWdQAt8@10].
- Design the critical flow backwards from its destination, fixing every dead end on the way back to entry [vYvzOyTA2z4@17].

### 5 · Write the remediation at token and component level

*A proposal that can't be diffed can't be scored.*

- Ask for a gap analysis against current state and a plan built from reusable design-system components rather than one-off pieces [qF5il_9IwME@132].
- Express every fix as a token added, retired or re-scoped, or a component variant added — so one edit propagates [vpU9lZY69O4@19].
- Give the agent the design system as context, not instructions. Once a system is attached, generated pages follow the colours, the font, the spacing [vpU9lZY69O4@19] without being re-told.
- Escalate model tier as the work spans more interacting components; fast models that handled scaffolding start failing [qF5il_9IwME@130].
- Split system work into one skill per component category — form elements, navigation, data display — rather than one skill for the library, which makes the agent bounce between components [r-MPiu0E4W4@85].

### 6 · Re-score, and force verification

*Agents grade themselves generously unless the harness forbids it.*

- Re-run the identical rubric after each round of fixes and diff the scores; the audit is only useful as a repeated measurement.
- Add a hook that fires before the agent is allowed to finish, forcing it to recheck the work [NovNcsKX8AU@32] — agents are prone to picking the easy way out in verification [NovNcsKX8AU@33]. Deterministic hooks that constrain bad behaviour are among the most underrated harness features [NovNcsKX8AU@70].
- Keep each remediation sub-task bounded to roughly 50k–150k tokens [NovNcsKX8AU@35]; larger subtasks drift into a high-context zone where quality degrades. Use files as the handoff surface between passes — the file system is the most foundational harness primitive [NovNcsKX8AU@39].
- Stop treating public benchmarks as evidence; build your own representative set of tasks [NovNcsKX8AU@83]. This rubric, applied to your own screens, is that set.
- If you pipe a review agent's findings straight back into a coding agent, supervise it: the two can correct each other into an infinite loop [gXwF0VbTKvs@37].
- Preview locally and never redeploy without explicit approval, so the live product can't drift from what was reviewed [TTeMU5iOE84@46].

---

## Prompts to copy

Four prompts: score, remediate, self-audit in a loop, and user-test without users. Paste them with your screenshots attached and the design system, if one exists, in context.

### Prompt 1 — Cold score

```
You are reviewing a front end against a fixed rubric. Do not propose solutions
in this pass. Scoring only.

CONTEXT
- Screens: [attach screenshots, numbered]
- Repo / URL: [...]
- Design system or tokens, if any: [attach or state "none"]
- Product type: [marketing site | internal tool | consumer app | dashboard]

RUBRIC — score each dimension Low (0), Medium (1) or High (2):
 1. Colour discipline       — does the accent have exactly one job?
 2. Hierarchy and emphasis  — one primary action; links look like links
 3. Typography              — 16px base, ratio-derived scale, <=2 families
 4. Spacing and alignment   — 8pt system; section rhythm re-scaled per breakpoint
 5. Tokens and design system— tiered tokens with written usage descriptions
 6. Components and states   — default/hover/active/disabled/focus; atoms not monoliths
 7. Responsive and touch    — 44x44 targets, safe area, per-breakpoint layout vars
 8. Accessibility           — 4.5:1 text, 3:1 graphics, visible focus, one H1, reading order
 9. Motion and feedback     — rhythm not saturation; two changes on state change
10. Performance and data    — Core Web Vitals stated; no polling; no per-keystroke calls
11. Content, trust, convert — real copy and imagery; minimal forms; instrumented events
12. Distinctiveness         — a chosen aesthetic direction, not the default generated look

OUTPUT (markdown table, then a summary line):
| # | Dimension | Score | Evidence (screen + what you saw) | Tier tell |

Then: TOTAL x/24. Under 10 = rebuild. 10-18 = remediate. 19+ = ship.
Be strict. A dimension you cannot verify from the evidence provided is scored
Low with the reason "unverifiable", not skipped.
```

Strictness is the point: the corpus's own audit prompt is deliberately harsh, which is why 67/100 counts as a real first build and 80 is the launch bar [bkTY9gYkWis@32].

### Prompt 2 — Remediation plan

```
Using the scorecard you just produced, write a remediation plan that would move
every Low to Medium and every Medium to High.

RULES
- First research current UX best practice for this product type, then perform a
  gap analysis against the current app, then plan.
- Every fix must be expressed as one of: a token added / retired / re-scoped,
  a component variant added, a state added, or a layout variable added.
  "Make it look more premium" is not a fix. Reject it in your own output.
- Plan reusable design-system components, never one-off isolated pieces.
- Order the work by score delta per unit of effort. Show the ordering.
- Name the single highest-leverage change first and justify why it is highest.
- For each dimension, state the expected new score and the one observation a
  reviewer would make to confirm it.

OUTPUT
1. Highest-leverage change (one paragraph)
2. Token diff: table of | token | action | value | usage description |
3. Component diff: table of | component | variants added | states added |
4. Per-dimension plan: | # | current | target | changes | confirming observation |
5. Anything you could NOT fix without information you don't have.
```

The gap-analysis-then-plan ordering, and the insistence on reusable components over isolated pieces, both come straight from the corpus [qF5il_9IwME@132]. Section 5 is the honesty clause; the corpus's own gaps list is the model for it.

### Prompt 3 — Self-audit loop

```
Audit the build you just produced against the twelve-dimension rubric.

- Run the app locally, screenshot each route at 1440px and at 390px, and score
  from what you actually see, not from the code you intended to write.
- Score out of 24. List concrete pre-launch fixes, ordered.
- Do not stop at the trivial check. Verify the hard cases: the longest label,
  the empty state, the error state, the slowest network path, keyboard-only
  navigation through the primary flow.
- Then apply the fixes and re-score. Report both scores and the diff.
- Do not deploy. Preview locally and wait for explicit approval.

Stop condition: total >= 19/24 AND no dimension scored Low.
```

Two corpus constraints are wired in on purpose: agents default to the easy way out in verification [NovNcsKX8AU@33], and local-preview-then-approve is what stops the live site drifting from what was reviewed [TTeMU5iOE84@46].

### Prompt 4 — Persona flow test

```
You are a specific user, not an assistant. Stay in character.

PERSONA
- Role: [e.g. operations manager at a 40-person logistics firm]
- Experience level with tools like this: [low / medium / high]
- Constraints: [on a phone, in a hurry, colour-blind, first time in the product]
- Goal for this session: [the one task]

TASK
Navigate from the flow home base and attempt the goal. At every step, state:
1. What you expected to happen
2. What actually happened
3. Where you hesitated, and for how long
4. Your confidence that you are on the right path, 1-5

Then repeat the whole run on Flow B. Finish with: which flow you recommend,
why, and the three moments that most damaged confidence in the losing flow.
Do not soften your critique.
```

Modelled on the corpus's persona-driven flow review, including the 1–5 confidence scale [Oo-5AWdQAt8@8] and the flow home base that lets an agent move between flows [Oo-5AWdQAt8@6]. Its output is a lens, not a verdict [Oo-5AWdQAt8@10].

---

## Where the corpus disagrees

Three splits matter enough to change how you apply the rubric. Don't paper over them.

- **Can AI build your design system?** One position: download a component you like and tell the agent to create a design system based on this component [hgIonrdRTSE@11], treated as reliable. The opposing position, from the two deepest design-system sources: AI cannot do those things [lsgVRCJ4fko@6], because it doesn't know your brand, your components or your product; and no, it cannot [r-MPiu0E4W4@92] — what gets called a generated design system is a few styles, some variables and maybe one button. Asking AI to build tokens is compared to hiring someone to build you a kitchen [r-MPiu0E4W4@95] on the brief "make it dark". The measured middle: out-of-the-box design-system sync works about 28% [lsgVRCJ4fko@214], while per-grouping skills get you to roughly 80–85% [lsgVRCJ4fko@215] with the last 15% manual. *Rubric implication:* a generated system scores Medium on dimension 5 at best until a human has assigned roles and written usage descriptions.
- **How much personality should an interface carry?** One side: interfaces are tools and should be simple, easy, straightforward [XZf5A0wcruE@7] and sometimes a little boring, with brand stripped out. The other: hover micro-interactions, gradient and liquid-metal treatments and playful motion are what mark an exceptional, non-generic product [bkTY9gYkWis@2]. *Rubric implication:* dimensions 9 and 12 are scored against product type. A dense internal dashboard scores High on motion by being restrained; a marketing landing page does not.
- **Is maximum reasoning effort always right?** One source calls extra-high the bare minimum [G-F5Qvy-7KM@13] for design output. Another measured the opposite under a time budget: max effort throughout scored 53.9% due to timeouts [NovNcsKX8AU@36] against 63.6% at high. *Rubric implication:* max effort for the scoring and design-direction passes; high for mechanical remediation.
- **A fourth, smaller split:** whether the craft must be hand-built at all. One source teaches the whole system manually — style guide, colour variables, type styles, component variants — before anything ships [23lJ2YAnlC8@1]. Another argues that being purely a front-end builder is no longer valuable because the build step is commoditised [omU3zR3K7-U@9], while conceding AI still lacks the design and UX judgement. The role the corpus converges on is design engineer [r-MPiu0E4W4@1].

---

## Sources

Twenty talks, read in full across five extraction passes. Grouped by what they contribute to the rubric.

### Craft and critique — the scoring dimensions

1. [Amateur vs Pro UI Design | with examples](https://www.youtube.com/watch?v=XZf5A0wcruE) `XZf5A0wcruE` — the junior-to-senior ladder across login, dashboard and complex screens; the source of most Low descriptors in dimensions 1, 2 and 9.
2. [Web Design for Beginners | FREE COURSE](https://www.youtube.com/watch?v=B-ytMSuwbf8) `B-ytMSuwbf8` — the only end-to-end fundamentals course: briefs, wireframes, colour theory, type scales, the eight-point system, UI patterns, responsive breakpoints.
3. [Top UI/UX Design Tips - How to Design a Great Bottom Mobile Navigation Bar](https://www.youtube.com/watch?v=wLJ40GV2XEc) `wLJ40GV2XEc` — the numeric touch specs: 44×44 targets, 24px icons, 3–5 tabs, 3:1 graphical contrast, safe area.
4. [How to Be A Web Designer in 2026 (Free Resources & My Best Advice)](https://www.youtube.com/watch?v=vYvzOyTA2z4) `vYvzOyTA2z4` — 60-30-10, font-count caps, contrast ratios, one-H1 SEO structure, think-aloud user testing, conversion instrumentation.

### Design systems and tokens — where fixes get written

5. [Learn Design Systems: Figma Variables, Components, AI, and More](https://www.youtube.com/watch?v=lsgVRCJ4fko) `lsgVRCJ4fko` — the deepest token treatment: tier architectures, colour-scale maths, the four mapped categories, focus states, mobile variants, and the sharpest limits on AI.
6. [Design with AI - Full Guide (Tools, Workflows, Design Systems)](https://www.youtube.com/watch?v=r-MPiu0E4W4) `r-MPiu0E4W4` — token/time benchmarks across AI design tools, per-category skills, variable descriptions, the accessibility-review pass that found 22 issues.
7. [Building beautiful UI using AI (My design workflow)](https://www.youtube.com/watch?v=vpU9lZY69O4) `vpU9lZY69O4` — the token definition used throughout, the propagation argument, and the AI-design "smell".
8. [Figma Website Design Tutorial 2026](https://www.youtube.com/watch?v=23lJ2YAnlC8) `23lJ2YAnlC8` — the hand-built path: style-guide page, named type styles, colour variables, combined component variants, auto layout, and the SVG-fidelity migration pitfall.
9. [These UI libraries + AI = beautiful looking web apps](https://www.youtube.com/watch?v=hgIonrdRTSE) `hgIonrdRTSE` — free component sources, and the "extract a design system from one component" workflow plus the reserve-your-signature rule.

### Front-end engineering — the non-functional dimensions

10. [14 Front End System Design Concepts | Explained in 10 Minutes](https://www.youtube.com/watch?v=YO7R0rYWDl8) `YO7R0rYWDl8` — rendering strategies, re-render control, lazy loading, caching, debouncing, cursor pagination.
11. [Real Senior Frontend System Design Interview 2026 (AI Coding Included)](https://www.youtube.com/watch?v=gXwF0VbTKvs) `gXwF0VbTKvs` — functional vs non-functional framing, Core Web Vitals, the polling/SSE decision, read-write-ratio caching, and the visual-intelligence limit on models.
12. [Build a Full-Stack GenAI Project in 4 Hours (FastAPI, React, Supabase)](https://www.youtube.com/watch?v=qF5il_9IwME) `qF5il_9IwME` — a coded stack in practice: shadcn/ui theming, numbered-screenshot bug reports, loading-state and citation-panel patterns, deployment sizing.
13. [The Never Ending Lore of Harness | Vivek Trivedy (Product Lead, Langchain)](https://www.youtube.com/watch?v=NovNcsKX8AU) `NovNcsKX8AU` — the harness side of the review loop: verification hooks, sub-agent sizing, the file system as primitive, and build-your-own-evals.

### AI build workflows — how the remediation gets executed

14. [How I Build INSANELY Beautiful Websites Using Claude Code](https://www.youtube.com/watch?v=bkTY9gYkWis) `bkTY9gYkWis` — the seven premium traits, anti-slop skills, plan-first briefing, section-by-section building, motion rhythm, and the self-audit score that this guide's thresholds borrow.
15. [Use AI Like a Senior Designer (3 Workflows)](https://www.youtube.com/watch?v=Oo-5AWdQAt8) `Oo-5AWdQAt8` — widget-first generation, the persona-driven flow test, and reading-order handoff with mandatory human verification.
16. [How I Build Insane Three.js Websites With Claude Opus 5](https://www.youtube.com/watch?v=G-F5Qvy-7KM) `G-F5Qvy-7KM` — the definition of AI slop, reference-first prompting, reasoning-effort settings, and the under-1MB proof that rich isn't heavy.
17. [Build AMAZING Websites Using Claude Code! (Full Guide)](https://www.youtube.com/watch?v=TTeMU5iOE84) `TTeMU5iOE84` — brand-brief-first design, generating five directions then converging, and the local-preview-before-deploy rule.
18. [The Best AI Automation Stack to Learn in 2026](https://www.youtube.com/watch?v=omU3zR3K7-U) `omU3zR3K7-U` — the React + Vite + shadcn/ui baseline, component ownership, and the claim that AI solved front-end code but not UX.
19. [Data Scientist to AI Engineer Roadmap for 2026](https://www.youtube.com/watch?v=yAOzupIW87E) `yAOzupIW87E` — career sequencing: shipping complete projects with a real front end, and front end as the least familiar layer for backend-leaning engineers.
20. [How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)](https://www.youtube.com/watch?v=a3SMraZWNNs) `a3SMraZWNNs` — read in full; contributed no citable UI-design claims. Listed for provenance completeness, not cited above.

---

## What this rubric cannot score

Honest limits. Every item below is something a reader would reasonably expect and the corpus does not supply. If your reviewing agent tries to score these, it is inventing.

- **Accessibility beyond contrast, focus and reading order.** No source in the corpus covers ARIA patterns, keyboard-navigation specifics, screen-reader testing protocol, or focus management in modals and dialogs. Dimension 8 is therefore narrower than WCAG.
- **Outcome evidence.** No source ties a specific design decision — a colour, a hierarchy change, a layout — to a measured conversion rate or task-completion time. Every High in this rubric is a craft judgement, not a proven lift.
- **Dense enterprise UI.** The pattern vocabulary in the corpus is marketing- and portfolio-oriented. Data tables, complex validation, bulk actions, permissions UI and empty/error states at scale are unaddressed.
- **Motion as a discipline.** Easing curves, durations and a framework for when to animate never appear. The corpus offers examples and a rhythm heuristic, nothing measurable.
- **Desktop and tablet navigation patterns.** The only navigation source scoped to numbers is mobile bottom nav. Sidebar, mega-menu and command-palette patterns are uncovered.
- **Design-system versioning and migration.** How to version tokens across breaking changes, or keep a system in sync post-launch across releases, is never discussed.
- **Team workflows.** No source covers several designers prompting against one shared system simultaneously, or handoff when designer and builder are different people — nearly every example is one person doing both roles.
- **Native mobile.** iOS and Android platform guidelines and platform-specific component libraries are absent; only responsive web variants are covered.
- **Real budgets.** The corpus gives relative token and time figures for AI design tools but no subscription pricing or cost-per-design, so you cannot build a budget from it alone.
- **Dark mode in practice.** Tokens are said to make theming easy, but no source works through an actual dark-mode palette, contrast re-check, or elevation model.

---

*The Interface Scorecard — a field guide from the transcript corpus. 20 source videos · 1,307 transcript chunks read in full · 5 parallel extraction passes · compiled 2026-09-16. Every cite resolves to a chunk you can open; claims the corpus does not support are listed under "What this rubric cannot score" rather than filled in.*
