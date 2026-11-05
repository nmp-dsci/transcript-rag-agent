---
title: Free-GPU Video Production
topic: Generating AI video end to end on free cloud GPUs — MiniMax H3 distillation variants, the dual-T4 notebook, StoryMem character memory, and when to pivot off the free tier
compiled_at: 2026-11-04
videos: 20
chunks: 1368
extraction_passes: 5
sources:
  - video_id: rEjasJZq1t8
    title: MiniMax H3 is Now Crazy Fast (Free Kaggle Notebook)
  - video_id: GM70MDyl94E
    title: "Minute Long AI Story Videos With Consistent Characters? Meet StoryMem"
  - video_id: JfaLQqfXqPA
    title: "RFT, DPO, SFT: Fine-tuning with OpenAI — Ilan Bigio, OpenAI"
  - video_id: Cu8kUBwLoTM
    title: Agentic AI made me an 100x engineer!
  - video_id: E8zpgNPx8jE
    title: Build a Complete End-to-End GenAI Project in 3 Hours
  - video_id: qF5il_9IwME
    title: Build a Full-Stack GenAI Project in 4 Hours (FastAPI, React, Supabase)
  - video_id: Q679gH7oszg
    title: How I Build and Ship Custom AI Solutions for Clients
  - video_id: a3SMraZWNNs
    title: How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)
  - video_id: omU3zR3K7-U
    title: The Best AI Automation Stack to Learn in 2026
  - video_id: 9LU90uI4WRU
    title: Fine-Tune Qwen3.5 0.8B Locally On Your Own Dataset - Full Easy Guide
  - video_id: OYvlznJ4IZQ
    title: 20 AI Concepts Explained in 40 Minutes
  - video_id: NovNcsKX8AU
    title: The Never Ending Lore of Harness | Vivek Trivedy (Product Lead, Langchain)
  - video_id: reISMhbZ2XE
    title: "Building Better AI Agents: Artificial Intelligence Observability How To"
  - video_id: TL527yTpxlk
    title: Complete Beginner's Course on AI Evaluations in 50 Minutes (2025) | Aman Khan
  - video_id: oQdkFPhqaGk
    title: Complete Guide to building an AI Trading Bot with Claude for Beginners
  - video_id: IBYHgDC5L0c
    title: How to make an $8K/month Youtube Channel with Claude
  - video_id: v7qMjy_RxOs
    title: The Honest Guide To Fine-Tuning Local AI In 2026
  - video_id: 4yNswvhPWCQ
    title: "Build an LLM from Scratch 7: Instruction Finetuning"
  - video_id: r-MPiu0E4W4
    title: Design with AI - Full Guide (Tools, Workflows, Design Systems)
  - video_id: lsgVRCJ4fko
    title: "Learn Design Systems: Figma Variables, Components, AI, and More"
---

# Four steps, two T4s, and a memory bank

How to generate AI video end to end on free cloud GPUs — picking a MiniMax H3
distillation variant, running the dual-T4 notebook, diagnosing the failures that
only show up at 480p, bolting long-term character memory onto an open backbone
with StoryMem, and knowing the exact moment the free tier stops being enough.
Distilled from AIQUEST's two open-video-model teardowns, with the fine-tuning,
eval and deployment practice around them drawn from Fahd Mirza, Zen van Riel,
Sebastian Raschka, Dave Ebbelaar, LangChain, Ray Fu, Gaurav Sen and others.

20 source videos · 1,368 transcript chunks read in full · 5 parallel extraction
passes · compiled 2026-11-04.

## The short answer

Yes — stage one runs for free, and no, you will not stay there. The corpus
contains exactly one working end-to-end path for MiniMax H3 on free hardware,
and it is a **Kaggle** notebook on **dual T4 GPUs**, not Colab. Two cells: cell
one installs dependencies and pulls roughly 40 GB of weights in about three
minutes [rEjasJZq1t8@13], cell two splits the diffusion transformer across both
cards at **9.3 GB of VRAM each** and hands you a web UI [rEjasJZq1t8@13]. A
five-second 480p clip came back in **182 seconds** [rEjasJZq1t8@15]. That is your
base case, and it is enough to do real comparative work.

On Colab specifically, be precise about what the corpus does and does not say.
The same creator keeps "all my updated notebooks for Collab and Kaggle" in one
GitHub repo [rEjasJZq1t8@12] — but the MiniMax H3 Turbo notebook is launched by
clicking the *Kaggle* badge, and every hardware number given assumes two T4s
with the text encoder pinned in ~21.9 GB of system RAM [rEjasJZq1t8@13]. No video
in this corpus demonstrates H3 on a single Colab GPU, states Colab's runtime
limits, or shows a single-card fallback. Treat Colab as unverified for this
pipeline and run the verified Kaggle path first; see Gaps.

> "Trying to run a massive model on a free T4 GPU is brutal." [rEjasJZq1t8@0]

That line is the whole reason the four-step distillation LoRAs exist, and why the
free tier works at all. Three moves follow from it.

**Buy speed with distillation, not hardware.** Four-step distillation LoRAs are
what make a massive model tractable on free cards, and their authors claim they
"match or beat the full base model" [rEjasJZq1t8@0]. Distillation is the standard
move — train a small student on a large teacher's outputs [OYvlznJ4IZQ@31].

**Pick the variant by failure mode.** Benchmarks do not decide this. PDMD leads
published leaderboards on visual quality, motion and prompt alignment
[rEjasJZq1t8@1], yet Fused Turbo won the squirrel-on-a-deck composition test
outright and lost the hard-cut test badly. Choose per shot type, not per
leaderboard.

**Buy memory with a LoRA, not a bigger model.** ByteDance's StoryMem is a LoRA
over an existing backbone that "generates videos shot by shot while maintaining a
compact memory bank" [GM70MDyl94E@5] — 0.7 B added parameters on a 14 B base
[GM70MDyl94E@6]. Minute-long continuity is an adapter, not a new model.

## The four variants, and what each one is actually for

Cell one of the notebook exposes a single variable where you type the active
LoRA: `PDMD`, `DMAD`, `fused`, or `official` [rEjasJZq1t8@12]. That one word
changes your failure profile more than any prompt edit will. Here is the
side-by-side, from the creator's own battery of text-rendering, physics, dialogue
and multi-shot tests.

### PDMD — sharpness and prompt obedience

- Top of the published leaderboards against H3's official Turbo LoRA and DMD,
  leading on visual quality, dynamic motion and semantic prompt alignment
  [rEjasJZq1t8@1].
- Won the hardest test in the set — a 10-second multi-character argument with a
  prompted hard camera cut: "strict prompt following, PDMD easily took the crown"
  [rEjasJZq1t8@11].
- Renders individual strands of hair crisply even at 480p; colour profile reads
  slightly washed out and muted next to Fused [rEjasJZq1t8@6].
- Technically runs in two steps instead of four, "if you don't mind a noticeable
  drop in quality" [rEjasJZq1t8@1]. Use it for thumbnail-grade previews only,
  never finals.
- Text-to-video and image-to-video only — no reference-to-video mode
  [rEjasJZq1t8@2].

**Default.** Start here. It won the two tests that matter most for narrative
work: legible text and obeyed cuts.

### DMAD — cinematic motion, softer image

- ByteDance's four-step distillation, claiming a **79% preference win rate**
  against rivals like DMD2 and RCM in their own published data [rEjasJZq1t8@2].
- Trades harsh digital sharpness for motion that "looks organic and fluid" — the
  cat-on-a-turntable test read natural rather than artificial [rEjasJZq1t8@7].
- Weakest on kinetic impact: in the kaiju-versus-skyscraper test "the building
  just sort of collapses on its own" with no sense of weight [rEjasJZq1t8@7].
- Hallucinated extra text across a fighter's back and put "AI Quest" on a
  waistband instead of a belt in the text-rendering test [rEjasJZq1t8@4].
- Cleans up the most of any variant when you move from 480p to 720p, "leaning
  into a remarkably smooth and realistic human render" [rEjasJZq1t8@9].

**Pick for** sci-fi and action where smooth cinematic motion beats micro-detail,
and for any shot you will finish at 720p.

### Fused Turbo — fastest, widest mode support, least predictable

- The LoRA is "baked directly into the model transformer safe tensors file"
  rather than loaded on top, cutting overhead [rEjasJZq1t8@2].
- Finishes a standard five-second render **10–15 seconds faster** than PDMD or
  DMAD, which share nearly identical times [rEjasJZq1t8@3].
- The only variant supporting all three modes — text-to-video, image-to-video and
  reference-to-video, the last taking reference audio plus up to three reference
  images [rEjasJZq1t8@2].
- Surprise winner on unhyped text rendering and on colour vibrancy — it
  "completely steals the show" on the spatial-composition test [rEjasJZq1t8@6].
- Breaks on the two things narrative video needs most: it faked a prompted hard
  cut by letting "the virtual camera just drift awkwardly back and forth"
  [rEjasJZq1t8@11], and "still lagged behind visually" in two-person dialogue
  close-ups [rEjasJZq1t8@10].

**Pick for** single-shot establishing frames, reference-driven generation, and any
pass where render throughput matters more than character fidelity.

### Official 8-step reference-to-video — the baseline you should keep

- Built into the same notebook alongside the three four-step options, selected by
  typing `official` [rEjasJZq1t8@2].
- Twice the steps means it is your control condition. Every "match or beat the
  base model" claim in this space is vendor-published; the corpus contains no
  independent third-party validation of any of the three win rates. Run the
  official model on your own prompts before you believe any of them.
- Selecting `fused` or `official` makes the UI reveal a reference-to-video tab for
  reference audio and up to three images [rEjasJZq1t8@14].

**Always** render one shot per variant including `official` before committing a
project to any single LoRA.

One result held across all four: audio is not the weak link. "The actual spoken
audio and ambient background generation on all three models sounded fantastic,"
even where faces fell apart in the same clip [rEjasJZq1t8@9]. Budget your
debugging time on pixels, not on sound.

## Stage one: the free dual-T4 run, start to finished clip

Five stages. Nothing here needs a credit card, local GPU, or command line — the
environment is "stripped down into just two executable cells" [rEjasJZq1t8@12].
Budget around ten minutes of setup before your first render starts.

### 1. Verify the account before you touch the notebook

This is the single step that silently blocks everyone. Kaggle gates GPU access
behind identity verification, and the failure is not an error message — it is a
greyed-out dropdown.

- Verify your Kaggle account with a phone number in account settings first. Skip
  it and "the GPU options will be grayed out and the notebook won't initialize"
  [rEjasJZq1t8@12].
- Set the accelerator to the **T4 × 2** option, not a single T4 [rEjasJZq1t8@12].
  The model is split across both cards; this is not optional.
- Launch from the creator's GitHub repo by clicking the Kaggle badge next to the
  MiniMax H3 Turbo notebook — it "will load the workspace directly into your
  account" [rEjasJZq1t8@12].

### 2. Choose the LoRA in cell one, then run all

The variant is a string literal, not a UI toggle, so it is decided before the
40 GB download rather than after. Pick deliberately; changing your mind means
re-running cell one.

- Default is `PDMD`. Backspace it and type `DMAD`, `fused`, or `official` to
  switch [rEjasJZq1t8@12].
- Cell one connects to the instance, installs dependencies and pulls ~40 GB of
  weights, "finishing the entire download in about 3 minutes" [rEjasJZq1t8@13].
- Cell two initialises the engine and takes "roughly 3 minutes" before it emits
  your UI links [rEjasJZq1t8@13].

### 3. Read the load logs — they are your capacity budget

These three numbers tell you exactly how much headroom you have, and they are why
the free tier works at all. Memorise them; they decide every pivot decision
later.

- The script "automatically splits the massive diffusion transformer evenly across
  both T4 GPUs," loading exactly 9.3 GB onto each card [rEjasJZq1t8@13].
- Auxiliary components — notably the text encoder — are pinned into system RAM,
  not VRAM [rEjasJZq1t8@13].
- Steady state: ~21.9 GB system memory, just over 10 GB per card
  [rEjasJZq1t8@13]. Roughly 5 GB of VRAM headroom per card is all you have for
  resolution and duration increases.

### 4. Take the Cloudflare link, never the Gradio one

Cell two emits two public URLs and they are not equivalent. Choosing wrong costs
you a mid-render session, and renders are measured in minutes.

- You get both a Gradio link and a Cloudflare tunnel URL. "Grad links have a habit
  of timing out or crashing midsession" — use Cloudflare [rEjasJZq1t8@13].
- The UI shows your selected LoRA name at the top — use it to confirm cell one
  actually took your edit [rEjasJZq1t8@13].
- The interface splits prompt input into dedicated fields for spoken lines, speech
  language, sound-effect prompts and background-music prompts. You can dump it all
  into the main box, but "splitting them out gives you granular creative control"
  [rEjasJZq1t8@14].

### 5. Prototype at 480p, finish at 540p or 720p

Resolution is the main dial you have and it is a two-pass workflow, not a single
choice. Everything you learn about framing at 480p transfers; everything you judge
about fidelity at 480p does not.

- "480p is your best friend for rapid prototyping," finishing a five-second clip in
  under three minutes; 540p is the middle ground, 720p maximum fidelity at longer
  waits [rEjasJZq1t8@14].
- A measured five-second 480p run completed in **182 seconds**, with full
  generation metadata saved below the video player [rEjasJZq1t8@15]. Keep that
  metadata — it is your only reproducibility record.
- Duration goes up to 15 seconds, with aspect-ratio auto-match and an
  image-to-video tab that takes first *and* last frames [rEjasJZq1t8@14].
- A style-LoRA picker ships preconfigured options (realism, people, camera motion,
  360 orbit, weapon combat, spatial physics); you can "stack up to two styles at
  once" or paste a custom Hugging Face repo link [rEjasJZq1t8@14].
- Both PDMD and DMAD "hit their peak performance at 720p" [rEjasJZq1t8@2], so
  re-render your final cuts at 540p or 720p to "clear up the artifacts"
  [rEjasJZq1t8@15].

## Delete on sight

Most 480p "bad model" verdicts are framing bugs. The corpus contains one
controlled experiment that proves it: the same jungle prompt with hikers framed
wide produced "noticeable facial morphing, glitching, and weird artifacts" because
"480p just doesn't provide enough pixel density for distant facial features"
[rEjasJZq1t8@8]. Forcing the camera into a close-up at the same 480p made "the
facial distortion disappear because the model has enough pixel real estate to
resolve the eyes, mouth, and skin texture properly" [rEjasJZq1t8@9]. Framing is a
resolution decision.

| Delete on sight | Replace with |
| --- | --- |
| Wide framing with distant faces at 480p | A close-up, or the same prompt re-run at 720p — which stabilises facial structure though "you can still catch slight warping" [rEjasJZq1t8@9] |
| The Gradio share link | The Cloudflare tunnel URL [rEjasJZq1t8@13] |
| Fused Turbo for prompted hard camera cuts | PDMD, which "handled this brilliantly" with crisp close-up expressions [rEjasJZq1t8@11] |
| Fused Turbo for two-person dialogue close-ups | PDMD or DMAD — Fused's "character rendering can be somewhat inconsistent depending on the scene" [rEjasJZq1t8@10] |
| Two-step PDMD output in a final cut | The full four steps; two steps carries "a noticeable drop in quality" [rEjasJZq1t8@1] |
| 480p as the delivery resolution | 540p or 720p for finals [rEjasJZq1t8@15] |
| One 60-second prompt and hope | Shot-by-shot generation with a memory bank [GM70MDyl94E@5] |
| Blaming StoryMem for visible glitches | Blaming the backbone — "those artifacts, they're coming from the underlying video model" [GM70MDyl94E@1] |
| A vendor's cherry-picked showcase page | Your own pre-generated side-by-side playback page [rEjasJZq1t8@3] |
| "The agent says the render pipeline works" | Watching the output yourself — agents cheat verification, including disabling the thing under test to fake a pass [Cu8kUBwLoTM@28] |

## Long-term and short-term memory, concretely

This is the part of your plan that has a real answer in the corpus, and it is more
specific than "give the model memory." Open video models have a *duration*
ceiling: most "can barely generate 5 to 10 seconds of footage," and when they
chain scenes you get "different faces, different outfits, completely broken
continuity" [GM70MDyl94E@0]. Even Sora 2 "has a hard 15-second limit. You can't
generate anything longer than that in one go" [GM70MDyl94E@1]. StoryMem's claimed
result against that baseline: **28.7% better consistency** than Sora 2 and
**9.4%** over prior state-of-the-art HoloCine [GM70MDyl94E@0], over stories of 20,
43, 45, 46 and 58 seconds with 10–15+ distinct scenes.

### Long-term memory: the memory bank between shots

The architectural choice is the lesson. Rather than train a giant model on long
footage — an approach that "require[s] huge computational costs and often
degrade[s] quality" [GM70MDyl94E@5] — StoryMem keeps the single-shot model and
adds state outside it.

- It is "a Laura that works with existing video models," not a new model — so your
  backbone choice stays open [GM70MDyl94E@5].
- The bank "stores key frames from previously generated shots, characters, scenes,
  stylistic cues" — a compact visual-context store, not a text summary
  [GM70MDyl94E@5].
- Memory enters the diffusion model via **latent concatenation** plus a **negative
  RoPE shift**, which "tells the model that these memory frames are from the past,
  not part of the current sequence" [GM70MDyl94E@6]. That positional tag is the
  whole trick: without it, past frames read as present ones.
- What goes *into* the bank is filtered twice: CLIP features for "semantic key
  frame selection to pick the most meaningful frames," and HPSv3 for "aesthetic
  filtering to make sure they're not storing blurry or lowquality frames"
  [GM70MDyl94E@6]. A memory bank is only as good as its admission policy.
- Training cost stayed small: LoRA fine-tuning on "just 400,000 5-second videos,"
  adding 0.7 B parameters to a 14 B Wan 2.2 base [GM70MDyl94E@6]. Short clips
  taught long-horizon memory.

### Short-term memory: first/last frames and reference conditioning

Within and across adjacent shots, the corpus gives you three conditioning handles
you already have in the H3 notebook — use these before reaching for StoryMem at
all.

- The image-to-video tab accepts first *and* last frames [rEjasJZq1t8@14].
  Chaining shot N's last frame into shot N+1's first frame is the cheapest
  continuity you can buy.
- Fused Turbo and the official model expose reference-to-video: reference audio
  plus "up to three reference images to steer your generation" [rEjasJZq1t8@2] — a
  three-slot short-term identity anchor.
- StoryMem's own memory-to-video mode starts from an explicit initial memory: a
  43-second lovers story began from "a male character and a female character with a
  garden background" [GM70MDyl94E@3]. You seed the bank; you don't hope it forms.
- Environment consistency comes free with character consistency: in the cafe story
  "the cafe environment stays identical" across cuts for the full 20 seconds
  [GM70MDyl94E@4].
- Set expectations honestly. Consistency is "quite good. Not absolutely perfect,
  but very solid" [GM70MDyl94E@5], and glitches do occur — one scene had a
  character "suddenly replace someone who was already standing" [GM70MDyl94E@1].

### What StoryMem costs you to run

- Three weight sets on Hugging Face: text-to-video with Wan 2.2 14B,
  image-to-video, and memory-to-video — plus the backbone weights for Wan 2.2 14B
  itself [GM70MDyl94E@6].
- Unquantised, "you can't run the full model unless you have serious VRAM"
  [GM70MDyl94E@7].
- With an FP8 or FP4 quantised backbone, "you should be able to run this whole
  thing within 16 GB of VRAM" [GM70MDyl94E@7]. That is the number to design your
  hardware around.
- Why quantisation buys that much: dropping 32-bit weights to 8-bit means "75% of
  your memory is expected to be saved," applied after training purely to cut
  inference cost [OYvlznJ4IZQ@33].
- **Blocker:** 16 GB in *one* card is not what the free H3 notebook gives you — it
  gives 9.3 GB on each of two cards with the model sharded across them
  [rEjasJZq1t8@13]. The corpus never demonstrates StoryMem on a free dual-T4
  notebook, and never shows a dual-GPU split for it. This is the hard boundary of
  your base case.

## When to leave the free notebook — and where to go

Four triggers, each with a destination the corpus actually supports. None of them
arrives on day one: do stage one first, learn your prompts and framing at 480p,
and only then spend money.

### Trigger: you need StoryMem's minute-long memory

- Target one card with **≥16 GB VRAM** plus an FP8/FP4 quantised Wan 2.2 14B
  backbone [GM70MDyl94E@7].
- Stay on Nvidia. "Nvidia GPUs have access to CUDA" and it is the path of least
  resistance; AMD's ROCm is a viable but less mature middle ground
  [v7qMjy_RxOs@15].
- Do not buy a Mac for this. "What I would definitely avoid is to fine-tune on
  Apple silicon" — missing ports, much slower than comparable Nvidia
  [v7qMjy_RxOs@17]. The corpus disagrees with itself here: Raschka trains happily
  on a MacBook Air CPU in ~15 minutes for a tiny 355M text model [4yNswvhPWCQ@46]
  — a different scale of problem entirely.

### Trigger: you want your own look, as a style LoRA

- Exhaust the cheaper options first. The decision order is: better prompt, then
  RAG/context, then an agentic loop, and only fine-tune if all of those fail
  [v7qMjy_RxOs@8]. For video, your "better prompt" rung includes the built-in style
  picker and stacking two styles.
- LoRA trains roughly "half to 2% of the parameters" [v7qMjy_RxOs@4] by injecting
  "small trainable matrices into specific layers" while the base stays frozen
  [9LU90uI4WRU@1].
- Starting hyperparameters from the corpus: rank 16 — "16 is a solid sweet spot. I
  always use it" [9LU90uI4WRU@4] — with alpha equal to rank [9LU90uI4WRU@10], and a
  fixed seed for reproducibility [9LU90uI4WRU@9].
- Wall-clock reference point: LoRA fine-tuning a medium model took "2 to 3 hours on
  a 5090" [v7qMjy_RxOs@14]. Note these figures are all from text-model fine-tunes;
  the corpus gives no video-LoRA training benchmarks.
- "Always always inspect your data before training. Garbage in garbage out"
  [9LU90uI4WRU@5] — and dedupe near-duplicates before you start [4yNswvhPWCQ@49].

### Trigger: you need unattended, scheduled batch rendering

- A notebook is an interactive session, not a job runner. The one
  scheduled-automation example in the corpus carries the same caveat — automate the
  run on a clock, but the host machine has to stay on [oQdkFPhqaGk@15].
- The mental model is unglamorous: "deploying is nothing more than pushing all that
  code to some kind of like cloud server," setting env vars, and running the same
  startup command you ran locally [qF5il_9IwME@180].
- Containerise. "The really industry standard way of doing this is first of all to
  use docker" [omU3zR3K7-U@13]; Railway for a first deploy, with its own MCP server
  for agent-managed deployment [omU3zR3K7-U@14].
- For steady-state cost, a bare-metal provider beats hyperscalers at this scale —
  one team has run Hetzner for three years [omU3zR3K7-U@14] and calls it "literally
  10 times cheaper" than Azure/AWS for their workload [Q679gH7oszg@37].
- Expect the memory crunch to move, not vanish: a heavy document-processing library
  on a small cloud instance produced exactly the failure you'd predict — "we were
  running out of memory" — and the fix was a lighter library, not a bigger box
  [E8zpgNPx8jE@104].

### Trigger: clips need to become a channel

- Pick the niche on revenue data, not taste: YouTube RPM ranges from "finance,
  that's $15 to $50 per 1,000 views" down to $3–8 for entertainment
  [IBYHgDC5L0c@1] — a 10× payout spread for identical effort.
- Script to four rules: a hook in the first 30–60 seconds, an early open loop,
  visual tags on every scene for the editor, and "a soft CTA at the end so no smash
  that like energy" [IBYHgDC5L0c@6]. The visual tags are what turn a script into a
  shot list for H3.
- H3 generates its own spoken audio, so voice cloning is optional — but if you want
  a consistent host voice, instant cloning "requires you to record a 10-second
  audio clip for minimum," with at least four clips recommended [IBYHgDC5L0c@7].
- Watch advertised-versus-real tool pricing. An AI-avatar tool advertises from
  $15/month but realistic use needs "something like the $100 a month plan" for
  enough API credits [IBYHgDC5L0c@14]; most other tools' free tiers cover "80% of
  what you need anyway" [IBYHgDC5L0c@3].
- Cadence and patience: "post five to six times a week," with results typically
  visible after two to three months [IBYHgDC5L0c@16]. At 182 seconds per
  five-second 480p clip, do that arithmetic before you commit.
- For assembly and post, ffmpeg is the workhorse — one workflow generates start/end
  frames, runs them through a video model, then "break[s] out the video into
  individual frames using ffmpeg" for frame-level control [r-MPiu0E4W4@110].

## How to actually tell which variant is better

Every number in the Variants section above is vendor-published. The creator's own
method is the one worth copying: he "put together a dedicated comparison web page
playing back my own pre-generated clips so we can watch all three four-step models
side by side" [rEjasJZq1t8@3], on his own prompts, "not cherrypicked marketing
clips" [rEjasJZq1t8@1]. That is a human eval harness. The eval videos in this
corpus tell you how to make it rigorous.

- Start in a spreadsheet, not a platform. "Spreadsheets are the ultimate sort of
  product to use for evaluating LLMs" before any automated judge exists
  [TL527yTpxlk@13].
- Write the rubric before you grade. Define explicit criteria and "define what's
  good or bad for any of these" — for video, that means naming your axes:
  distant-face integrity, text legibility, cut obedience, motion weight, colour
  [TL527yTpxlk@14].
- Size the prompt set small at first: "somewhere between you know five
  at least usually it's around 10" golden examples, and only push past 100 when
  you're chasing production confidence [TL527yTpxlk@23]. At three minutes a render,
  ten prompts × four variants is already a two-hour batch.
- Grade it yourself. Outsourcing human eval entirely "never [proved] useful" — the
  person who owns the product has to be in the sheet [TL527yTpxlk@5]. The strongest
  practitioner habit in the corpus is a principal engineer who "spends three hours
  per day looking at traces and data" [a3SMraZWNNs@34].
- Repeat each prompt. For probabilistic long-horizon generation where one bad step
  derails the whole run, "people usually run it three to five times" to get a
  confidence interval [reISMhbZ2XE@40]. A single clip is an anecdote, not a result.
- Take notes while grading, then compress them: write short notes next to the bad
  outputs and "use the LM to summarize all my notes" into the top few things to fix
  [TL527yTpxlk@21].
- Ship something early. "Ship early and shipped often," even to a tiny cohort —
  production usage is where you discover what you actually needed to test for
  [reISMhbZ2XE@10].
- Calibrate your own expectations with the generic curve: a first build "typically
  gets to 70 or 80% accuracy. After iteration, you can usually get to 90%"
  [Q679gH7oszg@10]. Your first batch of clips will not be your last.
- Keep the artefacts on disk. Generation metadata saves alongside each video
  [rEjasJZq1t8@15], and the file system is "arguably the most foundational harness
  primitive" for persistence across runs [NovNcsKX8AU@39]. A Kaggle session is
  ephemeral; your prompt/seed/clip ledger should not be.

## Copy blocks

Three artefacts to paste. The first is your stage-one checklist, the second is the
shot-list format that makes multi-shot generation tractable, the third is the
variant-comparison rubric.

### Stage one — free dual-T4 run

```
STAGE ONE CHECKLIST — MiniMax H3 on free dual T4

BEFORE THE NOTEBOOK
[ ] Kaggle account verified with a phone number
    (unverified = GPU dropdown greyed out, notebook will not initialise)
[ ] Accelerator set to T4 x 2  — NOT single T4
[ ] Launched via the Kaggle badge from the notebook repo

CELL 1  (~3 min)
[ ] LoRA variable set deliberately: PDMD | DMAD | fused | official
    PDMD    -> sharpest, best prompt/cut obedience      (default)
    DMAD    -> smoothest motion, best 480p->720p gain
    fused   -> ~10-15s faster per 5s render, t2v+i2v+r2v
    official-> 8-step reference-to-video = control condition
[ ] Run all; ~40 GB of weights download

CELL 2  (~3 min)
[ ] Confirm log: 9.3 GB VRAM loaded per card, ~21.9 GB system RAM
[ ] Confirm the LoRA name shown at the top of the UI matches cell 1
[ ] Open the CLOUDFLARE tunnel link — not the Gradio link

FIRST RENDER
[ ] 480p, 5 seconds  -> expect under 3 minutes (measured: 182 s)
[ ] Close-up framing for anything with a face
[ ] Split prompt fields: visual / spoken lines / language / SFX / music
[ ] Save the generation metadata next to the clip

FINALS
[ ] Re-render the keepers at 540p or 720p
[ ] Duration ceiling: 15 s. Style LoRAs: stack max 2.
```

Every number here is measured in [rEjasJZq1t8@13] and [rEjasJZq1t8@15] on one
creator's run. Treat them as the expected shape, not a guarantee.

### Shot list for memory-consistent multi-scene video

```
SHOT LIST — one row per generated shot, max 15 s each

PROJECT MEMORY (seed this before shot 1)
  character_refs:  up to 3 reference images, one per recurring character
  environment_ref: 1 image per recurring set
  style_lora:      max 2 stacked (realism | people | camera motion |
                   360 orbit | weapon combat | spatial physics)
  reference_audio: optional voice/ambience anchor

PER SHOT
  id:              S01
  duration:        5 s        (ceiling 15 s)
  framing:         CLOSE-UP unless no face is in frame
                   -- wide + distant face at 480p = facial morphing
  visual prompt:   <action, subject, camera move, lighting>
  spoken lines:    <dialogue only — own field, not the main prompt>
  language:        <speech language>
  sfx prompt:      <diegetic sound>
  music prompt:    <bed>
  first frame:     <path | S00 last frame>   <- short-term continuity
  last frame:      <path | leave blank>
  seed:            <record it>
  variant:         PDMD | DMAD | fused | official

CUTS
  Want a hard cut inside one shot? Use PDMD.
  Fused Turbo fakes cuts by drifting the camera instead.

CONTINUITY ESCALATION (cheapest first)
  1. Chain shot N last frame -> shot N+1 first frame
  2. Reference-to-video: reference audio + up to 3 reference images
     (fused / official only)
  3. StoryMem LoRA over a quantised Wan 2.2 14B backbone
     -- needs ~16 GB VRAM on one card, FP8/FP4. Not the free notebook.
```

Fields map to the notebook UI described in [rEjasJZq1t8@14]; the escalation
ladder's top rung is StoryMem's documented requirement [GM70MDyl94E@7].

### Variant comparison rubric

```
VARIANT BAKE-OFF — your prompts, your sheet, not their showcase page

SETUP
  prompts: 10 of your own, covering the known failure axes below
  variants: PDMD, DMAD, fused, official  (official = control)
  repeats: 3 per prompt per variant      (one clip is an anecdote)
  resolution: 480p for the bake-off, 720p for the finalists only

ONE ROW PER CLIP
  prompt_id | variant | seed | resolution | render_seconds | notes
  then score 1-5 on each axis:

AXES  (define good/bad BEFORE you watch anything)
  face_integrity_wide   distant faces intact in a wide frame?
  face_integrity_close  eyes/mouth/skin resolved in a close-up?
  text_legibility       prompted on-screen words correct and placed?
  cut_obedience         did a prompted hard cut actually cut?
  motion_weight         does impact read as mass, or does it collapse?
  colour                vibrancy and grading without over-saturation
  audio_sync            dialogue lip/timing match
  prompt_adherence      did it build the scene you asked for?

AFTER GRADING
  - Write one short note beside every clip scoring <=2.
  - Summarise all the notes into the top 3 fixes (framing? resolution?
    variant? prompt structure?). Fix one at a time.
  - Only then change the variant. Framing bugs masquerade as model bugs.
```

The method is lifted from the eval discipline in [TL527yTpxlk@14],
[TL527yTpxlk@21] and the repeat-count guidance in [reISMhbZ2XE@40]. The axes are
the failure modes this corpus actually observed.

## Sources

Twenty talks, read in full across five parallel extraction passes. Only two are
about video generation; they carry the spine of this guide. The other eighteen
supply the fine-tuning, eval, hardware and deployment practice that the two video
talks assume.

### Cluster 1 — Video models and fine-tuning methods

1. [MiniMax H3 is Now Crazy Fast (Free Kaggle Notebook)](https://www.youtube.com/watch?v=rEjasJZq1t8) `rEjasJZq1t8` — AIQUEST. The spine of this guide: head-to-head tests of the three four-step distillation LoRAs, plus exact specs, timings and setup steps for the free dual-T4 notebook.
2. [Minute Long AI Story Videos With Consistent Characters? Meet StoryMem](https://www.youtube.com/watch?v=GM70MDyl94E) `GM70MDyl94E` — AIQUEST. The memory answer: the memory-bank mechanism, negative RoPE shift, CLIP + HPSv3 keyframe filtering, and the 16 GB quantised VRAM figure.
3. [RFT, DPO, SFT: Fine-tuning with OpenAI — Ilan Bigio, OpenAI](https://www.youtube.com/watch?v=JfaLQqfXqPA) `JfaLQqfXqPA` — AI Engineer. Why LoRA underlies every tuning method, data-volume thresholds, and the "avoid fine-tuning until you need it" default.
4. [Agentic AI made me an 100x engineer!](https://www.youtube.com/watch?v=Cu8kUBwLoTM) `Cu8kUBwLoTM` — Bits inside by René Rebe. The verification warning: an agent that faked a passing test by disabling the thing under test, plus real token-cost figures.

### Cluster 2 — End-to-end builds, client delivery and evals

5. [Build a Complete End-to-End GenAI Project in 3 Hours](https://www.youtube.com/watch?v=E8zpgNPx8jE) `E8zpgNPx8jE` — Dave Ebbelaar. The memory-exhaustion failure on a small cloud instance, and the architect-not-passenger stance toward coding agents.
6. [Build a Full-Stack GenAI Project in 4 Hours (FastAPI, React, Supabase)](https://www.youtube.com/watch?v=qF5il_9IwME) `qF5il_9IwME` — Dave Ebbelaar. The deployment mental model, and the reminder that a demo one-shot is a day of real validation.
7. [How I Build and Ship Custom AI Solutions for Clients](https://www.youtube.com/watch?v=Q679gH7oszg) `Q679gH7oszg` — Dave Ebbelaar. The 70–80%-then-90% accuracy curve, and the Hetzner-is-10×-cheaper deployment economics.
8. [How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)](https://www.youtube.com/watch?v=a3SMraZWNNs) `a3SMraZWNNs` — Dave Ebbelaar. The three eval levels and the look-at-your-own-data habit this guide's bake-off rests on.

### Cluster 3 — Stack, LoRA mechanics and concepts

9. [The Best AI Automation Stack to Learn in 2026](https://www.youtube.com/watch?v=omU3zR3K7-U) `omU3zR3K7-U` — Dave Ebbelaar. Docker as the shipping default, Railway for a first deploy, Hetzner for production.
10. [Fine-Tune Qwen3.5 0.8B Locally On Your Own Dataset - Full Easy Guide](https://www.youtube.com/watch?v=9LU90uI4WRU) `9LU90uI4WRU` — Fahd Mirza. Parameter-by-parameter LoRA settings — rank 16, alpha = rank, fixed seed, inspect-your-data-first.
11. [20 AI Concepts Explained in 40 Minutes](https://www.youtube.com/watch?v=OYvlznJ4IZQ) `OYvlznJ4IZQ` — Gaurav Sen. Why distillation and quantisation exist, including the 75%-memory-saved figure for 32→8-bit.
12. [The Never Ending Lore of Harness | Vivek Trivedy (Product Lead, Langchain)](https://www.youtube.com/watch?v=NovNcsKX8AU) `NovNcsKX8AU` — GroundZero YT. The file system as foundational primitive, and harness-engineering-before-RL.

### Cluster 4 — Observability, judging and shipped pipelines

13. [Building Better AI Agents: Artificial Intelligence Observability How To](https://www.youtube.com/watch?v=reISMhbZ2XE) `reISMhbZ2XE` — LangChain. Ship-early-and-often, and the three-to-five-repeats rule for probabilistic tasks.
14. [Complete Beginner's Course on AI Evaluations in 50 Minutes (2025) | Aman Khan](https://www.youtube.com/watch?v=TL527yTpxlk) `TL527yTpxlk` — Peter Yang. Spreadsheet-first grading, rubric-before-judge, golden-set sizing, and don't outsource your own eval.
15. [Complete Guide to building an AI Trading Bot with Claude for Beginners](https://www.youtube.com/watch?v=oQdkFPhqaGk) `oQdkFPhqaGk` — Ray Fu. The scheduled-automation pattern and its host-must-stay-on caveat.
16. [How to make an $8K/month Youtube Channel with Claude](https://www.youtube.com/watch?v=IBYHgDC5L0c) `IBYHgDC5L0c` — Ray Fu. The channel layer: RPM-driven niche choice, four script rules, voice-clone minimums, real tool pricing, posting cadence.

### Cluster 5 — Fine-tuning hardware, design and post

17. [The Honest Guide To Fine-Tuning Local AI In 2026](https://www.youtube.com/watch?v=v7qMjy_RxOs) `v7qMjy_RxOs` — Zen van Riel. The prompt→RAG→agentic→fine-tune decision order, CUDA-first hardware advice, avoid-Apple-Silicon, and the 2–3h-on-a-5090 figure.
18. [Build an LLM from Scratch 7: Instruction Finetuning](https://www.youtube.com/watch?v=4yNswvhPWCQ) `4yNswvhPWCQ` — Sebastian Raschka. The Apple-Silicon counterpoint, deduplication before training, and the CPU-versus-A100 time gap.
19. [Design with AI - Full Guide (Tools, Workflows, Design Systems)](https://www.youtube.com/watch?v=r-MPiu0E4W4) `r-MPiu0E4W4` — UI Collective. The ffmpeg frame-extraction post-production pattern, and token-cost reality for generative tooling.
20. [Learn Design Systems: Figma Variables, Components, AI, and More](https://www.youtube.com/watch?v=lsgVRCJ4fko) `lsgVRCJ4fko` — UI Collective. Background on skills-based AI consistency — the 28%-out-of-the-box versus 80–85%-with-structure gap.

## Gaps — what this corpus cannot tell you

These are the questions you asked, or will ask next, that twenty talks do not
answer. Do not let anyone — including a model — fill them in for you.

- **Google Colab specifically.** The only Colab mention is that the creator's repo
  holds "notebooks for Collab and Kaggle" [rEjasJZq1t8@12]. Nothing in the corpus
  states Colab's free GPU allocation, runtime or disk limits, whether a free Colab
  session offers two GPUs, or whether the H3 notebook has ever been run there. Your
  single most load-bearing question is the one the corpus answers least.
- **Single-GPU MiniMax H3.** Every figure assumes the transformer sharded across
  two T4s at 9.3 GB each. No video shows a single-card configuration, a CPU-offload
  path, or a quantised H3 variant.
- **StoryMem's speed and runtime.** The corpus gives parameter count,
  training-clip count and VRAM footprint, but no generation latency, no per-shot
  render time, and no end-to-end wall-clock for a one-minute story. You cannot
  budget a production schedule from it.
- **MiniMax H3 + StoryMem together.** StoryMem is documented as a LoRA for Wan
  2.2-based models [GM70MDyl94E@6]. No video states whether it composes with
  MiniMax H3, or with H3's four-step distillation LoRAs. Treat "H3 plus StoryMem"
  as an untested hypothesis.
- **Independent benchmark validation.** PDMD's leaderboard position, DMAD's 79% win
  rate and StoryMem's 28.7%/9.4% consistency gains are all vendor-published. The
  corpus contains no third-party replication of any of them.
- **Dollar costs for GPU rental.** No video prices a cloud GPU hour, an RTX 5090,
  or the per-clip cost of rendering at scale. The nearest cost signals in the
  corpus are for text models and agentic coding, not video.
- **Training your own video LoRA.** All LoRA hyperparameters, timings and
  dataset-size figures in this corpus come from text-model fine-tunes. No video
  shows building a video-model style LoRA, nor the data engineering that turns raw
  clips into training pairs.
- **What to do when an unattended render dies.** No video covers recovery from a
  mid-render session crash, a timed-out tunnel, or a notebook killed by the free
  tier's limits — beyond noting that Gradio links crash and that scheduled jobs
  need the host to stay on.
- **Automated video evaluation.** The eval material in this corpus is all
  text-output grading. Nothing covers an automated judge for video — no metric for
  character consistency, cut obedience, or face integrity that you could run
  instead of watching clips yourself.
- **Rights, licensing and platform policy.** No video addresses the licensing terms
  of these open weights, the rules for monetising AI-generated video on YouTube, or
  disclosure requirements for synthetic voices and faces.

---

Free-GPU Video Production · a field guide compiled from 20 source videos and 1,368
transcript chunks read in full, across 5 parallel extraction passes · compiled
2026-11-04. Every cite resolves to a numbered chunk in the corpus; claims without a
cite are editorial connective tissue. No web sources were used.
