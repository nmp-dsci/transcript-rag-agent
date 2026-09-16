---
title: LLM-as-a-Judge
topic: "LLM-as-a-judge: using language models to evaluate language model outputs"
compiled_at: 2026-09-14
revised_at: 2026-09-15
revision: 4
videos: 19
chunks: 703
sources:
  - video_id: N_DwZR--XCc
    title: "LLM-as-a-Judge / Autoraters • Guest Lecture at @Northeastern University • March 24, 2026"
  - video_id: EjFsjfOBZFE
    title: "UnlikelyAI Webinar: Is LLM-as-a-Judge Ready for Production in Financial Services?"
  - video_id: pnlT_xatpVQ
    title: "LLM-as-a-Judge 101"
  - video_id: qoPYlLg7_rY
    title: "LLM-as-a-Judge Evals with LangSmith"
  - video_id: a3SMraZWNNs
    title: "How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)"
  - video_id: TL527yTpxlk
    title: "Complete Beginner's Course on AI Evaluations in 50 Minutes (2025) | Aman Khan"
  - video_id: vBJF2sy1Pyw
    title: "The challenges in using LLM-as-a-Judge - Sourabh Agrawal | Vector Space Talk #013"
  - video_id: X4dEHRzBLmc
    title: "Judge the Judge: Building LLM Evaluators That Actually Work with GEPA — Mahmoud Mabrouk, Agenta AI"
  - video_id: kP_aaFnXLmY
    title: "3. Tutorial: How to create an LLM judge and align with human labels"
  - video_id: 3FcYdRQPMCo
    title: "LLM as a Judge Explained | Hands-On GenAI Evaluation with Real Code"
  - video_id: mX9HzIRdBpw
    title: "Evaluating Netflix Show Synopses with LLM-as-a-Judge"
  - video_id: kIvjiH8yJoU
    title: "From LLM-as-a-Judge To Human-in-the-Loop: Rethinking Evaluat... - Eric Pugh & Fernando Rejon Barrera"
  - video_id: spvXj9tnWAQ
    title: "Engineering Better Evals: Scalable LLM Evaluation Pipelines That Work — Dat Ngo, Aman Khan, Arize"
  - video_id: 8fNP4N46RRo
    title: "Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 8 - LLM Evaluation"
  - video_id: uQFLY8rQVYA
    title: "LLM Eval Methods | LLM-as-a-Judge | Reference Based Evals Vs Reference Free Evals | CampusX"
  - video_id: PjCwlX0XT8o
    title: "Chapter 6: Metrics, LLM Judges, and Statistical Confidence | Evals For AI Engineers"
  - video_id: HI5b0g_mELA
    title: "Amazon Bedrock RAG Evaluation ( LLM as a Judge ) | Step-by-Step Guide"
  - video_id: q2JrUKBMf0w
    title: "The Future of Evals: From LLM as a Judge to Agent as a Judge — Aparna Dhinakaran, Arize AI"
  - video_id: cRz0BWkuwHg
    title: "Key Metrics and Evaluation Methods for RAG"
---

# Your judge is an *instrument*, not an oracle

A field guide from the transcript corpus.

How to build an LLM that grades another LLM's output — and how to prove the grade means anything. Distilled from nineteen talks by Arize, Stanford CME295, Netflix, Agenta, Evidently, UnlikelyAI, OpenSearch, UpTrain, LangSmith practitioners, CampusX and others who have shipped judges, calibrated them against humans, and published the numbers where they failed.

19 source videos · 703 transcript chunks read in full · 5 parallel extraction passes · compiled 2026-09-14 · revision 4, 2026-09-15.

## The thesis: narrow it, calibrate it, characterise its error

Everyone reaches for LLM-as-a-judge for the same reason. Deterministic code evals work only when the success criteria are clear-cut and the data is well structured [pnlT_xatpVQ@2], and human annotation — hands-down the best signal you can collect — is either sparse or slow and expensive [pnlT_xatpVQ@3]. The judge is the middle ground: the thing you use when a task is too ambiguous for a regex and too costly for a person [uQFLY8rQVYA@22]. And it does buy real speed — with prompt tuning and the right model you get to roughly 90% of what you need very fast [EjFsjfOBZFE@11].

The failure is always the same too. The dev loop is bottlenecked on evaluation, so you swap the human for a judge and the loop gets fast — but if the judge doesn't correlate with human annotation you end up with useless signal and the loop "won't go anywhere" [X4dEHRzBLmc@1]. A first-pass judge rates every sampled output "good" [TL527yTpxlk@35]; a naive seed judge scores 61% and calls the trajectory compliant 98% of the time, which is just the majority class with extra steps [X4dEHRzBLmc@23]. The most common mistake in the field is assuming the judge works out of the box [a3SMraZWNNs@35].

> Before you create a judge, you have to be the judge. — Evidently AI [kP_aaFnXLmY@2]

The corpus converges on one reframing that makes everything else follow. An evaluator is a measurement instrument with a failure profile, not simply a scoring function [PjCwlX0XT8o@9]. Instruments get narrowed so they can be calibrated, calibrated against a reference you trust, and characterised — sensitivity, specificity, bias, repeatability — before anyone reads a number off them. Three moves, in that order.

### 1 · Narrow it

You do not need an overall judge for your product; you can create multiple small judges where each evaluates one specific thing [kP_aaFnXLmY@0]. Netflix builds a prompt per criterion because one overloaded prompt degrades quality [mX9HzIRdBpw@10]. Combining concerns in one judge makes a failure impossible to diagnose [PjCwlX0XT8o@27].

### 2 · Calibrate it

Compare judge labels to human labels on a golden set and compute exact-match accuracy — human alignment — then read the disagreements and fix the prompt [pnlT_xatpVQ@17]. It is a five-step loop: predictions, critiques, human labels, comparison, iterate [a3SMraZWNNs@20].

### 3 · Characterise it

Report TP/FP/TN/FN — true and false positives and negatives, defined below — with sensitivity, specificity and balanced accuracy instead of one headline number [PjCwlX0XT8o@57], test the four known biases [PjCwlX0XT8o@36], and re-run — one successful trace does not establish a reliable workflow [PjCwlX0XT8o@85].

The prize for doing this is compounding, not linear. Getting through four eval iterations in a month instead of two produces an exponentially better product [spvXj9tnWAQ@10]. The scale is real: Arize runs over 100 million evals a month, the average customer runs about 12 eval jobs, and the top teams run more than 3,800 distinct evaluators [q2JrUKBMf0w@0], with Duolingo running roughly 20 evals per trace [spvXj9tnWAQ@1].

### The four cells, in plain terms

TP, FP, TN and FN are the four cells of a confusion matrix: your human labels crosstabulated against the judge's calls, one cell per combination. Name the positive class before you compute anything — the class you pick decides how every prediction and every error is interpreted [PjCwlX0XT8o@54]. Take the worked example's positive class — "the human label says this response contains an unsupported claim", so a positive prediction means the judge flagged it:

- **TP — true positive.** The label says unsupported and the judge flags the response. A catch [PjCwlX0XT8o@55].
- **FP — false positive.** The response is supported and the judge flags it anyway. A false alarm; it costs human review capacity and can suppress a useful answer [PjCwlX0XT8o@55].
- **TN — true negative.** The response is supported and the judge passes it. Correctly left alone [PjCwlX0XT8o@55].
- **FN — false negative.** The response contains an unsupported claim and the judge passes it [PjCwlX0XT8o@55]. This is the dangerous cell: the ungrounded statement reaches the user, exactly where the evaluator was supposed to be the safety signal [PjCwlX0XT8o@60].

The rates are just ratios of those cells. **Sensitivity (TPR) = TP / (TP + FN)** is the proportion of genuinely unsupported responses the judge flags [PjCwlX0XT8o@56]. **Specificity (TNR) = TN / (TN + FP)** is the proportion of supported responses it leaves unflagged [PjCwlX0XT8o@56]. **Balanced accuracy** is the mean of the two, giving equal weight to the positive and negative classes [PjCwlX0XT8o@56]. That is why the headline number is not enough: raw agreement can be high when unsupported claims are rare even while TPR is poor [PjCwlX0XT8o@57]. Stage 07 puts numbers in the cells — of 200 held-out responses, TP 34, FN 6, FP 16, TN 144, which is where TPR 0.85, TNR 0.90 and balanced accuracy 0.875 come from [PjCwlX0XT8o@57].

## The build: seven stages from raw traces to a release gate

Every talk in the corpus that actually shipped a judge walks some version of this path. Skipping stages 1–3 is why judges fail; the teams who rushed metric design and annotation all report those as the hardest part in hindsight [X4dEHRzBLmc@11].

### 01 · Look at your data before you define anything

You cannot write criteria for failure modes you have not seen. This stage is deliberately informal.

- Observe real inputs and outputs and note failure modes — no formal annotation yet. "This is more of a vibe check." [pnlT_xatpVQ@6]
- But a vibe check alone is not enough; it doesn't tell you the whole story and you need rigorous methods and metrics on top of it [cRz0BWkuwHg@1].
- Sample real traffic — especially downvoted or rejected responses — rather than synthetic data or public benchmarks [vBJF2sy1Pyw@13]. Public benchmarks decay because they leak into training data [vBJF2sy1Pyw@13].
- Topic-cluster your logged queries before you decide what to evaluate. One client found around 80% of their questions fell into a narrow subset of topics [cRz0BWkuwHg@8].
- This never stops, and it is not beneath you: a principal engineer at Zapier spends three hours per day looking at traces and data [a3SMraZWNNs@34], and manual inspection has the highest value-to-prestige ratio of any eval activity [a3SMraZWNNs@36].
- Read transcripts, not just scores — that is how you catch a support agent telling a user their 45-minute-old order had passed a 60-minute cancellation window [TL527yTpxlk@21].

### 02 · Derive the metric from a failure, not from a template

Generic criteria produce generic results. The metric is the product decision; everything downstream is plumbing.

- Start with a failure, not with a favour evaluator — a general-purpose judge will happily report that your assistant improved while it is quietly rewarding length [PjCwlX0XT8o@0].
- Metrics must be specific, answerable and actionable — finer-grained beats general, a human must be able to label it from the context, and a low score must point at what to fix [pnlT_xatpVQ@9].
- Kill the catch-all. One metric called "quality" gives the judge nothing to focus on and gives you no direction when it drops [pnlT_xatpVQ@10]. A single omnibus score cannot show whether retrieval failed, extraction over-generated, or the evaluator just prefers verbosity [PjCwlX0XT8o@6].
- Kill the sprawl too. Ten overlapping metrics — fluency, style, tone, politeness — waste tokens and cause cognitive overload, and LLMs are already fluent out of the box [pnlT_xatpVQ@11].
- Derive criteria from the business use case, ideally with subject-matter experts, rather than reaching for "hallucination" [X4dEHRzBLmc@7]. Error analysis on real support transcripts surfaced four distinct error types — policy adherence, response style, information delivery, tool-call correctness — and each wants its own judge [X4dEHRzBLmc@8].
- Define the positive class explicitly. A claim is not positive merely because it is plausible; and decide your unit of analysis — claim-level and response-level are different measurements [PjCwlX0XT8o@1].
- Write the spec to a machine-readable file and have a repository test load it, confirm the positive class is declared, and reject an empty evidence contract [PjCwlX0XT8o@3].
- For a new RAG app, four or five metrics are the right starting set: response relevance, response completeness, context relevance, hallucination/groundedness, conversation satisfaction [vBJF2sy1Pyw@11].

### 03 · Be the judge yourself — build the golden set

Annotating does double duty: it proves your guidelines are unambiguous, and it gives you the reference the judge is measured against.

- Both purposes matter. If you cannot unambiguously label your own data on your own criteria, the LLM will struggle too; and the labels are what let you quantify how well the judge performs [pnlT_xatpVQ@8].
- Use binary or very few classes. A 1–10 scale is hard to apply consistently; binary labels make annotation far easier [kP_aaFnXLmY@2]. Netflix names the switch from a 1–4 Likert scale to pass/fail as a key change that lifted inter-annotator agreement [mX9HzIRdBpw@7].
- Record *why*, not just the label. Without the annotator's reasoning, a prompt-optimisation algorithm has no way to recover the underlying rule [X4dEHRzBLmc@10].
- Run two graders against the same rubric. Frequent disagreement means the rubric is ambiguous, not that the graders are bad [uQFLY8rQVYA@16].
- Treat calibration as iterative: annotate, measure disagreement, refine written guidelines with concrete examples, repeat. Netflix collected roughly 1,000 synopsis evaluations this way [mX9HzIRdBpw@5], landing a 600-item golden set sampled uniformly across content categories and genres for diversity rather than raw count [mX9HzIRdBpw@27].
- Sizing, honestly: around 10 examples is enough to decide whether to invest further [TL527yTpxlk@23], roughly 100+ before production depending on your regulatory bar [TL527yTpxlk@24], and 50–100 samples is a good starting test set [vBJF2sy1Pyw@12]. Do not let sizing stop you: "five is better than nothing" [TL527yTpxlk@29].
- Balance the classes so you can see both error directions — the code-review study used 27 bad and 23 good examples [kP_aaFnXLmY@8].
- Deliberately include hard cases — typos, long threads, conflicting fields, negation, near-duplicates, correct-but-less-verbose answers — because random sampling alone can produce an easy test [PjCwlX0XT8o@48].
- Split the labels *before* you touch the judge prompt. One worked allocation: 108 for judge design, 36 for development checks, 36 sealed for the final test [PjCwlX0XT8o@44].

### 04 · Write the judge prompt as guidelines for a human

Keep it minimal, keep it one decision, and make the output machine-checkable at the API level rather than by pleading in prose.

- The standard anatomy: role, task description, evaluation criteria, a scoring rubric per criterion, optional few-shot rubric examples, the candidate output, explicit instructions, and a structured JSON output spec [N_DwZR--XCc@30].
- Scope it to one decision and explicitly exclude helpfulness or writing quality if that is not what you are measuring [PjCwlX0XT8o@27], and use explicit labels like supported / unsupported / not_verifiable instead of vague scores [PjCwlX0XT8o@28].
- Ask only what a model can actually assess. "Is this answer medically correct?" is out of scope; "is this answer based on the context?" is in scope [kP_aaFnXLmY@1].
- Cap the judge's context at 4,000 tokens or less — even million-token models struggle to reason over the full length, so trim rather than dumping your knowledge base in [pnlT_xatpVQ@13]. The rule comes from a benchmark where reasoning performance dropped steeply after about 4,000 tokens regardless of advertised window [pnlT_xatpVQ@29].
- Bound the evidence. Give the judge the task instruction, the retrieved records and the target output in delimited fields — never hidden application metadata or the variant's name [PjCwlX0XT8o@29].
- Demand reasoning before the score. Rationale-first mirrors chain-of-thought and empirically improves the judgment [8fNP4N46RRo@18]; requiring a rationale makes judges better at the whole task, not just more explainable [N_DwZR--XCc@13]. Adding the single line "Always explain your reasoning" to an already-detailed prompt moved accuracy from 95% to 98% [kP_aaFnXLmY@19].
- Write general criteria, not your examples. Pasting labelled examples in makes the judge score those correctly and fail to generalise [kP_aaFnXLmY@17]. In the GEPA study, a seed prompt that did *not* contain the agent's policy text beat one that did, because pasting it trapped the optimiser in a local minimum [X4dEHRzBLmc@30].
- Tell it not to suggest improvements. Left unsaid, judges default to giving improvement feedback instead of a clean label, which breaks parsing [N_DwZR--XCc@35].
- Force structure at the API, not in prose. Prompting alone does not guarantee a parseable response — use constrained/guided decoding and structured output [8fNP4N46RRo@20], and stop spending tokens telling the judge to "respond in exactly one word" [pnlT_xatpVQ@14].
- Reject bad output loudly. A parser that rejects unknown labels, missing keys or malformed evidence arrays must trigger retry or review — never a silent pass [PjCwlX0XT8o@30]. Without an explicit structure constraint, a judge LLM can return a response of any shape [3FcYdRQPMCo@19].
- A little domain knowledge in the prompt goes a long way [kP_aaFnXLmY@3], and punish the things you don't want: the UPSC grading prompt explicitly instructs the judge not to reward verbosity or keyword stuffing [uQFLY8rQVYA@32].

### 05 · Choose the judge model — and don't let it grade itself

Judge quality is prompt and model together. The corpus disagrees on tier; it does not disagree on family.

- Never use the same model for the application and its judge. Intuitively it is "letting a student grade their own essay" [pnlT_xatpVQ@15]; formally it is self-enhancement bias, where a model prefers responses generated by itself [8fNP4N46RRo@26], and the remedy is a different model for generation and judging [8fNP4N46RRo@26].
- Model tier is not free. The same winning prompt on GPT-3.5 Turbo instead of GPT-4 mini collapsed to 48% recall and 72% accuracy [kP_aaFnXLmY@20], and in the GEPA study small or older models as judge or refiner were "a complete failure" on complex-policy work [X4dEHRzBLmc@28].
- Where the corpus splits: use the most powerful model you can afford [a3SMraZWNNs@18] versus don't use an expensive frontier model for routine evaluation because the cost adds up [vBJF2sy1Pyw@5]. The workable synthesis from the GEPA run: a bigger model for the refinement/reflection step, a smaller cheaper model for the judge itself, because the judge runs at online volume [X4dEHRzBLmc@31].
- Don't start with reasoning models. Their extra reasoning overhead is often overkill for eval when the metrics are well defined [pnlT_xatpVQ@16]. Bigger and proprietary models do tend to align better with humans [pnlT_xatpVQ@23].
- Consider the cheaper tier entirely. Encoder-only (BERT-style) evaluators run about 10× cheaper and one to two orders of magnitude faster than judge calls [spvXj9tnWAQ@5], and a trained classifier sits between rules and a judge — people skip it because of a lack of labelled data, not because it doesn't work [EjFsjfOBZFE@23].
- Run the judge cold. Low temperature (roughly 0.0–0.2) keeps scores reproducible across repeated runs [8fNP4N46RRo@31], and pin candidate models at 0.0 so the same question returns the same answer [3FcYdRQPMCo@11].
- Speed is a selection criterion too: with the same winning prompt, Claude performed nearly as well as GPT-4 mini as a judge but OpenAI ran noticeably faster [kP_aaFnXLmY@22].

### 06 · Meta-evaluate: measure the judge against your labels

This is the step that converts a plausible prompt into a trusted instrument. It is also the step teams skip.

- The simplest version: exact-match accuracy of judge labels against human labels, then read the judge's explanations on the disagreements and roll the fix into a new evaluator version [pnlT_xatpVQ@17].
- Track agreement as an explicit metric. A first-pass judge hit only 70% agreement with human labels on 10 examples [a3SMraZWNNs@25].
- Prefer classification metrics over a single correlation number: accuracy, precision and recall are intuitive and let you weight the error direction that matters [kP_aaFnXLmY@4].
- Watch for a judge that is precise but blind. The first code-review prompt had 100% precision and 36% recall — it caught almost nothing [kP_aaFnXLmY@15]. Rewriting it with explicit generalised criteria took it to 95% accuracy and 92% recall [kP_aaFnXLmY@18], from a starting point of 64% [kP_aaFnXLmY@12].
- Measure alignment per criterion, not in aggregate. In one walkthrough, product-knowledge labels matched humans 100% of the time while tone matched only once — same judge, wildly different reliability [TL527yTpxlk@39].
- Do not report raw agreement. Two raters guessing at random already agree 50% of the time [8fNP4N46RRo@7], so use chance-corrected metrics — Cohen's kappa for two raters, Fleiss' kappa or Krippendorff's alpha for more [8fNP4N46RRo@9].
- For numeric graders, mean absolute error against the human marks is the alignment metric: an MAE of 2.3 means the judge deviates from the human by ±2.3 marks on average, and you iterate the model, system prompt or rubric to drive it down [uQFLY8rQVYA@34].
- Close the gap with a meta-prompt: hand a powerful model the inputs, outputs, human critiques, the judge's current outcomes and the current judge prompt, and ask it to rewrite the prompt to maximise agreement [a3SMraZWNNs@26].
- Automated optimisation works but is not magic. GEPA lifted validation accuracy from 69% to 74% and dropped the judge's bias toward "compliant" from 98% to 64% [X4dEHRzBLmc@26] — still far from the ~95% you would want to call it human-aligned [X4dEHRzBLmc@27], at roughly $200–300 of tokens for a small-scale run [X4dEHRzBLmc@31]. And a prompt optimiser that makes the prompt longer and more rule-laden is not automatically better — check its output by hand [TL527yTpxlk@27].
- Always test a fancy method from a paper against the simple baseline on your own labels [kP_aaFnXLmY@3].
- Where this can land: Netflix reached an 85% agreement rate with expert creative writers on an extremely subjective task [mX9HzIRdBpw@2].

### 07 · Gate on it — with a threshold you chose from costs

A score becomes a release signal only when the instrument has a documented error profile and a maintenance policy.

- Report the confusion matrix. A worked example on 200 held-out responses (40 unsupported, 160 supported) where the judge catches 34 and wrongly flags 16 gives TPR 0.85, TNR 0.90, balanced accuracy 0.875 [PjCwlX0XT8o@57].
- Pick the threshold from documented costs of false negatives versus false positives, not from observed accuracy — for a safety floor, first retain only thresholds with TPR ≥ 0.90, then minimise cost among those [PjCwlX0XT8o@59].
- Never ship the judge's observed positive rate as the true rate. Correct it with sensitivity and specificity: a 78% judge positive rate with sensitivity 0.90 and specificity 0.80 corrects to roughly 83% [PjCwlX0XT8o@66].
- And know when correction is impossible: at sensitivity and specificity near 0.50 the correction denominator collapses, and bootstrap resampling cannot repair an uninformative instrument [PjCwlX0XT8o@70].
- Compare variants with paired, case-level differences and a paired bootstrap interval — resample the differences between paired cases, not A and B independently [PjCwlX0XT8o@80].
- Predeclare the primary metric, the evaluation set and the minimum worthwhile improvement before you run the comparison; keep the full ledger including discarded trials [PjCwlX0XT8o@83].
- Run each case multiple times — five is a starting protocol — to separate difficulty (consistent failure) from instability (alternating pass/fail on identical runs) [PjCwlX0XT8o@85]. With an 0.8 single-run success rate over five attempts, "at least one passes" is 0.99968 while "all five pass" is 0.32768 — a retryable product cares about the first number, an unattended workflow about the second [PjCwlX0XT8o@87].
- Repeat runs are also a discovery tool: run the same eval 10, 20, 50 times over a subset to find the examples the judge is unstable on, and turn those into clearer criteria [pnlT_xatpVQ@26]. Raising temperature deliberately amplifies the signal while you probe [pnlT_xatpVQ@26].
- Ship the evaluator like production code — package boundary, tests, version, cost record, validation results, maintenance policy. Without those artefacts, a score is an observation, not a dependable release signal [PjCwlX0XT8o@91].
- Define maintenance triggers in advance — excessive disagreement with human review, a new high-risk failure mode, changed provider behaviour, a score-distribution shift, rising cost, repeated evaluator errors — and never silently retune thresholds to preserve a green dashboard [PjCwlX0XT8o@97].
- Then roll out behind a real experiment: dogfood internally and launch as an A/B test on a small slice (~10%, or ~1% at large companies) rather than straight to full production [TL527yTpxlk@43]. Decide in advance how you break the tie when the eval says good and the business metric goes down [TL527yTpxlk@45].
- Run two loops, not one: one improving the application from eval signal, a second annotating eval failures to improve the eval prompt template itself [spvXj9tnWAQ@9].

## Scoring: what you ask the judge to output

The output format is not cosmetic. It determines how consistently a human can produce ground truth, how consistently the model can reproduce it, and whether a disagreement is interpretable.

### Binary over Likert

Move away from 1–5 scores and percentages toward binary pass/fail; even binary is hard to calibrate, and two human annotators often disagree on a 1–5 score anyway [X4dEHRzBLmc@9]. The academic consensus agrees: binary is easier for both the judge and the human raters to apply consistently [8fNP4N46RRo@29].

### Word labels over numbers

Numeric ratings don't work well for LLM judges; use a small set of discrete word labels — yes/no, positive/negative/neutral [pnlT_xatpVQ@9]. If your evaluation can be answered with yes/no, that is a sign your specificity is right.

### Pairwise beats scoring in isolation

Judges scoring a single answer tend to be too optimistic and agree with whatever they're shown [kIvjiH8yJoU@6], so ask "left or right — which one is better?" like an eye exam [kIvjiH8yJoU@8]. Pairwise gives the model a baseline and a clearer benchmark of quality [cRz0BWkuwHg@5]. Pairwise evals are a sports tournament; direct scoring is Olympic gymnastics [pnlT_xatpVQ@4]. What it buys in discrimination it gives back in order sensitivity: the judge can pick response A purely because A was mentioned first [8fNP4N46RRo@23]. Control for that before you believe a win rate — next section.

### Pairwise's own bias: position

Position bias is the tax on the format: judges always favour one or the other, the first or the second, depending on the ordering [N_DwZR--XCc@46]. Four controls, cheapest first. **Randomise** which response appears first, and allow ties so the judge is never forced to invent a winner between two equally good answers [cRz0BWkuwHg@6]. **Run both orderings** — ask A vs B, then B vs A, and take the majority across the two runs; if the two disagree, you do not have a verdict [8fNP4N46RRo@24]. **Keep the swap paired**: hold rubric, evidence, decoding settings and whitespace fixed so the only thing that changed is position — a new seed or altered context makes the diagnosis ambiguous, and a flipped verdict then means neither answer has demonstrated superiority [PjCwlX0XT8o@37]. **Predeclare the aggregation rule**: a production comparison either aggregates both orders or excludes and separately reports unresolved reversals under a documented rule [PjCwlX0XT8o@37]. Anonymise candidates as response_A / response_B while you are at it, and randomise their order during the experiment [PjCwlX0XT8o@32]. If you train a judge rather than prompt one, the training-time mitigation is shuffling candidate positions intentionally so the model cannot overfit to "first" [N_DwZR--XCc@46]. The flip rate belongs on the model card as a measured number, not an assumption [PjCwlX0XT8o@36].

### ELO when you have more than two variants

Treat each system variant as a chess player and apply ELO to the pairwise outcomes for a global ranking [kIvjiH8yJoU@8]. The open-sourced ragelo toolkit scores retrieved-hit relevance, feeds that plus both answers to a pairwise judge, then runs ELO to rank agents [kIvjiH8yJoU@9]. Do not average per-query ELO — that is not what ELO means [kIvjiH8yJoU@24].

### Pointwise, pairwise, listwise

Judge design inherits the pointwise/pairwise/listwise learning-to-rank paradigms from IR [N_DwZR--XCc@15]. Pointwise is cheap and scalable but its absolute scores can be inconsistent or miscalibrated [N_DwZR--XCc@21]; pairwise correlates better with human judgment at higher comparison cost [N_DwZR--XCc@26]; listwise ranks whole lists with NDCG but is the most expensive and hardest to scale [N_DwZR--XCc@28].

### Score atomic facts, not whole passages

Decompose the output into a list of atomic facts with one LLM call, then check each [8fNP4N46RRo@35], weighting by importance if you need partial credit — a 4-fact passage with 2 correct facts scored 6 out of 10 [8fNP4N46RRo@37]. The same move fixes relevance: break the response into facts and score each fact's relevance, then aggregate into a ratio, rather than asking for a subjective holistic label [vBJF2sy1Pyw@7].

### Never collapse to one number

Averaging metrics into a single weighted quality score loses the diagnostic signal about where the app is failing; plot several metrics together instead [pnlT_xatpVQ@22]. Keep dimensions separate and independently owned; a composite decision may be useful but must be built visibly from component results with an explicit policy [PjCwlX0XT8o@7]. A scorer should report check-level evidence — name, expected, observed, pass/fail — rather than an aggregate like 0.75 [PjCwlX0XT8o@20].

### Longer reasoning, then a short summary

Netflix asked for a reasoning trace first because the reasoning becomes context for the score [mX9HzIRdBpw@11]; pushing the trace from a couple of sentences to four or five paragraphs improved judge accuracy by about 7% absolute, with diminishing returns at the longest setting [mX9HzIRdBpw@13]. Then summarise the long trace into ~2 readable sentences — the summarisation step itself improved accuracy further [mX9HzIRdBpw@14].

### Consensus and panels

Generating five judge outputs and taking the rounded average gave a noticeable accuracy boost, and paired best with the longer rationale that introduces the variance worth aggregating over [mX9HzIRdBpw@15]. A panel of different judge models aggregated by vote reduces individual-model bias, trading compute for reliability [N_DwZR--XCc@44].

### Confidence, when the API gives it to you

For autoregressive judges the log-probability of the returned label token is a solid confidence proxy; small encoder-only models hand you a classification probability directly [spvXj9tnWAQ@22]. And remember a useful metric need not be a number at all — a label or explanation string surfaced to a human is a metric [kIvjiH8yJoU@16].

## Delete on sight

Thirteen things the corpus tells you to stop doing, and what to do instead. Every row is someone's published mistake.

| Delete on sight | Replace with |
| --- | --- |
| `You are an AI evaluator` | `You are an expert evaluator` — the first primes the model to behave the way an AI would [pnlT_xatpVQ@14] |
| A single `quality` metric scored 1–10 | A named failure mode with a binary label the judge can actually apply [pnlT_xatpVQ@10] |
| The same model generating and judging | A judge from a different model family, chosen before you look at the scores [pnlT_xatpVQ@15] |
| Your labelled examples pasted into the judge prompt | Generalised written criteria that describe the rule behind the examples [kP_aaFnXLmY@17] |
| The agent's full policy pasted into the seed judge prompt | A policy-free seed the optimiser can still move away from [X4dEHRzBLmc@30] |
| Ten overlapping metrics (fluency, style, tone, politeness…) | A handful tied to business-critical failure modes [pnlT_xatpVQ@11] |
| `Respond in exactly one word` | Structured outputs / constrained decoding enforced by the provider [8fNP4N46RRo@21] |
| Dumping the knowledge base into the judge's context | ≤ 4,000 tokens of deliberately trimmed context [pnlT_xatpVQ@13] |
| One headline accuracy number | TPR, TNR and balanced accuracy off a confusion matrix [PjCwlX0XT8o@57] |
| Raw human agreement rate | Cohen's kappa, Fleiss' kappa or Krippendorff's alpha [8fNP4N46RRo@9] |
| An LLM judge for PII detection | A rule-based matcher — cheap, fast, free [EjFsjfOBZFE@22] |
| Averaging per-query ELO into an aggregate | The ELO algorithm, run properly over the pairwise outcomes [kIvjiH8yJoU@24] |
| Quietly retuning the threshold when the dashboard goes red | A predeclared maintenance trigger and a new judge version [PjCwlX0XT8o@97] |

One more that is easy to miss: out-of-the-box eval templates produce generic, low-value results, so customise criteria heavily to your application [spvXj9tnWAQ@12].

## Validate the instrument

A judge that agrees with your labels can still be measuring the wrong thing. Two suites belong in every judge's model card: bias diagnostics and statistical reliability.

### The bias suite

Bias here is not only an incorrect label — it is a systematic tendency to reward an irrelevant property. Four diagnostics belong on the card: position bias, verbosity bias, self-preference, and task-specific bias [PjCwlX0XT8o@36].

#### Position bias → the swap test

In pairwise judging the model can pick response A just because it was first [8fNP4N46RRo@23]. Ask both orderings and take the majority across the two runs [8fNP4N46RRo@24], or simply randomise which appears first [cRz0BWkuwHg@6]. Make it a paired operation, not a fresh sample: if the verdict changes when only the positions change, that is evidence of position sensitivity [PjCwlX0XT8o@37]. At training time, shuffling candidate order intentionally is the mitigation [N_DwZR--XCc@46].

#### Verbosity bias → matched length slices

Judges prefer longer, more elaborate responses even when they aren't more correct [8fNP4N46RRo@25]. Keep compared responses similar in length so the comparison is fair [cRz0BWkuwHg@7], and diagnose properly with matched slices — concise vs verbose carrying equivalent facts — reporting preference within each slice rather than the aggregate win rate [PjCwlX0XT8o@38]. This is not theoretical: in one dry run the judge favoured longer answers, letting verbose responses with unsupported claims beat short correct ones [PjCwlX0XT8o@91].

#### Self-preference → cross-family checks

A judge gives elevated scores to its own family's style and instruction conventions [8fNP4N46RRo@26]. Run the same fixture suite through a judge from the generator's family and one from a different family, and against non-LLM checks — agreement across families is useful evidence of stability but not proof of correctness [PjCwlX0XT8o@41].

#### Task-specific bias → strip the surface

Judges reward features that correlate with quality but aren't quality: polished greetings, disclaimers, a particular formatting style [PjCwlX0XT8o@36]. A 2024-era study of four leading judge models found none reliable across the board, including judges that flipped their verdict on formatting changes that left meaning unchanged [EjFsjfOBZFE@9].

#### Leakage → anonymise the candidates

Present candidates as anonymous response_A / response_B, randomise order, and evaluate each independently against the same context. Hidden identity removes one obvious source of preference leakage but does not eliminate the other three biases [PjCwlX0XT8o@32]. Keep every test item, paraphrase and label rationale out of the judge's few-shot prompt and regression fixtures [PjCwlX0XT8o@48].

#### Vague rubrics → reward hacking

A judge's reliability depends entirely on how objectively its criteria and rubrics are specified; ambiguity there is a direct path to reward hacking [N_DwZR--XCc@48]. And the rationale is not a defence — the judge's rationale is also model output and must be validated like the verdict [PjCwlX0XT8o@29], coming out of the same tangled generation path as the verdict rather than a trace of real reasoning [EjFsjfOBZFE@12].

### The reliability suite

Bias tests tell you what the judge is measuring. These tell you how much to believe the number it produced.

#### Freeze a test split first

Split labelled data before touching the judge prompt, and treat the final partition as a sealed decision surface opened only after the configuration is frozen — otherwise it is no longer a clean estimate on unseen cases [PjCwlX0XT8o@44].

#### Correct for the judge's own error

The observed positive rate is not the true rate. Correct it with sensitivity and specificity — P = (Q + C − 1) / (S + C − 1) — and know that when both approach 0.50 the correction carries no information [PjCwlX0XT8o@66].

#### Pair your comparisons

Pairing removes baseline task-difficulty noise that independent samples leave in; predeclare a minimum improvement (e.g. 0.03) before calling a variant a release candidate [PjCwlX0XT8o@80].

#### Measure the distribution, not one run

Don't check whether a single run gave the answer you wanted — run it multiple times and dig into the distribution of outcomes [EjFsjfOBZFE@14]. Re-running the same evaluation can give different results from non-determinism alone, observed frequently on 4o-mini-class models [kIvjiH8yJoU@26].

#### Mutation-test the scorer

Remove a required key, change a number's type, empty the recommendations list, add an unallowed amenity, raise the price above the limit, change the recipient — each mutation should produce the intended failure and matching evidence, not a false pass [PjCwlX0XT8o@24].

#### Keep a human in the loop, forever

LLM evals scale human judgment; they do not replace it, so some human-in-the-loop is always required for the best results [pnlT_xatpVQ@16]. Even where a judge is not strictly required, periodically collect human ratings alongside judge scores and run a correlation analysis so you don't over-optimise against a flawed proxy [8fNP4N46RRo@31]. Human judgment remains the gold standard for fluency, naturalness and real-world usefulness [cRz0BWkuwHg@9].

## In production: cost, guardrails and agents

Offline calibration is the easy half. The hard half is what the judge costs, where it sits in the request path, which way it fails, and whether a fixed rubric can describe an agent at all.

Those four questions are positional — they are answered by *where* the judge sits, not by how good its prompt is. Every placement in the cards below is drawn in "Draw the system: where the judge attaches": the attachment points on an agent (input, retrieval, tool calls, output, trajectory), the three request-path topologies and what each one does to p95, the instrument ladder, and the calibration loop that feeds them.

### Price it at production volume

Judge checks run in seconds, not milliseconds, and you pay per check forever — stacking multiple judges scales expensively fast [EjFsjfOBZFE@17]. Price at actual production call volume, not demo volume: a chatbot check runs on every message and may need several checks per message [EjFsjfOBZFE@28]. Even a toy harness makes three paid calls per test question [3FcYdRQPMCo@27].

### Budget the tail, not the mean

When a check sits in the user's critical path, consistency matters as much as average latency — measure the 95th percentile, because one person waiting two minutes ruins the experience even if most get sub-five-second responses [EjFsjfOBZFE@29].

### Evals are not guardrails

Guardrails act at inference time to prevent problems; evals are a check after the fact, good for offline evaluation and continuous monitoring — an eval can detect a prompt injection but not prevent it [pnlT_xatpVQ@31]. Practically, only safety and jailbreak checks are cheap enough to run in the moment; relevance, hallucination and completeness run post-hoc [vBJF2sy1Pyw@3]. Fire the guardrail on the query in parallel with generation rather than checking the response afterwards, so you don't add a latency layer for the 95% of users who aren't misusing the product [vBJF2sy1Pyw@19].

### Decide which way you fail

Nondeterministic guardrails tend to fail toward leniency — letting a bad output through rather than holding it back, which is the more dangerous direction [EjFsjfOBZFE@17]. Overblocking costs an annoyed customer and a contact-centre call; a breach reaching a regulator costs vastly more. Decide explicitly: do you fail open or fail closed, and does that argue for two independent checks? [EjFsjfOBZFE@27] A judge is best used as one voice among several rather than the single decision-maker [EjFsjfOBZFE@18].

### Decompose the hard question

Instead of asking one frontier judge "is this message a breach", decompose the policy into small narrow checks a model answers reliably — "did you name a specific product?" — which lets you use smaller, cheaper models and shrinks latency [EjFsjfOBZFE@35]. Push as much of the criterion as possible into deterministic checks; they are fast, reproducible and easier to debug than a global quality score [PjCwlX0XT8o@12]. Hard requirements need explicit code, because without it a judge will reward longer and more persuasive answers that still violate the task [PjCwlX0XT8o@18].

### Pick the least interpretive instrument

Choose the least interpretive tool that can answer the criterion — executable check, reference check, classifier, judge, human — and add others only when the criterion has independent parts [PjCwlX0XT8o@9]. The compact rule: output shape → executable check; exact identity or database state → reference check; known policy categories → classifier; borderline meaning, relevance or evidence support → a judge with reference checks and sampled human review; high-consequence ambiguity → humans plus deterministic checks [PjCwlX0XT8o@12]. The toolbox is bigger than the judge: encoder-only classifiers, human feedback, golden datasets and code heuristics all belong in it [spvXj9tnWAQ@5].

### Make it survivable in an audit

Audit fitness rests on three things: did the guardrail explain *why* it failed, is that explanation stable over time, and is the decision stored and retrievable — could you defend a verdict pulled from eight months ago? [EjFsjfOBZFE@30] You need to name the exact rule and condition, and store the verbatim customer message rather than a summary [EjFsjfOBZFE@36]. Keep the system model-agnostic so it survives deprecations — heavily prompt-tuned systems are the most fragile to a model swap [EjFsjfOBZFE@39]. And prompt tuning has a ceiling: it moves metrics but cannot fix the black-box problem, and there will always be cases outside the expected boundaries [EjFsjfOBZFE@16].

### Route failures to a root-cause battery

Take the 10–20% of production cases that fail a high-level quality check and run a battery of 5–10 targeted checks on just those, to tell retrieval failure from citation failure from hallucination from a bad embedding model [vBJF2sy1Pyw@6]. Filter all zero-scoring failures and cluster their common topics to turn a pile of failures into a diagnosis [vBJF2sy1Pyw@25]. Instrument every stage — query rewrite, sub-query generation, retrieval, reranking — not just the final response [vBJF2sy1Pyw@16].

### Skip evaluating what's already broken

If an agent's control-flow step fails, don't evaluate everything downstream of it — those results are likely invalid anyway, and skipping saves compute [spvXj9tnWAQ@12]. And when a guardrail fires, the fix is almost always in the orchestration and prompt, not more guardrail tuning [spvXj9tnWAQ@19].

### Judging agents and trajectories

What is being evaluated has changed — from single prompt-answering, to tool calls and deep research, to long-horizon loops with sub-agents on real-world data [q2JrUKBMf0w@1]. One camp says an LLM judge handles it: feed it what happened and what was expected, with or without a reference path [spvXj9tnWAQ@16]. The other says a fixed rubric with fixed scores fundamentally cannot cover adaptive behaviour [q2JrUKBMf0w@3] and proposes agent-as-judge — a long-running agent that reads traces and discovers patterns a rubric would never catch [q2JrUKBMf0w@4], such as an agent calling the same tool repeatedly in a loop [q2JrUKBMf0w@4], and can then open a PR with a fix [q2JrUKBMf0w@4].

### The decomposition that actually won

Netflix's largest single gain came from the same move in miniature: split factuality into narrow sub-types (plot, metadata, talent, awards), run a targeted judge-agent per sub-type, and take the minimum score across them — over a 10% boost, the biggest of all their interventions [mX9HzIRdBpw@17]. And they validated the whole thing against reality: causal inference showed synopses with higher judge precision or clarity scores had significantly higher take fractions [mX9HzIRdBpw@19] — a signal available before a show ever launches, unlike behavioural metrics which are inherently retroactive [mX9HzIRdBpw@21].

### What happens with no evaluation at all

A car dealership chatbot with no real-time evaluation agreed that a competitor's car was better and then agreed to sell a car for a dollar [vBJF2sy1Pyw@18]. Jailbreak attempts carry telltale signs — unusually long prompts, bursts of repeated traffic, lower sentiment and coherence than genuine queries [vBJF2sy1Pyw@21]. Human review handles genuine ambiguity well, but it doesn't scale — design for what fraction of decisions escalate, because that governs cost and response time [EjFsjfOBZFE@19].

## Draw the system: where the judge attaches

A judge is not a stage in your app; it is a set of probes clipped onto one. Where you clip them decides cost, latency and what you can diagnose. Teams that do this well instrument every part of the pipeline — query rewrite, sub-query generation, retrieval, reranking — not just the final response [vBJF2sy1Pyw@16], which is how Duolingo ends up running roughly 20 evals per trace [spvXj9tnWAQ@1]. Five drawings: the attachment points, the request-path topologies, the instrument ladder, the calibration loop, and the four cells.

**Figure 1 — an agent and its seven judge attachment points.** Solid boxes are your application; dashed (`+ +`) boxes are judges. Nothing in the corpus draws this diagram — it is assembled from the placements each talk describes, cited one by one below.

```
                      online . in the request path
        + + + + + + + + + +            + + + + + + + + + + + + +
        + J1 . input check +            + J4 . output guardrail  +
        + jailbreak/PII/   +            + inline . blocking .    +
        + topic            +            + fail open or closed    +
        + + + + +|+ + + + +             + + + + + +|+ + + + + + +
                 v                                 v
  +----------+   +--------------------+   +----------------+
  |User query|-->|     Agent loop     |-->| Final response |--> user
  +----------+   |  plan . act . obs  |   +-------+--------+
                 +--+-------+------+--+           | (spans)
           +--------+       |      +-------+      v
           v                v              v   +------------+
    +-------------+  +------------+  +--------+|Trace store |
    | Retrieval   |  | Tool calls |  |Sub-    ||every span  |
    | rewrite .   |  | / APIs     |  |agents  |+-----+------+
    | search.rank |  +-----+------+  +--------+      |
    +------+------+        |                         |
           v               v                         |
    + + + + + + +   + + + + + + + +                  |
    + J2 context+   + J3 tool-call+                  |
    + relevance +   + check       +                  |
    + + + + + + +   + + + + + + + +                  |
                                                     |
  offline . async . sampled traces <-----------------+
        |                    |                    |
        v                    v                    v
  + + + + + + + +   + + + + + + + + +   + + + + + + + + + +
  + J5 . response+   + J6 . trajectory+   + J7 . agent-as- +
  + quality      +   + judge          +   + judge          +
  + + + + + + + +   + + + + + + + + +   + + + + + + + + + +
```

- **J1 · input check** — fire the jailbreak/safety check on the query *in parallel* with generation, not after the response, so you don't add a latency layer for the 95% of users who aren't misusing the product [vBJF2sy1Pyw@19]. Jailbreak traffic has telltale shape — unusually long prompts, repeated bursts, lower sentiment and coherence [vBJF2sy1Pyw@21].
- **J2 · context relevance** — clip a check onto retrieval itself, because context relevance is one of the four or five metrics any RAG app should start with [vBJF2sy1Pyw@11], and the platform should plug into every part of the pipeline rather than only the final response [vBJF2sy1Pyw@16].
- **J3 · tool-call check** — tool-call correctness is one of the four error types error analysis surfaced on real support transcripts, and it wants its own judge [X4dEHRzBLmc@8]. Attribute the failure correctly: when an agent fails to pick an available tool, that is a recall error in the router, not a fault of the final LLM call [8fNP4N46RRo@40].
- **J4 · output guardrail** — the only judge in the blocking path, because blocking bad outputs in real time is one of the three production uses of evals [a3SMraZWNNs@9]. Being here forces the two questions the other placements dodge: fail open or fail closed, and do you need a second independent check [EjFsjfOBZFE@27] — nondeterministic guardrails drift toward leniency on their own [EjFsjfOBZFE@17].
- **J5 · response quality, post-hoc** — relevance, hallucination and completeness are too expensive to run before showing the response, so they run after the fact on stored traces [vBJF2sy1Pyw@3]; that is the difference between an eval and a guardrail — an eval can detect a prompt injection but not prevent it [pnlT_xatpVQ@31].
- **J6 · trajectory judge** — feed the judge what the agent did and what was expected, with or without a reference path [spvXj9tnWAQ@16], and make it conditional: if a control-flow step failed, skip everything downstream because those results are invalid anyway [spvXj9tnWAQ@12].
- **J7 · agent-as-judge** — a long-running agent that reads traces and discovers patterns a fixed rubric would never encode [q2JrUKBMf0w@4], such as the same tool being called repeatedly in a loop [q2JrUKBMf0w@4]. This box exists because the target moved: from single prompt-answering to tool calls to long-horizon loops with sub-agents on real-world data [q2JrUKBMf0w@1].
- **The trace store is the load-bearing box.** Everything below the bus is optional until spans are recorded; and for audit, what you store must be the verbatim message, not a summary [EjFsjfOBZFE@36].

**Figure 2 — three request-path topologies.**

```
A . inline      [input check] -> [generate] -> [output check] -> user sees answer
  blocking      serial: every check lands inside the number the user feels (p95)

B . parallel                 + input check +
  non-blocking  [query] --+--                --+--> user sees answer
                          +--   [generate]   --+
                the check overlaps generation: no extra layer for the 95%
                who aren't misusing it

C . post-hoc    [query] -> [generate] -> user sees answer
  async                        :
                               v
                 + sampled traces -> 5-10 targeted checks +
                 the 10-20% that fail a high-level check get the battery
```

Judge checks run in seconds, not milliseconds, and you pay per check forever [EjFsjfOBZFE@17], so topology is a cost decision before it is a quality one. In lane A the number to watch is p95, not the mean — one person waiting two minutes ruins the experience even if most get sub-five-second responses [EjFsjfOBZFE@29]. Lane B is the recommended shape for safety checks [vBJF2sy1Pyw@19]; lane C is where relevance, hallucination and completeness belong [vBJF2sy1Pyw@3], feeding the root-cause battery on the 10–20% of cases that fail a high-level check [vBJF2sy1Pyw@6].

**Figure 3 — the instrument ladder, and decomposition inside the judge column.**

```
                    [ one criterion, one decision ]
         +------------+------------+------------+------------+
         v            v            v            v            v
  [executable  ] [reference  ] [classifier ] [LLM judge  ] [human review]
  [check       ] [check      ] [           ] [ (judge)   ] [            ]
   output shape   identity/DB   known cats    borderline    high-conseq.

  <-- cheap . deterministic . debuggable ... interpretive . needs calibration -->

  inside the judge column, decompose again:
     [plot] [metadata] [talent] [awards]  ->  min() across sub-scores
```

Read the top row left to right and stop at the first instrument that can answer the criterion: an evaluator is a measurement instrument with a failure profile, so pick the least interpretive one that works [PjCwlX0XT8o@9], and push as much of the criterion as possible into deterministic checks [PjCwlX0XT8o@12]. When the judge column is unavoidable, decompose the hard question into narrow checks a smaller model answers reliably [EjFsjfOBZFE@35] — the bottom row is Netflix's factuality split, whose minimum-across-sub-types aggregation was their single biggest gain, over 10% [mX9HzIRdBpw@17].

**Figure 4 — the two calibration loops.**

```
 [Production traces] --> [Sample + hard cases] --> [Human labels . frozen split]
        ^                                                      |
        |                                                      v
        |                                             [ Judge prompt vN ] <--+
        |                                                      |             |
        |                                                      v             |
        |                                             [ Judge labels ]       |
        |                                               (design split)      |
        |                                                      |             |
        |             [Confusion matrix + kappa] <-------------+             |
        |                        |                                           |
        |                        v                                           |
        |          [Disagreements -> meta-prompt] --- loop 2 ----------------+
        |                        |
        |                        v
        +------ loop 1 ---- [Frozen test -> release gate]
```

Two loops, not one: one improving the application from eval signal, a second annotating eval failures to improve the eval prompt itself [spvXj9tnWAQ@9]. Sample from real traffic, not benchmarks [vBJF2sy1Pyw@13], and plant hard cases deliberately because random sampling alone produces an easy test [PjCwlX0XT8o@48]. The comparison box is exact-match human alignment [pnlT_xatpVQ@17] reported chance-corrected [8fNP4N46RRo@9] and per criterion, since tone and product knowledge can differ wildly on the same judge [TL527yTpxlk@39]. The rewrite arrow is the meta-prompt [a3SMraZWNNs@26]; the split is frozen before the judge prompt is touched [PjCwlX0XT8o@44] and the gate predeclares its minimum worthwhile improvement [PjCwlX0XT8o@83].

**Figure 5 — the four cells with the worked numbers.**

```
 positive class = the human label says "contains an unsupported claim"

                    human: unsupported  |  human: supported
  judge flags       TP 34               |  FP 16                TPR = 34/40   = 0.85
                    a catch             |  false alarm, review  TNR = 144/160 = 0.90
  judge passes      FN 6  << danger     |  TN 144               balanced acc  = 0.875
                    reaches the user    |  correctly left alone n = 200 held out
```

Name the positive class before you compute anything — the class you pick decides how every error is read [PjCwlX0XT8o@54]. These are the worked numbers: TP 34, FN 6, FP 16 out of 200 [PjCwlX0XT8o@57], giving TPR 0.85, TNR 0.90 and balanced accuracy 0.875 [PjCwlX0XT8o@57]. FN is the marked cell because a miss puts an ungrounded statement into a user-facing answer, exactly where the evaluator was supposed to be the safety signal [PjCwlX0XT8o@60].

One honest caveat on all five. No talk in this corpus presents a reference architecture — every box above is a placement someone described in prose, drawn here and cited to the sentence that justifies it. Where the corpus is silent it stays silent: nothing here tells you how many judge calls a single agent turn should cost, how to combine J1–J7 into one release decision, or what the agent's own memory and planner should look like. Those are in the gaps.

## Copy these

Four artefacts. The first is the judge; the second closes the gap with your humans; the third and fourth are what turn a score into a release signal.

### Narrow judge prompt — one decision, explicit labels, bounded evidence

```
ROLE
You are an expert evaluator of real-estate assistant emails, applying a
written rubric. (Do not say "AI evaluator" — it primes AI-like behaviour.)

THE ONE DECISION
Does this email contain a material claim about a listing that is NOT
entailed by the retrieved listing records below?
You are NOT assessing helpfulness, tone, or writing quality. Other judges
own those. Do not suggest improvements.

EVIDENCE CONTRACT — nothing outside these fields is evidence
<task_instruction> ... </task_instruction>
<retrieved_records> ... </retrieved_records>
<response>          ... </response>
A claim is not supported merely because it is plausible or conventional.
Total context here must stay under ~4,000 tokens. Trim, don't dump.

LABELS — use these exact strings, nothing else
  supported | unsupported | not_verifiable
"unsupported" means the claim lacks support in the evidence, NOT that it
is known to be false. "not_verifiable" means no permitted field addresses it.

PROCEDURE
1. Write your reasoning first, then the labels.
2. List each material claim in the response.
3. For each claim, name the exact field in <retrieved_records> that
   entails it. If no field entails it, say so.
4. Label each claim.
5. Derive the response label MECHANICALLY:
     response_label = unsupported  if ANY claim is unsupported
   Do not form an independent holistic opinion of the response.

OUTPUT
JSON only, enforced by the provider's structured-output API — not by this
instruction:
{"reasoning": str,
 "claims": [{"text": str, "label": str, "evidence_field": str}],
 "response_label": str}
```

Built from the standard judge-prompt anatomy [N_DwZR--XCc@30], the one-decision rule [PjCwlX0XT8o@27], the explicit-label vocabulary [PjCwlX0XT8o@28], the 4K context cap [pnlT_xatpVQ@13], rationale-before-score [8fNP4N46RRo@18], and structured output enforced at the API [8fNP4N46RRo@21].

### Meta-prompt — rewrite the judge to agree with your humans

```
You are optimising an LLM judge prompt so that it agrees with human labels.

INPUTS
- CURRENT_JUDGE_PROMPT: <paste verbatim>
- EXAMPLES: for each case —
    the model input
    the model output
    the HUMAN label
    the HUMAN's written reason for that label   (required — not optional)
    the CURRENT judge's label and rationale

TASK
1. Read ONLY the disagreements. For each one, name the pattern behind it in
   a single sentence. Group the patterns.
2. Rewrite CURRENT_JUDGE_PROMPT so the next run maximises exact-match
   agreement with the HUMAN labels.
3. Express every fix as GENERAL criteria. Do NOT paste any case from
   EXAMPLES into the new prompt: a prompt that memorises these cases will
   score them correctly and fail on everything else.
4. Do not add instructions that are not justified by an observed
   disagreement. Minimal prompt, maximal agreement.
5. Return: the new prompt, plus a changelog of what changed and which
   disagreement motivated it.

THEN, BEFORE YOU BELIEVE IT
Re-run on the development split (never the frozen test split) and report
accuracy, precision, recall — per criterion, not in aggregate.
```

This is the meta-prompting loop [a3SMraZWNNs@26] with the annotator-reasoning requirement [X4dEHRzBLmc@10] and the no-memorisation rule [kP_aaFnXLmY@17]. Check what it returns by hand — optimisers happily produce longer, more rule-laden prompts that aren't better [TL527yTpxlk@27].

### Metric spec + data split + release gate

```
# metric spec — checked into the repo, loaded by a repository test
id:               listing_claim_support_v1
criterion:        "material listing claim entailed by retrieved records"
positive_class:   "claim entailed by permitted evidence"   # must be declared
unit_of_analysis: [claim, response]                        # both, explicitly
evidence_sources: [retrieved_listing_records]              # must be non-empty
boundary_policy:  "not_verifiable when no permitted field addresses the claim"
aggregation:      "response = unsupported if any claim is unsupported"

# the test that loads this file must FAIL on:
#   - missing positive_class
#   - wrong unit_of_analysis
#   - empty evidence_sources
#   - missing entailment/boundary policy

# data split — frozen BEFORE the judge prompt is touched
design:      108   # prompt iteration happens here
development:  36   # development checks happen here
final_test:   36   # sealed; opened once, after the config is frozen
composition: include typos, long threads, conflicting fields, negation,
             near-duplicates, correct-but-less-verbose answers
leakage:     no test item, paraphrase or label rationale may appear in the
             judge's few-shot prompt, rubric, or regression fixtures

# release gate
report:   TP FP TN FN, TPR (sensitivity), TNR (specificity), balanced_accuracy
floor:    TPR >= 0.90            # safety floor first...
then:     minimise C_FN*FN + C_FP*FP among surviving thresholds
correct:  P = (Q + C - 1) / (S + C - 1)   # observed rate -> estimated true rate
compare:  paired bootstrap on case-level differences, not independent means
predeclare: primary metric, eval set, minimum worthwhile improvement (0.03)
runs_per_case: 5          # separates difficulty from instability
ledger:   keep discarded trials; reporting only the favourable interval is
          selection bias
```

Assembled from the tested metric spec [PjCwlX0XT8o@3], the frozen split [PjCwlX0XT8o@44], hard-case composition [PjCwlX0XT8o@48], the confusion matrix [PjCwlX0XT8o@57], cost-weighted thresholds [PjCwlX0XT8o@59], prevalence correction [PjCwlX0XT8o@66], paired bootstrap [PjCwlX0XT8o@80] and predeclaration [PjCwlX0XT8o@83].

### Judge model card + maintenance triggers

```
judge_id:          listing_claim_support_v1
judge_model:       <family, version>     temperature: 0.0-0.2
generator_family:  <MUST differ from judge_model's family>
context_budget:    <= 4000 tokens
output:            structured JSON, strict parser; parse failure => retry or
                   review, NEVER a silent pass

bias_diagnostics                       # hold everything fixed but one property
  position:        same pair as A/B and B/A; verdict flip rate .......... __
  verbosity:       matched-length slices, equal facts; pref within slice . __
  self_preference: same fixtures via an out-of-family judge; agreement ... __
  task_specific:   fixtures differing only in greeting / formatting ...... __

calibration
  human_alignment (exact match) ......................................... __
  cohen_kappa (chance-corrected — raw agreement is not a number) ........ __
  TPR ____   TNR ____   balanced_accuracy ____
  per-criterion match rate (NOT aggregate — tone and factuality differ) .. __

cost
  $ per judgement at production volume (not demo volume) ................ __
  p95 latency if in the user's critical path ............................ __

failure_direction
  fail_open or fail_closed on ambiguity? ................................ __
  second independent check required? .................................... __

maintenance_triggers          # any one fires a re-validation, not a retune
  - disagreement with sampled human review exceeds ____
  - a new high-risk failure mode appears
  - the provider or judge model version changes
  - the score distribution shifts unexpectedly
  - evaluator cost per run rises above ____
  - repeated evaluator errors or parse failures
policy: thresholds are NEVER silently retuned to preserve a green dashboard
```

The four bias diagnostics [PjCwlX0XT8o@36], chance-corrected agreement [8fNP4N46RRo@9], per-criterion match rate [TL527yTpxlk@39], production pricing [EjFsjfOBZFE@28], p95 latency [EjFsjfOBZFE@29], fail-open/fail-closed [EjFsjfOBZFE@27], and predeclared maintenance triggers [PjCwlX0XT8o@97].

## Where the corpus disagrees

Four live splits. Ignore anyone who tells you these are settled.

### How big should the judge model be?

**Big:** use the most powerful model you can afford, not mini-tier [a3SMraZWNNs@18]; the Stanford position is a judge with much bigger capacity than what it evaluates, so it cannot be fooled by plausible-but-wrong answers [8fNP4N46RRo@26].

**Small:** don't use an expensive frontier model for routine evaluation — the cost defeats the purpose of automating it [vBJF2sy1Pyw@5]. Evidently's data splits the difference: GPT-4 mini hit 95–98% accuracy once the prompt was detailed enough [kP_aaFnXLmY@18], but one step down to GPT-3.5 Turbo collapsed it [kP_aaFnXLmY@20].

*Our read:* the tier floor is real and task-dependent, so find it empirically on your own labels. Use the expensive model for offline prompt refinement and the cheapest model that clears your calibration bar for the online judge [X4dEHRzBLmc@31].

### Numeric scores: broken, or fine?

**Broken:** numeric ratings don't work well for LLM judges — use discrete word labels [pnlT_xatpVQ@9], and move away from 1–5 scores and percentages entirely [X4dEHRzBLmc@9].

**Fine:** the UPSC grading case study runs entirely on continuous marks with MAE against human scores and reports no reliability problem [uQFLY8rQVYA@34], and Bedrock's built-in metrics are all continuous 0–1 scores used directly [HI5b0g_mELA@18].

*Our read:* the split tracks whether a human reference already uses numbers. Exam marking has a real numeric ground truth; "quality 1–10" does not. Netflix's result is the tiebreaker for subjective criteria — switching annotators from a 1–4 Likert scale to binary was a key agreement win [mX9HzIRdBpw@7].

### Score one answer, or compare two?

**Compare:** single-answer judges are too optimistic and agree with whatever they're shown, so use pairwise plus ELO for complex RAG and agent systems [kIvjiH8yJoU@6]. Even when absolute scores disagree with humans, sorting variants worst-to-best shows real correlation with expert rankings [kIvjiH8yJoU@6].

**Score:** a single-answer binary judge reached 95–98% accuracy against human labels through iterative prompt refinement, no pairwise needed [kP_aaFnXLmY@18].

*Our read:* pairwise for choosing between variants, pointwise for monitoring one system in production — you cannot rank a live trace against anything. Note that pairwise output doubles as synthetic preference data for reward-model training [8fNP4N46RRo@22].

### Can a fixed rubric evaluate an agent?

**Yes:** trajectory evaluation is just another application of LLM-as-judge — grade against a reference path, or without one by matching hit nodes to the expected process [spvXj9tnWAQ@16].

**No:** classical fixed-rubric judging could not catch the failures Arize's own agent produced — forgetting context, getting stuck in loops, generating different dynamic UI every turn [q2JrUKBMf0w@2] — so a third eval category is needed alongside deterministic checks and LLM-as-judge [q2JrUKBMf0w@3].

*Our read:* both camps are describing the same fix at different granularity — decompose the judgment until each piece is answerable. Netflix's factuality agents did exactly that and produced their biggest gain [mX9HzIRdBpw@17].

## Sources

Nineteen talks, 703 chunks, read in full across five parallel extraction passes. Each line says what that talk uniquely contributed.

### Cluster 1 — production scale, optimisation, theory

1. [Engineering Better Evals: Scalable LLM Evaluation Pipelines That Work — Dat Ngo, Aman Khan, Arize](https://www.youtube.com/watch?v=spvXj9tnWAQ) [spvXj9tnWAQ] — the full production toolbox (judge, encoder-only classifiers, code evals, human feedback, golden datasets), conditional eval strategy for agents, guardrail-vs-root-cause framing.
2. [Judge the Judge: Building LLM Evaluators That Actually Work with GEPA — Mahmoud Mabrouk, Agenta AI](https://www.youtube.com/watch?v=X4dEHRzBLmc) [X4dEHRzBLmc] — the only fully worked judge-calibration case study with before/after accuracy, dollar cost, and a catalogue of failed experiments.
3. [The Future of Evals: From LLM as a Judge to Agent as a Judge — Aparna Dhinakaran, Arize AI](https://www.youtube.com/watch?v=q2JrUKBMf0w) [q2JrUKBMf0w] — production eval statistics and the argument that fixed-rubric judging breaks on long-horizon multi-agent trajectories.
4. [LLM-as-a-Judge / Autoraters • Guest Lecture at @Northeastern University • March 24, 2026](https://www.youtube.com/watch?v=N_DwZR--XCc) [N_DwZR--XCc] — the academic backbone: pointwise/pairwise/listwise taxonomy, canonical judge-prompt anatomy, APO and RL training methods, the three canonical bias sources.

### Cluster 2 — the build process and the platforms

1. [LLM-as-a-Judge 101](https://www.youtube.com/watch?v=pnlT_xatpVQ) [pnlT_xatpVQ] — the four-step build (data, metrics, prompt/model, meta-eval), the prompting and model-selection pitfalls, the 4K-token rule and its source study.
2. [LLM-as-a-Judge Evals with LangSmith](https://www.youtube.com/watch?v=qoPYlLg7_rY) [qoPYlLg7_rY] — datasets/evaluators/experiments as building blocks, plus a controlled multi-model experiment where the model ranking flipped on a bigger test set.
3. [LLM Eval Methods | LLM-as-a-Judge | Reference Based Evals Vs Reference Free Evals](https://www.youtube.com/watch?v=uQFLY8rQVYA) [uQFLY8rQVYA] — the programmatic/human/model-graded taxonomy worked through three parallel case studies, plus the reference-based vs reference-free distinction and MAE as an alignment metric.
4. [Amazon Bedrock RAG Evaluation ( LLM as a Judge ) | Step-by-Step Guide](https://www.youtube.com/watch?v=HI5b0g_mELA) [HI5b0g_mELA] — cloud implementation mechanics: golden-dataset JSONL schema, IAM/CORS requirements, the built-in metric catalogue, comparing configurations before rollout.

### Cluster 3 — alignment with human labels

1. [How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)](https://www.youtube.com/watch?v=a3SMraZWNNs) [a3SMraZWNNs] — the three-level framework (unit tests → human/model eval → A/B), and the human-vs-model alignment workflow with meta-prompting to close the gap.
2. [3. Tutorial: How to create an LLM judge and align with human labels](https://www.youtube.com/watch?v=kP_aaFnXLmY) [kP_aaFnXLmY] — the numbers-heavy code-review case study: exact accuracy/precision/recall deltas across three prompt iterations and two model swaps.
3. [From LLM-as-a-Judge To Human-in-the-Loop: Rethinking Evaluat... - Eric Pugh & Fernando Rejon Barrera](https://www.youtube.com/watch?v=kIvjiH8yJoU) [kIvjiH8yJoU] — the case for pairwise/ELO over single-answer scoring, and for surfacing judge explanations to humans as an exploration tool rather than a dashboard number.
4. [Complete Beginner's Course on AI Evaluations in 50 Minutes (2025) | Aman Khan](https://www.youtube.com/watch?v=TL527yTpxlk) [TL527yTpxlk] — the PM-oriented live build of a golden dataset and rubric in a spreadsheet before any judge exists, including a caught arithmetic failure and per-criterion match rates.

### Cluster 4 — operations, code, theory, and an industrial case study

1. [The challenges in using LLM-as-a-Judge - Sourabh Agrawal | Vector Space Talk #013](https://www.youtube.com/watch?v=vBJF2sy1Pyw) [vBJF2sy1Pyw] — where in the request lifecycle to run judge calls, cost control, root-cause analysis on failure clusters, and the RAG starter metric set.
2. [LLM as a Judge Explained | Hands-On GenAI Evaluation with Real Code](https://www.youtube.com/watch?v=3FcYdRQPMCo) [3FcYdRQPMCo] — the fully worked code walkthrough: two candidate models plus one judge, a structured-JSON judge prompt, and a scored comparison loop.
3. [Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 8 - LLM Evaluation](https://www.youtube.com/watch?v=8fNP4N46RRo) [8fNP4N46RRo] — why raw agreement misleads, why BLEU/ROUGE/METEOR fail, the bias taxonomy with remedies, constrained decoding, and atomic-fact factuality scoring.
4. [Evaluating Netflix Show Synopses with LLM-as-a-Judge](https://www.youtube.com/watch?v=mX9HzIRdBpw) [mX9HzIRdBpw] — the most detailed industrial case study: expert-calibrated golden dataset, per-criteria judges, inference-time scaling, factuality agents, and causal validation against real member behaviour.

### Cluster 5 — regulated production and statistical rigour

1. [UnlikelyAI Webinar: Is LLM-as-a-Judge Ready for Production in Financial Services?](https://www.youtube.com/watch?v=EjFsjfOBZFE) [EjFsjfOBZFE] — the regulated-industry lens: black-box audit risk, fail-open vs fail-closed, production pricing and p95 latency, and decomposition into narrow deterministic sub-checks.
2. [Key Metrics and Evaluation Methods for RAG](https://www.youtube.com/watch?v=cRz0BWkuwHg) [cRz0BWkuwHg] — the RAG metric checklist (precision, recall, hit rate, MRR, NDCG; faithfulness, answer relevancy, answer correctness) plus concrete pairwise-judging practice.
3. [Chapter 6: Metrics, LLM Judges, and Statistical Confidence | Evals For AI Engineers](https://www.youtube.com/watch?v=PjCwlX0XT8o) [PjCwlX0XT8o] — the engineering playbook: criterion-first judge prompts, frozen calibration splits, bias diagnostics, sensitivity/specificity correction, paired bootstrap, and evaluator-as-production-code.

## What this corpus does not tell you

Nineteen talks, and these questions still go unanswered. Where you see a confident answer elsewhere, ask for the evidence.

- **When is a judge calibrated enough?** No generalisable threshold — no required correlation, kappa or agreement score — for trusting a judge in production. The only figure offered is one presenter's informal ~95% aspiration on a single narrow case study [X4dEHRzBLmc@27].
- **How many labels are enough?** No sample-size formula or power analysis anywhere. The rules of thumb range from 10 to 600 and are all explicitly presented as informal; Arize declines to give a rule at all [pnlT_xatpVQ@6].
- **What does a judge actually cost?** No dollar-per-call or token-per-call figures at volume. The corpus offers "seconds, not milliseconds" [EjFsjfOBZFE@17] and one $200–300 optimisation run [X4dEHRzBLmc@31]. You cannot budget from this.
- **Which model is the best judge?** No head-to-head benchmark of specific judge models on bias or accuracy. Recommendations stay generic — bigger, cheaper, different family.
- **Judge drift.** No cadence, trigger or worked monitoring example for re-validating a judge once live, or for what to do when a provider silently updates the model you calibrated against. PjCwlX0XT8o names maintenance triggers but no video runs the loop [PjCwlX0XT8o@97].
- **Combining narrow judges into one decision.** The corpus insists on many small judges but never works an example of aggregating tone, policy and factuality into a release gate beyond a passing mention of averaging.
- **Adversarial pressure on the judge.** Nothing on defending a judge prompt against prompt injection, or on a generator learning to exploit judge biases during RLHF-style optimisation — despite reward hacking being named as the risk [N_DwZR--XCc@48].
- **Self-hosted judges.** No cost, latency or quality comparison of open-weight judge models against hosted APIs, and no comparison of eval frameworks — Ragas is named [cRz0BWkuwHg@9] but never compared to alternatives.
- **Cost of the judging paradigms.** No concrete dollar or millisecond comparison across pointwise, pairwise and listwise judging, and the ELO/ragelo approach is described without its K-factor or scoring maths, so you cannot reproduce it from the transcript.
- **Statistical significance in the platform demos.** A jump from 79% to 94% on 20 examples is reported without any significance treatment [qoPYlLg7_rY@14] — and on a 168-example set the ranking reversed, with the older GPT-4o-mini beating the newer model [qoPYlLg7_rY@16]. Treat every small-n eval result this way.
- **A reference architecture.** No talk draws where judges sit in an agent, how many judge calls one agent turn should cost, or how J1–J7 in "Draw the system" combine into a single release decision. The placements are described one at a time — instrument every stage [vBJF2sy1Pyw@16], skip what is downstream of a failed step [spvXj9tnWAQ@12] — and never assembled into one diagram or costed as a whole.

---

LLM-as-a-Judge · a field guide compiled from the transcript corpus.

19 source videos · 703 transcript chunks read in full · 5 parallel extraction passes · compiled 2026-09-14 · revision 4, 2026-09-15.

Every claim on this page resolves to a chunk you can open. Where it does not, it is in the gaps.
