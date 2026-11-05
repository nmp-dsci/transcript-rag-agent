---
title: The Fine-Tuning Pipeline
topic: "Fine-tuning LLMs end to end: input sets, method options, parameters, hyperparameterisation, hosting and scoring"
compiled_at: 2026-10-22
videos: 20
chunks: 1482
sources:
  - video_id: JfaLQqfXqPA
    title: "RFT, DPO, SFT: Fine-tuning with OpenAI — Ilan Bigio, OpenAI"
  - video_id: uLrOI65XbDw
    title: "Everything you need to know about Fine-tuning and Merging LLMs: Maxime Labonne"
  - video_id: cO6Ly7mIziQ
    title: "LoRA & QLoRA Explained Simply | Full Fine-Tuning vs PEFT + Intuition + Practical (Complete Guide)"
  - video_id: 4yNswvhPWCQ
    title: "Build an LLM from Scratch 7: Instruction Finetuning"
  - video_id: YxXBojdD1as
    title: "Fine-Tuning a Model: The Easiest Way to Start (Unsloth Tutorial)"
  - video_id: v7qMjy_RxOs
    title: "The Honest Guide To Fine-Tuning Local AI In 2026"
  - video_id: R78NYFkWH10
    title: "Fine-Tuning LLMs for Real-World Tool Calling: Lessons from tau2-bench"
  - video_id: CvRngaQZQ3Y
    title: "Improving Agents is a Data Mining Problem — Vivek Trivedy, LangChain"
  - video_id: NovNcsKX8AU
    title: "The Never Ending Lore of Harness | Vivek Trivedy (Product Lead, Langchain)"
  - video_id: T0HhO4YtTfE
    title: "AI System Design: From Idea to Production - Apoorva Joshi, MongoDB"
  - video_id: a3SMraZWNNs
    title: "How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)"
  - video_id: vBJF2sy1Pyw
    title: "The challenges in using LLM-as-a-Judge - Sourabh Agrawal | Vector Space Talk #013"
  - video_id: 1jvxxa7tdjw
    title: "Exploring MLOps and LLMOps: Architectures and Best Practices"
  - video_id: uaiq1HvQKoI
    title: "GenAI in Production: How RAG & AI Agents Are Shaping the Future"
  - video_id: Q679gH7oszg
    title: "How I Build and Ship Custom AI Solutions for Clients"
  - video_id: qF5il_9IwME
    title: "Build a Full-Stack GenAI Project in 4 Hours (FastAPI, React, Supabase)"
  - video_id: E8zpgNPx8jE
    title: "Build a Complete End-to-End GenAI Project in 3 Hours"
  - video_id: NRQVTxl14x8
    title: "GenAI in Production: How AI Is Transforming the Entire Filmmaking Pipeline"
  - video_id: lsgVRCJ4fko
    title: "Learn Design Systems: Figma Variables, Components, AI, and More"
  - video_id: r-MPiu0E4W4
    title: "Design with AI - Full Guide (Tools, Workflows, Design Systems)"
---

# Train the behaviour, not the knowledge

A staff-engineer walkthrough of fine-tuning as a system: when it earns its place, how to construct the input set, which method to pick (SFT, DPO, RFT, GRPO, merging), which parameters you actually set, how to host the result, and how to score it honestly. Distilled from twenty talks — OpenAI, LangChain, Databricks, MongoDB, Hugging Face practitioners, Unsloth and from-scratch tutorialists, and a tau²-bench case study.

20 source videos · 1,482 transcript chunks read in full · 5 parallel extraction passes · compiled 2026-10-22

## Thesis

Yes — and the corpus answers it in a specific order. Fine-tuning is continued training [JfaLQqfXqPA@4] that specialises a model you already have. It is not a knowledge transplant: every method in this corpus retrains only a subset of the weights [JfaLQqfXqPA@11], the base stays frozen under an adapter [YxXBojdD1as@2], and the honest summary of what you get is a methodology and a format — not "the best way to give correct information" [JfaLQqfXqPA@65]. So the practical sequence is: earn the fine-tune behind prompting, retrieval and harness work; build a narrow, machine-scorable input set and treat its quality as the whole job; pick the method by what you are teaching (imitation, preference, or reasoning); keep the parameter list to three or four knobs; host the adapter where your cost curve lives; and score the identical evaluation before and after, or you have learned nothing.

Do you need hyperparameterisation? Barely. The corpus converges on a short list — epochs, learning rate, LoRA rank — and one discipline: change exactly one thing per iteration [YxXBojdD1as@24]. Nobody in these twenty talks runs a sweep. The reason is that the dominant term is data, not configuration: feeding a model uncleaned input doesn't give you a neutral result, it means the end result is much worse [v7qMjy_RxOs@10], and a fine-tuned model inherits whatever errors or biases lurk in the training data [uaiq1HvQKoI@46]. Budget your effort accordingly: a weekend of dataset engineering beats an afternoon of sweeps.

> "Fine tuning is not an event, it's essentially an experimental loop." [YxXBojdD1as@23]

**Earn it.** Prompt, then retrieve, then engineer the harness — then fine-tune. Harness work gives feedback in minutes and is cheap relative to training, so the recommendation is to start with harness engineering [NovNcsKX8AU@52] and only escalate to break through that ceiling [CvRngaQZQ3Y@15].

**Build the set.** The deliverable of a fine-tuning project is a dataset, not a checkpoint. Mask the prompt and train on the outputs only [uLrOI65XbDw@3], and grade every sample on accuracy, diversity and complexity before it enters the file.

**Score it twice.** Run the baseline first. The comparison is only valid with the same questions, same scoring rules, comparable generation settings [YxXBojdD1as@4] — because you can't improve what you can't measure [T0HhO4YtTfE@22].

## The gate: when fine-tuning is the right tool

Treat fine-tuning as the specialised machine, not the hammer: it is a high-upfront-cost tool next to prompting's fast-iteration toolkit. Concretely it will cost you at least a weekend [v7qMjy_RxOs@8] and a competent GPU. Use these six tests before you commit.

- **Run the escalation ladder first.** Four steps, in order: try a better prompt [v7qMjy_RxOs@8], then RAG, then an agentic loop, and only fine-tune if all three fail. OpenAI's own guidance is the same shape — always start with prompt engineering and see how far you can go [JfaLQqfXqPA@32], then run evals.
- **Confirm the failure is behavioural.** Reach for fine-tuning when the LLM's failures are behavioral rather than data [T0HhO4YtTfE@12] or orchestration problems. Otherwise you are training around a retrieval bug. Design the simplest system first, then find gaps during the evaluation [T0HhO4YtTfE@10].
- **Check you can even assemble the data.** If you cannot build the instruction dataset, that is a scoping answer, not a tooling problem: it's a good sign that you need to rescope the project [uLrOI65XbDw@2].
- **Fine-tune for guarantees, not facts.** A system prompt is not going to cover every type of question [v7qMjy_RxOs@1]. If you need a model that writes like you every time with a real guarantee [v7qMjy_RxOs@8], that is a weights problem. If you need current facts, that is retrieval.
- **Price the inference, not the training.** A closed model can be four or five times more expensive [R78NYFkWH10@1] per task, and at high volume you can move from token costs to hardware costs [CvRngaQZQ3Y@9] with a cluster you spin down when idle.
- **Expect to win only on the narrow task.** On a curated vertical slice a fine-tuned open model can reach and then go beyond frontier performance [CvRngaQZQ3Y@9], and depending on training-mixture coverage your fine-tuned model will be better [NovNcsKX8AU@51] than the unfine-tuned alternative. Outside that slice, anything you don't include might regress [JfaLQqfXqPA@36].

And know the ceiling of the artefact itself: fine-tuning only changes the behaviour of the model [YxXBojdD1as@32]. The system around it still needs context, tools, guardrails and evaluation. Also beware vendors: many platforms that claim to train on your data are really just doing retrieval, since a lot of them just inject your information into a prompt [v7qMjy_RxOs@4].

## The option space

Two orthogonal choices: *what you teach* (the objective) and *how much of the model you touch* (the adaptation method). Get the objective wrong and no amount of rank tuning saves you. The lifecycle context: pretraining produces a base model that will autocomplete your question instead of answering it [uLrOI65XbDw@0]; SFT teaches instruction-following; preference alignment produces the chat model. You can intervene at any point — after any stage we can pick up the model [cO6Ly7mIziQ@9] and train on top.

### What you teach

- **SFT — imitation.** A direct mapping from input to output. Mechanically it is the pretraining objective again: instruction fine-tuning uses exactly the same loss used during pre-training [4yNswvhPWCQ@44]; only the text structure changes. Fits classification, structured extraction and distillation into small models.
- **DPO — preference.** Pairs a chosen and a rejected answer. The model learns the delta between the examples [JfaLQqfXqPA@8], not the examples. Popular over RLHF because it skips the reward model, which otherwise means holding two LLMs in compute memory [4yNswvhPWCQ@53]. Use it for tone, style, refusals and safety.
- **RFT — reasoning.** Reinforces chains of thought that score well against a grader. Built for gradable hard problems [JfaLQqfXqPA@10] with unambiguous answers, and it is startlingly data-efficient — 40 to 80 very high quality, very high signal examples [JfaLQqfXqPA@42] can generalise. Two hard limits: it works on reasoning models only (no interplay [JfaLQqfXqPA@47] with non-reasoning bases), and agentic use cases are limited by a single turn [JfaLQqfXqPA@48].
- **SFT → GRPO — staged RL.** The tau²-bench tool-calling study trains SFT on teacher trajectories first, then pushes performance further with reinforcement learning [R78NYFkWH10@8]. The reward is shaped, not pass/fail: action fraction [R78NYFkWH10@11], a termination bonus, tool-name accuracy, tool-argument accuracy and a step penalty.
- **Merging — no GPU at all.** Combine existing fine-tuned checkpoints by weight arithmetic, with no training. On one leaderboard generation the entire top eight or ten were merge models [uLrOI65XbDw@9]. Use mergekit [uLrOI65XbDw@10]: SLERP (spherical linear interpolation [uLrOI65XbDw@10], two models, per-layer factors) or DARE, recommended not with three but with seven or eight models [uLrOI65XbDw@11].
- **Exotic merges — know the limits.** Passthrough merging concatenates layers with zero training: one presenter built a 120B model by repeating 10 layers six times [uLrOI65XbDw@12], and it went viral for being particularly good at creative writing [uLrOI65XbDw@12]. You can also graft feed-forward layers into a FrankenMoE by adding a router [uLrOI65XbDw@13] — but for raw performance prefer SLERP and DARE, because MoE merging is more experimental [uLrOI65XbDw@14].

### How much of the model you touch

- **Full fine-tuning.** Every weight is trainable, so cost scales linearly with size: a 13-billion model means 13 billion weights [cO6Ly7mIziQ@13]. It still wins on peak performance, but it needs clusters and is argued to be not a practical choice [cO6Ly7mIziQ@15] for most teams.
- **LoRA.** Freeze the pretrained weights and add adapters to each targeted layer [uLrOI65XbDw@7], exploiting the fact that a matrix's rank is its number of linearly independent rows and columns [cO6Ly7mIziQ@36]. On GPT-3 scale that is about 4.7 million trainable parameters [cO6Ly7mIziQ@62] instead of 175 billion — typically half to 2% of the parameters [v7qMjy_RxOs@4] — for accuracy matching the 89.5% full-fine-tune baseline [cO6Ly7mIziQ@63].
- **QLoRA.** LoRA with the frozen base loaded in 4-bit [cO6Ly7mIziQ@63], so work that needed 24GB+ fits in 8 to 16 GB [cO6Ly7mIziQ@64]. Do not pretend it is free: it also degrades performance, so there's a trade-off [uLrOI65XbDw@8]. Reach for it when VRAM is the binding constraint, not by default.
- **The wider PEFT shelf.** LoRA is the famous one, but the shelf also holds DoRA, prefix tuning, prompt tuning, BitFit, IA3 and plain adapters — LoRA is one method of achieving PEFT [cO6Ly7mIziQ@16], not the definition of it. The payoff you are buying is less memory, faster training, lower cost, easy customization [cO6Ly7mIziQ@18].

### Tooling, by where you want the job to run

- Open-source trainers: TRL from Hugging Face, built on top of Transformers [uLrOI65XbDw@3], plus Axolotl (YAML-driven) and LLaMA Factory (GUI).
- A LoRA run on Hugging Face needs transformers, datasets, peft, accelerate, bitsandbytes and TRL [cO6Ly7mIziQ@66].
- No-code: Unsloth's UI runs the whole loop and ships an OpenAI-compatible API locally [YxXBojdD1as@30].
- Managed: a guided API such as Mosaic AI Model Training [1jvxxa7tdjw@60] for continued pretraining and instruction tuning, versus DIY Spark/Ray/DeepSpeed.
- Split infrastructure is normal: the tau²-bench team used a serverless supervised fine-tuning [R78NYFkWH10@9] service for SFT and the OpenPipe SDK on self-managed H100s for the RL stage.
- Base-model shortlist for open fine-tunes: Qwen3 30B [R78NYFkWH10@1] for tool-calling work, and the GLM series [NovNcsKX8AU@71] as a credible agentic base.

## Building the input set

This is the part that decides the outcome. There are roughly five steps [v7qMjy_RxOs@10] in a first project — collection, dataset engineering, LoRA training, evaluation, export. Below is the dataset half expanded into the seven moves the corpus actually demonstrates.

### 1. Pick one behaviour, and make it machine-scorable

If you cannot grade an output with code, you cannot run the loop. Define the label space before you open a tool.

- Start with the behaviour you want to measure [YxXBojdD1as@4] — categories, severity levels, required keys — and only then choose tooling.
- Force a structured target: in the worked example, exactly one JSON object containing three fields: category, severity and next action [YxXBojdD1as@3]. Subjective "does this feel nicer" targets have no scoreboard.
- Start small on the base model too — a deliberately tiny base like Qwen 2.5 0.5B [YxXBojdD1as@8] is for learning the loop, not shipping.
- Where success is a world-state change, grade the world: tau²-bench asks whether the customer's data allowance actually increased by 2 GB [R78NYFkWH10@3], not whether the transcript sounded right.

### 2. Choose the record shape for your objective

Each objective has exactly one file format. Getting the masking wrong silently trains the model to parrot your prompts.

- SFT: a system+user prompt as context and the assistant message as the target — the assistant message contains the correct answer [YxXBojdD1as@4]. The prompt is masked out of the loss; you train on the outputs only [uLrOI65XbDw@3].
- On a hosted API the file is JSONL with a messages array, and the trainer takes the last assistant message and learns from it [JfaLQqfXqPA@27]. Upload with purpose set to fine-tune [JfaLQqfXqPA@28].
- DPO: one instruction, a chosen answer and a rejected answer [uLrOI65XbDw@4]; the chosen response is the preferred response [4yNswvhPWCQ@55].
- Local/from-scratch SFT: wrap raw JSON into text, because models read text — apply the Alpaca prompt-style template [4yNswvhPWCQ@7]. Prefer terser templates (Phi-3 style) since generation is token-by-token and shorter is cheaper [4yNswvhPWCQ@9].

### 3. Generate candidates — synthetic data is the norm

Hand-labelling does not scale and is not required. Most SFT corpora are model-made.

- Be unembarrassed about it: most SFT datasets use synthetic data and that's perfectly fine [uLrOI65XbDw@4].
- From raw source material (transcripts, docs), invert the task: use a local language model to generate a question relevant [v7qMjy_RxOs@11] to each snippet, then pair question with snippet to get chat-format pairs.
- When the output space is enumerable, enumerate it. For a 120-function assistant — the first time anybody fine-tuned a model with over 100 functions [JfaLQqfXqPA@13] — OpenAI generated every valid permutation of call arguments deterministically, then had a frontier model reverse-generate a plausible user command for each.
- Distil from a stronger model's production behaviour: take the good traces and the good examples [CvRngaQZQ3Y@10] from the large model's runs and train a 9–13B student to mimic them.
- For a DPO negative, construct the contrast deliberately — generate the positives, then manually filter [JfaLQqfXqPA@36] and synthesise a matched weak answer on the same topic so the delta is about the quality you care about.
- Tooling exists for the cold start: Unsloth ships a recipe builder, though the presenter prefers building his own datasets [YxXBojdD1as@9].

### 4. Mine and filter traces if you already have production

A shipped system is a dataset generator. Most teams fine-tune on their own traces, not public corpora.

- The order is ship first, then collect a ton of traces [CvRngaQZQ3Y@0] — tool calls, outputs, CLI/API usage — before any mining.
- Cheapest possible first step: just turn on tracing and point an agent at it [CvRngaQZQ3Y@18], then ask targeted questions — find good and bad interactions where users got upset [CvRngaQZQ3Y@4], or whether the agent degrades after compaction.
- Budget for it: reading traces at scale is super expensive [CvRngaQZQ3Y@6] — millions of traces times millions of tokens. Mine with a cheap judge; a harness-engineered open model matched Opus-level trace-judging at one to two orders of magnitude lower cost [CvRngaQZQ3Y@7].
- Log rollouts as first-class artefacts: the tau²-bench team logs the traces of the interaction between user and agent [R78NYFkWH10@4] in an experiment tracker.
- Filter hard. Of ~822 raw rollouts they kept only 669 clean trajectories for supervised fine-tuning [R78NYFkWH10@10], screening for efficiency and hallucinated tool calls.

### 5. Grade every sample on three axes

Accuracy, diversity, complexity. These are the only quality dimensions the corpus names, and all three are load-bearing.

- **Accuracy** — the outputs must be factually correct [uLrOI65XbDw@5], typo-free, no invented knowledge.
- **Diversity** — many topics and styles unless the use case is genuinely narrow; this is the direct defence against regression, because anything you don't include might regress [JfaLQqfXqPA@36].
- **Complexity** — tasks that force reasoning, e.g. where the output will have chain-of-thought reasoning [uLrOI65XbDw@6], rather than Wikipedia-style lookups.
- For RFT, add a fourth bar: unambiguous grading, where any qualified grader following the rubric would agree. The email-triage failure case was ~600 personally labelled emails where the author conceded he wouldn't hit the same classifications for all of them [JfaLQqfXqPA@45].

### 6. Deduplicate, filter, cluster, regenerate

The curation pipeline is mechanical and cheap, and it is where most of the quality comes from.

- Start from open-source datasets, combine them, then deduplicate — deduplication can be either exact [uLrOI65XbDw@6] or fuzzy. Hand-built sets need it too: find near duplicates and remove them [4yNswvhPWCQ@49].
- Apply quality filters — rule-based, or reward-model / LLM-as-judge scoring — then explore the set with clustering before generating more.
- Add programmatic guardrails inside the data stage, not after: more programmatic evaluation and guardrails in your SFT stage [R78NYFkWH10@23] — flag tool-not-found errors, check response tone.
- Never trust a teacher trace because it passed: RL rollout trajectories are actually hallucinating a lot [R78NYFkWH10@22], and that noise propagates into the RL stage and destabilises it. Their teacher solved 68 of 74 tasks [R78NYFkWH10@9] — and the passing traces still needed filtering.
- Keep the prompt context rich. Stripping function schemas out of the prompt to save latency backfired: removing them from context makes them much harder to learn through fine-tuning [JfaLQqfXqPA@20].

### 7. Split three ways, and check the licence

Two splits is one too few, and one legal check can invalidate the whole corpus.

- Three splits: training updates the adapter, validation checks generalisation while you iterate, and you keep a final test set completely untouched [YxXBojdD1as@6]. A concrete ratio from the from-scratch build: 85% training, 10% test, 5% validation [4yNswvhPWCQ@12].
- Pass the validation file to the job explicitly — it is what tells you how well it's doing [JfaLQqfXqPA@29] so you don't overfit.
- Licence contamination is real: outputs generated under one model's terms mean you can't use those outputs to fine-tune some other architectures [1jvxxa7tdjw@40]. Audit which model produced which synthetic rows before you mix providers.
- Data quality is the gate, not the sequencing: a model cannot self-correct a bad corpus, and feeding it bad data makes the end result much worse [v7qMjy_RxOs@10].

## How many examples — the corpus numbers

Every number below is a reported figure from a specific talk, not a general rule. They disagree by an order of magnitude because they are answering different questions: a hosted SFT on a narrow classifier needs hundreds, a from-scratch instruction-tuned production model needs six figures. Find the row that matches your objective and base.

| Scale | Context | Source |
| --- | --- | --- |
| 40–80 | RFT, high-signal unambiguous examples — enough to generalise | [JfaLQqfXqPA@42] |
| 20–50 | A first local LoRA experiment, plus a separate eval set | [YxXBojdD1as@33] |
| 50–100 | Hosted SFT — where you start to see "signs of life" | [JfaLQqfXqPA@18] |
| 150 / 500 | The two sizes actually trained in the banking-intent demo | [JfaLQqfXqPA@30] |
| 500+ | Recommended floor for production-grade SFT | [JfaLQqfXqPA@36] |
| 669 / 822 | Clean trajectories kept from raw rollouts for tool-calling SFT | [R78NYFkWH10@10] |
| ~1,000s | Instruction fine-tuning to tailor a model to a task (enterprise MLOps view) | [1jvxxa7tdjw@43] |
| 1–2M+ tokens | Raw source text before transformation, for an 8B local fine-tune | [v7qMjy_RxOs@10] |
| 100,000+ | What production LLMs are actually instruction-tuned on | [4yNswvhPWCQ@5] |
| 50–100 | Separate figure: the *evaluation* set, drawn from real traffic | [vBJF2sy1Pyw@12] |

The staff-engineer read: the floor is far lower than people expect, and the method moves it more than the task does. RFT buys data efficiency at the price of requiring a reasoning base and an unambiguous grader; SFT is cheap to label and expensive in volume; continued pretraining is a different sport entirely, measured in millions of tokens and billions for pretraining.

## The parameters you actually set

Short answer to the hyperparameterisation question: you need a small, ordered set of decisions and one disciplined loop — not a sweep. No talk in this corpus runs a hyperparameter search. What they all do is fix three knobs, read the loss curve, and change one variable at a time.

- **The three that matter.** The three parameters to recognise immediately are epochs, learning rate and rank [YxXBojdD1as@14]. Everything else is defaults until a curve tells you otherwise.
- **Learning rate: push until it breaks.** The most important SFT hyperparameter, and the search procedure is one line — go as high as you can until your loss explodes [uLrOI65XbDw@8], then back off. Pair it with AdamW, which combines the weight-decay concept [cO6Ly7mIziQ@57] with a scheduler and warmup.
- **Epochs: two, then watch.** In the from-scratch run two was better than one [4yNswvhPWCQ@48], and further epochs push into overfitting as training loss drops below validation loss. The Unsloth tutorial also runs 3 epochs. Start at 2–3.
- **Rank: capacity, not quality.** Rank controls roughly how much capacity the adapter has [YxXBojdD1as@14] — and more capacity is not automatically better, it just costs more memory and parameters. Rank 16 with alpha 32 on a 1.1B base left just 0.277% of parameters trainable [cO6Ly7mIziQ@76].
- **Batch size and accumulation.** On real GPUs, maximise batch size to maximise GPU utilisation [uLrOI65XbDw@9]; sequence length trades against VRAM and need not match the base model's full context. On small hardware, fake it: collect gradients from four batches, making one update [YxXBojdD1as@12], so a physical batch of 2 effectively sees eight examples before one full update [YxXBojdD1as@15].
- **Dropout and padding.** Dropout randomly ignores some training signals to reduce overfitting [YxXBojdD1as@12]; the tutorial leaves it at zero for a small run. Pad per batch, not globally — pad only to the longest element in each batch [4yNswvhPWCQ@14] — and mask the rest with -100, the cross-entropy default ignore index [4yNswvhPWCQ@27], keeping exactly one pad token so the model learns to stop.
- **The reward function, if you do RL.** With GRPO the reward components *are* your hyperparameters, and their weights are the dangerous part. Weighting them evenly produced reward hacking [R78NYFkWH10@13] — the model farmed the action-fraction term by re-calling a correct-but-redundant tool. The fix was upweighting task completion.
- **The grader, if you do RFT.** Graders are the tunable surface: string checks, text similarity, sandboxed Python where you write Python code sandboxed [JfaLQqfXqPA@42], or model graders, nestable into a weighted multi-grader. Partial credit works: half credit per correct field moved hydrogen-bond prediction from roughly 65% [JfaLQqfXqPA@46] to around 80%.

### Read the loss curve; nothing will stop the job for you

Overfitting protection is manual. Asked directly whether hosted jobs check for it: "do the fine tuning runs check for overfitting? No, they don't" [JfaLQqfXqPA@31] — you watch the curves and stop the job yourself. And a wrong curve is diagnostic, not cosmetic: a training loss chart trending upwards instead of downwards [YxXBojdD1as@16] means the run config is broken. In that case the cause was max steps set to 3 [YxXBojdD1as@17] instead of 3 epochs — three optimizer updates total. After the fix, the training loss gradually went down [YxXBojdD1as@18], confirming the diagnosis. Expect to wait: hosted jobs run anywhere from 30 minutes to 24 hours [JfaLQqfXqPA@29].

### Hardware, because it constrains the parameters

- LoRA is what makes the loop iterable: instead of waiting 20 hours [v7qMjy_RxOs@13] you can finish in about two.
- Measured: a medium model took 2 to 3 hours on a 5090 [v7qMjy_RxOs@14]. A 355M full fine-tune for 2 epochs took about 15 minutes [4yNswvhPWCQ@46] on an M3 MacBook Air CPU versus roughly a minute on an A100.
- VRAM is the hard wall: a 27B model will likely exceed 14 GB [v7qMjy_RxOs@14] even quantized, and offloading to system RAM will grind fine-tuning to a halt [v7qMjy_RxOs@16].
- Vendor verdict: AMD is actively developing CUDA counterparts [v7qMjy_RxOs@16] and ROCm can already train many models; definitely avoid fine-tuning on Apple silicon [v7qMjy_RxOs@17].
- Don't chase bit-identical reproducibility across backends — results are very close but not identical between CPU and GPU [4yNswvhPWCQ@47], and vary by library version.
- If results are weak, a bigger base can beat better hyperparameters: when 124M GPT-2 produced unusable output, the fix was that he went one model size up [4yNswvhPWCQ@38].

### Delete on sight

| Delete on sight | Replace with |
| --- | --- |
| `max_steps = 3` when you meant three epochs | Epochs. Three updates is not three passes [YxXBojdD1as@17] |
| Padding the whole dataset to one global max length | Per-batch padding to that batch's longest element [4yNswvhPWCQ@14] |
| Loss computed over the prompt tokens | Mask the prompt; train on outputs only [uLrOI65XbDw@3] |
| Offloading training state to system RAM | A model sized to your VRAM [v7qMjy_RxOs@16] |
| Assuming the trainer halts on overfit | Your own eyes on train/validation loss [JfaLQqfXqPA@31] |
| Changing the data and the config in one iteration | One change per run [YxXBojdD1as@24] |
| Unfiltered teacher trajectories | Clean-trajectory filtering before SFT [R78NYFkWH10@22] |
| Evenly weighted RL reward components | Task completion upweighted over sub-rewards [R78NYFkWH10@13] |
| Rewards or penalties applied inside the chain of thought | Grade the final result only — optimising the CoT drops your ability to catch bad behaviour [JfaLQqfXqPA@58] |
| Fine-tuning to install facts | RAG or agentic retrieval [JfaLQqfXqPA@65] |
| `pass@1` as your reliability number | pass^k, which is significantly lower than pass one [R78NYFkWH10@14] |
| A judge you never aligned to a human | An agreement score against human labels [a3SMraZWNNs@35] |
| Dashboards of "helpfulness 4.2, truthfulness 4.5" | Specific boolean checks tied to a decision [a3SMraZWNNs@33] |
| Benchmarking against a model released after your eval set | A baseline that predates the benchmark [R78NYFkWH10@19] |

## Hosting the model

Four deployment shapes appear in this corpus, and the choice is driven by where your cost curve and your control requirements sit — not by training convenience. Note that the fine-tuned artefact is usually an adapter, not a model: on a hosted platform the job is generating a LoRA component [JfaLQqfXqPA@55] that is loaded alongside unchanged base weights at inference.

- **Adapter on a hosted base.** You train and serve on the provider's infrastructure; the adapter rides on their base model. Lowest operational burden, no VRAM planning, and the per-token cost model stays. This is the default for hosted SFT, DPO and RFT.
- **Merged weights, local runtime.** After LoRA training, merge the LoRA adapter into the base model [cO6Ly7mIziQ@80] to ship one standalone artefact. Then export using the GGUF file format [YxXBojdD1as@25] at Q4_K_M [YxXBojdD1as@26] quantization (distinct from the 4-bit used in training), which makes it easier to run on platforms like LM Studio [v7qMjy_RxOs@19] and Ollama. Point the hub at the local export folder so you aren't downloading the model again [YxXBojdD1as@27].
- **Adapters kept separate and swapped.** Because the base stays frozen you can hold a different LoRA adapter per task without duplicating the full model [cO6Ly7mIziQ@55]. This is the right shape when you have many narrow behaviours and one base — but you give up the single-artefact simplicity of merging.
- **Your own cluster, tiered routing.** At high volume, move from token costs to hardware costs [CvRngaQZQ3Y@9]: a dedicated cluster, spun down when idle, for effectively unlimited inference. The common end state is tiered — teams break their workload up [R78NYFkWH10@20] so frontier models take the hard tasks and the fine-tuned small model takes the simple ones.
- **Serve behind a promotion gate.** Don't flip traffic. Serving should support online strategies like A/B testing [1jvxxa7tdjw@12] and canaries at one endpoint, and the new checkpoint should carry a challenger alias [1jvxxa7tdjw@15] against the proven champion so a human gates promotion. The same pattern from the RAG-production side: blue-green or canary deployments [uaiq1HvQKoI@43], validated before the switch.
- **Promote code, not checkpoints.** The enterprise MLOps recommendation is explicit: deploy code [1jvxxa7tdjw@7], promoting the training pipeline through dev/staging/prod and retraining in each, because it locks down your production training data [1jvxxa7tdjw@8] and makes retraining reproducible.
- **Modest infrastructure is fine.** One consultancy's baseline for client systems: a bare-metal VM, no managed services [Q679gH7oszg@36] with Docker Compose and Caddy for automatic HTTPS — roughly 10× cheaper than hyperscalers at their scale — plus a hard network baseline: block all IPs by default [Q679gH7oszg@38], whitelist a static VPN IP, expose only 80/443.
- **Instrument before you need it.** Pair LLM trace monitoring with application error tracking — Langfuse for all LLM traces [Q679gH7oszg@40] plus Sentry piped to Slack. Traces are also your next training set, so this is dual-purpose spend. Add a user feedback channel — thumbs up, thumbs down, comments [uaiq1HvQKoI@41].
- **Gate retraining on a number.** Drift triggers should be concrete, not vibes: one reference architecture retriggers training when a metric crosses a threshold — the R-squared threshold is 0.7 [1jvxxa7tdjw@29] across a rolling detection window. Pick the equivalent scalar for your task and wire it to the pipeline.

Risks to budget for once it is live, from the production-RAG side and applicable verbatim to a fine-tuned endpoint: hallucinations, data leaks, bias, prompt injection [uaiq1HvQKoI@44], IP exposure and runaway usage cost. And one architectural stabiliser: models churn monthly, so keep the surrounding stack boring — AI is going to change every month but your process shouldn't [Q679gH7oszg@49].

## Scoring the model

Scoring is where fine-tuning projects are won or quietly wasted. The corpus gives a five-stage ladder that starts before training and ends in production slices. Build it first; you can't improve what you can't measure [T0HhO4YtTfE@22].

### 1. Baseline the untrained model, with the exact harness

The before/after comparison is the entire product of a fine-tuning run. It is also the easiest thing to invalidate.

- Hold everything constant: same evaluation, same questions, same scoring rules, comparable generation settings [YxXBojdD1as@4] — including the system prompt you trained with.
- The point of the baseline is attribution: write proper tests to figure out whether your fine-tuned model [v7qMjy_RxOs@19] beats vanilla, so you can tell a training problem from a dataset problem.
- Calibrate expectations. One honest first-iteration result: the fine-tune scored 75 composite against the base model's 74 [YxXBojdD1as@23], correctly classifying 12 of 25 cases versus 11. Real, tiny, and only visible because the baseline existed.

### 2. Level one — unit tests on structured output

Cheap, deterministic, and run on every change. These catch format regression, which is the most common fine-tuning failure mode.

- Assert the shape: check whether the category is in the allowed categories [a3SMraZWNNs@14], confidence is a float in [0,1], and the label matches the expected value for a known input.
- Score several axes at once rather than one aggregate: correct category, correct severity, valid JSON, whether it used the exact required keys [YxXBojdD1as@5], and usefulness of the free-text field.
- Prefer reference-based metrics where the golden answer exists — exact string matching, semantic similarity, code execution results [a3SMraZWNNs@31], SQL correctness, schema validation — because they automate for free.
- Make structured output a precondition of the architecture: when every step's output is a Pydantic model [Q679gH7oszg@28], every step is testable.
- Cadence: unit tests on every code or prompt change; human-and-model evals weekly or bi-weekly [a3SMraZWNNs@12]; A/B testing for major releases.

### 3. Level two — an LLM judge you have aligned

For free-form output, exact match is useless. But an unaligned judge is worse than no judge, because it manufactures confidence.

- The judge must explain itself: produce detailed critiques, not just scores [a3SMraZWNNs@19]. The critique is what you diagnose and align against.
- The alignment loop: collect model predictions, generate model critiques [a3SMraZWNNs@20], collect human judgments on the same rows, compare, revise the judge prompt, repeat. A worked 10-example pass landed at a 70% agreement score [a3SMraZWNNs@25] on the first attempt.
- To close the gap, export the disagreements and create a meta-prompt [a3SMraZWNNs@26] asking a strong model to rewrite the judge prompt — then re-measure agreement.
- Reduce subjectivity structurally: rather than a Likert rating, break the response into individual facts [vBJF2sy1Pyw@7], score each, and aggregate into a ratio.
- Scale matters more than prompt polish: a 3B judge averaged 51.6/100 [4yNswvhPWCQ@83] across 110 test rows and 14 responses couldn't be parsed into a number at all. A tighter rubric helps — a 1-to-5 scale [4yNswvhPWCQ@78] instead of 0–100 — but bigger models are usually better [4yNswvhPWCQ@78].
- Scoring free-form output at all requires the judge pattern: have a second model score each response on a scale from 0 to 100 [4yNswvhPWCQ@76] against the reference answer.

### 4. Measure reliability, not just accuracy

A single successful run hides consistency problems. For anything agentic or tool-calling, accuracy is the wrong statistic.

- Use pass^k — success required k consecutive times. Empirically pass three is significantly lower than pass one [R78NYFkWH10@14], and the gap *is* the finding.
- The headline result on that benchmark: SFT lifted pass^3 from 37.5% to 75% [R78NYFkWH10@15], with GRPO adding an incremental 2.5 points to 77.5% — beating the closed baseline at lower cost. Note the shape: SFT did almost all the work; RL was the last few points.
- Log the operational metrics too, not only pass/fail: total time, how many tokens did it take [NovNcsKX8AU@57], and overall cost — plus read the trajectory itself.
- Watch for silent failures where neither side errors — an agent that never enables roaming on the agent side [R78NYFkWH10@21] produces no error signal at all.
- Avoid contaminated comparisons: that team chose their closed baseline specifically because the benchmark was released after GPT-4.1 mini [R78NYFkWH10@19].
- Public benchmarks are a vibe check. Build a representative set of evals and tasks [NovNcsKX8AU@83] tagged to axes that map to your product.

### 5. Run it in production, on slices

Post-deployment scoring has a different economics problem: you cannot afford every check on every request.

- Two-tier it: run one high-level check across all traffic, then on the 10–20% that fail run a battery of 5–10 deeper checks [vBJF2sy1Pyw@6] to localise the failure to retrieval, citation, utilisation or hallucination.
- Most quality checks can't be pre-response: checking requires extra time and you don't want to add latency [vBJF2sy1Pyw@3]. Only safety/jailbreak checks are cheap enough to run in parallel with generation.
- Across the lifecycle: golden datasets and manual reviews [uaiq1HvQKoI@43] in development, adversarial end-to-end tests in staging, continuous A/B and re-evaluation after deployment.
- Feed the results back as labels — one of the three jobs of an eval system is to label data for fine-tuning [a3SMraZWNNs@9] or custom test sets, which closes the loop into the next dataset.
- Operating loop for the whole thing: the analyze, measure, improve life cycle [a3SMraZWNNs@9] — categorise failures, convert them to quantitative metrics, then change prompts, data, or weights.
- Don't automate away the reading. A principal engineer is cited as spending three hours per day looking at traces [a3SMraZWNNs@34].

One strategic note on eval design, which matters more for a fine-tune than for a prompt: you can define agent behaviour by showing the evals you ran on it [CvRngaQZQ3Y@11], because teams iterate until they pass. Your eval suite is a specification of the model you are about to train. And before you spend a GPU-weekend, remember binary signals are weak teachers — densifying feedback [CvRngaQZQ3Y@14] by letting the system read and reason over its own traces is often the cheaper win.

## Templates to copy

Five artefacts, each assembled from the corpus's demonstrated practice. Adapt the names; keep the structure.

### Dataset spec — fill this in before opening a trainer

```
BEHAVIOUR (one, narrow):
  Input:   <what arrives>
  Output:  exactly one JSON object with keys: <k1>, <k2>, <k3>
  Label space: <enumerate every allowed value>

SCORABLE BY CODE?  yes / no     # if no, stop and redefine
GRADER:            <string check | schema check | python | model grader>
PASS BAR:          <baseline score> + <delta you need to ship>

SOURCE OF EXAMPLES:
  [ ] production traces (filtered: how?)
  [ ] synthetic from a frontier model (which? licence checked?)
  [ ] enumerated permutations of a bounded output space
  [ ] hand-written

QUALITY BARS (reject a row if it fails any):
  accuracy    - factually correct, no typos, no invented knowledge
  diversity   - covers topics/styles we must not regress on
  complexity  - forces reasoning, not lookup

SPLITS:  train __% / validation __% / test __% (test untouched)
COUNTS:  train ____  val ____  test ____   eval set ____ (50-100 real rows)
```

Three quality axes from [uLrOI65XbDw@5]; scorable-output and label-space-first from [YxXBojdD1as@4]; untouched test split from [YxXBojdD1as@6]; 50–100 real eval rows from [vBJF2sy1Pyw@12].

### Record formats — SFT and preference

```
# SFT (JSONL, one object per line). Prompt is context; only the final
# assistant message is the training target.
{"messages": [
  {"role": "developer", "content": "<the system prompt you will ALSO use at inference>"},
  {"role": "user",      "content": "<the input>"},
  {"role": "assistant", "content": "{\"category\": \"...\", \"severity\": \"...\", \"next_action\": \"...\"}"}
]}

# Preference / DPO: same prompt, two completions.
{"prompt": "<the instruction>",
 "chosen":   "<the preferred answer>",
 "rejected": "<a matched weak answer on the SAME topic>"}

# Local/from-scratch SFT: render to text before tokenising. Prefer a terse
# template over Alpaca boilerplate - fewer tokens per row is cheaper.
### Instruction:
<instruction>
### Input:
<input, omit the header if empty>
### Response:
<target>
```

Trainer learns from the last assistant message [JfaLQqfXqPA@27]; chosen/rejected pairing from [uLrOI65XbDw@4]; matched negatives from [JfaLQqfXqPA@36]; Alpaca template and the terseness argument from [4yNswvhPWCQ@9].

### Run config — a defensible starting point

```
method:               qlora        # lora if VRAM is not the constraint
load_in_4bit:         true         # training-time quantization
epochs:               3            # NOT max_steps. 2-3 to start.
max_steps:            null         # leave unset; setting it to 3 = 3 updates
learning_rate:        <raise until the loss curve explodes, then back off>
optimizer:            adamw        # weight decay + scheduler + warmup
lora_rank:            16           # capacity, not quality
lora_alpha:           32
lora_dropout:         0.0          # 0 for a small experimental run
per_device_batch:     2            # maximise on real GPUs
grad_accum_steps:     4            # effective batch = 8
max_seq_length:       512          # trades against VRAM; need not equal ctx
validation_file:      required     # the only overfit signal you get

STOP CONDITIONS (manual - nothing stops the job for you):
  - validation loss diverges from training loss  -> kill the job
  - training loss trending UP                    -> config is broken
CHANGE EXACTLY ONE THING PER RUN: data spec OR one parameter. Never both.
```

Values from the Unsloth run (rank 16, grad-accum 4, dropout 0) [YxXBojdD1as@12] and rank 16 / alpha 32 [cO6Ly7mIziQ@76]; learning-rate procedure from [uLrOI65XbDw@8]; no automatic overfit stop from [JfaLQqfXqPA@31]; the max_steps trap from [YxXBojdD1as@17].

### Before/after scorecard

```
HELD CONSTANT: eval set, questions, scoring rules, generation settings,
               system prompt (identical to training)

                              base      tuned     delta
valid JSON                    __ / N    __ / N    ____
exact required keys           __ / N    __ / N    ____
correct primary label         __ / N    __ / N    ____
correct secondary label       __ / N    __ / N    ____
free-text usefulness (judge)  __ / 5    __ / 5    ____
composite                     ____      ____      ____

pass^1                        ____      ____      ____
pass^3                        ____      ____      ____   # the real number
tokens / task                 ____      ____      ____
latency p50 / p95             ____      ____      ____
cost / task                   ____      ____      ____

REGRESSION CHECK: score the tuned model on 20 rows of behaviour that were
NOT in the training data. Anything you did not include might regress.

ATTRIBUTION: tuned < base  -> dataset problem or training problem?
             Re-run the base eval first. If base moved, the harness moved.
```

Identical-harness rule and multi-axis scoring from [YxXBojdD1as@4]; pass^k from [R78NYFkWH10@14]; token/time/cost logging from [NovNcsKX8AU@57]; regression risk from [JfaLQqfXqPA@36].

### Judge-alignment meta-prompt

```
You are improving an evaluator prompt.

Below are cases where a human grader and the current LLM judge disagreed.
Each row has: INPUT, MODEL OUTPUT, HUMAN LABEL + HUMAN CRITIQUE,
JUDGE LABEL + JUDGE CRITIQUE.

<paste disagreement rows>

CURRENT JUDGE PROMPT:
<paste it>

Tasks:
1. For each disagreement, state in one line why the judge diverged from the
   human. Name the specific criterion that was missing, vague, or misweighted.
2. Rewrite the judge prompt so those cases would be graded as the human did,
   WITHOUT special-casing these examples.
3. The rewritten judge must output a detailed critique and then a label -
   critique first, label last. Use a 1-5 rubric with each level defined.
4. List any case where you believe the HUMAN label is wrong, and why.

Return only the rewritten judge prompt and the numbered diagnosis list.

# Then: re-run on the same rows and recompute the human/judge agreement
# score. Ship the judge only when agreement stops improving.
```

Meta-prompt method and the agreement-score loop from [a3SMraZWNNs@26] and [a3SMraZWNNs@20]; critique-before-label from [a3SMraZWNNs@19]; 1–5 rubric from [4yNswvhPWCQ@78].

## Where the corpus disagrees

Four live disagreements. Each is a real engineering fork, and the resolution depends on your constraints — not on who is right.

### Is fine-tuning a last resort or a strategy?

The strict-escalation position puts it fourth: prompt, RAG, agentic loop, then fine-tune, because it costs at least a weekend [v7qMjy_RxOs@8]. The enterprise position is the inverse: frontier models are commodity, so proprietary data plus fine-tuning is the competitive edge you should plan for. And a third argument sidesteps technical merit entirely — the top two enterprise reasons for open models are control and customizability [uLrOI65XbDw@2], which holds even when prompting is technically good enough. **Resolution:** the ladder is right for one task; the strategic position is right for a portfolio. Note the corpus's own counter-evidence: a complete production RAG copilot gets built with no fine-tuning step anywhere, relying on hybrid retrieval and a grounding validator that rejects unretrieved citations [qF5il_9IwME@143].

### Full fine-tuning or PEFT by default?

One camp calls full-parameter training not a practical choice [cO6Ly7mIziQ@15] and makes LoRA/QLoRA the default even for modest setups. The from-scratch camp fully fine-tunes a 355M GPT-2 end to end and, when quality was poor, reached for a bigger base rather than an adapter — one model size up [4yNswvhPWCQ@38]. **Resolution:** the crossover is model size. Below a few hundred million parameters full fine-tuning is simpler and the memory argument evaporates; above a billion, PEFT wins on iteration speed, and full fine-tuning's only remaining claim is peak performance.

### Expensive judge or cheap judge?

One side: use the most powerful model that you can afford [a3SMraZWNNs@18], explicitly not a mini model, because you must trust the judge's reasoning. The other: you don't want to use an expensive model like GPT-4 [vBJF2sy1Pyw@5] for evaluation at scale — if judging costs more than generating, hire a human. **Resolution:** these are different jobs. Use the strongest model for the small aligned judge that grades your eval set and for the meta-prompt rewrite; use cheap models for the high-level check over all production traffic, escalating only the failing slice. The trace-mining result supports this: a harness-engineered open model matched Opus-level trace judging at far lower cost [CvRngaQZQ3Y@7].

### Is passing the task metric enough to accept a training trajectory?

The judge-the-final-answer approach treats a good score as sufficient evidence. The tool-calling case study says explicitly that it is not: trajectories can pass while hallucinating a lot [R78NYFkWH10@22], or succeed by luck, and that noise destabilises the later RL stage. **Resolution:** outcome scoring is adequate for accepting an *answer* and inadequate for accepting a *training row*. Any trajectory entering a dataset needs process-level checks, not just an outcome check. The same logic kills chain-of-thought optimisation: policing the reasoning reduces bad behaviour slightly while dropping your ability to catch it to roughly zero [JfaLQqfXqPA@58].

## Sources

Twenty talks, grouped by what they contributed. Three are included for completeness and contributed nothing to this topic — said plainly rather than quietly dropped.

### Method and mechanics

1. [RFT, DPO, SFT: Fine-tuning with OpenAI — Ilan Bigio, OpenAI](https://www.youtube.com/watch?v=JfaLQqfXqPA) [JfaLQqfXqPA] — the most granular build guide here: live SFT/DPO/RFT demos with real metrics, sample-count rules of thumb, grader design, platform mechanics, and two honest failures.
2. [Everything you need to know about Fine-tuning and Merging LLMs: Maxime Labonne](https://www.youtube.com/watch?v=uLrOI65XbDw) [uLrOI65XbDw] — SFT/DPO data formats, the three-feature sample recipe, the curation pipeline, full/LoRA/QLoRA trade-offs, key hyperparameters, and the deep dive on merging.
3. [LoRA & QLoRA Explained Simply | Full Fine-Tuning vs PEFT + Intuition + Practical (Complete Guide)](https://www.youtube.com/watch?v=cO6Ly7mIziQ) [cO6Ly7mIziQ] — matrix rank as the intuition under LoRA, the parameter arithmetic, the PEFT landscape, and a hands-on TinyLlama PEFT script.
4. [Build an LLM from Scratch 7: Instruction Finetuning](https://www.youtube.com/watch?v=4yNswvhPWCQ) [4yNswvhPWCQ] — the mechanics nobody else shows: Alpaca formatting, batch-level padding, loss masking with ignore_index, epoch/overfit behaviour, and an Ollama-based judge pipeline.

### Hands-on runs

5. [Fine-Tuning a Model: The Easiest Way to Start (Unsloth Tutorial)](https://www.youtube.com/watch?v=YxXBojdD1as) [YxXBojdD1as] — the full loop end to end: baseline before training, the hyperparameters actually set, a real max-steps mistake and its loss-chart symptom, before/after scores, GGUF export, and calling the result from Python.
6. [The Honest Guide To Fine-Tuning Local AI In 2026](https://www.youtube.com/watch?v=v7qMjy_RxOs) [v7qMjy_RxOs] — the decision flowchart, the five-step pipeline, token/VRAM/wall-clock benchmarks, and a blunt hardware verdict across Nvidia, AMD and Apple Silicon.
7. [Fine-Tuning LLMs for Real-World Tool Calling: Lessons from tau2-bench](https://www.youtube.com/watch?v=R78NYFkWH10) [R78NYFkWH10] — the only production-grade SFT→GRPO workflow in the corpus, with reward shaping, trajectory filtering counts, pass^k numbers, and a reward-hacking post-mortem.

### Deciding and sequencing

8. [Improving Agents is a Data Mining Problem — Vivek Trivedy, LangChain](https://www.youtube.com/watch?v=CvRngaQZQ3Y) [CvRngaQZQ3Y] — fine-tuning as step two of a trace-mining pipeline, the harness→fine-tune→harness sandwich, and the token-cost-versus-hardware-cost economics.
9. [The Never Ending Lore of Harness | Vivek Trivedy (Product Lead, Langchain)](https://www.youtube.com/watch?v=NovNcsKX8AU) [NovNcsKX8AU] — defines model-plus-harness, gives the sub-agent context budget, and states plainly when RL/fine-tuning beats more context engineering.
10. [AI System Design: From Idea to Production - Apoorva Joshi, MongoDB](https://www.youtube.com/watch?v=T0HhO4YtTfE) [T0HhO4YtTfE] — frames fine-tuning as one design-pattern option and names the exact trigger: behavioural failures, or a domain ceiling prompting can't clear.

### Scoring

11. [How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)](https://www.youtube.com/watch?v=a3SMraZWNNs) [a3SMraZWNNs] — the three-level eval ladder, the judge-alignment workflow with a real agreement score, the meta-prompt fix, and four named eval mistakes.
12. [The challenges in using LLM-as-a-Judge - Sourabh Agrawal | Vector Space Talk #013](https://www.youtube.com/watch?v=vBJF2sy1Pyw) [vBJF2sy1Pyw] — judge economics at scale, root-cause analysis on the failing slice, fact-decomposition scoring, and eval dataset sizing.

### Shipping and operations

13. [Exploring MLOps and LLMOps: Architectures and Best Practices](https://www.youtube.com/watch?v=1jvxxa7tdjw) [1jvxxa7tdjw] — deploy-code-not-models, champion/challenger promotion, a concrete drift-based retrain threshold, data-volume thresholds per customisation step, and the licence cross-contamination trap.
14. [GenAI in Production: How RAG & AI Agents Are Shaping the Future](https://www.youtube.com/watch?v=uaiq1HvQKoI) [uaiq1HvQKoI] — the garbage-in-garbage-out warning specific to fine-tuned models inheriting training-data errors, plus golden datasets, canary/blue-green rollout, and feedback loops.
15. [How I Build and Ship Custom AI Solutions for Clients](https://www.youtube.com/watch?v=Q679gH7oszg) [Q679gH7oszg] — the accuracy ladder clients should expect, structured-output-first eval design, the bare-metal deployment and security baseline, and the Langfuse/Sentry monitoring pair.
16. [Build a Full-Stack GenAI Project in 4 Hours (FastAPI, React, Supabase)](https://www.youtube.com/watch?v=qF5il_9IwME) [qF5il_9IwME] — the counter-example: a production-grade document copilot built entirely with hybrid retrieval, agentic retrieval and a grounding validator, with no fine-tuning anywhere.
17. [Build a Complete End-to-End GenAI Project in 3 Hours](https://www.youtube.com/watch?v=E8zpgNPx8jE) [E8zpgNPx8jE] — data-pipeline craft upstream of any training set (dedup-on-conflict, two-stage ingestion, correct source timestamps) but no fine-tuning content.

### Adjacent and off-topic

18. [GenAI in Production: How AI Is Transforming the Entire Filmmaking Pipeline](https://www.youtube.com/watch?v=NRQVTxl14x8) [NRQVTxl14x8] — one production LoRA use case (per-actor digital doubles) and a real cost/control pendulum, but no training detail; the LoRA work is attributed to a team not on the call.
19. [Learn Design Systems: Figma Variables, Components, AI, and More](https://www.youtube.com/watch?v=lsgVRCJ4fko) [lsgVRCJ4fko] — no LLM fine-tuning content; a Figma design-systems tutorial that uses "training" in the design-tooling sense. Contributed nothing to this guide.
20. [Design with AI - Full Guide (Tools, Workflows, Design Systems)](https://www.youtube.com/watch?v=r-MPiu0E4W4) [r-MPiu0E4W4] — no LLM fine-tuning content; AI-assisted UI/UX design tooling. Contributed nothing to this guide.

## Gaps

What twenty talks do not tell you. These are the questions a staff engineer would have to answer elsewhere — and the reason this guide gives you procedures where it cannot give you numbers.

- **No dollar costs, anywhere.** Not for hosted SFT/DPO/RFT jobs, not for cloud GPU hours, not for a self-managed cluster. Every cost claim in the corpus is relative ("4–5× more expensive", "10× cheaper") or in wall-clock minutes. You cannot size a budget from this material.
- **No break-even analysis for the token-cost-versus-hardware-cost switch.** The economics are asserted at high volume but never quantified, so there is no request-per-second threshold at which owning GPUs starts to pay.
- **No method for choosing LoRA rank or alpha.** Rank 16 / alpha 32 appears twice as a demo value, and rank is correctly described as capacity — but no scaling rule, no search procedure, no relationship to dataset size or task difficulty.
- **No concrete learning-rate schedule, warmup, or weight-decay values.** AdamW is named and never tuned on screen. "Raise it until the loss explodes" is the only procedure offered.
- **No GRPO hyperparameters.** The RL case study discloses its reward components but not group size, KL penalty coefficient, learning rate, training duration or cost.
- **No exact thresholds for dedup or quality filtering.** Exact and fuzzy dedup are recommended, and clustering tools are named but never demonstrated; there are no similarity cut-offs or filter-pass rates.
- **No distributed or multi-GPU training.** Every demo runs on one machine — a laptop, a single Colab GPU, or one H100 node. Sharding, FSDP/DeepSpeed configuration and multi-node failure modes are absent.
- **No catastrophic-forgetting or regression-suite methodology specific to a fine-tuned checkpoint.** Regression risk is flagged ("anything you don't include might regress") but no one builds the suite that would detect it.
- **No safety or red-team evaluation of a fine-tuned model.** Preference tuning is mentioned as a safety lever; nobody tests whether a fine-tune degraded existing refusals.
- **No multimodal or vision-language fine-tuning.** All worked examples are text-only. Image fine-tuning is mentioned in passing on the hosted platform and never demonstrated.
- **No continued / domain-adaptive pretraining as a step before instruction tuning.** Token volumes for it are quoted; the procedure is not.
- **No framework comparison.** TRL, Axolotl, LLaMA Factory, Unsloth, PEFT and Mosaic AI are all named; no criteria are offered for choosing among them, and no two are benchmarked against each other.
- **No roadmap for multi-turn RFT.** The single-turn limitation is stated as current fact with no timeline, and no workaround for feeding tool results back mid-episode.
- **No licensing guidance for redistributing a fine-tuned open-weight model.** The input-side contamination risk is flagged; the output-side distribution question is untouched.
- **No absolute benchmark numbers for merged models.** The merging results cite leaderboard rank, never downstream task scores against frontier models — so you cannot tell whether merging is a real alternative to training or a leaderboard artefact.

---

The Fine-Tuning Pipeline — a field guide from the transcript corpus. 20 source videos · 1,482 transcript chunks read in full · 5 parallel extraction passes · compiled 2026-10-22. Every claim above resolves to a chunk you can open; where the corpus is silent, the Gaps section says so instead of guessing.
