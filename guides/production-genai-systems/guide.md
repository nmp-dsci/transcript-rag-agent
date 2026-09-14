---
title: Production GenAI Systems
topic: "production generative AI systems: taking LLM and agent applications from pilot to production"
compiled_at: 2026-09-14
videos: 17
chunks: 737
extraction_passes: 6
sources:
  - video_id: T0HhO4YtTfE
    title: "AI System Design: From Idea to Production - Apoorva Joshi, MongoDB"
  - video_id: vSx5IULvBns
    title: "Always-on agents run production without the on-call tax — Justin Smith, Resolve AI"
  - video_id: vW8wLsb3Nnc
    title: "POC to PROD: Hard Lessons from 200+ Enterprise GenAI Deployments - Randall Hunt, Caylent"
  - video_id: _2tZaDs-w5s
    title: "RAG at scale: production ready GenAI apps with Azure AI Search"
  - video_id: ow1we5PzK-o
    title: "The Multi-Agent Architecture That Actually Ships — Luke Alvoeiro, Factory"
  - video_id: ObTPqBGsEbA
    title: "The Production AI Playbook: Deploying Agents at Enterprise Scale — Sandipan Bhaumik, Databricks"
  - video_id: tLK5jyhQOgA
    title: "Make GenAI Production-Ready With Kubernetes Patterns - Roland Huss, Red Hat & Bilgin Ibryam, Diagrid"
  - video_id: 1LhvqZvDT5w
    title: "From GenAI Pilots to Production - Nikita Kozodoi"
  - video_id: 1jvxxa7tdjw
    title: "Exploring MLOps and LLMOps: Architectures and Best Practices"
  - video_id: sk6HBdmVmL8
    title: "Omnigent: Open-Source Meta-Harness for AI Agents | Matei Zaharia"
  - video_id: E8zpgNPx8jE
    title: "Build a Complete End-to-End GenAI Project in 3 Hours"
  - video_id: Q679gH7oszg
    title: "How I Build and Ship Custom AI Solutions for Clients"
  - video_id: cvPEiPt7HXo
    title: "Large Language Model Operations (LLMOps) Explained"
  - video_id: hBzUokVYQkI
    title: "AI Infrastructure Explained (GPUs, vLLM, and LLM-D)"
  - video_id: 7YYVgH0_9CA
    title: "Scalable GenAI in Production - Kamelia Aryafar"
  - video_id: uaiq1HvQKoI
    title: "GenAI in Production: How RAG & AI Agents Are Shaping the Future"
  - video_id: mvleESOUTRw
    title: "Production-Grade AI Project Tutorial – Build & Deploy"
---

# Production GenAI Systems

Seventeen practitioners — Databricks, AWS, MongoDB, Red Hat, Factory, Caylent, Azure AI Search, IBM, Resolve AI, and two agencies that ship client work every fortnight — on the distance between a pilot that impressed the room and a system that survives Monday morning. Everything below is traceable to a chunk you can open.

*17 source videos · 737 transcript chunks read in full · 6 parallel extraction passes · compiled 2026-09-14*

## The thesis: it is never the model

Every talk in this corpus was given by someone who has watched a GenAI pilot die. None of them blame the model. A two-year agency practice puts it flatly: when a project fails, it is the execution, not the model [Q679gH7oszg@2]. A Netflix- and Google-scale practitioner draws the same architecture diagram and notes that the model is a very small part of an end-to-end architecture [7YYVgH0_9CA@2]. A consultant who watched 200+ enterprise deployments opens with the warning that generative AI is not the magical pill [vW8wLsb3Nnc@0].

The pattern that kills pilots is legible and repeatable: pick a model, build features against predictable data, demo it, and then discover in production that nobody can check what the AI actually did [ObTPqBGsEbA@13]. One retail bank spent roughly $85K over six months on a POC that did not succeed [ObTPqBGsEbA@23], and no one could say why. Demos are, in the corpus's own phrase, the happy path [7YYVgH0_9CA@10]: they are designed to impress [uaiq1HvQKoI@45], and impressing is not a system property.

> "I like to leave you with **stop building demos**. We are beyond demos. We have many cool demos. Start building systems." — Kamelia Aryafar [7YYVgH0_9CA@24]

Three moves separate a system from a demo. Each one inverts a default that feels productive and is not.

**Spec before stack.** The eval set *is* the specification — evaluation is basically specification for your AI system [ObTPqBGsEbA@6]. In the successful redo of that failed bank chatbot, the team selected the model in week seven of an eight-week POC [ObTPqBGsEbA@24], and spent weeks one and two building the evaluation layer. Specs are the new code [T0HhO4YtTfE@1].

**Trace before scale.** Observability is not a dashboard you add later; every step in an agentic pipeline needs its own logging service attached [1LhvqZvDT5w@51]. Without traces a disputed AI decision cannot be reconstructed at all, and teams default to refunding the customer. The closing line of the framework talk holds: you can't improve what you can't measure [T0HhO4YtTfE@22].

**Price before ship.** Cost and latency are design inputs, not post-launch cleanup. Applications quietly die because they don't take cost into account and capacity estimates [7YYVgH0_9CA@6], defaulting to the biggest reasoning model for every call. The market verdict is blunt: if you are slower and more expensive, you will not be used [vW8wLsb3Nnc@12].

## Pilot to production in six stages

Two talks in the corpus offer full frameworks — a four-phase design loop from MongoDB and a five-pillar enterprise playbook from Databricks. They agree more than they differ. What follows merges them and folds in the field reports from everyone else. Run the stages in order the first time; after that you will be iterating across all six at once.

### 01 — Quantify the problem. Forbid the solution.

A business problem that names an architecture has already skipped the thinking. Write it so that an agent, a control flow, and a logistic regression all remain live options.

- Name the user, state the current state, quantify the pain — and stop there. A good problem statement should not prescribe what the system is going to be [T0HhO4YtTfE@3], whether agent, multi-agent, or otherwise.
- Put a number on the baseline. The worked example: reviewers spend two days per claim, four times the industry standard for non-urgent cases [T0HhO4YtTfE@3] and twelve times for urgent ones.
- Collect the constraints that will veto your design before you draw it: data residency, approved vendors, and the rule that denial decisions must be reviewed by a human reviewer [T0HhO4YtTfE@4].
- Classify the role of AI on three axes — is AI critical to your product or complementary [T0HhO4YtTfE@5], reactive or proactive, and what maximum autonomy the constraints permit.
- Write one SMART success metric tied to the baseline: reduce the average processing time for urgent claim review requests [T0HhO4YtTfE@6] from two days to one hour within 90 days of launch.
- Treat "we can't tell what good looks like" as a stop signal. The agency red flag is a project where there is no real way to track the impact [Q679gH7oszg@6] — and the standing advice is to always pick the quick wins over the moonshots [Q679gH7oszg@6].
- Reset the accuracy expectation in the first meeting. Your first build typically gets to 70 or 80% accuracy [Q679gH7oszg@10]; iteration takes it to about 90. Clients arriving from deterministic software expect magic from day one [Q679gH7oszg@7].

### 02 — Build the eval before you choose the model

Model choice is the most reversible decision in the project and the one teams make first. Invert it. The eval set outlives every model you will run behind it.

- Define success in hard numbers before writing code: the bank chatbot targeted 60% of user queries handled at around 85% accuracy [ObTPqBGsEbA@24], plus latency and operational targets.
- Build the golden set from what humans actually answered. Talk with the domain experts and find what is actually happening [ObTPqBGsEbA@7] on the ground — including the gray areas. That project started from 200 real agent-answered cases.
- Expect the set to grow forever: your evaluation data set is a living system [ObTPqBGsEbA@25], and the bigger it grows the better the system gets.
- Start it on day one, not after the pipeline works. At AWS's GenAI Innovation Center, one of the very first things we try to do is to actually build [1LhvqZvDT5w@25] an evaluation rubric.
- The cheapest first eval is the vibe check you already ran, written down and varied: 20 minutes later, you do have some form of eval set [vW8wLsb3Nnc@15] you can begin running.
- This is the layer where you earn the right to claim the system works — this is where we prove that the system is robust and not just a vibe check [vW8wLsb3Nnc@8] that got lucky on one prompt.

### 03 — Design the simplest thing the flow allows

Map how one real request moves end to end, then choose the least machinery that satisfies it. Agents are a design pattern, not a starting position.

- Don't let hype or a coding agent pick the architecture — you risk ending up with an over-engineered system by doing this [T0HhO4YtTfE@10]. Start simple, evaluate, find gaps, iterate.
- Trace the request first: map out how an insurance claim would flow through the system [T0HhO4YtTfE@10], because the flow informs the architecture and not the reverse.
- Match the technique to the data source, not to fashion: vector search with metadata pre-filtering or hybrid search [T0HhO4YtTfE@9] for policy documents full of codes; exact match on patient ID for structured history.
- Some problems are not GenAI problems. For demand forecasting, you don't need to run Claude Opus to predict [1LhvqZvDT5w@9] a time series — and the standing rule is that you should always pick up the simplest tool possible [1LhvqZvDT5w@45].
- What you commit to is the contract, not the wiring. Inputs and outputs are the most fundamental part [vW8wLsb3Nnc@8]; the architecture and the specific LLM are incidental and will change.
- Freeze the surrounding stack so the model can churn. One agency has run FastAPI, Celery, Postgres and Redis for two years across client projects — "We just swap out the model" [Q679gH7oszg@50] — which also keeps coding agents inside a scaffold they can't wander out of.

### 04 — Pay the data tax up front

This is the majority of the project and almost nobody budgets for it. Humans route around bad data silently; agents do not.

- Budget the time honestly: on enterprise agent projects 60% of project time [ObTPqBGsEbA@14] goes into the data foundation — split between the data that answers questions and the data that traces behaviour.
- Data built for humans does not survive contact with agents: agents will find it wrong and give you the wrong answer confidently [ObTPqBGsEbA@14], with no flag raised. With GenAI, garbage in, garbage out really bites [uaiq1HvQKoI@46].
- Set each pipeline's cadence from the source's real update frequency — guidelines annually, policies quarterly, claims hourly — or the system works off stale information [T0HhO4YtTfE@8].
- The canonical production incident in this corpus is a freshness bug: after a rate-policy update, the new policy document was not updated in the vector database [ObTPqBGsEbA@28], the embeddings never came through, and the bot kept answering from the old policy. It surfaced as a thumbs-down trend, not an alert.
- Keeping the index in sync with new, edited and archived documents without full reindexing is an ongoing operational job [uaiq1HvQKoI@25].
- Prefer a pipeline that detects deltas itself. An integrated ingestion pipeline can automatically track changes so it's not a one-shot thing [_2tZaDs-w5s@19], paying incremental cost rather than reindexing everything.

### 05 — Instrument every step before you scale any of it

Three gaps sink pilots: you can't see what the AI did, you have no metric that matters to the business, and no one owns the failure. The first is the one you fix with code.

- Name the gap out loud. The first one is the observability gap [ObTPqBGsEbA@2] — and it is followed by the evaluation gap and the governance gap.
- Trace at the granularity of the decision. One overdraft-fee request should show intent classification, the account API call, the policy-document retrieval from the RAG vector database [ObTPqBGsEbA@12], the reasoning step and the final guardrail check.
- Watch behaviour, not just answers. An agent can return the right answer while making three calls to the database to find that answer [ObTPqBGsEbA@10] — invisible in a demo, expensive at thousands of queries a day.
- Log every request and response automatically; that logging is what later feeds monitoring, retraining and debugging [1jvxxa7tdjw@12].
- Track the human's disagreement rate as your live quality signal — how often a human reviewer overrides the AI verdict [T0HhO4YtTfE@19]. Alongside it, watch average token cost, token usage, and average number of turns [T0HhO4YtTfE@18].
- Two tools cover most of it in practice: Langfuse for all LLM traces [Q679gH7oszg@40] plus Sentry piped into Slack for application errors. A Sentry trace can be copied as markdown and pasted straight into the coding agent [Q679gH7oszg@43].
- Wire detection to action. With online monitoring in place, a duplicate or failing tool call can trigger a fallback that will retry three times, not more than three times [ObTPqBGsEbA@14], then escalate to a human.

### 06 — Harden, then roll out slowly

Guardrails are a runtime engineering requirement, and rollout is a separate discipline from building. Both exist because LLM systems fail differently from the software around them.

- The reason guardrails exist at all: unlike traditional software, LLM-based systems are probabilistic and can produce outputs that are unexpected, incorrect, or even harmful [T0HhO4YtTfE@17].
- One layer is never enough. A managed guardrail service alone is usually not enough — you need multiple layers of defense [1LhvqZvDT5w@11]: prompt instructions, a classifier fleet, and observability for what slips through.
- Put classifiers on both sides of the model. They must also sit between the LLM response and the user [1LhvqZvDT5w@16], since web-search or hallucinated output can itself be harmful.
- Red-team before launch and turn it into a number: one system passed only five or six out of 20 stress scenarios [1LhvqZvDT5w@18], which is the signal that guardrails aren't done.
- Cheap deterministic checks catch expensive things: NER-based PII pre-validation meant the team already detected 47 PII breaches during the testing phase [ObTPqBGsEbA@21].
- Version prompts like code. You have to treat prompt versioning as change management [ObTPqBGsEbA@21] in an enterprise-grade solution — not a casual commit.
- Roll out behind a switch: use versioning with blue-green or canary deployments [uaiq1HvQKoI@43], reindexing in parallel and cutting over only after validation. On the serving side, assign a challenger alias [1jvxxa7tdjw@15] while the champion keeps taking traffic.
- Rehearse the incident loop: detect using your eval dashboard [ObTPqBGsEbA@29], diagnose with tracing, contain by rolling back the prompt version, fix from the test-case library, then add the case to the eval suite so it is caught automatically next time.

## The evaluation stack

Evaluation in this corpus is three layers deep, runs continuously, and has a CI budget. Treat any one of those three as optional and you have a vibe check with a dashboard.

- **Layer 1 — deterministic, and first.** Formats, regex, classic NER and PII models. The first layer is deterministic [ObTPqBGsEbA@8] and it is cheap, so run it before you spend a token on anything smarter.
- **Layer 2 — semantic judges.** LLM-as-judge for groundedness, safety and relevance. Every enterprise needs a very scalable framework of LLM judges [7YYVgH0_9CA@13], because routing everything to humans stopped scaling long ago.
- **Layer 3 — behavioural, and skipped.** Are tool calls correct, duplicated, or looping? This is the layer most teams never build, and the one that catches three database calls where one would do [ObTPqBGsEbA@10].
- **Humans are the gold standard, judges are the loop.** Human experts scoring against a rubric set the standard; an LLM judge reusing the same rubric [1LhvqZvDT5w@25] gives you fast, cheap iteration on every prompt change.
- **Aligning a judge costs examples.** Getting an LLM judge to agree with humans took one engagement almost 50 examples of how different human judges judge [1LhvqZvDT5w@29], cached in the system prompt to control cost.
- **Score components, not just answers.** Evaluate both the components and the system as a whole [1jvxxa7tdjw@45]: did retrieval fetch the right context, and was the response faithful to it?
- **Measure guardrail compliance as a metric.** Turn the boundary into a number — a missing-citation rate [T0HhO4YtTfE@18] for output compliance, a faithfulness score for grounding, a rejection rate for inputs.
- **Never stop at launch.** Models drift. It's not okay to just monitor things and evaluate at the launch [7YYVgH0_9CA@9] — teams that evaluate once lose track of where quality actually stands.
- **Give CI a cost ceiling.** Behavioural evals get expensive past 300–500 rows. Run a subset on each change and only do the full test when you merge to the main branch [ObTPqBGsEbA@33].
- **Re-run your own set on every model upgrade.** Published leaderboards are not evidence about your system. Keep more than one provider live, because you cannot really rely on one single model [ObTPqBGsEbA@22].

## The retrieval floor

RAG is a single-turn grounded conversation [7YYVgH0_9CA@5] — the foundation the agent layer stands on. Retrieval quality is therefore the ceiling on everything above it: your app works when they ask a question and they get the answer they're looking for [_2tZaDs-w5s@10]. These are the levers that move that number, roughly in order of payoff.

- **Scope before you rank.** The single most effective lever is narrowing the candidate set: the other dimension of getting quality out of the system is to narrow the data set [_2tZaDs-w5s@13], then do the ranking tricks on top. Filters stay fast even if you have hundreds of millions of documents [_2tZaDs-w5s@9].
- **Two stages, always.** The first stage is recall oriented and uses vectors and keywords [_2tZaDs-w5s@11]; the second reranks the small candidate set. Benchmarked on one query set, quality runs BM25 < vectors < fusion < fusion+rerank, with better results just out of the box when reranking is enabled [_2tZaDs-w5s@12].
- **Rerankers are cross-encoders, and that's the point.** These rerankers are cross encoders [_2tZaDs-w5s@12] — they see query and document together, which is why they rank better and can't run over the whole corpus. Budget about 100 milliseconds give or take for a model like this [_2tZaDs-w5s@13]; in an interactive app that hides behind the LLM call.
- **Decouple candidates from returns.** Separate how many candidates you want from how many you want to return [_2tZaDs-w5s@10]. Recall depth and prompt size are different budgets.
- **Quantize, then oversample back.** Single-bit quantization is a 32× density gain that still lands in the low to mid 90% of the original performance [_2tZaDs-w5s@17]. Keep full-precision vectors on the side so you can do oversampling where we query at the quantized [_2tZaDs-w5s@18] index and rerank at full precision.
- **Embeddings are not a query engine.** Embeddings alone do not a great query system make [vW8wLsb3Nnc@12] — you still need OpenSearch or Postgres underneath for facets and filters.
- **Chunk with overlap, then tune both.** A workable default is about a thousand words [mvleESOUTRw@71] per chunk with a 200-word overlap, because without overlap [mvleESOUTRw@73] a narrative splits mid-thought. Then tune: if your chunks are too small, they might lack sufficient context [uaiq1HvQKoI@24]; too large adds noise.
- **Return citations or don't ship.** Good RAG systems will also provide citations or links [uaiq1HvQKoI@21] back to the source. They are also your feedback channel: let reviewers flag any irrelevant citations [T0HhO4YtTfE@15] and you get a hallucination signal for free.
- **The payoff is measurable.** On one engagement with scattered, unversioned documentation, a RAG retrieval layer meant 60% of the time that they spent initially was reduced [uaiq1HvQKoI@11] for complex queries.

## Agent architectures that survive contact

The corpus contains exactly one multi-agent system with published production numbers — Factory's "missions", whose longest mission ran for 16 days [ow1we5PzK-o@7]. Its design choices are the most concrete evidence here about what makes long-horizon agent work hold together, and they are mostly about structure, not intelligence. The stated bottleneck is not model capability: the bottleneck in software engineering nowadays is not intelligence [ow1we5PzK-o@0], it's human attention.

- **Write the validation contract before the code.** A contract written during planning before any code [ow1we5PzK-o@5] defines correctness independently of implementation — hundreds of assertions on a complex project. The alternative is worthless: tests written after implementation don't catch bugs. They confirm decisions [ow1we5PzK-o@5].
- **Three roles, one active at a time.** Missions uses a three-role architecture. There's orchestrator, there's workers [ow1we5PzK-o@3] and validators. Each worker gets clean context and commits via Git so the next inherits a working codebase.
- **Validate behaviour, not just code.** Two validators run per milestone: scrutiny (tests, types, lints, per-feature review agents) and a user-testing validator that spawns the application and interacts with it through computer use [ow1we5PzK-o@6]. Most wall clock time is actually spent waiting for this like real world execution [ow1we5PzK-o@6], not generating tokens.
- **Make validation adversarial on purpose.** Neither validator has seen the code before [ow1we5PzK-o@7], so neither is invested in it. Go further and let validation might use a different model provider entirely [ow1we5PzK-o@11], so the check isn't biased by the same training data.
- **Handoffs in writing, not in memory.** A finishing worker files what was completed, what was left, which commands ran and their exit codes. The system self-heals at milestone boundaries by forcing them to write it down [ow1we5PzK-o@7].
- **Serial beats parallel for mutating work.** Ten agents is not ten times throughput: agents conflict. They step on each other's changes. They duplicate work [ow1we5PzK-o@8]. Missions runs features serially and parallelizes only read-only operations. It seems slower on paper, but the error rate drops dramatically [ow1we5PzK-o@9].
- **Keep orchestration in prompts.** Almost all of the orchestration logic is defined in prompts and skills [ow1we5PzK-o@13] — about 700 lines of text — so the system improves with each model release instead of being obsoleted by it. The structure is load-bearing enough to run missions very very successfully even using open-weight models [ow1we5PzK-o@11].
- **Fresh context per task, not one long session.** When the context window gets bigger, the model gets stupider [sk6HBdmVmL8@6] — and you pay for the whole context every turn. Kick off sub-agents so they each have like a fresh start [sk6HBdmVmL8@6].
- **Policies that read session state.** Static allow/deny lists only ask whether does the tool call it's making match this pattern [sk6HBdmVmL8@10]. Risk lives in history: an agent that just pulled a new npm package and read 10,000 company docs, and now it's trying to send an email [sk6HBdmVmL8@11] is suspicious even though sending email is normally fine.
- **Risk-score toward a human.** As you take actions, your score goes up [sk6HBdmVmL8@12], and past a threshold the session needs supervision. Separately, narrow down what it can do based on that initial prompt [sk6HBdmVmL8@12] — then any out-of-scope attempt is an injection signal.
- **Cap the spend inside the agent.** Agents can spend like $1,000 by mistake [sk6HBdmVmL8@13] generating a giant log file. Tell the agent to ask me for permission after like every $10 spent [sk6HBdmVmL8@13].
- **Failure handling is the boring 80%.** Production agent pipelines need certain logic to do retry and exponential [1LhvqZvDT5w@52] backoff for throttling, plus session-memory retrieval to resume mid-conversation. Reliable tool use and error handling [uaiq1HvQKoI@33] is a core agent-building problem, not an edge case.
- **Point agents at the ops long tail.** A survey puts 70% of the time from an engineer [vSx5IULvBns@1] into running already-shipped code. Background agents absorb it: trigger them on a schedule [vSx5IULvBns@11], on a deploy event, or by message — and let the agent choose its own cadence, since none of this is hard-coded in [vSx5IULvBns@20].
- **The expensive part is the environment.** Model capability isn't the wall; truly understanding your environment and the way that your services interact [vSx5IULvBns@6] is. The cost of production AI ops is not just in the task execution, it's in the environment complexity [vSx5IULvBns@25].

## Serving economics: the three numbers

Below the application layer the constraints are physical and unforgiving. Every workload is sized by three numbers that size every AI workload [hBzUokVYQkI@11]: compute, capacity, and bandwidth. And set the target before the architecture, because Latency is a product decision [7YYVgH0_9CA@11], not an engineering one.

- **Memory caps concurrency, not compute.** Every batched user needs a KV-cache scratch pad alongside the fixed weights, so it's the memory that decides how many people one GPU can actually serve [hBzUokVYQkI@28]. Which is why teams end up buying far more GPUs than the math can actually calls for [hBzUokVYQkI@28].
- **Cache-aware routing is the cheapest 3×.** Round-robin treats servers as interchangeable, so it throws away saved work that was perfectly good [hBzUokVYQkI@33]. Routing a follow-up back to the server holding the conversation cache gives you around three times the throughput and a first response that is twice as fast [hBzUokVYQkI@35].
- **Split prefill from decode.** Prefill is compute-bound and parallelizable; decode is sequential and memory-bound. You can split these two phases into different group of pods [tLK5jyhQOgA@21] and hand off the KV work, for up to 70% more tokens per second on the same hardware [hBzUokVYQkI@36].
- **Start with the smallest change.** Most people start with the first path because it's the smallest change [hBzUokVYQkI@39]: one pool of identical vLLM servers with cache-aware routing on and nothing else touched.
- **Prompt caching, and where you put the volatile bits.** A cacheed input token only costs about tenth of what a fresh one does [hBzUokVYQkI@25], so turn 10 costs roughly what turn 2 did. You forfeit that by putting per-request data at the top — move it at the bottom of the prompt after the instructions [vW8wLsb3Nnc@13]. Long-running agent systems take advantage of prompt caching heavily [ow1we5PzK-o@12] to offset multi-day costs.
- **Batch anything that isn't interactive.** One model read can serve many users at once, and batch on bedrock is a 50% off [vW8wLsb3Nnc@18] whatever model inference you're running. For repeat-shaped requests, add semantic caching could be useful to expedite decisions [T0HhO4YtTfE@21] on cases similar to past ones.
- **Know the price band you must hit.** Some applications justify dollars per query, others must cut below 1 cent per query [1jvxxa7tdjw@38] for the ROI to work. Ask early whether is this inference going to bankrupt my company [vW8wLsb3Nnc@14]. Trainium and Inferentia come at about a 60% price performance improvement over using Nvidia GPUs [vW8wLsb3Nnc@9], at the cost of less HBM and the Neuron SDK.
- **Models break your deployment assumptions.** A served model is 10 to 100 GB of read-only trained data [tLK5jyhQOgA@3]. Startup can be, you know, up to 10 minutes or more [tLK5jyhQOgA@9], so readiness probes must signal that weights are loaded and warm-up is done.
- **Stop shipping weights per pod.** You cannot download 50 GB of model data for every pod start [tLK5jyhQOgA@10]. Pre-populated persistent volumes remain the common answer; image volumes, enabled by default in Kubernetes 1.35 [tLK5jyhQOgA@16], are the intended end state.
- **Route on LLM state, not connection count.** An inference-aware endpoint picker sends work to whichever replica has the least amount of in-flight requests, has the shortest queue [tLK5jyhQOgA@19], with cache quality factored in.
- **Fine-tune late, and cheaply.** There are only two main reasons for fine-tuning [1LhvqZvDT5w@30]: push past what prompting achieves, or swap a large model for a small one. It takes about a couple of thousand examples to take the accuracy to the next level [1LhvqZvDT5w@32], and on a small model with LoRA the cost of it is below $2 per hour [1LhvqZvDT5w@34].
- **Prompt engineering outperformed the forecast.** Across 200+ deployments the team expected to fine-tune and never did: prompt engineering has proven unreasonably effective for us [vW8wLsb3Nnc@13], and moving prompts from Claude 3.7 to Claude 4 was a drop-in with zero regressions.

## Delete on sight

| Delete on sight | Replace with |
| --- | --- |
| A `get_current_date` tool call — "defining a tool called get current date is infuriating to me" [vW8wLsb3Nnc@13] | It's `time.now()`. Format the string into the prompt you already control. |
| Arithmetic inside the LLM — "It is the most expensive possible way of doing math" [vW8wLsb3Nnc@17] | A deterministic function. Hand the model the result. |
| Volatile content at the top of the prompt | Time-sensitive info goes at the bottom of the prompt after the instructions [vW8wLsb3Nnc@13], so the cache prefix survives. |
| Round-robin in front of a vLLM fleet | Cache-aware routing — otherwise it throws away saved work that was perfectly good [hBzUokVYQkI@33]. |
| One long chat session for a whole project | Sub-agents with clean context: they each have like a fresh start [sk6HBdmVmL8@6]. |
| Tests written after the implementation | A validation contract written during planning before any code [ow1we5PzK-o@5]. |
| "Shall we use GPT? Shall we use Claude?" in week one | The evaluation layer in weeks one and two; we selected the model in week seven [ObTPqBGsEbA@24]. |
| Benchmark leaderboards as upgrade evidence | Your own eval set, rerun — because you cannot really rely on one single model [ObTPqBGsEbA@22]. |
| A single managed guardrail, ticked on and forgotten | Layers: it is usually not enough. You usually need to have multiple layers of defense [1LhvqZvDT5w@11]. |
| A `create_tables` script as your migration path | A real migration tool. This only creates tables if they don't exist [E8zpgNPx8jE@59] — new columns silently never appear. |
| A production database listening on 0.0.0.0 | Set the inbound IP restrictions [E8zpgNPx8jE@141] before it holds anything real; the agency default is to block all IPs by default [Q679gH7oszg@38] and allowlist. |
| The job-run date as your `created_at` | The source's publish date — I want that to be the original date of the article itself [E8zpgNPx8jE@93] — or date filters break. |
| A chatbot with no business function behind it | Automate a durable workflow. "if you just build a chatbot, you know, sayanara" [vW8wLsb3Nnc@7]. |
| Thousand-word skill and tool descriptions | Read your traces. One team found tool descriptions and stuff for our engineers were just like super long [sk6HBdmVmL8@14], burning tokens on every call. |

## Templates to copy

Three artefacts assembled from the corpus: the one-pager that has to exist before design, the POC schedule that inverts model selection, and the readiness checklist that decides whether you ship.

### Product requirements one-pager

```
BUSINESS PROBLEM  (solution-agnostic — no architecture words allowed)
  Who:        [user role]
  Today:      [current process]
  Baseline:   [number] [unit] per [task], which is [N]x [comparator]
  Cost of it: [downstream consequence]

CONSTRAINTS  (gather before designing, not after)
  Regulatory / residency: [e.g. data must stay in approved cloud]
  Approved vendors/models: [procurement limits]
  Mandatory human review: [which decisions can never be automated]
  Performance:            [latency ceiling] [monthly inference budget] [uptime SLA]

ROLE OF AI  (three axes)
  Critical or complementary: [...]
  Reactive or proactive:     [...]
  Max autonomy permitted:    [full | semi | advisory]   <- set by constraints above

SUCCESS METRIC  (one, SMART, tied to the baseline)
  Reduce [metric] from [baseline] to [target] within [window] of launch.

DATA SOURCES
  | source | where it lives | raw format | update frequency | pipeline cadence |
  Retrieval technique per source: [vector+prefilter | hybrid | exact match | SQL]

REQUEST FLOW  (one real request, end to end, before any architecture is chosen)
  1. [trigger] -> 2. [retrieve] -> 3. [reason] -> 4. [escalate if ...] -> 5. [log]

UX + FEEDBACK
  Input / Output / Where it lives / What triggers it / Human's role /
  How it explains itself / How a user disagrees with it
```

Structure follows the four-phase framework — Four phases. You start with product requirements [T0HhO4YtTfE@1] — plus the UX questions from the design phase, beginning with what does the system take as input [T0HhO4YtTfE@14].

### Eight-week POC, model chosen in week 7

```
WEEK 1-2   EVALUATION LAYER
  - Interview domain experts; collect ~200 real human-answered cases,
    including the gray areas and the edge cases they argue about.
  - Define success in numbers: % of queries handled, accuracy target,
    latency target, tolerable false-positive rate.
  - Stand up the automated scoring pipeline: capture question + response,
    score against the golden set, route anything below threshold to a human.

WEEK 2-4   DATA FOUNDATION            (expect ~60% of total effort to land here)
  - "Question data": what the system answers from. Freshness cadence per source.
  - "Tracking data": traces, spans, tool calls, token counts. Built now, not later.
  - Fix the source data. Agents do not silently correct what humans route around.

WEEK 4-6   ORCHESTRATION + GUARDRAILS
  - Simplest architecture that satisfies the request flow. Control flow before agents.
  - Deterministic checks (format, PII/NER) -> semantic judges -> behavioural checks.
  - Guardrail classifiers on BOTH sides of the model. Fallback: retry 3x, then human.

WEEK 7     MODEL SELECTION
  - Now, and only now, run candidates against YOUR eval set. Ignore leaderboards.
  - Keep a second provider wired up as a hedge.

WEEK 8     GOVERNANCE + ROLLOUT
  - Prompt versioning as change management. Canary or blue/green. Champion/challenger.
  - Incident playbook rehearsed: detect -> diagnose -> contain -> fix -> add to evals.
```

The week-7 inversion is taken directly from the banking-chatbot redo, where the team selected the model in week seven [ObTPqBGsEbA@24] of an eight-week POC after a prior attempt burned $85K over six months [ObTPqBGsEbA@23]. The five pillars it walks through are evaluation, observability, data foundation, orchestration, and governance [ObTPqBGsEbA@4].

### Pre-launch readiness checklist

```
EVALUATION
  [ ] Golden set exists, built from real human answers, and is growing weekly
  [ ] Three layers running: deterministic -> LLM-judge -> behavioural (tool calls)
  [ ] LLM judge calibrated against human scores (budget ~50 few-shot examples)
  [ ] CI runs a subset per change; full suite only on merge to main

OBSERVABILITY
  [ ] Every pipeline step emits its own trace, including each tool call
  [ ] One request reconstructable end to end: intent -> retrieval -> reason -> guardrail
  [ ] Cost per request, tokens, and turn count on a dashboard
  [ ] Human-override rate tracked as a first-class quality metric

SAFETY
  [ ] Guardrail classifiers before AND after the model
  [ ] PII detection run in testing, not just production
  [ ] Red-team scenarios written; pass rate recorded (x of 20)
  [ ] Non-English jailbreak attempts tested
  [ ] Agent tool scope narrowed to the stated task; out-of-scope = injection signal
  [ ] Per-agent spend cap with a human checkpoint

COST + LATENCY
  [ ] Target $/query agreed with product, not inferred after launch
  [ ] Prompt cache prefix stable; volatile content at the bottom of the prompt
  [ ] Non-interactive work moved to batch
  [ ] Cache-aware routing on if self-hosting

ROLLOUT
  [ ] Prompts versioned under change management, rollback rehearsed
  [ ] Canary or blue/green; reindex in parallel, cut over after validation
  [ ] Incident playbook written and the last incident added to the eval suite
  [ ] Named owner for the agent's actions
```

Assembled from the guardrail architecture (classifiers on both sides, maybe it takes like 100 milliseconds or 50 milliseconds [1LhvqZvDT5w@16] when run in parallel), the incident playbook, and the accountability question the corpus raises but does not answer in depth: clarify accountability who owns the agents actions [uaiq1HvQKoI@50].

## Where the corpus disagrees

Five live disagreements between people who have all shipped. Each is a real fork, not a misunderstanding — which means the right answer is situational and you should know which situation you are in.

### How much autonomy to grant

- **Constrain it.** Start with the simplest, most controlled design and iterate, because you risk ending up with an over-engineered system by doing this [T0HhO4YtTfE@10] otherwise.
- **Release it.** Let a background agent choose its own monitoring cadence — check hourly, or come back in three days for an intermittent fault — because none of this is hard-coded in [vSx5IULvBns@20], and that judgement is the value.
- **Reading:** the split tracks blast radius. Autonomy over *when to look* is cheap; autonomy over *what to decide* is not.

### Parallel agents: throughput or chaos

- **Serial.** For software work, agents conflict. They step on each other's changes. They duplicate work [ow1we5PzK-o@8]; missions run one worker at a time.
- **Parallel.** In the choreography pattern, independent agents on a shared message bus run concurrently, and the advantage it brings you is the latency is reduced [ObTPqBGsEbA@20] because nothing waits on an orchestrator.
- **Reading:** it comes down to shared mutable state. Agents writing to one codebase must serialize; agents reading disjoint sources should not.

### How many coding sessions a human can hold

- **Build for many.** Sub-agents, orchestrators mixing providers, and routing by known model strengths — GPT is like widely considered to be better at debugging [sk6HBdmVmL8@5] — are treated as a maturity pattern worth building infrastructure for.
- **Stay under three.** A shipping agency deliberately runs one or three sessions at a time [Q679gH7oszg@33], plan-mode first, because past that you stop being able to oversee the work.
- **Reading:** the first position assumes a validation harness exists; the second assumes the human *is* the harness.

### Fine-tuning: plan for it or wait for it

- **Reactively.** Prompting plus few-shot usually suffices; fine-tune only for the two main reasons for fine-tuning [1LhvqZvDT5w@30] — accuracy beyond prompting, or a smaller cheaper model.
- **Proactively.** Advice from the platform side is working on platforms which do support fine tuning and training [1jvxxa7tdjw@44] from the start, and collecting proprietary feedback data now, to avoid a migration later.
- **Reading:** these only conflict on spend. Choosing a capable platform is nearly free; running the tuning job is not. There is also a licensing trap — outputs blended across model families mean you can't use those outputs to F tune some other architectures [1jvxxa7tdjw@40].

### Where the real discipline lives

- **Infrastructure.** Having walked the full stack: almost none of it was machine learning [hBzUokVYQkI@44] — it was GPUs, routing, caching and Kubernetes from the first lesson to the last.
- **Data flywheel.** The differentiator is that your application is going to learn from the data that's generated from interaction [7YYVgH0_9CA@8] and feeds it back — which is what makes the system yours rather than rented.
- **Reading:** both are downstream of the same claim in the thesis. Infrastructure sets your cost floor; the flywheel sets your ceiling. Neither is the model.

## Sources

Seventeen talks, read in full, grouped as they were extracted.

**Frameworks and field reports**

1. [AI System Design: From Idea to Production - Apoorva Joshi, MongoDB](https://www.youtube.com/watch?v=T0HhO4YtTfE) — `T0HhO4YtTfE` — the four-phase framework (requirements → design → eval/monitoring → cost/latency/reliability), walked end to end on a claims-review system.
2. [Always-on agents run production without the on-call tax — Justin Smith, Resolve AI](https://www.youtube.com/watch?v=vSx5IULvBns) — `vSx5IULvBns` — the background-agent pattern: always-on, cloud-sandboxed agents triggered by schedule, event or message, choosing their own cadence.
3. [POC to PROD: Hard Lessons from 200+ Enterprise GenAI Deployments - Randall Hunt, Caylent](https://www.youtube.com/watch?v=vW8wLsb3Nnc) — `vW8wLsb3Nnc` — war stories with numbers: vector-store cost tradeoffs, custom silicon, prompt-caching mechanics, generative UI, know-your-user failures.

**Retrieval and agent architecture at scale**

4. [RAG at scale: production ready GenAI apps with Azure AI Search](https://www.youtube.com/watch?v=_2tZaDs-w5s) — `_2tZaDs-w5s` — two-stage recall+rerank, cross-encoder latency budgets, quantization and oversampling, index density limits, incremental ingestion.
5. [The Multi-Agent Architecture That Actually Ships — Luke Alvoeiro, Factory](https://www.youtube.com/watch?v=ow1we5PzK-o) — `ow1we5PzK-o` — the only production multi-agent system here with published numbers: orchestrator/worker/validator, adversarial validation contracts, serial execution, 16-day missions.
6. [The Production AI Playbook: Deploying Agents at Enterprise Scale — Sandipan Bhaumik, Databricks](https://www.youtube.com/watch?v=ObTPqBGsEbA) — `ObTPqBGsEbA` — five pillars, three eval layers, the $85K failed POC and its successful redo, the stale-embedding incident.

**Platform, MLOps and Kubernetes**

7. [Make GenAI Production-Ready With Kubernetes Patterns - Roland Huss, Red Hat & Bilgin Ibryam, Diagrid](https://www.youtube.com/watch?v=tLK5jyhQOgA) — `tLK5jyhQOgA` — model-data loading patterns, LLM-state-aware routing, prefill/decode disaggregation, RAG as ordinary Kubernetes primitives.
8. [From GenAI Pilots to Production - Nikita Kozodoi](https://www.youtube.com/watch?v=1LhvqZvDT5w) — `1LhvqZvDT5w` — layered guardrails, red-teaming with a pass-rate metric, human-then-judge evaluation, concrete fine-tuning cost and data thresholds.
9. [Exploring MLOps and LLMOps: Architectures and Best Practices](https://www.youtube.com/watch?v=1jvxxa7tdjw) — `1jvxxa7tdjw` — deploy-code-not-models, catalog governance and lineage, drift-triggered retraining thresholds, what changes moving to LLMOps.

**Harnesses and hands-on builds**

10. [Omnigent: Open-Source Meta-Harness for AI Agents | Matei Zaharia](https://www.youtube.com/watch?v=sk6HBdmVmL8) — `sk6HBdmVmL8` — infrastructure for running many coding agents together: session portability, contextual security policies, per-agent spend caps.
11. [Build a Complete End-to-End GenAI Project in 3 Hours](https://www.youtube.com/watch?v=E8zpgNPx8jE) — `E8zpgNPx8jE` — a warts-and-all build: the OOM from a heavy dependency, the migration against the wrong database, the committed `.env`.
12. [How I Build and Ship Custom AI Solutions for Clients](https://www.youtube.com/watch?v=Q679gH7oszg) — `Q679gH7oszg` — the business layer: discovery red flags, the 70/80→90% accuracy conversation, PoC vs MVP, sprint pricing, a standardized deploy/monitor/secure stack.

**Infrastructure and scale**

13. [Large Language Model Operations (LLMOps) Explained](https://www.youtube.com/watch?v=cvPEiPt7HXo) — `cvPEiPt7HXo` — a short vendor-agnostic primer naming the LLMOps lifecycle stages and how LLM requirements diverge from classic MLOps.
14. [AI Infrastructure Explained (GPUs, vLLM, and LLM-D)](https://www.youtube.com/watch?v=hBzUokVYQkI) — `hBzUokVYQkI` — VRAM bandwidth math, prefill/decode/KV-cache mechanics, batching limits, sharding, cache-aware routing gains.
15. [Scalable GenAI in Production - Kamelia Aryafar](https://www.youtube.com/watch?v=7YYVgH0_9CA) — `7YYVgH0_9CA` — the production stack above the model, a five-stage maturity model, "latency is a product decision", the data-flywheel argument.

**RAG in practice and enterprise-grade code**

16. [GenAI in Production: How RAG & AI Agents Are Shaping the Future](https://www.youtube.com/watch?v=uaiq1HvQKoI) — `uaiq1HvQKoI` — the RAG stack from ingestion to citations, a 60%-search-time-reduction case study, a checklist of production hurdles including non-technical ones.
17. [Production-Grade AI Project Tutorial – Build & Deploy](https://www.youtube.com/watch?v=mvleESOUTRw) — `mvleESOUTRw` — software-architecture patterns for a GenAI data pipeline: unified loaders, Pydantic models everywhere, chunk metadata, per-call usage tracking.

## What this corpus does not tell you

Seventeen talks, and these questions still have no answer here. Anything below is a gap you must close with your own numbers.

- **Steady-state inference cost.** No talk gives $/1M tokens, $/GPU-hour, or $/query at production volume. Fine-tuning hourly cost is quantified; serving cost is not.
- **A worked eval methodology.** Accuracy figures — 70–80%, 90%, 85% — are quoted as outcomes but never as measurements. No talk shows how those percentages were computed, or names a framework for scoring RAG answers.
- **Prompt-injection defence for tool-calling agents.** Injection is named as a risk and hidden-instruction attacks are described, but no talk gives a defence architecture or tested mitigations.
- **Guardrail accuracy.** Classifier fleets are described with latency numbers and layering, but no false-positive or false-negative rates.
- **Who is on call for the agent.** Accountability is raised as a question. No talk says who debugs the debugger, or how an always-on agent's own judgement is evaluated for correctness.
- **A proactive freshness strategy.** The stale-embedding incident was caught after a satisfaction drop. Nobody prescribes staleness alarms, refresh SLAs, or index-lag monitoring.
- **Reranker and inference-engine selection.** ~100ms is the only guidance on choosing a cross-encoder. vLLM is never compared against TensorRT-LLM, SGLang or TGI.
- **Capacity planning and failover.** No GPU autoscaling thresholds, no traffic-spike headroom numbers, no multi-region or disaster-recovery strategy for GenAI serving.
- **Named orchestration frameworks.** LangGraph, CrewAI and AutoGen are never named.
- **How validation contracts stay honest.** Factory's contracts can span hundreds of assertions over a 16–30 day run, but the talk does not say how they are authored, reviewed, or kept current.
- **Prompt and agent rollback in practice.** Versioning-as-change-management is asserted; no talk walks through canarying a prompt change or rolling one back mid-incident.
- **Model deprecation at scale.** One anecdote — Claude 3.7 to 4 as a drop-in — stands in for the whole migration-risk question.

---

*Production GenAI Systems — compiled from a transcript corpus of 17 talks, 737 chunks read in full across 6 parallel extraction passes, compiled 2026-09-14. Every claim above resolves to a cited chunk; where the corpus was silent, the gaps section says so instead of guessing.*
