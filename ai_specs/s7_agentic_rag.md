# Spec: S7 Agentic RAG Agent

Status: draft
Date: 2026-06-03

## Summary

S7 introduces a new **agentic RAG agent** (`rag_agent`) that replaces the fixed retrieval-then-answer pipeline with a LangGraph ReAct research loop. Instead of doing one retrieval and answering, the agent drives its own research:

1. It retrieves on the original question to get initial context and identify sub-topics.
2. For each sub-topic worth exploring, it calls the retrieval tool again with a focused query.
3. It continues iterating — each loop adding more evidence — until it judges it has enough information.
4. Only then does it produce a comprehensive, cited answer.

This means a single `rag_agent` invocation may call `retrieve_transcript_chunks` five or more times, each time targeting a different angle of the question. The LLM is in control of what to research next and when to stop — the graph does not impose a fixed retrieval count or a grader that forces the agent to answer early.

Because the research loop can take several seconds per iteration, the CLI streams progress in real time using LangGraph's `graph.stream()`. The user sees each retrieval call as it happens — the query the agent chose, which iteration it is on, and how many chunks came back — before the final answer is printed. This makes the agent's reasoning transparent and distinguishes it clearly from the single-shot `rag_llm` path.

The existing `RagTranscriptAgent` is **not touched**. It is labelled `rag_llm` in the CLI and evaluation context (documentation label only — no code change) so the two approaches can be compared side-by-side.

One tool is exposed to the LangGraph agent:

- **`retrieve_transcript_chunks(query)`** — wraps `MultiTranscriptRagContextProvider.get_context(...)`. Returns top-k chunks relevant to `query`. The agent calls this tool multiple times, once per topic it wants to research. Because the agent controls the query string on every call, it can refine or sharpen a query simply by choosing a different string on the next iteration — no separate rewrite tool is needed.

## Current Source Of Truth

Build on:

- `src/rag/context.py`
  - `MultiTranscriptRagContextProvider.get_context(question, source_url, top_k, filter_transcripts, ...)` — the retrieval entry point. S7 wraps this as a LangChain `@tool`. Do not change this file.
- `src/agents/models.py`
  - `RagQuestionRequest`, `RagAnswerReference`, `RagTranscriptAnswer` — reuse for the agent's output shape.
- `src/agents/rag_transcript_agent.py`
  - `RagTranscriptAgent` — the existing LLM agent. S7 does **not** touch this file. It is the `rag_llm` baseline for comparison.
- `src/agents/prompts.py`
  - `RAG_SYSTEM_PROMPT`, `build_rag_question_prompt`, `build_transcript_context_prompt` — available for reference; the agentic agent uses its own prompts tuned for the ReAct research loop.
- `src/cli.py`
  - `rag-ask` subcommand — extended with mutually exclusive `--rag_llm` / `--rag_agent` flags to select the agent. Default behavior (no flag, or `--rag_llm`) is unchanged.
- `src/config.py`
  - Extended with agentic-agent defaults. No behavior change when env vars are absent.
- `src/observability.py`
  - Existing helpers reused. New per-node logging added alongside.

## Goals

- Add a `RagAgent` class in `src/agents/rag_agent.py` that uses a LangGraph ReAct loop for iterative transcript research.
- The agent must be capable of calling `retrieve_transcript_chunks` multiple times — once per sub-topic — accumulating evidence across loops before answering.
- The LLM drives the research plan: it decides which sub-topics to retrieve, in what order, and when it has enough information to stop.
- Stream each retrieval iteration to the terminal in real time so the user can see the agent's research plan as it unfolds.
- Make the new agent callable from `rag-ask --rag_agent` without changing the default `rag-ask` path.
- Preserve the existing `RagTranscriptAgent` (`rag_llm`) in its current state so both agents answer the same question for comparison.
- Reuse `MultiTranscriptRagContextProvider` for all retrieval inside the new agent.
- Return a `RagTranscriptAnswer` so the output shape is identical between `rag_llm` and `rag_agent`.

## Non-Goals

- Do not change `RagTranscriptAgent`. S7 is additive only.
- Do not change `TranscriptAgent`.
- Do not change retrieval, chunking, embeddings, or storage.
- Do not impose a fixed retrieval count. The agent decides how many retrieval loops to perform.
- Do not use a separate `grade_documents` node to force early termination. The LLM decides when research is complete.
- Do not introduce a streaming UI or multi-turn conversation in S7.
- Do not support parallel tool calls. The ReAct loop is sequential.
- Do not add a new vector store. The agent uses the same Chroma collection via the existing context provider.
- Do not add async execution. S7 is synchronous to match the existing CLI and test patterns.
- Do not add a new dashboard tab in S7. Comparison is done via CLI and the eval HTML report.

## Agent Naming

To enable comparison between the two approaches, the convention from this point forward is:

| Label | Class | File | Change in S7 |
|-------|-------|------|--------------|
| `rag_llm` | `RagTranscriptAgent` | `src/agents/rag_transcript_agent.py` | **None. Zero changes.** |
| `rag_agent` | `RagAgent` | `src/agents/rag_agent.py` | New file, new class. |

`rag_llm` is a **documentation and CLI label only**. It is the shorthand used in specs, eval reports, and the comparison table to refer to the existing pipeline agent. It does **not** involve any code rename:

- The class `RagTranscriptAgent` keeps its name.
- The file `src/agents/rag_transcript_agent.py` keeps its name.
- No import path, alias, or symbol changes anywhere in the codebase.
- A diff of `src/agents/rag_transcript_agent.py` against `main` must be empty after S7 is implemented.

The CLI flag `--rag_agent` (added to `rag-ask`) selects `rag_agent`. The flag `--rag_llm` selects the baseline explicitly; it is also the default when neither flag is passed, so `rag-ask` continues to use the existing `RagTranscriptAgent` exactly as before. `--rag_llm` and `--rag_agent` are mutually exclusive.

## Architecture

### Research loop design

The `rag_agent` graph is a **pure ReAct loop**. There is no separate grader node and no fixed-phase pipeline. The LLM is the sole decision-maker for what to research and when to stop:

```
START
  │
  ▼
generate_query_or_respond          ← LLM with tools bound
  │
  ├── tool_calls present?
  │      │
  │      ▼  YES
  │   retrieve                     ← ToolNode executes retrieve_transcript_chunks
  │      └──────────────────────────► back to generate_query_or_respond
  │                                   (loop: agent sees tool result and decides next action)
  │
  └── no tool_calls?
         │
         ▼  NO (agent is done researching)
        END                         ← final AI message becomes the answer
```

On each pass through `generate_query_or_respond`, the LLM:

- Sees the full message history: original question + all prior tool calls and results.
- Decides to call `retrieve_transcript_chunks` with a new focused query (continue researching), OR
- Produces a final answer with no tool call (research is complete, exit loop).

This means the agent naturally supports the intended research pattern:

```
User question
  → retrieve(original question)          # broad initial retrieval
  → [LLM sees chunks, identifies topics A, B, C, D, E]
  → retrieve("topic A focused query")    # drill into topic A
  → retrieve("topic B focused query")    # drill into topic B
  → retrieve("topic C focused query")    # drill into topic C
  → ... (as many as the agent judges necessary)
  → [LLM decides it has enough evidence]
  → final answer (no tool call)
```

The number of retrieval loops is not fixed. For a broad question the agent may do 5–8 calls. For a narrow question it may do 1. The `max_iterations` guard is the only hard cap.

### LangGraph graph

```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

tools = [retrieve_transcript_chunks]

workflow = StateGraph(MessagesState)

workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode(tools))

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    route_on_tool_calls,          # returns "retrieve" or END
    {"retrieve": "retrieve", END: END},
)

workflow.add_edge("retrieve", "generate_query_or_respond")   # always loop back

graph = workflow.compile()
```

```python
def route_on_tool_calls(state: MessagesState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "retrieve"
    return END
```

No `grade_documents` node. No `generate_answer` node. The graph has three elements: one LLM node, one ToolNode, and one conditional edge. The final answer is the last AI message when the agent exits the loop with no tool call.

### Streaming

The CLI drives the graph with `graph.stream(inputs, stream_mode="updates")` instead of `graph.invoke()`. LangGraph yields one dict per node execution: `{node_name: node_output}`. The CLI inspects these events and prints a progress line for each retrieval iteration before the final answer is printed.

**Color palette**

Each iteration is assigned a color from a fixed cycle so the user can visually distinguish retrieval passes at a glance. The palette cycles when `max_iterations` exceeds its length.

| Iteration mod 6 | ANSI color | Name |
|-----------------|-----------|------|
| 1 | `\033[96m` | Bright cyan |
| 2 | `\033[93m` | Bright yellow |
| 3 | `\033[92m` | Bright green |
| 4 | `\033[95m` | Bright magenta |
| 5 | `\033[94m` | Bright blue |
| 0 | `\033[97m` | Bright white |

Color is applied to the entire iteration line: the `[N]` badge, the label, the query, and the `→ K chunks` suffix. The reset code `\033[0m` is appended at the end of each line. Colors are suppressed entirely when stdout is not a TTY (`sys.stdout.isatty()` is false) so piped and CI output is plain text with no escape codes. No new dependencies — use ANSI codes directly.

**Event handling per node:**

| Node | What to print |
|------|--------------|
| `generate_query_or_respond` — `retrieve_transcript_chunks` call | Line in iteration color: `[N] Retrieving: "<query>"` |
| `retrieve` — after `retrieve_transcript_chunks` | Append to same line (via `\r` on TTY): `→  K chunks` then reset |
| `generate_query_or_respond` — no tool calls (final) | blank line then print `Answer` block in default color |

The `\r` overwrite keeps each retrieval call on one line: the query appears immediately when the LLM decides to call the tool, and the chunk count is appended once the tool returns. On non-TTY output, query and chunk count are printed as two separate plain lines so the output is readable without a terminal.

**Example terminal output for a broad question (colors described in comments):**

```
Researching...
                                                           # colors reset between lines
[1] Retrieving: "AI engineers leveraging Claude feature development"  →  8 chunks   # bright cyan
[2] Retrieving: "Claude code generation risks silent failures"        →  6 chunks   # bright yellow
[3] Retrieving: "AI engineer autonomy level human review gates"       →  7 chunks   # bright green
[4] Retrieving: "spec-driven workflow Claude checklist"               →  5 chunks   # bright magenta
[5] Retrieving: "feature shipping Claude regression testing"          →  8 chunks   # bright blue

Answer
The corpus shows three recurring patterns...

References
[1] https://www.youtube.com/watch?v=...&t=120s
...

Agent: 5 iterations (rag_agent)
```

**`RagAgent.answer_streaming()` — the streaming entry point used by the CLI:**

```python
def answer_streaming(
    self,
    request: RagQuestionRequest,
    on_event: Callable[[AgentProgressEvent], None] | None = None,
) -> RagTranscriptAnswer:
    """Run the research loop, calling on_event for each node update.

    on_event is called synchronously in the streaming loop. The CLI passes a
    function that prints progress lines; tests pass a list-accumulator.
    Returns the same RagTranscriptAnswer as answer().
    """
```

`answer()` remains unchanged — it wraps `graph.invoke()` for programmatic and test use where streaming output is not wanted.

**`AgentProgressEvent` — the progress event model:**

`AgentProgressEvent` is a Pydantic `BaseModel`. It lives in `src/agents/models.py` alongside the other agent data models so it follows the same import and convention pattern as the rest of the project.

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class AgentProgressEvent(BaseModel):
    iteration: int = Field(description="1-based retrieval counter.")
    event_type: Literal["retrieval_start", "retrieval_complete", "answer_start"]
    query: str | None = Field(default=None, description="The retrieval query for this iteration.")
    chunk_count: int | None = Field(default=None, description="Populated on retrieval_complete.")
```

Using `Literal` for `event_type` instead of `str`:
- Makes valid values explicit and machine-checkable.
- Allows exhaustive matching in the CLI's event handler without silent no-ops on typos.
- Consistent with how the project uses Pydantic for structured, validated data.

`query` and `chunk_count` default to `None` because they are not populated for all event types (`answer_start` has neither; `retrieval_start` has `query` but not yet `chunk_count`). Required fields have no default; optional fields use `| None = None`.

The CLI imports `AgentProgressEvent` from `src.agents.models`. Tests instantiate it directly and assert on its fields without any terminal output.

### `generate_query_or_respond` node

```python
def generate_query_or_respond(state: MessagesState) -> dict:
    response = llm.bind_tools([retrieve_transcript_chunks]).invoke(state["messages"])
    return {"messages": [response]}
```

On the first pass `state["messages"]` contains only the user's question. On subsequent passes it contains the full conversation history: question + all tool call/result pairs. The LLM uses this accumulating context to decide what to research next. Because the agent owns the query string on every call, it refines or rephrases queries by simply choosing a different string — no separate rewrite tool is required.

### Tool

#### `retrieve_transcript_chunks`

```python
@tool
def retrieve_transcript_chunks(query: str) -> str:
    """Search indexed YouTube transcript chunks for content relevant to the query.

    Returns formatted chunk references with timestamps and source URLs.

    Call this tool whenever you need evidence from the transcript corpus.
    You should call it multiple times — once for the original question to identify
    the key topics, then once per sub-topic that needs deeper evidence.
    Keep calling it until you have enough information to write a comprehensive answer.
    """
    context = context_provider.get_context(
        question=query,
        source_url=source_url,
        top_k=top_k,
        filter_transcripts=filter_transcripts,
    )
    return context.context_text
```

The tool is a closure constructed at `RagAgent._build_graph(request)` time so `context_provider`, `source_url`, `top_k`, and `filter_transcripts` are injected from the `RagQuestionRequest`. The tool signature seen by the LLM is `retrieve_transcript_chunks(query: str)`.

The docstring is the agent's primary signal for when and how to use this tool. It explicitly instructs the agent to call it multiple times for sub-topic research.

### `generate_query_or_respond` system prompt

The system prompt does two jobs: it guides the research loop and it prescribes the structure of the final answer. Both must be explicit.

```text
AGENTIC_RAG_SYSTEM_PROMPT = """You are a YouTube transcript research agent.

You have one tool:
- retrieve_transcript_chunks(query): search the indexed transcript corpus for chunks
  relevant to a query. Call it with a focused, specific query each time.

Research protocol:
1. Start by calling retrieve_transcript_chunks with the user's question to get initial context
   and understand which topics the transcripts cover.
2. From the initial results, identify the key sub-topics, claims, or angles that deserve
   deeper investigation. Plan a focused retrieval query for each one.
3. Call retrieve_transcript_chunks once per sub-topic. Each call should use a focused query
   that targets that sub-topic specifically — not a paraphrase of the original question.
4. Continue retrieving until you have enough evidence to write a comprehensive answer.
   You decide when you have enough. For a broad question this may be 5–8 calls.
   For a narrow question it may be 1–2.
5. Once you have sufficient evidence, produce your final answer — do not call any tool.

Answer structure (for your final response, with no tool call):
Your answer must be structured markdown in this exact order:

## Key Findings
A numbered list of the most important insights from across all your research.
Each finding is one concise sentence with inline citations. Example:
1. AI engineers primarily use Claude for spec-driven feature development [1][3].
2. The main risk cited is silent regression in untested code paths [2][5].

## Finding 1: <short title>
2–4 sentences expanding on finding 1, grounded only in the chunks that support it.
Cite inline with the labels from the retrieved chunks (e.g. [1], [3]).

## Finding 2: <short title>
2–4 sentences expanding on finding 2, with its own citations.

## Finding 3: <short title>
...and so on, one section per finding in the Key Findings list.

Answer rules:
- Use only the retrieved transcript chunks accumulated in this conversation.
- Every claim must have at least one inline citation.
- Do not invent names, dates, claims, or conclusions.
- Do not repeat the same evidence under multiple findings.
- If the transcripts do not contain enough information on a finding, say so in that section.
- Number of findings: write as many as the evidence supports. Do not pad with thin findings.

Return JSON with this exact shape — the answer field contains the structured markdown above:
  {"question": "...", "answer": "## Key Findings\n1. ...\n\n## Finding 1: ...\n...",
   "references": [{"label": "[1]", "source_url": "...", "timestamp_url": "...",
   "start_seconds": 0.0, "end_seconds": 0.0, "chunk_index": 0, "video_id": "..."}]}
"""
```

### Answer structure

The `answer` field of `RagTranscriptAnswer` contains structured markdown produced by the agent at the end of its research loop. The prescribed format is:

```
## Key Findings
1. <insight statement> [citation]
2. <insight statement> [citation]
...

## Finding 1: <short title>
<2–4 sentences of explanation with inline citations>

## Finding 2: <short title>
<2–4 sentences of explanation with inline citations>

...one section per finding
```

**Why this structure:** The research loop retrieves evidence across multiple sub-topics. The Key Findings section gives the user a scannable summary of what was learned. Each Finding section then provides the depth and cited evidence behind a single insight, so users who want to understand where a finding comes from can read further and follow the timestamp links to the source video.

The `references` list remains a flat array of all cited chunks in label order — unchanged from `RagTranscriptAgent`. The `answer` field is richer but the `RagTranscriptAnswer` schema is not changed; the structure lives inside the string.

The `_parse_answer` method does not validate the internal markdown structure — it validates only that the JSON is well-formed and that `references` labels match citations in the answer text. Structure enforcement is via the prompt; format regression is caught by manual review and the completion test.

### Answer parsing

`RagAgent._parse_answer(state)` reads the last AI message from the final graph state. Because the agent is instructed to return JSON in the final answer, the same `_json_object` + `_fallback_references` helper pattern used by `RagTranscriptAgent` is applied to deserialise the message content into a `RagTranscriptAnswer`.

If the last message is not valid JSON (e.g. the agent produced a plain-text response), `_fallback_references` extracts an answer string and populates references from any chunks cited inline, matching the existing fallback behaviour.

### Iteration guard

`max_iterations` is enforced by passing `recursion_limit` to `graph.compile()` or `graph.invoke()`. LangGraph raises `GraphRecursionError` when the cycle count exceeds the limit. `RagAgent.answer()` catches this, extracts the best available last AI message from the partial state, parses it into a `RagTranscriptAnswer`, and sets `last_terminated_reason = "max_iterations_reached"` on the agent instance for observability.

Default: `max_iterations = 10` (enough for initial retrieval + up to 9 sub-topic passes). Configurable via `--max-iterations` CLI flag or `YT_AGENT_RAG_AGENT_MAX_ITERATIONS` env var.

## `RagAgent` Interface

```python
class RagAgent:
    def __init__(
        self,
        llm,
        context_provider: MultiTranscriptRagContextProvider,
        max_context_chars: int = 40_000,
        max_iterations: int = 10,
    ) -> None:
        ...

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        context_provider: MultiTranscriptRagContextProvider | None = None,
    ) -> "RagAgent":
        ...

    def answer(self, request: RagQuestionRequest) -> RagTranscriptAnswer:
        """Run the research loop via graph.invoke() and return the final answer.

        No streaming output. Used by tests and programmatic callers.
        """
        ...

    def answer_streaming(
        self,
        request: RagQuestionRequest,
        on_event: Callable[[AgentProgressEvent], None] | None = None,
    ) -> RagTranscriptAnswer:
        """Run the research loop via graph.stream(), calling on_event per node update.

        Used by the CLI to print live progress. on_event is called synchronously.
        Returns the same RagTranscriptAnswer as answer().
        """
        ...

    def _build_graph(self, request: RagQuestionRequest) -> CompiledGraph:
        """Construct and compile the LangGraph graph for this request.

        Tools are closures that capture request parameters so retrieve calls
        inherit source_url, top_k, and filter_transcripts from the request.
        """
        ...

    def _parse_answer(self, final_state: MessagesState) -> RagTranscriptAnswer:
        """Deserialise the last AI message into a RagTranscriptAnswer."""
        ...
```

`last_context` is populated after `answer()` returns. It holds a `TranscriptContext` built from the **union** of all chunks retrieved across all loop iterations — the same interface used by `RagTranscriptAgent.last_context` so observability and dashboard consumers work without branching.

`last_iteration_count` is the number of `retrieve_transcript_chunks` calls the agent made before producing its final answer.

`last_terminated_reason` is `"completed"` on normal exit or `"max_iterations_reached"` on guard fire.

## Source Of New Files

```text
src/
  agents/
    rag_agent.py              # New: RagAgent — LangGraph ReAct research agent

tests/
  agents/
    test_rag_agent.py         # New: unit tests for RagAgent
```

Files updated:

```text
src/cli.py                    # Add --rag_llm, --rag_agent, --max-iterations flags to rag-ask
src/config.py                 # Add YT_AGENT_RAG_AGENT_MAX_ITERATIONS
src/agents/models.py          # Add AgentProgressEvent (Pydantic BaseModel)
src/agents/prompts.py         # Add AGENTIC_RAG_SYSTEM_PROMPT
readme.md                     # Add rag-agent invocations and rag_llm vs rag_agent section
```

Files **not** touched:

```text
src/agents/rag_transcript_agent.py   # rag_llm baseline — zero changes
src/agents/transcript_agent.py       # unchanged
src/rag/context.py                   # unchanged
src/rag/storage.py                   # unchanged
```

## CLI Interface

```bash
# Baseline — uses rag_llm / RagTranscriptAgent. This is also the default
# when neither --rag_llm nor --rag_agent is passed.
uv run python -m src.cli rag-ask "$question" --top-k 20
uv run python -m src.cli rag-ask "$question" --rag_llm --top-k 20
uv run python -m src.cli rag-ask "$question" --rag_llm --url "$url" --top-k 10
uv run python -m src.cli rag-ask "$question" --rag_llm --filter-transcripts --top-k 20

# New: route to rag_agent
uv run python -m src.cli rag-ask "$question" --rag_agent
uv run python -m src.cli rag-ask "$question" --rag_agent --url "$url" --top-k 10
uv run python -m src.cli rag-ask "$question" --rag_agent --filter-transcripts
uv run python -m src.cli rag-ask "$question" --rag_agent --max-iterations 8
```

New argparse additions — `--rag_llm` and `--rag_agent` are a mutually exclusive group; `rag_llm` is the default when neither is passed:

```python
agent_group = rag_ask.add_mutually_exclusive_group()
agent_group.add_argument(
    "--rag_agent",
    dest="rag_agent",
    action="store_true",
    default=False,
    help="Use the agentic LangGraph RAG agent (rag_agent) instead of the pipeline agent (rag_llm).",
)
agent_group.add_argument(
    "--rag_llm",
    dest="rag_llm",
    action="store_true",
    default=False,
    help="Use the pipeline RAG agent (rag_llm). This is the default when neither --rag_llm nor --rag_agent is passed.",
)
rag_ask.add_argument(
    "--max-iterations",
    type=int,
    default=None,
    help="Max ReAct loop iterations for --rag_agent mode (default: YT_AGENT_RAG_AGENT_MAX_ITERATIONS or 10).",
)
```

When `--rag_agent` is **not** passed (i.e. `--rag_llm` is passed or no flag at all), `rag-ask` behaves exactly as today. `--max-iterations` is ignored without `--rag_agent`.

Output format with `--rag_agent` — streamed live to the terminal, each iteration line in its cycle color:

```text
Researching...

[1] Retrieving: "AI engineers leveraging Claude feature development"  →  8 chunks   ← bright cyan
[2] Retrieving: "Claude code generation risks silent failures"        →  6 chunks   ← bright yellow
[3] Retrieving: "AI engineer autonomy level human review gates"       →  7 chunks   ← bright green
[4] Retrieving: "spec-driven workflow Claude checklist"               →  5 chunks   ← bright magenta
[5] Retrieving: "feature shipping Claude regression testing"          →  8 chunks   ← bright blue

Answer

## Key Findings
1. AI engineers primarily use Claude for spec-driven feature development [1][3].
2. The dominant risk cited is silent regression in untested code paths [2][5].
3. Human review gates are applied at PR-merge time rather than during generation [4][7].

## Finding 1: Spec-driven feature development
Engineers report writing a short spec before delegating work to Claude, including
a checklist of acceptance criteria the agent must satisfy [1]. The most common
failure mode is omitting test commands from the spec, which allows the agent to
skip verification [3].

## Finding 2: Silent regression risk
Multiple speakers flag that Claude-generated code can silently break adjacent
paths not covered by tests [2]. The risk is described as higher when the agent
is given write access to shared utilities [5].

## Finding 3: Human review at merge time
The dominant pattern is a review-only gate: a human reads the diff and runs
the tests before merge, but does not re-prompt the agent [4]. One speaker
contrasts this with an autonomous-merge-over-feature-flag approach [7].

References
[1] https://www.youtube.com/watch?v=...&t=120s
[2] https://www.youtube.com/watch?v=...&t=480s
...

Agent: 5 iterations (rag_agent)
```

Each `[N] Retrieving: …` line appears — in its iteration color — as soon as the LLM decides to call the tool. The `→ K chunks` count is appended on the same line once the tool returns (`\r` overwrite on a TTY; plain newline on non-TTY). Colors are suppressed entirely on non-TTY stdout. The final `Answer` / `References` blocks print in the default terminal color and are identical in format to the existing `rag-ask` output.

## Config

Add to `src/config.py`:

```text
YT_AGENT_RAG_AGENT_MAX_ITERATIONS=10   # hard cap on LangGraph ReAct loop cycles
```

Read only when `--rag_agent` is active. No behavior change when unset.

## Data

- Inputs: same `RagQuestionRequest` as `RagTranscriptAgent`. No new fields required.
- Outputs: same `RagTranscriptAnswer`. The `recursion` field is `None`.
- `last_context`: union of all retrieved chunks across all iterations, built after the loop exits.
- Persistence: none. No new files written.

## Constraints

- Do not change `RagTranscriptAgent` or any file it depends on.
- Each call to `retrieve_transcript_chunks` must delegate to `MultiTranscriptRagContextProvider.get_context(...)` with the same `source_url`, `top_k`, and `filter_transcripts` from the original request. The query string changes per call; everything else is fixed.
- `max_iterations` must be enforced in code (via LangGraph's `recursion_limit`), not by the LLM.
- The same DeepSeek model used by `RagTranscriptAgent` is used for the ReAct LLM. No new model provider.
- LangGraph is already available via `uv sync`. Do not add new packages.
- Tests mock the LLM and context provider. No live DeepSeek or embedding calls in CI.

## Testing Requirements

All external calls (LLM, context provider) must be mocked. No live API keys needed in CI.

Required tests:

- `RagAgent.answer(request)` returns a valid `RagTranscriptAnswer` with `question`, `answer`, `references` populated.
- When the LLM produces no tool call on the first turn, the graph exits immediately and returns the LLM's message as the answer.
- When the LLM calls `retrieve_transcript_chunks` once, `ToolNode` executes it, the result is appended to messages, and the graph loops back to `generate_query_or_respond`.
- When the LLM calls `retrieve_transcript_chunks` three times before producing a final answer, `last_iteration_count` is `3` and `last_context` contains the union of all three retrieval results.
- `max_iterations` guard: when the loop exceeds the cap, `answer()` returns a valid `RagTranscriptAnswer` (not a raised exception) and `last_terminated_reason == "max_iterations_reached"`.
- The `retrieve_transcript_chunks` tool closure passes the correct `source_url`, `top_k`, and `filter_transcripts` from the original request regardless of the `query` argument.
- CLI: `rag-ask --rag_agent` instantiates `RagAgent`; `rag-ask --rag_llm` and `rag-ask` with no flag instantiate `RagTranscriptAgent`.
- CLI: `Agent: N iterations (rag_agent)` footer appears with `--rag_agent`; absent without it.
- CLI: passing both `--rag_llm` and `--rag_agent` together is rejected by argparse (mutually exclusive group).
- Malformed final AI message (non-JSON): `_fallback_references` is invoked and a valid `RagTranscriptAnswer` is returned.
- When the mocked LLM returns a well-formed structured answer, `result.answer` contains `## Key Findings` and at least one `## Finding` heading.
- When the mocked LLM returns a flat answer (no headings — fallback path), `_parse_answer` still returns a valid `RagTranscriptAnswer` without raising.
- `answer_streaming()` with a mock that makes 3 retrieval calls emits exactly 3 `retrieval_start` events and 3 `retrieval_complete` events in alternating order before an `answer_start` event, and returns the same `RagTranscriptAnswer` as `answer()` on the same input.
- `AgentProgressEvent.chunk_count` is populated on `retrieval_complete` events and equals the number of chunks returned by the mocked context provider.
- `answer_streaming()` with `on_event=None` does not raise and returns a valid answer (the callback is optional).
- `answer()` and `answer_streaming()` produce identical `RagTranscriptAnswer` objects given the same mocked inputs.
- When `sys.stdout.isatty()` returns `True`, the CLI output for iteration 1 contains `\033[96m` (bright cyan) and ends with `\033[0m` (reset).
- When `sys.stdout.isatty()` returns `False`, the CLI output contains no ANSI escape codes at all.
- The color for iteration N is determined by `(N - 1) % 6` mapping to the fixed palette; iteration 7 gets the same color as iteration 1.
- The `Answer`, `References`, and `Agent:` footer lines contain no ANSI codes regardless of TTY state.

## Completion Test

From a clean clone with env set up and the existing corpus indexed:

```bash
question="what does this corpus say about how AI engineers leverage agentic coding to fully develop features, what is the best workflow for agentic coding"

# 1. Baseline — existing rag_llm (unchanged behavior). Default with no flag,
#    or explicit via --rag_llm.
uv run python -m src.cli rag-ask "$question" --rag_llm --top-k 30

# 2. Baseline — existing rag_llm with recursive rag retrieval 
uv run python -m src.cli rag-ask "$question" --rag_llm --recursive --top-k 10

# 3. Agentic agent — same question
uv run python -m src.cli rag-ask "$question" --rag_agent --top-k 10

```

Expected:

- Run 1: existing `Answer` + `References` blocks, no footer, no streaming output. Behavior byte-identical to pre-S7 main.
- Runs 2–4: terminal shows `Researching…` header, then `[N] Retrieving: …  →  K chunks` lines appearing live as each iteration completes. After the loop exits, the standard `Answer` / `References` blocks print, followed by `Agent: N iterations (rag_agent)` where `N >= 2`.
- The answer from run 2 is demonstrably more comprehensive than run 1 — it cites chunks from multiple distinct retrieval passes covering different sub-topics, not just the top-10 from a single retrieval.
- The answer from run 2 contains a `## Key Findings` section followed by at least two `## Finding N:` sections, each with inline citations traceable to the `References` block.
- Re-running run 2 with stdout piped to a file (`... --rag_agent > out.txt`) produces clean output with no `\r` artifacts — TTY detection is working.

## Acceptance Criteria

- `src/agents/rag_agent.py` is added with a `RagAgent` class.
- `RagAgent.answer(RagQuestionRequest)` returns a `RagTranscriptAnswer` produced by a LangGraph ReAct research loop.
- The LangGraph graph has exactly three elements: `generate_query_or_respond` node, `retrieve` ToolNode, and a `route_on_tool_calls` conditional edge. No `grade_documents` node. No `generate_answer` node.
- The graph loops: `generate_query_or_respond` → `retrieve` → `generate_query_or_respond` → ... until the LLM produces a response with no tool calls.
- One tool is defined: `retrieve_transcript_chunks`. Its docstring explicitly instructs multi-loop research behaviour.
- `retrieve_transcript_chunks` delegates to `MultiTranscriptRagContextProvider.get_context(...)`. `source_url`, `top_k`, and `filter_transcripts` are fixed from the request; only `query` changes per call.
- `AGENTIC_RAG_SYSTEM_PROMPT` instructs the agent to: (1) start with an initial broad retrieval, (2) identify sub-topics, (3) call `retrieve_transcript_chunks` once per sub-topic, (4) continue until evidence is sufficient, (5) then produce a final JSON answer with a structured markdown `answer` field.
- The `answer` field of the returned `RagTranscriptAnswer` contains structured markdown in the prescribed format: a `## Key Findings` numbered list followed by one `## Finding N: <title>` section per finding. The `RagTranscriptAnswer` schema is not changed — the structure lives inside the `answer` string.
- `max_iterations` is enforced via LangGraph's `recursion_limit`. Exceeding it returns the best available answer, not an exception.
- `last_context` on the agent instance reflects the union of all chunks retrieved across all loop iterations.
- `last_iteration_count` reflects the number of tool calls made before the final answer.
- `RagAgent` exposes two public entry points: `answer()` (uses `graph.invoke()`, no output) and `answer_streaming()` (uses `graph.stream()`, calls `on_event` per node update). Both return a `RagTranscriptAnswer`.
- `AgentProgressEvent` is a Pydantic `BaseModel` defined in `src/agents/models.py` with fields: `iteration: int`, `event_type: Literal["retrieval_start", "retrieval_complete", "answer_start"]`, `query: str | None = None`, `chunk_count: int | None = None`. It follows the same `BaseModel` + `Field` conventions as the rest of the file.
- The CLI calls `answer_streaming()` when `--rag_agent` is set and prints a `Researching…` header, one `[N] Retrieving: …` line per retrieval call (with chunk count appended on completion), and the standard `Answer` / `References` / footer blocks.
- Each iteration line is colored using the 6-color ANSI cycle defined in the Streaming section (`\033[96m` for iteration 1, `\033[93m` for 2, etc., cycling by `(N-1) % 6`). No new packages — raw ANSI codes only.
- TTY detection governs both color and `\r` overwrite: on a TTY, colors are applied and chunk counts overwrite the query line; on non-TTY stdout, output is plain text with newlines only.
- `rag-ask --rag_agent` routes to `RagAgent`; `rag-ask --rag_llm` and `rag-ask` with no flag route to `RagTranscriptAgent`. `--rag_llm` and `--rag_agent` are mutually exclusive. The existing agent path is unchanged.
- `AGENTIC_RAG_SYSTEM_PROMPT` is added to `src/agents/prompts.py` without modifying existing prompt constants.
- `src/agents/rag_transcript_agent.py` is **not modified** by S7. A diff of that file against `main` must be empty. The class name `RagTranscriptAgent`, the file name, and all import paths are unchanged. `rag_llm` is a documentation label only — it corresponds to no code symbol.
- Tests pass with LLM and context provider mocked. No live API calls in CI.
- `readme.md` documents `rag-ask --rag_llm` / `rag-ask --rag_agent`, `--max-iterations`, `YT_AGENT_RAG_AGENT_MAX_ITERATIONS`, and the `rag_llm` vs `rag_agent` naming convention.

## Tasks

S7 is split into five sequential tasks. Each task is scoped to a minimal set of files so a coding agent receives only the context it needs. Complete each task in order — later tasks read outputs from earlier ones.

---

### Task 1 — Data models, prompts, config

**Goal:** Add all new data types and constants. No logic, no agent, no CLI changes.

**Files to read before starting:**
- `src/agents/models.py`
- `src/agents/prompts.py`
- `src/config.py`

**Changes:**
- `src/agents/models.py` — add `AgentProgressEvent` Pydantic `BaseModel` (see *AgentProgressEvent* section)
- `src/agents/prompts.py` — add `AGENTIC_RAG_SYSTEM_PROMPT` constant (see *generate_query_or_respond system prompt* section). Do not modify existing constants.
- `src/config.py` — add `YT_AGENT_RAG_AGENT_MAX_ITERATIONS` env var with default `10` (see *Config* section)

**Relevant spec sections:** *AgentProgressEvent*, *generate_query_or_respond system prompt*, *Config*

**Done when:** `AgentProgressEvent` is importable from `src.agents.models`, `AGENTIC_RAG_SYSTEM_PROMPT` is importable from `src.agents.prompts`, and `YT_AGENT_RAG_AGENT_MAX_ITERATIONS` is readable from settings.

---

### Task 2 — Core `RagAgent` class

**Goal:** Implement the LangGraph ReAct research loop. No CLI changes.

**Files to read before starting:**
- `src/agents/models.py` (after Task 1)
- `src/agents/prompts.py` (after Task 1)
- `src/config.py` (after Task 1)
- `src/rag/context.py`
- `src/agents/rag_transcript_agent.py` (read-only reference — do not modify)

**Changes:**
- `src/agents/rag_agent.py` — new file containing `RagAgent` (see *Architecture*, *RagAgent Interface*, *Tool*, *Answer parsing*, *Iteration guard* sections)

**Relevant spec sections:** *Architecture*, *Research loop design*, *LangGraph graph*, *Tool*, *generate_query_or_respond node*, *Answer parsing*, *Iteration guard*, *RagAgent Interface*, *Source Of New Files*

**Done when:** `RagAgent.answer(request)` and `RagAgent.answer_streaming(request, on_event)` are callable, `last_context`, `last_iteration_count`, and `last_terminated_reason` are populated after `answer()` returns, and `_parse_answer` handles both valid JSON and malformed fallback cases.

---

### Task 3 — CLI integration

**Goal:** Wire `RagAgent` into the CLI and implement streaming terminal output with TTY-aware color.

**Files to read before starting:**
- `src/cli.py`
- `src/agents/rag_agent.py` (after Task 2)
- `src/agents/models.py` (after Task 1 — for `AgentProgressEvent`)

**Changes:**
- `src/cli.py` — add mutually exclusive `--rag_llm` / `--rag_agent` flags and a `--max-iterations` flag to `rag-ask`; route to `RagAgent` when `--rag_agent` is set; implement the `on_event` callback that prints streamed progress with ANSI color cycling and TTY detection (see *CLI Interface* and *Streaming* sections)

**Relevant spec sections:** *CLI Interface*, *Streaming*, *Color palette*, *Event handling per node*

**Constraints:**
- The default `rag-ask` path (no flag, or `--rag_llm`) must be byte-identical to pre-S7 behavior.
- No new packages — raw ANSI codes only.
- Colors suppressed entirely when `sys.stdout.isatty()` is `False`.

**Done when:** `rag-ask --rag_agent` streams `[N] Retrieving: …  →  K chunks` lines in the correct cycle colors, prints `Agent: N iterations (rag_agent)` footer, and `rag-ask --rag_llm` / `rag-ask` with no flag is unchanged.

---

### Task 4 — Tests

**Goal:** Write the full unit test suite. All LLM and context provider calls must be mocked — no live API calls.

**Files to read before starting:**
- `src/agents/rag_agent.py` (after Task 2)
- `src/agents/models.py` (after Task 1)
- `src/cli.py` (after Task 3)
- Any existing test in `tests/agents/` for fixture and mock conventions

**Changes:**
- `tests/agents/test_rag_agent.py` — new file covering all cases listed in *Testing Requirements*

**Relevant spec sections:** *Testing Requirements* (authoritative list of required tests)

**Done when:** `uv run pytest tests/agents/test_rag_agent.py` passes with no live API calls. Every case in *Testing Requirements* has a corresponding test.

---

### Task 5 — Documentation

**Goal:** Update `readme.md` so users know how to invoke the new agent and understand the `rag_llm` vs `rag_agent` naming.

**Files to read before starting:**
- `readme.md`

**Changes:**
- `readme.md` — add `rag-ask --rag_llm` / `rag-ask --rag_agent`, `--max-iterations`, and `YT_AGENT_RAG_AGENT_MAX_ITERATIONS` to the usage section; add a `rag_llm vs rag_agent` comparison table or note explaining the two approaches and the CLI flag that selects each

**Relevant spec sections:** *CLI Interface*, *Agent Naming*, *Acceptance Criteria* (last bullet)

**Done when:** A new user can read `readme.md` and know how to run both agents side-by-side and understand what `rag_llm` and `rag_agent` refer to.

---

## Open Questions

- Should `last_context` expose the per-iteration chunk lists (for trace inspection) in addition to the union? S7 keeps it as the flat union to match the `RagTranscriptAgent` interface; per-iteration detail can be added in an eval spec.
- Should the system prompt cap the number of `retrieve_transcript_chunks` calls via instruction (e.g. "call at most 5 times") in addition to the hard `max_iterations` guard? A soft instruction helps the model self-pace; the hard guard is the safety net. This is a prompt-tuning question best answered by eval runs.
