"""The Prompts tab's data source: the live prompt registry, grouped by system.

Serves :data:`src.agents.prompts.PROMPT_REGISTRY` — the same constants the
agents import — plus the one prompt that lives outside that module
(``src.rag.summaries.SUMMARY_SYSTEM_PROMPT``, appended here to keep
``prompts.py`` free of rag imports). Because the API reads the running
constants directly, the tab cannot drift from what the engines actually send.

RAGAS judge prompts live inside the ``ragas`` library rather than this repo,
so they are represented by a link-out note instead of duplicated text. The
``depth-v2`` rubric's depth prompt *is* ours, so it is served like any other.
"""

from __future__ import annotations

from typing import Any

from src.agents.prompts import PROMPT_REGISTRY
from src.evals.judge import DEPTH_JUDGE_SYSTEM_PROMPT, DEPTH_JUDGE_USER_PROMPT
from src.rag.summaries import SUMMARY_SYSTEM_PROMPT

#: Tab grouping metadata, in display order.
SYSTEMS: list[dict[str, str]] = [
    {
        "key": "chat",
        "title": "Chat — direct transcript",
        "description": "Q&A and summarization over one full transcript.",
    },
    {
        "key": "vector_rag",
        "title": "Vector RAG — single-hop",
        "description": "One retrieval across the corpus, then a single answer call "
        "(rag_llm). Includes the history and query-rewrite prompts.",
    },
    {
        "key": "recursive_rag",
        "title": "Recursive RAG",
        "description": "Multi-hop follow-up retrieval and the final synthesis call "
        "(rag_llm --recursive).",
    },
    {
        "key": "agentic_rag",
        "title": "Agentic RAG — LangGraph ReAct",
        "description": "The research-loop system prompt behind rag_agent.",
    },
    {
        "key": "summary_filter",
        "title": "Summary filter",
        "description": "Per-video summaries used to pre-filter retrieval scope.",
    },
    {
        "key": "graph_rag",
        "title": "GraphRAG — knowledge graph (P4)",
        "description": "Entity/claim extraction, community summaries, the "
        "local/global/temporal router, and the graph answer paths.",
    },
    {
        "key": "doc_review",
        "title": "Document review — a shared page against the corpus",
        "description": "The prompt behind a chat message that contains a URL: "
        "the document supplies what is being reviewed, the corpus supplies the "
        "criteria, and the two are cited differently.",
    },
    {
        "key": "depth_judge",
        "title": "Depth judge — the depth-v2 rubric",
        "description": "The five depth metrics (insight, specificity, coverage, "
        "evidence breadth, calibration) that make up 60% of a depth-v2 "
        "composite, scored in one structured call per answer.",
    },
    {
        "key": "retrieval_variants",
        "title": "Retrieval variants — HyDE, multi-query, contextual",
        "description": "The prompts that change what gets embedded rather than "
        "how the answer is written: HyDE's hypothetical passage, multi-query's "
        "paraphrases, and the per-chunk situating sentence behind "
        "index-contextual.",
    },
]

_EXTRA_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "SUMMARY_SYSTEM_PROMPT",
        "system": "summary_filter",
        "role": "system",
        "template_vars": [],
        "text": SUMMARY_SYSTEM_PROMPT,
        "module": "src/rag/summaries.py",
    },
    {
        "name": "DEPTH_JUDGE_SYSTEM_PROMPT",
        "system": "depth_judge",
        "role": "system",
        "template_vars": [],
        "text": DEPTH_JUDGE_SYSTEM_PROMPT,
        "module": "src/evals/judge.py",
    },
    {
        "name": "DEPTH_JUDGE_USER_PROMPT",
        "system": "depth_judge",
        "role": "user_template",
        "template_vars": ["question", "context_count", "contexts_block", "answer"],
        "text": DEPTH_JUDGE_USER_PROMPT,
        "module": "src/evals/judge.py",
    },
]

_NOTES = [
    "RAGAS judge prompts (faithfulness, answer relevancy, context precision) "
    "live inside the ragas library, not this repo — see "
    "https://github.com/explodinggradients/ragas.",
    "The depth-v2 rubric's five depth metrics are judged by this repo's own "
    "prompt, shown above — one structured call per answer, scored 0-1 with a "
    "one-sentence reason each.",
]

#: Fields PromptsView.tsx reads off every prompt entry. Validated at load time
#: so a malformed registry entry fails fast server-side instead of throwing in
#: the frontend (e.g. ``prompt.role.replace(...)`` on a missing ``role``).
_REQUIRED_FIELDS = ("name", "system", "role", "template_vars", "text")


def _validate_prompt_shape(entry: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in _REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(
            f"prompt entry {entry.get('name', '<unnamed>')!r} is missing required "
            f"field(s): {', '.join(missing)}"
        )
    return entry


def load_prompts() -> dict[str, Any]:
    """The registry grouped by system, in SYSTEMS display order."""
    prompts: list[dict[str, Any]] = [
        _validate_prompt_shape({**entry, "module": "src/agents/prompts.py"})
        for entry in PROMPT_REGISTRY
    ] + [_validate_prompt_shape(entry) for entry in _EXTRA_PROMPTS]
    groups = []
    for system in SYSTEMS:
        members = [prompt for prompt in prompts if prompt["system"] == system["key"]]
        groups.append({**system, "prompts": members, "count": len(members)})
    return {"systems": groups, "total": len(prompts), "notes": _NOTES}
