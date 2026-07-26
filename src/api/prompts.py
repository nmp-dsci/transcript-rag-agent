"""The Prompts tab's data source: the live prompt registry, grouped by system.

Serves :data:`src.agents.prompts.PROMPT_REGISTRY` — the same constants the
agents import — plus the one prompt that lives outside that module
(``src.rag.summaries.SUMMARY_SYSTEM_PROMPT``, appended here to keep
``prompts.py`` free of rag imports). Because the API reads the running
constants directly, the tab cannot drift from what the engines actually send.

RAGAS judge prompts live inside the ``ragas`` library rather than this repo,
so they are represented by a link-out note instead of duplicated text.
"""

from __future__ import annotations

from typing import Any

from src.agents.prompts import PROMPT_REGISTRY
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
]

_NOTES = [
    "RAGAS judge prompts (faithfulness, answer relevancy, context precision) "
    "live inside the ragas library, not this repo — see "
    "https://github.com/explodinggradients/ragas."
]


def load_prompts() -> dict[str, Any]:
    """The registry grouped by system, in SYSTEMS display order."""
    prompts: list[dict[str, Any]] = [
        {**entry, "module": "src/agents/prompts.py"} for entry in PROMPT_REGISTRY
    ] + _EXTRA_PROMPTS
    groups = []
    for system in SYSTEMS:
        members = [prompt for prompt in prompts if prompt["system"] == system["key"]]
        groups.append({**system, "prompts": members, "count": len(members)})
    return {"systems": groups, "total": len(prompts), "notes": _NOTES}
