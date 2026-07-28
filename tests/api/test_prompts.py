"""The Prompts tab serves the live registry, grouped and complete."""

from __future__ import annotations

from src.agents import prompts as prompt_module
from src.api.prompts import SYSTEMS, load_prompts


def test_every_registry_prompt_lands_in_exactly_one_group() -> None:
    payload = load_prompts()
    served = [prompt["name"] for system in payload["systems"] for prompt in system["prompts"]]
    registry_names = [entry["name"] for entry in prompt_module.PROMPT_REGISTRY]
    for name in registry_names:
        assert served.count(name) == 1, name
    assert payload["total"] == len(served)


def test_group_keys_cover_the_registry_systems() -> None:
    group_keys = {system["key"] for system in SYSTEMS}
    registry_systems = {entry["system"] for entry in prompt_module.PROMPT_REGISTRY}
    assert registry_systems <= group_keys


def test_prompt_text_is_the_live_constant() -> None:
    payload = load_prompts()
    by_name = {
        prompt["name"]: prompt for system in payload["systems"] for prompt in system["prompts"]
    }
    assert by_name["RAG_SYSTEM_PROMPT"]["text"] is prompt_module.RAG_SYSTEM_PROMPT
    assert by_name["SUMMARY_SYSTEM_PROMPT"]["module"] == "src/rag/summaries.py"
    graph = [p for p in by_name.values() if p["system"] == "graph_rag"]
    assert len(graph) >= 4  # extraction, community, router, answer paths
