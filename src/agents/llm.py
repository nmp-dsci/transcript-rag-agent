"""How every agent builds its chat client.

One function, because the settings that reach the client have to be identical
across the agents: they answer the same questions in the same matrix, and a
difference here would show up as a difference between *engines*.

The load-bearing part is ``timeout``. The OpenAI-compatible client defaults to
no read timeout, so a request the endpoint accepts and then never answers
blocks in the socket read forever — the process sits at 0% CPU with an
ESTABLISHED connection and nothing times it out. On an interactive question
that is a hung tab; on a judged matrix run it is worse, because the run sails
past its own time budget and has to be killed by hand. Both have happened here.
A call that has produced nothing in ``YT_AGENT_LLM_TIMEOUT_SECONDS`` is not
going to, so fail it and let the retry (or the failed cell) be visible.
"""

from __future__ import annotations

from typing import Any

from src.config import Settings


def chat_model_kwargs(settings: Settings, **overrides: Any) -> dict[str, Any]:
    """The ``ChatOpenAI`` keyword arguments an agent answers with.

    ``base_url`` is omitted rather than passed as ``None`` so the client keeps
    its own default when none is configured, which is what these call sites did
    before this was shared.
    """
    kwargs: dict[str, Any] = {
        "api_key": settings.deepseek_api_key,
        "model": settings.deepseek_model,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 2,
    }
    if settings.deepseek_base_url:
        kwargs["base_url"] = settings.deepseek_base_url
    kwargs.update(overrides)
    return kwargs
