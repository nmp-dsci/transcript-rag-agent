"""Compatibility shim so ragas imports against langchain-community >= 0.4.

ragas 0.4 imports the legacy Vertex AI classes at module import time purely
for isinstance checks in its token-usage handling; langchain-community 0.4
removed those integrations. Registering inert stand-ins keeps the real ragas
metrics importable without pulling the Google SDK stack. The stand-ins are
never instantiated — isinstance against them is simply always False, which is
correct because this project never uses Vertex AI models.

Call :func:`install` before any ``import ragas``.
"""

from __future__ import annotations

import sys
import types


class _VertexSentinel:
    """Placeholder class used only so ragas isinstance checks return False."""


def install() -> None:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except Exception:
        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = _VertexSentinel  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = module
        try:
            import langchain_community.chat_models as chat_models

            chat_models.vertexai = module  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:
        try:
            import langchain_community.llms as llms

            llms.VertexAI = _VertexSentinel  # type: ignore[attr-defined]
        except Exception:
            pass
