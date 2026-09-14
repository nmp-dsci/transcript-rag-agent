from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("claude_agent_sdk")

from src.guides.tools import (
    QUALIFIED_TOOL,
    RetrievalLog,
    build_retrieval_server,
    build_retrieval_tool,
    describe_tool_call,
)


def test_retrieve_chunks_tool_scopes_logs_and_formats():
    calls = []

    def retrieve(question, video_ids, top_k):
        calls.append((question, video_ids, top_k))
        return [{"video_id": "v1", "chunk_index": 2, "text": "the words", "title": "T"}]

    log = RetrievalLog()
    seen = []
    tool = build_retrieval_tool(
        retrieve, default_video_ids=["v1", "v2"], log=log, on_call=seen.append
    )
    assert tool.name == "retrieve_chunks"
    assert QUALIFIED_TOOL == "mcp__corpus__retrieve_chunks"
    handler = tool.handler

    out = asyncio.run(handler({"question": "why?"}))
    assert "### v1@2 · T" in out["content"][0]["text"]
    assert calls[-1] == ("why?", ["v1", "v2"], 10)
    asyncio.run(handler({"question": "whole corpus", "video_ids": [], "top_k": 99}))
    assert calls[-1] == ("whole corpus", None, 30)
    asyncio.run(handler({"question": "one", "video_ids": ["v9"], "top_k": 3}))
    assert calls[-1] == ("one", ["v9"], 3)
    assert len(log.calls) == 3 and log.calls[0]["videos"] == ["v1"]
    assert seen[0]["tool"] == "retrieve_chunks" and seen[0]["results"] == 1
    assert asyncio.run(handler({"question": ""}))["is_error"] is True
    server = build_retrieval_server(retrieve, default_video_ids=None)
    assert server["type"] == "sdk" and server["name"] == "corpus"


def test_describe_tool_call():
    assert describe_tool_call("Read", {"file_path": "corpus/a.md"}) == "Read corpus/a.md"
    assert describe_tool_call("Grep", {"pattern": "judge"}) == "Grep judge"
    assert (
        describe_tool_call("mcp__corpus__retrieve_chunks", {"question": "q"})
        == 'retrieve_chunks "q"'
    )
    assert describe_tool_call("WebSearch", {"query": "x"}) == "WebSearch"
