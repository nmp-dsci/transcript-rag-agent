from __future__ import annotations

from src.chat.frontend import render_chat_html, write_chat_html
from src.chat.history import build_entry
from src.chat.setups import SetupResult


def _entry(question: str = "What is agentic RAG?", answer: str = "It loops."):
    result = SetupResult(
        key="rag_agent",
        title="rag_agent (agentic)",
        command="uv run python -m src.cli rag-ask ... --rag_agent",
        answer=answer,
        token_estimate=200,
        chunk_count=5,
        iterations=3,
        elapsed_seconds=2.0,
    )
    return build_entry(question, [result])


def test_render_includes_question_and_setup_title() -> None:
    html = render_chat_html([_entry()])
    assert "Transcript RAG Chat" in html
    assert "What is agentic RAG?" in html
    assert "rag_agent (agentic)" in html
    assert '"conversations"' in html


def test_render_escapes_script_terminator() -> None:
    html = render_chat_html([_entry(answer="break </script> me")])
    # The embedded JSON must not contain a raw closing script tag.
    assert "<\\/script>" in html


def test_render_handles_empty_history() -> None:
    html = render_chat_html([])
    assert "0 question(s)" in html
    assert '"conversations"' in html


def test_render_ships_readability_renderer() -> None:
    # The page renders answers client-side; assert the readability features and
    # their styling are present in the generated document.
    html = render_chat_html([_entry()])
    for marker in [
        "function cleanAnswer",  # strips embedded JSON / preamble noise
        "function extractJsonAnswer",  # prefers the canonical JSON answer field
        "function renderAnswer",  # markdown -> headings/lists/paragraphs
        "function toggleClamp",  # progressive disclosure
        ".ans-section.summary",  # Key Findings callout styling
        "a.cite",  # clickable inline citation styling
        ".body.clamp",  # long-answer clamp styling
    ]:
        assert marker in html, marker


def test_render_omits_the_trace_this_page_cannot_show() -> None:
    """Only the React Chat tab renders traces; embedding them here is dead weight."""
    result = SetupResult(
        key="rag_llm",
        title="rag_llm (single-hop)",
        command="uv run python -m src.cli rag-ask ... --rag_llm",
        answer="An answer.",
        trace=[{"phase": "retrieve", "label": "Retrieve candidates", "detail": "30 candidates"}],
    )
    html = render_chat_html([build_entry("q", [result])])

    assert "Retrieve candidates" not in html
    assert '"trace"' not in html
    # The rest of the answer payload is untouched.
    assert "rag_llm (single-hop)" in html


def test_write_chat_html_creates_file(tmp_path) -> None:
    path = tmp_path / "out" / "chat.html"
    written = write_chat_html([_entry()], path)
    assert written == path
    assert path.exists()
    assert "Transcript RAG Chat" in path.read_text(encoding="utf-8")
