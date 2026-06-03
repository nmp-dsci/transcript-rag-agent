from __future__ import annotations

from src.agents.models import RagAnswerReference
from src.evals.evaluation import (
    SETUP_COMMANDS,
    AgentRun,
    _title_from_command,
    render_html_report,
)


def test_title_from_command_uses_flags_after_question() -> None:
    assert _title_from_command(SETUP_COMMANDS[0]) == "--rag_llm --top-k 30"
    assert _title_from_command(SETUP_COMMANDS[1]) == "--rag_llm --recursive --top-k 10"
    assert _title_from_command(SETUP_COMMANDS[2]) == "--rag_agent --top-k 10"


def test_render_html_report_three_columns_question_command_and_answer() -> None:
    reference = RagAnswerReference(
        label="[1]",
        source_url="https://www.youtube.com/watch?v=abc",
        timestamp_url="https://www.youtube.com/watch?v=abc&t=10s",
        start_seconds=10,
        end_seconds=20,
        chunk_index=1,
        video_id="abc",
    )
    runs = [
        AgentRun(
            title=_title_from_command(SETUP_COMMANDS[0]),
            command=SETUP_COMMANDS[0],
            answer="rag_llm answer",
            references=[reference],
            token_estimate=120,
            chunk_count=30,
            llm_calls=1,
        ),
        AgentRun(
            title=_title_from_command(SETUP_COMMANDS[1]),
            command=SETUP_COMMANDS[1],
            answer="rag_llm recursive answer",
            token_estimate=80,
            chunk_count=10,
            llm_calls=4,
            terminated_reason="max_depth_reached",
        ),
        AgentRun(
            title=_title_from_command(SETUP_COMMANDS[2]),
            command=SETUP_COMMANDS[2],
            answer="## Key Findings\n1. agentic answer",
            token_estimate=60,
            chunk_count=18,
            iterations=5,
            terminated_reason="completed",
        ),
    ]

    output = render_html_report(question="how is agentic coding used", runs=runs)

    # Question shown at the top.
    assert "how is agentic coding used" in output
    # Three columns, one per setup, titled from the command flags.
    assert output.count('class="answer-col"') == 3
    assert "--rag_llm --top-k 30" in output
    assert "--rag_llm --recursive --top-k 10" in output
    assert "--rag_agent --top-k 10" in output
    # The full bash command is shown inside an expandable details element.
    assert "<details><summary>Command</summary>" in output
    assert "src.cli rag-ask" in output
    # Answers and per-setup metadata are rendered.
    assert "rag_llm answer" in output
    assert "## Key Findings" in output
    assert "iterations 5" in output
    assert "LLM calls 4" in output
    # References carry the traceable timestamp link.
    assert "https://www.youtube.com/watch?v=abc&amp;t=10s" in output
    # Dark theme is applied.
    assert "color-scheme:dark" in output
