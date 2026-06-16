from __future__ import annotations

from src import cli
from src.chat import session as session_module
from src.chat.history import load_history
from src.chat.session import run_session
from src.chat.setups import SetupResult


class ScriptedInput:
    """Return queued answers in order, raising EOFError when exhausted."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)

    def __call__(self, prompt: str = "") -> str:
        if not self._answers:
            raise EOFError
        return self._answers.pop(0)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list = []

    def run_many(self, keys, question, *, url=None, top_k=None, on_progress=None):
        self.calls.append((keys, question, url))
        if on_progress:
            on_progress("Running ...")
        return [
            SetupResult(
                key=key,
                title=f"{key} title",
                command=f"cmd {key}",
                answer=f"answer from {key}",
                token_estimate=10,
                chunk_count=2,
            )
            for key in keys
        ]


def _paths(tmp_path):
    return tmp_path / "chat_history.json", tmp_path / "chat.html"


def test_ask_flow_captures_history_and_writes_html(settings, tmp_path) -> None:
    history_path, chat_html_path = _paths(tmp_path)
    runner = FakeRunner()
    outputs: list[str] = []
    inputs = ScriptedInput(
        [
            "1",  # main menu -> ask
            "How do agents retrieve?",  # question
            "",  # no url restriction
            "1,3",  # setups
            "q",  # quit
        ]
    )

    code = run_session(
        settings,
        input_fn=inputs,
        output=outputs.append,
        history_path=history_path,
        chat_html_path=chat_html_path,
        runner_factory=lambda: runner,
        index_fn=lambda argv: 0,
    )

    assert code == 0
    assert runner.calls == [(["rag_llm", "rag_agent"], "How do agents retrieve?", None)]
    entries = load_history(history_path)
    assert len(entries) == 1
    assert entries[0].question == "How do agents retrieve?"
    assert [a.key for a in entries[0].answers] == ["rag_llm", "rag_agent"]
    assert chat_html_path.exists()
    assert any("Captured 2 answer(s)" in line for line in outputs)


def test_ask_flow_invalid_then_valid_setup(settings, tmp_path) -> None:
    history_path, chat_html_path = _paths(tmp_path)
    runner = FakeRunner()
    inputs = ScriptedInput(["1", "Q", "", "nope", "a", "q"])

    run_session(
        settings,
        input_fn=inputs,
        output=lambda _line: None,
        history_path=history_path,
        chat_html_path=chat_html_path,
        runner_factory=lambda: runner,
        index_fn=lambda argv: 0,
    )

    assert runner.calls[0][0] == ["rag_llm", "rag_llm_recursive", "rag_agent"]


def test_fetch_single_url_invokes_index(settings, tmp_path) -> None:
    history_path, chat_html_path = _paths(tmp_path)
    invoked: list = []
    inputs = ScriptedInput(["2", "1", "https://youtu.be/abc", "q"])

    run_session(
        settings,
        input_fn=inputs,
        output=lambda _line: None,
        history_path=history_path,
        chat_html_path=chat_html_path,
        runner_factory=lambda: FakeRunner(),
        index_fn=lambda argv: invoked.append(argv) or 0,
    )

    assert invoked == [["index-rag", "https://youtu.be/abc"]]


def test_fetch_bulk_channel_invokes_index(settings, tmp_path) -> None:
    history_path, chat_html_path = _paths(tmp_path)
    invoked: list = []
    inputs = ScriptedInput(["2", "2", "@channel", "", "q"])

    run_session(
        settings,
        input_fn=inputs,
        output=lambda _line: None,
        history_path=history_path,
        chat_html_path=chat_html_path,
        runner_factory=lambda: FakeRunner(),
        index_fn=lambda argv: invoked.append(argv) or 0,
    )

    assert invoked == [
        ["bulk-index", "channel", "--channel", "@channel", "--latest", "5"]
    ]


def test_quit_immediately_returns_zero(settings, tmp_path) -> None:
    history_path, chat_html_path = _paths(tmp_path)
    code = run_session(
        settings,
        input_fn=ScriptedInput(["q"]),
        output=lambda _line: None,
        history_path=history_path,
        chat_html_path=chat_html_path,
        runner_factory=lambda: FakeRunner(),
        index_fn=lambda argv: 0,
    )
    assert code == 0


def test_cli_chat_dispatches_to_session(monkeypatch, settings) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda require_keys=True: settings)
    captured: dict = {}

    def fake_run_session(passed_settings) -> int:
        captured["settings"] = passed_settings
        return 0

    monkeypatch.setattr(session_module, "run_session", fake_run_session)

    assert cli.main(["chat"]) == 0
    assert captured["settings"] is settings
