from __future__ import annotations

from src.rag.contextualize import (
    ChunkContextualizer,
    build_contextual_index,
    context_cache_key,
    document_window,
    situate,
)
from src.rag.models import TranscriptChunk


def _chunk(index: int, text: str = "spoken words", header: str | None = None) -> TranscriptChunk:
    return TranscriptChunk(
        transcript_id="raw_transcript:video",
        video_id="video",
        source_url="https://www.youtube.com/watch?v=video",
        chunk_index=index,
        text=text,
        segment_count=1,
        title="A video",
        channel_name="A channel",
        context_header=header,
    )


class FakeLlm:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.prompts: list[str] = []

    def invoke(self, messages: list):
        self.prompts.append(str(messages[-1].content))

        class Response:
            content = self.contents.pop(0) if self.contents else "a situating sentence"

        return Response()


class BrokenLlm:
    def invoke(self, messages: list):
        raise RuntimeError("upstream is down")


class FakeStore:
    def __init__(
        self,
        chunks: list[TranscriptChunk] | None = None,
        stale: dict[str, list[str]] | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.upserted: list[TranscriptChunk] = []
        #: ``video_id -> chunk ids already stored that the write will not cover``,
        #: i.e. the tail a shrinking re-chunk left behind in this mirror.
        self.stale = stale or {}
        self.replaced: list[str] = []

    def all_chunks(self) -> list[TranscriptChunk]:
        return list(self.chunks)

    def upsert_chunks(self, chunks: list[TranscriptChunk]) -> None:
        self.upserted.extend(chunks)

    def replace_chunks(self, video_id: str, chunks: list[TranscriptChunk]) -> list[str]:
        self.upserted.extend(chunks)
        removed = self.stale.pop(video_id, [])
        self.replaced.extend(removed)
        return removed


# ── the situating call ────────────────────────────────────────────────────────


def test_the_prompt_carries_the_chunk_and_its_surrounding_transcript() -> None:
    llm = FakeLlm()

    ChunkContextualizer(llm).contextualize(_chunk(1, "the chunk"), "before the chunk after")

    prompt = llm.prompts[0]
    assert "the chunk" in prompt
    assert "before the chunk after" in prompt
    assert "A video" in prompt


def test_a_failed_call_is_recorded_as_an_error_not_an_empty_context() -> None:
    result = ChunkContextualizer(BrokenLlm()).contextualize(_chunk(0), "excerpt")

    assert result.error is not None
    assert result.context == ""


# ── the situating sentence's place in the chunk ───────────────────────────────


def test_situating_keeps_the_deterministic_header_and_adds_to_it() -> None:
    """The two say different things — which video, versus what about it."""
    situated = situate(_chunk(0, header="[A channel — A video @ 00:10-00:40]"), "About the cap.")

    assert situated.context_header == "[A channel — A video @ 00:10-00:40]\nAbout the cap."


def test_situating_never_touches_the_spoken_text() -> None:
    """Only the embedding changes, so the answer and its citations do not."""
    situated = situate(_chunk(0, "what was actually said"), "About the cap.")

    assert situated.text == "what was actually said"
    assert "About the cap." in situated.embedding_text
    assert situated.embedding_text.endswith("what was actually said")


def test_a_chunk_with_no_header_gets_the_situating_sentence_alone() -> None:
    assert situate(_chunk(0), "About the cap.").context_header == "About the cap."


# ── the document window ───────────────────────────────────────────────────────


def test_the_window_is_centred_on_the_chunk_being_situated() -> None:
    chunks = [_chunk(index, f"part{index}") for index in range(5)]

    window = document_window(chunks, 2, window_chars=20)

    assert "part2" in window
    assert "part1" in window and "part3" in window
    assert "part0" not in window


def test_the_window_takes_what_it_can_at_the_start_of_a_video() -> None:
    chunks = [_chunk(index, f"part{index}") for index in range(5)]

    window = document_window(chunks, 0, window_chars=1000)

    assert window.startswith("part0")


def test_the_window_is_never_longer_than_its_budget() -> None:
    chunks = [_chunk(index, "x" * 500) for index in range(10)]

    assert len(document_window(chunks, 5, window_chars=600)) <= 600


def test_an_empty_video_has_an_empty_window() -> None:
    assert document_window([], 0) == ""


# ── caching ───────────────────────────────────────────────────────────────────


def test_a_cached_chunk_costs_no_second_call(tmp_path) -> None:
    llm = FakeLlm("about the cap", "a different sentence")
    contextualizer = ChunkContextualizer(llm, cache_dir=tmp_path)

    first = contextualizer.contextualize(_chunk(0), "excerpt")
    second = contextualizer.contextualize(_chunk(0), "excerpt")

    assert len(llm.prompts) == 1
    assert second.context == first.context == "about the cap"
    assert second.cached is True and first.cached is False


def test_editing_a_chunks_text_invalidates_its_cached_context(tmp_path) -> None:
    contextualizer = ChunkContextualizer(FakeLlm("first", "second"), cache_dir=tmp_path)

    contextualizer.contextualize(_chunk(0, "original text"), "excerpt")
    result = contextualizer.contextualize(_chunk(0, "edited text"), "excerpt")

    assert result.context == "second"


def test_the_cache_key_tracks_both_the_chunk_id_and_its_text() -> None:
    assert context_cache_key(_chunk(0, "a")) != context_cache_key(_chunk(0, "b"))
    assert context_cache_key(_chunk(0, "a")) != context_cache_key(_chunk(1, "a"))


def test_a_failure_is_not_cached_so_the_chunk_is_retried(tmp_path) -> None:
    assert (
        ChunkContextualizer(BrokenLlm(), cache_dir=tmp_path)
        .contextualize(_chunk(0), "excerpt")
        .error
    )

    result = ChunkContextualizer(FakeLlm("about the cap"), cache_dir=tmp_path).contextualize(
        _chunk(0), "excerpt"
    )

    assert result.context == "about the cap"


# ── the indexing pass ─────────────────────────────────────────────────────────


def test_every_chunk_reaches_the_contextual_index_situated() -> None:
    source = FakeStore([_chunk(0, "one"), _chunk(1, "two")])
    target = FakeStore()

    result = build_contextual_index(source, target, ChunkContextualizer(FakeLlm()))

    assert result.chunks == 2 and result.contextualized == 2 and result.failed == 0
    assert [chunk.chunk_index for chunk in target.upserted] == [0, 1]
    assert all(chunk.context_header for chunk in target.upserted)


def test_a_chunk_whose_situating_failed_is_still_indexed_unsituated() -> None:
    """A missing chunk would depress recall and read as a retrieval result."""
    source = FakeStore([_chunk(0, "one")])
    target = FakeStore()

    result = build_contextual_index(source, target, ChunkContextualizer(BrokenLlm()))

    assert result.failed == 1 and result.contextualized == 0
    assert [chunk.chunk_index for chunk in target.upserted] == [0]
    assert target.upserted[0].context_header is None


def test_max_chunks_bounds_a_smoke_test_run() -> None:
    source = FakeStore([_chunk(index) for index in range(5)])
    target = FakeStore()

    result = build_contextual_index(source, target, ChunkContextualizer(FakeLlm()), max_chunks=2)

    assert result.chunks == 2
    assert len(target.upserted) == 2


def test_chunks_are_situated_per_video_so_the_window_is_their_own() -> None:
    first = _chunk(0, "video one text")
    second = TranscriptChunk(
        transcript_id="raw_transcript:other",
        video_id="other",
        source_url="https://www.youtube.com/watch?v=other",
        chunk_index=0,
        text="video two text",
        segment_count=1,
    )
    llm = FakeLlm()

    build_contextual_index(FakeStore([first, second]), FakeStore(), ChunkContextualizer(llm))

    assert len(llm.prompts) == 2
    by_video = {"video one text" in prompt: prompt for prompt in llm.prompts}
    assert "video two text" not in by_video[True]


def test_a_transient_empty_response_is_retried_once() -> None:
    """Both failure modes seen in practice are transient, so one retry is
    the difference between a situated chunk and an unsituated one."""
    llm = FakeLlm("", "about the cap")

    result = ChunkContextualizer(llm).contextualize(_chunk(0), "excerpt")

    assert result.context == "about the cap"
    assert len(llm.prompts) == 2


def test_a_call_that_raises_is_retried_once() -> None:
    class FlakyLlm:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection reset")

            class Response:
                content = "about the cap"

            return Response()

    llm = FlakyLlm()

    assert ChunkContextualizer(llm).contextualize(_chunk(0), "excerpt").context == "about the cap"
    assert llm.calls == 2


def test_two_empty_responses_are_reported_as_a_failure() -> None:
    result = ChunkContextualizer(FakeLlm("", "")).contextualize(_chunk(0), "excerpt")

    assert result.error == "empty context"
    assert result.context == ""


def test_a_full_pass_drops_contextual_chunks_the_baseline_no_longer_has() -> None:
    """The contextual index is a mirror; a stale tail here is a chunk the
    contextual arm of an ablation can retrieve and the baseline arm cannot."""
    source = FakeStore([_chunk(index) for index in range(3)])
    target = FakeStore(stale={"video": ["chunk:video:3", "chunk:video:4"]})

    result = build_contextual_index(source, target, ChunkContextualizer(FakeLlm()))

    assert target.replaced == ["chunk:video:3", "chunk:video:4"]
    assert result.removed == 2
    assert "2 stale removed" in result.summary()


def test_a_max_chunks_pass_never_deletes_from_the_mirror() -> None:
    """It reads a prefix, so its last video is partial — replacing would delete
    real chunks on the strength of a truncation."""
    source = FakeStore([_chunk(index) for index in range(5)])
    target = FakeStore(stale={"video": ["chunk:video:4"]})

    result = build_contextual_index(source, target, ChunkContextualizer(FakeLlm()), max_chunks=2)

    assert target.replaced == []
    assert result.removed == 0
    assert len(target.upserted) == 2
