from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings
from src.rag.models import TranscriptChunk
from src.rag.storage import RawTranscriptStore, TranscriptChunkStore, raw_document_from_transcript
from src.transcripts.models import Transcript

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backfill_chunk_metadata.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("backfill_chunk_metadata", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeEmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float("agent" in text.lower()), 1.0]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        superdata_api_key="super",
        deepseek_api_key="deep",
        deepseek_model="deepseek-v4",
        deepseek_base_url=None,
        chroma_path=tmp_path / "chroma",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
        mlflow_experiment_name="test-backfill",
        log_transcript_artifacts=False,
    )


def _seed(tmp_path: Path) -> Settings:
    settings = _settings(tmp_path)

    transcript = Transcript(
        video_id="vid1",
        url="https://www.youtube.com/watch?v=vid1",
        title="Video Title",
        channel_id="channel-1",
        channel_name="Channel One",
        upload_date="2026-01-01",
        raw_text="some spoken text",
        fetched_at=datetime.now(timezone.utc),
    )
    raw_store = RawTranscriptStore(settings.chroma_path, collection_name=settings.raw_transcript_collection)
    raw_store.upsert_raw_document(raw_document_from_transcript(transcript))

    chunk_store = TranscriptChunkStore(
        settings.chroma_path,
        FakeEmbeddingModel(),
        collection_name=settings.chunk_collection,
    )
    chunk_store.upsert_chunks(
        [
            TranscriptChunk(
                transcript_id="raw_transcript:vid1",
                video_id="vid1",
                source_url="https://www.youtube.com/watch?v=vid1",
                chunk_index=0,
                text="some spoken text",
                start_seconds=1.0,
                end_seconds=5.0,
                start_segment_index=0,
                end_segment_index=0,
                segment_count=1,
            )
        ]
    )
    return settings


def _stored_embedding(settings: Settings) -> list[float]:
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_collection(settings.chunk_collection)
    result = collection.get(ids=["chunk:vid1:0"], include=["embeddings"])
    return list(result["embeddings"][0])


def _stored_metadata(settings: Settings) -> dict:
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_collection(settings.chunk_collection)
    result = collection.get(ids=["chunk:vid1:0"], include=["metadatas"])
    return dict(result["metadatas"][0] or {})


def test_re_embed_rebuilds_vectors_after_metadata_already_stamped(monkeypatch, tmp_path) -> None:
    module = _load_module()
    settings = _seed(tmp_path)
    monkeypatch.setattr(module, "load_settings", lambda require_keys=True: settings)
    monkeypatch.setattr("src.rag.embeddings.HuggingFaceEmbeddingModel", FakeEmbeddingModel)

    result = module.main([])
    assert result == 0
    stamped_meta = _stored_metadata(settings)
    assert stamped_meta.get("channel_name") == "Channel One"
    assert stamped_meta.get("context_header")

    embedding_before_re_embed = _stored_embedding(settings)

    result = module.main(["--re-embed"])
    assert result == 0

    embedding_after_re_embed = _stored_embedding(settings)
    assert embedding_after_re_embed != embedding_before_re_embed


def test_plain_run_stamps_identity_metadata(monkeypatch, tmp_path) -> None:
    module = _load_module()
    settings = _seed(tmp_path)
    monkeypatch.setattr(module, "load_settings", lambda require_keys=True: settings)

    result = module.main([])

    assert result == 0
    meta = _stored_metadata(settings)
    assert meta.get("channel_id") == "channel-1"
    assert meta.get("channel_name") == "Channel One"
    assert meta.get("title") == "Video Title"
    assert meta.get("upload_date") == "2026-01-01"


def test_plain_run_is_noop_on_second_pass(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    settings = _seed(tmp_path)
    monkeypatch.setattr(module, "load_settings", lambda require_keys=True: settings)

    assert module.main([]) == 0
    embedding_after_first_run = _stored_embedding(settings)

    assert module.main([]) == 0
    embedding_after_second_run = _stored_embedding(settings)

    assert "nothing to do" in capsys.readouterr().out
    assert embedding_after_second_run == embedding_after_first_run
