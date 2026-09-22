from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, load_settings


def test_loads_settings_from_env_file(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "SUPERDATA_API_KEY=super",
                "DEEPSEEK_API_KEY=deep",
                "DEEPSEEK_MODEL=deepseek-v4",
                "YT_AGENT_CHROMA_PATH=.cache/chroma",
                "SUPADATA_TIMEOUT_SECONDS=150",
                "SUPADATA_POLL_INTERVAL_SECONDS=3",
                "SUPADATA_MAX_POLL_SECONDS=900",
                "YT_AGENT_RAG_RECURSIVE_DEFAULT=true",
                "YT_AGENT_RAG_MAX_DEPTH=1",
                "YT_AGENT_RAG_MAX_FOLLOWUPS=4",
                "YT_AGENT_RAG_FOLLOWUP_TOP_K=6",
                "YT_AGENT_RAG_NOVELTY_MIN_CHUNKS=1",
                "YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS=5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(env))
    monkeypatch.delenv("SUPERDATA_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = load_settings()

    assert settings.superdata_api_key == "super"
    assert settings.deepseek_api_key == "deep"
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.chroma_path.name == "chroma"
    assert settings.supadata_timeout_seconds == 150
    assert settings.supadata_poll_interval_seconds == 3
    assert settings.supadata_max_poll_seconds == 900
    assert settings.rag_recursive_default is True
    assert settings.rag_max_depth == 1
    assert settings.rag_max_followups == 4
    assert settings.rag_followup_top_k == 6
    assert settings.rag_novelty_min_chunks == 1
    assert settings.rag_max_total_followups == 5


def test_accepts_supadata_api_key_alias(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "SUPADATA_API_KEY=super",
                "DEEPSEEK_API_KEY=deep",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(env))
    monkeypatch.delenv("SUPERDATA_API_KEY", raising=False)
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = load_settings()

    assert settings.superdata_api_key == "super"


def test_missing_env_file_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(tmp_path / "missing.env"))

    with pytest.raises(ConfigError):
        load_settings()


def test_numbered_supadata_keys_form_an_ordered_fallback_list(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "SUPADATA_API_KEY=first",
                "SUPADATA_API_KEY_2=second",
                "SUPADATA_API_KEY_3=  third  ",
                "SUPADATA_API_KEY_5=never-reached",
                "DEEPSEEK_API_KEY=deep",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(env))
    for name in ("SUPERDATA_API_KEY", "SUPADATA_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for index in range(2, 7):
        monkeypatch.delenv(f"SUPADATA_API_KEY_{index}", raising=False)

    settings = load_settings()

    assert settings.superdata_api_key == "first"
    # _4 is absent, so _5 is not read: the numbering stops at the first gap.
    assert settings.supadata_api_keys == ("first", "second", "third")


def test_single_key_is_a_one_entry_ring(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SUPERDATA_API_KEY=only\nDEEPSEEK_API_KEY=deep\n", encoding="utf-8")
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(env))
    for name in ("SUPERDATA_API_KEY", "SUPADATA_API_KEY", "SUPADATA_API_KEY_2", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert load_settings().supadata_api_keys == ("only",)


def test_demo_mode_without_keys_has_an_empty_ring(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YT_AGENT_ENV_PATH", str(tmp_path / "missing.env"))
    monkeypatch.setenv("YT_AGENT_DEMO_MODE", "1")
    for name in ("SUPERDATA_API_KEY", "SUPADATA_API_KEY", "SUPADATA_API_KEY_2"):
        monkeypatch.delenv(name, raising=False)

    assert load_settings().supadata_api_keys == ()


def test_the_web_collections_are_separate_from_the_transcript_ones(monkeypatch) -> None:
    # The separation is the whole point: transcript_chunks is the collection
    # the committed eval snapshots were measured against.
    monkeypatch.setenv("SUPADATA_API_KEY", "k1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    settings = load_settings()
    assert settings.web_source_collection == "web_sources"
    assert settings.web_chunk_collection == "web_chunks"
    assert settings.web_chunk_collection != settings.chunk_collection
    assert settings.web_source_collection != settings.raw_transcript_collection


def test_the_web_collections_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("SUPADATA_API_KEY", "k1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("YT_AGENT_WEB_CHUNK_COLLECTION", "web_chunks_v2")
    assert load_settings().web_chunk_collection == "web_chunks_v2"


def test_the_channels_file_defaults_to_the_repo_convention(monkeypatch) -> None:
    # None means "use channels.yaml at the repo root", resolved by the
    # registry rather than baked into settings.
    monkeypatch.setenv("SUPADATA_API_KEY", "k1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.delenv("YT_AGENT_CHANNELS_FILE", raising=False)
    assert load_settings().channels_file is None


def test_a_relative_channels_file_resolves_against_the_project_root(monkeypatch) -> None:
    monkeypatch.setenv("SUPADATA_API_KEY", "k1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("YT_AGENT_CHANNELS_FILE", "custom-channels.yaml")
    resolved = load_settings().channels_file
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.name == "custom-channels.yaml"
