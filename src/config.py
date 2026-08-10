from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    superdata_api_key: str
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str | None
    chroma_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    log_transcript_artifacts: bool
    raw_transcript_collection: str = "raw_transcripts"
    chunk_collection: str = "transcript_chunks"
    # Contextual Retrieval's parallel index: the same chunks, embedded with an
    # LLM-written situating sentence. A separate collection on purpose — the
    # two are compared against each other, so neither may overwrite the other.
    contextual_chunk_collection: str = "transcript_chunks_contextual"
    transcript_summary_collection: str = "transcript_summaries"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 10
    transcript_filter_top_k: int = 5
    transcript_filter_min_score: float = 0.25
    rag_recursive_default: bool = False
    rag_max_depth: int = 1
    rag_max_followups: int = 3
    rag_followup_top_k: int | None = None
    rag_novelty_min_chunks: int = 2
    rag_max_total_followups: int | None = None
    rag_agent_max_iterations: int = 10
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 150
    # Retrieval strategy. "semantic" is the historical behaviour; "hybrid" fuses
    # semantic and BM25 rankings with RRF before the answer call.
    retrieval_mode: str = "semantic"
    # Candidates pulled per retriever before fusion/reranking collapse them to
    # top_k. Wider than top_k on purpose — reranking can only reorder what it sees.
    retrieval_candidates: int = 30
    # On by default: retrieve wide, then let the local cross-encoder reorder to
    # top_k. The eval-ablation harness measures the recall/NDCG lift this buys;
    # set YT_AGENT_RERANK_ENABLED=false to retrieve without it.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Adjacent chunks pasted around each hit so answers stop cutting off
    # mid-thought. 0 disables neighbour expansion.
    neighbor_span: int = 0
    # Query-side retrieval transform applied before embedding: "hyde",
    # "multi_query", or None to retrieve on the question as asked.
    query_transform: str | None = None
    # Paraphrases multi-query asks for. The original question is always
    # retrieved too, so the fan-out is one more search than this.
    multi_query_variants: int = 3
    # Read timeout (seconds) on LLM calls. Without one an OpenAI-compatible
    # endpoint that accepts the request and never answers hangs the process
    # forever, which is how a judged matrix run sails past its own time budget.
    llm_timeout_seconds: float = 120.0
    # Independent judge samples per metric; >1 reports mean plus spread.
    judge_samples: int = 1
    supadata_timeout_seconds: float = 120.0
    supadata_poll_interval_seconds: float = 2.0
    supadata_max_poll_seconds: float = 600.0
    discovery_cache_ttl_hours: float = 24.0
    # RAGAS judge model; each falls back to the DeepSeek value when unset.
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str | None = None
    # GraphRAG (P4) knowledge-graph store. Neo4j runs locally via docker-compose;
    # the graph is derived from the chunk corpus and rebuildable with index-graph,
    # so these are conveniences, not secrets worth guarding at load time.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "yt-agent-graph"
    graph_cache_dir: Path | None = None
    # Disk caches for the LLM passes the retrieval variants make: one expansion
    # per question, one situating sentence per chunk. Both are derived state —
    # deleting either only costs the calls to regenerate it.
    query_cache_dir: Path | None = None
    context_cache_dir: Path | None = None
    # RAPTOR level 2: the cross-video theme layer, written by `index-themes`.
    # A single derived JSON artifact — deleting it only costs the clustering
    # (free) plus one LLM call per theme to rebuild.
    theme_path: Path = Path(".yt-agent/themes.json")
    # The disagreement layer, written by `index-conflicts`. Also derived, but
    # unlike the theme artifact it is rebuilt from the *cached* GraphRAG claim
    # extractions plus one adjudication call per candidate pair — deleting it
    # costs those calls and nothing else.
    conflict_path: Path = Path(".yt-agent/conflicts.json")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return _project_root() / path


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


RETRIEVAL_MODES = ("semantic", "hybrid")
QUERY_TRANSFORMS = ("hyde", "multi_query")


def _retrieval_mode_env(name: str, default: str) -> str:
    value = (os.environ.get(name) or "").strip().lower()
    if not value:
        return default
    if value not in RETRIEVAL_MODES:
        raise ConfigError(f"{name} must be one of: {', '.join(RETRIEVAL_MODES)}")
    return value


def _query_transform_env(name: str) -> str | None:
    """The configured query transform, or ``None`` for no transform.

    Both an unset variable and an explicit ``none`` mean "retrieve on the
    question as asked" — the historical behaviour — so the setting can be
    turned off in an env file without deleting the line.
    """
    value = (os.environ.get(name) or "").strip().lower()
    if not value or value == "none":
        return None
    if value not in QUERY_TRANSFORMS:
        raise ConfigError(f"{name} must be one of: {', '.join(QUERY_TRANSFORMS)}, or none")
    return value


def load_settings(require_keys: bool = True) -> Settings:
    env_path = Path(os.environ.get("YT_AGENT_ENV_PATH", "~/.env")).expanduser()
    if env_path.exists():
        load_dotenv(env_path, override=False)
    elif require_keys:
        raise ConfigError(f"Missing env file: {env_path}")

    superdata_api_key = os.environ.get("SUPERDATA_API_KEY") or os.environ.get(
        "SUPADATA_API_KEY", ""
    )
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    missing: list[str] = []
    if require_keys and not superdata_api_key:
        missing.append("SUPERDATA_API_KEY or SUPADATA_API_KEY")
    if require_keys and not deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required configuration: {joined}")

    configured_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4")
    api_model = "deepseek-v4-flash" if configured_model == "deepseek-v4" else configured_model

    return Settings(
        superdata_api_key=superdata_api_key,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=api_model,
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        chroma_path=_resolve_project_path(
            os.environ.get("YT_AGENT_CHROMA_PATH", ".yt-agent/chroma")
        ),
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "file:.yt-agent/mlruns"),
        mlflow_experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "yt-agent-v1"),
        log_transcript_artifacts=_bool_env(
            os.environ.get("YT_AGENT_LOG_TRANSCRIPT_ARTIFACTS"), default=False
        ),
        raw_transcript_collection=os.environ.get(
            "YT_AGENT_RAW_TRANSCRIPT_COLLECTION", "raw_transcripts"
        ),
        chunk_collection=os.environ.get("YT_AGENT_CHUNK_COLLECTION", "transcript_chunks"),
        contextual_chunk_collection=os.environ.get(
            "YT_AGENT_CONTEXTUAL_CHUNK_COLLECTION", "transcript_chunks_contextual"
        ),
        transcript_summary_collection=os.environ.get(
            "YT_AGENT_TRANSCRIPT_SUMMARY_COLLECTION", "transcript_summaries"
        ),
        embedding_model=os.environ.get(
            "YT_AGENT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        rag_top_k=_int_env("YT_AGENT_RAG_TOP_K", 10),
        transcript_filter_top_k=_int_env("YT_AGENT_TRANSCRIPT_FILTER_TOP_K", 5),
        transcript_filter_min_score=_float_env("YT_AGENT_TRANSCRIPT_FILTER_MIN_SCORE", 0.25),
        rag_recursive_default=_bool_env(
            os.environ.get("YT_AGENT_RAG_RECURSIVE_DEFAULT"), default=False
        ),
        rag_max_depth=_int_env("YT_AGENT_RAG_MAX_DEPTH", 1),
        rag_max_followups=_int_env("YT_AGENT_RAG_MAX_FOLLOWUPS", 3),
        rag_followup_top_k=_optional_int_env("YT_AGENT_RAG_FOLLOWUP_TOP_K"),
        rag_novelty_min_chunks=_int_env("YT_AGENT_RAG_NOVELTY_MIN_CHUNKS", 2),
        rag_max_total_followups=_optional_int_env("YT_AGENT_RAG_MAX_TOTAL_FOLLOWUPS"),
        rag_agent_max_iterations=_int_env("YT_AGENT_RAG_AGENT_MAX_ITERATIONS", 10),
        chunk_target_chars=_int_env("YT_AGENT_CHUNK_TARGET_CHARS", 1200),
        chunk_overlap_chars=_int_env("YT_AGENT_CHUNK_OVERLAP_CHARS", 150),
        retrieval_mode=_retrieval_mode_env("YT_AGENT_RETRIEVAL_MODE", "semantic"),
        retrieval_candidates=_int_env("YT_AGENT_RETRIEVAL_CANDIDATES", 30),
        rerank_enabled=_bool_env(os.environ.get("YT_AGENT_RERANK_ENABLED"), default=True),
        rerank_model=os.environ.get(
            "YT_AGENT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        neighbor_span=_int_env("YT_AGENT_NEIGHBOR_SPAN", 0),
        query_transform=_query_transform_env("YT_AGENT_QUERY_TRANSFORM"),
        multi_query_variants=_int_env("YT_AGENT_MULTI_QUERY_VARIANTS", 3),
        llm_timeout_seconds=_float_env("YT_AGENT_LLM_TIMEOUT_SECONDS", 120.0),
        judge_samples=_int_env("YT_AGENT_JUDGE_SAMPLES", 1),
        supadata_timeout_seconds=_float_env("SUPADATA_TIMEOUT_SECONDS", 120.0),
        supadata_poll_interval_seconds=_float_env("SUPADATA_POLL_INTERVAL_SECONDS", 2.0),
        supadata_max_poll_seconds=_float_env("SUPADATA_MAX_POLL_SECONDS", 600.0),
        discovery_cache_ttl_hours=_float_env("YT_AGENT_DISCOVERY_CACHE_TTL_HOURS", 24.0),
        judge_model=os.environ.get("YT_AGENT_JUDGE_MODEL") or None,
        judge_api_key=os.environ.get("YT_AGENT_JUDGE_API_KEY") or None,
        judge_base_url=os.environ.get("YT_AGENT_JUDGE_BASE_URL") or None,
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "yt-agent-graph"),
        graph_cache_dir=_resolve_project_path(
            os.environ.get("YT_AGENT_GRAPH_CACHE_PATH", ".yt-agent/graph_cache")
        ),
        query_cache_dir=_resolve_project_path(
            os.environ.get("YT_AGENT_QUERY_CACHE_PATH", ".yt-agent/query_cache")
        ),
        context_cache_dir=_resolve_project_path(
            os.environ.get("YT_AGENT_CONTEXT_CACHE_PATH", ".yt-agent/context_cache")
        ),
        theme_path=_resolve_project_path(
            os.environ.get("YT_AGENT_THEME_PATH", ".yt-agent/themes.json")
        ),
        conflict_path=_resolve_project_path(
            os.environ.get("YT_AGENT_CONFLICT_PATH", ".yt-agent/conflicts.json")
        ),
    )
