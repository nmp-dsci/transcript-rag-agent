"""Draft candidate golden questions from the indexed corpus with RAGAS.

The golden set (``src/evals/golden_dataset.json``) is deliberately small and
hand-curated: every entry carries a reference answer and the ``expected_chunk_ids``
a good retriever must surface, and both need a human who has watched the video.
Growing it to 40+ is therefore a *curation* task, not an automation one.

This script does the part a machine can do well — it uses RAGAS'
``TestsetGenerator`` to draft candidate questions and reference answers grounded in
the corpus, so curation starts from a list instead of a blank page. What it emits
are **candidates, not golden entries**: they have no ``expected_chunk_ids``, no
domain, and are unverified. Promoting one into the golden set is the manual step
documented in ``docs/golden-set-curation.md``.

It calls an LLM, so it needs an API key and is never run in CI.

    uv run python scripts/generate_golden_candidates.py --size 20 --out candidates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def corpus_documents(chunk_records: list[dict[str, Any]]) -> list[Any]:
    """One LangChain ``Document`` per video: its chunks joined in index order.

    RAGAS builds its question knowledge-graph from documents, so grouping chunks
    back into per-video transcripts gives it coherent source material rather than
    hundreds of disconnected fragments. Videos with no text are skipped.
    """
    from langchain_core.documents import Document

    by_video: dict[str, dict[str, Any]] = {}
    for record in chunk_records:
        video_id = str(record.get("video_id") or "")
        if not video_id:
            continue
        bucket = by_video.setdefault(
            video_id,
            {"title": record.get("title"), "chunks": []},
        )
        bucket["chunks"].append(
            (int(record.get("chunk_index", 0) or 0), str(record.get("text") or ""))
        )

    documents: list[Any] = []
    for video_id, bucket in by_video.items():
        ordered = [text for _index, text in sorted(bucket["chunks"]) if text.strip()]
        if not ordered:
            continue
        documents.append(
            Document(
                page_content="\n".join(ordered),
                metadata={"video_id": video_id, "title": bucket["title"]},
            )
        )
    return documents


def candidate_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape RAGAS testset rows into review-ready candidate records.

    Intentionally *not* the golden schema: a candidate still needs a human to add
    ``expected_chunk_ids`` (which video chunks actually answer it), a ``domain``,
    and a fact-check of the reference. ``notes`` says exactly that.
    """
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question = str(row.get("user_input") or row.get("question") or "").strip()
        if not question:
            continue
        candidates.append(
            {
                "id": f"candidate-{index:03d}",
                "question": question,
                "reference_answer": str(row.get("reference") or "").strip(),
                "domain": "",
                "expected_video_ids": [],
                "expected_chunk_ids": [],
                "notes": "UNVERIFIED candidate — add expected_chunk_ids + domain and "
                "fact-check the reference before promoting (see docs/golden-set-curation.md)",
            }
        )
    return candidates


def _load_chunk_records(chroma_path: Path, collection_name: str) -> list[dict[str, Any]]:
    import chromadb
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(collection_name)
    except NotFoundError:
        return []
    result = collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    records: list[dict[str, Any]] = []
    for index, meta in enumerate(metadatas):
        meta = meta or {}
        records.append(
            {
                "video_id": str(meta.get("video_id", "")),
                "chunk_index": int(meta.get("chunk_index", index) or 0),
                "title": meta.get("title") or None,
                "text": documents[index] if index < len(documents) else "",
            }
        )
    return records


def _build_generator(settings: Any) -> Any:
    """A RAGAS ``TestsetGenerator`` on the project's judge LLM + embedding model."""
    from src.evals import _ragas_compat

    _ragas_compat.install()

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator

    model = settings.judge_model or settings.deepseek_model
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            api_key=settings.judge_api_key or settings.deepseek_api_key,
            base_url=settings.judge_base_url or settings.deepseek_base_url,
            temperature=0.0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )
    return TestsetGenerator(llm=llm, embedding_model=embeddings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20, help="Number of candidates to draft")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("golden_candidates.json"),
        help="Where to write the candidate records for review",
    )
    args = parser.parse_args(argv)

    from src.config import load_settings

    settings = load_settings(require_keys=True)
    records = _load_chunk_records(settings.chroma_path, settings.chunk_collection)
    documents = corpus_documents(records)
    if not documents:
        print("No indexed chunks found — index a corpus before generating candidates.")
        return 1

    print(f"Drafting {args.size} candidates from {len(documents)} videos…")
    generator = _build_generator(settings)
    testset = generator.generate_with_langchain_docs(documents, testset_size=args.size)
    candidates = candidate_records(testset.to_list())

    args.out.write_text(json.dumps({"entries": candidates}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(candidates)} UNVERIFIED candidates to {args.out}")
    print("Next: curate each into the golden set per docs/golden-set-curation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
