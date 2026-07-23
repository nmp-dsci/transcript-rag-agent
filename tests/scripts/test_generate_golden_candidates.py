from __future__ import annotations

from scripts.generate_golden_candidates import candidate_records, corpus_documents


class TestCorpusDocuments:
    def test_one_document_per_video_with_chunks_in_index_order(self) -> None:
        records = [
            {"video_id": "v1", "chunk_index": 1, "text": "second", "title": "V1"},
            {"video_id": "v1", "chunk_index": 0, "text": "first", "title": "V1"},
            {"video_id": "v2", "chunk_index": 0, "text": "only", "title": "V2"},
        ]
        docs = corpus_documents(records)

        by_video = {doc.metadata["video_id"]: doc for doc in docs}
        assert set(by_video) == {"v1", "v2"}
        # Chunks are stitched back in index order, not retrieval order.
        assert by_video["v1"].page_content == "first\nsecond"
        assert by_video["v1"].metadata["title"] == "V1"

    def test_videos_without_text_are_skipped(self) -> None:
        records = [
            {"video_id": "v1", "chunk_index": 0, "text": "   "},
            {"video_id": "", "chunk_index": 0, "text": "orphan"},
        ]
        assert corpus_documents(records) == []


class TestCandidateRecords:
    def test_maps_ragas_rows_to_unverified_candidates(self) -> None:
        rows = [
            {"user_input": "What changed for CGT?", "reference": "The rate rose."},
            {"user_input": "  ", "reference": "ignored — no question"},
        ]
        candidates = candidate_records(rows)

        assert len(candidates) == 1
        entry = candidates[0]
        assert entry["id"] == "candidate-001"
        assert entry["question"] == "What changed for CGT?"
        assert entry["reference_answer"] == "The rate rose."
        # Candidates are explicitly not golden entries yet.
        assert entry["expected_chunk_ids"] == []
        assert entry["domain"] == ""
        assert "UNVERIFIED" in entry["notes"]
