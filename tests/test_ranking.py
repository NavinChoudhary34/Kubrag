from unittest.mock import MagicMock

from app.services.retrieval import ranking_service

DOCS = [
    {"content": "alpha", "source": "a.txt", "score": 0.5},
    {"content": "beta", "source": "b.txt", "score": 0.4},
    {"content": "gamma", "source": "c.txt", "score": 0.3},
]


def test_empty_input_returns_empty():
    assert ranking_service.rerank_documents("q", []) == []


def test_source_metadata_survives_reranking(monkeypatch):
    fake = MagicMock()
    # reranker prefers doc id 2, then 0
    fake.rerank.return_value = [{"id": 2, "text": "gamma", "score": 0.99},
                                {"id": 0, "text": "alpha", "score": 0.50}]
    monkeypatch.setattr(ranking_service, "_get_ranker", lambda: fake)

    out = ranking_service.rerank_documents("q", DOCS, top_n=2)

    assert [d["source"] for d in out] == ["c.txt", "a.txt"]
    assert out[0]["score"] == 0.99          # reranker score replaces score
    assert out[0]["vector_score"] == 0.3    # original vector score is kept


def test_reranker_failure_falls_back_to_original_order(monkeypatch):
    def boom():
        raise RuntimeError("onnx missing")
    monkeypatch.setattr(ranking_service, "_get_ranker", boom)
    assert ranking_service.rerank_documents("q", DOCS, top_n=2) == DOCS[:2]
