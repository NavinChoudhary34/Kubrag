from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.retrieval import qdrant_service
from app.services.retrieval.errors import RetrievalError


def _point(text, source="a.txt", score=0.9, payload_none=False):
    payload = None if payload_none else {"text": text, "source": source}
    return SimpleNamespace(payload=payload, score=score)


@pytest.fixture
def qdrant(monkeypatch):
    """Fake Qdrant client + embedding so no network is needed."""
    fake = MagicMock()
    monkeypatch.setattr(qdrant_service, "client", fake)
    monkeypatch.setattr(qdrant_service, "embed_query", lambda q: [0.1] * 4)
    monkeypatch.setattr(qdrant_service, "_collection_dim", 4)  # matches the vector above
    return fake


def test_returns_content_source_and_score(qdrant):
    qdrant.query_points.return_value = SimpleNamespace(points=[_point("hello", "k8s.html", 0.7)])
    assert qdrant_service.search_enterprise_knowledge("q") == [
        {"content": "hello", "source": "k8s.html", "score": 0.7}
    ]


def test_genuine_no_results_is_an_empty_list_not_an_error(qdrant):
    qdrant.query_points.return_value = SimpleNamespace(points=[])
    assert qdrant_service.search_enterprise_knowledge("q") == []


def test_points_with_missing_payload_or_text_are_skipped(qdrant):
    qdrant.query_points.return_value = SimpleNamespace(
        points=[_point("", payload_none=True), _point(""), _point("good")])
    results = qdrant_service.search_enterprise_knowledge("q")
    assert [r["content"] for r in results] == ["good"]


def test_qdrant_outage_raises_instead_of_returning_empty(qdrant):
    qdrant.query_points.side_effect = ConnectionError("qdrant down")
    with pytest.raises(RetrievalError):
        qdrant_service.search_enterprise_knowledge("q")


def test_embedding_dimension_mismatch_is_reported(qdrant, monkeypatch):
    monkeypatch.setattr(qdrant_service, "_collection_dim", 3072)  # collection is 3072-dim
    with pytest.raises(RetrievalError, match="dimension mismatch"):
        qdrant_service.search_enterprise_knowledge("q")
    qdrant.query_points.assert_not_called()
