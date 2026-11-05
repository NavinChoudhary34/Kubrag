import sys
from unittest.mock import MagicMock

# responder imports the Portkey gateway; replace it with a fake
fake_gateway = MagicMock()
sys.modules["app.gateway"] = fake_gateway

from app.agents.nodes import responder, retriever  # noqa: E402


def _state(query, documents=None):
    return {
        "messages": [{"role": "user", "content": "how do pods scale?"}],
        "current_query": query, "documents": documents or [],
        "plan": ["Intent: Technical"], "status": "",
    }


def test_no_documents_skips_llm_and_says_so():
    fake_gateway.portkey_client.reset_mock()
    out = responder.generate_node(_state("pod scaling", documents=[]))
    assert out["final_answer"] == responder.NO_CONTEXT_ANSWER
    fake_gateway.portkey_client.chat.completions.create.assert_not_called()


def test_with_documents_calls_llm(monkeypatch):
    resp = MagicMock()
    resp.choices[0].message.content = "Use the HPA."
    fake_gateway.portkey_client.chat.completions.create.return_value = resp
    fake_gateway.extract_cache_status.return_value = "MISS"

    out = responder.generate_node(_state("pod scaling", ["SOURCE: a\nCONTENT: HPA docs"]))
    assert out["final_answer"] == "Use the HPA."


def test_retriever_reports_empty_results_explicitly(monkeypatch):
    monkeypatch.setattr(retriever, "search_enterprise_knowledge", lambda q, limit: [])
    out = retriever.retrieve_node(_state("pod scaling"))
    assert out["documents"] == [] and out["citations"] == []
    assert "none" in out["plan"][-1]


def test_retriever_formats_documents_with_source(monkeypatch):
    raw = [{"content": "HPA text", "source": "pods.html", "score": 0.8}]
    monkeypatch.setattr(retriever, "search_enterprise_knowledge", lambda q, limit: raw)
    monkeypatch.setattr(retriever, "rerank_documents", lambda q, d, top_n: d)
    out = retriever.retrieve_node(_state("pod scaling"))
    assert out["documents"] == ["SOURCE: pods.html\nCONTENT: HPA text"]
    assert out["citations"] == [{"source": "pods.html", "score": 0.8}]
    assert out["plan"][-1] == "Context Retrieved"  # string the eval pipeline looks for
