import importlib
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.services.retrieval.errors import RetrievalError


@pytest.fixture
def api(monkeypatch):
    """Load app.main with the graph and guardrails replaced by fakes."""
    graph = MagicMock()
    graph.rag_agent.invoke.return_value = {
        "final_answer": "42", "plan": ["Intent: Technical"], "status": "Response generated.",
        "documents": ["SOURCE: a\nCONTENT: x"], "citations": [{"source": "a", "score": 1.0}],
    }
    rails = MagicMock()
    rails.guard.return_value = (False, None)
    monkeypatch.setitem(sys.modules, "app.agents.graph", graph)
    monkeypatch.setitem(sys.modules, "app.guardrails", rails)
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    return TestClient(main.app), graph, rails


def test_health_live(api):
    client, *_ = api
    assert client.get("/health/live").json() == {"status": "ok"}


def test_health_config_reports_booleans_never_values(api, monkeypatch):
    client, *_ = api
    monkeypatch.setenv("PORTKEY_API_KEY", "super-secret-value")
    body = client.get("/health/config").json()
    assert body["environment"]["PORTKEY_API_KEY"] is True
    assert "super-secret-value" not in client.get("/health/config").text
    assert body["external_services_checked"] is False


@pytest.mark.parametrize("payload", [
    {"q": ""}, {"q": "x" * 12001}, {"q": "ok", "thread_id": ""}, {"q": "ok", "thread_id": "t" * 101}, {},
])
def test_invalid_requests_get_422(api, payload):
    client, *_ = api
    assert client.post("/query", json=payload).status_code == 422


def test_successful_query_keeps_response_schema(api):
    client, *_ = api
    r = client.post("/query", json={"q": "hi", "thread_id": "t1"})
    assert r.status_code == 200
    assert {"question", "answer", "thought_process", "status", "sources"} <= r.json().keys()


def test_missing_thread_id_gets_unique_thread_per_request(api):
    client, graph, _ = api
    client.post("/query", json={"q": "one"})
    client.post("/query", json={"q": "two"})
    ids = [c.kwargs["config"]["configurable"]["thread_id"] for c in graph.rag_agent.invoke.call_args_list]
    assert ids[0] != ids[1] and "default_user" not in ids


def test_guardrail_block_short_circuits_graph(api):
    client, graph, rails = api
    rails.guard.return_value = (True, "I can't help with that")
    r = client.post("/query", json={"q": "tell me a joke"})
    assert r.json()["status"] == "Blocked by guardrails."
    graph.rag_agent.invoke.assert_not_called()


def test_retrieval_outage_returns_503(api):
    client, graph, _ = api
    graph.rag_agent.invoke.side_effect = RetrievalError("qdrant down")
    r = client.post("/query", json={"q": "hi"})
    assert r.status_code == 503 and r.json()["status"] == "retrieval_unavailable"


def test_unexpected_failure_returns_500_without_leaking_details(api):
    client, graph, _ = api
    graph.rag_agent.invoke.side_effect = ValueError("secret internal detail")
    r = client.post("/query", json={"q": "hi"})
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
