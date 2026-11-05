"""
Test setup: replace heavy third-party packages (Qdrant, FlashRank, Portkey,
NeMo, LangGraph, logfire...) with light stand-ins so the unit tests run in
seconds, offline, with no API keys. Only OUR logic is being tested.
"""
import contextlib
import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_stubs():
    # logfire: every call is a no-op; span() works as a context manager
    lf = types.ModuleType("logfire")
    for name in ("info", "warning", "warn", "error", "debug", "configure"):
        setattr(lf, name, lambda *a, **k: None)
    lf.span = lambda *a, **k: contextlib.nullcontext()
    sys.modules["logfire"] = lf

    # python-dotenv: don't read any real .env during tests
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv

    # Everything else can be a MagicMock module
    for name in (
        "qdrant_client", "qdrant_client.http", "qdrant_client.http.models",
        "flashrank", "portkey_ai", "langchain_openai", "langchain_groq",
        "langchain_google_genai", "nemoguardrails",
    ):
        sys.modules[name] = MagicMock()


_install_stubs()


@pytest.fixture
def fake_ranker(monkeypatch):
    """Give ranking_service a controllable Ranker."""
    from app.services.retrieval import ranking_service
    monkeypatch.setattr(ranking_service, "_ranker", None)
    return ranking_service
