"""Dependency-free helpers for API health and configuration diagnostics."""
from __future__ import annotations
import os
from typing import Iterable

# Variable NAMES only — values are never returned, so secrets can't leak.
# Required = the API cannot answer queries without these.
REQUIRED_KEYS = (
    "PORTKEY_API_KEY", "QDRANT_CLUSTER_ENDPOINT", "QDRANT_API_KEY",
    "GROQ_API_KEY", "GEMINI_API_KEY",
)
# Optional = features degrade gracefully without these.
OPTIONAL_KEYS = ("GROQ_FALLBACK_API_KEY", "LOGFIRE_TOKEN", "LANGSMITH_API_KEY")


def configuration_presence(keys: Iterable[str] = REQUIRED_KEYS + OPTIONAL_KEYS) -> dict[str, bool]:
    """Report whether each environment variable has a non-blank value."""
    return {key: bool(os.getenv(key, "").strip()) for key in keys}


def missing_required(keys: Iterable[str] = REQUIRED_KEYS) -> list[str]:
    """Names of required variables that are unset or blank."""
    return [k for k, present in configuration_presence(keys).items() if not present]


def liveness_payload() -> dict[str, str]:
    """Process liveness only; intentionally does not probe external services."""
    return {"status": "ok"}
