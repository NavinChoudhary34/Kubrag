# Improvements (drop these files over the repo, same paths)

| File | Change |
|---|---|
| app/ingestion/chunking/splitter.py | Oversized paragraphs are split (HTML/PDF docs were stored as ONE chunk, up to 62k chars) + overlap |
| app/ingestion/processor.py | Deterministic point ids (re-ingest no longer duplicates) + chunk_index payload |
| app/services/retrieval/qdrant_service.py | Raises RetrievalError on outage / embedding-dim mismatch instead of silently returning [] |
| app/services/retrieval/errors.py | New RetrievalError |
| app/services/retrieval/ranking_service.py | Reranker keeps source + scores (was returning bare strings) |
| app/agents/nodes/retriever.py | Passes sources through, reports "no results" explicitly, adds `citations` |
| app/agents/nodes/responder.py | No context -> honest answer, no LLM call; context-only prompt; history capped |
| app/agents/nodes/planner.py | Robust CONVERSATIONAL parsing; falls back to raw query if planner LLM fails |
| app/agents/history.py, state.py | Shared history formatter (last 10 msgs); `citations` state field |
| app/main.py | Real 422/503/500 codes, input limits, unique thread_id by default (was shared "default_user"), lifespan, /health/live, /health/config |
| app/reliability.py | Health/config helper (supersedes the patch pack) |
| ui/app.py | Answer reveal no longer sleeps 5 ms per character |
| requirements-prod.txt | Adds langchain-google-genai (embedding.py imports it; missing from prod image) |
| tests/ | 39 offline unit tests (`python -m pytest tests`) |

After deploying: re-run ingestion with `--wipe` so HTML/PDF docs are re-chunked.
