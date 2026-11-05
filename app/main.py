# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard
from app.reliability import configuration_presence, liveness_payload, missing_required
from app.services.retrieval.errors import RetrievalError


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup hook (replaces the deprecated @app.on_event('startup'))."""
    initialize_rails()
    yield


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    # Bound user input; out-of-range values get an automatic HTTP 422.
    q: str = Field(min_length=1, max_length=12000)
    # If the client sends no thread_id we create a fresh one. The old shared
    # default ("default_user") put every anonymous caller in ONE conversation,
    # so strangers could see each other's chat memory.
    thread_id: Optional[str] = Field(default=None, min_length=1, max_length=100)


def _error_response(q: str, answer: str, status_code: int, status: str) -> JSONResponse:
    """Same JSON shape as a success response, but with a real HTTP error code."""
    return JSONResponse(
        status_code=status_code,
        content={
            "question": q,
            "answer": answer,
            "thought_process": ["Error encountered during execution."],
            "status": status,
            "sources": [],
            "citations": [],
        },
    )


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/health/live")
def health_live():
    """Process liveness only; no external dependency checks."""
    return liveness_payload()


@app.get("/health/config")
def health_config():
    """Report which env vars are set (never their values). No network calls."""
    missing = missing_required()
    return {
        "status": "ok" if not missing else "degraded",
        "missing_required": missing,
        "environment": configuration_presence(),
        "external_services_checked": False,
    }


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        # 502: the Mermaid rendering service (external) failed
        return JSONResponse(status_code=502, content={"error": f"Could not generate graph image: {e}"})


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id or str(uuid.uuid4())

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "citations": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "citations": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "citations": final_output.get("citations", [])
        }
    except RetrievalError as e:
        # Vector DB / embedding problem: a dependency outage, not a bad request
        logfire.error(f"❌ Retrieval unavailable: {e}")
        return _error_response(
            q, "The knowledge base is temporarily unavailable. Please try again shortly.",
            503, "retrieval_unavailable")
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return _error_response(
            q, "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            500, "error")
