import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    RetrievalError (outage / dimension mismatch) is deliberately NOT caught
    here: it propagates so the API can answer with a proper 503.
    """
    query = state["current_query"]

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        # Genuine "nothing found": report it explicitly so the responder
        # can say so instead of letting the LLM improvise without context.
        if not raw_results:
            return {
                "documents": [],
                "citations": [],
                "status": "No relevant documents found.",
                "plan": state["plan"] + ["Context Retrieved: none"],
            }

        with logfire.span("⚖️ Semantic Reranking"):
            reranked = rerank_documents(query, raw_results, top_n=5)
            logfire.info(f"Reranking complete. Kept top {len(reranked)} chunks.")

    # Keep the "CONTENT:" prefix the evals/UI already expect, and add the source
    formatted_docs = [f"SOURCE: {d['source']}\nCONTENT: {d['content']}" for d in reranked]
    citations = [{"source": d["source"], "score": d.get("score")} for d in reranked]

    return {
        "documents": formatted_docs,
        "citations": citations,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
