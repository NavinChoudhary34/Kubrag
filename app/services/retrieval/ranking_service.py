import time
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-macro-MiniLM-L-6-v2) for fast cross-encoder reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # Specific cache directory avoids permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()
    return _ranker


def rerank_documents(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-score candidate chunks against the query with a cross-encoder.

    CHANGED: takes and returns dicts ({"content", "source", "score", ...})
    instead of bare strings, so source metadata survives reranking.
    The original vector score is kept as "vector_score"; "score" becomes the
    reranker score. On any reranker failure we fall back to the Qdrant order.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()

        # FlashRank wants 'id' + 'text'; the id lets us map results back to metadata
        passages = [{"id": i, "text": doc["content"]} for i, doc in enumerate(documents)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))

        reranked = []
        for res in results[:top_n]:
            original = documents[int(res["id"])]
            reranked.append({
                **original,
                "vector_score": original.get("score"),
                "score": float(res["score"]),
            })

        top_score = results[0]["score"] if results else "N/A"
        logfire.info(f"✅ [Reranker] Done in {time.time() - start_time:.2f}s. Top score: {top_score}")
        return reranked

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fall back to the original Qdrant order so the user still gets an answer
        return documents[:top_n]
