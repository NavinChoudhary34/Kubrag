import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.errors import RetrievalError


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

# Cached vector size of the collection (looked up once, on first search)
_collection_dim: int | None = None


def _get_collection_dim() -> int | None:
    """Return the vector size the Qdrant collection was created with."""
    global _collection_dim
    if _collection_dim is None:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        # Single unnamed vector config exposes .size; named configs are dicts
        _collection_dim = getattr(vectors, "size", None)
    return _collection_dim


def search_enterprise_knowledge(query: str, limit: int = 8) -> list[dict]:
    """
    Search the enterprise knowledge base.

    Returns a list of {"content", "source", "score"} dicts.
    An empty list means "searched fine, nothing found".
    Raises RetrievalError if the search itself failed, so callers (and the
    API) can tell an outage apart from a genuine no-results case.
    """
    try:
        query_vector = embed_query(query)

        # Guard: if Gemini was unreachable and the 768-dim fallback model was
        # used, the vector no longer matches a 3072-dim collection. Fail loudly
        # with a clear message instead of returning silent empty results.
        expected_dim = _get_collection_dim()
        if expected_dim is not None and len(query_vector) != expected_dim:
            raise RetrievalError(
                f"Embedding dimension mismatch: query vector is {len(query_vector)}-dim "
                f"but collection '{settings.QDRANT_COLLECTION}' expects {expected_dim}-dim. "
                "The embedding model likely fell back to a different model."
            )

        # query_points is the modern Qdrant search interface
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        results = []
        for res in response.points:
            payload = res.payload or {}  # tolerate points with no payload
            text = payload.get("text", "")
            if not text:
                continue  # nothing usable to show the LLM
            results.append({
                "content": text,
                "source": payload.get("source", "Unknown"),
                "score": res.score
            })
        return results

    except RetrievalError:
        logfire.error("❌ Retrieval error (see message above)")
        raise
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        raise RetrievalError(f"Vector search failed: {e}") from e
