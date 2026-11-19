class RetrievalError(RuntimeError):
    """
    Raised when the vector store or embedding step is broken (outage, bad
    credentials, embedding-dimension mismatch).
    This is different from a genuine "no results" answer, which is just [].
    """
