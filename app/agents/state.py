from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict, total=False):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]      # formatted context strings (shown in the UI)
    citations: List[dict]     # NEW: [{"source", "score"}] for each kept chunk
    plan: List[str]
    status: str
    final_answer: str
