from app.agents.state import AgentState
from app.agents.history import format_history
from app.gateway import get_langchain_llm
import logfire

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")


def parse_planner_decision(raw: str, fallback_query: str) -> str:
    """
    Turn the raw LLM output into either 'CONVERSATIONAL' or a search query.
    LLMs often add quotes/punctuation ('"CONVERSATIONAL".'), which used to be
    treated as a literal search query. We normalise before comparing, and fall
    back to the user's own message if the model returned nothing usable.
    """
    # Strip quotes, backticks, whitespace AND trailing punctuation in one pass,
    # so '"CONVERSATIONAL".' and "`conversational`" both normalise correctly.
    cleaned = (raw or "").strip().strip("`'\" \n.!")
    if cleaned.upper() == "CONVERSATIONAL":
        return "CONVERSATIONAL"
    return cleaned or fallback_query


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    history = format_history(state["messages"])
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.

    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("🧠 Planner Decision"):
        try:
            raw = llm.invoke(prompt).content
        except Exception as e:
            # If the planner LLM is down, searching with the raw question is a
            # safer default than failing the whole request.
            logfire.warning(f"Planner LLM failed ({e}); falling back to raw query search.")
            raw = ""
        decision = parse_planner_decision(raw, fallback_query=user_message)
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
