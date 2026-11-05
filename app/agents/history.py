# Shared helper for turning stored chat messages into prompt text.

# Only the most recent messages go into a prompt, so long conversations
# can't grow without limit and blow past the model's token limits.
MAX_HISTORY_MESSAGES = 10


def format_history(messages: list[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    """
    Format every message EXCEPT the latest one (the current user question)
    as 'User: ...' / 'Assistant: ...' lines, keeping only the last N.
    """
    previous = messages[:-1][-max_messages:]
    lines = []
    for msg in previous:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + ("\n" if lines else "")
