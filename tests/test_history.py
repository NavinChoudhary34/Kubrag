from app.agents.history import format_history


def test_excludes_latest_message():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "latest"}]
    out = format_history(msgs)
    assert "latest" not in out
    assert "User: hi" in out and "Assistant: hello" in out


def test_only_most_recent_messages_are_kept():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(50)]
    out = format_history(msgs, max_messages=3)
    assert out.count("User:") == 3
    assert "m48" in out and "m0" not in out


def test_empty_history_is_empty_string():
    assert format_history([{"role": "user", "content": "only"}]) == ""
