import sys
from unittest.mock import MagicMock

# planner.py builds an LLM at import time; give it a fake gateway first
sys.modules["app.gateway"] = MagicMock()

from app.agents.nodes.planner import parse_planner_decision  # noqa: E402


def test_plain_conversational():
    assert parse_planner_decision("CONVERSATIONAL", "hi") == "CONVERSATIONAL"


def test_quoted_and_punctuated_conversational_is_still_recognised():
    # LLMs often answer '"CONVERSATIONAL".' — this used to be searched literally
    assert parse_planner_decision('"CONVERSATIONAL".', "hi") == "CONVERSATIONAL"
    assert parse_planner_decision("`conversational`", "hi") == "CONVERSATIONAL"


def test_search_query_is_returned_cleaned():
    assert parse_planner_decision('  "k8s HPA scaling"  ', "x") == "k8s HPA scaling"


def test_empty_output_falls_back_to_user_message():
    assert parse_planner_decision("", "how do pods scale?") == "how do pods scale?"
    assert parse_planner_decision(None, "how do pods scale?") == "how do pods scale?"
