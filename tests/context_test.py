"""context_test.py — unit tests for ConversationContext."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from conversation_context import ConversationContext


def test_history_and_mode():
    ctx = ConversationContext()
    assert ctx.mode == config.MODE_GENERAL

    ctx.set_mode(config.MODE_INTERVIEW)
    assert ctx.mode == config.MODE_INTERVIEW

    ctx.add_message("Interviewer", "Tell me about yourself.")
    ctx.add_message("You", "I'm a computer science student.")

    history = ctx.get_history()
    assert len(history) == 2
    assert history[0]["speaker"] == "Interviewer"

    formatted = ctx.format_history_for_prompt()
    assert "Interviewer" in formatted and "computer science" in formatted

    ctx.clear()
    assert ctx.get_history() == []

    print("✅ context_test passed.")


if __name__ == "__main__":
    test_history_and_mode()
