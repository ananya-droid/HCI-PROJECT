"""communication_test.py — unit tests for CommunicationEngine."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication_engine import CommunicationEngine


def test_add_and_clear():
    engine = CommunicationEngine()
    assert engine.is_empty()

    engine.add_token("HELLO")
    engine.add_token("I")
    engine.add_token("NEED")
    engine.add_token("WATER")

    assert engine.get_tokens() == ["HELLO", "I", "NEED", "WATER"]
    assert not engine.is_empty()

    engine.clear()
    assert engine.is_empty()
    assert engine.get_tokens() == []

    print("✅ communication_test passed.")


if __name__ == "__main__":
    test_add_and_clear()
