"""intent_test.py — unit tests for the rule-based intent engine."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import intent_engine as intent


def test_intents():
    cases = [
        (["HELLO"], intent.GREETING),
        (["BYE"], intent.FAREWELL),
        (["I", "NEED", "HELP"], intent.HELP_REQUEST),
        (["I", "NEED", "WATER"], intent.WATER_REQUEST),
        (["I", "NEED", "FOOD"], intent.FOOD_REQUEST),
        (["THANK_YOU"], intent.GRATITUDE),
        (["YES"], intent.AFFIRMATION),
        (["NO"], intent.NEGATION),
        ([], intent.UNKNOWN),
    ]

    for tokens, expected in cases:
        result = intent.detect_intent(tokens)
        assert result == expected, f"tokens={tokens} expected={expected} got={result}"

    print("✅ intent_test passed.")


if __name__ == "__main__":
    test_intents()
