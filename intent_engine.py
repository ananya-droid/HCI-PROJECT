"""
intent_engine.py
------------------
Classifies the high-level INTENT of a recognized token sequence.
Intent is metadata alongside the generated sentence — it never
replaces or overrides the actual message (see spec section 25-26).
"""

GREETING = "GREETING"
FAREWELL = "FAREWELL"
HELP_REQUEST = "HELP_REQUEST"
WATER_REQUEST = "WATER_REQUEST"
FOOD_REQUEST = "FOOD_REQUEST"
GRATITUDE = "GRATITUDE"
AFFIRMATION = "AFFIRMATION"
NEGATION = "NEGATION"
EMPLOYMENT_INTENT = "EMPLOYMENT_INTENT"
STATEMENT = "STATEMENT"
UNKNOWN = "UNKNOWN"


def detect_intent(tokens):
    """
    tokens: list of recognized sign tokens, e.g. ["I", "NEED", "WATER"]

    Simple, explainable rule-based classifier. Order matters — more
    specific rules are checked before generic fallbacks.
    """
    if not tokens:
        return UNKNOWN

    token_set = set(t.upper() for t in tokens)

    if token_set == {"HELLO"} or tokens[:1] == ["HELLO"]:
        if len(tokens) == 1:
            return GREETING

    if "BYE" in token_set:
        return FAREWELL

    if "HELP" in token_set:
        return HELP_REQUEST

    if "NEED" in token_set and "WATER" in token_set:
        return WATER_REQUEST

    if "NEED" in token_set and "FOOD" in token_set:
        return FOOD_REQUEST

    if "THANK_YOU" in token_set:
        return GRATITUDE

    if "JOB" in token_set or "WORK" in token_set:
        return EMPLOYMENT_INTENT

    if token_set == {"YES"}:
        return AFFIRMATION

    if token_set == {"NO"}:
        return NEGATION

    if len(tokens) >= 2:
        return STATEMENT

    return UNKNOWN
