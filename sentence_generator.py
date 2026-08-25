"""
sentence_generator.py
---------------------

Converts recognized sign tokens into natural English.

Recognition and language generation are deliberately separate:

    SIGN TOKENS
        ↓
    SENTENCE GENERATOR
        ↓
    NATURAL ENGLISH
"""


# ---------------------------------------------------------
# OFFLINE SENTENCE GENERATION
# ---------------------------------------------------------

def generate_offline_sentence(tokens, context=None):
    """
    Convert sign tokens into a simple natural English sentence.

    This is deterministic and works completely offline.
    """

    if not tokens:
        return ""

    # Normalize tokens
    tokens = [str(token).upper() for token in tokens]

    token_set = set(tokens)

    # -----------------------------------------------------
    # Specific combinations
    # -----------------------------------------------------

    # HELLO + HELP
    if "HELLO" in token_set and "HELP" in token_set:
        return "Hello, I need some help."

    # I + HELP
    if "I" in token_set and "HELP" in token_set:
        return "I need some help."

    # HELLO alone
    if tokens == ["HELLO"]:
        return "Hello."

    # HELP alone
    if tokens == ["HELP"]:
        return "I need some help."

    # THANK YOU
    if "THANK_YOU" in token_set:
        if "HELLO" in token_set:
            return "Hello, thank you."
        return "Thank you."

    # YES
    if tokens == ["YES"]:
        return "Yes."

    # NO
    if tokens == ["NO"]:
        return "No."

    # PLEASE
    if tokens == ["PLEASE"]:
        return "Please."

    # I alone
    if tokens == ["I"]:
        return "I."

    # -----------------------------------------------------
    # Basic combinations
    # -----------------------------------------------------

    if "HELLO" in token_set:
        remaining = [
            t for t in tokens
            if t != "HELLO"
        ]

        if remaining:
            return "Hello, " + _basic_phrase(remaining)

        return "Hello."

    return _basic_phrase(tokens)


# ---------------------------------------------------------
# BASIC FALLBACK
# ---------------------------------------------------------

def _basic_phrase(tokens):
    """
    Simple deterministic fallback for combinations
    that do not yet have a dedicated rule.
    """

    phrases = []

    for token in tokens:

        if token == "YES":
            phrases.append("yes")

        elif token == "NO":
            phrases.append("no")

        elif token == "PLEASE":
            phrases.append("please")

        elif token == "THANK_YOU":
            phrases.append("thank you")

        elif token == "HELP":
            phrases.append("help")

        elif token == "I":
            phrases.append("I")

        else:
            phrases.append(token.lower())

    if not phrases:
        return ""

    # Special case:
    # I + HELP
    if "I" in tokens and "HELP" in tokens:
        return "I need some help."

    return " ".join(phrases).capitalize() + "."


# ---------------------------------------------------------
# PUBLIC FUNCTION
# ---------------------------------------------------------

def generate_sentence(tokens, context=None, use_ai=True):
    """
    Main interface used by live_sign_test.py.

    Returns:

        sentence, used_ai

    Example:

        generate_sentence(
            ["HELLO", "HELP"],
            context=context,
            use_ai=False
        )

        -> ("Hello, I need some help.", False)
    """

    if not tokens:
        return "", False

    # -----------------------------------------------------
    # AI generation
    # -----------------------------------------------------
    #
    # We intentionally keep the first version offline.
    # Once the deterministic pipeline is working perfectly,
    # AI generation can be added here.
    #
    # This prevents an API failure from breaking the demo.
    # -----------------------------------------------------

    sentence = generate_offline_sentence(tokens, context)

    return sentence, False