# sentence_generator.py

class SentenceGenerator:

    def __init__(self):
        pass

    def generate(self, tokens, context=None):

        if not tokens:
            return ""

        # Convert tokens to strings
        tokens = [str(token).upper() for token in tokens]

        # Temporary natural-language fallback.
        # This is NOT the final AI system.
        text = " ".join(tokens)

        # Basic formatting only
        sentence = text.capitalize()

        if not sentence.endswith((".", "!", "?")):
            sentence += "."

        return sentence