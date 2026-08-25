# communication_engine.py


class CommunicationEngine:

    def __init__(self):
        self.tokens = []


    # ==========================================
    # ADD SIGN
    # ==========================================

    def add_token(self, token):

        if token not in ["UNKNOWN", "NO HAND"]:
            self.tokens.append(token)


    # ==========================================
    # GET RAW TEXT
    # ==========================================

    def get_text(self):

        return " ".join(self.tokens)


    # ==========================================
    # GENERATE NATURAL SENTENCE
    # ==========================================

    def finish_sentence(self):

        if not self.tokens:
            return ""

        tokens = [
            token.upper()
            for token in self.tokens
        ]


        # ======================================
        # COMMON PHRASES
        # ======================================

        phrases = {

            ("HELLO",):
                "Hello.",

            ("YES",):
                "Yes.",

            ("NO",):
                "No.",

            ("THANK_YOU",):
                "Thank you.",

            ("PLEASE",):
                "Please.",

            ("HELP",):
                "I need help.",

            ("I", "NEED", "HELP"):
                "I need help.",

            ("I", "NEED", "WATER"):
                "I need water.",

            ("I", "NEED", "FOOD"):
                "I need food.",

            ("I", "NEED"):
                "I need something.",

            ("I", "WANT", "WATER"):
                "I want water.",

            ("I", "WANT", "FOOD"):
                "I want food.",

            ("HELLO", "I", "NEED", "HELP"):
                "Hello, I need help.",

            ("HELLO", "I", "NEED", "WATER"):
                "Hello, I need water.",

            ("HELLO", "I", "NEED", "FOOD"):
                "Hello, I need food."
        }


        token_tuple = tuple(tokens)


        # ======================================
        # EXACT PHRASE MATCH
        # ======================================

        if token_tuple in phrases:

            return phrases[token_tuple]


        # ======================================
        # SIMPLE FALLBACK RULES
        # ======================================

        if (
            "I" in tokens
            and "NEED" in tokens
            and "HELP" in tokens
        ):

            return "I need help."


        if (
            "I" in tokens
            and "NEED" in tokens
            and "WATER" in tokens
        ):

            return "I need water."


        if (
            "I" in tokens
            and "NEED" in tokens
            and "FOOD" in tokens
        ):

            return "I need food."


        # ======================================
        # GENERAL FALLBACK
        # ======================================

        words = []

        for token in tokens:

            word = token.replace(
                "_",
                " "
            ).lower()

            words.append(word)


        sentence = " ".join(words)

        sentence = sentence.capitalize()

        if not sentence.endswith(
            (".", "!", "?")
        ):

            sentence += "."


        return sentence


    # ==========================================
    # GET CURRENT TOKENS
    # ==========================================

    def get_tokens(self):

        return self.tokens.copy()


    # ==========================================
    # CLEAR
    # ==========================================

    def clear(self):

        self.tokens = []