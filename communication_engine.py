"""
communication_engine.py
-------------------------
Deliberately dumb by design (see spec section 17): it only holds the
sequence of recognized sign tokens for the CURRENT message. It does
NOT contain hardcoded phrase mappings — all language generation
happens downstream in sentence_generator.py.
"""


class CommunicationEngine:
    def __init__(self):
        self.tokens = []

    def add_token(self, token: str):
        if token:
            self.tokens.append(token)

    def get_tokens(self):
        return list(self.tokens)

    def clear(self):
        self.tokens = []

    def is_empty(self):
        return len(self.tokens) == 0

    def __repr__(self):
        return f"CommunicationEngine(tokens={self.tokens})"
