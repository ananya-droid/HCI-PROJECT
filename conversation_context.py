"""
conversation_context.py
-------------------------
Tracks conversation history and current mode (GENERAL, INTERVIEW, ...),
so the AI/NLP layer can phrase things appropriately.
"""

import config


class ConversationContext:
    def __init__(self, mode: str = config.MODE_GENERAL):
        self.history = []
        self.mode = mode

    def add_message(self, speaker: str, text: str):
        self.history.append({"speaker": speaker, "text": text})

    def get_history(self):
        return list(self.history)

    def get_recent_history(self, n: int = 6):
        return self.history[-n:]

    def set_mode(self, mode: str):
        self.mode = mode

    def clear(self):
        self.history = []

    def format_history_for_prompt(self, n: int = 6) -> str:
        lines = []
        for entry in self.get_recent_history(n):
            lines.append(f"{entry['speaker']}: {entry['text']}")
        return "\n".join(lines) if lines else "(no prior conversation)"
