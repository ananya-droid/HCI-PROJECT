# conversation_context.py


class ConversationContext:

    def __init__(self):

        self.mode = "GENERAL"

        self.history = []


    # ==========================================
    # SET CONVERSATION MODE
    # ==========================================

    def set_mode(self, mode):

        self.mode = mode.upper()


    # ==========================================
    # GET CURRENT MODE
    # ==========================================

    def get_mode(self):

        return self.mode


    # ==========================================
    # ADD MESSAGE
    # ==========================================

    def add_message(self, speaker, text):

        self.history.append({
            "speaker": speaker,
            "text": text
        })


    # ==========================================
    # GET CONVERSATION HISTORY
    # ==========================================

    def get_history(self):

        return self.history


    # ==========================================
    # GET RECENT HISTORY
    # ==========================================

    def get_recent_history(self, limit=5):

        return self.history[-limit:]


    # ==========================================
    # CLEAR CONVERSATION
    # ==========================================

    def clear(self):

        self.history = []


    # ==========================================
    # CLEAR EVERYTHING
    # ==========================================

    def reset(self):

        self.mode = "GENERAL"
        self.history = []