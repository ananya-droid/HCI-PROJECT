# communication_pipeline.py

from communication_engine import CommunicationEngine
from ai_sentence_generator import AISentenceGenerator
from conversation_context import ConversationContext


class CommunicationPipeline:

    def __init__(self):

        self.communication = CommunicationEngine()

        self.sentence_generator = AISentenceGenerator()

        self.context = ConversationContext()


    # ==========================================
    # ADD SIGN
    # ==========================================

    def add_sign(self, sign):

        self.communication.add_token(sign)


    # ==========================================
    # SET MODE
    # ==========================================

    def set_mode(self, mode):

        self.context.set_mode(mode)


    # ==========================================
    # FINISH MESSAGE
    # ==========================================

    def finish_message(self):

        tokens = self.communication.tokens.copy()

        if not tokens:
            return ""

        sentence = self.sentence_generator.generate(
            tokens,
            self.context
        )

        # Store generated sentence
        self.context.add_message(
            "USER",
            sentence
        )

        # Clear signs after sentence generation
        self.communication.clear()

        return sentence


    # ==========================================
    # ADD INTERVIEWER MESSAGE
    # ==========================================

    def add_interviewer_message(self, text):

        self.context.add_message(
            "INTERVIEWER",
            text
        )


    # ==========================================
    # GET CONTEXT
    # ==========================================

    def get_context(self):

        return self.context


    # ==========================================
    # GET CURRENT TOKENS
    # ==========================================

    def get_tokens(self):

        return self.communication.tokens.copy()


    # ==========================================
    # CLEAR
    # ==========================================

    def clear(self):

        self.communication.clear()
        self.context.clear()