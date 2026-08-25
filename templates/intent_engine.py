# intent_engine.py


class IntentEngine:

    def __init__(self):
        self.last_intent = None
        self.conversation_history = []

    # ==========================================
    # DETECT INTENT
    # ==========================================

    def detect_intent(self, tokens):

        if not tokens:
            return "NONE"

        tokens = [token.upper() for token in tokens]

        # Greeting
        if "HELLO" in tokens:
            return "GREETING"

        # Future HELP support
        if "HELP" in tokens:
            return "HELP_REQUEST"

        # Future WATER support
        if "WATER" in tokens:
            return "WATER_REQUEST"

        # Future FOOD support
        if "FOOD" in tokens:
            return "FOOD_REQUEST"

        # Future NEED support
        if "NEED" in tokens:
            return "NEED_REQUEST"

        # YES depends on previous context
        if "YES" in tokens:

            if self.last_intent == "HELP_REQUEST":
                return "HELP_CONFIRMATION"

            return "CONFIRMATION"

        # NO depends on previous context
        if "NO" in tokens:

            if self.last_intent == "HELP_REQUEST":
                return "HELP_DECLINED"

            return "NEGATION"

        # Future THANK_YOU
        if "THANK_YOU" in tokens:
            return "GRATITUDE"

        return "GENERAL_COMMUNICATION"


    # ==========================================
    # GENERATE RESPONSE
    # ==========================================

    def generate_response(self, intent):

        responses = {

            "GREETING":
                "Hello! How can I help you?",

            "CONFIRMATION":
                "Okay.",

            "NEGATION":
                "Okay, I understand.",

            "HELP_REQUEST":
                "Sure. I am here to help you.",

            "HELP_CONFIRMATION":
                "Okay. I will help you.",

            "HELP_DECLINED":
                "Okay. I understand.",

            "WATER_REQUEST":
                "Sure. You need water.",

            "FOOD_REQUEST":
                "Sure. You need food.",

            "NEED_REQUEST":
                "Okay. What do you need?",

            "GRATITUDE":
                "You're welcome.",

            "GENERAL_COMMUNICATION":
                "I understand.",

            "NONE":
                ""
        }

        return responses.get(
            intent,
            "I understand."
        )


    # ==========================================
    # PROCESS COMMUNICATION
    # ==========================================

    def process(self, tokens):

        intent = self.detect_intent(tokens)

        response = self.generate_response(intent)

        self.last_intent = intent

        self.conversation_history.append({

            "tokens": tokens.copy(),

            "intent": intent,

            "response": response
        })

        return intent, response


    # ==========================================
    # CLEAR CONTEXT
    # ==========================================

    def clear(self):

        self.last_intent = None

        self.conversation_history = []