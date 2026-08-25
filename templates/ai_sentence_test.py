# ai_sentence_test.py

from ai_sentence_generator import AISentenceGenerator
from conversation_context import ConversationContext


generator = AISentenceGenerator()
context = ConversationContext()


print("==============================")
print("AI SENTENCE GENERATOR TEST")
print("==============================")


# ==========================================
# GENERAL
# ==========================================

tokens = [
    "I",
    "NEED",
    "WATER"
]

sentence = generator.generate(
    tokens,
    context
)

print()
print("Mode:", context.get_mode())
print("Tokens:", tokens)
print("Generated:", sentence)


# ==========================================
# INTERVIEW
# ==========================================

context.set_mode("INTERVIEW")

context.add_message(
    "INTERVIEWER",
    "Tell me about yourself."
)


tokens = [
    "I",
    "STUDENT",
    "COMPUTER_SCIENCE",
    "AI"
]

sentence = generator.generate(
    tokens,
    context
)

print()
print("Mode:", context.get_mode())
print("Tokens:", tokens)
print("Generated:", sentence)