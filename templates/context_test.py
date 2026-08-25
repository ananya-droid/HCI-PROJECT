# context_test.py

from conversation_context import ConversationContext


context = ConversationContext()


print("==============================")
print("CONVERSATION CONTEXT TEST")
print("==============================")


# Default mode
print()
print("Current mode:")
print(context.get_mode())


# Change mode
context.set_mode("INTERVIEW")

print()
print("Changed mode:")
print(context.get_mode())


# Add conversation
context.add_message(
    "INTERVIEWER",
    "Tell me about yourself."
)

context.add_message(
    "USER",
    "I'm a computer science student."
)


# Display history
print()
print("Conversation history:")

for message in context.get_history():

    print(
        message["speaker"],
        ":",
        message["text"]
    )


# Recent history
print()
print("Recent history:")

print(
    context.get_recent_history()
)


# Reset
context.reset()

print()
print("After reset:")

print("Mode:", context.get_mode())
print("History:", context.get_history())