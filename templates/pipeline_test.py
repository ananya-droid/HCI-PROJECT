from communication_pipeline import CommunicationPipeline


pipeline = CommunicationPipeline()


print("==============================")
print("COMMUNICATION PIPELINE TEST")
print("==============================")


# ------------------------------------------
# GENERAL CONVERSATION
# ------------------------------------------

pipeline.add_sign("HELLO")
pipeline.add_sign("YES")

sentence = pipeline.finish_message()

print()
print("Generated:")
print(sentence)


# ------------------------------------------
# INTERVIEW MODE
# ------------------------------------------

pipeline.set_mode("INTERVIEW")

pipeline.add_interviewer_message(
    "Tell me about yourself."
)

pipeline.add_sign("I")
pipeline.add_sign("NEED")
pipeline.add_sign("HELP")

sentence = pipeline.finish_message()

print()
print("Interview Generated:")
print(sentence)


# ------------------------------------------
# HISTORY
# ------------------------------------------

print()
print("Conversation History:")

for message in pipeline.get_context().get_history():

    print(
        message["speaker"],
        ":",
        message["text"]
    )