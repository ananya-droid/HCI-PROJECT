from communication_pipeline import CommunicationPipeline


print("==============================")
print("CONTEXTUAL INTERVIEW TEST")
print("==============================")
print()


pipeline = CommunicationPipeline()


# ==========================================
# INTERVIEW MODE
# ==========================================

pipeline.set_mode("INTERVIEW")


# ==========================================
# INTERVIEWER QUESTION
# ==========================================

pipeline.add_interviewer_message(
    "Tell me about yourself."
)


# ==========================================
# USER SIGNS
# ==========================================

pipeline.add_sign("I")
pipeline.add_sign("STUDENT")
pipeline.add_sign("COMPUTER_SCIENCE")
pipeline.add_sign("AI")


# ==========================================
# GENERATE ANSWER
# ==========================================

answer = pipeline.finish_message()


print()
print("Interviewer:")
print("Tell me about yourself.")
print()

print("User signs:")
print(
    "I STUDENT COMPUTER_SCIENCE AI"
)

print()

print("Generated answer:")
print(answer)

print()


# ==========================================
# SHOW HISTORY
# ==========================================

print("==============================")
print("CONVERSATION HISTORY")
print("==============================")

for message in pipeline.get_context().get_history():

    print(
        message["speaker"],
        ":",
        message["text"]
    )

print()