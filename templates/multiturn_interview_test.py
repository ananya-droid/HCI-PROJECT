from communication_pipeline import CommunicationPipeline


print("==============================")
print("MULTI-TURN INTERVIEW TEST")
print("==============================")


pipeline = CommunicationPipeline()

pipeline.set_mode("INTERVIEW")


# ==========================================
# QUESTION 1
# ==========================================

question1 = "Tell me about yourself."

pipeline.add_interviewer_message(
    question1
)


# User signs answer
pipeline.add_sign("I")
pipeline.add_sign("STUDENT")
pipeline.add_sign("COMPUTER_SCIENCE")
pipeline.add_sign("AI")


answer1 = pipeline.finish_message()


print()
print("INTERVIEWER:")
print(question1)

print()
print("USER:")
print(answer1)


# ==========================================
# QUESTION 2
# ==========================================

question2 = "Why are you interested in AI?"

pipeline.add_interviewer_message(
    question2
)


# User signs answer
pipeline.add_sign("I")
pipeline.add_sign("LIKE")
pipeline.add_sign("AI")


answer2 = pipeline.finish_message()


print()
print("INTERVIEWER:")
print(question2)

print()
print("USER:")
print(answer2)


# ==========================================
# CONVERSATION HISTORY
# ==========================================

print()
print("==============================")
print("FULL CONVERSATION")
print("==============================")


for message in pipeline.get_context().get_history():

    print(
        message["speaker"],
        ":",
        message["text"]
    )