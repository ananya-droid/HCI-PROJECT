# ai_sentence_generator.py

import os
from google import genai


class AISentenceGenerator:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.6-flash"


    def generate(self, tokens, context):

        if not tokens:
            return ""

        mode = context.get_mode()

        history = context.get_recent_history(
            limit=6
        )

        token_text = " ".join(
            str(token).upper()
            for token in tokens
        )

        history_text = ""

        for message in history:

            history_text += (
                f"{message['speaker']}: "
                f"{message['text']}\n"
            )


        prompt = f"""
You are the language-generation component of a
sign-language communication system.

The user cannot speak and communicates through
recognized sign-language concepts.

Your task is to convert the recognized concepts
into natural, grammatically correct English.

CONVERSATION MODE:
{mode}

RECENT CONVERSATION:
{history_text}

CURRENT SIGN CONCEPTS:
{token_text}

RULES:

1. Preserve the user's intended meaning.
2. Do not invent facts.
3. Do not invent qualifications.
4. Do not invent work experience.
5. Do not add information that the user did not communicate.
6. Do not answer questions on behalf of the user.
7. Do not change the user's intention.
8. You may rearrange words to make natural English.
9. Correct grammar.
10. Consider the conversation context.
11. If the mode is INTERVIEW, use professional,
    natural English.
12. Keep the response concise.
13. Return ONLY the final sentence.

Convert the sign concepts into natural English.
"""


        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        sentence = response.text.strip()

        return sentence