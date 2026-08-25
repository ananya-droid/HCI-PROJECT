# ai_sentence_generator.py

import os
import time
import re

from google import genai


class AISentenceGenerator:

    def __init__(self):

        # ==========================================
        # LOAD GEMINI API KEY
        # ==========================================

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:

            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )


        # ==========================================
        # GEMINI CLIENT
        # ==========================================

        self.client = genai.Client(
            api_key=api_key
        )


        # ==========================================
        # MODEL
        # ==========================================

        self.model = "gemini-3.6-flash"


    # ==============================================
    # GENERATE NATURAL ENGLISH
    # ==============================================

    def generate(self, tokens, context):

        if not tokens:

            return ""


        # ==========================================
        # CURRENT MODE
        # ==========================================

        mode = context.get_mode()


        # ==========================================
        # RECENT CONVERSATION
        # ==========================================

        history = context.get_recent_history(
            limit=8
        )


        # ==========================================
        # CONVERT TOKENS TO TEXT
        # ==========================================

        token_text = " ".join(
            str(token).upper()
            for token in tokens
        )


        # ==========================================
        # BUILD CONVERSATION HISTORY
        # ==========================================

        history_lines = []

        for message in history:

            speaker = message.get(
                "speaker",
                "UNKNOWN"
            )

            text = message.get(
                "text",
                ""
            )

            history_lines.append(
                f"{speaker}: {text}"
            )


        if history_lines:

            history_text = "\n".join(
                history_lines
            )

        else:

            history_text = (
                "No previous conversation."
            )


        # ==========================================
        # MODE-SPECIFIC INSTRUCTIONS
        # ==========================================

        if mode == "INTERVIEW":

            mode_instruction = """
The user is participating in a job interview.

Use professional, natural spoken English.

If the signs represent an answer to an
interview question, phrase them naturally as
an interview answer.

Do not make up qualifications, projects,
experience, achievements, skills, or facts.

Only express information represented by the
user's signs or clearly implied by them.
"""

        else:

            mode_instruction = """
This is a normal conversation.

Use natural everyday spoken English.

Keep the user's meaning exactly the same.
"""


        # ==========================================
        # GEMINI PROMPT
        # ==========================================

        prompt = f"""
You are the natural-language generation component
of a real-time sign-language communication system.

The user cannot communicate verbally and is using
recognized sign concepts to communicate.

Your job is NOT to translate each sign literally.

Your job is to reconstruct the user's intended
message as natural spoken English.

------------------------------------------
CONVERSATION MODE
------------------------------------------

{mode}

{mode_instruction}

------------------------------------------
RECENT CONVERSATION
------------------------------------------

{history_text}

------------------------------------------
CURRENT USER SIGN CONCEPTS
------------------------------------------

{token_text}

------------------------------------------
IMPORTANT RULES
------------------------------------------

1. Preserve the user's intended meaning.

2. Produce natural, grammatically correct
   English.

3. Rearrange sign concepts when necessary.

4. Add small grammatical words such as:
   "a", "an", "the", "is", "am", "are",
   "to", "some", etc. when required.

5. Do NOT translate signs word-for-word when
   that would produce unnatural English.

6. Do NOT invent facts.

7. Do NOT invent experiences.

8. Do NOT invent qualifications.

9. Do NOT invent emotions.

10. Do NOT invent actions that the user did
    not communicate.

11. Conversation history may be used to understand
    the meaning of the current signs.

12. Conversation history must NOT cause you to
    add information that the user did not communicate.

13. Generate ONLY the user's message.

14. Never answer on behalf of the interviewer.

15. Never continue the conversation yourself.

16. Never generate dialogue such as:
    "Interviewer: ..."
    or
    "You could say ..."

17. Return ONLY the final sentence.

18. Keep the output concise and natural.

------------------------------------------
EXAMPLES
------------------------------------------

Signs:
HELLO

Natural English:
Hello.


Signs:
I NEED WATER

Natural English:
I need some water.


Signs:
I NEED HELP

Natural English:
I need help.


Signs:
HELLO YES

Natural English:
Hello, yes.


Signs:
I STUDENT COMPUTER_SCIENCE AI

Natural English:
I'm a computer science student interested in AI.


Signs:
I LIKE AI

Natural English:
I'm interested in AI.


Signs:
THANK_YOU

Natural English:
Thank you.


Signs:
PLEASE HELP

Natural English:
Please help me.


------------------------------------------
CONTEXT EXAMPLE
------------------------------------------

Previous conversation:

INTERVIEWER:
Tell me about yourself.

Current signs:

I STUDENT COMPUTER_SCIENCE AI

Good output:

I'm a computer science student interested in AI.

Notice that the answer is based only on the
information communicated by the user.

------------------------------------------
CURRENT TASK
------------------------------------------

Convert these current sign concepts into
one natural English sentence:

{token_text}

Return ONLY the sentence.
"""


        # ==========================================
        # GEMINI REQUEST
        # ==========================================

        for attempt in range(3):

            try:

                response = (
                    self.client
                    .models
                    .generate_content(
                        model=self.model,
                        contents=prompt
                    )
                )


                # ==================================
                # GET RESPONSE
                # ==================================

                if response.text:

                    sentence = response.text.strip()


                    # ==================================
                    # CLEAN GEMINI OUTPUT
                    # ==================================

                    sentence = (
                        sentence
                        .replace(
                            "```",
                            ""
                        )
                        .strip()
                    )


                    # Remove accidental prefixes

                    sentence = re.sub(
                        r"^(Answer|Response|Sentence)\s*:\s*",
                        "",
                        sentence,
                        flags=re.IGNORECASE
                    ).strip()


                    # Remove surrounding quotes

                    if (
                        len(sentence) >= 2
                        and sentence[0] in "\"'"
                        and sentence[-1] == sentence[0]
                    ):

                        sentence = sentence[1:-1].strip()


                    return sentence


            except Exception as e:

                print(
                    f"Gemini attempt "
                    f"{attempt + 1}/3 failed:"
                )

                print(e)


                if attempt < 2:

                    time.sleep(2)


        # ==========================================
        # FALLBACK
        # ==========================================

        print(
            "Gemini unavailable. "
            "Using fallback."
        )

        return self._fallback(tokens)


    # ==============================================
    # FALLBACK
    # ==============================================

    def _fallback(self, tokens):

        text = " ".join(
            str(token).upper()
            for token in tokens
        )


        if not text:

            return ""


        sentence = text.capitalize()


        if not sentence.endswith(
            (".", "!", "?")
        ):

            sentence += "."


        return sentence