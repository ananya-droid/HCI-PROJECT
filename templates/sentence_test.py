from sentence_generator import SentenceGenerator


generator = SentenceGenerator()


tests = [
    ["HELLO"],
    ["HELLO", "YES"],
    ["I", "NEED", "WATER"],
    ["I", "NEED", "HELP"],
]


for tokens in tests:

    sentence = generator.generate(tokens)

    print("Tokens:", tokens)
    print("Sentence:", sentence)
    print()