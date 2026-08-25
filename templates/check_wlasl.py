import json
import urllib.request


URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"


TARGET_SIGNS = [
    "hello",
    "yes",
    "no",
    "please",
    "thank_you",
    "bye",
    "help",
    "sorry",
    "i",
    "you",
    "need",
    "want",
    "water",
    "food",
    "like",
    "good",
    "bad",
    "student",
    "computer",
    "ai",
    "work",
    "learn",
    "understand"
]


print("==============================")
print("WLASL DATASET CHECK")
print("==============================")
print()
print("Downloading WLASL metadata...")
print()


try:

    with urllib.request.urlopen(URL) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

except Exception as e:

    print("ERROR downloading metadata:")
    print(e)
    exit()


# Create lookup of available glosses

available = {}

for item in data:

    gloss = item.get("gloss", "").lower().strip()

    available[gloss] = item


print(
    f"Total WLASL glosses found: {len(available)}"
)

print()
print("==============================")
print("TARGET SIGNS")
print("==============================")
print()


found = []
missing = []


for sign in TARGET_SIGNS:

    if sign.lower() in available:

        print(
            f"{sign.upper():25} -> FOUND"
        )

        found.append(sign)

    else:

        print(
            f"{sign.upper():25} -> NOT FOUND"
        )

        missing.append(sign)


print()
print("==============================")
print("SUMMARY")
print("==============================")

print(
    f"Found   : {len(found)}"
)

print(
    f"Missing : {len(missing)}"
)

print()
print("Found signs:")
print(found)

print()
print("Missing signs:")
print(missing)

print()