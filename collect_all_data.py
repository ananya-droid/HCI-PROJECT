"""
collect_all_data.py
---------------------
Walks through the full vocabulary (config.ALL_VOCABULARY) in one sitting,
so one contributor can record all 15 core signs without re-launching the
script each time.

Usage:
    python collect_all_data.py --person person2 --samples 30

For each label it will:
    1. Print the label and wait for you to press ENTER when ready.
    2. Reuse the same recording loop as collect_data.py.
    3. Move on to the next label automatically.

Press 'q' at any point during a label's recording window to skip
to the next label early, or Ctrl+C to stop the whole session.
"""

import argparse

import config
from collect_data import collect


def collect_all(person: str, samples_per_sign: int, vocabulary=None):
    vocabulary = vocabulary or config.ALL_VOCABULARY

    print("=" * 50)
    print(f"COLLECT ALL DATA — person: {person}")
    print(f"Signs to record: {', '.join(vocabulary)}")
    print("=" * 50)

    for label in vocabulary:
        input(f"\nNext sign: '{label}'. Press ENTER when ready to start "
              f"(a window will open; SPACE to record each sample, 'q' to move on)...")
        try:
            collect(label=label, person=person, num_samples=samples_per_sign)
        except KeyboardInterrupt:
            print("\n[collect_all_data] Interrupted by user. Stopping session.")
            return

    print("\n[collect_all_data] All signs recorded for this session.")
    print("Run 'python check_dataset.py' next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record every sign in the vocabulary in one session.")
    parser.add_argument("--person", default="person1", help="Contributor id")
    parser.add_argument("--samples", type=int, default=30, help="Samples per sign")
    args = parser.parse_args()

    collect_all(args.person, args.samples)
