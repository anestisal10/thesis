import json
import random
import os
from pathlib import Path

NAMES = [
    "John", "David", "Michael", "Daniel", "James", "Robert", "William", "Joseph",
    "Thomas", "Charles", "Christopher", "Matthew", "Andrew", "Anthony", "Mark",
    "Paul", "Steven", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy",
    "Jason", "Kenneth",
    "Mary", "Sarah", "Emma", "Olivia", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Emily", "Abigail", "Elizabeth", "Sofia", "Avery",
    "Ella", "Scarlett", "Grace", "Chloe", "Victoria", "Riley", "Aria", "Lily",
    "Zoey", "Penelope"
]

PLACES = ["store", "park", "school", "hospital", "station", "market", "library", "cinema"]
OBJECTS = ["drink", "book", "bag", "ring", "letter", "box", "key", "flower"]
VERBS = ["gave", "handed", "passed", "offered"]

TEMPLATES = [
    # A gives to B
    ("A_to_B", lambda A, B, place, obj, verb:
        f"When {A} and {B} went to the {place}, {A} {verb} a {obj} to"
    ),

    # B gives to A
    ("B_to_A", lambda A, B, place, obj, verb:
        f"When {A} and {B} went to the {place}, {B} {verb} a {obj} to"
    ),

    # Variation 1
    ("A_to_B_alt", lambda A, B, place, obj, verb:
        f"{A} and {B} were at the {place}, and {A} {verb} a {obj} to"
    ),

    # Variation 2
    ("B_to_A_alt", lambda A, B, place, obj, verb:
        f"At the {place}, {A} and {B} met, and {B} {verb} a {obj} to"
    ),
]


def generate_ioi_dataset(num_samples=1000, output_dir=None, seed=42):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "ioi"

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    dataset = []

    while len(dataset) < num_samples:
        A, B, C = random.sample(NAMES, 3)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        verb = random.choice(VERBS)

        template_name, template_fn = random.choice(TEMPLATES)

        # CLEAN
        clean_prompt = template_fn(A, B, place, obj, verb)

        if "A_to_B" in template_name:
            target_clean = B

            # Corrupt by replacing IO (B → C)
            corrupted_prompt = template_fn(A, C, place, obj, verb)
            target_corrupted = C

        else:  # B_to_A
            target_clean = A

            # To corrupt IO, we must change the "other" entity (A → C)
            corrupted_prompt = template_fn(C, B, place, obj, verb)
            target_corrupted = C

        dataset.append({
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "target_clean": target_clean,
            "target_corrupted": target_corrupted
        })

    file_path = os.path.join(output_dir, "dataset.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} samples at {file_path}")
    return file_path


if __name__ == "__main__":
    generate_ioi_dataset()