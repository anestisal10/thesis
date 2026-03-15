import json
import random
import os
from pathlib import Path

# 50 names (mix of male/female) — matches PROGRESS.md spec
NAMES = [
    # Male
    "John", "David", "Michael", "Daniel", "James", "Robert", "William", "Joseph",
    "Thomas", "Charles", "Christopher", "Matthew", "Andrew", "Anthony", "Mark",
    "Paul", "Steven", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy",
    "Jason", "Kenneth",
    # Female
    "Mary", "Sarah", "Emma", "Olivia", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Emily", "Abigail", "Elizabeth", "Sofia", "Avery",
    "Ella", "Scarlett", "Grace", "Chloe", "Victoria", "Riley", "Aria", "Lily",
    "Zoey", "Penelope"
]

PLACES = ["store", "park", "school", "hospital", "station", "market", "library", "cinema"]
OBJECTS = ["drink", "book", "bag", "ring", "letter", "box", "key", "flower"]

def generate_ioi_dataset(num_samples=1000, output_dir=None, seed=42):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "ioi"
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    dataset = []

    while len(dataset) < num_samples:
        s_name, io_name, distractor = random.sample(NAMES, 3)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)

        # Clean: "When John and Mary went to the store, John gave a drink to" -> Target: Mary
        clean_text = f"When {s_name} and {io_name} went to the {place}, {s_name} gave a {obj} to"

        # Corrupted: Swap IO with distractor so the circuit sees a different IO
        corrupted_text = f"When {s_name} and {distractor} went to the {place}, {s_name} gave a {obj} to"

        dataset.append({
            "clean_prompt": clean_text,
            "corrupted_prompt": corrupted_text,
            "target_clean": io_name,       # correct answer for clean
            "target_corrupted": distractor  # correct answer for corrupted
        })

    file_path = os.path.join(output_dir, "dataset.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"Generated {len(dataset)} IOI samples at {file_path}")
    return file_path

if __name__ == "__main__":
    generate_ioi_dataset()
