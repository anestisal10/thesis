import json
import random
import os
from pathlib import Path

NOUNS = ["war", "famine", "campaign", "reign", "epidemic", "crisis", "festival",
         "tournament", "project", "expedition"]

def generate_greater_than_dataset(num_samples=1000, output_dir=None, seed=42):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "greater_than"
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    dataset = []

    while len(dataset) < num_samples:
        noun = random.choice(NOUNS)
        # Choose a century prefix, e.g. 17, 18, 19
        prefix = random.choice([16, 17, 18, 19])

        # year1 = prefix + (02 to 70) to ensure we can always pick a smaller year
        y1_suffix = random.randint(2, 70)
        year1 = f"{prefix}{y1_suffix:02d}"

        # clean_year2 > year1
        clean_y2_suffix = random.randint(y1_suffix + 1, y1_suffix + 15)

        # corrupted_year2 < year1 (to break the "greater than" expectation)
        corrupted_y2_suffix = random.randint(max(1, y1_suffix - 15), y1_suffix - 1)

        clean_text = f"The {noun} lasted from the year {year1} to the year {prefix}"
        corrupted_text = f"The {noun} lasted from the year {year1} to the year {prefix}"

        dataset.append({
            "clean_prompt": clean_text,
            "corrupted_prompt": corrupted_text,
            "year1_suffix": y1_suffix,
            "target_clean_suffix": clean_y2_suffix,
            "target_corrupted_suffix": corrupted_y2_suffix
        })

    file_path = os.path.join(output_dir, "dataset.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"Generated {len(dataset)} Greater-Than samples at {file_path}")
    return file_path

if __name__ == "__main__":
    generate_greater_than_dataset()
