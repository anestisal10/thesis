import json
import random
import os
from pathlib import Path

# Provide a tokenizer instance if you strictly want to check single-token.
# However, Gemma-3 tokenizes numbers digit-by-digit. To guarantee a "Single-Token"
# Mechanistic Interpretability metric (as requested) while still generating 1000 pairs,
# we ensure the FIRST digit of the answer is different between the clean and corrupted pairs,
# and use that single digit (which is always 1 token in Gemma) as the target.

def generate_arithmetic_dataset(num_pairs=1000, output_dir=None, seed=42):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "arithmetic"
    
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = []
    
    # We want num_pairs total pairs: half addition, half subtraction
    # A single "pair" means a Clean prompt and a Corrupted prompt.
    half = num_pairs // 2
    
    seen_add = set()
    while len(dataset) < half:
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)
        if num1 < num2:
            num1, num2 = num2, num1  # Ensure num1 >= num2 for positive subtraction
            
        if (num1, num2) in seen_add:
            continue
            
        ans_clean = num1 + num2
        ans_corr = num1 - num2
        
        # We need the first digit to be different so logit diff is meaningful.
        target_clean_token = str(ans_clean)[0]
        target_corr_token = str(ans_corr)[0]
        
        if target_clean_token == target_corr_token:
            continue
            
        clean_prompt = f"Calculate the result of the following arithmetic expression and provide only the final answer: {num1} + {num2} ="
        corr_prompt = f"Calculate the result of the following arithmetic expression and provide only the final answer: {num1} - {num2} ="
        
        dataset.append({
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corr_prompt,
            "answer_clean": ans_clean,
            "answer_corrupted": ans_corr,
            "target_clean_token": target_clean_token,
            "target_corrupted_token": target_corr_token,
            "operator": "+"
        })
        seen_add.add((num1, num2))

    seen_sub = set()
    while len(dataset) < num_pairs:
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)
        if num1 < num2:
            num1, num2 = num2, num1
            
        if (num1, num2) in seen_sub:
            continue
            
        ans_clean = num1 - num2
        ans_corr = num1 + num2
        
        target_clean_token = str(ans_clean)[0]
        target_corr_token = str(ans_corr)[0]
        
        if target_clean_token == target_corr_token:
            continue
            
        clean_prompt = f"Calculate the result of the following arithmetic expression and provide only the final answer: {num1} - {num2} ="
        corr_prompt = f"Calculate the result of the following arithmetic expression and provide only the final answer: {num1} + {num2} ="
        
        dataset.append({
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corr_prompt,
            "answer_clean": ans_clean,
            "answer_corrupted": ans_corr,
            "target_clean_token": target_clean_token,
            "target_corrupted_token": target_corr_token,
            "operator": "-"
        })
        seen_sub.add((num1, num2))
        
    random.shuffle(dataset)
    
    file_path = os.path.join(output_dir, "dataset.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Generated {len(dataset)} pairs at {file_path}")
    return file_path

if __name__ == "__main__":
    generate_arithmetic_dataset()
