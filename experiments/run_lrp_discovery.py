import os
import json
import torch
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Robust absolute path resolution — works from any working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from circuits.lrp import get_lrp_scores
from utils.logging import log_experiment, set_seed


def load_data(task, n_samples):
    path = REPO_ROOT / "data" / task / "dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:n_samples]


def run_lrp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model name/path. Defaults to gemma-3-270m-it for tool_selection, else gemma-3-270m.")
    parser.add_argument("--task", type=str, choices=["ioi", "greater_than", "tool_selection"], default="ioi")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.model is None:
        args.model = "google/gemma-3-270m-it" if args.task == "tool_selection" else "google/gemma-3-270m"

    print(f"Loading data for {args.task}...")
    dataset = load_data(args.task, args.n_samples)

    print(f"Loading tokenizer and model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    aggregated_scores = {}

    print(f"Running LRP on {args.task} ({args.n_samples} samples)...")
    for i, sample in enumerate(dataset):
        if args.task == "ioi":
            text = sample["clean_prompt"]
            target_id = tokenizer.encode(" " + sample["target_clean"], add_special_tokens=False)[-1]
            distractor_id = tokenizer.encode(" " + sample["target_corrupted"], add_special_tokens=False)[-1]
        elif args.task == "greater_than":
            text = sample["clean_prompt"]
            target_id = tokenizer.encode(str(sample["target_clean_suffix"]), add_special_tokens=False)[-1]
            distractor_id = tokenizer.encode(str(sample["target_corrupted_suffix"]), add_special_tokens=False)[-1]
        elif args.task == "tool_selection":
            tools = sample["tools"]
            text = tokenizer.apply_chat_template(
                sample["clean_messages"], tools=tools, tokenize=False, add_generation_prompt=True
            )
            target_id = tokenizer.encode(sample["target_clean_tool"], add_special_tokens=False)[-1]
            distractor_id = tokenizer.encode(sample["target_corrupted_tool"], add_special_tokens=False)[-1]

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        scores = get_lrp_scores(model, input_ids, target_id, distractor_id)

        for k, v in scores.items():
            if isinstance(v, list):
                if k not in aggregated_scores:
                    aggregated_scores[k] = list(v)  # explicit copy — avoid reference aliasing
                else:
                    aggregated_scores[k] = [a + b for a, b in zip(aggregated_scores[k], v)]
            else:
                aggregated_scores[k] = aggregated_scores.get(k, 0.0) + v

        max_head = max([k for k in scores.keys() if "attn" in k], key=lambda k: scores[k])
        print(f"Sample {i+1}: Top Attn Head = {max_head} (score: {scores[max_head]:.4f})")

    # Average accumulated scores
    for k in aggregated_scores:
        if isinstance(aggregated_scores[k], list):
            aggregated_scores[k] = [v / args.n_samples for v in aggregated_scores[k]]
        else:
            aggregated_scores[k] = aggregated_scores[k] / args.n_samples

    head_scores = {k: v for k, v in aggregated_scores.items() if "attn" in k}
    top_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n=== LRP TOP 5 GLOBAL ATTENTION HEADS ===")
    for head, score in top_heads:
        print(f"  {head}: {score:.4f}")

    # Save circuit
    circuits_dir = REPO_ROOT / "results" / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    circuit_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_lrp_circuit.json"
    with open(circuit_file, "w") as f:
        json.dump(aggregated_scores, f)
    print(f"Saved LRP circuit to {circuit_file}")

    log_experiment(
        task=args.task,
        method="LRP",
        config={**vars(args), "circuit_file": str(circuit_file)},
        metrics={"status": "success", "samples": args.n_samples},
        base_dir=str(REPO_ROOT / "results")
    )


if __name__ == "__main__":
    run_lrp()
