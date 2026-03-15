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

from circuits.eap_ig import get_eap_ig_scores
from utils.logging import log_experiment, set_seed


def load_data(task, n_samples):
    path = REPO_ROOT / "data" / task / "dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:n_samples]


def tokenize_and_embed(tokenizer, model, text_a, text_b):
    """
    Tokenize two texts and return embeddings aligned to the same length
    via right-padding (not truncation). Returns:
        clean_embeds, corrupted_embeds, attention_mask  — all shape [1, max_len, ...]
    """
    enc_a = tokenizer(text_a, return_tensors="pt").input_ids.to(model.device)
    enc_b = tokenizer(text_b, return_tensors="pt").input_ids.to(model.device)

    len_a, len_b = enc_a.shape[1], enc_b.shape[1]
    max_len = max(len_a, len_b)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Pad the shorter sequence on the right
    if len_a < max_len:
        pad = torch.full((1, max_len - len_a), pad_id, dtype=torch.long, device=model.device)
        enc_a = torch.cat([enc_a, pad], dim=1)
    if len_b < max_len:
        pad = torch.full((1, max_len - len_b), pad_id, dtype=torch.long, device=model.device)
        enc_b = torch.cat([enc_b, pad], dim=1)

    # Attention mask: 1 for real tokens, 0 for padding
    mask_a = (enc_a != pad_id).long()
    mask_b = (enc_b != pad_id).long()
    # Use the union mask (any position real in either input) for the IG pass
    attention_mask = (mask_a | mask_b).long()

    with torch.no_grad():
        clean_embeds = model.get_input_embeddings()(enc_a).detach()
        corrupted_embeds = model.get_input_embeddings()(enc_b).detach()

    return clean_embeds, corrupted_embeds, attention_mask


def run_discovery():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model name/path. Defaults to gemma-3-270m-it for tool_selection, else gemma-3-270m.")
    parser.add_argument("--task", type=str, choices=["ioi", "greater_than", "tool_selection"], default="ioi")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=5, help="IG interpolation steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
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

    print("Running EAP-IG Loop...")
    for i, sample in enumerate(dataset):
        # Build text and target token IDs
        if args.task == "ioi":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            target_clean_id = tokenizer.encode(" " + sample["target_clean"], add_special_tokens=False)[-1]
            target_corrupted_id = tokenizer.encode(" " + sample["target_corrupted"], add_special_tokens=False)[-1]
        elif args.task == "greater_than":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            target_clean_id = tokenizer.encode(str(sample["target_clean_suffix"]), add_special_tokens=False)[-1]
            target_corrupted_id = tokenizer.encode(str(sample["target_corrupted_suffix"]), add_special_tokens=False)[-1]
        elif args.task == "tool_selection":
            tools = sample["tools"]
            clean_text = tokenizer.apply_chat_template(
                sample["clean_messages"], tools=tools, tokenize=False, add_generation_prompt=True
            )
            corrupted_text = tokenizer.apply_chat_template(
                sample["corrupted_messages"], tools=tools, tokenize=False, add_generation_prompt=True
            )
            target_clean_id = tokenizer.encode(sample["target_clean_tool"], add_special_tokens=False)[-1]
            target_corrupted_id = tokenizer.encode(sample["target_corrupted_tool"], add_special_tokens=False)[-1]

        # Padding-based alignment (was: min_len truncation — scientifically incorrect for tool_selection)
        clean_len = tokenizer(clean_text, return_tensors="pt").input_ids.shape[1]
        clean_embeds, corrupted_embeds, attention_mask = tokenize_and_embed(
            tokenizer, model, clean_text, corrupted_text
        )

        def logit_diff_metric(logits):
            final_logits = logits[0, clean_len - 1, :]
            return final_logits[target_clean_id] - final_logits[target_corrupted_id]

        if not args.dry_run:
            scores = get_eap_ig_scores(
                model=model,
                clean_embeds=clean_embeds,
                corrupted_embeds=corrupted_embeds,
                attention_mask=attention_mask,
                metric_fn=logit_diff_metric,
                n_steps=args.n_steps
            )

            # Aggregate scores across samples
            for k, v in scores.items():
                if isinstance(v, list):
                    if k not in aggregated_scores:
                        aggregated_scores[k] = list(v)  # explicit copy — avoid reference aliasing
                    else:
                        aggregated_scores[k] = [a + b for a, b in zip(aggregated_scores[k], v)]
                else:
                    aggregated_scores[k] = aggregated_scores.get(k, 0.0) + v

            print(f"Sample {i+1} EAP-IG Run Complete.")
            max_head = max([k for k in scores.keys() if "attn" in k], key=lambda k: scores[k])
            print(f"  Top Attention Head for this sample: {max_head} (score: {scores[max_head]:.4f})")
        else:
            clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                out = model(input_ids=clean_ids)
                val = logit_diff_metric(out.logits)
                print(f"Sample {i+1} Original Logit Diff: {val.item():.4f}")

    # Finalize: average accumulated scores
    if not args.dry_run:
        for k in aggregated_scores:
            if isinstance(aggregated_scores[k], list):
                aggregated_scores[k] = [v / args.n_samples for v in aggregated_scores[k]]
            else:
                aggregated_scores[k] = aggregated_scores[k] / args.n_samples

        head_scores = {k: v for k, v in aggregated_scores.items() if "attn" in k}
        top_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n=== TOP 5 GLOBAL ATTENTION HEADS ===")
        for head, score in top_heads:
            print(f"  {head}: {score:.4f}")

        # Save the full circuit
        circuits_dir = REPO_ROOT / "results" / "circuits"
        circuits_dir.mkdir(parents=True, exist_ok=True)
        circuit_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_circuit.json"
        with open(circuit_file, "w") as f:
            json.dump(aggregated_scores, f)
        print(f"Saved full aggregated circuit to {circuit_file}")

        log_experiment(
            task=args.task,
            method="EAP-IG",
            config={**vars(args), "circuit_file": str(circuit_file)},
            metrics={"status": "success", "samples_tested": args.n_samples},
            base_dir=str(REPO_ROOT / "results")
        )
    else:
        log_experiment(
            task=args.task,
            method="EAP-IG",
            config=vars(args),
            metrics={"status": "dry_run", "samples_tested": args.n_samples},
            base_dir=str(REPO_ROOT / "results")
        )


if __name__ == "__main__":
    run_discovery()
