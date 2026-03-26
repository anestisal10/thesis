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
from utils.data import load_dataset_split
from utils.logging import log_experiment, set_seed
from utils.tool_selection import build_tool_call_message, find_subsequence
from utils.tokenization import get_single_token_id


def load_data(task, n_samples, split, split_seed):
    path = REPO_ROOT / "data" / task / "dataset.json"
    return load_dataset_split(path, split=split, n_samples=n_samples, split_seed=split_seed)


def tokenize_and_embed(tokenizer, model, text_a, text_b):
    """
    Tokenize two texts and return embeddings aligned to the same length
    via LEFT-padding ensuring semantic alignment of the final tokens.
    """
    enc_a = tokenizer(text_a, return_tensors="pt").input_ids.to(model.device)
    enc_b = tokenizer(text_b, return_tensors="pt").input_ids.to(model.device)

    len_a, len_b = enc_a.shape[1], enc_b.shape[1]
    max_len = max(len_a, len_b)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id. Set tokenizer.pad_token before calling.")
    pad_id = tokenizer.pad_token_id

    # Left Pad
    if len_a < max_len:
        pad = torch.full((1, max_len - len_a), pad_id, dtype=torch.long, device=model.device)
        enc_a = torch.cat([pad, enc_a], dim=1)
    if len_b < max_len:
        pad = torch.full((1, max_len - len_b), pad_id, dtype=torch.long, device=model.device)
        enc_b = torch.cat([pad, enc_b], dim=1)

    # Attention masks
    mask_a = (enc_a != pad_id).long()
    mask_b = (enc_b != pad_id).long()
    ig_mask = (mask_a | mask_b).long()

    with torch.no_grad():
        clean_embeds = model.get_input_embeddings()(enc_a).detach()
        corrupted_embeds = model.get_input_embeddings()(enc_b).detach()

    return clean_embeds, corrupted_embeds, mask_a, mask_b, ig_mask


def pad_and_embed_ids(tokenizer, model, ids_a, ids_b):
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id. Set tokenizer.pad_token before calling.")
    pad_id = tokenizer.pad_token_id
    ids_a = torch.tensor([ids_a], device=model.device)
    ids_b = torch.tensor([ids_b], device=model.device)
    len_a, len_b = ids_a.shape[1], ids_b.shape[1]
    max_len = max(len_a, len_b)
    pad_len_a = max_len - len_a
    pad_len_b = max_len - len_b
    if pad_len_a > 0:
        pad = torch.full((1, pad_len_a), pad_id, dtype=torch.long, device=model.device)
        ids_a = torch.cat([pad, ids_a], dim=1)
    if pad_len_b > 0:
        pad = torch.full((1, pad_len_b), pad_id, dtype=torch.long, device=model.device)
        ids_b = torch.cat([pad, ids_b], dim=1)

    mask_a = (ids_a != pad_id).long()
    mask_b = (ids_b != pad_id).long()
    ig_mask = (mask_a | mask_b).long()

    with torch.no_grad():
        embeds_a = model.get_input_embeddings()(ids_a).detach()
        embeds_b = model.get_input_embeddings()(ids_b).detach()

    return embeds_a, embeds_b, mask_a, mask_b, ig_mask, pad_len_a, pad_len_b


def run_once(args):
    set_seed(args.seed)

    if args.model is None:
        args.model = "google/functiongemma-270m-it" if args.task == "tool_selection" else "google/gemma-3-270m"

    print(f"Loading data for {args.task} ({args.split} split)...")
    dataset = load_data(args.task, args.n_samples, args.split, args.split_seed)

    print(f"Loading tokenizer and model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token_id or eos_token. Cannot safely pad.")
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # Removed greater_than pre-calculated token IDs as they are no longer needed

    aggregated_scores = {}
    n_used = 0

    print("Running EAP-IG Loop...")
    for i, sample in enumerate(dataset):
        if args.task == "ioi":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            try:
                target_clean_id = get_single_token_id(tokenizer, sample["target_clean"], prefer_space=True)
                target_corrupted_id = get_single_token_id(tokenizer, sample["target_corrupted"], prefer_space=True)
            except ValueError:
                print(f"Sample {i+1} skipped: IOI targets are not single-token.")
                continue
            
            def logit_diff_metric(logits):
                final_logits = logits[0, -1, :]
                return final_logits[target_clean_id] - final_logits[target_corrupted_id]

        elif args.task == "arithmetic":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            
            target_clean_id = get_single_token_id(tokenizer, sample["target_clean_token"], prefer_space=False)
            target_corrupted_id = get_single_token_id(tokenizer, sample["target_corrupted_token"], prefer_space=False)
            
            def logit_diff_metric(logits):
                final_logits = logits[0, -1, :]
                return final_logits[target_clean_id] - final_logits[target_corrupted_id]

        elif args.task == "tool_selection":
            # 1. Load keys from the new EAP dataset
            tools = sample["tools_schema"]
            
            # 2. Apply Chat Template to get the base prompts (returns list of ints)
            clean_prompt_ids = tokenizer.apply_chat_template(
                sample["clean_messages"], tools=tools, tokenize=True, add_generation_prompt=True
            )
            corrupted_prompt_ids = tokenizer.apply_chat_template(
                sample["corrupted_messages"], tools=tools, tokenize=True, add_generation_prompt=True
            )
            
            # 3. Append the forced prefix ("<start_function_call>call:")
            prefix_ids = tokenizer.encode(sample["required_prefix"], add_special_tokens=False)
            clean_full_ids = clean_prompt_ids + prefix_ids
            corrupted_full_ids = corrupted_prompt_ids + prefix_ids
            
            # 4. Get the Target Token IDs (e.g., "measure" vs "check")
            # We use get_single_token_id to ensure it's mapped to exactly 1 vocabulary id
            target_clean_id = get_single_token_id(tokenizer, sample["clean_target_token"])
            target_corrupted_id = get_single_token_id(tokenizer, sample["corrupted_target_token"])

            # 5. Pad and Embed
            clean_embeds, corrupted_embeds, clean_mask, corrupted_mask, ig_mask, pad_len_a, pad_len_b = pad_and_embed_ids(
                tokenizer, model, clean_full_ids, corrupted_full_ids
            )

            # 6. The New Logit Difference Metric
            def logit_diff_metric(logits):
                # logits shape: [batch=1, seq_len, vocab_size]
                # We extract the logit distribution for the very LAST token (the decision point)
                final_logits = logits[0, -1, :]
                return final_logits[target_clean_id] - final_logits[target_corrupted_id]

        if args.task != "tool_selection":
            clean_embeds, corrupted_embeds, clean_mask, corrupted_mask, ig_mask = tokenize_and_embed(
                tokenizer, model, clean_text, corrupted_text
            )

        if not args.dry_run:
            scores = get_eap_ig_scores(
                model=model,
                clean_embeds=clean_embeds,
                corrupted_embeds=corrupted_embeds,
                clean_attention_mask=clean_mask,
                corrupted_attention_mask=corrupted_mask,
                ig_attention_mask=ig_mask,
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
            n_used += 1
        else:
            if args.task == "tool_selection":
                print(f"Sample {i+1} Dry Run: tool_selection skipped (no text prompt).")
                continue
            clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                out = model(input_ids=clean_ids)
                val = logit_diff_metric(out.logits)
                print(f"Sample {i+1} Original Logit Diff: {val.item():.4f}")

    # Finalize: average accumulated scores
    if not args.dry_run:
        if n_used == 0:
            print("No samples were processed successfully; skipping aggregation.")
            log_experiment(
                task=args.task,
                method="EAP-IG",
                config=vars(args),
                metrics={"status": "no_samples", "samples_tested": 0},
                base_dir=str(REPO_ROOT / "results")
            )
            return None
        for k in aggregated_scores:
            if isinstance(aggregated_scores[k], list):
                aggregated_scores[k] = [v / n_used for v in aggregated_scores[k]]
            else:
                aggregated_scores[k] = aggregated_scores[k] / n_used

        head_scores = {k: v for k, v in aggregated_scores.items() if "attn" in k}
        top_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n=== TOP 5 GLOBAL ATTENTION HEADS ===")
        for head, score in top_heads:
            print(f"  {head}: {score:.4f}")

        # Save the full circuit
        circuits_dir = REPO_ROOT / "results" / "circuits"
        circuits_dir.mkdir(parents=True, exist_ok=True)
        seed_tag = f"_seed{args.seed}" if args.seeds else ""
        circuit_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_circuit{seed_tag}.json"
        with open(circuit_file, "w") as f:
            json.dump(aggregated_scores, f)
        print(f"Saved full aggregated circuit to {circuit_file}")

        log_experiment(
            task=args.task,
            method="EAP-IG",
            config={**vars(args), "circuit_file": str(circuit_file)},
            metrics={"status": "success", "samples_tested": n_used},
            base_dir=str(REPO_ROOT / "results")
        )
        return aggregated_scores
    else:
        log_experiment(
            task=args.task,
            method="EAP-IG",
            config=vars(args),
            metrics={"status": "dry_run", "samples_tested": args.n_samples},
            base_dir=str(REPO_ROOT / "results")
        )
        return None


def run_discovery():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model name/path. Defaults to gemma-3-270m-it for tool_selection, else gemma-3-270m.")
    parser.add_argument("--task", type=str, choices=["ioi", "arithmetic", "tool_selection"], default="ioi")
    parser.add_argument("--n-samples", type=int, default=1000) # Reduced default for discovery
    parser.add_argument("--n-steps", type=int, default=5, help="IG interpolation steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed seed used only for train/val/test splitting")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-run discovery")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        all_scores = []
        for s in seeds:
            args.seed = s
            scores = run_once(args)
            if scores is not None:
                all_scores.append(scores)
        if all_scores:
            mean_scores = {}
            std_scores = {}
            keys = all_scores[0].keys()
            for k in keys:
                if isinstance(all_scores[0][k], list):
                    arr = torch.tensor([s[k] for s in all_scores], dtype=torch.float)
                    mean_scores[k] = arr.mean(dim=0).tolist()
                    std_scores[k] = arr.std(dim=0, unbiased=len(all_scores) > 1).tolist()
                else:
                    arr = torch.tensor([s[k] for s in all_scores], dtype=torch.float)
                    mean_scores[k] = float(arr.mean().item())
                    std_scores[k] = float(arr.std(unbiased=len(all_scores) > 1).item())
            circuits_dir = REPO_ROOT / "results" / "circuits"
            circuits_dir.mkdir(parents=True, exist_ok=True)
            mean_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_circuit_seedmean.json"
            std_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_circuit_seedstd.json"
            with open(mean_file, "w") as f:
                json.dump(mean_scores, f)
            with open(std_file, "w") as f:
                json.dump(std_scores, f)
            # Also write a default circuit file for downstream tooling.
            base_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_circuit.json"
            with open(base_file, "w") as f:
                json.dump(mean_scores, f)
            print(f"Saved mean circuit to {mean_file}")
            print(f"Saved std circuit to {std_file}")
            print(f"Saved mean circuit alias to {base_file}")
    else:
        run_once(args)


if __name__ == "__main__":
    run_discovery()
