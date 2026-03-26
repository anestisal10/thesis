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
from utils.data import load_dataset_split
from utils.logging import log_experiment, set_seed
from utils.tool_selection import build_tool_call_message, find_subsequence
from utils.tokenization import get_single_token_id


def load_data(task, n_samples, split, split_seed):
    path = REPO_ROOT / "data" / task / "dataset.json"
    return load_dataset_split(path, split=split, n_samples=n_samples, split_seed=split_seed)


def run_once(args):
    set_seed(args.seed)

    if args.model is None:
        args.model = "google/gemma-3-270m-it" if args.task == "tool_selection" else "google/gemma-3-270m"

    print(f"Loading data for {args.task} ({args.split} split)...")
    dataset = load_data(args.task, args.n_samples, args.split, args.split_seed)

    print(f"Loading tokenizer and model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()

    # Removed greater_than pre-calculated token IDs as they are no longer needed

    aggregated_scores = {}
    n_used = 0

    print(f"Running LRP on {args.task} ({args.n_samples} samples)...")
    for i, sample in enumerate(dataset):
        if args.task == "ioi":
            text = sample["clean_prompt"]
            try:
                target_id = get_single_token_id(tokenizer, sample["target_clean"], prefer_space=True)
                distractor_id = get_single_token_id(tokenizer, sample["target_corrupted"], prefer_space=True)
            except ValueError:
                print(f"Sample {i+1} skipped: IOI targets are not single-token.")
                continue
        elif args.task == "arithmetic":
            text = sample["clean_prompt"]
            target_id = get_single_token_id(tokenizer, sample["target_clean_token"], prefer_space=False)
            distractor_id = get_single_token_id(tokenizer, sample["target_corrupted_token"], prefer_space=False)
        elif args.task == "tool_selection":
            tools = sample["tools"]
            target_tool = sample["target_clean_tool"]
            tool_tokens = tokenizer.encode(target_tool, add_special_tokens=False)
            if not tool_tokens:
                raise ValueError(f"Tool name tokenized to empty sequence: {target_tool}")

            prompt_ids = tokenizer.apply_chat_template(
                sample["clean_messages"], tools=tools, tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                sample["clean_messages"] + [build_tool_call_message(target_tool)],
                tools=tools, tokenize=True, add_generation_prompt=False
            )

            start_idx = find_subsequence(full_ids, tool_tokens, start=len(prompt_ids))
            if start_idx < 0:
                raise ValueError("Could not locate tool name tokens in tool-call sequence.")
            positions = list(range(start_idx, start_idx + len(tool_tokens)))

            text = None
            target_id = None
            distractor_id = None

        if args.task == "tool_selection":
            input_ids = torch.tensor([full_ids], device=model.device)
        else:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        if args.task == "tool_selection":
            scores, rel_val = get_lrp_scores(
                model, input_ids, target_id, distractor_id,
                positions=positions, token_ids=tool_tokens,
                lrp_mode=args.lrp_mode, epsilon=args.lrp_epsilon,
            )
        else:
            scores, rel_val = get_lrp_scores(
                model, input_ids, target_id, distractor_id,
                lrp_mode=args.lrp_mode, epsilon=args.lrp_epsilon,
            )
        
        # Normalization factor: LRP scores sum to the relevance signal. 
        # Normalize by abs(rel_val) to make scores comparable across samples.
        norm_factor = abs(rel_val) if abs(rel_val) > 1e-12 else 1.0

        for k, v in scores.items():
            if isinstance(v, list):
                # Normalize list (neurons)
                v_norm = [x / norm_factor for x in v]
                if k not in aggregated_scores:
                    aggregated_scores[k] = list(v_norm)
                else:
                    aggregated_scores[k] = [a + b for a, b in zip(aggregated_scores[k], v_norm)]
            else:
                # Normalize scalar (heads)
                v_norm = v / norm_factor
                aggregated_scores[k] = aggregated_scores.get(k, 0.0) + v_norm

        # Find max head among head-level scores only (exclude neuron/weight lists)
        head_keys = [k for k in scores.keys() if "_H" in k]
        if head_keys:
            max_head = max(head_keys, key=lambda k: scores[k])
            print(f"Sample {i+1}: Top Attn Head = {max_head} (score: {scores[max_head]:.4f})")
        else:
            print(f"Sample {i+1}: No attention head scores found.")
        n_used += 1

    # Average accumulated scores
    if n_used == 0:
        print("No samples were processed successfully; skipping aggregation.")
        log_experiment(
            task=args.task,
            method="LRP",
            config=vars(args),
            metrics={"status": "no_samples", "samples": 0},
            base_dir=str(REPO_ROOT / "results")
        )
        return None

    for k in aggregated_scores:
        if isinstance(aggregated_scores[k], list):
            aggregated_scores[k] = [v / n_used for v in aggregated_scores[k]]
        else:
            aggregated_scores[k] = aggregated_scores[k] / n_used

    head_scores = {k: v for k, v in aggregated_scores.items() if "_H" in k}
    top_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n=== LRP TOP 5 GLOBAL ATTENTION HEADS ===")
    for head, score in top_heads:
        print(f"  {head}: {score:.4f}")

    # Save circuit
    circuits_dir = REPO_ROOT / "results" / "circuits"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{args.seed}" if args.seeds else ""
    circuit_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_lrp_circuit{seed_tag}.json"
    
    # Filter out Tensors (weights) to prevent JSON serialization crash
    json_scores = {k: v for k, v in aggregated_scores.items() if not k.endswith("_weights")}
    
    with open(circuit_file, "w") as f:
        json.dump(json_scores, f)
    print(f"Saved LRP circuit to {circuit_file}")

    log_experiment(
        task=args.task,
        method="LRP",
        config={**vars(args), "circuit_file": str(circuit_file)},
        metrics={"status": "success", "samples": n_used},
        base_dir=str(REPO_ROOT / "results")
    )
    return aggregated_scores


def run_lrp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Model name/path. Defaults to gemma-3-270m-it for tool_selection, else gemma-3-270m.")
    parser.add_argument("--task", type=str, choices=["ioi", "arithmetic", "tool_selection"], default="ioi")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed seed used only for train/val/test splitting")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-run discovery")
    parser.add_argument(
        "--lrp_mode",
        type=str,
        choices=["lrp_eps", "input_x_grad"],
        default="lrp_eps",
        help=(
            "LRP implementation to use. "
            "'lrp_eps' (default): true layer-wise LRP-ε with conservation checking. "
            "'input_x_grad': legacy backward-hook Input×Gradient approximation "
            "(retained for ablation comparison)."
        )
    )
    parser.add_argument(
        "--lrp_epsilon",
        type=float,
        default=1e-9,
        help="Numerical stabiliser ε in the LRP-ε denominator (default 1e-9)."
    )
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
            mean_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_lrp_circuit_seedmean.json"
            std_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_lrp_circuit_seedstd.json"
            with open(mean_file, "w") as f:
                json.dump(mean_scores, f)
            with open(std_file, "w") as f:
                json.dump(std_scores, f)
            # Also write a default LRP circuit file for downstream tooling.
            base_file = circuits_dir / f"{args.model.replace('/', '_')}_{args.task}_lrp_circuit.json"
            with open(base_file, "w") as f:
                json.dump(mean_scores, f)
            print(f"Saved mean circuit to {mean_file}")
            print(f"Saved std circuit to {std_file}")
            print(f"Saved mean circuit alias to {base_file}")
    else:
        run_once(args)


if __name__ == "__main__":
    run_lrp()
