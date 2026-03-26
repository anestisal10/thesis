import os
import json
import torch
import argparse
import sys
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Robust absolute path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compression.pruning_core import MagnitudePruner
from evaluation.metrics import compute_logit_diff, compute_prob_diff, compute_normalized_faithfulness, compute_kl_divergence, compute_task_accuracy, compute_perplexity
from utils.aggregation import summarize_multi_seed_metrics
from utils.data import load_dataset_split
from utils.logging import log_experiment, set_seed
from utils.tool_selection import build_tool_call_message, find_subsequence, tool_call_logprob
from utils.tokenization import get_single_token_id

def load_circuit_union(model_name, task):
    """Loads EAP-IG and LRP circuits and returns a dict representing the union of their top 10% nodes."""
    eap_file = REPO_ROOT / "results" / "circuits" / f"{model_name.replace('/', '_')}_{task}_circuit.json"
    lrp_file = REPO_ROOT / "results" / "circuits" / f"{model_name.replace('/', '_')}_{task}_lrp_circuit.json"
    
    if not (eap_file.exists() and lrp_file.exists()):
         raise FileNotFoundError(f"Missing circuit files for {model_name} on {task}. Run EAP-IG and LRP discovery first.")
         
    with open(eap_file, "r") as f:
         eap_scores = json.load(f)
    with open(lrp_file, "r") as f:
         lrp_scores = json.load(f)
         
    def normalize_key(k):
        # LRP key example: model.model.layers.5.self_attn.o_proj_H7 -> L5_attn_H7
        if "layers." in k:
            import re
            m = re.search(r"layers\.(\d+)\.", k)
            if m:
                i = m.group(1)
                if "attn" in k:
                    h = re.search(r"_H(\d+)$", k)
                    if h: return f"L{i}_attn_H{h.group(1)}"
                if "mlp" in k:
                    n = re.search(r"_neurons_(\d+)$", k)
                    if n: return f"L{i}_mlp_neurons_{n.group(1)}"
                    if k.endswith("_neurons"):
                        return f"L{i}_mlp_neurons"
        return k

    def get_top_10_percent_keys(scores_dict):
        all_vals = []
        for k, v in scores_dict.items():
            if isinstance(v, list):
                all_vals.extend([abs(x) for x in v])
            else:
                all_vals.append(abs(v))
        if not all_vals: return set()
        
        sorted_vals = sorted(all_vals, reverse=True)
        k_idx = max(0, int(len(sorted_vals) * 0.1) - 1)
        threshold = sorted_vals[k_idx]
        
        top_keys = set()
        for k, v in scores_dict.items():
            k_norm = normalize_key(k)
            if isinstance(v, list):
                for i, score in enumerate(v):
                    if abs(score) >= threshold:
                        top_keys.add(f"{k_norm}_{i}")
            else:
                if abs(v) >= threshold:
                    top_keys.add(k_norm)
        return top_keys
        
    eap_keys = get_top_10_percent_keys(eap_scores)
    lrp_keys = get_top_10_percent_keys(lrp_scores)
    
    union_keys = eap_keys.union(lrp_keys)
    print(f"EAP-IG top 10% nodes: {len(eap_keys)}")
    print(f"LRP top 10% nodes: {len(lrp_keys)}")
    print(f"Union (Circuit-Locked Protected Nodes): {len(union_keys)}")
    
    return {k: 1.0 for k in union_keys}




def run_once(args):
    set_seed(args.seed)

    if args.model is None:
        args.model = "google/gemma-3-270m-it" if args.task == "tool_selection" else "google/gemma-3-270m"

    keep_ratio = 1.0 - args.sparsity
    print(f"Target Sparsity: {args.sparsity*100:.1f}% -> Keep Ratio: {keep_ratio*100:.1f}%")

    # Determine circuit scores if circuit_locked
    circuit_scores = None
    if args.strategy == "circuit_locked":
        circuit_scores = load_circuit_union(args.model, args.task)

    print(f"Loading tokenizer and model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    year_token_ids = {}
    if args.task == "greater_than":
        for i in range(100):
            year_str = f"{i:02d}"
            year_token_ids[i] = tokenizer.encode(year_str, add_special_tokens=False)[-1]

    # Apply Pruning Engine
    print(f"Initializing MagnitudePruner with strategy={args.strategy}...")
    pruner = MagnitudePruner(model, keep_ratio=keep_ratio, circuit_scores=circuit_scores)
    
    dataset_path = REPO_ROOT / "data" / args.task / "dataset.json"
    task_data = load_dataset_split(
        dataset_path,
        split=args.split,
        n_samples=args.n_samples,
        split_seed=args.split_seed,
    )

    clean_ld_list, corrupted_ld_list, pruned_ld_list, acc_list, kl_div_list = [], [], [], [], []

    print("Running Task Evaluation...")
    # Baseline Clean and Corrupted Runs Collection for Faithfulness
    pruner.remove_hooks()
    
    input_ids_cache = []
    target_clean_ids = []
    target_corrupted_ids = []
    
    # For Greater-Than multi-answer
    gt_target_years = []
    gt_distractor_years = []

    for sample in task_data:
        if args.task == "ioi":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            try:
                tc = get_single_token_id(tokenizer, sample["target_clean"], prefer_space=True)
                td = get_single_token_id(tokenizer, sample["target_corrupted"], prefer_space=True)
            except ValueError:
                print("Skipping IOI sample: target/distractor not single-token.")
                continue
            target_clean_ids.append(tc)
            target_corrupted_ids.append(td)
        elif args.task == "greater_than":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            # Multi-answer logic
            y1 = sample["year1_suffix"]
            # Targets: years > y1. Model predicts the 2-digit suffix next.
            target_years = [year_token_ids[i] for i in range(y1 + 1, 100)]
            distractor_years = [year_token_ids[i] for i in range(0, y1 + 1)]
            gt_target_years.append(target_years)
            gt_distractor_years.append(distractor_years)
            # For LD baseline
            tc = year_token_ids[sample["target_clean_suffix"]]
            td = year_token_ids[sample["target_corrupted_suffix"]]
            target_clean_ids.append(tc)
            target_corrupted_ids.append(td)
        elif args.task == "tool_selection":
            tools = sample["tools"]
            clean_text = None
            corrupted_text = None
            tc = None
            td = None
            target_clean_ids.append(sample["target_clean_tool"])
            target_corrupted_ids.append(sample["target_corrupted_tool"])
            
        # Left-padding would be better for EAP-IG alignment, but for inference [batch=1] 
        # we can just use the raw tokens and index -1.
        if args.task != "tool_selection":
            clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(model.device)
            corr_ids = tokenizer(corrupted_text, return_tensors="pt").input_ids.to(model.device)
            
            with torch.no_grad():
                clean_logits = model(clean_ids).logits
                corr_logits = model(corr_ids).logits
                
                input_ids_cache.append((clean_ids, clean_logits.cpu()))
                
                if args.task == "greater_than":
                    clean_val = compute_prob_diff(clean_logits, target_years, distractor_years).item()
                    corr_val = compute_prob_diff(corr_logits, target_years, distractor_years).item()
                else:
                    clean_val = compute_logit_diff(clean_logits, tc, td).item()
                    corr_val = compute_logit_diff(corr_logits, tc, td).item()
                    
                clean_ld_list.append(clean_val)
                corrupted_ld_list.append(corr_val)
        else:
            input_ids_cache.append((sample, None))
            clean_val = tool_call_logprob(model, tokenizer, sample["clean_messages"], tools, sample["target_clean_tool"]) - \
                        tool_call_logprob(model, tokenizer, sample["clean_messages"], tools, sample["target_corrupted_tool"])
            corr_val = tool_call_logprob(model, tokenizer, sample["corrupted_messages"], tools, sample["target_corrupted_tool"]) - \
                       tool_call_logprob(model, tokenizer, sample["corrupted_messages"], tools, sample["target_clean_tool"])
            clean_ld_list.append(clean_val)
            corrupted_ld_list.append(corr_val)
            
    if not input_ids_cache:
        print("No samples were processed successfully; skipping pruning evaluation.")
        log_experiment(
            task=args.task,
            method=f"Pruning_{args.strategy}",
            config=vars(args),
            metrics={"status": "no_samples"},
            base_dir=str(REPO_ROOT / "results")
        )
        return None

    # Pruned Run Check
    pruner.apply_hooks()
    for i, cached_data in enumerate(input_ids_cache):
        tc = target_clean_ids[i]
        td = target_corrupted_ids[i]
        
        if args.task != "tool_selection":
            input_ids, clean_logits_cpu = cached_data
            with torch.no_grad():
                pruned_logits = model(input_ids).logits
            kl = compute_kl_divergence(clean_logits_cpu.to(model.device), pruned_logits)
            kl_div_list.append(kl)
                
            if args.task == "greater_than":
                target_years = gt_target_years[i]
                distractor_years = gt_distractor_years[i]
                pruned_val = compute_prob_diff(pruned_logits, target_years, distractor_years).item()
                target_tensor = torch.tensor(target_years, device=model.device).unsqueeze(0)
                acc = compute_task_accuracy(pruned_logits, target_tensor)
            else:
                pruned_val = compute_logit_diff(pruned_logits, tc, td).item()
                target_tensor = torch.tensor([tc], device=model.device).unsqueeze(0) if isinstance(tc, int) else torch.tensor(tc, device=model.device).unsqueeze(0)
                acc = compute_task_accuracy(pruned_logits, target_tensor)
        else:
            sample, _ = cached_data
            tools = sample["tools"]
            pruned_val = tool_call_logprob(model, tokenizer, sample["clean_messages"], tools, tc) - \
                         tool_call_logprob(model, tokenizer, sample["clean_messages"], tools, td)
            tool_names = [t["function"]["name"] for t in tools]
            scores = {name: tool_call_logprob(model, tokenizer, sample["clean_messages"], tools, name) for name in tool_names}
            pred = max(scores.items(), key=lambda kv: kv[1])[0]
            acc = 1.0 if pred == tc else 0.0
            
        pruned_ld_list.append(pruned_val)
        acc_list.append(acc)

    # ── Overall task metrics ───────────────────────────────────────────────
    m = sum(pruned_ld_list) / len(pruned_ld_list)
    b = sum(clean_ld_list) / len(clean_ld_list)
    b_prime = sum(corrupted_ld_list) / len(corrupted_ld_list)

    normalized_faithfulness = compute_normalized_faithfulness(m, b, b_prime)
    mean_acc = sum(acc_list) / len(acc_list)
    mean_kl = sum(kl_div_list) / len(kl_div_list) if kl_div_list else 0.0

    # ── Perplexity: UNPRUNED baseline (hooks temporarily removed) ─────────
    # Remove hooks so we measure the unmodified, un-pruned model. This is the
    # perplexity ceiling — any increase in perplexity_pruned is attributed to
    # structural degradation introduced by this sparsity level.
    print("\nComputing UNPRUNED baseline perplexity on WikiText-2 (hooks off)...")
    pruner.remove_hooks()
    try:
        wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
        wiki_texts = [t for t in wiki_data["text"] if len(t.strip()) > 50]
        perplexity_unpruned = compute_perplexity(model, tokenizer, wiki_texts, max_samples=200)
    except Exception as e:
        print(f"Error computing unpruned perplexity: {e}")
        perplexity_unpruned = float("inf")
        wiki_texts = []

    # ── Perplexity: PRUNED model (hooks re-applied) ───────────────────────
    # Re-apply pruning hooks so perplexity is measured on the structurally
    # degraded model. The delta (pruned - unpruned) quantifies catastrophic
    # forgetting on general language at this sparsity level.
    print("Computing PRUNED model perplexity on WikiText-2 (hooks active = pruned model)...")
    pruner.apply_hooks()
    try:
        if not wiki_texts:
            wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
            wiki_texts = [t for t in wiki_data["text"] if len(t.strip()) > 50]
        perplexity_pruned = compute_perplexity(model, tokenizer, wiki_texts, max_samples=200)
    except Exception as e:
        print(f"Error computing pruned perplexity: {e}")
        perplexity_pruned = float("inf")

    pruner.remove_hooks()

    metrics = {
        "strategy": args.strategy,
        "sparsity": args.sparsity,
        "task_accuracy": mean_acc,
        "kl_divergence": mean_kl,
        "normalized_faithfulness": normalized_faithfulness,
        # perplexity_unpruned: base model, no structural changes.
        # perplexity_pruned:   same model with pruning masks active.
        # Delta = forgetting cost of this sparsity level.
        "perplexity_unpruned": perplexity_unpruned,
        "perplexity_pruned": perplexity_pruned,
        "clean_baseline": b,
        "corrupted_baseline": b_prime,
        "pruned_metric": m
    }

    print("\n=== Pruning Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    log_experiment(
        task=args.task,
        method=f"Pruning_{args.strategy}",
        config=vars(args),
        metrics=metrics,
        base_dir=str(REPO_ROOT / "results")
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["ioi", "greater_than", "tool_selection"], required=True)
    parser.add_argument("--model", type=str, default=None, help="E.g., google/gemma-3-270m-it")
    parser.add_argument("--strategy", type=str, choices=["magnitude", "circuit_locked"], required=True)
    parser.add_argument("--sparsity", type=float, required=True, help="Fraction of nodes to REMOVE (e.g. 0.4 for 40% stripped, keeping 60%)")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed seed used only for train/val/test splitting")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-run pruning")
    args = parser.parse_args()

    if getattr(args, "seeds", None):
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        all_metrics = []
        for s in seeds:
            setattr(args, "seed", s)
            metrics = run_once(args)
            if metrics is None:
                print(f"[WARN] Skipping seed {s}: no successful samples.")
                continue
            all_metrics.append(metrics)
        if not all_metrics:
            print("[WARN] No successful pruning runs; skipping aggregation.")
            return
        agg = summarize_multi_seed_metrics(all_metrics, seeds)
        log_experiment(
            task=args.task,
            method=f"Pruning_{args.strategy}_MultiSeed",
            config=vars(args),
            metrics=agg,
            base_dir=str(REPO_ROOT / "results")
        )
        print("\n=== Multi-Seed Summary ===")
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        run_once(args)


if __name__ == "__main__":
    main()
