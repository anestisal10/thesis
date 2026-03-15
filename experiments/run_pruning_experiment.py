import os
import json
import torch
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Robust absolute path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compression.pruning_core import MagnitudePruner
from evaluation.metrics import compute_logit_diff, compute_faithfulness, compute_kl_divergence, compute_task_accuracy, compute_perplexity
from utils.logging import log_experiment, set_seed

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
            if isinstance(v, list):
                for i, score in enumerate(v):
                    if abs(score) >= threshold:
                        top_keys.add(f"{k}_{i}")
            else:
                if abs(v) >= threshold:
                    top_keys.add(k)
        return top_keys
        
    eap_keys = get_top_10_percent_keys(eap_scores)
    lrp_keys = get_top_10_percent_keys(lrp_scores)
    
    union_keys = eap_keys.union(lrp_keys)
    print(f"EAP-IG top 10% nodes: {len(eap_keys)}")
    print(f"LRP top 10% nodes: {len(lrp_keys)}")
    print(f"Union (Circuit-Locked Protected Nodes): {len(union_keys)}")
    
    # Return a dict where everything is 1.0 (so when MagnitudePruner does top-10% of this dict, threshold is 1.0)
    # But wait, MagnitudePruner expects the format containing lists for neurons if we pass it directly.
    # Actually, MagnitudePruner._parse_protected_nodes supports both direct keys and lists.
    # But I modified it to look for lists if the value is a list.
    # To be safe, let's just create a flat dict since MagnitudePruner checks:
    # "if abs(v) >= threshold: self.protected_nodes.add(k)"
    # We will pass a dict mapping union keys to 1.0.
    union_dict = {k: 1.0 for k in union_keys}
    return union_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["ioi", "greater_than", "tool_selection"], required=True)
    parser.add_argument("--model", type=str, default=None, help="E.g., google/gemma-3-270m-it")
    parser.add_argument("--strategy", type=str, choices=["magnitude", "circuit_locked"], required=True)
    parser.add_argument("--sparsity", type=float, required=True, help="Fraction of nodes to REMOVE (e.g. 0.4 for 40% stripped, keeping 60%)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    # Apply Pruning Engine
    print(f"Initializing MagnitudePruner with strategy={args.strategy}...")
    pruner = MagnitudePruner(model, keep_ratio=keep_ratio, circuit_scores=circuit_scores)
    
    dataset_path = REPO_ROOT / "data" / args.task / "dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)[:args.n_samples]

    clean_ld_list, ablated_ld_list, kl_div_list, acc_list = [], [], [], []

    print("Running Task Evaluation...")
    # Baseline Clean Run Collection for Faithfulness
    # We remove hooks for the clean run
    pruner.remove_hooks()
    
    clean_logits_cache = []
    target_clean_ids = []
    target_corrupted_ids = []
    input_ids_cache = []
    
    for sample in task_data:
        if args.task == "ioi":
            text = sample["clean_prompt"]
            tc = tokenizer.encode(" " + sample["target_clean"], add_special_tokens=False)[-1]
            td = tokenizer.encode(" " + sample["target_corrupted"], add_special_tokens=False)[-1]
        elif args.task == "greater_than":
            text = sample["clean_prompt"]
            tc = tokenizer.encode(str(sample["target_clean_suffix"]), add_special_tokens=False)[-1]
            td = tokenizer.encode(str(sample["target_corrupted_suffix"]), add_special_tokens=False)[-1]
        elif args.task == "tool_selection":
            tools = sample["tools"]
            text = tokenizer.apply_chat_template(
                sample["clean_messages"], tools=tools, tokenize=False, add_generation_prompt=True
            )
            tc = tokenizer.encode(sample["target_clean_tool"], add_special_tokens=False)[-1]
            td = tokenizer.encode(sample["target_corrupted_tool"], add_special_tokens=False)[-1]
            
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        input_ids_cache.append(input_ids)
        target_clean_ids.append(tc)
        target_corrupted_ids.append(td)
        
        with torch.no_grad():
            clean_logits_cache.append(model(input_ids).logits)
            
    # Pruned Run Check
    pruner.apply_hooks()
    for i, input_ids in enumerate(input_ids_cache):
        tc = target_clean_ids[i]
        td = target_corrupted_ids[i]
        clean_logits = clean_logits_cache[i]
        
        with torch.no_grad():
            pruned_logits = model(input_ids).logits
            
        clean_ld = compute_logit_diff(clean_logits, tc, td)
        pruned_ld = compute_logit_diff(pruned_logits, tc, td)
        kl = compute_kl_divergence(clean_logits, pruned_logits)
        
        # calculate accuracy
        tc_tensor = torch.tensor([tc], device=model.device)
        acc = compute_task_accuracy(pruned_logits, tc_tensor)
        
        clean_ld_list.append(clean_ld.item())
        ablated_ld_list.append(pruned_ld.item())
        kl_div_list.append(kl)
        acc_list.append(acc)

    # Calculate overall task metrics
    clean_ld_tensor = torch.tensor(clean_ld_list)
    ablated_ld_tensor = torch.tensor(ablated_ld_list)
    mean_faithfulness = compute_faithfulness(clean_ld_tensor, ablated_ld_tensor)
    mean_kl = sum(kl_div_list) / len(kl_div_list)
    mean_acc = sum(acc_list) / len(acc_list)
    
    print("\nRunning General Perplexity Evaluation on WikiText-2 (Test Set)...")
    try:
        wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # take only non-empty paragraphs
        wiki_texts = [t for t in wiki_data["text"] if len(t.strip()) > 50] 
        perplexity = compute_perplexity(model, tokenizer, wiki_texts, max_samples=40)
    except Exception as e:
        print(f"Error loading wikitext or calculating perplexity: {e}")
        perplexity = float("inf")
        
    pruner.remove_hooks()

    metrics = {
        "strategy": args.strategy,
        "sparsity": args.sparsity,
        "task_accuracy": mean_acc,
        "faithfulness": mean_faithfulness,
        "kl_divergence": mean_kl,
        "perplexity": perplexity,
        "clean_logit_diff": clean_ld_tensor.mean().item(),
        "ablated_logit_diff": ablated_ld_tensor.mean().item()
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

if __name__ == "__main__":
    main()
