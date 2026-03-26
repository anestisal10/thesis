import os
import json
import torch
import argparse
import numpy as np
import sys
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Robust absolute path resolution — works from any working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.metrics import compute_logit_diff, compute_prob_diff, compute_normalized_faithfulness, compute_kl_divergence, compute_task_accuracy
from utils.data import load_dataset_split
from utils.logging import log_experiment, set_seed
from utils.tool_selection import build_tool_call_message, find_subsequence, tool_call_logprob
from utils.tokenization import get_single_token_id


from compression.ablator import StructuralAblator

class CircuitAblator(StructuralAblator):
    def __init__(self, model, circuit_scores, keep_ratio=0.1):
        super().__init__(model)
        self.circuit_scores = circuit_scores
        self.keep_ratio = keep_ratio
        self._build_masks()

    def _build_masks(self):
        """Build per-layer boolean masks for attention heads and MLP neurons."""
        all_scores = []
        for k, v in self.circuit_scores.items():
            if isinstance(v, list):
                all_scores.extend([abs(x) for x in v])
            else:
                all_scores.append(abs(v))

        if not all_scores:
            self.threshold = 0.0
        else:
            sorted_scores = sorted(all_scores, reverse=True)
            k_idx = max(0, int(len(sorted_scores) * self.keep_ratio) - 1)
            self.threshold = sorted_scores[k_idx]

        print(f"Computed masking threshold: {self.threshold:.6f} for keep_ratio {self.keep_ratio:.2f}")

        self.attn_masks = {}
        self.mlp_masks = {}

        for layer_idx in range(self.num_layers):
            # Attention head mask: True = keep this head
            attn_mask = torch.zeros(self.num_heads, dtype=torch.bool)
            for h in range(self.num_heads):
                key = f"L{layer_idx}_attn_H{h}"
                if key in self.circuit_scores and abs(self.circuit_scores[key]) >= self.threshold:
                    attn_mask[h] = True
            self.attn_masks[layer_idx] = attn_mask

            # MLP neuron mask: True = keep this neuron
            mlp_key = f"L{layer_idx}_mlp_neurons"
            if mlp_key in self.circuit_scores:
                scores = np.array(self.circuit_scores[mlp_key])
                mlp_mask = torch.from_numpy(np.abs(scores) >= self.threshold)
            else:
                mlp_mask = torch.zeros(self.intermediate_size, dtype=torch.bool)

            self.mlp_masks[layer_idx] = mlp_mask





def get_circuit_file(model_name, task, method):
    suffix = "_lrp_circuit.json" if method == "lrp" else "_circuit.json"
    return REPO_ROOT / "results" / "circuits" / f"{model_name.replace('/', '_')}_{task}{suffix}"


def get_eval_method_label(method):
    return "LRP" if method == "lrp" else "EAP-IG"


def evaluate_once(args):
    set_seed(args.seed)

    # Determine model — can be overridden via --model for cross-model evaluation
    if args.model is None:
        args.model = "google/functiongemma-270m-it" if args.task == "tool_selection" else "google/gemma-3-270m"

    model_name = args.model
    circuit_file = get_circuit_file(model_name, args.task, args.method)

    if not circuit_file.exists():
        print(f"Circuit file not found: {circuit_file}\nRun run_circuit_discovery.py first.")
        return

    with open(circuit_file, "r") as f:
        circuit_scores = json.load(f)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    dataset_path = REPO_ROOT / "data" / args.task / "dataset.json"
    dataset = load_dataset_split(
        dataset_path,
        split=args.split,
        n_samples=args.n_samples,
        split_seed=args.split_seed,
    )

    # Removed greater_than pre-calculated token IDs as they are no longer needed

    ablator = CircuitAblator(model, circuit_scores, keep_ratio=args.keep_ratio)

    results = {
        "m_ld": [],          # Ablated Clean Logit Diff
        "b_ld": [],          # Full Clean Logit Diff
        "b_prime_ld": [],    # Full Corrupted Logit Diff
        "m_pd": [],          # Ablated Clean Prob Diff
        "b_pd": [],          # Full Clean Prob Diff
        "b_prime_pd": [],    # Full Corrupted Prob Diff
        "kl": [],
        "b_acc": [],         # Full Clean Task Accuracy
        "b_prime_acc": [],   # Full Corrupted Task Accuracy
        "m_acc": []          # Ablated Clean Task Accuracy
    }

    print(f"Evaluating circuit on {args.task} (keep_ratio={args.keep_ratio})...")

    for i, sample in enumerate(dataset):
        # ---------------------------------------------------------
        # 1. Prepare Inputs and Targets (Task-Specific Logic)
        # ---------------------------------------------------------
        if args.task == "ioi":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(model.device)
            corrupted_ids = tokenizer(corrupted_text, return_tensors="pt").input_ids.to(model.device)
            try:
                target_ids =[get_single_token_id(tokenizer, sample["target_clean"], prefer_space=True)]
                distractor_ids = [get_single_token_id(tokenizer, sample["target_corrupted"], prefer_space=True)]
            except ValueError:
                print(f"Sample {i+1} skipped: IOI targets are not single-token.")
                continue

        elif args.task == "arithmetic":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(model.device)
            corrupted_ids = tokenizer(corrupted_text, return_tensors="pt").input_ids.to(model.device)
            target_ids = [get_single_token_id(tokenizer, sample["target_clean_token"], prefer_space=False)]
            distractor_ids = [get_single_token_id(tokenizer, sample["target_corrupted_token"], prefer_space=False)]

        elif args.task == "tool_selection":
            tools = sample["tools_schema"]
            
            # Apply chat template
            clean_prompt_ids = tokenizer.apply_chat_template(
                sample["clean_messages"], tools=tools, tokenize=True, add_generation_prompt=True
            )
            corrupted_prompt_ids = tokenizer.apply_chat_template(
                sample["corrupted_messages"], tools=tools, tokenize=True, add_generation_prompt=True
            )
            
            # Append forced prefix
            prefix_ids = tokenizer.encode(sample["required_prefix"], add_special_tokens=False)
            clean_ids = torch.tensor([clean_prompt_ids + prefix_ids], device=model.device)
            corrupted_ids = torch.tensor([corrupted_prompt_ids + prefix_ids], device=model.device)
            
            # Target vs Corrupted Tool Token
            target_ids = [get_single_token_id(tokenizer, sample["clean_target_token"])]
            distractor_ids = [get_single_token_id(tokenizer, sample["corrupted_target_token"])]

        # ---------------------------------------------------------
        # 2. Universal Evaluation Block (Shared by ALL tasks)
        # ---------------------------------------------------------
        with torch.no_grad():
            # -- Full Clean Run (b) --
            ablator.remove_hooks()
            clean_full_logits = model(input_ids=clean_ids).logits
            b_ld = compute_logit_diff(clean_full_logits, target_ids, distractor_ids).item()
            b_pd = compute_prob_diff(clean_full_logits, target_ids, distractor_ids).item()
            b_acc = compute_task_accuracy(clean_full_logits, torch.tensor(target_ids, device=model.device))

            # -- Full Corrupted Run (b') --
            corrupted_full_logits = model(input_ids=corrupted_ids).logits
            b_prime_ld = compute_logit_diff(corrupted_full_logits, target_ids, distractor_ids).item()
            b_prime_pd = compute_prob_diff(corrupted_full_logits, target_ids, distractor_ids).item()
            b_prime_acc = compute_task_accuracy(corrupted_full_logits, torch.tensor(target_ids, device=model.device))

            # -- Ablated Clean Run (m) --
            ablator.apply_hooks()
            ablated_logits = model(input_ids=clean_ids).logits
            m_ld = compute_logit_diff(ablated_logits, target_ids, distractor_ids).item()
            m_pd = compute_prob_diff(ablated_logits, target_ids, distractor_ids).item()
            m_acc = compute_task_accuracy(ablated_logits, torch.tensor(target_ids, device=model.device))
            
            kl = compute_kl_divergence(clean_full_logits, ablated_logits)

        # ---------------------------------------------------------
        # 3. Store Results
        # ---------------------------------------------------------
        results["b_ld"].append(b_ld)
        results["b_prime_ld"].append(b_prime_ld)
        results["m_ld"].append(m_ld)
        results["b_pd"].append(b_pd)
        results["b_prime_pd"].append(b_prime_pd)
        results["m_pd"].append(m_pd)
        results["kl"].append(kl.item() if isinstance(kl, torch.Tensor) else kl)
        results["b_acc"].append(b_acc)
        results["b_prime_acc"].append(b_prime_acc)
        results["m_acc"].append(m_acc)

    ablator.remove_hooks()

    # Final summary metrics
    if not results["b_ld"]:
        print("No samples were processed successfully; skipping evaluation.")
        return None
    mean_b_ld = np.mean(results["b_ld"])
    mean_b_prime_ld = np.mean(results["b_prime_ld"])
    mean_m_ld = np.mean(results["m_ld"])
    
    mean_b_pd = np.mean(results["b_pd"])
    mean_b_prime_pd = np.mean(results["b_prime_pd"])
    mean_m_pd = np.mean(results["m_pd"])

    mean_b_acc = np.mean(results["b_acc"])
    mean_b_prime_acc = np.mean(results["b_prime_acc"])
    mean_m_acc = np.mean(results["m_acc"])

    faithfulness_ld = compute_normalized_faithfulness(mean_m_ld, mean_b_ld, mean_b_prime_ld)
    faithfulness_pd = compute_normalized_faithfulness(mean_m_pd, mean_b_pd, mean_b_prime_pd)
    mean_kl = np.mean(results["kl"])

    summary_metrics = {
        "keep_ratio": args.keep_ratio,
        "clean_baseline_ld": mean_b_ld,
        "corrupted_baseline_ld": mean_b_prime_ld,
        "ablated_ld": mean_m_ld,
        "faithfulness_ld": faithfulness_ld,
        "clean_baseline_pd": mean_b_pd,
        "corrupted_baseline_pd": mean_b_prime_pd,
        "ablated_pd": mean_m_pd,
        "faithfulness_pd": faithfulness_pd,
        "kl_divergence": mean_kl,
        "clean_baseline_acc": mean_b_acc,
        "corrupted_baseline_acc": mean_b_prime_acc,
        "ablated_acc": mean_m_acc,
    }

    print("\n=== Evaluation Results (Hanna et al., 2024 Metrics) ===")
    for k, v in summary_metrics.items():
        print(f"  {k:25}: {v:.4f}")

    return summary_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["ioi", "arithmetic", "tool_selection"], default="ioi")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name/path. Defaults to gemma-3-270m-it for tool_selection, else gemma-3-270m.")
    parser.add_argument("--method", type=str, choices=["eap_ig", "lrp"], default="eap_ig",
                        help="Which discovered circuit family to evaluate.")
    parser.add_argument("--keep-ratios", type=str, default="0.1", help="Comma-separated fraction of nodes to keep (0.0 to 1.0)")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed seed used only for train/val/test splitting")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-run evaluation")
    args = parser.parse_args()
    method_label = get_eval_method_label(args.method)

    keep_ratios = [float(r.strip()) for r in args.keep_ratios.split(",")]

    for kr in keep_ratios:
        args.keep_ratio = kr
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            all_metrics = []
            for s in seeds:
                args.seed = s
                metrics = evaluate_once(args)
                if metrics is None:
                    print(f"[WARN] Skipping seed {s}: circuit file missing or evaluation failed.")
                    continue
                all_metrics.append(metrics)
                circuit_file = get_circuit_file(args.model, args.task, args.method)
                log_experiment(
                    task=args.task,
                    method=f"{method_label}_Eval",
                    config={**vars(args), "circuit_file": str(circuit_file)},
                    metrics=metrics,
                    base_dir=str(REPO_ROOT / "results")
                )
            if not all_metrics:
                print(f"[WARN] No successful evaluations for keep_ratio {kr}; skipping aggregation.")
                continue
            agg = {"seeds": seeds}
            for k in all_metrics[0].keys():
                vals = np.array([m[k] for m in all_metrics], dtype=float)
                agg[f"{k}_mean"] = float(vals.mean())
                agg[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                agg[f"{k}_ci95"] = float(1.96 * agg[f"{k}_std"] / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            log_experiment(
                task=args.task,
                method=f"{method_label}_Eval_MultiSeed",
                config={**vars(args), "circuit_file": str(get_circuit_file(args.model, args.task, args.method))},
                metrics=agg,
                base_dir=str(REPO_ROOT / "results")
            )
            print(f"\n=== Multi-Seed Summary (keep_ratio={kr}) ===")
            for k, v in agg.items():
                if isinstance(v, float):
                    print(f"  {k:25}: {v:.4f}")
                else:
                    print(f"  {k:25}: {v}")
        else:
            metrics = evaluate_once(args)
            circuit_file = get_circuit_file(args.model, args.task, args.method)
            log_experiment(
                task=args.task,
                method=f"{method_label}_Eval",
                config={**vars(args), "circuit_file": str(circuit_file)},
                metrics=metrics,
                base_dir=str(REPO_ROOT / "results")
            )


if __name__ == "__main__":
    main()
