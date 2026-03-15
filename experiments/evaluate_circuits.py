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

from evaluation.metrics import compute_logit_diff, compute_prob_diff, compute_normalized_faithfulness, compute_kl_divergence
from utils.logging import log_experiment, set_seed


class CircuitAblator:
    def __init__(self, model, circuit_scores, keep_ratio=0.1):
        self.model = model
        self.circuit_scores = circuit_scores
        self.keep_ratio = keep_ratio
        self.hooks = []

        # Config-driven — works for Gemma 3 and any well-formed HF model config
        if hasattr(model.config, "num_attention_heads"):
            self.num_heads = model.config.num_attention_heads
            if hasattr(model.config, "head_dim"):
                self.head_dim = model.config.head_dim
            elif hasattr(model.config, "hidden_size"):
                self.head_dim = model.config.hidden_size // self.num_heads
            else:
                raise NotImplementedError("Cannot determine head_dim from config.")
        else:
            raise NotImplementedError("Model config does not expose num_attention_heads.")

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

        for layer_idx in range(len(self.model.model.layers)):
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
                int_size = getattr(self.model.config, "intermediate_size", 2048)
                mlp_mask = torch.zeros(int_size, dtype=torch.bool)

            self.mlp_masks[layer_idx] = mlp_mask

    def apply_hooks(self):
        """Register forward pre-hooks to zero-ablate non-circuit nodes."""
        self.remove_hooks()
        for i, layer in enumerate(self.model.model.layers):
            def make_attn_pre_hook(mask):
                def pre_hook(module, args):
                    x = args[0]
                    batch, seq, hidden = x.shape
                    x_reshaped = x.view(batch, seq, self.num_heads, self.head_dim)
                    mask_dev = mask.to(x.device).view(1, 1, self.num_heads, 1)
                    x_ablated = (x_reshaped * mask_dev).view(batch, seq, hidden)
                    return (x_ablated,)
                return pre_hook

            def make_mlp_pre_hook(mask):
                def pre_hook(module, args):
                    x = args[0]
                    mask_dev = mask.to(x.device).view(1, 1, -1)
                    x_ablated = x * mask_dev
                    return (x_ablated,)
                return pre_hook

            h1 = layer.self_attn.o_proj.register_forward_pre_hook(make_attn_pre_hook(self.attn_masks[i]))
            self.hooks.append(h1)

            h2 = layer.mlp.down_proj.register_forward_pre_hook(make_mlp_pre_hook(self.mlp_masks[i]))
            self.hooks.append(h2)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["ioi", "greater_than", "tool_selection"], default="ioi")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name/path. Defaults to gemma-3-270m-it for tool_selection, else gemma-3-270m.")
    parser.add_argument("--keep_ratio", type=float, default=0.1, help="Fraction of nodes to keep (0.0 to 1.0)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Determine model — can be overridden via --model for cross-model evaluation
    if args.model is None:
        args.model = "google/gemma-3-270m-it" if args.task == "tool_selection" else "google/gemma-3-270m"

    model_name = args.model
    circuit_file = REPO_ROOT / "results" / "circuits" / f"{model_name.replace('/', '_')}_{args.task}_circuit.json"

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
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)[:args.n_samples]

    # Pre-calculate year token IDs for Greater-Than (00-99)
    year_token_ids = {}
    if args.task == "greater_than":
        for i in range(100):
            year_str = f"{i:02d}"
            # Test both with and without space (Gemma tokenizer varies)
            token_id = tokenizer.encode(year_str, add_special_tokens=False)[-1]
            year_token_ids[i] = token_id

    ablator = CircuitAblator(model, circuit_scores, keep_ratio=args.keep_ratio)

    results = {
        "m_ld": [],          # Ablated Clean Logit Diff
        "b_ld": [],          # Full Clean Logit Diff
        "b_prime_ld": [],    # Full Corrupted Logit Diff
        "m_pd": [],          # Ablated Clean Prob Diff
        "b_pd": [],          # Full Clean Prob Diff
        "b_prime_pd": [],    # Full Corrupted Prob Diff
        "kl": []
    }

    print(f"Evaluating circuit on {args.task} (keep_ratio={args.keep_ratio})...")

    for i, sample in enumerate(dataset):
        # 1. Prepare inputs and targets
        if args.task == "ioi":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            target_ids = [tokenizer.encode(" " + sample["target_clean"], add_special_tokens=False)[-1]]
            distractor_ids = [tokenizer.encode(" " + sample["target_corrupted"], add_special_tokens=False)[-1]]
        elif args.task == "greater_than":
            clean_text = sample["clean_prompt"]
            corrupted_text = sample["corrupted_prompt"]
            y1 = sample["year1_suffix"]
            # Correct tokens: years > y1. Incorrect tokens: years <= y1.
            target_ids = [year_token_ids[y] for y in range(y1 + 1, 100)]
            distractor_ids = [year_token_ids[y] for y in range(0, y1 + 1)]
        elif args.task == "tool_selection":
            tools = sample["tools"]
            clean_text = tokenizer.apply_chat_template(
                sample["clean_messages"], tools=tools, tokenize=False, add_generation_prompt=True
            )
            corrupted_text = tokenizer.apply_chat_template(
                sample["corrupted_messages"], tools=tools, tokenize=False, add_generation_prompt=True
            )
            target_ids = [tokenizer.encode(sample["target_clean_tool"], add_special_tokens=False)[-1]]
            distractor_ids = [tokenizer.encode(sample["target_corrupted_tool"], add_special_tokens=False)[-1]]

        clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(model.device)
        corrupted_ids = tokenizer(corrupted_text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            # 2. Full Clean Run (b)
            ablator.remove_hooks()
            clean_full_logits = model(input_ids=clean_ids).logits
            b_ld = compute_logit_diff(clean_full_logits, target_ids[0], distractor_ids[0]).item()
            b_pd = compute_prob_diff(clean_full_logits, target_ids, distractor_ids).item()

            # 3. Full Corrupted Run (b')
            corrupted_full_logits = model(input_ids=corrupted_ids).logits
            b_prime_ld = compute_logit_diff(corrupted_full_logits, target_ids[0], distractor_ids[0]).item()
            b_prime_pd = compute_prob_diff(corrupted_full_logits, target_ids, distractor_ids).item()

            # 4. Ablated Clean Run (m)
            ablator.apply_hooks()
            ablated_logits = model(input_ids=clean_ids).logits
            m_ld = compute_logit_diff(ablated_logits, target_ids[0], distractor_ids[0]).item()
            m_pd = compute_prob_diff(ablated_logits, target_ids, distractor_ids).item()
            
            kl = compute_kl_divergence(clean_full_logits, ablated_logits)

        results["b_ld"].append(b_ld)
        results["b_prime_ld"].append(b_prime_ld)
        results["m_ld"].append(m_ld)
        results["b_pd"].append(b_pd)
        results["b_prime_pd"].append(b_prime_pd)
        results["m_pd"].append(m_pd)
        results["kl"].append(kl)

    ablator.remove_hooks()

    # Final summary metrics
    mean_b_ld = np.mean(results["b_ld"])
    mean_b_prime_ld = np.mean(results["b_prime_ld"])
    mean_m_ld = np.mean(results["m_ld"])
    
    mean_b_pd = np.mean(results["b_pd"])
    mean_b_prime_pd = np.mean(results["b_prime_pd"])
    mean_m_pd = np.mean(results["m_pd"])

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
        "kl_divergence": mean_kl
    }

    print("\n=== Evaluation Results (Hanna et al., 2024 Metrics) ===")
    for k, v in summary_metrics.items():
        print(f"  {k:25}: {v:.4f}")

    log_experiment(
        task=args.task,
        method="EAP-IG_Eval",
        config={**vars(args), "circuit_file": str(circuit_file)},
        metrics=summary_metrics,
        base_dir=str(REPO_ROOT / "results")
    )


if __name__ == "__main__":
    main()
