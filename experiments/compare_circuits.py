"""
Script to compare EAP-IG vs LRP circuits using Jaccard Similarity.
Reads both circuit JSON files and computes the intersection / union of top-K components.
"""
import json
import argparse
import sys
from pathlib import Path

# Robust absolute path resolution — works from any working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.logging import log_experiment


def load_circuit(path):
    """Load a circuit from a JSON file."""
    with open(path) as f:
        return json.load(f)


def get_top_k_nodes(circuit, k=10, node_type="attn"):
    """Return a set of the top-K attention heads or MLP layers by absolute score."""
    if node_type == "attn":
        scores = {key: abs(v) for key, v in circuit.items() if "attn" in key and isinstance(v, (int, float))}
    else:
        scores = {key: sum(abs(x) for x in v) for key, v in circuit.items() if "mlp" in key and isinstance(v, list)}

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return set(n for n, _ in sorted_nodes[:k])


def jaccard_similarity(set_a, set_b):
    """Intersection over Union."""
    if len(set_a | set_b) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_circuits():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["ioi", "greater_than", "tool_selection"], default="ioi")
    parser.add_argument("--top_k", type=int, default=10, help="Top K components to compare")
    parser.add_argument("--model", type=str, default=None, help="Model name/path. Defaults to gemma-3-270m-it for tool_selection, else gemma-3-270m.")
    args = parser.parse_args()

    if args.model is None:
        args.model = "google/gemma-3-270m-it" if args.task == "tool_selection" else "google/gemma-3-270m"
    model_name = args.model
    model_slug = model_name.replace("/", "_")

    circuits_dir = REPO_ROOT / "results" / "circuits"
    eap_path = circuits_dir / f"{model_slug}_{args.task}_circuit.json"
    lrp_path = circuits_dir / f"{model_slug}_{args.task}_lrp_circuit.json"

    if not eap_path.exists() or not lrp_path.exists():
        print(f"Missing circuit files. Expected:\n  {eap_path}\n  {lrp_path}")
        return

    eap_circuit = load_circuit(eap_path)
    lrp_circuit = load_circuit(lrp_path)

    print(f"\n{'='*50}")
    print(f"Circuit Comparison: {args.task.upper()} (Top-{args.top_k} nodes)")
    print(f"{'='*50}")

    eap_heads = get_top_k_nodes(eap_circuit, k=args.top_k, node_type="attn")
    lrp_heads = get_top_k_nodes(lrp_circuit, k=args.top_k, node_type="attn")

    j_attn = jaccard_similarity(eap_heads, lrp_heads)

    print(f"\nTop-{args.top_k} Attention Heads")
    print(f"  EAP-IG: {sorted(eap_heads)}")
    print(f"  LRP:    {sorted(lrp_heads)}")
    print(f"  Shared: {sorted(eap_heads & lrp_heads)}")
    print(f"  Jaccard Similarity (Attn): {j_attn:.4f}")

    eap_mlp = get_top_k_nodes(eap_circuit, k=args.top_k, node_type="mlp")
    lrp_mlp = get_top_k_nodes(lrp_circuit, k=args.top_k, node_type="mlp")
    j_mlp = jaccard_similarity(eap_mlp, lrp_mlp)

    print(f"\nTop-{args.top_k} MLP Layers")
    print(f"  EAP-IG: {sorted(eap_mlp)}")
    print(f"  LRP:    {sorted(lrp_mlp)}")
    print(f"  Shared: {sorted(eap_mlp & lrp_mlp)}")
    print(f"  Jaccard Similarity (MLP): {j_mlp:.4f}")

    log_experiment(
        task=args.task,
        method="Circuit_Comparison",
        config=vars(args),
        metrics={
            "jaccard_attn": j_attn,
            "jaccard_mlp": j_mlp,
            "eap_ig_top_heads": sorted(eap_heads),
            "lrp_top_heads": sorted(lrp_heads),
            "shared_heads": sorted(eap_heads & lrp_heads)
        },
        base_dir=str(REPO_ROOT / "results")
    )


if __name__ == "__main__":
    compare_circuits()
