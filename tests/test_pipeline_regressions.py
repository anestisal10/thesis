import json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.compare_circuits import get_top_k_nodes, jaccard_similarity
from experiments.evaluate_circuits import get_circuit_file
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.aggregation import summarize_multi_seed_metrics
from utils.data import load_dataset_split


def test_load_dataset_split_uses_split_seed():
    dataset_path = REPO_ROOT / ".agent" / f"test_dataset_{uuid.uuid4().hex}.json"
    dataset = [{"id": i} for i in range(20)]

    try:
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        split_a = load_dataset_split(dataset_path, split="test", n_samples=5, seed=1, split_seed=123)
        split_b = load_dataset_split(dataset_path, split="test", n_samples=5, seed=999, split_seed=123)

        assert [item["id"] for item in split_a] == [item["id"] for item in split_b]
    finally:
        if dataset_path.exists():
            dataset_path.unlink()


def test_compare_circuits_normalizes_attention_keys():
    eap_circuit = {"L0_attn_H1": 1.0}
    lrp_circuit = {"model.model.layers.0.self_attn.o_proj_H1": 2.0}

    eap_heads = get_top_k_nodes(eap_circuit, k=1, node_type="attn")
    lrp_heads = get_top_k_nodes(lrp_circuit, k=1, node_type="attn")

    assert eap_heads == {"L0_attn_H1"}
    assert lrp_heads == {"L0_attn_H1"}
    assert jaccard_similarity(eap_heads, lrp_heads) == 1.0


def test_summarize_multi_seed_metrics_keeps_string_fields():
    metrics = [
        {"strategy": "magnitude", "task_accuracy": 0.5, "kl_divergence": 1.0},
        {"strategy": "magnitude", "task_accuracy": 0.75, "kl_divergence": 3.0},
    ]

    agg = summarize_multi_seed_metrics(metrics, seeds=[1, 2])

    assert agg["strategy"] == "magnitude"
    assert agg["task_accuracy_mean"] == 0.625
    assert agg["kl_divergence_mean"] == 2.0


def test_get_circuit_file_selects_method_suffix():
    eap_path = get_circuit_file("google/gemma-3-270m", "ioi", "eap_ig")
    lrp_path = get_circuit_file("google/gemma-3-270m", "ioi", "lrp")

    assert eap_path.name == "google_gemma-3-270m_ioi_circuit.json"
    assert lrp_path.name == "google_gemma-3-270m_ioi_lrp_circuit.json"
