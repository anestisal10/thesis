import numpy as np


def summarize_multi_seed_metrics(all_metrics, seeds):
    agg = {"seeds": seeds}
    sample_metrics = all_metrics[0]
    for k, v in sample_metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            vals = np.array([m[k] for m in all_metrics], dtype=float)
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{k}_ci95"] = float(1.96 * agg[f"{k}_std"] / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        else:
            agg[k] = v
    return agg
