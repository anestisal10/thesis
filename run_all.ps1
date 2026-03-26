# ============================================================
# run_all.ps1 — Thesis Experiment Pipeline Orchestrator
# ============================================================
# All experiments are driven by canonical YAML config files
# under configs/. Parameters are specified there — not here.
# Use scripts/run_from_config.py to invoke any experiment:
#
#   python scripts/run_from_config.py <script.py> <config.yaml>
#
# To override a single parameter, append --key value:
#
#   python scripts/run_from_config.py ... --task greater_than
#
# Tasks that are not currently active are commented out.
# Re-enable by uncommenting the relevant block.
# ============================================================

# ── Phase 1: EAP-IG Circuit Discovery ───────────────────────
Write-Host "=== Phase 1: EAP-IG Circuit Discovery ==="

# IOI (uncomment to run)
# python scripts/run_from_config.py experiments/run_circuit_discovery.py configs/circuit_discovery_eap_ig.yaml --task ioi --seeds "1,2,3"

# Arithmetic (uncomment to run)
python scripts/run_from_config.py experiments/run_circuit_discovery.py configs/circuit_discovery_eap_ig.yaml --task arithmetic --seeds "1,2,3"

# Tool Selection (ACTIVE)
python scripts/run_from_config.py experiments/run_circuit_discovery.py configs/circuit_discovery_eap_ig.yaml --task tool_selection --seeds "1,2,3"

# ── Phase 1: LRP Circuit Discovery ──────────────────────────
Write-Host "`n=== Phase 1: LRP Circuit Discovery ==="

# IOI (uncomment to run)
# python scripts/run_from_config.py experiments/run_lrp_discovery.py configs/circuit_discovery_lrp.yaml --task ioi --seeds "1,2,3"

# Arithmetic (uncomment to run)
# python scripts/run_from_config.py experiments/run_lrp_discovery.py configs/circuit_discovery_lrp.yaml --task arithmetic --seeds "1,2,3"

# Tool Selection (uncomment to run)
# python scripts/run_from_config.py experiments/run_lrp_discovery.py configs/circuit_discovery_lrp.yaml --task tool_selection --seeds "1,2,3"

# ── Phase 1: Circuit Evaluation ─────────────────────────────
Write-Host "`n=== Phase 1: Circuit Evaluation ==="

# EAP-IG circuits
# python scripts/run_from_config.py experiments/evaluate_circuits.py configs/evaluate_circuits_eap_ig.yaml --task ioi --seeds "1,2,3"
python scripts/run_from_config.py experiments/evaluate_circuits.py configs/evaluate_circuits_eap_ig.yaml --task arithmetic --seeds "1,2,3"
python scripts/run_from_config.py experiments/evaluate_circuits.py configs/evaluate_circuits_eap_ig.yaml --task tool_selection --seeds "1,2,3"

# LRP circuits (uncomment to run)
# python scripts/run_from_config.py experiments/evaluate_circuits.py configs/evaluate_circuits_lrp.yaml --task ioi --seeds "1,2,3"
# python scripts/run_from_config.py experiments/evaluate_circuits.py configs/evaluate_circuits_lrp.yaml --task arithmetic --seeds "1,2,3"
# python scripts/run_from_config.py experiments/evaluate_circuits.py configs/evaluate_circuits_lrp.yaml --task tool_selection --seeds "1,2,3"

# ── Phase 1: Circuit Comparison ─────────────────────────────
# Write-Host "`n=== Phase 1: EAP-IG vs LRP Comparison ==="
# python experiments/compare_circuits.py --task ioi
# python experiments/compare_circuits.py --task arithmetic
# python experiments/compare_circuits.py --task tool_selection

# ── Pillar 1: Pruning ────────────────────────────────────────
# Write-Host "`n=== Pillar 1: Magnitude Pruning Baseline ==="
# foreach ($sparsity in @("0.2", "0.4", "0.6")) {
#     python scripts/run_from_config.py experiments/run_pruning_experiment.py configs/pruning_magnitude.yaml --task ioi --sparsity $sparsity
# }

# Write-Host "`n=== Pillar 1: Circuit-Locked Pruning ==="
# foreach ($sparsity in @("0.2", "0.4", "0.6")) {
#     python scripts/run_from_config.py experiments/run_pruning_experiment.py configs/pruning_circuit_locked.yaml --task ioi --sparsity $sparsity
# }

Write-Host "`n=== Pipeline completed successfully. ==="
