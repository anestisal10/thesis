# Run EAP-IG Circuit Discovery
Write-Host "Running EAP-IG..."
python experiments/run_circuit_discovery.py --task ioi
python experiments/run_circuit_discovery.py --task greater_than
python experiments/run_circuit_discovery.py --task tool_selection

# Run LRP Circuit Discovery
Write-Host "`nRunning LRP..."
python experiments/run_lrp_discovery.py --task ioi
python experiments/run_lrp_discovery.py --task greater_than
python experiments/run_lrp_discovery.py --task tool_selection

# Evaluate Circuits
Write-Host "`nEvaluating Circuits..."
python experiments/evaluate_circuits.py --task ioi
python experiments/evaluate_circuits.py --task greater_than
python experiments/evaluate_circuits.py --task tool_selection

# Compare Circuits
Write-Host "`nComparing EAP-IG vs LRP..."
python experiments/compare_circuits.py --task ioi
python experiments/compare_circuits.py --task greater_than
python experiments/compare_circuits.py --task tool_selection

Write-Host "`nPipeline completely successfully."
