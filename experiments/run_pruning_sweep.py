import subprocess
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

tasks = ["ioi", "greater_than", "tool_selection"]
strategies = ["magnitude", "circuit_locked"]
sparsities = [0.2, 0.4, 0.6]

total_runs = len(tasks) * len(strategies) * len(sparsities)
current_run = 1

for task, strategy, sparsity in itertools.product(tasks, strategies, sparsities):
    print(f"\n=========================================")
    print(f"Run {current_run}/{total_runs}: [Task: {task}] [Strategy: {strategy}] [Sparsity: {sparsity*100:.0f}%]")
    print(f"=========================================\n")
    
    cmd = [
        sys.executable, str(REPO_ROOT / "experiments" / "run_pruning_experiment.py"),
        "--task", task,
        "--strategy", strategy,
        "--sparsity", str(sparsity),
        "--n_samples", "50" # 50 to speed up this sweep
    ]
    
    # Run the pruning experiment
    subprocess.run(cmd, check=True)
    current_run += 1

print("\nAll pruning experiments completed successfully.")
