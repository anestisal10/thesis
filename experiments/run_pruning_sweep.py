import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Task 7: Update run_pruning_sweep to use Canonical configs
configs = [
    REPO_ROOT / "configs" / "pruning_magnitude.yaml",
    REPO_ROOT / "configs" / "pruning_circuit_locked.yaml"
]

print("Starting pruning sweep using canonical configs...")
for config_path in configs:
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        continue
    
    print(f"\n=========================================")
    print(f"Running config: {config_path.name}")
    print(f"=========================================\n")
    
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_from_config.py"), "--config", str(config_path)]
    subprocess.run(cmd, check=True)

print("\nSweep completed.")
