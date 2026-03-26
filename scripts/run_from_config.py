#!/usr/bin/env python3
"""
scripts/run_from_config.py
==========================
Thin orchestrator that reads a YAML config file and forwards
every key–value pair as CLI arguments to any experiment script.

Design goals:
  - Config files are the canonical source of truth for every
    parameter (GEMINI.md principle: "Track everything, assume nothing").
  - Experiment scripts themselves stay unchanged (argparse-based);
    this shim just bridges YAML → argv.
  - Any config key can be overridden on the command line by passing
    --key value after the config path.

Usage
-----
  python scripts/run_from_config.py \\
      experiments/run_circuit_discovery.py \\
      configs/circuit_discovery_eap_ig.yaml

  # Override a single key:
  python scripts/run_from_config.py \\
      experiments/run_circuit_discovery.py \\
      configs/circuit_discovery_eap_ig.yaml \\
      --task greater_than

  # Show constructed argv without running:
  python scripts/run_from_config.py \\
      experiments/run_circuit_discovery.py \\
      configs/circuit_discovery_eap_ig.yaml \\
      --dry-run-config
"""

import sys
import subprocess
from pathlib import Path

# We only import yaml inside the function so the failure message
# is clear if pyyaml is not installed.


def _load_yaml(config_path: Path) -> dict:
    try:
        import yaml  # pyyaml
    except ImportError:
        sys.exit(
            "[run_from_config] ERROR: pyyaml is not installed. "
            "Run: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_to_argv(config: dict) -> list[str]:
    """Convert a flat YAML dict to a list of --key value strings."""
    argv = []
    for key, value in config.items():
        cli_key = f"--{key.replace('_', '-')}"
        if value is None:
            # Omit null values entirely — let argparse use its default.
            continue
        if isinstance(value, bool):
            # Boolean flags: only add the flag itself (store_true style).
            if value:
                argv.append(cli_key)
        else:
            argv.extend([cli_key, str(value)])
    return argv


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    script_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    # Extra CLI overrides come after the config path
    extra_args = sys.argv[3:]

    if not script_path.exists():
        sys.exit(f"[run_from_config] ERROR: Script not found: {script_path}")
    if not config_path.exists():
        sys.exit(f"[run_from_config] ERROR: Config not found: {config_path}")

    config = _load_yaml(config_path)
    config_argv = _config_to_argv(config)

    # Extra CLI args override config values (they come last, so argparse
    # sees them after the config-derived args and overwrites as expected).
    # NOTE: for boolean flags this means --no-dry-run is not supported;
    # to disable a bool flag set it to false in the YAML.
    full_argv = [sys.executable, str(script_path)] + config_argv + extra_args

    # Handle the special --dry-run-config flag (our own flag, not passed on)
    if "--dry-run-config" in full_argv:
        full_argv.remove("--dry-run-config")
        print("[run_from_config] Constructed command:")
        print("  " + " ".join(full_argv))
        return

    print(f"[run_from_config] Config : {config_path}")
    print(f"[run_from_config] Script : {script_path}")
    print(f"[run_from_config] Command: {' '.join(full_argv)}")
    print()

    result = subprocess.run(full_argv, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
