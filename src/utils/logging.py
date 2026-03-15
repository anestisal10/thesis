import json
import uuid
import os
import random
import datetime
import numpy as np
import torch
from pathlib import Path

def setup_experiment_dir(base_dir="results"):
    """Ensures results directory exists."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def set_seed(seed: int = 42):
    """
    Sets random seed for reproducibility across Python, NumPy, and PyTorch.
    Call this at the start of every experiment script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[set_seed] Global seed set to {seed}")

def log_experiment(task, method, config, metrics, base_dir="results"):
    """
    Logs an experiment run uniformly with a unique ID and timestamp.
    """
    setup_experiment_dir(base_dir)
    exp_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().isoformat()
    
    log_data = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "task": task,
        "method": method,
        "config": config,
        "metrics": metrics
    }
    
    file_path = os.path.join(base_dir, f"{task}_{method}_{exp_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Logged experiment {exp_id} to {file_path}")
    return file_path
