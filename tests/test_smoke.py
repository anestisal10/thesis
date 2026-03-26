import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compression.pruning_core import MagnitudePruner
from circuits.eap_ig import EAP_IG_Tracker

class DummyConfig:
    def __init__(self):
        self.num_attention_heads = 2
        self.head_dim = 4
        self.hidden_size = 8
        self.intermediate_size = 16

class DummyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(8, 8)
    def forward(self, x):
        return self.o_proj(x)

class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(16, 8)
    def forward(self, x):
        return self.down_proj(x)

class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttn()
        self.mlp = DummyMLP()

    def forward(self, x):
        return self.self_attn(x) + self.mlp(torch.cat([x, x], dim=-1))

class DummyModelInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(), DummyLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.model = DummyModelInner()

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x

def test_magnitude_pruner_smoke():
    model = DummyModel()
    
    circuit_scores = {
        "L0_attn_H1": 1.0,
        "L0_mlp_neurons": [0.5, 2.0]  # Only one high score
    }
    
    pruner = MagnitudePruner(model, keep_ratio=0.5, circuit_scores=circuit_scores)
    assert len(pruner.attn_masks) == 2
    assert pruner.attn_masks[0].shape == (2,)
    
    pruner.apply_hooks()
    assert len(pruner.hooks) > 0
    pruner.remove_hooks()
    assert len(pruner.hooks) == 0

def test_eap_ig_tracker_smoke():
    model = DummyModel()
    tracker = EAP_IG_Tracker(model)
    tracker.set_hooks("ig")
    
    # Fake forward pass
    x = torch.randn(1, 5, 8, requires_grad=True)
    out = model(x)
    out.sum().backward()
    tracker.accumulate_grads()
    
    assert len(tracker.current_tensors) > 0
    tracker.remove_hooks()

if __name__ == "__main__":
    print("Testing MagnitudePruner...")
    test_magnitude_pruner_smoke()
    print("Testing EAP_IG_Tracker...")
    test_eap_ig_tracker_smoke()
    print("All smoke tests passed!")
