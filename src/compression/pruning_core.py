import torch
import torch.nn as nn
from typing import Dict, Optional, List

from compression.ablator import StructuralAblator

class MagnitudePruner(StructuralAblator):
    """
    A unified structural pruning engine that supports both:
    1. Baseline L2-Magnitude Pruning
    2. Circuit-Locked Pruning (preserving MI-discovered subgraphs)
    
    Operates by appending pre-hooks to zero-ablate the outputs of specific attention heads
    and MLP intermediate neurons.
    """
    def __init__(self, model, keep_ratio: float, circuit_scores: Optional[Dict] = None):
        super().__init__(model)
        self.keep_ratio = keep_ratio
        
        # Set of node keys that MUST NOT be pruned (the protected circuit)
        self.protected_nodes = set()
        if circuit_scores is not None:
            self._parse_protected_nodes(circuit_scores)
            
        self._build_masks()

    def _parse_protected_nodes(self, circuit_scores):
        """Extracts the node keys from the circuit dictionaries that should be protected.
        Assumes the provided dict is already parsed as the minimal circuit to protect.
        """
        for k, v in circuit_scores.items():
            if isinstance(v, list):
                 for neuron_idx, score in enumerate(v):
                     self.protected_nodes.add(f"{k}_{neuron_idx}")
            else:
                 self.protected_nodes.add(k)
                    
        print(f"[MagnitudePruner] Locked {len(self.protected_nodes)} nodes from the MI circuit.")

    def _build_masks(self):
        """Calculates L2 norms for all nodes, ranks them, and builds zero-masks using vectorized operations."""
        # Tensors to store all norms
        # Keep norms on CPU to avoid device mismatch when model is on CUDA.
        all_head_norms = torch.zeros(self.num_layers, self.num_heads, device="cpu")
        all_neuron_norms = torch.zeros(self.num_layers, self.intermediate_size, device="cpu")
        
        # 1. Calculate L2 Norms
        for i, layer in enumerate(self.model.model.layers):
            # Attention Heads (Output projection weight norm)
            o_proj_weight = layer.self_attn.o_proj.weight.detach()
            o_proj_reshaped = o_proj_weight.view(o_proj_weight.shape[0], self.num_heads, -1)
            head_norms = torch.norm(o_proj_reshaped, p=2, dim=(0, 2)).detach().cpu()
            all_head_norms[i] = head_norms
            
            # MLP Neurons (Output projection weight norm)
            down_proj_weight = layer.mlp.down_proj.weight.detach()
            neuron_norms = torch.norm(down_proj_weight, p=2, dim=0).detach().cpu()
            all_neuron_norms[i] = neuron_norms

        # 2. Inject Protection (Protected MI circuit nodes get INF norm)
        for node in self.protected_nodes:
            if "attn" in node: # e.g., L0_attn_H5
                try:
                    parts = node.split("_")
                    layer_idx = int(str(parts[0])[1:])
                    head_idx = int(str(parts[2])[1:])
                    all_head_norms[layer_idx, head_idx] = float('inf')
                except Exception as e:
                    raise ValueError(f"Failed to parse protected attn node '{node}': {e}")
            elif "mlp" in node: # e.g., L0_mlp_neurons_123
                try:
                    parts = node.split("_")
                    layer_idx = int(str(parts[0])[1:])
                    neuron_idx = int(str(parts[3]))
                    all_neuron_norms[layer_idx, neuron_idx] = float('inf')
                except Exception as e:
                    raise ValueError(f"Failed to parse protected mlp node '{node}': {e}")

        # 3. Rank and Threshold
        flat_norms = torch.cat([all_head_norms.flatten(), all_neuron_norms.flatten()])
        num_total = flat_norms.numel()
        num_keep = int(num_total * self.keep_ratio)
        
        # Sort values to find threshold
        sorted_norms, _ = torch.sort(flat_norms, descending=True)
        threshold = sorted_norms[num_keep - 1] if num_keep > 0 else float('inf')
        
        print(f"[MagnitudePruner] Global Threshold: {threshold:.6f} | Keeping {num_keep}/{num_total} nodes")

        # 4. Build Masks Vectorized
        for i in range(self.num_layers):
            self.attn_masks[i] = all_head_norms[i] >= threshold
            self.mlp_masks[i] = all_neuron_norms[i] >= threshold
            
        # Logging overlap for sanity
        protected_count = len(self.protected_nodes)
        survived = torch.sum(all_head_norms == float('inf')).item() + torch.sum(all_neuron_norms == float('inf')).item()
        print(f"[MagnitudePruner] Protected nodes: {protected_count} | Enforced survival: {survived}")
