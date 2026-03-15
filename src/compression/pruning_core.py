import torch
import torch.nn as nn
from typing import Dict, Optional, List

class MagnitudePruner:
    """
    A unified structural pruning engine that supports both:
    1. Baseline L2-Magnitude Pruning
    2. Circuit-Locked Pruning (preserving MI-discovered subgraphs)
    
    Operates by appending pre-hooks to zero-ablate the outputs of specific attention heads
    and MLP intermediate neurons.
    """
    def __init__(self, model, keep_ratio: float, circuit_scores: Optional[Dict] = None):
        self.model = model
        self.keep_ratio = keep_ratio
        
        # Determine model structure parameters
        if hasattr(model.config, "num_attention_heads"):
            self.num_heads = model.config.num_attention_heads
            if hasattr(model.config, "head_dim"):
                self.head_dim = model.config.head_dim
            elif hasattr(model.config, "hidden_size"):
                self.head_dim = model.config.hidden_size // self.num_heads
            else:
                raise NotImplementedError("Cannot determine head_dim from config.")
        else:
            raise NotImplementedError("Model config does not expose num_attention_heads.")
            
        self.num_layers = len(model.model.layers)
        self.intermediate_size = getattr(model.config, "intermediate_size", None)
        if self.intermediate_size is None:
             raise NotImplementedError("Model config does not expose intermediate_size.")

        # Set of node keys that MUST NOT be pruned (the protected circuit)
        self.protected_nodes = set()
        if circuit_scores is not None:
            self._parse_protected_nodes(circuit_scores)
            
        self.attn_masks = {}
        self.mlp_masks = {}
        self.hooks = []
        
        self._calculate_norms_and_build_masks()

    def _parse_protected_nodes(self, circuit_scores):
        """Extracts the node keys from the circuit dictionaries that should be protected."""
        # Typically, a circuit dict from EAP-IG/LRP assigns scores to nodes.
        # We need to compute the top 10% (keep_ratio) threshold of the original circuit
        # or assume the provided dict is ALREADY the minimal circuit we want to protect.
        # We assume the caller passes the minimal circuit dict (i.e. only keys > threshold).
        # Actually, let's look at `evaluate_circuits.py`: it thresholds at runtime.
        # We should threshold here using the same top 10% logic as evaluate_circuits.py
        
        all_scores = []
        for k, v in circuit_scores.items():
            if isinstance(v, list):
                all_scores.extend([abs(x) for x in v])
            else:
                all_scores.append(abs(v))

        if not all_scores:
            threshold = 0.0
        else:
            sorted_scores = sorted(all_scores, reverse=True)
            k_idx = max(0, int(len(sorted_scores) * 0.1) - 1) # Assuming circuits are always top 10%
            threshold = sorted_scores[k_idx]
            
        for k, v in circuit_scores.items():
            if isinstance(v, list):
                 for neuron_idx, score in enumerate(v):
                     if abs(score) >= threshold:
                         self.protected_nodes.add(f"{k}_{neuron_idx}")
            else:
                if abs(v) >= threshold:
                    self.protected_nodes.add(k)
                    
        print(f"[MagnitudePruner] Locked {len(self.protected_nodes)} nodes from the MI circuit.")

    def _calculate_norms_and_build_masks(self):
        """Calculates L2 norms for all nodes, ranks them, and builds zero-masks."""
        node_norms = {}
        
        # Calculate L2 Norms for all Attention Heads and MLP Neurons
        for i, layer in enumerate(self.model.model.layers):
            # Attention Heads
            # For Gemma, W_q, W_k, W_v, W_o. We can just use W_o (o_proj) and W_v (v_proj).
            # Usually, structural pruning looks at the output projection weights.
            o_proj_weight = layer.self_attn.o_proj.weight.detach() # shape: [hidden_dim, num_heads * head_dim]
            
            # Reshape into [hidden_dim, num_heads, head_dim] then compute norm per head
            # Norm is calculated over output features (dim 0) and head dimension (dim 2)
            o_proj_reshaped = o_proj_weight.view(o_proj_weight.shape[0], self.num_heads, self.head_dim)
            head_norms = torch.norm(o_proj_reshaped, p=2, dim=(0, 2)) # shape: [num_heads]
            
            for h in range(self.num_heads):
                node_norms[f"L{i}_attn_H{h}"] = head_norms[h].item()
                
            # MLP Neurons
            # For Gemma, it's (gate_proj(x) * up_proj(x)) * down_proj
            # We look at the output projection of the MLP stream: down_proj
            down_proj_weight = layer.mlp.down_proj.weight.detach() # shape: [hidden_dim, intermediate_size]
            
            # Norm per neuron (column of down_proj)
            neuron_norms = torch.norm(down_proj_weight, p=2, dim=0) # shape: [intermediate_size]
            
            for n in range(self.intermediate_size):
                node_norms[f"L{i}_mlp_neurons_{n}"] = neuron_norms[n].item()

        # Overwrite norms for protected nodes to infinity so they are never pruned
        for node in self.protected_nodes:
            if node in node_norms:
                node_norms[node] = float('inf')
                
        # Rank the nodes by norm
        all_node_names = list(node_norms.keys())
        all_node_values = list(node_norms.values())
        
        # Sort in descending order (highest norms first)
        sorted_indices = sorted(range(len(all_node_values)), key=lambda k: all_node_values[k], reverse=True)
        
        # Determine how many nodes to keep overall
        num_total_nodes = len(all_node_names)
        num_keep = int(num_total_nodes * self.keep_ratio)
        
        # Create a set of nodes to keep
        nodes_to_keep = set()
        for idx in sorted_indices[:num_keep]:
            nodes_to_keep.add(all_node_names[idx])
            
        print(f"[MagnitudePruner] Global Pruning Constraint: Keeping {num_keep}/{num_total_nodes} nodes (Sparsity: {(1 - self.keep_ratio)*100:.1f}%)")
        
        # Overlap check
        overlap = len(nodes_to_keep.intersection(self.protected_nodes))
        print(f"[MagnitudePruner] {overlap} of the {len(self.protected_nodes)} protected circuit nodes naturally survived L2 pruning.")

        # Build actual dense tensors for masking
        for i in range(self.num_layers):
            attn_mask = torch.zeros(self.num_heads, dtype=torch.bool)
            for h in range(self.num_heads):
                if f"L{i}_attn_H{h}" in nodes_to_keep:
                    attn_mask[h] = True
            self.attn_masks[i] = attn_mask
            
            mlp_mask = torch.zeros(self.intermediate_size, dtype=torch.bool)
            for n in range(self.intermediate_size):
                 if f"L{i}_mlp_neurons_{n}" in nodes_to_keep:
                     mlp_mask[n] = True
            self.mlp_masks[i] = mlp_mask

    def apply_hooks(self):
        """Attaches forward pre-hooks to zero-out pruned components dynamically."""
        self.remove_hooks()
        for i, layer in enumerate(self.model.model.layers):
            def make_attn_pre_hook(mask):
                def pre_hook(module, args):
                    x = args[0]
                    batch, seq, hidden = x.shape
                    x_reshaped = x.view(batch, seq, self.num_heads, self.head_dim)
                    mask_dev = mask.to(x.device).view(1, 1, self.num_heads, 1)
                    x_ablated = (x_reshaped * mask_dev).view(batch, seq, hidden)
                    return (x_ablated,)
                return pre_hook

            def make_mlp_pre_hook(mask):
                def pre_hook(module, args):
                    x = args[0]
                    mask_dev = mask.to(x.device).view(1, 1, -1)
                    x_ablated = x * mask_dev
                    return (x_ablated,)
                return pre_hook

            h1 = layer.self_attn.o_proj.register_forward_pre_hook(make_attn_pre_hook(self.attn_masks[i]))
            self.hooks.append(h1)

            h2 = layer.mlp.down_proj.register_forward_pre_hook(make_mlp_pre_hook(self.mlp_masks[i]))
            self.hooks.append(h2)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
