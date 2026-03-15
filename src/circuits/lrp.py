import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

class LRPAnalyzer:
    """
    Layer-wise Relevance Propagation (LRP) for Transformer models.
    
    Implements the true LRP-ε rule for linear layers as per the paper:
        z_j = sum_i (a_i * W_ij) + b_j
        R_{i <- j} = (a_i * W_ij) / (z_j + epsilon * sign(z_j)) * R_j
        
    This class utilizes PyTorch backward hooks to override standard gradients, 
    forcing the network to backpropagate relevance (R) instead of gradients.
    """
    
    def __init__(self, model: PreTrainedModel, epsilon: float = 1e-9):
        self.model = model
        self.epsilon = epsilon
        self.hooks =[]
        
        self.activations = {}
        self.weight_relevances = {}
        self.neuron_relevances = {}
        self.head_relevances = {}
        
        if hasattr(model.config, "num_attention_heads"):
            self.num_heads = model.config.num_attention_heads
            self.head_dim = getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // self.num_heads if hasattr(model.config, "hidden_size") else None
            )
            if self.head_dim is None:
                raise NotImplementedError(
                    f"Cannot determine head_dim for model config: {type(model.config).__name__}. "
                    "Ensure the model config has 'head_dim' or 'hidden_size'."
                )
        else:
            raise NotImplementedError(
                f"Model config ({type(model.config).__name__}) does not expose 'num_attention_heads'. "
                "LRPAnalyzer requires this field to decompose attention head attributions."
            )
            
    def _make_forward_hook(self, name):
        def hook(module, input, output):
            # Save the input activations (a_i) for the LRP rule
            self.activations[name] = input[0].detach()
        return hook
    
    def _make_backward_hook(self, name, module):
        def hook(mod, grad_input, grad_output):
            # grad_output[0] is the relevance (R_j) flowing from the upper layer
            if grad_output[0] is None or name not in self.activations:
                return grad_input
                
            a = self.activations[name]       #[batch, seq, in_features]
            R_j = grad_output[0]             #[batch, seq, out_features]
            
            W = mod.weight
            b = mod.bias
            
            # 1. Recompute forward pass z = a_i * W_ij + b_j
            z = F.linear(a, W, b)
            
            # 2. Stabilize denominator (LRP-ε rule)
            sign_z = torch.sign(z)
            sign_z[sign_z == 0] = 1.0        # Prevent 0 multiplier
            z_stable = z + self.epsilon * sign_z
            
            # 3. Compute the shared factor S_j = R_j / z_j
            S = R_j / z_stable               #[batch, seq, out_features]
            
            # ------------------------------------------------------------------
            # PARAMETER-LEVEL ATTRIBUTION (Unstructured Pruning - Eq 8)
            # R_w_ij = W_ij * \sum_{batch, seq} (S_j * a_i)
            # ------------------------------------------------------------------
            S_sum = S.reshape(-1, S.shape[-1]) # [N, out_features]
            a_sum = a.reshape(-1, a.shape[-1]) # [N, in_features]
            
            R_W = (S_sum.T @ a_sum) * W 
            self.weight_relevances[name] = R_W.detach().cpu()
            
            # ------------------------------------------------------------------
            # NEURON-LEVEL ATTRIBUTION (Structured Pruning - Eq 6)
            # R_j aggregated over batch and sequence
            # ------------------------------------------------------------------
            self.neuron_relevances[name] = R_j.sum(dim=(0, 1)).detach().cpu()
            
            # ------------------------------------------------------------------
            # PROPAGATE RELEVANCE TO LOWER LAYERS (Eq 5)
            # R_i = a_i * \sum_j (S_j * W_ij)
            # ------------------------------------------------------------------
            R_i = a * F.linear(S, W.t())
            
            # ------------------------------------------------------------------
            # ATTENTION HEAD EXTRACTION (Approximating AttnLRP)
            # ------------------------------------------------------------------
            if "o_proj" in name or "out_proj" in name:
                batch, seq, _ = R_i.shape
                # The input to o_proj is the concatenated heads, reshape to extract
                rel = R_i.view(batch, seq, self.num_heads, self.head_dim)
                head_scores = rel.sum(dim=(0, 1, 3))  #[num_heads]
                for h in range(self.num_heads):
                    self.head_relevances[f"{name}_H{h}"] = head_scores[h].item()
            
            # Override standard gradient with true relevance to continue backprop
            new_grad_input = list(grad_input)
            new_grad_input[0] = R_i
            return tuple(new_grad_input)
            
        return hook
            
    def attach_hooks(self):
        self.remove_hooks()
        self.activations.clear()
        self.weight_relevances.clear()
        self.neuron_relevances.clear()
        self.head_relevances.clear()
        
        # Dynamically attach hooks to ALL linear layers (covers up, down, gate, q, k, v, o)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                h1 = module.register_forward_hook(self._make_forward_hook(name))
                h2 = module.register_full_backward_hook(self._make_backward_hook(name, module))
                self.hooks.extend([h1, h2])
            
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        
    def compute_lrp_scores(self, epsilon: float = 1e-9):
        """
        Aggregate and return computed LRP relevance scores.
        """
        scores = {}
        
        # 1. Parameter-level scores (Unstructured Pruning)
        # Note: Kept as PyTorch tensors. Using .tolist() for 4096x4096 weight matrices 
        # causes massive python-list memory bloat.
        for name, weight_rel in self.weight_relevances.items():
            scores[f"{name}_weights"] = weight_rel 
            
        # 2. Neuron-level scores (Structured Pruning)
        for name, neuron_rel in self.neuron_relevances.items():
            scores[f"{name}_neurons"] = neuron_rel.tolist()
            
        # 3. Attention head scores
        for name, head_val in self.head_relevances.items():
            scores[name] = head_val
            
        return scores


def get_lrp_scores(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    target_token_id: int,
    distractor_token_id: int,
    epsilon: float = 1e-9
):
    """
    Compute LRP relevance scores for a given input sequence.
    
    The relevance signal is the logit difference (target - distractor) at the final token position.
    
    Args:
        model: The causal LM model.
        input_ids: Input token IDs of shape [1, seq_len].
        target_token_id: The token we want the model to predict.
        distractor_token_id: The competing token.
        epsilon: Small constant for numerical stability.
        
    Returns:
        dict mapping component names to relevance scores.
    """
    model.eval()
    
    # We pass epsilon directly to the analyzer now so hooks use it
    analyzer = LRPAnalyzer(model, epsilon=epsilon)
    analyzer.attach_hooks()
    
    input_ids_leaf = input_ids.clone().requires_grad_(False)
    
    outputs = model(input_ids=input_ids_leaf, output_hidden_states=False)
    logits = outputs.logits  # [1, seq_len, vocab_size]
    
    final_logits = logits[0, -1, :]  # [vocab_size]
    relevance_signal = final_logits[target_token_id] - final_logits[distractor_token_id]
    
    model.zero_grad()
    relevance_signal.backward()
    
    scores = analyzer.compute_lrp_scores(epsilon=epsilon)
    analyzer.remove_hooks()
    
    return scores