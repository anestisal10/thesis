import torch
from transformers import PreTrainedModel

class EAP_IG_Tracker:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.clean_activations = {}
        self.corrupted_activations = {}
        self.accumulated_grads = {}
        self.current_tensors = {}
        self.hooks = []
        
        # Figure out the number of heads dynamically from model config.
        # Raises NotImplementedError for unknown architectures rather than silently guessing,
        # which would produce wrong head decompositions for models like Qwen.
        if hasattr(model.config, "num_attention_heads"):
            self.num_heads = model.config.num_attention_heads
            if hasattr(model.config, "head_dim"):
                self.head_dim = model.config.head_dim
            elif hasattr(model.config, "hidden_size"):
                self.head_dim = model.config.hidden_size // self.num_heads
            else:
                raise NotImplementedError(
                    f"Cannot determine head_dim for model config: {type(model.config).__name__}. "
                    "Ensure the model config has 'head_dim' or 'hidden_size'."
                )
        else:
            raise NotImplementedError(
                f"Model config ({type(model.config).__name__}) does not expose 'num_attention_heads'. "
                "EAP_IG_Tracker requires this field to decompose attention head attributions."
            )
            
    def _make_hook(self, name, capture_type):
        def hook(module, args, output):
            x = args[0] # The input to the linear projection
            if capture_type == "clean":
                self.clean_activations[name] = x.detach().clone()
            elif capture_type == "corrupted":
                self.corrupted_activations[name] = x.detach().clone()
            elif capture_type == "ig":
                if x.requires_grad:
                    x.retain_grad()
                    self.current_tensors[name] = x
        return hook

    def set_hooks(self, capture_type):
        """Attaches hooks to o_proj (Attn) and down_proj (MLP) for every layer."""
        self.remove_hooks()
        self.current_tensors = {}
        
        for i, layer in enumerate(self.model.model.layers):
            # Hook the input of o_proj (this tensor contains the concatenated head outputs)
            h1 = layer.self_attn.o_proj.register_forward_hook(self._make_hook(f"L{i}_attn", capture_type))
            self.hooks.append(h1)
            
            # Hook the input of down_proj (this tensor contains the MLP neuron activations)
            h2 = layer.mlp.down_proj.register_forward_hook(self._make_hook(f"L{i}_mlp", capture_type))
            self.hooks.append(h2)
            
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def accumulate_grads(self):
        """Called after loss.backward() to accumulate gradients for the current step."""
        for name, tensor in self.current_tensors.items():
            if name not in self.accumulated_grads:
                self.accumulated_grads[name] = torch.zeros_like(tensor)
            if tensor.grad is not None:
                self.accumulated_grads[name] += tensor.grad.detach()

    def compute_scores(self, n_steps: int):
        """Computes the final EAP-IG node attribution scores."""
        scores = {}
        
        for name in self.clean_activations:
            diff = self.corrupted_activations[name] - self.clean_activations[name]
            avg_grad = self.accumulated_grads.get(name, torch.zeros_like(diff)) / n_steps
            
            # Element-wise attribution
            attribution = diff * avg_grad
            
            if "attn" in name:
                # Shape: [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
                batch, seq, _ = attribution.shape
                attr = attribution.view(batch, seq, self.num_heads, self.head_dim)
                
                # Sum over batch, seq, and head_dim to get the score per head
                head_scores = attr.sum(dim=(0, 1, 3))
                for h in range(self.num_heads):
                    scores[f"{name}_H{h}"] = head_scores[h].item()
                    
            elif "mlp" in name:
                # Shape: [batch, seq, intermediate_size]
                # Sum over batch and seq to get the score per neuron
                neuron_scores = attribution.sum(dim=(0, 1))
                # Store as list to be JSON serializable
                scores[f"{name}_neurons"] = neuron_scores.cpu().tolist()
                
        return scores


def get_eap_ig_scores(
    model: PreTrainedModel, 
    clean_embeds: torch.Tensor, 
    corrupted_embeds: torch.Tensor, 
    attention_mask: torch.Tensor,
    metric_fn, 
    n_steps: int = 5
):
    """
    Computes Node Attribution Patching scores using Integrated Gradients (IG).
    Uses native PyTorch hooks on HuggingFace models.
    """
    model.eval()
    tracker = EAP_IG_Tracker(model)
    
    # 1. Forward pass on clean inputs
    tracker.set_hooks("clean")
    with torch.no_grad():
        model(inputs_embeds=clean_embeds, attention_mask=attention_mask, output_hidden_states=False)
        
    # 2. Forward pass on corrupted inputs
    tracker.set_hooks("corrupted")
    with torch.no_grad():
        model(inputs_embeds=corrupted_embeds, attention_mask=attention_mask, output_hidden_states=False)
        
    # 3. IG Steps
    diff_embeds = corrupted_embeds - clean_embeds
    tracker.set_hooks("ig")
    
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interpolated_embeds = clean_embeds + alpha * diff_embeds
        interpolated_embeds.requires_grad_(True)
        
        outputs = model(
            inputs_embeds=interpolated_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        loss = metric_fn(outputs.logits)
        model.zero_grad()
        loss.backward()
        
        tracker.accumulate_grads()
        
    tracker.remove_hooks()
    
    # 4. Compute final scores
    scores = tracker.compute_scores(n_steps)
    print(f"Computed EAP-IG node attribution scores over {n_steps} steps.")
    return scores
