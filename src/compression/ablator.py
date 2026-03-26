import torch

class StructuralAblator:
    """
    Base class for structural ablation.
    Subclasses must define _build_masks() to populate self.attn_masks and self.mlp_masks.
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
        # Determine model structure parameters (Gemma 3 / Llama family)
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
        self.intermediate_size = getattr(model.config, "intermediate_size", 2048)

        self.attn_masks = {}
        self.mlp_masks = {}

    def _build_masks(self):
        raise NotImplementedError("Subclasses must implement _build_masks()")

    def apply_hooks(self):
        """Attaches forward pre-hooks to zero-out pruned components dynamically."""
        self.remove_hooks()
        for i, layer in enumerate(self.model.model.layers):
            def make_attn_pre_hook(mask):
                def pre_hook(module, args):
                    x = args[0]
                    batch, seq, hidden = x.shape
                    
                    # Hardening for GQA architectures (like Gemma 3)
                    if hidden % self.num_heads != 0:
                        raise RuntimeError(f"Hidden dim {hidden} not divisible by num_heads {self.num_heads}")
                    
                    h_dim = hidden // self.num_heads
                    if not printed[0]:
                        print(f"[StructuralAblator] Layer {i} o_proj input shape: {x.shape}, inferred head_dim: {h_dim}")
                        printed[0] = True

                    x_reshaped = x.view(batch, seq, self.num_heads, h_dim)
                    mask_dev = mask.to(x.device).view(1, 1, self.num_heads, 1)
                    x_ablated = (x_reshaped * mask_dev).view(batch, seq, hidden)
                    return (x_ablated,)
                
                # Captured list to only print once
                printed = [False]
                return pre_hook

            def make_mlp_pre_hook(mask):
                def pre_hook(module, args):
                    x = args[0]
                    mask_dev = mask.to(x.device).view(1, 1, -1)
                    x_ablated = x * mask_dev
                    return (x_ablated,)
                return pre_hook

            if i in self.attn_masks:
                h1 = layer.self_attn.o_proj.register_forward_pre_hook(make_attn_pre_hook(self.attn_masks[i]))
                self.hooks.append(h1)

            if i in self.mlp_masks:
                h2 = layer.mlp.down_proj.register_forward_pre_hook(make_mlp_pre_hook(self.mlp_masks[i]))
                self.hooks.append(h2)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
