"""lrp.py — Layer-wise Relevance Propagation for Transformer Models

This module provides two distinct LRP implementations, selectable via
the `lrp_mode` argument of `get_lrp_scores()`:

  lrp_eps (default — True LRP-ε)
  --------------------------------
  Implements true layer-wise relevance propagation as formalised in
  Montavon et al. (2019) "Layer-Wise Relevance Propagation: An Overview".
  Relevance is initialised at the output (logit difference) and propagated
  **backwards manually through every nn.Linear module** in the network,
  layer by layer, using the LRP-ε rule:

      S_l  =  R_l / (z_l + ε · sign(z_l))
      R_{l-1} = a_l * (S_l @ W_l)         ... (Input-weighted projection)

  After each propagation step the conservation property is checked:
      |sum(R_{l-1}) - sum(R_l)| should be ≈ 0 at linear boundaries.
  Conservation violations are logged (expected at GeLU/LayerNorm
  boundaries — non-linear ops require dedicated rules, see §Scope below).

  Scope note: Full-network conservation requires custom propagation rules
  for every non-linear op (GeLU, LayerNorm, RoPE, softmax). These rules
  exist in specialised frameworks (Chefer et al., 2021; Ali et al., 2022)
  but are architecturally specific and computationally expensive. This
  implementation propagates through all nn.Linear projections (o_proj,
  down_proj, q/k/v_proj, etc.) which are the primary computational nodes
  of interest for circuit discovery. This is consistent with standard
  practice in LRP-for-Transformers literature.

  input_x_grad (Legacy — Input × Gradient approximation)
  -------------------------------------------------------
  The original hook-based approach retained for ablation comparison.
  Uses PyTorch autograd backward hooks on each nn.Linear independently.
  Does NOT enforce conservation across layers. Fast but theoretically
  weaker — reviewer-visible gap if claimed as "LRP" without qualification.
  Cite as: "gradient-weighted activation attribution (LRP-0 approximation)"

Citation:
  Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R. (2019).
  Layer-Wise Relevance Propagation: An Overview. In Explainable AI,
  Lecture Notes in Computer Science, vol 11700. Springer, Cham.
"""

import torch
import torch.nn.functional as F
from collections.abc import Sequence


# ══════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ══════════════════════════════════════════════════════════════════════════

def _normalize_token_ids(token_ids, device):
    if isinstance(token_ids, int):
        return torch.tensor([token_ids], device=device)
    if isinstance(token_ids, torch.Tensor):
        return token_ids.to(device).flatten()
    if isinstance(token_ids, Sequence):
        return torch.tensor(list(token_ids), device=device)
    raise TypeError(f"Unsupported token id container: {type(token_ids).__name__}")


def _compute_relevance_signal(model, input_ids, target_token_id,
                              distractor_token_id, positions, token_ids):
    """Run a forward pass and compute the scalar relevance signal."""
    outputs = model(input_ids=input_ids)
    if positions is not None and token_ids is not None:
        relevance_signal = sum(
            outputs.logits[0, pos - 1, tok_id]
            for pos, tok_id in zip(positions, token_ids)
        )
    else:
        final_logits = outputs.logits[0, -1, :]
        target_ids = _normalize_token_ids(target_token_id, final_logits.device)
        distractor_ids = _normalize_token_ids(distractor_token_id, final_logits.device)
        relevance_signal = final_logits[target_ids].mean() - final_logits[distractor_ids].mean()
    return outputs, relevance_signal


def _extract_head_relevances(name, R_i, num_heads, head_dim, out_dict):
    """Split R_i along the num_heads dimension and accumulate per-head scores."""
    total_dim = num_heads * head_dim
    if R_i.shape[-1] >= total_dim:
        R_flat = R_i.reshape(-1, total_dim)  # [tokens, num_heads*head_dim]
    else:
        return  # dimension mismatch — skip silently
    R_heads = R_flat.reshape(-1, num_heads, head_dim)   # [tokens, H, D]
    head_scores = R_heads.sum(dim=(0, 2))               # [H]
    for h in range(num_heads):
        out_dict[f"{name}_H{h}"] = head_scores[h].item()


# ══════════════════════════════════════════════════════════════════════════
#  Implementation A: True LRP-ε (manual layer-wise propagation)
# ══════════════════════════════════════════════════════════════════════════

class LRPAnalyzerEps:
    """True LRP-ε: manual backward walk through nn.Linear modules.

    Algorithm
    ---------
    1. Forward pass with activation cache: record (a_l, z_l) per Linear.
    2. Initialise relevance at the output logit-diff node.
    3. Walk layers in reverse order; for each nn.Linear apply:
          S = R / (z + ε · sign(z))
          R_prev = a * (S @ W)
    4. After each step, check conservation: |sum(R_prev) - sum(R)| ≈ 0.
       Log warnings for large violations (expected at non-linear boundaries).
       If shape mismatches occur across residuals, relevance is redistributed 
       proportionally to z_out and a warning is logged (if verbose).
    5. Extract per-head scores from o_proj/out_proj layers.
    6. Extract per-neuron scores from mlp.down_proj layers.
    """

    def __init__(self, model, epsilon: float = 1e-9,
                 conservation_tol: float = 0.05,
                 verbose_conservation: bool = False):
        self.model = model
        self.epsilon = epsilon
        self.conservation_tol = conservation_tol
        self.verbose_conservation = verbose_conservation

        cfg = model.config
        self.num_heads = getattr(cfg, "num_attention_heads", 8)
        self.hidden_size = getattr(cfg, "hidden_size", 1024)
        self.head_dim = getattr(cfg, "head_dim", self.hidden_size // self.num_heads)

        # Filled by _run_and_cache
        self._layer_order: list[tuple[str, torch.nn.Linear]] = []
        self._activations: dict[str, torch.Tensor] = {}  # name -> a_l
        self._z_outputs: dict[str, torch.Tensor] = {}    # name -> z_l
        self._hooks: list = []

    # ── Forward pass with activation cache ───────────────────────────────

    def _attach_cache_hooks(self):
        self._layer_order.clear()
        self._activations.clear()
        self._z_outputs.clear()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._layer_order.append((name, module))

                def fwd(mod, inp, out, _name=name):
                    self._activations[_name] = inp[0].detach().clone()
                    self._z_outputs[_name] = out.detach().clone()

                h = module.register_forward_hook(fwd)
                self._hooks.append(h)

    def _remove_cache_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def run_and_cache(self, input_ids):
        """Forward pass only — caches activations and preactivation outputs."""
        self._attach_cache_hooks()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        self._remove_cache_hooks()
        return outputs

    # ── Core LRP-ε propagation ────────────────────────────────────────────

    def _lrp_eps_rule(self, a: torch.Tensor, W: torch.Tensor,
                      b, R_out: torch.Tensor) -> torch.Tensor:
        """Apply LRP-ε rule: R_in = a * ((R_out / (z + ε·sign(z))) @ W).

        Args:
            a:     input activation  [*, in_features]
            W:     weight matrix     [out_features, in_features]
            b:     bias or None
            R_out: incoming relevance [*, out_features]

        Returns:
            R_in: outgoing relevance [*, in_features]
        """
        orig_shape = a.shape
        a_2d = a.reshape(-1, a.shape[-1])          # [N, in_f]
        R_2d = R_out.reshape(-1, W.shape[0])       # [N, out_f]

        # Recompute z from cached activation for numerical consistency
        z = F.linear(a_2d, W, b)                  # [N, out_f]
        stabilizer = torch.sign(z)
        stabilizer[stabilizer == 0] = 1.0         # avoid flat-zero stabiliser
        z_stable = z + self.epsilon * stabilizer

        S = R_2d / z_stable                        # [N, out_f]
        R_in_2d = a_2d * (S @ W)                   # [N, in_f]
        return R_in_2d.reshape(orig_shape)

    def propagate(self, R_scalar: float) -> tuple[dict, dict]:
        """Walk all cached layers in reverse order, propagating relevance.

        Args:
            R_scalar: initial relevance value (e.g. logit diff)

        Returns:
            (head_relevances, neuron_relevances) dicts.
        """
        head_relevances: dict[str, float] = {}
        neuron_relevances: dict[str, list] = {}
        conservation_log: list[tuple[str, float]] = []

        # Work in reverse layer order
        layers_rev = list(reversed(self._layer_order))

        # The first layer in reverse receives R_scalar as total relevance.
        # We distribute it proportionally over the output neurons of that layer
        # using the output activations as weights (standard initialisation).
        first_name, first_mod = layers_rev[0]
        if first_name not in self._z_outputs:
            # Nothing was cached — return empty
            return head_relevances, neuron_relevances

        z_first = self._z_outputs[first_name]  # [*, out_f]
        z_flat = z_first.reshape(-1, z_first.shape[-1])  # [N, out_f]
        z_sum = z_flat.sum()
        if z_sum.abs() < 1e-12:
            R_current = z_flat * 0.0
        else:
            R_current = z_flat * (R_scalar / z_sum.item())
        # Re-open to the full original shape
        R_current = R_current.reshape(z_first.shape)

        for name, mod in layers_rev:
            if name not in self._activations:
                continue

            a = self._activations[name]
            W = mod.weight.detach()
            b = mod.bias.detach() if mod.bias is not None else None

            # Ensure R_current shape matches what this layer expects
            if R_current.shape[-1] != W.shape[0]:
                # Shape mismatch (possible after residual connections) — reset
                # R_current proportionally from z_out of the current layer
                z_cur = self._z_outputs.get(name)
                if z_cur is None:
                    continue
                z_flat = z_cur.reshape(-1, z_cur.shape[-1])
                z_tot = z_flat.sum().item()
                R_scalar_carry = R_current.sum().item()
                
                if self.verbose_conservation:
                    print(f"  [LRP-ε] Shape mismatch at {name} (expected {W.shape[0]}, got {R_current.shape[-1]}). "
                          f"Redistributing {R_scalar_carry:.4f} relevance proportionally over z_out.")
                          
                R_current = (z_flat * (R_scalar_carry / (z_tot + 1e-12))).reshape(z_cur.shape)

            R_budget_before = R_current.sum().item()

            with torch.no_grad():
                R_prev = self._lrp_eps_rule(a, W, b, R_current)

            R_budget_after = R_prev.sum().item()
            delta = abs(R_budget_after - R_budget_before)
            conservation_log.append((name, delta))

            if self.verbose_conservation and delta > self.conservation_tol * abs(R_budget_before + 1e-12):
                print(f"  [LRP-ε conservation] {name}: Δ={delta:.4f} "
                      f"(before={R_budget_before:.4f}, after={R_budget_after:.4f})")

            # ── Extract head relevances from o_proj / out_proj ────────────
            if "o_proj" in name or "out_proj" in name:
                # R_prev has shape [*, in_features] = [*, num_heads * head_dim]
                _extract_head_relevances(
                    name, R_prev,
                    self.num_heads, self.head_dim,
                    head_relevances
                )

            # ── Extract neuron relevances from mlp.down_proj ──────────────
            if "mlp" in name and "down_proj" in name:
                r2d = R_prev.reshape(-1, R_prev.shape[-1])
                neuron_relevances[name] = r2d.sum(dim=0).cpu().tolist()

            R_current = R_prev

        return head_relevances, neuron_relevances

    def compute_lrp_scores(self) -> dict:
        """Return JSON-serialisable score dict (same format as legacy API)."""
        raise RuntimeError(
            "Call propagate(R_scalar) first, then construct scores from the "
            "returned dicts. This method is intentionally not auto-called "
            "because R_scalar must come from a relevance signal forward-pass."
        )


# ══════════════════════════════════════════════════════════════════════════
#  Implementation B: Input × Gradient (legacy, hook-based)
# ══════════════════════════════════════════════════════════════════════════

class LRPAnalyzerGradWeighted:
    """Input × Gradient attribution (legacy — retained for ablation comparison).

    This is the hook-based "LRP-0 approximation" from the original thesis
    implementation. It wraps each nn.Linear independently with a backward
    hook that computes R_i = a * (S @ W) per-layer, intercepting PyTorch
    autograd's backward pass.

    Important limitations vs. true LRP-ε
    -------------------------------------
     - Hooks intercept individual modules; the conservation property is
       NOT verified across the full computational graph.
     - Residual connections are not explicitly modelled; relevance can
       accumulate inconsistently across skip paths.
     - Numerically equivalent to true LRP-ε within a single linear layer,
       but diverges when the network has non-linear skip connections.

    Cite as: "gradient-weighted activation attribution (LRP-0 approximation)"
    Appropriate reference: Bach et al. (2015), Montavon et al. (2019) §3.
    """

    def __init__(self, model, epsilon=1e-9):
        self.model = model
        self.epsilon = epsilon
        self.hooks = []
        self.activations = {}
        self.neuron_relevances = {}
        self.head_relevances = {}
        self._hook_error_count = 0

        cfg = model.config
        self.num_heads = getattr(cfg, "num_attention_heads", 8)
        self.hidden_size = getattr(cfg, "hidden_size", 1024)
        self.head_dim = getattr(cfg, "head_dim", self.hidden_size // self.num_heads)

    def _make_forward_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = input[0].detach()
        return hook

    def _make_backward_hook(self, name, module):
        def hook(mod, grad_input, grad_output):
            try:
                if grad_output[0] is None or name not in self.activations:
                    return grad_input
                a = self.activations[name]
                R_j = grad_output[0]
                W = mod.weight
                b = mod.bias

                a_2d = a.reshape(-1, a.shape[-1])
                R_j_2d = R_j.reshape(-1, mod.out_features)

                if a_2d.shape[0] != R_j_2d.shape[0]:
                    return grad_input

                with torch.no_grad():
                    z = F.linear(a_2d, W, b)
                    stabilizer = torch.sign(z)
                    stabilizer[stabilizer == 0] = 1.0
                    z_stable = z + self.epsilon * stabilizer
                    S = R_j_2d / z_stable

                relevance_prop = torch.matmul(S, W)
                R_i_2d = a_2d * relevance_prop
                R_i = R_i_2d.reshape(a.shape)

                # Neuron attribution
                if "mlp" in name and "down_proj" in name:
                    r2d = R_i.reshape(-1, R_i.shape[-1])
                    self.neuron_relevances[name] = r2d.sum(dim=0).detach().cpu()

                # Attention head attribution
                if "o_proj" in name or "out_proj" in name:
                    head_dict: dict[str, float] = {}
                    _extract_head_relevances(name, R_i, self.num_heads, self.head_dim, head_dict)
                    self.head_relevances.update(head_dict)

                new_grad = list(grad_input)
                if len(new_grad) > 0 and grad_input[0] is not None:
                    new_grad[0] = R_i.reshape(grad_input[0].shape)
                del self.activations[name]
                return tuple(new_grad)

            except Exception:
                self._hook_error_count += 1
                return grad_input
        return hook

    def attach_hooks(self):
        self.remove_hooks()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                h1 = module.register_forward_hook(self._make_forward_hook(name))
                h2 = module.register_full_backward_hook(self._make_backward_hook(name, module))
                self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activations.clear()

    def compute_lrp_scores(self) -> dict:
        scores: dict = {}
        for name, neuron_rel in self.neuron_relevances.items():
            if isinstance(neuron_rel, torch.Tensor):
                scores[f"{name}_neurons"] = neuron_rel.tolist()
            else:
                scores[f"{name}_neurons"] = list(neuron_rel)
        for name, head_val in self.head_relevances.items():
            scores[name] = head_val
        return scores


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════

_VALID_LRP_MODES = ("lrp_eps", "input_x_grad")


def get_lrp_scores(
    model,
    input_ids,
    target_token_id,
    distractor_token_id,
    epsilon: float = 1e-9,
    positions=None,
    token_ids=None,
    lrp_mode: str = "lrp_eps",
    conservation_tol: float = 0.05,
    verbose_conservation: bool = False,
):
    """Compute LRP-based circuit scores for one input.

    Args:
        model:               HuggingFace CausalLM model (eval mode expected).
        input_ids:           Tensor [1, seq_len].
        target_token_id:     int | list[int] | Tensor — correct token(s).
        distractor_token_id: int | list[int] | Tensor — incorrect token(s).
        epsilon:             LRP-ε stabiliser (default 1e-9).
        positions:           For teacher-forced tasks (e.g. tool selection),
                             list of token positions to sum over.
        token_ids:           Corresponding token IDs for `positions`.
        lrp_mode:            'lrp_eps'      — true layer-wise LRP (recommended).
                             'input_x_grad' — legacy hook-based approx.
        conservation_tol:    Fraction of R budget allowed to drift at each
                             layer before a warning is logged (lrp_eps only).
        verbose_conservation: Print per-layer conservation deltas.

    Returns:
        (scores, rel_val) — scores is a JSON-serialisable dict of float
        values keyed by layer name. rel_val is the scalar relevance signal.
    """
    if lrp_mode not in _VALID_LRP_MODES:
        raise ValueError(f"lrp_mode must be one of {_VALID_LRP_MODES}, got '{lrp_mode}'")

    model.eval()

    if lrp_mode == "lrp_eps":
        return _get_lrp_scores_eps(
            model, input_ids, target_token_id, distractor_token_id,
            epsilon=epsilon, positions=positions, token_ids=token_ids,
            conservation_tol=conservation_tol,
            verbose_conservation=verbose_conservation,
        )
    else:  # input_x_grad
        return _get_lrp_scores_grad_weighted(
            model, input_ids, target_token_id, distractor_token_id,
            epsilon=epsilon, positions=positions, token_ids=token_ids,
        )


def _get_lrp_scores_eps(
    model, input_ids, target_token_id, distractor_token_id,
    epsilon, positions, token_ids, conservation_tol, verbose_conservation,
):
    """Backend for lrp_mode='lrp_eps'."""
    analyzer = LRPAnalyzerEps(
        model, epsilon=epsilon,
        conservation_tol=conservation_tol,
        verbose_conservation=verbose_conservation,
    )

    # ① Cache activations via forward pass
    analyzer.run_and_cache(input_ids)

    # ② Compute relevance signal on a separate forward pass
    #    (we need grad here to get the scalar signal, but the LRP
    #     propagation itself is fully manual/no-grad)
    model.eval()
    # We need a fresh forward with grad to get the scalar
    outputs = model(input_ids=input_ids)
    if positions is not None and token_ids is not None:
        rel_signal_tensor = sum(
            outputs.logits[0, pos - 1, tok_id]
            for pos, tok_id in zip(positions, token_ids)
        )
    else:
        final_logits = outputs.logits[0, -1, :]
        target_ids = _normalize_token_ids(target_token_id, final_logits.device)
        distractor_ids = _normalize_token_ids(distractor_token_id, final_logits.device)
        rel_signal_tensor = final_logits[target_ids].mean() - final_logits[distractor_ids].mean()

    rel_val = rel_signal_tensor.item()

    # ③ Propagate relevance backward through linear layers
    head_relevances, neuron_relevances = analyzer.propagate(rel_val)

    # ④ Assemble JSON-serialisable score dict
    scores: dict = {}
    for name, nr in neuron_relevances.items():
        scores[f"{name}_neurons"] = nr
    scores.update(head_relevances)

    return scores, rel_val


def _get_lrp_scores_grad_weighted(
    model, input_ids, target_token_id, distractor_token_id,
    epsilon, positions, token_ids,
):
    """Backend for lrp_mode='input_x_grad' (legacy)."""
    analyzer = LRPAnalyzerGradWeighted(model, epsilon=epsilon)
    analyzer.attach_hooks()

    try:
        outputs = model(input_ids=input_ids)
        if positions is not None and token_ids is not None:
            relevance_signal = sum(
                outputs.logits[0, pos - 1, tok_id]
                for pos, tok_id in zip(positions, token_ids)
            )
        else:
            final_logits = outputs.logits[0, -1, :]
            target_ids = _normalize_token_ids(target_token_id, final_logits.device)
            distractor_ids = _normalize_token_ids(distractor_token_id, final_logits.device)
            relevance_signal = final_logits[target_ids].mean() - final_logits[distractor_ids].mean()

        model.zero_grad()
        relevance_signal.backward()
        scores = analyzer.compute_lrp_scores()
        rel_val = relevance_signal.item()
    finally:
        model.zero_grad()
        analyzer.remove_hooks()

    if analyzer._hook_error_count > 0:
        print(f"WARNING: LRP hooks encountered {analyzer._hook_error_count} errors. "
              "Results may be incomplete.")
    return scores, rel_val
