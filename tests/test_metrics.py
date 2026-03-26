import torch
import pytest
from evaluation.metrics import (
    compute_logit_diff,
    compute_prob_diff,
    compute_normalized_faithfulness,
    compute_kl_divergence,
    compute_task_accuracy
)

def test_compute_logit_diff():
    # Shape: [batch=2, seq=3, vocab=4]
    logits = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 0.0, 0.5]], # Batch 0 final step: targets 0 vs 1 -> 1.0 - 2.0 = -1.0
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [3.0, 1.0, 0.0, 0.5]]  # Batch 1 final step: targets 0 vs 1 -> 3.0 - 1.0 = 2.0
    ])
    
    # Use lists for token IDs as expected by the new API
    diff = compute_logit_diff(logits, [[0], [0]], [[1], [1]])
    assert torch.allclose(diff, torch.tensor([-1.0, 2.0]))
    
    # Test batch of targets
    diff2 = compute_logit_diff(logits, [[0], [1]], [[1], [0]])
    # Batch 0: target 0, distractor 1 -> 1.0 - 2.0 = -1.0
    # Batch 1: target 1, distractor 0 -> 1.0 - 3.0 = -2.0
    assert torch.allclose(diff2, torch.tensor([-1.0, -2.0]))

def test_compute_prob_diff():
    # Simplified logits so softmax is easy to calculate
    logits = torch.tensor([
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [10.0, -10.0, -10.0, -10.0]] # Batch 0 final step: ~1.0 for index 0
    ])
    
    prob_diff = compute_prob_diff(logits, [[0]], [[1, 2]])
    assert prob_diff.item() > 0.99
    
def test_compute_normalized_faithfulness():
    # (m - b') / (b - b')
    b = 0.8
    b_prime = 0.2
    m = 0.5
    
    faith = compute_normalized_faithfulness(m, b, b_prime)
    assert abs(faith - 0.5) < 1e-5
    
    # Zero division prevention
    faith_zero = compute_normalized_faithfulness(0.5, 0.8, 0.8)
    assert faith_zero == 0.0

def test_compute_kl_divergence():
    clean_logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])
    ablated_logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])
    
    kl = compute_kl_divergence(clean_logits, ablated_logits)
    assert kl < 1e-6 # Should be zero for identical distributions
    
    ablated_logits_diff = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]])
    kl_diff = compute_kl_divergence(clean_logits, ablated_logits_diff)
    assert kl_diff > 0.0 # Should be > 0 for different distributions

def test_compute_task_accuracy():
    logits = torch.tensor([
        [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0]], # Pred: 2
        [[0.0, 0.0, 0.0, 0.0], [4.0, 2.0, 1.0, 0.0]]  # Pred: 0
    ])
    
    # Needs list of lists for multiple valid options
    valid_tokens = [[2], [1]] # Batch 0 correct (pred 2 in [2]), Batch 1 incorrect (pred 0 in [1])
    acc = compute_task_accuracy(logits, valid_tokens)
    assert acc == 0.5
    
    # 2D target (multiple correct options)
    valid_tokens_multi = [[2, 3], [0, 1]] # Both correct (pred 2 in [2, 3], pred 0 in [0, 1])
    acc2 = compute_task_accuracy(logits, valid_tokens_multi)
    assert acc2 == 1.0
