import torch
import torch.nn.functional as F

def compute_logit_diff(logits, target_token_id, distractor_token_id):
    """
    Computes the logit difference between the target token and the distractor token
    for the final position in the sequence.
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        target_token_id: int or Tensor of shape (batch,)
        distractor_token_id: int or Tensor of shape (batch,)
        
    Returns:
        Tensor of shape (batch,) containing the logit differences.
    """
    final_logits = logits[:, -1, :] # [batch, vocab_size]
    
    if isinstance(target_token_id, int):
        clean_logits = final_logits[:, target_token_id]
        distractor_logits = final_logits[:, distractor_token_id]
    else:
        # If batch of targets
        batch_size = logits.shape[0]
        batch_indices = torch.arange(batch_size, device=logits.device)
        clean_logits = final_logits[batch_indices, target_token_id]
        distractor_logits = final_logits[batch_indices, distractor_token_id]
        
    return clean_logits - distractor_logits


def compute_prob_diff(logits, correct_token_ids, incorrect_token_ids):
    """
    Computes probability difference for multi-answer settings (Hanna et al., 2024).
    Formula: Sum(P(correct_tokens)) - Sum(P(incorrect_tokens))
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        correct_token_ids: list or 1D Tensor of correct vocabulary indices
        incorrect_token_ids: list or 1D Tensor of incorrect vocabulary indices
        
    Returns:
        Tensor of shape (batch,) containing the prob diffs.
    """
    final_logits = logits[:, -1, :] # [batch, vocab_size]
    
    # Convert logits to probabilities
    probs = F.softmax(final_logits, dim=-1)
    
    # Sum probabilities of correct and incorrect classes
    # If correct_token_ids is a single ID, we index directly; if list/tensor, we sum.
    if isinstance(correct_token_ids, (int, torch.Tensor)) and (not isinstance(correct_token_ids, torch.Tensor) or correct_token_ids.dim() == 0):
        correct_probs = probs[:, correct_token_ids]
    else:
        correct_probs = probs[:, correct_token_ids].sum(dim=-1)
        
    if isinstance(incorrect_token_ids, (int, torch.Tensor)) and (not isinstance(incorrect_token_ids, torch.Tensor) or incorrect_token_ids.dim() == 0):
        incorrect_probs = probs[:, incorrect_token_ids]
    else:
        incorrect_probs = probs[:, incorrect_token_ids].sum(dim=-1)
    
    return correct_probs - incorrect_probs


def compute_normalized_faithfulness(m, b, b_prime):
    """
    Computes normalized faithfulness as defined in Hanna et al., 2024.
    Formula: (m - b') / (b - b')
    
    Args:
        m: float (mean logit diff or prob diff of the ablated circuit)
        b: float (mean logit diff or prob diff of the full unablated model on clean inputs)
        b_prime: float (mean logit diff or prob diff of the full unablated model on corrupted inputs)
        
    Returns:
        float: Normalized faithfulness score. Values near 1.0 mean fully faithful.
    """
    # Prevent division by zero if clean and corrupted baselines are identical
    if abs(b - b_prime) < 1e-8:
        return 0.0 
        
    return (m - b_prime) / (b - b_prime)


def compute_kl_divergence(clean_logits, ablated_logits):
    """
    Computes the KL divergence between the full model's probability distribution
    and the ablated model's probability distribution over the vocabulary for the final token.
    
    Args:
        clean_logits: Tensor of shape (batch, seq_len, vocab_size)
        ablated_logits: Tensor of shape (batch, seq_len, vocab_size)
        
    Returns:
        float: The batch mean KL divergence.
    """
    clean_final = clean_logits[:, -1, :]
    ablated_final = ablated_logits[:, -1, :]
    
    clean_probs = F.softmax(clean_final, dim=-1)
    ablated_log_probs = F.log_softmax(ablated_final, dim=-1)
    
    # KLDivLoss expects input in log-space and target in prob-space
    # reduction="batchmean" divides the total loss by the batch size
    kl_loss = F.kl_div(ablated_log_probs, clean_probs, reduction="batchmean")
    return kl_loss.item()

def compute_task_accuracy(logits, target_token_ids):
    """
    Computes binary task accuracy (whether the argmax of the final logit matches the target).
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        target_token_ids: Tensor of shape (batch,)
        
    Returns:
        float: Accuracy score in [0.0, 1.0]
    """
    final_logits = logits[:, -1, :] # [batch, vocab_size]
    predictions = torch.argmax(final_logits, dim=-1)
    correct = (predictions == target_token_ids).float()
    return correct.mean().item()

def compute_perplexity(model, tokenizer, dataset, max_samples=100, ctx_len=512):
    """
    Computes general model perplexity on a given dataset (e.g. wikitext)
    to measure catastrophic forgetting during pruning.
    """
    model.eval()
    nlls = []
    
    with torch.no_grad():
        for i, text in enumerate(dataset):
            if i >= max_samples:
                break
                
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len)
            input_ids = encodings.input_ids.to(model.device)
            
            # Skip empty or very short sequences
            if input_ids.shape[1] < 2:
                continue
                
            outputs = model(input_ids, labels=input_ids)
            # outputs.loss is the cross entropy loss
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            
    if not nlls:
        return float('inf')
        
    avg_nll = torch.stack(nlls).mean()
    perplexity = torch.exp(avg_nll)
    return perplexity.item()
