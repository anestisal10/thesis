import torch
import torch.nn.functional as F

def compute_logit_diff(logits, valid_token_ids_list, invalid_token_ids_list):
    """
    Computes the Log-Sum-Exp (LSE) difference for multi-token classes.
    This provides a healthy, continuous gradient for EAP-IG when there are multiple right answers.
    """
    final_logits = logits[:, -1, :] # [batch, vocab_size]
    batch_size = final_logits.shape[0]
    logit_diffs = []
    
    for i in range(batch_size):
        valid_logits = final_logits[i, valid_token_ids_list[i]]
        invalid_logits = final_logits[i, invalid_token_ids_list[i]]
        
        # Use logsumexp to safely aggregate logits for multiple valid/invalid tokens
        # Formula: log(sum(exp(logits_valid))) - log(sum(exp(logits_invalid)))
        lse_valid = torch.logsumexp(valid_logits, dim=-1)
        lse_invalid = torch.logsumexp(invalid_logits, dim=-1)
        
        logit_diffs.append(lse_valid - lse_invalid)
        
    return torch.stack(logit_diffs)


def compute_prob_diff(logits, valid_token_ids_list, invalid_token_ids_list):
    """
    Computes probability difference: Sum(P(valid_tokens)) - Sum(P(invalid_tokens))
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        valid_token_ids_list: List of lists (or 1D tensors) of valid vocabulary indices per batch item
        invalid_token_ids_list: List of lists (or 1D tensors) of invalid vocabulary indices per batch item
    """
    final_logits = logits[:, -1, :] # [batch, vocab_size]
    probs = F.softmax(final_logits, dim=-1) # [batch, vocab_size]
    
    batch_size = probs.shape[0]
    prob_diffs = []
    
    for i in range(batch_size):
        # Sum the probability mass for all valid greater-than years
        p_valid = probs[i, valid_token_ids_list[i]].sum()
        # Sum the probability mass for all invalid less-than-or-equal years
        p_invalid = probs[i, invalid_token_ids_list[i]].sum()
        
        prob_diffs.append(p_valid - p_invalid)
        
    return torch.stack(prob_diffs)


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

def compute_task_accuracy(logits, valid_token_ids_list):
    """
    Computes accuracy by checking if the top predicted token is ANY of the valid years.
    """
    final_logits = logits[:, -1, :] # [batch, vocab_size]
    predictions = torch.argmax(final_logits, dim=-1) # [batch]
    
    batch_size = final_logits.shape[0]
    correct = 0.0
    
    for i in range(batch_size):
        pred_token = predictions[i].item()
        # Check if the model's top choice is mathematically greater than the prompt year
        if pred_token in valid_token_ids_list[i]:
            correct += 1.0
            
    return correct / batch_size

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
                
            labels = input_ids.clone()
            # Mask out the first token (BOS) from loss calculation for fair benchmark comparison
            labels[:, 0] = -100
            outputs = model(input_ids, labels=labels)
            # outputs.loss is the cross entropy loss
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            
    if not nlls:
        return float('inf')
        
    avg_nll = torch.stack(nlls).mean()
    perplexity = torch.exp(avg_nll)
    return perplexity.item()
