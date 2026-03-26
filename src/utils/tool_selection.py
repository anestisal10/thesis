from typing import Dict, List, Sequence
import torch
import torch.nn.functional as F


def build_tool_selection_prompt(messages: List[Dict], tools: List[Dict]) -> str:
    """
    Build a plain-text prompt that makes the next token the tool name.
    This avoids relying on model-specific tool-call chat templates.
    """
    tool_names = [t["function"]["name"] for t in tools]
    user_query = messages[-1]["content"] if messages else ""
    tools_str = ", ".join(tool_names)

    prompt = (
        "Select the best tool for the user query. "
        f"Available tools: {tools_str}. "
        f"Query: {user_query}\n"
        "Tool: "
    )
    return prompt


def build_tool_call_message(tool_name: str, arguments: str = "{}") -> Dict:
    """
    Build an assistant tool call message for HF chat templates.
    """
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }


def find_subsequence(seq: Sequence[int], subseq: Sequence[int], start: int = 0) -> int:
    """
    Find the start index of subseq in seq starting at start. Returns -1 if not found.
    """
    if not subseq:
        return -1
    max_i = len(seq) - len(subseq)
    for i in range(start, max_i + 1):
        if list(seq[i : i + len(subseq)]) == list(subseq):
            return i
    return -1


def tool_call_logprob(model, tokenizer, messages, tools, target_tool: str) -> float:
    """
    Teacher-forced logprob of a specific tool call.
    Uses build_tool_call_message to create the exact target response,
    then tokenizes the conversation and extracts the logprob of the target
    tool name tokens.
    """
    clean_full_ids = tokenizer.apply_chat_template(
        messages + [build_tool_call_message(target_tool)],
        tools=tools, tokenize=True, add_generation_prompt=False
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True
    )
    
    tool_tokens = tokenizer.encode(target_tool, add_special_tokens=False)
    start_idx = find_subsequence(clean_full_ids, tool_tokens, start=len(prompt_ids))
    if start_idx < 0:
        raise ValueError(f"Could not locate tool name tokens in tool-call sequence for {target_tool}.")
        
    input_tensor = torch.tensor([clean_full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_tensor).logits
        
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    total_lp = 0.0
    for pos, tok_id in zip(range(start_idx, start_idx + len(tool_tokens)), tool_tokens):
        # Logits at index `pos - 1` predict the token at index `pos`
        total_lp += logprobs[0, pos - 1, tok_id].item()
        
    return total_lp
