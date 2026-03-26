from typing import Optional


def get_single_token_id(tokenizer, text: str, prefer_space: bool = False) -> int:
    """
    Returns a single token id for a text fragment.
    Tries with and without a leading space; raises if neither is single-token.
    """
    if text.startswith(" "):
        candidates = [text]
    else:
        candidates = [f" {text}", text] if prefer_space else [text, f" {text}"]

    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]

    raise ValueError(f"Text fragment does not map to a single token: '{text}'")
