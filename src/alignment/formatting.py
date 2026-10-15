"""Prompt and chat formatting helpers."""

from __future__ import annotations

IM_END = "<|im_end|>"


def format_prompt(tokenizer, instruction):
    """Return the chat-formatted prompt for one user turn, ending with the assistant header.

    The tokenizer's template adds its own default system message; we add none.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )


def stop_token_ids(tokenizer):
    """Return [eos_token_id, <|im_end|> id], with missing ids and duplicates removed."""
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    if im_end is not None and im_end == getattr(tokenizer, "unk_token_id", None):
        im_end = None
    ids = []
    for token_id in (tokenizer.eos_token_id, im_end):
        if token_id is not None and token_id not in ids:
            ids.append(token_id)
    return ids
