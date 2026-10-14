"""Tokenization utilities."""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def encode_prompt_response(tokenizer, prompt, response, max_length, max_prompt_length=None):
    """Return (input_ids, labels) for one prompt/response pair with prompt tokens masked."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length :]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def resolve_pad_id(tokenizer):
    """Return pad_token_id, falling back to eos_token_id, then 0."""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def pad_and_stack(rows, pad_id):
    """Right-pad (input_ids, labels) rows to the longest row and stack into long tensors.

    The attention mask and padded labels come from row lengths, never from comparing
    token values to pad_id, since pad and EOS may share an id.
    """
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, row_labels) in enumerate(rows):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(row_labels, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return a right-padded batch with response-only labels."""
    rows = [
        encode_prompt_response(tokenizer, prompt, response, max_length)
        for prompt, response in zip(prompts, responses)
    ]
    return pad_and_stack(rows, resolve_pad_id(tokenizer))


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate SFT-style batches for chosen and rejected responses."""
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }
