"""Sequence log-probability computation."""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def response_token_logps(model, batch):
    """Return (token_logps, mask), both [B, T-1], for the response tokens of a batch.

    logits[:, t] predicts labels[:, t + 1]. Positions whose target label is -100
    (prompt and padding) are masked out and set to 0 in token_logps.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX

    log_probs = torch.log_softmax(logits, dim=-1)
    safe_targets = targets.masked_fill(~mask, 0)
    token_logps = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return token_logps * mask, mask


def sequence_logps(model, batch):
    """Return the summed response-token log-probability of each row, shape [B]."""
    token_logps, _ = response_token_logps(model, batch)
    return token_logps.sum(dim=-1)


def sft_loss(model, batch):
    """Return the response-only SFT negative log-likelihood.

    Reduction: mean over all response tokens in the batch (token-level mean),
    i.e. -(sum of masked token log-probs) / (number of response tokens).
    Longer responses therefore carry proportionally more weight. A batch with no
    response tokens returns 0 rather than NaN.
    """
    token_logps, mask = response_token_logps(model, batch)
    return -(token_logps * mask).sum() / mask.sum().clamp_min(1)
