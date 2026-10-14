"""Tests for response-token log-probabilities and the SFT loss."""

from __future__ import annotations

import torch
from torch import nn
from types import SimpleNamespace

from alignment.logprobs import response_token_logps, sequence_logps, sft_loss
from alignment.tokenization import build_sft_batch
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class BigramToyLM(nn.Module):
    """Logits at position t depend only on input_ids[t]."""

    def __init__(self, vocab_size, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Embedding(vocab_size, vocab_size)
        with torch.no_grad():
            self.table.weight.copy_(torch.randn(vocab_size, vocab_size, generator=generator))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(logits=self.table(input_ids))


class RecordingToyLM(ToyCausalLM):
    def __init__(self, vocab_size):
        super().__init__(vocab_size, bias_scale=0.5)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, labels=None):
        self.calls.append({"attention_mask": attention_mask, "labels": labels})
        return super().forward(input_ids, attention_mask=attention_mask, labels=labels)


class PadIsEosTokenizer(ToyTokenizer):
    pad_token_id = 1  # same as eos_token_id


def enc(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def hand_logp(model, prompt_ids, response_ids):
    """Sum of log p(response token | previous token) under a BigramToyLM."""
    ids = prompt_ids + response_ids
    log_probs = torch.log_softmax(model.table.weight.float(), dim=-1)
    total = torch.tensor(0.0)
    for position in range(len(prompt_ids), len(ids)):
        if position == 0:
            continue  # the first token has no predecessor
        total = total + log_probs[ids[position - 1], ids[position]]
    return total


def logps_for(model, tokenizer, prompts, responses, max_length=32):
    return sequence_logps(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def test_token_logps_shapes_and_mask_are_shifted():
    tok = ToyTokenizer()
    model = BigramToyLM(tok.vocab_size)
    batch = build_sft_batch(tok, ["Q: ", "Longer: "], ["ok", "yes"], max_length=32)

    token_logps, mask = response_token_logps(model, batch)

    B, T = batch["input_ids"].shape
    assert token_logps.shape == mask.shape == (B, T - 1)
    assert torch.equal(mask, batch["labels"][:, 1:].ne(-100))
    assert token_logps[~mask].eq(0).all()


def test_bigram_logp_matches_hand_computation():
    tok = ToyTokenizer()
    model = BigramToyLM(tok.vocab_size)
    prompt_ids, response_ids = enc(tok, "Q: "), enc(tok, "ok") + [tok.eos_token_id]

    logp = logps_for(model, tok, ["Q: "], ["ok"])

    torch.testing.assert_close(logp[0], hand_logp(model, prompt_ids, response_ids))


def test_mixed_length_rows_match_rows_computed_alone():
    tok = ToyTokenizer()
    model = BigramToyLM(tok.vocab_size)
    prompts = ["A: ", "A much longer prompt: ", "B"]
    responses = ["xyz", "y", "a fairly long response"]

    batched = logps_for(model, tok, prompts, responses)

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        alone = logps_for(model, tok, [prompt], [response])
        torch.testing.assert_close(batched[i], alone[0])
        expected = hand_logp(model, enc(tok, prompt), enc(tok, response) + [tok.eos_token_id])
        torch.testing.assert_close(batched[i], expected)


def test_prompt_tokens_do_not_contribute():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)

    logps = logps_for(
        model, tok, ["aaaa: ", "a completely different prompt: "], ["same", "same"], max_length=64
    )

    torch.testing.assert_close(logps[0], logps[1])


def test_extra_padding_does_not_contribute():
    tok = ToyTokenizer()
    model = BigramToyLM(tok.vocab_size)
    batch = build_sft_batch(tok, ["Q: "], ["ok"], max_length=32)
    extra = 5
    padded = {
        "input_ids": torch.cat([batch["input_ids"], torch.zeros(1, extra, dtype=torch.long)], dim=1),
        "attention_mask": torch.cat([batch["attention_mask"], torch.zeros(1, extra, dtype=torch.long)], dim=1),
        "labels": torch.cat([batch["labels"], torch.full((1, extra), -100, dtype=torch.long)], dim=1),
    }

    torch.testing.assert_close(sequence_logps(model, padded), sequence_logps(model, batch))


def test_pad_equal_to_eos_still_counts_eos():
    tok = PadIsEosTokenizer()
    model = BigramToyLM(tok.vocab_size)

    logps = logps_for(model, tok, ["Q: ", "A longer prompt: "], ["ok", "ok"])

    expected = hand_logp(model, enc(tok, "Q: "), enc(tok, "ok") + [tok.eos_token_id])
    without_eos = hand_logp(model, enc(tok, "Q: "), enc(tok, "ok"))
    torch.testing.assert_close(logps[0], expected)
    assert not torch.isclose(logps[0], without_eos)


def test_prompt_longer_than_max_length_gives_zero_logp_and_finite_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    batch = build_sft_batch(tok, ["a very long prompt", "another long prompt"], ["ok", "no"], max_length=4)

    logps = sequence_logps(model, batch)
    loss = sft_loss(model, batch)

    assert torch.equal(logps, torch.zeros(2))
    assert torch.isfinite(loss)


def test_sft_loss_is_mean_over_response_tokens_with_gradients():
    tok = ToyTokenizer()
    model = BigramToyLM(tok.vocab_size)
    batch = build_sft_batch(tok, ["Do: ", "Say: "], ["a", "longer"], max_length=32)

    loss = sft_loss(model, batch)

    token_count = batch["labels"][:, 1:].ne(-100).sum()
    expected = -sequence_logps(model, batch).sum() / token_count
    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.table.weight.grad is not None
    assert model.table.weight.grad.abs().sum() > 0


def test_row_order_preserved():
    tok = ToyTokenizer()
    model = BigramToyLM(tok.vocab_size)
    prompts = ["A: ", "Bb: ", "Ccc: "]
    responses = ["one", "two two", "x"]

    forward = logps_for(model, tok, prompts, responses)
    backward = logps_for(model, tok, prompts[::-1], responses[::-1])

    torch.testing.assert_close(backward, forward.flip(0))


def test_model_receives_attention_mask_and_no_labels():
    tok = ToyTokenizer()
    model = RecordingToyLM(tok.vocab_size)
    batch = build_sft_batch(tok, ["Q: ", "Longer: "], ["ok", "ok"], max_length=32)

    sequence_logps(model, batch)

    assert len(model.calls) == 1
    assert torch.equal(model.calls[0]["attention_mask"], batch["attention_mask"])
    assert model.calls[0]["labels"] is None
