"""Tests for prompt/response encoding and batch building."""

from __future__ import annotations

import torch

from alignment.tokenization import (
    build_dpo_batch,
    build_sft_batch,
    encode_prompt_response,
    pad_and_stack,
    resolve_pad_id,
)
from tests.toy_lm import ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class PadIsEosTokenizer(ToyTokenizer):
    pad_token_id = 1  # same as eos_token_id


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class NoPadNoEosTokenizer(ToyTokenizer):
    pad_token_id = None
    eos_token_id = None


def enc(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def test_encode_masks_prompt_and_labels_response_plus_eos():
    tok = ToyTokenizer()
    p, r = enc(tok, "Q: "), enc(tok, "ok")

    ids, labels = encode_prompt_response(tok, "Q: ", "ok", max_length=32)

    assert ids == p + r + [tok.eos_token_id]
    assert labels == [-100] * len(p) + r + [tok.eos_token_id]


def test_truncation_mid_response_drops_eos():
    tok = ToyTokenizer()
    p, r = enc(tok, "Q: "), enc(tok, "hello")
    max_length = len(p) + 2

    ids, labels = encode_prompt_response(tok, "Q: ", "hello", max_length=max_length)

    assert ids == p + r[:2]
    assert labels == [-100] * len(p) + r[:2]
    assert tok.eos_token_id not in labels


def test_exact_fit_keeps_eos():
    tok = ToyTokenizer()
    p, r = enc(tok, "Q: "), enc(tok, "hello")
    max_length = len(p) + len(r) + 1

    ids, labels = encode_prompt_response(tok, "Q: ", "hello", max_length=max_length)

    assert len(ids) == max_length
    assert labels[-1] == tok.eos_token_id


def test_prompt_longer_than_max_length_has_no_response_labels():
    tok = ToyTokenizer()

    ids, labels = encode_prompt_response(tok, "a long prompt here", "ok", max_length=4)

    assert ids == enc(tok, "a long prompt here")[:4]
    assert labels == [-100] * 4

    batch = build_sft_batch(tok, ["a long prompt here"], ["ok"], max_length=4)
    assert batch["labels"].eq(-100).all()
    assert batch["attention_mask"].eq(1).all()


def test_max_prompt_length_keeps_last_prompt_tokens():
    tok = ToyTokenizer()
    p, r = enc(tok, "abcdef"), enc(tok, "x")

    ids, labels = encode_prompt_response(tok, "abcdef", "x", max_length=32, max_prompt_length=3)

    assert ids == p[-3:] + r + [tok.eos_token_id]
    assert labels == [-100] * 3 + r + [tok.eos_token_id]


def test_empty_response_labels_only_eos():
    tok = ToyTokenizer()
    p = enc(tok, "Q: ")

    ids, labels = encode_prompt_response(tok, "Q: ", "", max_length=32)

    assert ids == p + [tok.eos_token_id]
    assert labels == [-100] * len(p) + [tok.eos_token_id]


def test_no_eos_token_appends_nothing():
    tok = NoEosTokenizer()
    p, r = enc(tok, "Q: "), enc(tok, "ok")

    ids, labels = encode_prompt_response(tok, "Q: ", "ok", max_length=32)

    assert ids == p + r
    assert labels == [-100] * len(p) + r


def test_resolve_pad_id_fallbacks():
    assert resolve_pad_id(ToyTokenizer()) == 0
    assert resolve_pad_id(NoPadTokenizer()) == NoPadTokenizer.eos_token_id
    assert resolve_pad_id(NoPadNoEosTokenizer()) == 0


def test_pad_and_stack_right_pads_by_length_not_by_value():
    # Row 0 legitimately contains the pad value as a real token.
    rows = [([5, 7, 9], [-100, 7, 9]), ([9, 9, 9, 9, 9], [-100, -100, 9, 9, 9])]

    batch = pad_and_stack(rows, pad_id=9)

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert all(v.dtype == torch.long for v in batch.values())
    assert batch["input_ids"].tolist() == [[5, 7, 9, 9, 9], [9, 9, 9, 9, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["labels"].tolist() == [[-100, 7, 9, -100, -100], [-100, -100, 9, 9, 9]]


def test_sft_batch_masks_prompt_labels_response_and_masks_padding():
    tok = ToyTokenizer()
    p0, r0 = enc(tok, "Task: "), enc(tok, "yes")
    p1, r1 = enc(tok, "Q: "), enc(tok, "ok")

    batch = build_sft_batch(tok, ["Task: ", "Q: "], ["yes", "ok"], max_length=32)

    eos = tok.eos_token_id
    len0, len1 = len(p0) + len(r0) + 1, len(p1) + len(r1) + 1
    width = max(len0, len1)
    assert batch["input_ids"].shape == (2, width)
    assert batch["labels"][0].tolist() == [-100] * len(p0) + r0 + [eos] + [-100] * (width - len0)
    assert batch["labels"][1].tolist() == [-100] * len(p1) + r1 + [eos] + [-100] * (width - len1)
    assert batch["attention_mask"][1].tolist() == [1] * len1 + [0] * (width - len1)
    assert batch["input_ids"][1, len1:].eq(tok.pad_token_id).all()


def test_pad_token_none_falls_back_to_eos():
    tok = NoPadTokenizer()

    batch = build_sft_batch(tok, ["Q: ", "Longer: "], ["ok", "ok"], max_length=32)

    short_len = len(enc(tok, "Q: ok")) + 1
    assert batch["input_ids"][0, short_len:].eq(tok.eos_token_id).all()
    assert batch["attention_mask"][0, short_len:].eq(0).all()
    assert batch["labels"][0, short_len:].eq(-100).all()


def test_pad_equal_to_eos_keeps_real_eos_label():
    tok = PadIsEosTokenizer()

    batch = build_sft_batch(tok, ["Q: ", "Longer: "], ["ok", "ok"], max_length=32)

    short_len = len(enc(tok, "Q: ok")) + 1
    assert batch["labels"][0, short_len - 1] == tok.eos_token_id
    assert batch["attention_mask"][0, short_len - 1] == 1
    assert batch["attention_mask"][0, short_len:].eq(0).all()
    assert batch["labels"][0, short_len:].eq(-100).all()
    # Long row's EOS also survives.
    assert batch["labels"][1, -1] == tok.eos_token_id
    assert batch["attention_mask"][1].eq(1).all()


def test_mixed_lengths_each_row_matches_row_built_alone():
    tok = ToyTokenizer()
    prompts = ["A: ", "Much longer prompt: ", "B"]
    responses = ["x", "yy", "a much longer response"]

    batch = build_sft_batch(tok, prompts, responses, max_length=24)

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        alone = build_sft_batch(tok, [prompt], [response], max_length=24)
        n = alone["input_ids"].shape[1]
        assert alone["attention_mask"].eq(1).all()
        for key in ("input_ids", "attention_mask", "labels"):
            assert batch[key][i, :n].tolist() == alone[key][0].tolist()
        assert batch["attention_mask"][i, n:].eq(0).all()
        assert batch["labels"][i, n:].eq(-100).all()


def test_row_order_preserved():
    tok = ToyTokenizer()
    prompts = ["short", "a much much longer prompt", "mid prompt"]
    responses = ["a", "b", "c"]

    batch = build_sft_batch(tok, prompts, responses, max_length=64)

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        ids, _ = encode_prompt_response(tok, prompt, response, max_length=64)
        assert batch["input_ids"][i, : len(ids)].tolist() == ids


def test_dpo_batch_builds_chosen_and_rejected_sft_batches():
    tok = ToyTokenizer()
    prompts = ["Task: ", "Q: "]
    chosen = ["good", "yes"]
    rejected = ["bad", "no"]

    batch = build_dpo_batch(tok, prompts, chosen, rejected, max_length=16)

    assert set(batch) == {"chosen", "rejected"}
    for side, responses in (("chosen", chosen), ("rejected", rejected)):
        expected = build_sft_batch(tok, prompts, responses, max_length=16)
        for key in ("input_ids", "attention_mask", "labels"):
            assert torch.equal(batch[side][key], expected[key])
