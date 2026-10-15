"""Offline tests for batched, resumable generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from alignment.generation import generate_outputs, make_hf_generate_fn
from alignment.io_utils import append_jsonl, read_jsonl
from tests.toy_lm import ToyTokenizer

STOP_A, STOP_B = 1, 5  # eos and a stand-in for <|im_end|>


def eval_rows():
    return [
        {"id": "r0", "instruction": "short", "dataset": "alpaca_eval", "source": "x"},
        {"id": "r1", "instruction": "a much much longer instruction", "dataset": "alpaca_eval"},
        {"id": "r2", "instruction": "mid length one", "dataset": "hh", "category": "harmful",
         "expected_behavior": "Refuse."},
        {"id": "r3", "instruction": "tiny", "dataset": "hh", "category": "benign", "expected_behavior": "Help."},
        {"id": "r4", "instruction": "another fairly long one", "dataset": "alpaca_eval"},
    ]


class FakeGenerate:
    def __init__(self):
        self.calls = []

    def __call__(self, instructions):
        self.calls.append(list(instructions))
        return [(f"out:{text}", len(text), len(text) % 2 == 0) for text in instructions]


# --- generate_outputs ---------------------------------------------------------


def test_outputs_keep_original_order_and_batches_are_length_sorted():
    rows = eval_rows()
    fake = FakeGenerate()

    outputs = generate_outputs(rows, fake, batch_size=2, generator_name="sft")

    assert [o["id"] for o in outputs] == [r["id"] for r in rows]
    assert [o["output"] for o in outputs] == [f"out:{r['instruction']}" for r in rows]
    flat = [text for call in fake.calls for text in call]
    assert [len(t) for t in flat] == sorted((len(r["instruction"]) for r in rows), reverse=True)
    assert [len(call) for call in fake.calls] == [2, 2, 1]


def test_output_row_fields():
    outputs = generate_outputs(eval_rows(), FakeGenerate(), batch_size=3, generator_name="dpo_hh")
    by_id = {o["id"]: o for o in outputs}

    assert by_id["r2"] == {
        "id": "r2",
        "instruction": "mid length one",
        "output": "out:mid length one",
        "generator": "dpo_hh",
        "dataset": "hh",
        "category": "harmful",
        "expected_behavior": "Refuse.",
        "num_new_tokens": len("mid length one"),
        "hit_stop": True,
    }
    assert "category" not in by_id["r0"] and "expected_behavior" not in by_id["r0"]
    assert "source" not in by_id["r0"]


def test_resume_skips_done_ids(tmp_path):
    rows = eval_rows()
    partial = tmp_path / "out.partial.jsonl"
    first = FakeGenerate()

    class Interrupt(Exception):
        pass

    def crash_on_second_batch(instructions):
        if first.calls:
            raise Interrupt
        return first(instructions)

    with pytest.raises(Interrupt):
        generate_outputs(rows, crash_on_second_batch, batch_size=2, generator_name="g", partial_path=partial)
    saved = read_jsonl(partial)
    assert len(saved) == 2

    second = FakeGenerate()
    outputs = generate_outputs(rows, second, batch_size=2, generator_name="g", partial_path=partial)

    done_ids = {row["id"] for row in saved}
    resumed = {text for call in second.calls for text in call}
    assert resumed == {r["instruction"] for r in rows if r["id"] not in done_ids}
    assert [o["id"] for o in outputs] == [r["id"] for r in rows]
    assert len(read_jsonl(partial)) == len(rows)


def test_wrong_number_of_results_raises():
    with pytest.raises(ValueError, match="returned 1 results for 2"):
        generate_outputs(eval_rows(), lambda batch: [("x", 1, True)], batch_size=2, generator_name="g")


def test_append_jsonl_appends(tmp_path):
    path = tmp_path / "a" / "x.jsonl"
    append_jsonl(path, [{"id": 1}])
    append_jsonl(path, [{"id": 2}, {"id": 3}])
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}, {"id": 3}]


# --- make_hf_generate_fn ------------------------------------------------------


class ChatToyTokenizer(ToyTokenizer):
    """ToyTokenizer plus the bits of the HF API generation uses."""

    padding_side = "right"

    def __init__(self):
        super().__init__()
        self.prompts = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return f"U:{messages[0]['content']}-A:"

    def convert_tokens_to_ids(self, token):
        return STOP_B if token == "<|im_end|>" else None

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        self.prompts.append({"texts": list(texts), "padding_side": self.padding_side,
                             "add_special_tokens": add_special_tokens})
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        width = max(len(e) for e in encoded)
        ids, mask = [], []
        for e in encoded:
            pad = [self.pad_token_id] * (width - len(e))
            ids.append(pad + e if self.padding_side == "left" else e + pad)
            mask.append([0] * len(pad) + [1] * len(e) if self.padding_side == "left" else [1] * len(e) + [0] * len(pad))
        return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))

    def decode(self, ids, skip_special_tokens=False):
        special = {self.pad_token_id, self.eos_token_id, STOP_B}
        return "".join(
            self.inv_vocab.get(int(i), "?") for i in ids if not (skip_special_tokens and int(i) in special)
        )


class FakeHFModel(torch.nn.Module):
    """Returns prompt + fixed continuations; records generate kwargs."""

    def __init__(self, continuations):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.continuations = continuations
        self.kwargs = None

    def generate(self, input_ids, attention_mask, **kwargs):
        assert not torch.is_grad_enabled()
        self.kwargs = {"attention_mask": attention_mask, **kwargs}
        new = torch.tensor(self.continuations[: input_ids.shape[0]])
        return torch.cat([input_ids, new], dim=1)


def test_make_hf_generate_fn_decodes_new_tokens_and_detects_stop():
    tok = ChatToyTokenizer()
    v = tok.vocab
    pad = tok.pad_token_id
    continuations = [
        [v[" "], v["h"], v["i"], v[" "], STOP_A, pad],  # stops on eos, then padding
        [v["o"], v["k"], STOP_B, pad, pad, pad],  # stops on <|im_end|>
        [v["x"], v["y"], v["z"], v["x"], v["y"], v["z"]],  # runs out of tokens
    ]
    model = FakeHFModel(continuations)
    gen_cfg = {"max_new_tokens": 6, "repetition_penalty": 1.1, "no_repeat_ngram_size": 0}

    results = make_hf_generate_fn(model, tok, gen_cfg)(["Hi", "A longer one", "Q"])

    assert results == [("hi", 5, True), ("ok", 3, True), ("xyzxyz", 6, False)]
    call = tok.prompts[0]
    assert call["texts"] == ["U:Hi-A:", "U:A longer one-A:", "U:Q-A:"]
    assert call["padding_side"] == "left" and call["add_special_tokens"] is False
    assert tok.padding_side == "right"  # restored
    kwargs = model.kwargs
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 6
    assert kwargs["repetition_penalty"] == 1.1
    assert kwargs["eos_token_id"] == [STOP_A, STOP_B]
    assert kwargs["pad_token_id"] == pad
    assert "no_repeat_ngram_size" not in kwargs  # 0 means off


def test_make_hf_generate_fn_passes_no_repeat_ngram_and_pad_fallback():
    tok = ChatToyTokenizer()
    tok.pad_token_id = None
    model = FakeHFModel([[STOP_A]])

    make_hf_generate_fn(model, tok, {"max_new_tokens": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 3})(["Hi"])

    assert model.kwargs["no_repeat_ngram_size"] == 3
    assert model.kwargs["pad_token_id"] == tok.eos_token_id


def test_make_hf_generate_fn_with_tiny_real_hf_model():
    transformers = pytest.importorskip("transformers")
    tok = ChatToyTokenizer()
    config = transformers.Qwen2Config(
        vocab_size=tok.vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    gen_cfg = {"max_new_tokens": 4, "repetition_penalty": 1.1, "no_repeat_ngram_size": 2}

    results = make_hf_generate_fn(model, tok, gen_cfg)(["Hi there", "Q"])

    assert len(results) == 2
    for text, num_new, hit_stop in results:
        assert isinstance(text, str) and text == text.strip()
        assert 1 <= num_new <= 4
        assert hit_stop or num_new == 4
