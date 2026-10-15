"""Tests for formatting helpers and training utilities (offline, CPU-only)."""

from __future__ import annotations

import json
import math
import random

import pytest
import torch
from torch import nn

from alignment.formatting import format_prompt, stop_token_ids
from alignment.train_utils import (
    MetricsLogger,
    cosine_with_warmup,
    load_config,
    param_groups,
    save_model,
    set_seed,
)


class StubChatTokenizer:
    eos_token_id = 151643
    unk_token_id = None

    def __init__(self, vocab=None):
        self.vocab = {"<|endoftext|>": 151643, "<|im_end|>": 151645} if vocab is None else vocab
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        self.calls.append({"messages": messages, "tokenize": tokenize, "add_generation_prompt": add_generation_prompt})
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")


# --- formatting -------------------------------------------------------------


def test_format_prompt_uses_chat_template_with_generation_prompt_and_no_system():
    tok = StubChatTokenizer()

    text = format_prompt(tok, "Hello?")

    assert tok.calls == [
        {"messages": [{"role": "user", "content": "Hello?"}], "tokenize": False, "add_generation_prompt": True}
    ]
    assert text == "<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n"


def test_stop_token_ids_includes_eos_and_im_end():
    assert stop_token_ids(StubChatTokenizer()) == [151643, 151645]


def test_stop_token_ids_dedupes_and_drops_missing():
    no_im_end = StubChatTokenizer(vocab={"<|endoftext|>": 151643})
    assert stop_token_ids(no_im_end) == [151643]

    same = StubChatTokenizer(vocab={"<|im_end|>": 151643})
    assert stop_token_ids(same) == [151643]

    no_eos = StubChatTokenizer()
    no_eos.eos_token_id = None
    assert stop_token_ids(no_eos) == [151645]


# --- schedule -----------------------------------------------------------------


def lrs_over(warmup, total, peak=1.0):
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=peak)
    scheduler = cosine_with_warmup(optimizer, warmup, total)
    lrs = []
    for _ in range(total + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_cosine_with_warmup_shape():
    lrs = lrs_over(warmup=4, total=20, peak=2e-5)

    assert lrs[0] == 0.0
    assert lrs[2] == pytest.approx(1e-5)
    assert lrs[4] == pytest.approx(2e-5)
    assert max(lrs) == pytest.approx(2e-5)
    assert all(a >= b for a, b in zip(lrs[4:], lrs[5:]))  # monotone decay after warmup
    assert lrs[12] == pytest.approx(1e-5)  # halfway through the cosine
    assert lrs[20] == pytest.approx(0.0, abs=1e-12)


def test_cosine_without_warmup_starts_at_peak():
    lrs = lrs_over(warmup=0, total=10)
    assert lrs[0] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(0.0, abs=1e-12)


# --- param groups -------------------------------------------------------------


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 4)
        self.proj = nn.Linear(4, 4)
        self.layer_norm = nn.LayerNorm(4)
        self.scale = nn.Parameter(torch.ones(4))  # 1-D, not called norm
        self.frozen = nn.Parameter(torch.ones(2, 2), requires_grad=False)


def test_param_groups_split_decay_and_no_decay():
    model = TinyNet()

    groups = param_groups(model, weight_decay=0.1)

    by_decay = {group["weight_decay"]: {id(p) for p in group["params"]} for group in groups}
    assert by_decay[0.1] == {id(model.embed.weight), id(model.proj.weight)}
    assert by_decay[0.0] == {
        id(model.proj.bias),
        id(model.layer_norm.weight),
        id(model.layer_norm.bias),
        id(model.scale),
    }
    torch.optim.AdamW(groups, lr=1e-3)  # accepted by the optimizer


def test_param_groups_drops_empty_groups():
    model = nn.Module()
    model.bias_only = nn.Parameter(torch.zeros(3))
    groups = param_groups(model, weight_decay=0.1)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0


# --- logging, config, seeding, saving -------------------------------------------


def test_metrics_logger_writes_json_lines(tmp_path):
    path = tmp_path / "sub" / "metrics.jsonl"
    logger = MetricsLogger(path)

    logger.log({"type": "train", "step": 1, "train_loss": 2.5})
    first_read = path.read_text()  # visible immediately, before close
    logger.log({"type": "val", "step": 1, "val_loss": 2.0})
    logger.close()

    assert first_read.count("\n") == 1
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [{"type": "train", "step": 1, "train_loss": 2.5}, {"type": "val", "step": 1, "val_loss": 2.0}]


def test_load_config_applies_non_none_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("lr: 2.0e-5\nmax_steps: null\neval_every: 100\ndevice: cuda:0\n")

    cfg = load_config(path, {"max_steps": 3, "eval_every": None, "device": "cpu", "extra": 1})

    assert cfg == {"lr": 2e-5, "max_steps": 3, "eval_every": 100, "device": "cpu", "extra": 1}


def test_set_seed_is_reproducible():
    set_seed(123)
    first = (random.random(), torch.rand(3))
    set_seed(123)
    second = (random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_save_model_prefers_save_pretrained(tmp_path):
    calls = []

    class Saveable:
        def save_pretrained(self, out_dir, **kwargs):
            calls.append((self, str(out_dir), kwargs))

    model, tok = Saveable(), Saveable()
    save_model(model, tok, tmp_path / "out")

    assert (tmp_path / "out").is_dir()
    assert calls[0] == (model, str(tmp_path / "out"), {"safe_serialization": True})
    assert calls[1][0] is tok


def test_save_model_falls_back_to_state_dict(tmp_path):
    model = nn.Linear(2, 2)
    save_model(model, object(), tmp_path / "out")
    state = torch.load(tmp_path / "out" / "model.pt")
    assert torch.equal(state["weight"], model.weight)


def test_cosine_value_formula_midpoint():
    lrs = lrs_over(warmup=2, total=10)
    progress = (5 - 2) / (10 - 2)
    assert lrs[5] == pytest.approx(0.5 * (1 + math.cos(math.pi * progress)))
