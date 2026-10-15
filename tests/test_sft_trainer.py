"""CPU smoke tests for the SFT training loop using the toy model and tokenizer."""

from __future__ import annotations

import json
import math

import pytest
import torch

from alignment.sft_trainer import train_sft
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def identity_format(tokenizer, instruction):
    return instruction


def state_dict_save(model, tokenizer, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")


def fake_rows(count, prefix):
    return [{"id": f"{prefix}-{i}", "instruction": f"Q{i % 3}: ", "response": "yes ok"} for i in range(count)]


def base_cfg(**changes):
    cfg = {
        "max_length": 32,
        "max_prompt_length": 16,
        "batch_size": 2,
        "grad_accum": 2,
        "lr": 0.1,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "epochs": 1,
        "max_steps": 3,
        "eval_every": 1,
        "log_every": 1,
        "seed": 0,
        "device": "cpu",
        "fp16": True,  # must be ignored on CPU
    }
    cfg.update(changes)
    return cfg


def read_metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(tmp_path, **changes):
    torch.manual_seed(0)
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    result = train_sft(
        base_cfg(**changes), model, tok, fake_rows(12, "train"), fake_rows(4, "val"), tmp_path / "run",
        identity_format, state_dict_save,
    )
    return result, model, tok


def test_train_sft_smoke(tmp_path, capsys):
    result, model, tok = run(tmp_path)
    out_dir = tmp_path / "run"

    metrics = read_metrics(out_dir / "metrics.jsonl")
    train = [m for m in metrics if m["type"] == "train"]
    val = [m for m in metrics if m["type"] == "val"]
    assert [m["step"] for m in train] == [1, 2, 3]
    assert [m["step"] for m in val] == [0, 1, 2, 3]
    for m in train:
        assert math.isfinite(m["train_loss"]) and math.isfinite(m["grad_norm"])
        assert {"lr", "loss_scale", "tokens_seen", "elapsed_s"} <= set(m)
    assert all(math.isfinite(m["val_loss"]) for m in val)
    assert val[-1]["val_loss"] < val[0]["val_loss"]
    assert train[-1]["tokens_seen"] > train[0]["tokens_seen"] > 0

    assert (out_dir / "best" / "model.pt").exists()
    assert (out_dir / "final" / "model.pt").exists()
    assert (out_dir / "config_resolved.yaml").exists()
    best = json.loads((out_dir / "best.json").read_text())
    assert best == {"step": 3, "val_loss": pytest.approx(min(m["val_loss"] for m in val))}
    assert result["best_val_loss"] == pytest.approx(best["val_loss"])
    assert result["steps"] == 3

    printed = capsys.readouterr().out
    assert "labelled" in printed.lower()


def test_second_call_returns_early(tmp_path, capsys):
    run(tmp_path)
    before = (tmp_path / "run" / "metrics.jsonl").read_text()
    capsys.readouterr()

    result, model, _ = run(tmp_path)

    assert result is None
    assert "already exists" in capsys.readouterr().out
    assert (tmp_path / "run" / "metrics.jsonl").read_text() == before


def test_steps_follow_epochs_and_grad_accum_without_max_steps(tmp_path):
    # 12 rows / batch 2 = 6 micro-batches; grad_accum 4 -> ceil(6 / 4) = 2 steps per epoch.
    result, _, _ = run(tmp_path, max_steps=None, grad_accum=4, epochs=2, eval_every=100)
    metrics = read_metrics(tmp_path / "run" / "metrics.jsonl")
    assert result["steps"] == 4
    assert [m["step"] for m in metrics if m["type"] == "val"] == [0, 4]


def test_val_loss_is_token_weighted_not_mean_of_batch_means(tmp_path):
    from alignment.sft_trainer import evaluate
    from alignment.tokenization import encode_prompt_response, pad_and_stack
    from alignment.logprobs import response_token_logps

    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    rows = [encode_prompt_response(tok, "Q: ", r, 64) for r in ("a", "a much longer response here")]
    loader = torch.utils.data.DataLoader(rows, batch_size=1, collate_fn=lambda b: pad_and_stack(b, 0))

    val_loss = evaluate(model, loader, torch.device("cpu"), use_fp16=False)

    batch = pad_and_stack(rows, 0)
    token_logps, mask = response_token_logps(model, batch)
    expected = (-(token_logps * mask).sum() / mask.sum()).item()
    assert val_loss == pytest.approx(expected)
    assert model.training


def test_non_finite_loss_aborts(tmp_path):
    class NanLM(ToyCausalLM):
        def forward(self, input_ids, attention_mask=None, labels=None):
            out = super().forward(input_ids, attention_mask=attention_mask, labels=labels)
            out.logits = out.logits * float("nan")
            return out

    tok = ToyTokenizer()
    with pytest.raises(RuntimeError, match="non-finite"):
        train_sft(
            base_cfg(max_steps=10, eval_every=1000), NanLM(tok.vocab_size), tok,
            fake_rows(40, "train"), fake_rows(4, "val"), tmp_path / "run", identity_format, state_dict_save,
        )
