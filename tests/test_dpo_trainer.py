"""CPU smoke tests for the DPO training loop using the toy model and tokenizer."""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from alignment.dpo import dpo_forward, dpo_loss_from_logps, implicit_rewards
from alignment.dpo_trainer import collate_pairs, evaluate, train_dpo
from alignment.tokenization import encode_prompt_response
from tests.toy_lm import ToyCausalLM, ToyTokenizer

LOG2 = math.log(2.0)


def identity_format(tokenizer, instruction):
    return instruction


def state_dict_save(model, tokenizer, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")


def fake_pairs(count, prefix):
    return [
        {"id": f"{prefix}-{i}", "instruction": f"Q{i % 4}: ", "chosen": "aaab a" * (1 + i % 2), "rejected": "bbba b" * (1 + i % 2)}
        for i in range(count)
    ]


def base_cfg(**changes):
    cfg = {
        "beta": 0.1,
        "max_length": 48,
        "max_prompt_length": 16,
        "batch_size": 1,
        "eval_batch_size": 3,
        "grad_accum": 2,
        "optimizer": "adamw",
        "lr": 0.5,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "epochs": 1,
        "max_steps": 4,
        "eval_every": 2,
        "log_every": 1,
        "seed": 0,
        "policy_device": "cpu",
        "reference_device": "cpu",
        "fp16": True,  # must be ignored on CPU
    }
    cfg.update(changes)
    return cfg


def models():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    reference = copy.deepcopy(policy)
    return tok, policy, reference


def run(tmp_path, train_rows=None, val_rows=None, **changes):
    tok, policy, reference = models()
    result = train_dpo(
        base_cfg(**changes),
        policy,
        reference,
        tok,
        fake_pairs(16, "train") if train_rows is None else train_rows,
        fake_pairs(6, "val") if val_rows is None else val_rows,
        tmp_path / "run",
        identity_format,
        state_dict_save,
    )
    return result, tok, policy, reference


def read_metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_step_zero_matches_reference_and_training_improves(tmp_path, capsys):
    result, *_ = run(tmp_path)

    val = [m for m in read_metrics(tmp_path / "run" / "metrics.jsonl") if m["type"] == "val"]
    assert [m["step"] for m in val] == [0, 2, 4]
    assert val[0]["loss"] == pytest.approx(LOG2, abs=1e-6)
    assert val[0]["reward_accuracy"] == 0.0
    assert val[-1]["loss"] < LOG2
    assert val[-1]["reward_accuracy"] > 0.5
    assert val[-1]["reward_margin"] > 0
    assert "expected ~0.6931 and 0.0" in capsys.readouterr().out
    assert result["steps"] == 4


def test_reference_is_frozen_and_never_gets_grad(tmp_path):
    tok, policy, reference = models()
    before = {name: p.detach().clone() for name, p in reference.named_parameters()}
    policy_before = policy.logit_bias.detach().clone()

    train_dpo(base_cfg(), policy, reference, tok, fake_pairs(16, "t"), fake_pairs(6, "v"),
              tmp_path / "run", identity_format, state_dict_save)

    for name, param in reference.named_parameters():
        assert torch.equal(param.detach(), before[name])
        assert param.grad is None
        assert not param.requires_grad
    assert not reference.training
    assert not torch.equal(policy.logit_bias.detach(), policy_before)


def test_outputs_written_and_second_call_returns_early(tmp_path, capsys):
    run(tmp_path)
    out_dir = tmp_path / "run"
    for name in ("best", "final"):
        assert (out_dir / name / "model.pt").exists()
    assert (out_dir / "config_resolved.yaml").exists()

    metrics = read_metrics(out_dir / "metrics.jsonl")
    train = [m for m in metrics if m["type"] == "train"]
    assert [m["step"] for m in train] == [1, 2, 3, 4]
    expected_train_keys = {
        "loss", "reward_accuracy", "reward_margin", "chosen_reward", "rejected_reward",
        "policy_chosen_logps", "policy_rejected_logps", "lr", "grad_norm", "loss_scale", "elapsed_s",
    }
    assert all(expected_train_keys <= set(m) for m in train)
    assert all(math.isfinite(m["loss"]) and math.isfinite(m["grad_norm"]) for m in train)

    best = json.loads((out_dir / "best.json").read_text())
    val = [m for m in metrics if m["type"] == "val"]
    top = max(val, key=lambda m: (m["reward_accuracy"], m["reward_margin"]))
    assert best["step"] == top["step"]
    assert best["reward_accuracy"] == top["reward_accuracy"]

    before = (out_dir / "metrics.jsonl").read_text()
    capsys.readouterr()
    result, *_ = run(tmp_path)
    assert result is None
    assert "already exists" in capsys.readouterr().out
    assert (out_dir / "metrics.jsonl").read_text() == before


def test_pairs_without_response_tokens_are_dropped(tmp_path, capsys):
    long_prompt = {"id": "long", "instruction": "x" * 60, "chosen": "a", "rejected": "b"}
    result, *_ = run(tmp_path, train_rows=fake_pairs(16, "t") + [long_prompt], max_prompt_length=None)

    assert result["train_pairs"] == 16
    assert result["dropped_train_pairs"] == 1
    assert "dropped 1" in capsys.readouterr().out.lower()


def test_evaluate_aggregates_over_all_pairs():
    tok = ToyTokenizer()
    torch.manual_seed(0)
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=-0.2)
    rows = fake_pairs(5, "v")
    pairs = [
        (
            encode_prompt_response(tok, r["instruction"], r["chosen"], 48),
            encode_prompt_response(tok, r["instruction"], r["rejected"], 48),
        )
        for r in rows
    ]
    loader = torch.utils.data.DataLoader(pairs, batch_size=3, collate_fn=lambda b: collate_pairs(b, 0))
    beta = 0.1

    metrics = evaluate(policy, reference, loader, beta, use_fp16=False)

    per_pair = [dpo_forward(policy, reference, collate_pairs([p], 0), beta) for p in pairs]
    logps = [
        torch.cat([out[key].detach() for out in per_pair])
        for key in ("policy_chosen_logps", "policy_rejected_logps", "reference_chosen_logps", "reference_rejected_logps")
    ]
    r_w, r_l = implicit_rewards(*logps, beta)
    assert metrics["loss"] == pytest.approx(dpo_loss_from_logps(*logps, beta).item())
    assert metrics["reward_accuracy"] == pytest.approx((r_w > r_l).float().mean().item())
    assert metrics["reward_margin"] == pytest.approx((r_w - r_l).mean().item())
    assert metrics["chosen_reward"] == pytest.approx(r_w.mean().item())
    assert metrics["rejected_reward"] == pytest.approx(r_l.mean().item())
    assert policy.training  # evaluate restores train mode


def test_collate_pairs_builds_chosen_and_rejected_batches():
    tok = ToyTokenizer()
    pairs = [
        (encode_prompt_response(tok, "Q: ", "aa", 32), encode_prompt_response(tok, "Q: ", "b", 32)),
        (encode_prompt_response(tok, "Long Q: ", "a", 32), encode_prompt_response(tok, "Long Q: ", "bbbb", 32)),
    ]
    batch = collate_pairs(pairs, 0)
    assert set(batch) == {"chosen", "rejected"}
    assert batch["chosen"]["input_ids"].shape[0] == batch["rejected"]["input_ids"].shape[0] == 2


def test_rmsprop_with_warmup_runs(tmp_path):
    result, *_ = run(tmp_path, optimizer="rmsprop", lr=0.01, warmup_steps=2)
    metrics = read_metrics(tmp_path / "run" / "metrics.jsonl")
    lrs = [m["lr"] for m in metrics if m["type"] == "train"]
    assert lrs == pytest.approx([0.005, 0.01, 0.01, 0.01])
    assert result["steps"] == 4


def test_unknown_optimizer_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="optimizer"):
        run(tmp_path, optimizer="sgd")
