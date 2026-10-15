"""Tests for the DPO loss, implicit rewards, metrics, and the model-level forward."""

from __future__ import annotations

import copy
import math

import pytest
import torch

import submission_adapter as adapter
from alignment.dpo import dpo_forward, dpo_loss_from_logps, implicit_rewards, preference_metrics
from alignment.logprobs import sequence_logps
from alignment.tokenization import build_dpo_batch, build_sft_batch
from tests.toy_lm import ToyCausalLM, ToyTokenizer

PROMPTS = ["Pick: ", "Choose: ", "Q: "]
CHOSEN = ["abc", "yes", "good answer"]
REJECTED = ["zz", "no no", "bad"]


def t(*values):
    return torch.tensor(values, dtype=torch.float32)


def hand_loss(pc, pr, rc, rr, beta):
    z = beta * ((pc - pr) - (rc - rr))
    return math.log1p(math.exp(-z))


def models(tokenizer):
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.0)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=-0.5)
    return policy, reference


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 2.0])
def test_equal_logps_give_log_two(beta):
    values = t(-2.0, -5.0, -0.5)
    loss = dpo_loss_from_logps(values, values, values, values, beta)
    assert loss.ndim == 0
    torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))


def test_beta_zero_gives_log_two_for_any_logps():
    loss = dpo_loss_from_logps(t(-1.0), t(-9.0), t(-4.0), t(-2.0), 0.0)
    torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))


def test_hand_computed_single_example():
    loss = dpo_loss_from_logps(t(-1.0), t(-3.0), t(-2.0), t(-2.0), 0.5)
    torch.testing.assert_close(loss, torch.tensor(math.log1p(math.exp(-1.0))))
    assert abs(loss.item() - 0.31326) < 1e-5


def test_batch_loss_is_mean_of_per_example_losses():
    pc, pr, rc, rr = [-1.0, -4.0, -2.5], [-3.0, -2.0, -2.5], [-2.0, -3.0, -1.0], [-2.0, -3.5, -4.0]
    beta = 0.3
    expected = sum(hand_loss(*row, beta) for row in zip(pc, pr, rc, rr)) / 3

    loss = dpo_loss_from_logps(t(*pc), t(*pr), t(*rc), t(*rr), beta)

    torch.testing.assert_close(loss, torch.tensor(expected))


def test_loss_direction():
    base = dict(pc=-2.0, pr=-2.0, rc=-2.0, rr=-2.0)

    def loss_with(**changes):
        v = {**base, **changes}
        return dpo_loss_from_logps(t(v["pc"]), t(v["pr"]), t(v["rc"]), t(v["rr"]), 0.5).item()

    baseline = loss_with()
    assert loss_with(pc=-1.0) < baseline
    assert loss_with(pr=-1.0) > baseline
    assert loss_with(rc=-1.0) > baseline


def test_analytic_gradient_wrt_policy_chosen():
    beta = 0.7
    pc = t(-1.0, -3.0, -2.0).requires_grad_()
    pr, rc, rr = t(-2.0, -1.0, -2.5), t(-1.5, -2.0, -2.0), t(-2.0, -2.0, -1.0)

    dpo_loss_from_logps(pc, pr, rc, rr, beta).backward()

    z = beta * ((pc.detach() - pr) - (rc - rr))
    expected = -beta * torch.sigmoid(-z) / len(z)
    torch.testing.assert_close(pc.grad, expected)


def test_extreme_margins_are_stable():
    zero = t(0.0)
    big = dpo_loss_from_logps(t(1000.0), zero, zero, zero, 1.0)
    small = dpo_loss_from_logps(t(-1000.0), zero, zero, zero, 1.0)

    assert torch.isfinite(big) and big.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(small) and small.item() == pytest.approx(1000.0, rel=1e-6)


def test_implicit_rewards_are_detached_and_correct():
    pc = t(-1.0, -2.0).requires_grad_()
    pr = t(-3.0, -1.0).requires_grad_()
    rc, rr = t(-2.0, -2.0), t(-2.0, -2.0)

    r_w, r_l = implicit_rewards(pc, pr, rc, rr, 0.5)

    assert not r_w.requires_grad and not r_l.requires_grad
    torch.testing.assert_close(r_w, t(0.5, 0.0))
    torch.testing.assert_close(r_l, t(-0.5, 0.5))


def test_metrics_values_and_types():
    metrics = preference_metrics(t(1.0, 0.0, 2.0), t(0.0, 1.0, 2.0))

    assert set(metrics) == {"reward_accuracy", "reward_margin", "chosen_reward", "rejected_reward"}
    assert all(type(v) is float for v in metrics.values())
    assert metrics["reward_accuracy"] == pytest.approx(1 / 3)  # tie at index 2 is not a win
    assert metrics["reward_margin"] == pytest.approx(0.0)
    assert metrics["chosen_reward"] == pytest.approx(1.0)
    assert metrics["rejected_reward"] == pytest.approx(1.0)


def test_identical_policy_and_reference_metrics_are_zero():
    logps_c, logps_r = t(-2.0, -5.0), t(-3.0, -1.0)
    metrics = preference_metrics(*implicit_rewards(logps_c, logps_r, logps_c, logps_r, 0.2))

    assert metrics["reward_accuracy"] == 0.0
    assert metrics["reward_margin"] == 0.0


def test_compute_dpo_loss_with_identical_models_is_log_two():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.8)
    reference = copy.deepcopy(policy)

    loss = adapter.compute_dpo_loss(policy, reference, tok, PROMPTS, CHOSEN, REJECTED, 0.2, 32)

    torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))


def test_reference_gets_no_gradient_policy_does():
    tok = ToyTokenizer()
    policy, reference = models(tok)
    assert reference.logit_bias.requires_grad

    loss = adapter.compute_dpo_loss(policy, reference, tok, PROMPTS, CHOSEN, REJECTED, 0.2, 32)
    loss.backward()

    assert reference.logit_bias.grad is None
    assert policy.logit_bias.grad is not None
    assert policy.logit_bias.grad.abs().sum() > 0


def test_compute_dpo_loss_matches_manual_composition():
    tok = ToyTokenizer()
    policy, reference = models(tok)
    beta = 0.3

    loss = adapter.compute_dpo_loss(policy, reference, tok, PROMPTS, CHOSEN, REJECTED, beta, 32)

    pc = sequence_logps(policy, build_sft_batch(tok, PROMPTS, CHOSEN, 32))
    pr = sequence_logps(policy, build_sft_batch(tok, PROMPTS, REJECTED, 32))
    rc = sequence_logps(reference, build_sft_batch(tok, PROMPTS, CHOSEN, 32))
    rr = sequence_logps(reference, build_sft_batch(tok, PROMPTS, REJECTED, 32))
    torch.testing.assert_close(loss, dpo_loss_from_logps(pc, pr, rc, rr, beta))
    torch.testing.assert_close(loss, adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta))


def test_dpo_forward_returns_loss_logps_and_metrics():
    tok = ToyTokenizer()
    policy, reference = models(tok)
    batch = build_dpo_batch(tok, PROMPTS, CHOSEN, REJECTED, 32)

    out = dpo_forward(policy, reference, batch, 0.2)

    expected_keys = {
        "loss",
        "policy_chosen_logps",
        "policy_rejected_logps",
        "reference_chosen_logps",
        "reference_rejected_logps",
        "reward_accuracy",
        "reward_margin",
        "chosen_reward",
        "rejected_reward",
    }
    assert set(out) == expected_keys
    assert out["policy_chosen_logps"].requires_grad
    assert not out["reference_chosen_logps"].requires_grad


def test_one_sgd_step_lowers_loss_and_leaves_reference_unchanged():
    tok = ToyTokenizer()
    policy, reference = models(tok)
    reference_before = reference.logit_bias.detach().clone()
    batch = build_dpo_batch(tok, PROMPTS, CHOSEN, REJECTED, 32)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)

    loss_before = dpo_forward(policy, reference, batch, 0.2)["loss"]
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    with torch.no_grad():
        loss_after = dpo_forward(policy, reference, batch, 0.2)["loss"]

    assert loss_after < loss_before.detach()
    assert torch.equal(reference.logit_bias.detach(), reference_before)


def test_row_order_of_rewards_follows_inputs():
    tok = ToyTokenizer()
    policy, reference = models(tok)
    beta = 0.2

    def rewards(prompts, chosen, rejected):
        out = dpo_forward(policy, reference, build_dpo_batch(tok, prompts, chosen, rejected, 32), beta)
        return implicit_rewards(
            out["policy_chosen_logps"],
            out["policy_rejected_logps"],
            out["reference_chosen_logps"],
            out["reference_rejected_logps"],
            beta,
        )

    r_w, r_l = rewards(PROMPTS, CHOSEN, REJECTED)
    r_w_rev, r_l_rev = rewards(PROMPTS[::-1], CHOSEN[::-1], REJECTED[::-1])

    torch.testing.assert_close(r_w_rev, r_w.flip(0))
    torch.testing.assert_close(r_l_rev, r_l.flip(0))
    for i, (chosen, rejected) in enumerate(zip(CHOSEN, REJECTED)):
        single_w, single_l = rewards([PROMPTS[i]], [chosen], [rejected])
        torch.testing.assert_close(r_w[i], single_w[0])
        torch.testing.assert_close(r_l[i], single_l[0])
