"""DPO loss and related utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.logprobs import sequence_logps


def dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Return the mean DPO loss over examples.

    z = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
    loss = mean(-log sigmoid(z)), computed with logsigmoid for numerical stability.
    """
    device = policy_chosen.device
    policy_chosen = policy_chosen.float()
    policy_rejected = policy_rejected.float()
    ref_chosen = ref_chosen.to(device).float()
    ref_rejected = ref_rejected.to(device).float()

    z = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(z).mean()


def implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Return detached implicit rewards (r_w, r_l) = beta * (policy - reference) logps."""
    device = policy_chosen.device
    r_w = beta * (policy_chosen.detach().float() - ref_chosen.detach().to(device).float())
    r_l = beta * (policy_rejected.detach().float() - ref_rejected.detach().to(device).float())
    return r_w, r_l


def preference_metrics(r_w, r_l):
    """Return reward accuracy (strict r_w > r_l), mean margin, and mean rewards as floats."""
    return {
        "reward_accuracy": (r_w > r_l).float().mean().item(),
        "reward_margin": (r_w - r_l).mean().item(),
        "chosen_reward": r_w.mean().item(),
        "rejected_reward": r_l.mean().item(),
    }


def dpo_forward(policy, reference, batch, beta):
    """Run policy (with gradients) and frozen reference on a DPO batch.

    Returns the loss, the four per-example log-prob tensors, and preference metrics.
    """
    policy_chosen = sequence_logps(policy, batch["chosen"])
    policy_rejected = sequence_logps(policy, batch["rejected"])
    with torch.no_grad():
        ref_chosen = sequence_logps(reference, batch["chosen"])
        ref_rejected = sequence_logps(reference, batch["rejected"])

    loss = dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    r_w, r_l = implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return {
        "loss": loss,
        "policy_chosen_logps": policy_chosen,
        "policy_rejected_logps": policy_rejected,
        "reference_chosen_logps": ref_chosen,
        "reference_rejected_logps": ref_rejected,
        **preference_metrics(r_w, r_l),
    }
