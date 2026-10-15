"""DPO training loop (plain PyTorch, no HF Trainer / TRL)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from alignment.dpo import dpo_forward, dpo_loss_from_logps, implicit_rewards, preference_metrics
from alignment.tokenization import IGNORE_INDEX, encode_prompt_response, pad_and_stack, resolve_pad_id
from alignment.train_utils import MetricsLogger, fp16_autocast, grad_scaler

MAX_NONFINITE_STREAK = 5
LOGP_KEYS = ("policy_chosen_logps", "policy_rejected_logps", "reference_chosen_logps", "reference_rejected_logps")
WINDOW_KEYS = ("loss", "reward_accuracy", "reward_margin", "chosen_reward", "rejected_reward",
               "policy_chosen_logps", "policy_rejected_logps")


def collate_pairs(pairs, pad_id):
    """[(chosen_row, rejected_row), ...] -> {"chosen": batch, "rejected": batch}."""
    return {
        "chosen": pad_and_stack([chosen for chosen, _ in pairs], pad_id),
        "rejected": pad_and_stack([rejected for _, rejected in pairs], pad_id),
    }


def _model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def evaluate(policy, reference, loader, beta, use_fp16=False):
    """DPO loss and reward metrics over ALL val pairs (per-example values concatenated)."""
    policy.eval()
    device = _model_device(policy)
    collected = {key: [] for key in LOGP_KEYS}
    for batch in loader:
        with fp16_autocast(device, use_fp16):
            out = dpo_forward(policy, reference, batch, beta)
        for key in LOGP_KEYS:
            collected[key].append(out[key].float().to(device))
    policy.train()

    logps = [torch.cat(collected[key]) for key in LOGP_KEYS]
    metrics = preference_metrics(*implicit_rewards(*logps, beta))
    return {"loss": dpo_loss_from_logps(*logps, beta).item(), **metrics}


def _decode(tokenizer, ids):
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids, skip_special_tokens=False)
    return str(ids)


def _labelled(row):
    ids, labels = row
    return [token for token, label in zip(ids, labels) if label != IGNORE_INDEX]


def _show_example(tokenizer, pair):
    (chosen_ids, chosen_labels), rejected = pair
    prompt = [token for token, label in zip(chosen_ids, chosen_labels) if label == IGNORE_INDEX]
    print("=== Example prompt (unlabelled tokens) ===")
    print(_decode(tokenizer, prompt))
    print("=== Example chosen (labelled tokens) ===")
    print(_decode(tokenizer, _labelled(pair[0])))
    print("=== Example rejected (labelled tokens) ===")
    print(_decode(tokenizer, _labelled(rejected)))


def _encode_pairs(name, rows, tokenizer, format_fn, cfg):
    """Tokenize (chosen, rejected) per row; drop pairs with no labelled tokens on either side."""
    eos = tokenizer.eos_token_id
    pairs, dropped, chosen_truncated, rejected_truncated = [], 0, 0, 0
    for row in rows:
        prompt = format_fn(tokenizer, row["instruction"])
        chosen, rejected = (
            encode_prompt_response(tokenizer, prompt, row[side], cfg["max_length"], cfg.get("max_prompt_length"))
            for side in ("chosen", "rejected")
        )
        if eos is not None:
            chosen_truncated += eos not in _labelled(chosen)
            rejected_truncated += eos not in _labelled(rejected)
        if not _labelled(chosen) or not _labelled(rejected):
            dropped += 1
            continue
        pairs.append((chosen, rejected))
    print(
        f"{name}: kept {len(pairs)} pairs, dropped {dropped} with no response tokens; "
        f"truncated (no EOS label): {chosen_truncated} chosen, {rejected_truncated} rejected"
    )
    return pairs, dropped


def _make_optimizer(cfg, params):
    name = cfg.get("optimizer", "rmsprop").lower()
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg["lr"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"Unknown optimizer {name!r}; use 'rmsprop' or 'adamw'")


def _constant_with_warmup(optimizer, warmup_steps):
    if warmup_steps <= 0:
        return LambdaLR(optimizer, lambda step: 1.0)
    return LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))


def train_dpo(cfg, policy, reference, tokenizer, train_rows, val_rows, output_dir, format_fn, save_fn):
    """Train the policy with the DPO loss against a frozen reference.

    Returns a summary dict, or None if output_dir/final already exists. fp16 mixed precision
    (autocast for both models, GradScaler for the policy) is used only when the policy is on CUDA.
    Best checkpoint = highest val reward_accuracy, ties broken by higher reward_margin.
    """
    output_dir = Path(output_dir)
    if (output_dir / "final").exists():
        print(f"{output_dir / 'final'} already exists; skipping training.")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(cfg), handle, sort_keys=True)

    policy_device = torch.device(cfg["policy_device"])
    policy.to(policy_device)
    policy.train()
    reference.to(torch.device(cfg["reference_device"]))
    reference.eval()
    reference.requires_grad_(False)
    use_fp16 = bool(cfg.get("fp16")) and policy_device.type == "cuda"
    beta = cfg["beta"]

    train_pairs, dropped_train = _encode_pairs("train", train_rows, tokenizer, format_fn, cfg)
    val_pairs, _ = _encode_pairs("val", val_rows, tokenizer, format_fn, cfg)
    _show_example(tokenizer, train_pairs[0])

    pad_id = resolve_pad_id(tokenizer)
    collate = lambda pairs: collate_pairs(pairs, pad_id)  # noqa: E731
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_pairs, batch_size=cfg.get("batch_size", 1), shuffle=True, generator=generator, collate_fn=collate
    )
    val_loader = DataLoader(val_pairs, batch_size=cfg.get("eval_batch_size", 1), collate_fn=collate)

    grad_accum = cfg["grad_accum"]
    num_batches = len(train_loader)
    total_steps = cfg.get("max_steps") or math.ceil(num_batches / grad_accum) * cfg["epochs"]
    trainable = [param for param in policy.parameters() if param.requires_grad]
    optimizer = _make_optimizer(cfg, trainable)
    scheduler = _constant_with_warmup(optimizer, cfg.get("warmup_steps", 0))
    scaler = grad_scaler(use_fp16)
    logger = MetricsLogger(output_dir / "metrics.jsonl")
    cuda_devices = sorted(
        {d for d in (policy_device, torch.device(cfg["reference_device"])) if d.type == "cuda"}, key=str
    )
    print(
        f"DPO: {len(train_pairs)} train / {len(val_pairs)} val pairs, {num_batches} micro-batches/epoch, "
        f"{total_steps} optimizer steps, beta={beta}, fp16={use_fp16}"
    )

    start = time.time()
    best_key = None
    step = 0
    window = {key: [] for key in WINDOW_KEYS}
    nonfinite_streak = 0
    last_eval_step = None

    def run_eval():
        nonlocal best_key, last_eval_step
        metrics = evaluate(policy, reference, val_loader, beta, use_fp16)
        last_eval_step = step
        logger.log({"type": "val", "step": step, **metrics, "elapsed_s": time.time() - start})
        print(f"[step {step}] val " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if step == 0:
            print(
                f"Step-0 val loss {metrics['loss']:.4f}, reward_accuracy {metrics['reward_accuracy']:.4f} "
                "(expected ~0.6931 and 0.0 when policy == reference)"
            )
        key = (metrics["reward_accuracy"], metrics["reward_margin"])
        if best_key is None or key > best_key:
            best_key = key
            save_fn(policy, tokenizer, output_dir / "best")
            with open(output_dir / "best.json", "w", encoding="utf-8") as handle:
                json.dump({"step": step, **metrics}, handle)

    run_eval()
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < total_steps:
            for batch_index, batch in enumerate(train_loader):
                # The final group of an epoch may be smaller than grad_accum.
                group_start = batch_index - batch_index % grad_accum
                group_size = min(grad_accum, num_batches - group_start)

                with fp16_autocast(policy_device, use_fp16):
                    out = dpo_forward(policy, reference, batch, beta)
                loss = out["loss"]
                if torch.isfinite(loss):
                    nonfinite_streak = 0
                    window["loss"].append(loss.item())
                    for key in ("reward_accuracy", "reward_margin", "chosen_reward", "rejected_reward"):
                        window[key].append(out[key])
                    window["policy_chosen_logps"].append(out["policy_chosen_logps"].mean().item())
                    window["policy_rejected_logps"].append(out["policy_rejected_logps"].mean().item())
                    scaler.scale(loss / group_size).backward()
                else:
                    nonfinite_streak += 1
                    if nonfinite_streak >= MAX_NONFINITE_STREAK:
                        raise RuntimeError(
                            f"Loss was non-finite for {MAX_NONFINITE_STREAK} consecutive micro-steps "
                            f"(at optimizer step {step}); aborting."
                        )

                if batch_index + 1 - group_start < group_size:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % cfg["log_every"] == 0:
                    record = {
                        "type": "train",
                        "step": step,
                        **{key: sum(vals) / max(1, len(vals)) for key, vals in window.items()},
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm),
                        "loss_scale": scaler.get_scale() if use_fp16 else 1.0,
                        "elapsed_s": time.time() - start,
                    }
                    for device in cuda_devices:
                        record[f"max_mem_gb_{device}"] = torch.cuda.max_memory_allocated(device) / 1e9
                    logger.log(record)
                    print(json.dumps(record))
                    window = {key: [] for key in WINDOW_KEYS}

                if step % cfg["eval_every"] == 0:
                    run_eval()
                if step >= total_steps:
                    break

        if last_eval_step != step:
            run_eval()
        save_fn(policy, tokenizer, output_dir / "final")
    finally:
        logger.close()

    return {
        "steps": step,
        "train_pairs": len(train_pairs),
        "dropped_train_pairs": dropped_train,
        "best_reward_accuracy": best_key[0],
        "best_reward_margin": best_key[1],
        "elapsed_s": time.time() - start,
    }
