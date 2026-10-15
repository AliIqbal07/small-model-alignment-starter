"""Supervised fine-tuning loop (plain PyTorch, no HF Trainer)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from alignment.logprobs import response_token_logps, sft_loss
from alignment.tokenization import IGNORE_INDEX, encode_prompt_response, pad_and_stack, resolve_pad_id
from alignment.train_utils import MetricsLogger, cosine_with_warmup, fp16_autocast, grad_scaler, param_groups

MAX_NONFINITE_STREAK = 5


def _decode(tokenizer, ids):
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids, skip_special_tokens=False)
    return str(ids)


def _show_example(tokenizer, ids, labels):
    labelled = [token for token, label in zip(ids, labels) if label != IGNORE_INDEX]
    print("=== Example input (full) ===")
    print(_decode(tokenizer, ids))
    print("=== Example labelled tokens only (what the loss sees) ===")
    print(_decode(tokenizer, labelled))
    print(f"=== {len(ids)} tokens, {len(labelled)} labelled ===")


@torch.no_grad()
def evaluate(model, loader, device, use_fp16):
    """Return total response-token NLL / total response tokens over the whole loader."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        with fp16_autocast(device, use_fp16):
            token_logps, mask = response_token_logps(model, batch)
        total_nll += -(token_logps * mask).sum().item()
        total_tokens += int(mask.sum().item())
    model.train()
    return total_nll / max(1, total_tokens)


def train_sft(cfg, model, tokenizer, train_rows, val_rows, output_dir, format_fn, save_fn):
    """Train with response-only SFT loss; returns a summary dict, or None if already finished.

    cfg keys: max_length, max_prompt_length, batch_size, grad_accum, lr, warmup_ratio,
    weight_decay, max_grad_norm, epochs, max_steps (optional), eval_every, log_every,
    seed, device, fp16. fp16 mixed precision is used only on CUDA; weights stay fp32.
    """
    output_dir = Path(output_dir)
    if (output_dir / "final").exists():
        print(f"{output_dir / 'final'} already exists; skipping training.")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(cfg), handle, sort_keys=True)

    device = torch.device(cfg["device"])
    use_fp16 = bool(cfg.get("fp16")) and device.type == "cuda"
    model.to(device)
    model.train()

    def encode(rows):
        return [
            encode_prompt_response(
                tokenizer,
                format_fn(tokenizer, row["instruction"]),
                row["response"],
                cfg["max_length"],
                cfg.get("max_prompt_length"),
            )
            for row in rows
        ]

    train_data, val_data = encode(train_rows), encode(val_rows)
    _show_example(tokenizer, *train_data[0])
    no_label_rows = sum(1 for _, labels in train_data if all(l == IGNORE_INDEX for l in labels))
    if no_label_rows:
        print(f"Warning: {no_label_rows} train rows have no labelled response tokens after truncation.")

    pad_id = resolve_pad_id(tokenizer)
    collate = lambda rows: pad_and_stack(rows, pad_id)  # noqa: E731
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_data, batch_size=cfg["batch_size"], shuffle=True, generator=generator, collate_fn=collate
    )
    val_loader = DataLoader(val_data, batch_size=cfg.get("eval_batch_size", cfg["batch_size"]), collate_fn=collate)

    grad_accum = cfg["grad_accum"]
    num_batches = len(train_loader)
    steps_per_epoch = math.ceil(num_batches / grad_accum)
    total_steps = cfg.get("max_steps") or steps_per_epoch * cfg["epochs"]
    warmup_steps = math.ceil(cfg["warmup_ratio"] * total_steps)

    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = grad_scaler(use_fp16)
    logger = MetricsLogger(output_dir / "metrics.jsonl")
    print(
        f"SFT: {len(train_data)} train / {len(val_data)} val rows, {num_batches} micro-batches/epoch, "
        f"{total_steps} optimizer steps (warmup {warmup_steps}), fp16={use_fp16}"
    )

    start = time.time()
    best_val = math.inf
    step = 0
    tokens_seen = 0
    window_losses = []
    nonfinite_streak = 0
    last_eval_step = None

    def run_eval():
        nonlocal best_val, last_eval_step
        val_loss = evaluate(model, val_loader, device, use_fp16)
        last_eval_step = step
        logger.log({"type": "val", "step": step, "val_loss": val_loss, "elapsed_s": time.time() - start})
        print(f"[step {step}] val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_fn(model, tokenizer, output_dir / "best")
            with open(output_dir / "best.json", "w", encoding="utf-8") as handle:
                json.dump({"step": step, "val_loss": val_loss}, handle)

    run_eval()
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < total_steps:
            for batch_index, batch in enumerate(train_loader):
                # The final group of an epoch may be smaller than grad_accum.
                group_start = batch_index - batch_index % grad_accum
                group_size = min(grad_accum, num_batches - group_start)

                with fp16_autocast(device, use_fp16):
                    loss = sft_loss(model, batch)
                if torch.isfinite(loss):
                    nonfinite_streak = 0
                    window_losses.append(loss.item())
                    scaler.scale(loss / group_size).backward()
                else:
                    nonfinite_streak += 1
                    if nonfinite_streak >= MAX_NONFINITE_STREAK:
                        raise RuntimeError(
                            f"Loss was non-finite for {MAX_NONFINITE_STREAK} consecutive micro-steps "
                            f"(at optimizer step {step}); aborting."
                        )
                tokens_seen += int(batch["attention_mask"].sum().item())

                if batch_index + 1 - group_start < group_size:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % cfg["log_every"] == 0:
                    record = {
                        "type": "train",
                        "step": step,
                        "train_loss": sum(window_losses) / max(1, len(window_losses)),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "loss_scale": scaler.get_scale() if use_fp16 else 1.0,
                        "tokens_seen": tokens_seen,
                        "elapsed_s": time.time() - start,
                    }
                    if device.type == "cuda":
                        record["max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    logger.log(record)
                    print(json.dumps(record))
                    window_losses = []

                if step % cfg["eval_every"] == 0:
                    run_eval()
                if step >= total_steps:
                    break

        if last_eval_step != step:
            run_eval()
        save_fn(model, tokenizer, output_dir / "final")
    finally:
        logger.close()

    return {"steps": step, "best_val_loss": best_val, "elapsed_s": time.time() - start}
