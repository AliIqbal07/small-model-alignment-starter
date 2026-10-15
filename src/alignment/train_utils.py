"""Shared training utilities."""

from __future__ import annotations

import contextlib
import json
import math
import random
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR


def fp16_autocast(device, enabled):
    """torch.autocast to float16 on the device's type when enabled, else a no-op context."""
    if enabled:
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.float16)
    return contextlib.nullcontext()


def grad_scaler(enabled):
    """CUDA GradScaler (a pass-through when disabled)."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # torch < 2.3


def set_seed(seed):
    """Seed python, numpy (if installed), torch, and CUDA."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(yaml_path, overrides):
    """Load a YAML config; every override whose value is not None replaces the YAML value."""
    with open(yaml_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    return cfg


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup from 0 to the peak lr over warmup_steps, then cosine decay to 0 at total_steps."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def param_groups(model, weight_decay):
    """Split trainable params: no weight decay for biases, norm weights, and any param with ndim < 2."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return [group for group in groups if group["params"]]


class MetricsLogger:
    """Append one JSON object per line and flush after every write."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")

    def log(self, record):
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()


def save_model(model, tokenizer, out_dir):
    """Save with save_pretrained(safe_serialization=True) when available, else a state_dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir, safe_serialization=True)
    else:
        torch.save(model.state_dict(), out_dir / "model.pt")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)
