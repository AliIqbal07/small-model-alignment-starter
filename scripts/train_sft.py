"""Supervised fine-tuning of Qwen2.5-0.5B with a plain PyTorch loop.

Example (Kaggle T4):
    python scripts/train_sft.py --config configs/sft.yaml --output_dir outputs/sft
Pilot:
    python scripts/train_sft.py --output_dir outputs/sft_pilot --max_steps 20 \\
        --limit_train 1000 --limit_val 100 --eval_every 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.formatting import format_prompt  # noqa: E402
from alignment.io_utils import read_jsonl  # noqa: E402
from alignment.sft_trainer import train_sft  # noqa: E402
from alignment.train_utils import load_config, save_model, set_seed  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Response-only SFT (no HF Trainer).")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--limit_train", type=int, help="use only the first N train rows")
    parser.add_argument("--limit_val", type=int, help="use only the first N val rows")
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--device")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "eval_every": args.eval_every,
        "device": args.device,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    cfg = load_config(args.config, overrides)
    set_seed(cfg["seed"])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32)
    model.to(cfg["device"])
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_rows = read_jsonl(cfg["train_file"])[: cfg.get("limit_train")]
    val_rows = read_jsonl(cfg["val_file"])[: cfg.get("limit_val")]
    print(f"Loaded {len(train_rows)} train / {len(val_rows)} val rows")

    result = train_sft(cfg, model, tokenizer, train_rows, val_rows, cfg["output_dir"], format_prompt, save_model)
    if result is not None:
        print(f"Done: {result}")


if __name__ == "__main__":
    main()
