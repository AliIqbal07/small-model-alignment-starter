"""DPO training on top of the SFT checkpoint with a plain PyTorch loop.

Example (Kaggle T4 x2):
    python scripts/train_dpo.py --config configs/dpo_hh.yaml
Pilot:
    python scripts/train_dpo.py --config configs/dpo_hh.yaml --output_dir outputs/dpo_hh_pilot \\
        --max_steps 10 --limit_train 400 --limit_val 100 --eval_every 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.dpo_trainer import train_dpo  # noqa: E402
from alignment.formatting import format_prompt  # noqa: E402
from alignment.io_utils import read_jsonl  # noqa: E402
from alignment.train_utils import load_config, save_model, set_seed  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DPO from scratch (no HF Trainer / TRL).")
    parser.add_argument("--config", default="configs/dpo_hh.yaml")
    parser.add_argument("--output_dir")
    parser.add_argument("--sft_model_path")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--limit_train", type=int, help="use only the first N train pairs")
    parser.add_argument("--limit_val", type=int, help="use only the first N val pairs")
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "output_dir": args.output_dir,
        "sft_model_path": args.sft_model_path,
        "max_steps": args.max_steps,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "eval_every": args.eval_every,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    cfg = load_config(args.config, overrides)
    set_seed(cfg["seed"])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"float32": torch.float32, "float16": torch.float16}
    path = cfg["sft_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right")
    policy = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(cfg["policy_device"])
    reference = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtypes[cfg["reference_dtype"]])
    reference.to(cfg["reference_device"])
    if cfg.get("gradient_checkpointing"):
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    train_rows = read_jsonl(cfg["train_file"])[: cfg.get("limit_train")]
    val_rows = read_jsonl(cfg["val_file"])[: cfg.get("limit_val")]
    print(f"Loaded {len(train_rows)} train / {len(val_rows)} val pairs")

    result = train_dpo(
        cfg, policy, reference, tokenizer, train_rows, val_rows, cfg["output_dir"], format_prompt, save_model
    )
    if result is not None:
        print(f"Done: {result}")


if __name__ == "__main__":
    main()
