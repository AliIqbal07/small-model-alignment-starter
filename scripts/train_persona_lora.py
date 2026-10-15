"""Persona track (Task 5): DPO with TRL's DPOTrainer + a PEFT LoRA adapter.

Runs on Kaggle (needs trl, peft, transformers, datasets). Example:
    python scripts/train_persona_lora.py --config configs/persona_lora.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from alignment.io_utils import read_json, write_json, write_text  # noqa: E402
from alignment.persona import split_rows, to_trl_conversational  # noqa: E402
from alignment.train_utils import load_config, set_seed  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Persona DPO with TRL + LoRA.")
    parser.add_argument("--config", default="configs/persona_lora.yaml")
    parser.add_argument("--output_dir")
    return parser.parse_args(argv)


def build_dpo_config(DPOConfig, cfg, output_dir):
    """DPOConfig from cfg, keeping only arguments this TRL version accepts (dropped ones are printed)."""
    wanted = {
        "output_dir": str(output_dir),
        "beta": cfg["beta"],
        "learning_rate": cfg["learning_rate"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "num_train_epochs": cfg["num_train_epochs"],
        "max_length": cfg["max_length"],
        "max_prompt_length": cfg["max_prompt_length"],
        "logging_steps": cfg["logging_steps"],
        "eval_strategy": "steps",
        "eval_steps": cfg["eval_steps"],
        "save_strategy": "no",
        "fp16": True,  # T4: fp32 weights + fp16 autocast via TrainingArguments
        "seed": cfg["seed"],
        "report_to": "none",
    }
    accepted = {field.name for field in dataclasses.fields(DPOConfig)}
    if "eval_strategy" not in accepted and "evaluation_strategy" in accepted:  # older transformers
        wanted["evaluation_strategy"] = wanted.pop("eval_strategy")
    dropped = sorted(set(wanted) - accepted)
    if dropped:
        print(f"WARNING: this TRL version's DPOConfig does not accept {dropped}; they are ignored.")
    return DPOConfig(**{key: value for key, value in wanted.items() if key in accepted})


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, {"output_dir": args.output_dir})
    output_dir = Path(cfg["output_dir"])
    if (output_dir / "adapter").exists():
        print(f"{output_dir / 'adapter'} already exists; skipping training.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "config_resolved.yaml", yaml.safe_dump(dict(cfg), sort_keys=True))
    set_seed(cfg["seed"])

    import datasets
    import peft
    import torch
    import transformers
    import trl
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    print(f"trl {trl.__version__}, peft {peft.__version__}, transformers {transformers.__version__}, "
          f"datasets {datasets.__version__}, torch {torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32)

    train_rows, eval_rows = split_rows(read_json(cfg["data_path"]), n_eval=cfg["n_eval"], seed=cfg["seed"])
    train_dataset = datasets.Dataset.from_list([to_trl_conversational(row) for row in train_rows])
    eval_dataset = datasets.Dataset.from_list([to_trl_conversational(row) for row in eval_rows])
    print(f"{len(train_dataset)} train / {len(eval_dataset)} eval preference pairs")

    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=list(cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    dpo_config = build_dpo_config(DPOConfig, cfg, output_dir)

    # The tokenizer argument was renamed from `tokenizer` to `processing_class` in newer TRL.
    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    tokenizer_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = DPOTrainer(
        model=model,
        # ref_model=None together with peft_config: TRL uses the same base model with the LoRA
        # adapter disabled as the frozen reference, so no second copy of the model is loaded.
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        **{tokenizer_kwarg: tokenizer},
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        print(f"trainable params: {trainable:,} / {total:,} ({trainable / total:.2%})")

    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(output_dir / "log_history.json", trainer.state.log_history)
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
