"""Generate greedy outputs for an eval set with one model.

Example:
    python scripts/generate.py --model_path outputs/sft/best --generator_name sft \\
        --eval_set eval_sets/alpaca_eval_300_seed42.json --output outputs/gen/sft_alpaca.json
With a LoRA adapter (persona task):
    python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct \\
        --adapter_path outputs/persona_lora/adapter --generator_name persona_lora ...
Sanity check (prints prompts and outputs, writes nothing):
    python scripts/generate.py ... --limit 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.formatting import format_prompt  # noqa: E402
from alignment.generation import generate_outputs, make_hf_generate_fn  # noqa: E402
from alignment.io_utils import read_json, write_json  # noqa: E402
from alignment.train_utils import load_config  # noqa: E402

DECODING_KEYS = ("max_new_tokens", "repetition_penalty", "no_repeat_ngram_size", "batch_size")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Greedy generation for an eval set.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--generator_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/generate.yaml")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--limit", type=int, help="sanity mode: first N rows, print them, write nothing")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter_path", help="optional LoRA adapter, merged into the base model")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and args.limit is None:
        print(f"{output} already exists; skipping.")
        return
    cfg = load_config(args.config, {"batch_size": args.batch_size})
    rows = read_json(args.eval_set)
    if args.limit is not None:
        rows = rows[: args.limit]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
        print(f"Merged LoRA adapter from {args.adapter_path}")
    model.to(args.device).eval()
    generate_batch = make_hf_generate_fn(model, tokenizer, cfg)

    start = time.time()
    if args.limit is not None:
        outputs = generate_outputs(rows, generate_batch, cfg["batch_size"], args.generator_name)
        for out in outputs:
            print("=" * 80)
            print(format_prompt(tokenizer, out["instruction"]))
            print("-" * 80)
            print(out["output"])
            print(f"[{out['num_new_tokens']} new tokens, hit_stop={out['hit_stop']}]")
        print("Sanity mode: nothing written.")
        return

    partial = output.with_name(output.name + ".partial.jsonl")
    outputs = generate_outputs(rows, generate_batch, cfg["batch_size"], args.generator_name, partial)
    elapsed = time.time() - start

    write_json(output, outputs)
    meta = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "generator_name": args.generator_name,
        "eval_set": args.eval_set,
        "decoding": {"do_sample": False, **{key: cfg[key] for key in DECODING_KEYS}},
        "num_rows": len(outputs),
        "num_hit_stop": sum(o["hit_stop"] for o in outputs),
        "frac_hit_stop": sum(o["hit_stop"] for o in outputs) / max(1, len(outputs)),
        "mean_num_new_tokens": sum(o["num_new_tokens"] for o in outputs) / max(1, len(outputs)),
        "elapsed_s": elapsed,
    }
    write_json(output.with_suffix(".meta.json"), meta)
    partial.unlink(missing_ok=True)
    print(f"Wrote {len(outputs)} outputs to {output} ({meta['frac_hit_stop']:.1%} hit a stop token)")


if __name__ == "__main__":
    main()
