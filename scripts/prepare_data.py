"""Prepare SFT, preference, and AlpacaEval data as raw text (no chat template).

Runs on Kaggle (needs internet, `datasets`, `transformers`, `huggingface_hub`).
Those libraries are imported inside functions so this module imports offline.

Example:
    python scripts/prepare_data.py --out_dir outputs/data --only all
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.data_eval import sample_alpaca_eval  # noqa: E402
from alignment.data_pref import parse_hh_pair, ultrafeedback_row_to_example  # noqa: E402
from alignment.data_sft import ultrachat_row_to_example  # noqa: E402
from alignment.data_utils import fill_val, percentiles, select_examples  # noqa: E402
from alignment.io_utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402

MAX_PROMPT_TOKENS = 256
MAX_SFT_TOKENS = 512
ALPACA_N = 300
ALPACA_REPO = "tatsu-lab/alpaca_eval"
ALPACA_FILE = "alpaca_eval.json"
SAMPLE_CHARS = 300


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dir", default="outputs/data")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_train", type=int, default=30000)
    parser.add_argument("--sft_val", type=int, default=1000)
    parser.add_argument("--pref_train", type=int, default=15000)
    parser.add_argument("--pref_val", type=int, default=1000)
    parser.add_argument("--only", choices=["all", "sft", "hh", "uf", "alpaca"], default="all")
    return parser.parse_args(argv)


@functools.lru_cache(maxsize=None)
def get_tokenizer(name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate(row):
    return {
        key: value[:SAMPLE_CHARS] + "..." if isinstance(value, str) and len(value) > SAMPLE_CHARS else value
        for key, value in row.items()
    }


class StatsFile:
    """data_stats.json, updated and saved after every output so partial runs keep their stats."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = read_json(self.path) if self.path.exists() else {}

    def record(self, name, raw, parsed, filtered, final, lengths, **extra):
        stats = {
            "raw_rows": raw,
            "parsed_rows": parsed,
            "length_filtered_rows": filtered,
            "final_rows": len(final),
            "samples": [_truncate(row) for row in final[:2]],
            "token_length_percentiles": {key: percentiles(values) for key, values in lengths.items()},
            **extra,
        }
        print(f"\n=== {name} ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        self.data[name] = stats
        write_json(self.path, self.data)


def _length_lists(examples, token_lengths):
    lengths = {}
    for example in examples:
        for key, value in token_lengths(example).items():
            lengths.setdefault(key, []).append(value)
    return lengths


def build_train_val(
    prefix, out_dir, load_train, load_val, parse_row, keep, token_lengths, n_train, n_val, seed, stats
):
    """Build {prefix}_train.jsonl then {prefix}_val.jsonl; skip files that already exist.

    Train: load -> parse -> length filter -> deterministic select.
    Val: the same from the val split, minus any instruction used in train. If that leaves
    fewer than n_val, the rest comes from train-pool examples not selected for train.
    """
    out_dir = Path(out_dir)
    train_path, val_path = out_dir / f"{prefix}_train.jsonl", out_dir / f"{prefix}_val.jsonl"

    def pool(load_rows):
        rows = load_rows()
        parsed = [example for example in map(parse_row, rows) if example is not None]
        return rows, parsed, [example for example in parsed if keep(example)]

    train_pool = None
    if train_path.exists():
        print(f"[skip] {train_path} already exists")
        train = read_jsonl(train_path)
    else:
        print(f"[build] {prefix}_train -> {train_path}")
        rows, parsed, train_pool = pool(load_train)
        train = select_examples(train_pool, n_train, seed, id_prefix=f"{prefix}_train")
        write_jsonl(train_path, train)
        stats.record(
            f"{prefix}_train", len(rows), len(parsed), len(train_pool), train, _length_lists(train, token_lengths)
        )

    if val_path.exists():
        print(f"[skip] {val_path} already exists")
        return
    print(f"[build] {prefix}_val -> {val_path}")
    rows, parsed, val_pool = pool(load_val)

    def load_train_pool():
        if train_pool is not None:
            return train_pool
        print(f"[fill] {prefix}_val is short; loading the train pool to top it up")
        return pool(load_train)[2]

    val, sources = fill_val(val_pool, train, n_val, seed, f"{prefix}_val", load_train_pool)
    write_jsonl(val_path, val)
    stats.record(
        f"{prefix}_val",
        len(rows),
        len(parsed),
        len(val_pool),
        val,
        _length_lists(val, token_lengths),
        val_sources=sources,
    )


# --- SFT: UltraChat -----------------------------------------------------------


def prepare_sft(args, stats):
    from datasets import load_dataset

    tokenizer = get_tokenizer(args.tokenizer)

    def keep(example):
        prompt_tokens = count_tokens(tokenizer, example["instruction"])
        if prompt_tokens > MAX_PROMPT_TOKENS:
            return False
        return prompt_tokens + count_tokens(tokenizer, example["response"]) <= MAX_SFT_TOKENS

    def token_lengths(example):
        prompt_tokens = count_tokens(tokenizer, example["instruction"])
        return {"instruction": prompt_tokens, "total": prompt_tokens + count_tokens(tokenizer, example["response"])}

    def load_rows(split):
        return list(load_dataset("HuggingFaceH4/ultrachat_200k", split=split).select_columns(["messages"]))

    build_train_val(
        "sft",
        args.out_dir,
        lambda: load_rows("train_sft"),
        lambda: load_rows("test_sft"),
        ultrachat_row_to_example,
        keep,
        token_lengths,
        args.sft_train,
        args.sft_val,
        args.seed,
        stats,
    )


# --- Preferences: HH-RLHF and UltraFeedback ---------------------------------------


def _pref_length_fns(tokenizer):
    def keep(example):
        return count_tokens(tokenizer, example["instruction"]) <= MAX_PROMPT_TOKENS

    def token_lengths(example):
        prompt_tokens = count_tokens(tokenizer, example["instruction"])
        return {
            "instruction": prompt_tokens,
            "chosen_total": prompt_tokens + count_tokens(tokenizer, example["chosen"]),
            "rejected_total": prompt_tokens + count_tokens(tokenizer, example["rejected"]),
        }

    return keep, token_lengths


def prepare_hh(args, stats):
    from datasets import load_dataset

    keep, token_lengths = _pref_length_fns(get_tokenizer(args.tokenizer))

    def load_rows(split):
        rows = []
        for source in ("helpful-base", "harmless-base"):
            for row in load_dataset("Anthropic/hh-rlhf", data_dir=source, split=split):
                rows.append({"chosen": row["chosen"], "rejected": row["rejected"], "source": source})
        return rows

    def parse_row(row):
        example = parse_hh_pair(row["chosen"], row["rejected"])
        return None if example is None else {**example, "source": row["source"]}

    build_train_val(
        "hh",
        args.out_dir,
        lambda: load_rows("train"),
        lambda: load_rows("test"),
        parse_row,
        keep,
        token_lengths,
        args.pref_train,
        args.pref_val,
        args.seed,
        stats,
    )


def prepare_uf(args, stats):
    from datasets import load_dataset

    keep, token_lengths = _pref_length_fns(get_tokenizer(args.tokenizer))
    columns = ["chosen", "rejected", "score_chosen", "score_rejected"]

    def load_rows(split):
        return list(load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split).select_columns(columns))

    build_train_val(
        "uf",
        args.out_dir,
        lambda: load_rows("train_prefs"),
        lambda: load_rows("test_prefs"),
        ultrafeedback_row_to_example,
        keep,
        token_lengths,
        args.pref_train,
        args.pref_val,
        args.seed,
        stats,
    )


# --- AlpacaEval -------------------------------------------------------------------


def load_alpaca_rows():
    """Download alpaca_eval.json directly (the dataset's loading script is broken)."""
    from huggingface_hub import HfApi, hf_hub_download

    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(repo_id=ALPACA_REPO, filename=ALPACA_FILE, repo_type="dataset")
    except EntryNotFoundError:
        files = HfApi().list_repo_files(ALPACA_REPO, repo_type="dataset")
        raise SystemExit(
            f"{ALPACA_FILE} not found in dataset repo {ALPACA_REPO}. Files in the repo:\n  "
            + "\n  ".join(files)
        ) from None
    return read_json(path)


def prepare_alpaca(args, stats):
    out_path = ROOT / "eval_sets" / f"alpaca_eval_{ALPACA_N}_seed{args.seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return
    print(f"[build] alpaca_eval -> {out_path}")
    rows = load_alpaca_rows()
    sampled = sample_alpaca_eval(rows, n=ALPACA_N, seed=args.seed)
    tokenizer = get_tokenizer(args.tokenizer)
    lengths = {"instruction": [count_tokens(tokenizer, row["instruction"]) for row in sampled]}
    write_json(out_path, sampled)
    stats.record("alpaca_eval", len(rows), len(rows), len(rows), sampled, lengths)


def main(argv=None):
    args = parse_args(argv)
    stats = StatsFile(Path(args.out_dir) / "data_stats.json")
    steps = {"sft": prepare_sft, "hh": prepare_hh, "uf": prepare_uf, "alpaca": prepare_alpaca}
    for key, step in steps.items():
        if args.only in ("all", key):
            step(args, stats)
    print(f"\nStats: {stats.path}")


if __name__ == "__main__":
    main()
