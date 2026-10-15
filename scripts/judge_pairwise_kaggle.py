"""Blind pairwise judging of A/B pairs with an LLM judge (Kaggle) or a fake judge (dry runs).

Example:
    python scripts/judge_pairwise_kaggle.py --pairs_path outputs/pairs/sft_vs_dpo_hh_hh.jsonl \\
        --output_path outputs/judgments/sft_vs_dpo_hh_hh.jsonl --judge_model <model> \\
        --rubric safety --backend kaggle --reuse_judgments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.io_utils import read_jsonl  # noqa: E402
from alignment.judge_backends import make_call_judge  # noqa: E402
from alignment.judging import judge_pairs  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Blind pairwise LLM judging.")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reuse_judgments", action="store_true", help="resume: skip pairs already judged")
    parser.add_argument("--rubric", choices=["general", "safety", "persona"], default="general")
    parser.add_argument("--backend", choices=["kaggle", "fake"], default="kaggle")
    parser.add_argument("--failures_path", help="default: <output_path stem>_failures.jsonl")
    parser.add_argument("--limit", type=int, help="judge only the first N pairs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = Path(args.output_path)
    failures_path = args.failures_path or output_path.with_name(f"{output_path.stem}_failures.jsonl")
    pairs = read_jsonl(args.pairs_path)[: args.limit]
    print(f"Judging {len(pairs)} pairs with {args.judge_model} ({args.backend}, rubric={args.rubric})")

    counts = judge_pairs(
        pairs,
        make_call_judge(args.backend, args.judge_model),
        output_path,
        failures_path,
        rubric=args.rubric,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        reuse=args.reuse_judgments,
    )
    print(f"Done: {counts}. Judgments: {output_path}; failures: {failures_path}")


if __name__ == "__main__":
    main()
