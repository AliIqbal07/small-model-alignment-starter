"""Summarize pairwise judgments: win rates, bootstrap CI, per-category breakdown.

Example:
    python scripts/summarize_judgments.py --judgments outputs/judgments/sft_vs_dpo_hh_hh.jsonl \\
        --model_x dpo_hh --model_y sft --eval_set hh_rlhf_eval_300 --out_json outputs/judgments/sft_vs_dpo_hh_hh.summary.json \\
        --out_md outputs/judgments/sft_vs_dpo_hh_hh.summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.io_utils import read_jsonl, write_json, write_text  # noqa: E402
from alignment.judging import summarize, summary_markdown, unresolved_failures  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize pairwise judgments.")
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--model_x", required=True)
    parser.add_argument("--model_y", required=True)
    parser.add_argument("--failures_path", help="failures JSONL; counts pairs never judged successfully")
    parser.add_argument("--eval_set", help="eval set label stored in the summary (e.g. alpaca_eval_300)")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args(argv)

    judgments = read_jsonl(args.judgments)
    failures = 0
    if args.failures_path and Path(args.failures_path).exists():
        failures = unresolved_failures(read_jsonl(args.failures_path), judgments)

    summary = summarize(judgments, args.model_x, args.model_y, failures=failures)
    summary["eval_set"] = args.eval_set
    markdown = summary_markdown(summary)
    write_json(args.out_json, summary)
    write_text(args.out_md, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
