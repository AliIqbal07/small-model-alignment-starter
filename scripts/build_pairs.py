"""Build blind A/B pairs from two models' generation outputs.

Example:
    python scripts/build_pairs.py --a outputs/gen/sft_alpaca.json --b outputs/gen/dpo_uf_alpaca.json \\
        --name_a sft --name_b dpo_uf --out outputs/pairs/sft_vs_dpo_uf_alpaca.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.io_utils import read_json, write_jsonl  # noqa: E402
from alignment.pairs import build_pairs  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Blind A/B pairs for pairwise judging.")
    parser.add_argument("--a", required=True, help="outputs JSON of model a")
    parser.add_argument("--b", required=True, help="outputs JSON of model b")
    parser.add_argument("--name_a", required=True)
    parser.add_argument("--name_b", required=True)
    parser.add_argument("--out", required=True, help="pairs JSONL")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    pairs = build_pairs(read_json(args.a), read_json(args.b), args.name_a, args.name_b, seed=args.seed)
    write_jsonl(args.out, pairs)
    shown_a = sum(pair["meta"]["A"] == args.name_a for pair in pairs)
    print(f"Wrote {len(pairs)} pairs to {args.out}")
    print(f"{args.name_a} shown as A: {shown_a}/{len(pairs)}; {args.name_b} shown as A: {len(pairs) - shown_a}/{len(pairs)}")


if __name__ == "__main__":
    main()
