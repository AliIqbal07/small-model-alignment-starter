"""Evaluation dataset loading."""

from __future__ import annotations

import random


def sample_alpaca_eval(rows, n=300, seed=42):
    """Deterministically sample n AlpacaEval rows, returned in original index order."""
    indices = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [
        {
            "id": f"alpaca_eval-{idx:04d}",
            "instruction": rows[idx]["instruction"],
            "dataset": "alpaca_eval",
            "source": rows[idx].get("dataset"),
        }
        for idx in indices
    ]
