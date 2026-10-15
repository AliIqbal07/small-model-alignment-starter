"""Persona (Task 5) data helpers. No trl/peft imports here so tests run offline."""

from __future__ import annotations

import random
import re

_WORD = re.compile(r"[a-z0-9']+")


def to_trl_conversational(row):
    """{"prompt","chosen","rejected"} strings -> TRL conversational preference format."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def split_rows(rows, n_eval=20, seed=42):
    """Deterministic (train, eval) split; each keeps the original row order."""
    if not 0 < n_eval < len(rows):
        raise ValueError(f"n_eval must be between 1 and {len(rows) - 1}, got {n_eval}")
    eval_indices = set(random.Random(seed).sample(range(len(rows)), n_eval))
    train = [row for i, row in enumerate(rows) if i not in eval_indices]
    eval_rows = [row for i, row in enumerate(rows) if i in eval_indices]
    return train, eval_rows


def _words(text):
    return set(_WORD.findall(text.lower()))


def near_duplicate(a, b, threshold=0.6):
    """True if the lowercased word sets overlap by more than threshold.

    Overlap = shared words / words in the shorter text (stricter than Jaccard).
    """
    words_a, words_b = _words(a), _words(b)
    if not words_a or not words_b:
        return False
    return len(words_a & words_b) / min(len(words_a), len(words_b)) > threshold
