"""Judge backends: functions mapping a prompt string to the judge's reply text."""

from __future__ import annotations

import json
import re

_PAIR_HEADER = re.compile(r"^### Pair (.+)$", re.MULTILINE)


def _fake_judge(prompt):
    """Dry-run judge: picks "A" for every pair in the prompt."""
    ids = _PAIR_HEADER.findall(prompt)
    return json.dumps([{"pair_id": pid, "winner": "A", "reason": "fake judge always picks A"} for pid in ids])


# ---------------------------------------------------------------------------
# KAGGLE JUDGE PLACEHOLDER: edit this function only.
# ---------------------------------------------------------------------------
def _make_kaggle_judge(judge_model):
    def call_judge(prompt):
        import kaggle_benchmarks  # noqa: F401  (lazy: only available on Kaggle)

        # TODO: send `prompt` to `judge_model` and return the reply text (str).
        raise NotImplementedError(
            "Fill in using the kaggle_benchmarks example from the Kaggle utility-benchmark notebook"
        )

    return call_judge


def make_call_judge(backend, judge_model):
    """Return call_judge(prompt) -> str for backend "fake" or "kaggle"."""
    if backend == "fake":
        return _fake_judge
    if backend == "kaggle":
        return _make_kaggle_judge(judge_model)
    raise ValueError(f"Unknown judge backend {backend!r}; use 'kaggle' or 'fake'")
