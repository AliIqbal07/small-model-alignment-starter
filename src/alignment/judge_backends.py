"""Judge backends: functions mapping a prompt string to the judge's reply text."""

from __future__ import annotations

import json
import re

_PAIR_HEADER = re.compile(r"^### Pair (.+)$", re.MULTILINE)


def _fake_judge(prompt):
    """Dry-run judge: picks "A" for every pair in the prompt."""
    ids = _PAIR_HEADER.findall(prompt)
    return json.dumps([{"pair_id": pid, "winner": "A", "reason": "fake judge always picks A"} for pid in ids])


def _make_kaggle_judge(judge_model):
    """Judge via kaggle_benchmarks (Kaggle Benchmark Task notebooks only).

    kbench.llms maps model names to LLMChat objects; .prompt() calls do not share history,
    so every batch is judged independently.
    """
    import kaggle_benchmarks as kbench  # lazy: only available on Kaggle

    if judge_model not in kbench.llms:
        raise ValueError(f"Judge model {judge_model!r} not available; options: {sorted(kbench.llms)}")
    llm = kbench.llms[judge_model]

    def call_judge(prompt):
        return str(llm.prompt(prompt, temperature=0))

    return call_judge


def make_call_judge(backend, judge_model):
    """Return call_judge(prompt) -> str for backend "fake" or "kaggle"."""
    if backend == "fake":
        return _fake_judge
    if backend == "kaggle":
        return _make_kaggle_judge(judge_model)
    raise ValueError(f"Unknown judge backend {backend!r}; use 'kaggle' or 'fake'")
