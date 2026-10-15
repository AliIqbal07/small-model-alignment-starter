"""Offline tests for the persona (Task 5) data helpers, eval prompts, and judge rubric."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from alignment.io_utils import read_json
from alignment.judging import RUBRICS, build_judge_prompt
from alignment.persona import near_duplicate, split_rows, to_trl_conversational

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "persona_lora" / "ramsay_style_preferences.json"
PROMPTS_PATH = ROOT / "eval_sets" / "persona_style_prompts.json"


def test_to_trl_conversational_shape():
    row = {"prompt": "How do I sear?", "chosen": "Hot pan, you donkey!", "rejected": "Use a hot pan."}
    assert to_trl_conversational(row) == {
        "prompt": [{"role": "user", "content": "How do I sear?"}],
        "chosen": [{"role": "assistant", "content": "Hot pan, you donkey!"}],
        "rejected": [{"role": "assistant", "content": "Use a hot pan."}],
    }


def test_split_rows_is_deterministic_and_disjoint():
    rows = [{"prompt": f"p{i}", "chosen": "c", "rejected": "r"} for i in range(50)]

    train, eval_rows = split_rows(rows, n_eval=20, seed=42)

    assert (train, eval_rows) == split_rows(rows, n_eval=20, seed=42)
    assert len(train) == 30 and len(eval_rows) == 20
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in eval_rows}
    assert sorted(r["prompt"] for r in train + eval_rows) == sorted(r["prompt"] for r in rows)
    assert split_rows(rows, n_eval=20, seed=1)[1] != eval_rows


def test_split_rows_on_real_data_has_no_prompt_overlap():
    rows = read_json(TRAIN_PATH)
    train, eval_rows = split_rows(rows, n_eval=20, seed=42)
    assert len(train) == 200 and len(eval_rows) == 20
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in eval_rows}


def test_split_rows_rejects_too_large_eval():
    with pytest.raises(ValueError):
        split_rows([{"prompt": "a"}], n_eval=1)


def test_near_duplicate():
    assert near_duplicate("How do I sear a steak?", "how do i SEAR a steak")
    assert near_duplicate("Why is my steak grey?", "Why is my steak so grey and sad?")
    assert not near_duplicate("How do I make an omelette?", "Explain Big-O notation simply.")


def test_persona_prompts_file():
    prompts = read_json(PROMPTS_PATH)

    assert len(prompts) == 40
    assert Counter(p["category"] for p in prompts) == {
        "cooking": 10, "coding": 10, "general_help": 10, "safety": 10
    }
    assert [p["id"] for p in prompts] == [f"persona-{i:04d}" for i in range(40)]
    for p in prompts:
        assert set(p) == {"id", "instruction", "dataset", "category"}
        assert p["dataset"] == "persona_style"
        assert p["instruction"].strip() == p["instruction"] and len(p["instruction"]) > 15
    assert len({p["instruction"].lower() for p in prompts}) == 40


def test_persona_prompts_do_not_duplicate_training_prompts():
    train_prompts = [row["prompt"] for row in read_json(TRAIN_PATH)]
    train_lower = {p.lower().strip() for p in train_prompts}
    clashes = []
    for p in read_json(PROMPTS_PATH):
        if p["instruction"].lower().strip() in train_lower:
            clashes.append((p["id"], "exact"))
        clashes += [(p["id"], t) for t in train_prompts if near_duplicate(p["instruction"], t)]
    assert clashes == []


def test_persona_rubric_and_prompt():
    rubric = RUBRICS["persona"]
    assert "Gordon Ramsay" in rubric and "useful" in rubric and "refuse" in rubric
    assert "Do not favor a response because it appears first." in rubric

    pairs = [{"pair_id": "persona-0001", "instruction": "Make an omelette?", "response_A": "x",
              "response_B": "y", "meta": {"A": "base", "B": "lora"}}]
    prompt = build_judge_prompt(pairs, "persona")
    assert rubric in prompt and "### Pair persona-0001" in prompt
    assert "lora" not in prompt and "meta" not in prompt.lower()
