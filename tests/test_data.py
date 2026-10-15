"""Offline tests for data parsing, selection, sampling, and JSON I/O."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from alignment.data_eval import sample_alpaca_eval
from alignment.data_pref import parse_hh_pair, ultrafeedback_row_to_example
from alignment.data_sft import ultrachat_row_to_example
from alignment.data_utils import percentiles, select_examples
from alignment.io_utils import read_json, read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def msg(role, content):
    return {"role": role, "content": content}


def hh(human, assistant):
    return f"\n\nHuman: {human}\n\nAssistant: {assistant}"


# --- UltraChat ---------------------------------------------------------------


def test_ultrachat_uses_first_user_and_following_assistant():
    row = {
        "messages": [
            msg("user", "  What is 2+2?  "),
            msg("assistant", " 4. "),
            msg("user", "And 3+3?"),
            msg("assistant", "6."),
        ]
    }
    assert ultrachat_row_to_example(row) == {"instruction": "What is 2+2?", "response": "4."}


def test_ultrachat_skips_leading_system_message():
    row = {"messages": [msg("system", "Be nice."), msg("user", "Hi"), msg("assistant", "Hello!")]}
    assert ultrachat_row_to_example(row) == {"instruction": "Hi", "response": "Hello!"}


@pytest.mark.parametrize(
    "messages",
    [
        [msg("user", "Hi")],  # no assistant reply
        [msg("user", "Hi"), msg("user", "Hello?")],  # next message is not the assistant
        [msg("assistant", "Hello!")],  # no user message
        [msg("user", "   "), msg("assistant", "Hello!")],  # empty instruction
        [msg("user", "Hi"), msg("assistant", "  ")],  # empty response
        [],
    ],
)
def test_ultrachat_invalid_rows_return_none(messages):
    assert ultrachat_row_to_example({"messages": messages}) is None


# --- HH-RLHF -----------------------------------------------------------------


def test_hh_single_turn_pair_is_parsed():
    result = parse_hh_pair(hh("How do I bake bread?", "Use flour."), hh("How do I bake bread?", "No idea."))
    assert result == {"instruction": "How do I bake bread?", "chosen": "Use flour.", "rejected": "No idea."}


def test_hh_multi_turn_is_rejected():
    chosen = hh("Hi", "Hello") + hh("How are you?", "Good.")
    rejected = hh("Hi", "Hello") + hh("How are you?", "Bad.")
    assert parse_hh_pair(chosen, rejected) is None


def test_hh_multi_turn_in_only_one_side_is_rejected():
    assert parse_hh_pair(hh("Hi", "Hello"), hh("Hi", "Hey") + hh("More?", "No")) is None


def test_hh_prompt_mismatch_is_rejected():
    assert parse_hh_pair(hh("Question A", "Answer"), hh("Question B", "Answer 2")) is None


@pytest.mark.parametrize(
    "chosen, rejected",
    [
        (hh("Q", "same"), hh("Q", "same")),  # identical responses
        (hh("Q", "  "), hh("Q", "other")),  # empty chosen
        (hh("Q", "answer"), hh("Q", "")),  # empty rejected
        (hh("  ", "a"), hh("  ", "b")),  # empty instruction
        ("no markers at all", "nothing here"),
    ],
)
def test_hh_empty_or_identical_is_rejected(chosen, rejected):
    assert parse_hh_pair(chosen, rejected) is None


# --- UltraFeedback ---------------------------------------------------------


def uf_row(chosen, rejected, score_chosen=8.0, score_rejected=4.0, user="Explain gravity."):
    return {
        "prompt": user,
        "chosen": [msg("user", user), msg("assistant", chosen)],
        "rejected": [msg("user", user), msg("assistant", rejected)],
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
    }


def test_uf_valid_row_is_parsed():
    result = ultrafeedback_row_to_example(uf_row(" Mass attracts. ", "Magic."))
    assert result == {"instruction": "Explain gravity.", "chosen": "Mass attracts.", "rejected": "Magic."}


def test_uf_without_scores_is_kept():
    row = uf_row("Mass attracts.", "Magic.")
    del row["score_chosen"], row["score_rejected"]
    assert ultrafeedback_row_to_example(row) is not None


def test_uf_tie_is_rejected():
    assert ultrafeedback_row_to_example(uf_row("A", "B", score_chosen=5.0, score_rejected=5.0)) is None


def test_uf_multi_turn_is_rejected():
    row = uf_row("A", "B")
    row["chosen"] = row["chosen"] + [msg("user", "More?"), msg("assistant", "Sure.")]
    assert ultrafeedback_row_to_example(row) is None


def test_uf_user_mismatch_is_rejected():
    row = uf_row("A", "B")
    row["rejected"][0] = msg("user", "Something else")
    assert ultrafeedback_row_to_example(row) is None


def test_uf_wrong_roles_are_rejected():
    row = uf_row("A", "B")
    row["chosen"] = [msg("assistant", "A"), msg("user", "Explain gravity.")]
    assert ultrafeedback_row_to_example(row) is None


@pytest.mark.parametrize("chosen, rejected", [("same", "same"), ("", "B"), ("A", "   ")])
def test_uf_identical_or_empty_is_rejected(chosen, rejected):
    assert ultrafeedback_row_to_example(uf_row(chosen, rejected)) is None


# --- AlpacaEval sampling ----------------------------------------------------


def alpaca_rows(count):
    return [{"instruction": f"instruction {i}", "output": "x", "dataset": f"src{i % 3}"} for i in range(count)]


def test_alpaca_sampling_is_deterministic_sorted_and_formatted():
    rows = alpaca_rows(50)

    first = sample_alpaca_eval(rows, n=10, seed=42)
    second = sample_alpaca_eval(rows, n=10, seed=42)

    assert first == second
    assert len(first) == 10
    indices = [int(item["id"].split("-")[1]) for item in first]
    assert indices == sorted(indices)
    assert len(set(indices)) == 10
    for item, idx in zip(first, indices):
        assert item == {
            "id": f"alpaca_eval-{idx:04d}",
            "instruction": rows[idx]["instruction"],
            "dataset": "alpaca_eval",
            "source": rows[idx]["dataset"],
        }


def test_alpaca_sampling_matches_random_sample_with_seed():
    import random

    rows = alpaca_rows(40)
    expected = sorted(random.Random(7).sample(range(40), 5))
    assert [r["id"] for r in sample_alpaca_eval(rows, n=5, seed=7)] == [f"alpaca_eval-{i:04d}" for i in expected]


def test_alpaca_sampling_differs_across_seeds():
    rows = alpaca_rows(100)
    assert sample_alpaca_eval(rows, n=10, seed=1) != sample_alpaca_eval(rows, n=10, seed=2)


# --- select_examples --------------------------------------------------------


def test_select_examples_is_deterministic_and_assigns_ids():
    examples = [{"instruction": f"q{i}"} for i in range(20)]

    first = select_examples(examples, n=5, seed=3, id_prefix="sft-train")
    second = select_examples(examples, n=5, seed=3, id_prefix="sft-train")

    assert first == second
    assert [e["id"] for e in first] == [f"sft-train-{i:05d}" for i in range(5)]
    assert first != select_examples(examples, n=5, seed=4, id_prefix="sft-train")
    assert "id" not in examples[0]  # inputs are not mutated


def test_select_examples_applies_keep_fn():
    examples = [{"value": i} for i in range(20)]
    selected = select_examples(examples, n=5, seed=0, keep_fn=lambda e: e["value"] % 2 == 0)
    assert len(selected) == 5
    assert all(e["value"] % 2 == 0 for e in selected)


def test_select_examples_raises_when_not_enough_rows():
    examples = [{"value": i} for i in range(10)]
    with pytest.raises(ValueError, match="only 5 examples remain.*need 6"):
        select_examples(examples, n=6, seed=0, keep_fn=lambda e: e["value"] < 5)


def test_percentiles_nearest_rank():
    values = list(range(1, 101))
    assert percentiles(values) == {"p50": 50, "p90": 90, "p99": 99}
    assert percentiles([]) == {"p50": None, "p90": None, "p99": None}


# --- JSON I/O ---------------------------------------------------------------


def test_jsonl_roundtrip_is_atomic(tmp_path):
    rows = [{"id": "a", "text": "héllo\nworld"}, {"id": "b", "n": 2}]
    path = tmp_path / "nested" / "rows.jsonl"

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows
    assert not list(path.parent.glob("*.tmp"))


def test_json_roundtrip(tmp_path):
    data = {"a": [1, 2], "b": "ü"}
    path = tmp_path / "data.json"
    write_json(path, data)
    assert read_json(path) == data
    assert not list(tmp_path.glob("*.tmp"))


# --- offline guarantee --------------------------------------------------------


def test_data_modules_and_script_do_not_import_heavy_libraries():
    code = (
        "import sys, runpy\n"
        "import alignment.data_sft, alignment.data_pref, alignment.data_eval\n"
        "import alignment.data_utils, alignment.io_utils\n"
        "runpy.run_path('scripts/prepare_data.py', run_name='not_main')\n"
        "bad = [m for m in ('datasets', 'transformers', 'huggingface_hub') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- val fill from the train pool ---------------------------------------------


def test_fill_val_uses_val_split_first_then_unused_train_examples():
    from alignment.data_utils import fill_val

    train_pool = [{"instruction": f"t{i}"} for i in range(10)]
    train = select_examples(train_pool, n=5, seed=0, id_prefix="x_train")
    val_pool = [{"instruction": "v0"}, {"instruction": "v1"}]

    val, counts = fill_val(val_pool, train, n=4, seed=0, id_prefix="x_val", load_train_pool=lambda: train_pool)

    assert counts == {"val_split": 2, "train_split": 2}
    assert [e["id"] for e in val] == [f"x_val-{i:05d}" for i in range(4)]
    assert {e["instruction"] for e in val[:2]} == {"v0", "v1"}
    train_instructions = {e["instruction"] for e in train}
    assert not train_instructions & {e["instruction"] for e in val}
    assert val == fill_val(val_pool, train, n=4, seed=0, id_prefix="x_val", load_train_pool=lambda: train_pool)[0]


def test_fill_val_drops_val_rows_whose_instruction_is_in_train():
    from alignment.data_utils import fill_val

    train = [{"id": "x_train-00000", "instruction": "shared"}]
    val_pool = [{"instruction": "shared"}, {"instruction": "v1"}, {"instruction": "v2"}]
    train_pool = [{"instruction": "shared"}, {"instruction": "shared"}, {"instruction": "t1"}]

    val, counts = fill_val(val_pool, train, n=3, seed=1, id_prefix="x_val", load_train_pool=lambda: train_pool)

    assert counts == {"val_split": 2, "train_split": 1}
    assert sorted(e["instruction"] for e in val) == ["t1", "v1", "v2"]


def test_fill_val_does_not_load_train_pool_when_val_is_large_enough():
    from alignment.data_utils import fill_val

    def fail():
        raise AssertionError("train pool should not be loaded")

    val, counts = fill_val([{"instruction": f"v{i}"} for i in range(5)], [], n=3, seed=0, id_prefix="v", load_train_pool=fail)
    assert len(val) == 3 and counts == {"val_split": 3, "train_split": 0}


def test_fill_val_raises_when_train_pool_is_also_exhausted():
    from alignment.data_utils import fill_val

    train = [{"instruction": "t0"}]
    with pytest.raises(ValueError, match="x_val: only 2 examples available.*need 5"):
        fill_val([{"instruction": "v0"}], train, n=5, seed=0, id_prefix="x_val",
                 load_train_pool=lambda: [{"instruction": "t0"}, {"instruction": "t1"}])


def test_script_build_train_val_fills_val_and_records_sources(tmp_path):
    import runpy

    script = runpy.run_path(str(ROOT / "scripts" / "prepare_data.py"), run_name="not_main")
    train_rows = [{"q": f"train question {i}", "a": "answer"} for i in range(30)]
    train_rows.append({"q": "train question 0", "a": "duplicate instruction"})
    val_rows = [{"q": f"val question {i}", "a": "answer"} for i in range(3)] + [{"q": "", "a": "x"}]
    stats = script["StatsFile"](tmp_path / "data_stats.json")

    def parse(row):
        return {"instruction": row["q"], "response": row["a"]} if row["q"] else None

    script["build_train_val"](
        "toy",
        tmp_path,
        load_train=lambda: train_rows,
        load_val=lambda: val_rows,
        parse_row=parse,
        keep=lambda e: True,
        token_lengths=lambda e: {"instruction": len(e["instruction"].split())},
        n_train=20,
        n_val=8,
        seed=42,
        stats=stats,
    )

    train = read_jsonl(tmp_path / "toy_train.jsonl")
    val = read_jsonl(tmp_path / "toy_val.jsonl")
    assert len(train) == 20 and len(val) == 8
    assert not {e["instruction"] for e in train} & {e["instruction"] for e in val}
    saved = read_json(tmp_path / "data_stats.json")
    assert saved["toy_val"]["val_sources"] == {"val_split": 3, "train_split": 5}
    assert saved["toy_val"]["final_rows"] == 8

    # Rerun with train already on disk: val is rebuilt identically from the saved train file.
    (tmp_path / "toy_val.jsonl").unlink()
    script["build_train_val"](
        "toy", tmp_path, lambda: train_rows, lambda: val_rows, parse, lambda e: True,
        lambda e: {"instruction": 1}, 20, 8, 42, stats,
    )
    assert read_jsonl(tmp_path / "toy_val.jsonl") == val
