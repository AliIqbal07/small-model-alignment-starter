"""Tests for blind A/B pair construction."""

from __future__ import annotations

import json

import pytest

from alignment.pairs import build_pairs


def outputs(generator, count, start=0):
    rows = []
    for i in range(start, start + count):
        row = {
            "id": f"ex-{i:04d}",
            "instruction": f"instruction {i}",
            "output": f"{generator} answer {i}",
            "generator": generator,
            "dataset": "hh_rlhf_style_eval",
            "num_new_tokens": 10,
            "hit_stop": True,
        }
        if i % 2:
            row["category"] = "harmful_instructions"
            row["expected_behavior"] = "Refuse."
        rows.append(row)
    return rows


def test_pairs_are_deterministic_and_sorted():
    a, b = outputs("sft_model", 30), outputs("dpo_model", 30)[::-1]

    first = build_pairs(a, b, "sft_model", "dpo_model", seed=42)

    assert first == build_pairs(a, b, "sft_model", "dpo_model", seed=42)
    assert [p["pair_id"] for p in first] == sorted(r["id"] for r in a)
    assert first != build_pairs(a, b, "sft_model", "dpo_model", seed=7)


def test_a_b_assignment_is_roughly_balanced():
    pairs = build_pairs(outputs("m1", 200), outputs("m2", 200), "m1", "m2", seed=42)
    share = sum(p["meta"]["A"] == "m1" for p in pairs) / len(pairs)
    assert 0.4 <= share <= 0.6


def test_meta_maps_back_to_original_outputs():
    a, b = outputs("sft_model", 20), outputs("dpo_model", 20)
    by_name = {"sft_model": {r["id"]: r for r in a}, "dpo_model": {r["id"]: r for r in b}}

    for pair in build_pairs(a, b, "sft_model", "dpo_model"):
        meta = pair["meta"]
        assert {meta["A"], meta["B"]} == {"sft_model", "dpo_model"}
        assert meta["a_id"] == pair["pair_id"]
        assert pair["response_A"] == by_name[meta["A"]][pair["pair_id"]]["output"]
        assert pair["response_B"] == by_name[meta["B"]][pair["pair_id"]]["output"]
        source = by_name["sft_model"][pair["pair_id"]]
        assert pair["instruction"] == source["instruction"]
        assert pair["dataset"] == source["dataset"]
        assert pair.get("category") == source.get("category")
        assert pair.get("expected_behavior") == source.get("expected_behavior")


def test_judge_facing_fields_hide_model_names():
    pairs = build_pairs(outputs("alpha_sft", 20), outputs("beta_dpo", 20), "alpha_sft", "beta_dpo")
    for pair in pairs:
        visible = json.dumps({k: v for k, v in pair.items() if k != "meta"})
        assert "alpha_sft" not in visible.replace("alpha_sft answer", "")
        assert "beta_dpo" not in visible.replace("beta_dpo answer", "")
        assert set(pair) <= {"pair_id", "instruction", "dataset", "category", "expected_behavior",
                             "response_A", "response_B", "meta"}


def test_mismatched_ids_raise():
    with pytest.raises(ValueError, match="ids differ"):
        build_pairs(outputs("m1", 5), outputs("m2", 5, start=1), "m1", "m2")


def test_mismatched_instructions_raise():
    b = outputs("m2", 5)
    b[2]["instruction"] = "something else"
    with pytest.raises(ValueError, match="instruction"):
        build_pairs(outputs("m1", 5), b, "m1", "m2")
