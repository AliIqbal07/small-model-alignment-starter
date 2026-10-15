"""Blind A/B pairs for pairwise judging."""

from __future__ import annotations

import random

CARRIED_FIELDS = ("category", "expected_behavior")


def build_pairs(outputs_a, outputs_b, name_a, name_b, seed=42):
    """Match two models' outputs by id and randomly assign which model is shown as A.

    Rows are sorted by id; random.Random(seed) draws one A/B coin flip per row in that order.
    Model names appear only under "meta", never in the judge-facing fields.
    """
    by_id_a = {row["id"]: row for row in outputs_a}
    by_id_b = {row["id"]: row for row in outputs_b}
    if by_id_a.keys() != by_id_b.keys():
        only_a = sorted(by_id_a.keys() - by_id_b.keys())
        only_b = sorted(by_id_b.keys() - by_id_a.keys())
        raise ValueError(
            f"Output ids differ: {len(only_a)} only in {name_a} (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in {name_b} (e.g. {only_b[:3]})"
        )

    rng = random.Random(seed)
    pairs = []
    for pair_id in sorted(by_id_a):
        row_a, row_b = by_id_a[pair_id], by_id_b[pair_id]
        if row_a["instruction"] != row_b["instruction"]:
            raise ValueError(f"Id {pair_id!r} has a different instruction in {name_a} and {name_b}")
        a_first = rng.random() < 0.5
        shown_a, shown_b = (row_a, row_b) if a_first else (row_b, row_a)
        pair = {"pair_id": pair_id, "instruction": row_a["instruction"], "dataset": row_a.get("dataset")}
        pair.update({key: row_a[key] for key in CARRIED_FIELDS if key in row_a})
        pair.update(
            {
                "response_A": shown_a["output"],
                "response_B": shown_b["output"],
                "meta": {
                    "A": name_a if a_first else name_b,
                    "B": name_b if a_first else name_a,
                    "a_id": pair_id,
                },
            }
        )
        pairs.append(pair)
    return pairs
