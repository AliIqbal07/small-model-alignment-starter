"""Deterministic example selection and summary statistics."""

from __future__ import annotations

import math
import random


def select_examples(examples, n, seed, keep_fn=None, id_prefix="ex"):
    """Shuffle with random.Random(seed), filter with keep_fn, take n, and assign ids.

    Ids are f"{id_prefix}-{i:05d}" in selection order. Inputs are not mutated.
    Raises ValueError if fewer than n examples survive the filter.
    """
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    kept = [examples[i] for i in order if keep_fn is None or keep_fn(examples[i])]
    if len(kept) < n:
        raise ValueError(
            f"{id_prefix}: only {len(kept)} examples remain after filtering, need {n}"
        )
    return [{"id": f"{id_prefix}-{i:05d}", **example} for i, example in enumerate(kept[:n])]


def percentiles(values, points=(50, 90, 99)):
    """Return nearest-rank percentiles as {"p50": ..., ...}; None for empty input."""
    ordered = sorted(values)
    result = {}
    for point in points:
        if not ordered:
            result[f"p{point}"] = None
            continue
        rank = max(1, math.ceil(point / 100 * len(ordered)))
        result[f"p{point}"] = ordered[rank - 1]
    return result


def fill_val(val_pool, train_selected, n, seed, id_prefix, load_train_pool):
    """Select n val examples, topping up from unused train examples if the val split is short.

    Any example whose instruction text appears in train_selected is excluded, so train
    and val never share an instruction. load_train_pool() is called only when needed.
    Returns (examples with ids, {"val_split": count, "train_split": count}).
    Raises ValueError if both sources together have fewer than n examples.
    """
    train_instructions = {example["instruction"] for example in train_selected}

    def unused(pool):
        candidates = [example for example in pool if example["instruction"] not in train_instructions]
        random.Random(seed).shuffle(candidates)
        return candidates

    from_val = unused(val_pool)[:n]
    from_train = []
    if len(from_val) < n:
        from_train = unused(load_train_pool())[: n - len(from_val)]
    chosen = from_val + from_train
    if len(chosen) < n:
        raise ValueError(
            f"{id_prefix}: only {len(chosen)} examples available (val split + unused train), need {n}"
        )
    examples = [{"id": f"{id_prefix}-{i:05d}", **example} for i, example in enumerate(chosen)]
    return examples, {"val_split": len(from_val), "train_split": len(from_train)}
