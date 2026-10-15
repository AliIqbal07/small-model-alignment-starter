"""Text generation helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from alignment.formatting import format_prompt, stop_token_ids
from alignment.io_utils import append_jsonl, read_jsonl

CARRIED_FIELDS = ("category", "expected_behavior")


def generate_outputs(rows, generate_batch_fn, batch_size, generator_name, partial_path=None):
    """Generate one output per row, batching by instruction length; return in the original order.

    generate_batch_fn(instructions) -> [(text, num_new_tokens, hit_stop), ...].
    If partial_path is given, finished rows are appended there after every batch and rows whose
    id is already in it are skipped, so an interrupted run can resume.
    """
    done = {}
    if partial_path is not None and Path(partial_path).exists():
        done = {row["id"]: row for row in read_jsonl(partial_path)}
        print(f"Resuming: {len(done)} rows already in {partial_path}")

    pending = [row for row in rows if row["id"] not in done]
    pending.sort(key=lambda row: len(row["instruction"]), reverse=True)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results = generate_batch_fn([row["instruction"] for row in batch])
        if len(results) != len(batch):
            raise ValueError(f"generate_batch_fn returned {len(results)} results for {len(batch)} prompts")
        finished = []
        for row, (text, num_new_tokens, hit_stop) in zip(batch, results):
            output = {
                "id": row["id"],
                "instruction": row["instruction"],
                "output": text,
                "generator": generator_name,
                "dataset": row.get("dataset"),
            }
            output.update({key: row[key] for key in CARRIED_FIELDS if key in row})
            output.update({"num_new_tokens": num_new_tokens, "hit_stop": hit_stop})
            finished.append(output)
            done[row["id"]] = output
        if partial_path is not None:
            append_jsonl(partial_path, finished)
        print(f"Generated {len(done)}/{len(rows)}")

    return [done[row["id"]] for row in rows]


def make_hf_generate_fn(model, tokenizer, gen_cfg):
    """Return a greedy generate_batch_fn for a Hugging Face causal LM.

    gen_cfg: max_new_tokens, repetition_penalty, no_repeat_ngram_size (0 = off).
    """
    stops = stop_token_ids(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    kwargs = {
        "do_sample": False,
        "max_new_tokens": gen_cfg["max_new_tokens"],
        "repetition_penalty": gen_cfg["repetition_penalty"],
        "eos_token_id": stops,
        "pad_token_id": pad_id,
    }
    if gen_cfg.get("no_repeat_ngram_size"):
        kwargs["no_repeat_ngram_size"] = gen_cfg["no_repeat_ngram_size"]

    def generate_batch(instructions):
        prompts = [format_prompt(tokenizer, instruction) for instruction in instructions]
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tokenizer.padding_side = previous_side
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        with torch.no_grad():
            sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        results = []
        for row in sequences[:, input_ids.shape[1] :].tolist():
            stop_at = next((i for i, token in enumerate(row) if token in stops), None)
            new_tokens = row if stop_at is None else row[: stop_at + 1]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append((text, len(new_tokens), stop_at is not None))
        return results

    return generate_batch
