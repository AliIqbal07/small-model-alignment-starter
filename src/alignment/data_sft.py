"""SFT dataset loading and preparation."""

from __future__ import annotations


def ultrachat_row_to_example(row):
    """Return {"instruction", "response"} from the first user turn and its reply, or None."""
    messages = row.get("messages") or []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "assistant":
            return None
        instruction = (message.get("content") or "").strip()
        response = (messages[index + 1].get("content") or "").strip()
        if not instruction or not response:
            return None
        return {"instruction": instruction, "response": response}
    return None
