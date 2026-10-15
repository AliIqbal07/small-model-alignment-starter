"""Preference dataset loading and preparation."""

from __future__ import annotations

HUMAN = "\n\nHuman:"
ASSISTANT = "\n\nAssistant:"


def _split_single_turn_hh(text):
    """Return (prompt_part, instruction, response) for a single-turn HH text, or None."""
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    human_at = text.index(HUMAN)
    assistant_at = text.index(ASSISTANT)
    if assistant_at < human_at:
        return None
    prompt_part = text[: assistant_at + len(ASSISTANT)]
    instruction = text[human_at + len(HUMAN) : assistant_at].strip()
    response = text[assistant_at + len(ASSISTANT) :].strip()
    return prompt_part, instruction, response


def _pair_or_none(instruction, chosen, rejected):
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return {"instruction": instruction, "chosen": chosen, "rejected": rejected}


def parse_hh_pair(chosen_text, rejected_text):
    """Return {"instruction", "chosen", "rejected"} for a single-turn HH-RLHF pair, or None."""
    chosen = _split_single_turn_hh(chosen_text)
    rejected = _split_single_turn_hh(rejected_text)
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    return _pair_or_none(chosen[1], chosen[2], rejected[2])


def _user_assistant_pair(messages):
    if not messages or len(messages) != 2:
        return None
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    return (user.get("content") or "").strip(), (assistant.get("content") or "").strip()


def ultrafeedback_row_to_example(row):
    """Return {"instruction", "chosen", "rejected"} for a single-turn, non-tied pair, or None."""
    score_chosen, score_rejected = row.get("score_chosen"), row.get("score_rejected")
    if score_chosen is not None and score_rejected is not None and score_chosen == score_rejected:
        return None
    chosen = _user_assistant_pair(row.get("chosen"))
    rejected = _user_assistant_pair(row.get("rejected"))
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    return _pair_or_none(chosen[0], chosen[1], rejected[1])
