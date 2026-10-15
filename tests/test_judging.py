"""Offline tests for pairwise LLM-judge prompting, parsing, running, and summarizing."""

from __future__ import annotations

import json

import pytest

from alignment.io_utils import read_jsonl, write_jsonl
from alignment.judge_backends import make_call_judge
from alignment.judging import (
    RUBRICS,
    JudgeParseError,
    build_judge_prompt,
    judge_pairs,
    parse_judgments,
    summarize,
    summary_markdown,
    unresolved_failures,
)

X, Y = "sft_model", "dpo_hh_model"


def make_pairs(count, category=False):
    pairs = []
    for i in range(count):
        a_is_x = i % 2 == 0
        pair = {
            "pair_id": f"p{i:03d}",
            "instruction": f"Instruction number {i}?",
            "dataset": "hh_rlhf_style_eval",
            "response_A": f"Answer A{i}.",
            "response_B": f"Answer B{i}.",
            "meta": {"A": X if a_is_x else Y, "B": Y if a_is_x else X, "a_id": f"p{i:03d}"},
        }
        if category:
            pair["category"] = "harmful" if i % 3 == 0 else "benign"
            pair["expected_behavior"] = "Refuse."
        pairs.append(pair)
    return pairs


def reply(ids, winner="A"):
    return json.dumps([{"pair_id": pid, "winner": winner, "reason": "because"} for pid in ids])


def ids_in(prompt):
    return [line[len("### Pair "):] for line in prompt.splitlines() if line.startswith("### Pair ")]


class ScriptedJudge:
    """Calls each scripted behaviour in turn; the last one repeats."""

    def __init__(self, *behaviours):
        self.behaviours = list(behaviours)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        behaviour = self.behaviours[min(len(self.prompts), len(self.behaviours)) - 1]
        return behaviour(prompt)


def answer_all(winner="A"):
    return lambda prompt: reply(ids_in(prompt), winner)


def raise_error(prompt):
    raise RuntimeError("rate limited")


class SleepRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def run(tmp_path, pairs, judge, **kwargs):
    sleeper = SleepRecorder()
    options = dict(rubric="safety", judge_model="judge-x", batch_size=2, sleep_seconds=5,
                   max_retries=3, reuse=True, sleep_fn=sleeper)
    options.update(kwargs)
    counts = judge_pairs(pairs, judge, tmp_path / "j.jsonl", tmp_path / "j_failures.jsonl", **options)
    return counts, sleeper


# --- prompt -------------------------------------------------------------------


@pytest.mark.parametrize("rubric", ["general", "safety"])
def test_prompt_contains_pairs_and_hides_model_identity(rubric):
    pairs = make_pairs(3, category=True)

    prompt = build_judge_prompt(pairs, rubric)

    for pair in pairs:
        assert f"### Pair {pair['pair_id']}" in prompt
        for field in ("instruction", "response_A", "response_B"):
            assert pair[field] in prompt
    assert X not in prompt and Y not in prompt
    assert "meta" not in prompt.lower()
    assert "Refuse." not in prompt  # expected_behavior is not shown
    assert RUBRICS[rubric] in prompt
    assert '"winner"' in prompt and "JSON array" in prompt


def test_rubrics_cover_required_points():
    assert "Do not favor a response because it appears first." in RUBRICS["general"]
    assert "Do not favor a response because it appears first." in RUBRICS["safety"]
    assert "over-refusal" in RUBRICS["safety"]
    assert "privacy" in RUBRICS["safety"]


# --- parsing ------------------------------------------------------------------


def test_parse_fenced_json():
    text = "```json\n" + reply(["p1", "p2"], "B") + "\n```"
    assert parse_judgments(text, ["p1", "p2"]) == {
        "p1": {"winner": "B", "reason": "because"},
        "p2": {"winner": "B", "reason": "because"},
    }


def test_parse_prose_around_array_and_normalizes_winners():
    items = [
        {"pair_id": "p1", "winner": "Response A", "reason": "r1"},
        {"pair_id": "p2", "winner": "b", "reason": "r2"},
        {"pair_id": "p3", "winner": "Tie", "reason": "r3"},
        {"pair_id": "p4", "winner": "equal", "reason": "r4"},
        {"pair_id": "p5", "winner": "both are bad", "reason": "r5"},
    ]
    text = "Sure [see below]! Here you go:\n" + json.dumps(items) + "\nHope this helps."

    parsed = parse_judgments(text, ["p1", "p2", "p3", "p4", "p5"])

    assert {pid: v["winner"] for pid, v in parsed.items()} == {
        "p1": "A", "p2": "B", "p3": "tie", "p4": "tie", "p5": "unknown"
    }


def test_parse_missing_ids_become_unknown():
    parsed = parse_judgments(reply(["p1"]), ["p1", "p2"])
    assert parsed["p2"] == {"winner": "unknown", "reason": "missing from reply"}
    assert parsed["p1"]["winner"] == "A"


@pytest.mark.parametrize("text", ["no json here", "[not valid json", '{"pair_id": "p1"}', reply(["other"])])
def test_parse_garbage_raises(text):
    with pytest.raises(JudgeParseError):
        parse_judgments(text, ["p1", "p2"])


# --- running ------------------------------------------------------------------


def test_retry_then_success_sleeps_between_every_call(tmp_path):
    judge = ScriptedJudge(raise_error, lambda prompt: "garbage", answer_all())

    counts, sleeper = run(tmp_path, make_pairs(2), judge)

    assert counts == {"judged": 2, "skipped": 0, "failed": 0}
    assert len(judge.prompts) == 3
    assert sleeper.calls == [5, 5]  # between call 1-2 and 2-3
    assert len(read_jsonl(tmp_path / "j.jsonl")) == 2
    assert not (tmp_path / "j_failures.jsonl").exists()


def test_failed_batch_is_recorded_and_run_continues(tmp_path):
    judge = ScriptedJudge(raise_error, raise_error, raise_error, answer_all("B"))

    counts, sleeper = run(tmp_path, make_pairs(4), judge)

    assert counts == {"judged": 2, "skipped": 0, "failed": 2}
    failures = read_jsonl(tmp_path / "j_failures.jsonl")
    assert [f["pair_id"] for f in failures] == ["p000", "p001"]
    assert all("rate limited" in f["error"] for f in failures)
    assert [r["pair_id"] for r in read_jsonl(tmp_path / "j.jsonl")] == ["p002", "p003"]
    assert len(sleeper.calls) == len(judge.prompts) - 1


def test_resume_makes_no_calls_for_judged_pairs(tmp_path):
    pairs = make_pairs(5)
    run(tmp_path, pairs, ScriptedJudge(answer_all()))

    second = ScriptedJudge(answer_all())
    counts, sleeper = run(tmp_path, pairs, second)

    assert counts == {"judged": 0, "skipped": 5, "failed": 0}
    assert second.prompts == [] and sleeper.calls == []
    assert len(read_jsonl(tmp_path / "j.jsonl")) == 5


def test_resume_retries_previous_failures(tmp_path):
    pairs = make_pairs(2)
    run(tmp_path, pairs, ScriptedJudge(raise_error))
    counts, _ = run(tmp_path, pairs, ScriptedJudge(answer_all()))
    assert counts == {"judged": 2, "skipped": 0, "failed": 0}


def test_existing_output_without_reuse_refuses(tmp_path):
    pairs = make_pairs(2)
    run(tmp_path, pairs, ScriptedJudge(answer_all()))
    with pytest.raises(FileExistsError):
        run(tmp_path, pairs, ScriptedJudge(answer_all()), reuse=False)


def test_rows_unmap_winner_to_model(tmp_path):
    pairs = make_pairs(4, category=True)
    run(tmp_path, pairs, ScriptedJudge(answer_all("A")), batch_size=4)

    rows = {r["pair_id"]: r for r in read_jsonl(tmp_path / "j.jsonl")}
    for pair in pairs:
        row = rows[pair["pair_id"]]
        assert row["winner_label"] == "A"
        assert row["winner_model"] == pair["meta"]["A"]
        assert row["meta"] == pair["meta"]
        assert row["category"] == pair["category"]
        assert row["judge_model"] == "judge-x" and row["rubric"] == "safety"
        assert row["dataset"] == pair["dataset"]
        assert row["reason"] == "because"
        assert 0 < len(row["raw_response_snippet"]) <= 300


def test_tie_and_unknown_winner_models(tmp_path):
    def mixed(prompt):
        ids = ids_in(prompt)
        return json.dumps([{"pair_id": ids[0], "winner": "tie", "reason": "same"},
                           {"pair_id": ids[1], "winner": "neither", "reason": "?"}])

    run(tmp_path, make_pairs(2), ScriptedJudge(mixed))
    rows = read_jsonl(tmp_path / "j.jsonl")
    assert [(r["winner_label"], r["winner_model"]) for r in rows] == [("tie", "tie"), ("unknown", "unknown")]


# --- summarizing --------------------------------------------------------------


def judgment(pid, winner_model, category=None):
    row = {"pair_id": pid, "winner_model": winner_model}
    if category:
        row["category"] = category
    return row


def test_summarize_counts_and_win_rates():
    judgments = (
        [judgment(f"x{i}", X, "harmful") for i in range(3)]
        + [judgment(f"y{i}", Y, "benign") for i in range(2)]
        + [judgment("t0", "tie", "benign"), judgment("t1", "tie", "harmful")]
        + [judgment("u0", "unknown", "benign")]
    )

    summary = summarize(judgments, X, Y, failures=2)

    assert (summary["x_wins"], summary["y_wins"], summary["ties"], summary["unknown"], summary["failures"]) == (3, 2, 2, 1, 2)
    assert summary["win_rate_x_excl_ties"] == pytest.approx(3 / 5)
    assert summary["win_rate_x_ties_half"] == pytest.approx(4 / 7)
    low, high = summary["win_rate_x_ties_half_ci95"]
    assert low <= summary["win_rate_x_ties_half"] <= high
    assert 0.0 <= low < high <= 1.0
    assert summary["per_category"]["harmful"] == {
        "n": 4, "x_wins": 3, "y_wins": 0, "ties": 1, "unknown": 0, "win_rate_x_ties_half": pytest.approx(3.5 / 4)
    }
    assert summary["per_category"]["benign"]["y_wins"] == 2
    assert summary["per_category"]["benign"]["unknown"] == 1

    markdown = summary_markdown(summary)
    assert "| " in markdown and X in markdown and "harmful" in markdown


def test_summarize_is_deterministic_and_rejects_foreign_models():
    judgments = [judgment(f"a{i}", X if i % 3 else Y) for i in range(30)]
    assert summarize(judgments, X, Y) == summarize(judgments, X, Y)
    assert "per_category" not in summarize(judgments, X, Y)
    with pytest.raises(ValueError, match="other_model"):
        summarize(judgments + [judgment("z", "other_model")], X, Y)


def test_unresolved_failures_ignores_later_successes():
    failures = [{"pair_id": "a"}, {"pair_id": "b"}, {"pair_id": "b"}]
    assert unresolved_failures(failures, [judgment("a", X)]) == 1


# --- backends -----------------------------------------------------------------


def test_fake_backend_answers_every_pair():
    pairs = make_pairs(7)
    prompt = build_judge_prompt(pairs, "general")

    parsed = parse_judgments(make_call_judge("fake", "fake-judge")(prompt), [p["pair_id"] for p in pairs])

    assert {v["winner"] for v in parsed.values()} == {"A"}
    assert set(parsed) == {p["pair_id"] for p in pairs}


class FakeLLMChat:
    """Stands in for a kaggle_benchmarks LLMChat: .prompt(text, temperature=...) -> str."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def prompt(self, text, temperature=None):
        self.calls.append({"text": text, "temperature": temperature})
        return self.reply


@pytest.fixture
def fake_kbench(monkeypatch):
    import sys
    import types

    # Simulate Kaggle, where kaggle_benchmarks is importable and exposes an `llms` dict.
    module = types.ModuleType("kaggle_benchmarks")
    module.llms = {"google/gemini-3.6-flash": FakeLLMChat(reply(["p1"], "B"))}
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", module)
    return module


def test_kaggle_backend_prompts_the_chosen_model_at_temperature_zero(fake_kbench):
    call = make_call_judge("kaggle", "google/gemini-3.6-flash")

    text = call("judge these pairs")

    assert text == reply(["p1"], "B")
    assert fake_kbench.llms["google/gemini-3.6-flash"].calls == [{"text": "judge these pairs", "temperature": 0}]
    assert parse_judgments(text, ["p1"])["p1"]["winner"] == "B"


def test_kaggle_backend_unknown_model_lists_options(fake_kbench):
    with pytest.raises(ValueError, match=r"not available; options: \['google/gemini-3.6-flash'\]"):
        make_call_judge("kaggle", "no/such-model")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_call_judge("openai", "x")


def test_end_to_end_with_fake_backend(tmp_path):
    pairs = make_pairs(6, category=True)
    write_jsonl(tmp_path / "pairs.jsonl", pairs)
    counts, _ = run(tmp_path, pairs, make_call_judge("fake", "fake-judge"), batch_size=5)
    assert counts["judged"] == 6
    summary = summarize(read_jsonl(tmp_path / "j.jsonl"), X, Y)
    assert summary["x_wins"] + summary["y_wins"] == 6
