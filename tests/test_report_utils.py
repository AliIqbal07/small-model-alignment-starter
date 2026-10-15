"""Offline tests for report figures and tables, using fake files in the real trainers' formats."""

from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("matplotlib")

from alignment.report_utils import (  # noqa: E402
    config_table,
    load_metrics,
    plot_dpo,
    plot_persona,
    plot_sft,
    results_table,
    smooth,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_sft_run(run_dir):
    """Same keys as alignment.sft_trainer.train_sft writes."""
    run_dir.mkdir(parents=True)
    rows = [{"type": "val", "step": 0, "val_loss": 1.40, "elapsed_s": 1.0}]
    for step in range(10, 101, 10):
        rows.append({"type": "train", "step": step, "train_loss": 1.4 - step / 1000 + 0.02 * math.sin(step),
                     "lr": 2e-5, "grad_norm": 4.0, "loss_scale": 65536.0, "tokens_seen": step * 1000,
                     "elapsed_s": float(step), "max_mem_gb": 12.5})
        if step % 50 == 0:
            rows.append({"type": "val", "step": step, "val_loss": 1.40 - step / 2000, "elapsed_s": float(step)})
    write_jsonl(run_dir / "metrics.jsonl", rows)
    (run_dir / "best.json").write_text(json.dumps({"step": 100, "val_loss": 1.35}))
    (run_dir / "config_resolved.yaml").write_text(
        "model_name: Qwen/Qwen2.5-0.5B\ntrain_file: outputs/data/sft_train.jsonl\nval_file: outputs/data/sft_val.jsonl\n"
        "max_length: 512\nmax_prompt_length: 256\nbatch_size: 2\ngrad_accum: 16\nlr: 2.0e-05\n"
        "warmup_ratio: 0.03\nweight_decay: 0.1\nepochs: 1\nmax_steps: null\nfp16: true\nseed: 42\n"
    )
    return run_dir


def make_dpo_run(run_dir):
    """Same keys as alignment.dpo_trainer.train_dpo writes."""
    run_dir.mkdir(parents=True)

    def metrics(step):
        return {"loss": 0.693 - step / 1000, "reward_accuracy": min(1.0, step / 40), "reward_margin": step / 100,
                "chosen_reward": step / 200, "rejected_reward": -step / 200}

    rows = [{"type": "val", "step": 0, **metrics(0), "elapsed_s": 1.0}]
    for step in range(5, 41, 5):
        rows.append({"type": "train", "step": step, **metrics(step), "policy_chosen_logps": -50.0,
                     "policy_rejected_logps": -60.0, "lr": 1e-6, "grad_norm": 2.0, "loss_scale": 65536.0,
                     "elapsed_s": float(step), "max_mem_gb_cuda:0": 10.0, "max_mem_gb_cuda:1": 3.0})
        if step % 20 == 0:
            rows.append({"type": "val", "step": step, **metrics(step), "elapsed_s": float(step)})
    write_jsonl(run_dir / "metrics.jsonl", rows)
    (run_dir / "best.json").write_text(json.dumps({"step": 40, **metrics(40)}))
    (run_dir / "config_resolved.yaml").write_text(
        "sft_model_path: outputs/sft/best\ntrain_file: outputs/data/hh_train.jsonl\nval_file: outputs/data/hh_val.jsonl\n"
        "beta: 0.1\nmax_length: 512\nmax_prompt_length: 256\nbatch_size: 1\ngrad_accum: 32\noptimizer: rmsprop\n"
        "lr: 1.0e-06\nwarmup_steps: 50\nepochs: 1\nfp16: true\nreference_dtype: float32\nseed: 42\n"
    )
    return run_dir


def make_persona_run(run_dir):
    """TRL trainer.state.log_history format."""
    run_dir.mkdir(parents=True)
    history = []
    for step in range(2, 40, 2):
        history.append({"loss": 0.69 - step / 100, "rewards/accuracies": step / 40, "rewards/margins": step / 50,
                        "learning_rate": 5e-5, "epoch": step / 13, "step": step})
    for step in range(5, 40, 5):
        history.append({"eval_loss": 0.69 - step / 120, "eval_rewards/accuracies": step / 40,
                        "eval_rewards/margins": step / 60, "epoch": step / 13, "step": step})
    history.sort(key=lambda row: row["step"])
    history.append({"train_runtime": 100.0, "train_loss": 0.4, "epoch": 3.0, "step": 39})
    (run_dir / "log_history.json").write_text(json.dumps(history))
    (run_dir / "config_resolved.yaml").write_text(
        "model_name: HuggingFaceTB/SmolLM2-360M-Instruct\nn_eval: 20\nr: 16\nlora_alpha: 32\nlora_dropout: 0.05\n"
        "target_modules:\n- q_proj\n- v_proj\nbeta: 0.1\nlearning_rate: 5.0e-05\nper_device_train_batch_size: 4\n"
        "gradient_accumulation_steps: 4\nnum_train_epochs: 3\nmax_length: 512\nmax_prompt_length: 256\nseed: 42\n"
    )
    return run_dir


def assert_png(path):
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- loading and smoothing --------------------------------------------------------


def test_load_metrics_splits_on_type(tmp_path):
    run = make_sft_run(tmp_path / "sft")
    train, val = load_metrics(run / "metrics.jsonl")
    assert [r["step"] for r in train] == list(range(10, 101, 10))
    assert [r["step"] for r in val] == [0, 50, 100]
    assert all(r["type"] == "train" for r in train) and all(r["type"] == "val" for r in val)


def test_smooth_keeps_length_and_averages_centered():
    values = [0, 0, 3, 0, 0]
    smoothed = smooth(values, 3)
    assert len(smoothed) == 5
    assert smoothed == pytest.approx([0, 1, 1, 1, 0])
    assert smooth([1.0, 2.0], 1) == [1.0, 2.0]
    assert smooth([], 5) == []


# --- plots --------------------------------------------------------------------------


def test_plot_sft_writes_png(tmp_path):
    out = tmp_path / "figs" / "sft.png"
    assert plot_sft(make_sft_run(tmp_path / "sft"), out) is True
    assert_png(out)


def test_plot_dpo_writes_png(tmp_path):
    out = tmp_path / "dpo.png"
    assert plot_dpo(make_dpo_run(tmp_path / "dpo_hh"), out, "DPO on HH-RLHF") is True
    assert_png(out)


def test_plot_persona_writes_png(tmp_path):
    out = tmp_path / "persona.png"
    assert plot_persona(make_persona_run(tmp_path / "persona") / "log_history.json", out) is True
    assert_png(out)


def test_plots_without_best_json_still_work(tmp_path):
    run = make_sft_run(tmp_path / "sft")
    (run / "best.json").unlink()
    assert plot_sft(run, tmp_path / "x.png") is True


def test_missing_inputs_return_false(tmp_path, capsys):
    missing = tmp_path / "nope"
    assert plot_sft(missing, tmp_path / "a.png") is False
    assert plot_dpo(missing, tmp_path / "b.png", "t") is False
    assert plot_persona(missing / "log_history.json", tmp_path / "c.png") is False
    assert not list(tmp_path.glob("*.png"))
    assert "skip" in capsys.readouterr().out.lower()


# --- tables -------------------------------------------------------------------------


def table_rows(markdown):
    """{first cell: [other cells]} for a markdown table."""
    rows = {}
    for line in markdown.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 1 and not set(cells[0]) <= {"-", ":"}:
            rows[cells[0]] = cells[1:]
    return rows


def test_config_table(tmp_path):
    runs = {
        "SFT": make_sft_run(tmp_path / "sft"),
        "DPO-HH": make_dpo_run(tmp_path / "dpo_hh"),
        "Persona": make_persona_run(tmp_path / "persona"),
        "Missing": tmp_path / "missing",
    }
    stats = {"sft_train": {"final_rows": 30000}, "sft_val": {"final_rows": 1000},
             "hh_train": {"final_rows": 15000}, "hh_val": {"final_rows": 1000}}

    rows = table_rows(config_table(runs, data_stats=stats))

    assert rows["Setting"] == ["SFT", "DPO-HH", "Persona", "Missing"]
    assert rows["Effective batch"] == ["32", "32", "16", "—"]
    assert rows["Train / val examples"][:2] == ["30000 / 1000", "15000 / 1000"]
    assert rows["Init model"][:3] == ["Qwen/Qwen2.5-0.5B", "outputs/sft/best", "HuggingFaceTB/SmolLM2-360M-Instruct"]
    assert rows["Optimizer"][1] == "rmsprop"
    assert rows["Beta"] == ["—", "0.1", "0.1", "—"]
    assert rows["LoRA r / alpha / dropout"] == ["—", "—", "16 / 32 / 0.05", "—"]
    assert rows["LoRA targets"][2] == "q_proj, v_proj"
    assert rows["Reference dtype"][:2] == ["—", "float32"]
    assert rows["Warmup"][:2] == ["3% of steps", "50 steps"]
    assert all(cell == "—" for cell in (row[3] for row in rows.values() if row is not rows["Setting"]))


def test_results_table_formats_rates_and_ci(tmp_path):
    summary = {"model_x": "dpo_hh", "model_y": "sft", "eval_set": "hh_rlhf_eval_300", "n_judgments": 300,
               "x_wins": 150, "y_wins": 100, "ties": 40, "unknown": 10, "failures": 2,
               "win_rate_x_excl_ties": 0.6, "win_rate_x_ties_half": 150 / 290 + 20 / 290,
               "win_rate_x_ties_half_ci95": [0.53, 0.645]}
    empty = {**summary, "model_x": "a", "model_y": "b", "eval_set": None, "x_wins": 0, "y_wins": 0, "ties": 0,
             "win_rate_x_excl_ties": None, "win_rate_x_ties_half": None, "win_rate_x_ties_half_ci95": None}
    (tmp_path / "s1.json").write_text(json.dumps(summary))
    (tmp_path / "s2.json").write_text(json.dumps(empty))

    table = results_table({"dpo_hh vs sft": tmp_path / "s1.json", "a vs b": tmp_path / "s2.json",
                           "missing": tmp_path / "none.json"})
    rows = table_rows(table)

    assert rows["dpo_hh vs sft"] == ["hh_rlhf_eval_300", "300", "150", "100", "40", "10", "2",
                                     "60.0%", "58.6%", "[53.0%, 64.5%]"]
    assert rows["a vs b"][0] == "—" and rows["a vs b"][-3:] == ["—", "—", "—"]
    assert "missing" not in rows
