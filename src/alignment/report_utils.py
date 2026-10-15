"""Report figures and tables built from the trainers' logs and the judge summaries.

matplotlib is imported inside the plotting functions only (it is not a training dependency).
Figures use matplotlib.figure.Figure directly, so no GUI backend or pyplot state is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from alignment.io_utils import read_json, read_jsonl

DASH = "—"
DPI = 150
SMOOTH_WINDOW = 5


# --- loading -----------------------------------------------------------------------


def load_metrics(path):
    """metrics.jsonl from our trainers -> (train_rows, val_rows), split on "type"."""
    rows = read_jsonl(path)
    return [r for r in rows if r.get("type") == "train"], [r for r in rows if r.get("type") == "val"]


def smooth(values, window):
    """Centered moving average with the same length; the window shrinks at the edges."""
    values = list(values)
    if window <= 1 or not values:
        return values
    half = window // 2
    result = []
    for i in range(len(values)):
        chunk = values[max(0, i - half) : i + half + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def _best_step(run_dir):
    path = Path(run_dir) / "best.json"
    return read_json(path).get("step") if path.exists() else None


def _skip(what, path):
    print(f"skip {what}: {path} not found")
    return False


# --- plots -------------------------------------------------------------------------


def _new_figure(rows=1, cols=1, size=(7, 4.5)):
    from matplotlib.figure import Figure

    figure = Figure(figsize=size, layout="constrained")
    axes = figure.subplots(rows, cols, squeeze=False)
    return figure, axes


def _save(figure, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=DPI)
    print(f"wrote {out_png}")
    return True


def _mark_best(ax, best_step):
    if best_step is not None:
        ax.axvline(best_step, color="grey", linestyle="--", linewidth=1, label=f"best (step {best_step})")


def plot_sft(run_dir, out_png):
    """Train loss (raw + smoothed) and val loss vs optimizer step, best step marked."""
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return _skip("SFT plot", metrics_path)
    train, val = load_metrics(metrics_path)
    steps = [r["step"] for r in train]
    losses = [r["train_loss"] for r in train]

    figure, axes = _new_figure()
    ax = axes[0][0]
    ax.plot(steps, losses, color="tab:blue", alpha=0.3, linewidth=1, label="train loss (raw)")
    ax.plot(steps, smooth(losses, SMOOTH_WINDOW), color="tab:blue", linewidth=2,
            label=f"train loss (moving avg, {SMOOTH_WINDOW})")
    ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", color="tab:orange",
            label="val loss (token-weighted)")
    _mark_best(ax, _best_step(run_dir))
    ax.set_title("SFT: response-only NLL")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (nats / response token)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(figure, out_png)


def plot_dpo(run_dir, out_png, title):
    """2x2 panel: loss, reward accuracy, reward margin (train smoothed + val) and val rewards."""
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return _skip(f"DPO plot ({title})", metrics_path)
    train, val = load_metrics(metrics_path)
    best_step = _best_step(run_dir)
    train_steps = [r["step"] for r in train]
    val_steps = [r["step"] for r in val]

    figure, axes = _new_figure(2, 2, size=(11, 7.5))
    panels = [
        (axes[0][0], "loss", "(a) DPO loss", "loss"),
        (axes[0][1], "reward_accuracy", "(b) reward accuracy", "accuracy"),
        (axes[1][0], "reward_margin", "(c) reward margin", "margin (r_chosen - r_rejected)"),
    ]
    for ax, key, panel_title, ylabel in panels:
        ax.plot(train_steps, smooth([r[key] for r in train], SMOOTH_WINDOW), color="tab:blue",
                label=f"train (moving avg, {SMOOTH_WINDOW})")
        ax.plot(val_steps, [r[key] for r in val], "o-", color="tab:orange", label="val")
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
    axes[0][1].axhline(0.5, color="black", linestyle=":", linewidth=1, label="chance (0.5)")

    ax = axes[1][1]
    ax.plot(val_steps, [r["chosen_reward"] for r in val], "o-", color="tab:green", label="val chosen reward")
    ax.plot(val_steps, [r["rejected_reward"] for r in val], "s-", color="tab:red", label="val rejected reward")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_title("(d) val implicit rewards")
    ax.set_ylabel("beta * (log pi - log pi_ref)")

    for row in axes:
        for ax in row:
            _mark_best(ax, best_step)
            ax.set_xlabel("optimizer step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    figure.suptitle(title)
    return _save(figure, out_png)


def plot_persona(log_history_path, out_png):
    """TRL log_history: train/eval loss and eval reward accuracy vs step."""
    log_history_path = Path(log_history_path)
    if not log_history_path.exists():
        return _skip("persona plot", log_history_path)
    history = read_json(log_history_path)
    train = [r for r in history if "loss" in r and "step" in r]
    evals = [r for r in history if "eval_loss" in r and "step" in r]

    figure, axes = _new_figure(1, 2, size=(11, 4.5))
    ax = axes[0][0]
    ax.plot([r["step"] for r in train], [r["loss"] for r in train], "-", color="tab:blue", label="train loss")
    ax.plot([r["step"] for r in evals], [r["eval_loss"] for r in evals], "o-", color="tab:orange", label="eval loss")
    ax.set_title("(a) DPO loss")
    ax.set_ylabel("loss")

    ax = axes[0][1]
    train_acc = [r for r in train if "rewards/accuracies" in r]
    if train_acc:
        ax.plot([r["step"] for r in train_acc], [r["rewards/accuracies"] for r in train_acc], "-",
                color="tab:blue", alpha=0.5, label="train reward accuracy")
    eval_acc = [r for r in evals if "eval_rewards/accuracies" in r]
    ax.plot([r["step"] for r in eval_acc], [r["eval_rewards/accuracies"] for r in eval_acc], "o-",
            color="tab:orange", label="eval reward accuracy")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="chance (0.5)")
    ax.set_title("(b) reward accuracy")
    ax.set_ylabel("accuracy")

    for ax in axes[0]:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    figure.suptitle("Persona LoRA (TRL DPOTrainer)")
    return _save(figure, out_png)


# --- tables ------------------------------------------------------------------------


def _fmt(value):
    if value is None or value == "":
        return DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _first(cfg, *keys):
    for key in keys:
        if cfg.get(key) is not None:
            return cfg[key]
    return None


def _kind(cfg):
    if "r" in cfg or "lora_alpha" in cfg:
        return "persona"
    if "beta" in cfg:
        return "dpo"
    return "sft"


def _data_sizes(cfg, kind, data_stats):
    if kind == "persona":
        n_eval = cfg.get("n_eval")
        path = Path(cfg.get("data_path", ""))
        total = len(read_json(path)) if cfg.get("data_path") and path.exists() else None
        train = total - n_eval if total is not None and n_eval is not None else None
        return f"{_fmt(train)} / {_fmt(n_eval)}"
    stats = data_stats or {}

    def size(key):
        name = Path(cfg[key]).stem if cfg.get(key) else None
        return stats.get(name, {}).get("final_rows") if name else None

    train, val = size("train_file"), size("val_file")
    return DASH if train is None and val is None else f"{_fmt(train)} / {_fmt(val)}"


def _config_column(cfg, data_stats):
    if not cfg:
        return {}
    kind = _kind(cfg)
    batch = _first(cfg, "batch_size", "per_device_train_batch_size")
    accum = _first(cfg, "grad_accum", "gradient_accumulation_steps")
    warmup = None
    if cfg.get("warmup_ratio") is not None:
        warmup = f"{cfg['warmup_ratio']:.0%} of steps"
    elif cfg.get("warmup_steps") is not None:
        warmup = f"{cfg['warmup_steps']} steps"
    lora = None
    if kind == "persona":
        lora = f"{_fmt(cfg.get('r'))} / {_fmt(cfg.get('lora_alpha'))} / {_fmt(cfg.get('lora_dropout'))}"
    return {
        "Init model": _first(cfg, "sft_model_path", "model_name"),
        "Train / val examples": _data_sizes(cfg, kind, data_stats),
        "Max length": cfg.get("max_length"),
        "Max prompt length": cfg.get("max_prompt_length"),
        "Batch size": batch,
        "Grad accum": accum,
        "Effective batch": batch * accum if batch is not None and accum is not None else None,
        "Optimizer": {"sft": "adamw", "dpo": cfg.get("optimizer"), "persona": "adamw (TRL default)"}[kind],
        "Learning rate": _first(cfg, "lr", "learning_rate"),
        "Weight decay": cfg.get("weight_decay"),
        "Warmup": warmup,
        "LR schedule": {"sft": "linear warmup + cosine to 0", "dpo": "constant after linear warmup",
                        "persona": "linear decay (TRL default)"}[kind],
        "Beta": cfg.get("beta"),
        "Epochs": _first(cfg, "epochs", "num_train_epochs"),
        "Max steps": cfg.get("max_steps"),
        "Precision": "fp16 autocast, fp32 weights" if cfg.get("fp16", kind == "persona") else "fp32",
        "Reference dtype": {"sft": None, "dpo": cfg.get("reference_dtype"),
                            "persona": "base model, adapter disabled"}[kind],
        "LoRA r / alpha / dropout": lora,
        "LoRA targets": cfg.get("target_modules"),
        "Seed": cfg.get("seed"),
    }


CONFIG_ROWS = (
    "Init model", "Train / val examples", "Max length", "Max prompt length", "Batch size", "Grad accum",
    "Effective batch", "Optimizer", "Learning rate", "Weight decay", "Warmup", "LR schedule", "Beta",
    "Epochs", "Max steps", "Precision", "Reference dtype", "LoRA r / alpha / dropout", "LoRA targets", "Seed",
)


def config_table(run_dirs, data_stats=None):
    """Markdown table of hyperparameters actually used (from each run's config_resolved.yaml).

    run_dirs: {column name: run dir}. data_stats: parsed data_stats.json for train/val sizes.
    """
    columns = []
    for name, run_dir in run_dirs.items():
        path = Path(run_dir) / "config_resolved.yaml"
        cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
        if not cfg:
            print(f"config table: {path} not found; column '{name}' is empty")
        columns.append(_config_column(cfg or {}, data_stats))

    lines = ["| Setting | " + " | ".join(run_dirs) + " |", "|---" * (len(run_dirs) + 1) + "|"]
    for row in CONFIG_ROWS:
        lines.append(f"| {row} | " + " | ".join(_fmt(column.get(row)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _pct(value):
    return DASH if value is None else f"{value:.1%}"


def results_table(summary_paths):
    """Markdown table of pairwise judge results from summarize_judgments.py JSON files.

    summary_paths: {comparison label: summary path}. Missing files are skipped.
    """
    header = ("Comparison", "Eval set", "n", "Wins", "Losses", "Ties", "Unknown", "Failures",
              "Win rate (excl. ties)", "Win rate (ties = 0.5)", "95% CI (ties = 0.5)")
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for label, path in summary_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"results table: {path} not found; skipping '{label}'")
            continue
        s = json.loads(path.read_text())
        ci = s.get("win_rate_x_ties_half_ci95")
        cells = (
            label, _fmt(s.get("eval_set")), s["n_judgments"], s["x_wins"], s["y_wins"], s["ties"],
            s["unknown"], s.get("failures", 0), _pct(s.get("win_rate_x_excl_ties")),
            _pct(s.get("win_rate_x_ties_half")), DASH if not ci else f"[{ci[0]:.1%}, {ci[1]:.1%}]",
        )
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines += ["", "Wins / losses / win rates are for the first model named in each comparison."]
    return "\n".join(lines) + "\n"
