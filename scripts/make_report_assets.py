"""Build report figures and tables from whatever runs and judge summaries exist so far.

Inputs (all optional):
    report/runs/{sft,dpo_hh,dpo_uf}/        metrics.jsonl, best.json, config_resolved.yaml
    report/runs/persona_lora/               log_history.json, config_resolved.yaml
    report/results/*.summary.json           from scripts/summarize_judgments.py
    report/data_stats.json                  from scripts/prepare_data.py
Outputs:
    report/figures/*.png (150 dpi), report/tables/{config,generation,results}_table.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alignment.io_utils import read_json, write_text  # noqa: E402
from alignment.report_utils import (  # noqa: E402
    config_table,
    generation_table,
    plot_dpo,
    plot_persona,
    plot_sft,
    results_table,
)

RUNS = ROOT / "report" / "runs"
FIGURES = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
RESULTS = ROOT / "report" / "results"


def main():
    os.chdir(ROOT)  # config paths such as data_path are repo-relative
    made, skipped = [], []

    figures = [
        (FIGURES / "sft_training.png", lambda out: plot_sft(RUNS / "sft", out)),
        (FIGURES / "dpo_hh_training.png",
         lambda out: plot_dpo(RUNS / "dpo_hh", out, "DPO on HH-RLHF (helpful-base + harmless-base, single-turn)")),
        (FIGURES / "dpo_uf_training.png",
         lambda out: plot_dpo(RUNS / "dpo_uf", out, "DPO on UltraFeedback (binarized)")),
        (FIGURES / "persona_lora_training.png",
         lambda out: plot_persona(RUNS / "persona_lora" / "log_history.json", out)),
    ]
    for out, build in figures:
        (made if build(out) else skipped).append(out.relative_to(ROOT))

    stats_path = ROOT / "report" / "data_stats.json"
    data_stats = read_json(stats_path) if stats_path.exists() else None
    run_dirs = {"SFT": RUNS / "sft", "DPO-HH": RUNS / "dpo_hh", "DPO-UF": RUNS / "dpo_uf",
                "Persona LoRA": RUNS / "persona_lora"}
    config_md = TABLES / "config_table.md"
    write_text(config_md, config_table(run_dirs, data_stats))
    made.append(config_md.relative_to(ROOT))

    generation_md = TABLES / "generation_table.md"
    write_text(generation_md, generation_table(ROOT / "configs" / "generate.yaml"))
    made.append(generation_md.relative_to(ROOT))

    summaries = {}
    for path in sorted(RESULTS.glob("*.summary.json")):
        summary = json.loads(path.read_text())
        summaries[f"{summary['model_x']} vs {summary['model_y']} ({path.stem.removesuffix('.summary')})"] = path
    if not summaries:
        print(f"results table: no *.summary.json in {RESULTS.relative_to(ROOT)} yet; writing an empty table")
    results_md = TABLES / "results_table.md"
    write_text(results_md, results_table(summaries))
    made.append(results_md.relative_to(ROOT))

    print("\nMade:")
    for path in made:
        print(f"  {path}")
    print("Skipped (inputs missing):")
    for path in skipped or ["(none)"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
