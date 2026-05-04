"""
Summarize Pass 6 CommonVoice objective-ablation results and collapse modes.

Expected input naming convention:
  results/eval_<metric>_pass6_<condition>.csv

Outputs:
  - results/eval_commonvoice_objective_summary_pass6.csv
  - results/eval_commonvoice_objective_collapse_pass6.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


FILENAME_RE = re.compile(r"eval_(emotion|wer|mos|novelty)_pass6_(.+)\.csv$")
CONTENT_COLLAPSE_WER = 0.8
IDENTITY_COLLAPSE_GAIN = 0.05
SUMMARY_ORDER = [
    "combined",
    "commonvoice_cv500_init",
    "cv500_ft_short_low_lr",
    "cv500_obj_label2",
    "cv500_obj_label4",
    "cv500_obj_label_ramp",
    "cv500_obj_recon_half_label2",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing eval_*_pass6_*.csv files (default: results)",
    )
    ap.add_argument(
        "--summary-out",
        default="results/eval_commonvoice_objective_summary_pass6.csv",
        help="Condition-level summary CSV output",
    )
    ap.add_argument(
        "--collapse-out",
        default="results/eval_commonvoice_objective_collapse_pass6.csv",
        help="Per-file collapse taxonomy CSV output",
    )
    return ap.parse_args()


def discover_metric_files(results_dir):
    grouped = defaultdict(dict)
    for path in sorted(Path(results_dir).glob("eval_*_pass6_*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        metric, condition = match.groups()
        grouped[condition][metric] = path
    if not grouped:
        raise SystemExit(f"No Pass 6 result CSVs found under {results_dir}")
    return grouped


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_mean(values):
    return statistics.mean(values) if values else None


def safe_fmt(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def safe_delta(value, reference):
    if value is None or reference is None:
        return ""
    return f"{value - reference:.4f}"


def summarize_condition(condition, metric_paths):
    summary = {
        "condition": condition,
        "styles_present": "",
        "styles_count": "",
        "sources_count": "",
        "rows_total": "",
        "emotion_rows_scored": "",
        "emotion_recall": "",
        "mean_emo_sim": "",
        "mean_wer": "",
        "mean_mos": "",
        "mean_mos_delta_vs_baseline": "",
        "mean_novelty_gain_vs_baseline": "",
        "content_collapse_count": 0,
        "style_collapse_to_neutral_count": 0,
        "identity_collapse_to_baseline_count": 0,
        "mixed_collapse_count": 0,
        "files_with_any_collapse": 0,
    }
    collapse_flags = defaultdict(lambda: {
        "condition": condition,
        "file": "",
        "speaker": "",
        "style": "",
        "content_collapse": 0,
        "style_collapse_to_neutral": 0,
        "identity_collapse_to_baseline": 0,
        "mixed_collapse": 0,
    })

    if "emotion" in metric_paths:
        emotion_rows = read_csv(metric_paths["emotion"])
        styles = sorted({row["style"] for row in emotion_rows if row["style"] != "baseline"})
        summary["styles_present"] = ",".join(styles)
        summary["styles_count"] = str(len(styles))
        summary["sources_count"] = str(len({row["speaker"] for row in emotion_rows}))
        summary["rows_total"] = str(len(emotion_rows))

        recall_rows = [row for row in emotion_rows if row["match"] in {"0", "1"}]
        if recall_rows:
            matches = sum(int(row["match"]) for row in recall_rows)
            summary["emotion_rows_scored"] = str(len(recall_rows))
            summary["emotion_recall"] = safe_fmt(matches / len(recall_rows))

        emo_sims = [float(row["emo_sim"]) for row in emotion_rows if row["emo_sim"]]
        summary["mean_emo_sim"] = safe_fmt(safe_mean(emo_sims))

        for row in emotion_rows:
            file_key = row["file"]
            collapse_flags[file_key]["file"] = row["file"]
            collapse_flags[file_key]["speaker"] = row["speaker"]
            collapse_flags[file_key]["style"] = row["style"]
            if row["target"] and row["target"] != "neutral" and row["predicted"] == "neutral":
                collapse_flags[file_key]["style_collapse_to_neutral"] = 1

    if "wer" in metric_paths:
        wer_rows = read_csv(metric_paths["wer"])
        scored = [
            float(row["wer"])
            for row in wer_rows
            if row["wer"] and row["style"] != "baseline" and row["ref_source"] != "self"
        ]
        summary["mean_wer"] = safe_fmt(safe_mean(scored))
        for row in wer_rows:
            file_key = row["file"]
            collapse_flags[file_key]["file"] = row["file"]
            collapse_flags[file_key]["speaker"] = row["speaker"]
            collapse_flags[file_key]["style"] = row["style"]
            if row["wer"] and row["style"] != "baseline" and float(row["wer"]) >= CONTENT_COLLAPSE_WER:
                collapse_flags[file_key]["content_collapse"] = 1

    if "mos" in metric_paths:
        mos_rows = read_csv(metric_paths["mos"])
        mos_vals = [float(row["mos"]) for row in mos_rows if row["mos"] and row["style"] != "baseline"]
        delta_vals = [
            float(row["delta_vs_baseline"])
            for row in mos_rows
            if row["delta_vs_baseline"] and row["style"] != "baseline"
        ]
        summary["mean_mos"] = safe_fmt(safe_mean(mos_vals))
        summary["mean_mos_delta_vs_baseline"] = safe_fmt(safe_mean(delta_vals))

    if "novelty" in metric_paths:
        novelty_rows = read_csv(metric_paths["novelty"])
        novelty_vals = [
            float(row["novelty_gain_vs_baseline"])
            for row in novelty_rows
            if row["novelty_gain_vs_baseline"] and row["style"] != "baseline"
        ]
        summary["mean_novelty_gain_vs_baseline"] = safe_fmt(safe_mean(novelty_vals))
        for row in novelty_rows:
            file_key = Path(row["generated_file"]).name
            collapse_flags[file_key]["file"] = file_key
            collapse_flags[file_key]["speaker"] = row["speaker"]
            collapse_flags[file_key]["style"] = row["style"]
            if (
                row["style"] != "baseline"
                and row["novelty_gain_vs_baseline"]
                and float(row["novelty_gain_vs_baseline"]) <= IDENTITY_COLLAPSE_GAIN
            ):
                collapse_flags[file_key]["identity_collapse_to_baseline"] = 1

    collapse_rows = []
    for row in collapse_flags.values():
        if row["style"] in {"", "baseline"}:
            continue
        axes = (
            row["content_collapse"]
            + row["style_collapse_to_neutral"]
            + row["identity_collapse_to_baseline"]
        )
        row["mixed_collapse"] = 1 if axes >= 2 else 0
        collapse_rows.append(row)

    summary["content_collapse_count"] = sum(row["content_collapse"] for row in collapse_rows)
    summary["style_collapse_to_neutral_count"] = sum(
        row["style_collapse_to_neutral"] for row in collapse_rows
    )
    summary["identity_collapse_to_baseline_count"] = sum(
        row["identity_collapse_to_baseline"] for row in collapse_rows
    )
    summary["mixed_collapse_count"] = sum(row["mixed_collapse"] for row in collapse_rows)
    summary["files_with_any_collapse"] = sum(
        1
        for row in collapse_rows
        if row["content_collapse"] or row["style_collapse_to_neutral"] or row["identity_collapse_to_baseline"]
    )

    return summary, sorted(collapse_rows, key=lambda row: (row["speaker"], row["style"]))


def as_float(summary_row, key):
    value = summary_row.get(key, "")
    return float(value) if value not in ("", None) else None


def add_delta_columns(summary_rows):
    index = {row["condition"]: row for row in summary_rows}
    combined = index.get("combined")
    cv500 = index.get("commonvoice_cv500_init")
    best_ft = index.get("cv500_ft_short_low_lr")

    combined_recall = as_float(combined, "emotion_recall") if combined else None
    combined_novelty = as_float(combined, "mean_novelty_gain_vs_baseline") if combined else None

    cv500_recall = as_float(cv500, "emotion_recall") if cv500 else None
    cv500_novelty = as_float(cv500, "mean_novelty_gain_vs_baseline") if cv500 else None
    cv500_wer = as_float(cv500, "mean_wer") if cv500 else None
    cv500_mos_delta = as_float(cv500, "mean_mos_delta_vs_baseline") if cv500 else None

    best_ft_recall = as_float(best_ft, "emotion_recall") if best_ft else None
    best_ft_novelty = as_float(best_ft, "mean_novelty_gain_vs_baseline") if best_ft else None
    best_ft_wer = as_float(best_ft, "mean_wer") if best_ft else None
    best_ft_mos_delta = as_float(best_ft, "mean_mos_delta_vs_baseline") if best_ft else None

    for row in summary_rows:
        row["delta_recall_vs_cv500"] = safe_delta(as_float(row, "emotion_recall"), cv500_recall)
        row["delta_novelty_vs_cv500"] = safe_delta(
            as_float(row, "mean_novelty_gain_vs_baseline"), cv500_novelty
        )
        row["delta_wer_vs_cv500"] = safe_delta(as_float(row, "mean_wer"), cv500_wer)
        row["delta_mos_delta_vs_cv500"] = safe_delta(
            as_float(row, "mean_mos_delta_vs_baseline"), cv500_mos_delta
        )
        row["delta_recall_vs_best_ft"] = safe_delta(
            as_float(row, "emotion_recall"), best_ft_recall
        )
        row["delta_novelty_vs_best_ft"] = safe_delta(
            as_float(row, "mean_novelty_gain_vs_baseline"), best_ft_novelty
        )
        row["delta_wer_vs_best_ft"] = safe_delta(as_float(row, "mean_wer"), best_ft_wer)
        row["delta_mos_delta_vs_best_ft"] = safe_delta(
            as_float(row, "mean_mos_delta_vs_baseline"), best_ft_mos_delta
        )
        row["delta_recall_vs_combined"] = safe_delta(
            as_float(row, "emotion_recall"), combined_recall
        )
        row["delta_novelty_vs_combined"] = safe_delta(
            as_float(row, "mean_novelty_gain_vs_baseline"), combined_novelty
        )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No rows to write to {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sort_summary_rows(rows):
    order = {name: idx for idx, name in enumerate(SUMMARY_ORDER)}
    return sorted(rows, key=lambda row: (order.get(row["condition"], 999), row["condition"]))


def main():
    args = parse_args()
    grouped = discover_metric_files(args.results_dir)

    summary_rows = []
    collapse_rows = []
    for condition, metric_paths in grouped.items():
        summary, detail = summarize_condition(condition, metric_paths)
        summary_rows.append(summary)
        collapse_rows.extend(detail)

    summary_rows = sort_summary_rows(summary_rows)
    add_delta_columns(summary_rows)
    collapse_rows = sorted(collapse_rows, key=lambda row: (row["condition"], row["speaker"], row["style"]))

    write_csv(args.summary_out, summary_rows)
    write_csv(args.collapse_out, collapse_rows)

    print("Pass 6 CommonVoice objective summary")
    print("=" * 96)
    for row in summary_rows:
        print(
            f"{row['condition']:30s} "
            f"recall={row['emotion_recall'] or 'n/a':>6s} "
            f"novelty={row['mean_novelty_gain_vs_baseline'] or 'n/a':>7s} "
            f"wer={row['mean_wer'] or 'n/a':>6s} "
            f"mosΔ={row['mean_mos_delta_vs_baseline'] or 'n/a':>7s} "
            f"Δrecall(best-ft)={row['delta_recall_vs_best_ft'] or 'n/a':>7s} "
            f"Δnovelty(best-ft)={row['delta_novelty_vs_best_ft'] or 'n/a':>7s}"
        )
    print(f"\nWrote {args.summary_out}")
    print(f"Wrote {args.collapse_out}")


if __name__ == "__main__":
    main()
