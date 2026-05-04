"""
Summarize mixed-data training results and collapse modes.

Expected input naming convention:
  results/eval_<metric>_<tag>_<condition>.csv

Examples:
  - results/eval_emotion_pass9_mixed_static_balanced.csv
  - results/eval_emotion_mixed_quality_mixed_quality_labeled_guarded.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


CONTENT_COLLAPSE_WER = 0.8
IDENTITY_COLLAPSE_GAIN = 0.05
SUMMARY_ORDER = [
    "combined",
    "commonvoice_cv500_init",
    "cv500_ft_short_low_lr",
    "cv500_rich_free_anchor",
    "cv500_pl_meta",
    "mixed_static_balanced",
    "mixed_cv_warmup",
    "mixed_labeled_finish",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--input-tag",
        default="pass9",
        help="Result tag used in the CSV filenames, e.g. pass9 or mixed_quality",
    )
    ap.add_argument(
        "--summary-out",
        default=None,
    )
    ap.add_argument(
        "--collapse-out",
        default=None,
    )
    return ap.parse_args()


def discover_metric_files(results_dir, input_tag):
    grouped = defaultdict(dict)
    prefix = f"eval_"
    infix = f"_{input_tag}_"
    suffix = ".csv"
    for path in sorted(Path(results_dir).glob(f"eval_*_{input_tag}_*.csv")):
        name = path.name
        if not (name.startswith(prefix) and infix in name and name.endswith(suffix)):
            continue
        metric_condition = name[len(prefix):-len(suffix)]
        metric, condition = metric_condition.split(infix, 1)
        if metric not in {"emotion", "wer", "mos", "novelty"}:
            continue
        grouped[condition][metric] = path
    if not grouped:
        raise SystemExit(f"No mixed-data result CSVs found under {results_dir} for tag {input_tag}")
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
    collapse_flags = defaultdict(
        lambda: {
            "condition": condition,
            "file": "",
            "speaker": "",
            "style": "",
            "content_collapse": 0,
            "style_collapse_to_neutral": 0,
            "identity_collapse_to_baseline": 0,
            "mixed_collapse": 0,
        }
    )

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
    summary["style_collapse_to_neutral_count"] = sum(row["style_collapse_to_neutral"] for row in collapse_rows)
    summary["identity_collapse_to_baseline_count"] = sum(row["identity_collapse_to_baseline"] for row in collapse_rows)
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
    refs = {
        "combined": index.get("combined"),
        "cv500": index.get("commonvoice_cv500_init"),
        "best_ft": index.get("cv500_ft_short_low_lr"),
        "best_rich": index.get("cv500_rich_free_anchor"),
        "best_partial": index.get("cv500_pl_meta"),
    }
    for name, ref in list(refs.items()):
        refs[name] = {
            "recall": as_float(ref, "emotion_recall") if ref else None,
            "novelty": as_float(ref, "mean_novelty_gain_vs_baseline") if ref else None,
            "wer": as_float(ref, "mean_wer") if ref else None,
            "mos_delta": as_float(ref, "mean_mos_delta_vs_baseline") if ref else None,
        }

    for row in summary_rows:
        for prefix, ref in refs.items():
            row[f"delta_recall_vs_{prefix}"] = safe_delta(as_float(row, "emotion_recall"), ref["recall"])
            row[f"delta_novelty_vs_{prefix}"] = safe_delta(
                as_float(row, "mean_novelty_gain_vs_baseline"), ref["novelty"]
            )
            row[f"delta_wer_vs_{prefix}"] = safe_delta(as_float(row, "mean_wer"), ref["wer"])
            row[f"delta_mos_delta_vs_{prefix}"] = safe_delta(
                as_float(row, "mean_mos_delta_vs_baseline"), ref["mos_delta"]
            )


def write_csv(path, fieldnames, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.summary_out is None:
        args.summary_out = f"results/eval_{args.input_tag}_summary.csv"
    if args.collapse_out is None:
        args.collapse_out = f"results/eval_{args.input_tag}_collapse.csv"

    grouped = discover_metric_files(args.results_dir, args.input_tag)

    summary_rows = []
    collapse_rows = []
    for condition in SUMMARY_ORDER:
        metric_paths = grouped.get(condition)
        if not metric_paths:
            continue
        summary, collapses = summarize_condition(condition, metric_paths)
        summary_rows.append(summary)
        collapse_rows.extend(collapses)

    extras = sorted(set(grouped) - set(SUMMARY_ORDER))
    for condition in extras:
        summary, collapses = summarize_condition(condition, grouped[condition])
        summary_rows.append(summary)
        collapse_rows.extend(collapses)

    add_delta_columns(summary_rows)

    summary_fields = list(summary_rows[0].keys())
    collapse_fields = list(collapse_rows[0].keys())
    write_csv(args.summary_out, summary_fields, summary_rows)
    write_csv(args.collapse_out, collapse_fields, collapse_rows)

    print(f"Wrote summary to {args.summary_out}")
    print(f"Wrote collapse taxonomy to {args.collapse_out}")


if __name__ == "__main__":
    main()
