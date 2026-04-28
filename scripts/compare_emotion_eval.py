"""Compare two eval_emotion CSV files and report recall deltas."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
          'happy', 'neutral', 'sad', 'whisper']


def load_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    summary = {
        'files': len(rows),
        'overall_correct': 0,
        'overall_total': 0,
        'per_style': defaultdict(lambda: {'correct': 0, 'total': 0, 'emo_sim': []}),
    }
    for row in rows:
        style = row['style']
        if row.get('match') in ('0', '1'):
            summary['overall_total'] += 1
            summary['overall_correct'] += int(row['match'])
            summary['per_style'][style]['total'] += 1
            summary['per_style'][style]['correct'] += int(row['match'])
        if row.get('emo_sim'):
            summary['per_style'][style]['emo_sim'].append(float(row['emo_sim']))
    return summary


def rate(correct, total):
    return (correct / total) if total else None


def format_rate(value):
    if value is None:
        return 'n/a'
    return f"{100 * value:.1f}%"


def format_delta(base, candidate):
    if base is None or candidate is None:
        return 'n/a'
    delta = 100 * (candidate - base)
    return f"{delta:+.1f} pts"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True, help='Baseline eval_emotion CSV')
    ap.add_argument('--candidate', required=True, help='Candidate eval_emotion CSV')
    args = ap.parse_args()

    baseline_rows = load_rows(args.baseline)
    candidate_rows = load_rows(args.candidate)
    baseline = summarize(baseline_rows)
    candidate = summarize(candidate_rows)

    print('Emotion recall comparison')
    print('=' * 60)
    print(f"Baseline CSV : {Path(args.baseline).resolve()}")
    print(f"Candidate CSV: {Path(args.candidate).resolve()}")
    print()
    print(f"Files compared       : baseline={baseline['files']}  candidate={candidate['files']}")
    base_overall = rate(baseline['overall_correct'], baseline['overall_total'])
    cand_overall = rate(candidate['overall_correct'], candidate['overall_total'])
    print(f"Overall recall       : {format_rate(base_overall)} -> {format_rate(cand_overall)} ({format_delta(base_overall, cand_overall)})")
    print()
    print('Per-style recall:')
    for style in STYLES:
        base_style = baseline['per_style'][style]
        cand_style = candidate['per_style'][style]
        base_rate = rate(base_style['correct'], base_style['total'])
        cand_rate = rate(cand_style['correct'], cand_style['total'])
        print(
            f"  {style:12s} {format_rate(base_rate):>8s} -> {format_rate(cand_rate):>8s}  {format_delta(base_rate, cand_rate):>9s}"
        )


if __name__ == '__main__':
    main()
