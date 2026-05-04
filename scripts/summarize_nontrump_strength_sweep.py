#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_strength_map(items: str):
    mapping = []
    for chunk in items.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        tag, label = chunk.split('=', 1)
        mapping.append((tag.strip(), label.strip()))
    return mapping


def mean_or_none(series):
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def summarize_tag(results_dir: Path, tag: str, strength_label: str):
    emotion = pd.read_csv(results_dir / f'eval_nontrump_strength_{tag}_emotion.csv')
    novelty = pd.read_csv(results_dir / f'eval_nontrump_strength_{tag}_novelty.csv')
    wer = pd.read_csv(results_dir / f'eval_nontrump_strength_{tag}_wer.csv')
    mos = pd.read_csv(results_dir / f'eval_nontrump_strength_{tag}_mos.csv')

    emotion = emotion[emotion['style'] != 'baseline'].copy()
    novelty = novelty[novelty['style'] != 'baseline'].copy()
    wer = wer[wer['style'] != 'baseline'].copy()
    mos = mos[mos['style'] != 'baseline'].copy()

    merged = emotion.merge(
        novelty[['speaker', 'style', 'novelty_gain_vs_baseline', 'style_strength']],
        on=['speaker', 'style'],
        how='left',
    ).merge(
        wer[['speaker', 'style', 'wer']],
        on=['speaker', 'style'],
        how='left',
    ).merge(
        mos[['speaker', 'style', 'delta_vs_baseline']],
        on=['speaker', 'style'],
        how='left',
    )

    rows = []
    overall = {
        'strength_tag': tag,
        'strength': strength_label,
        'style': '__overall__',
        'n_rows': int(len(merged)),
        'recall': mean_or_none(pd.to_numeric(merged['match'], errors='coerce')),
        'emo_sim': mean_or_none(pd.to_numeric(merged['emo_sim'], errors='coerce')),
        'novelty_gain_vs_baseline': mean_or_none(pd.to_numeric(merged['novelty_gain_vs_baseline'], errors='coerce')),
        'wer': mean_or_none(pd.to_numeric(merged['wer'], errors='coerce')),
        'mos_delta': mean_or_none(pd.to_numeric(merged['delta_vs_baseline'], errors='coerce')),
    }
    rows.append(overall)

    for style, group in merged.groupby('style'):
        rows.append(
            {
                'strength_tag': tag,
                'strength': strength_label,
                'style': style,
                'n_rows': int(len(group)),
                'recall': mean_or_none(pd.to_numeric(group['match'], errors='coerce')),
                'emo_sim': mean_or_none(pd.to_numeric(group['emo_sim'], errors='coerce')),
                'novelty_gain_vs_baseline': mean_or_none(pd.to_numeric(group['novelty_gain_vs_baseline'], errors='coerce')),
                'wer': mean_or_none(pd.to_numeric(group['wer'], errors='coerce')),
                'mos_delta': mean_or_none(pd.to_numeric(group['delta_vs_baseline'], errors='coerce')),
            }
        )

    return pd.DataFrame(rows), sorted(merged['speaker'].unique().tolist())


def format_metric(value):
    if value is None or pd.isna(value):
        return '-'
    return f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--strengths', default='5p0=5.0,7p5=7.5,10p0=10.0,12p5=12.5')
    parser.add_argument('--focus-styles', default='whisper,happy,sad,confused')
    parser.add_argument('--out-csv', default='results/eval_nontrump_strength_sweep.csv')
    parser.add_argument('--out-md', default='results/eval_nontrump_strength_sweep_summary.md')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    strength_map = parse_strength_map(args.strengths)
    focus_styles = [style.strip() for style in args.focus_styles.split(',') if style.strip()]

    all_rows = []
    panel = None
    for tag, label in strength_map:
        df, speakers = summarize_tag(results_dir, tag, label)
        all_rows.append(df)
        if panel is None:
            panel = speakers

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(args.out_csv, index=False)

    overall = summary[summary['style'] == '__overall__'].copy()
    focus = summary[summary['style'].isin(focus_styles)].copy()

    lines = []
    lines.append('# Non-Trump Style-Strength Sweep Summary')
    lines.append('')
    lines.append('## Fixed panel')
    lines.append('')
    for speaker in panel or []:
        lines.append(f'- `{speaker}`')
    lines.append('')
    lines.append('## Overall metrics by strength')
    lines.append('')
    lines.append('| Strength | Recall | emo_sim | Novelty gain | Mean WER | Mean MOS delta |')
    lines.append('|----------|--------|---------|--------------|----------|----------------|')
    for _, row in overall.sort_values('strength', key=lambda s: s.astype(float)).iterrows():
        lines.append(
            f"| `{row['strength']}` | {format_metric(row['recall'])} | {format_metric(row['emo_sim'])} | {format_metric(row['novelty_gain_vs_baseline'])} | {format_metric(row['wer'])} | {format_metric(row['mos_delta'])} |"
        )
    lines.append('')
    lines.append('## Focus styles')
    lines.append('')
    lines.append('| Strength | Style | Recall | emo_sim | Novelty gain | WER | MOS delta |')
    lines.append('|----------|-------|--------|---------|--------------|-----|-----------|')
    focus_sorted = focus.sort_values(['strength', 'style'], key=lambda s: s if s.name == 'style' else s.astype(float))
    for _, row in focus_sorted.iterrows():
        lines.append(
            f"| `{row['strength']}` | `{row['style']}` | {format_metric(row['recall'])} | {format_metric(row['emo_sim'])} | {format_metric(row['novelty_gain_vs_baseline'])} | {format_metric(row['wer'])} | {format_metric(row['mos_delta'])} |"
        )
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- `Recall` is only defined for emotional styles that have a mapped target in the emotion classifier.')
    lines.append('- `Novelty gain` is positive when the styled output moves farther from the source than plain baseline conversion did.')
    lines.append('- `MOS delta` is relative to the same-speaker baseline conversion; values closer to `0.0` are better.')

    Path(args.out_md).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
