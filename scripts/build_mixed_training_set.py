"""
Build a sampled mixed-data OpenVoice training artifact from CommonVoice,
CREMA-D, and Expresso.

This script is the first implementation step for the mixed-data bootstrap line:
- keep CommonVoice for speaker breadth,
- keep CREMA-D and Expresso for emotion/style supervision,
- and preserve enough metadata that later schedule and evaluation results are
  interpretable.

The output artifact is designed for `examples/openvoice_train_vae_mixed.py`.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


UNIFIED_STYLES = [
    "anger",
    "confused",
    "disgust",
    "enunciated",
    "fear",
    "happy",
    "neutral",
    "sad",
    "whisper",
]

CREMAD_STYLES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EXPRESSO_STYLES = [
    "default",
    "confused",
    "enunciated",
    "happy",
    "laughing",
    "sad",
    "whisper",
    "emphasis",
    "essentials",
    "longform",
    "singing",
]
EXPRESSO_MAP = {
    "default": "neutral",
    "confused": "confused",
    "enunciated": "enunciated",
    "happy": "happy",
    "sad": "sad",
    "whisper": "whisper",
}
EXPRESSO_ONLY = {"confused", "enunciated", "whisper"}
EXPRESSO_SUPPORTED = ["confused", "enunciated", "happy", "neutral", "sad", "whisper"]
DATASETS = ["CommonVoice", "CREMA-D", "Expresso"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--commonvoice",
        default="embeddings/openvoice_commonvoice_cv500_pseudo.pt",
        help="Pseudo-labeled CommonVoice embedding artifact",
    )
    ap.add_argument(
        "--cremad",
        default="embeddings/openvoice_cremad_emb.pt",
        help="CREMA-D embedding artifact",
    )
    ap.add_argument(
        "--expresso",
        default="embeddings/openvoice_expresso_emb.pt",
        help="Expresso embedding artifact",
    )
    ap.add_argument(
        "--parquet-dir",
        default=None,
        help="Expresso parquet directory. Auto-detected from the HF cache if omitted.",
    )
    ap.add_argument(
        "--output",
        default="embeddings/openvoice_mixed_base.pt",
        help="Output mixed-data artifact (.pt)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for all sampling (default: 42)",
    )
    ap.add_argument(
        "--commonvoice-max-speakers",
        type=int,
        default=None,
        help="Optional cap on CommonVoice speaker count after speaker-first sampling",
    )
    ap.add_argument(
        "--commonvoice-min-clips-per-speaker",
        type=int,
        default=1,
        help="Minimum clips to try to keep per CommonVoice speaker (default: 1)",
    )
    ap.add_argument(
        "--commonvoice-max-clips-per-speaker",
        type=int,
        default=1,
        help="Maximum CommonVoice clips per speaker (default: 1)",
    )
    ap.add_argument(
        "--commonvoice-prefer-pseudo",
        action="store_true",
        help="Prefer pseudo-labeled CommonVoice rows when choosing per-speaker clips",
    )
    ap.add_argument(
        "--pseudo-style-threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for treating CommonVoice pseudo styles as labeled (default: 0.60)",
    )
    ap.add_argument(
        "--pseudo-style-thresholds",
        default="",
        help=(
            "Optional per-style pseudo-label thresholds, e.g. "
            "neutral=0.995,sad=0.98,happy=0.92"
        ),
    )
    ap.add_argument(
        "--commonvoice-style-caps",
        default="",
        help="Optional per-style cap for selected CommonVoice pseudo labels, e.g. neutral=150,sad=120",
    )
    ap.add_argument(
        "--expresso-only-cap",
        type=int,
        default=90,
        help="Cap Expresso-only styles (confused/enunciated/whisper) to this many samples (default: 90)",
    )
    ap.add_argument(
        "--pseudo-confidence-scale",
        action="store_true",
        help="Scale CommonVoice pseudo-label row weights by the pseudo-label confidence",
    )
    ap.add_argument(
        "--pseudo-row-weight",
        type=float,
        default=1.0,
        help="Base row-weight multiplier for CommonVoice pseudo labels (default: 1.0)",
    )
    ap.add_argument(
        "--true-row-weight",
        type=float,
        default=1.0,
        help="Base row-weight multiplier for true labels from CREMA-D/Expresso (default: 1.0)",
    )
    args = ap.parse_args()
    if (
        args.commonvoice_max_clips_per_speaker is not None
        and args.commonvoice_max_clips_per_speaker < args.commonvoice_min_clips_per_speaker
    ):
        ap.error(
            "--commonvoice-max-clips-per-speaker must be >= "
            "--commonvoice-min-clips-per-speaker"
        )
    return args


def parse_style_caps(raw: str) -> Dict[str, int]:
    if not raw.strip():
        return {}
    caps = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected style cap in name=count form, got: {item}")
        name, value = item.split('=', 1)
        style = name.strip()
        if style not in UNIFIED_STYLES:
            raise ValueError(f"Unknown style in --commonvoice-style-caps: {style}")
        caps[style] = int(value.strip())
    return caps


def parse_style_thresholds(raw: str, default: float) -> Dict[str, float]:
    thresholds = {style: float(default) for style in UNIFIED_STYLES}
    if not raw.strip():
        return thresholds
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected pseudo threshold in name=value form, got: {item}")
        name, value = item.split('=', 1)
        style = name.strip()
        if style not in UNIFIED_STYLES:
            raise ValueError(f"Unknown style in --pseudo-style-thresholds: {style}")
        thresholds[style] = float(value.strip())
    return thresholds


def resolve_expresso_parquet_dir(parquet_dir: Optional[str]) -> Path:
    if parquet_dir:
        path = Path(parquet_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Expresso parquet dir not found: {path}")
        return path

    matches = sorted(
        glob(
            str(
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "datasets--ylacombe--expresso"
                / "snapshots"
                / "*"
                / "read"
            )
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-detect an Expresso parquet cache. Pass --parquet-dir explicitly."
        )
    return Path(matches[-1])


def onehot_target(style: str) -> List[float]:
    values = [-1.0] * len(UNIFIED_STYLES)
    values[UNIFIED_STYLES.index(style)] = 1.0
    return values


def build_empty_target() -> List[float]:
    return [0.0] * len(UNIFIED_STYLES)


def load_expresso_metadata(expresso_data, parquet_dir: Path) -> pd.DataFrame:
    files = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.endswith('.parquet')
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    df = pd.concat(
        [pd.read_parquet(path, columns=['speaker_id', 'style']) for path in files],
        ignore_index=True,
    )
    if len(df) != len(expresso_data['data']):
        if 'ids' not in expresso_data:
            raise ValueError(
                "Expresso parquet metadata length does not match the extracted embeddings, "
                "and no ids field is available to realign them."
            )
        row_indices = [int(x) for x in expresso_data['ids'].view(-1).tolist()]
        df = df.iloc[row_indices].reset_index(drop=True)
        if len(df) != len(expresso_data['data']):
            raise ValueError(
                "Failed to realign Expresso parquet metadata with the extracted embeddings."
            )
    return df


def resolve_cremad_rows(cremad_data):
    rows = []
    for idx in range(len(cremad_data['data'])):
        style = None
        for candidate in CREMAD_STYLES:
            if cremad_data[f'emotion_{candidate}'][idx].item() > 0:
                style = candidate
                break
        if style is None:
            raise ValueError(f"Failed to resolve a CREMA-D style for row {idx}")
        speaker_id = str(int(cremad_data['speakers'][idx].item())) if 'speakers' in cremad_data else f"cremad_{idx}"
        rows.append({
            'dataset': 'CREMA-D',
            'row_idx': idx,
            'speaker_id': speaker_id,
            'clip_path': None,
            'style': style,
            'label_source': 'true',
            'label_confidence': 1.0,
        })
    return rows


def resolve_expresso_rows(expresso_data, parquet_df, cap, seed):
    expresso_by_label = defaultdict(list)
    seen_shared = set()
    for row_idx in range(len(expresso_data['data'])):
        source_style = None
        for style in EXPRESSO_STYLES:
            if expresso_data[f'style_{style}'][row_idx].item() > 0:
                source_style = style
                break
        if source_style not in EXPRESSO_MAP:
            continue
        unified = EXPRESSO_MAP[source_style]
        if unified in EXPRESSO_ONLY:
            expresso_by_label[unified].append(row_idx)
            continue
        speaker_id = str(parquet_df.iloc[row_idx]['speaker_id'])
        key = (speaker_id, unified)
        if key not in seen_shared:
            seen_shared.add(key)
            expresso_by_label[unified].append(row_idx)

    rng = np.random.default_rng(seed)
    rows = []
    for style in EXPRESSO_SUPPORTED:
        indices = list(expresso_by_label.get(style, []))
        if style in EXPRESSO_ONLY and len(indices) > cap:
            indices = sorted(rng.choice(indices, cap, replace=False).tolist())
        for row_idx in indices:
            rows.append({
                'dataset': 'Expresso',
                'row_idx': row_idx,
                'speaker_id': str(parquet_df.iloc[row_idx]['speaker_id']),
                'clip_path': None,
                'style': style,
                'label_source': 'true',
                'label_confidence': 1.0,
            })
    return rows


def accepted_commonvoice_style(cv_data, row_idx: int, threshold_map: Dict[str, float]):
    style = cv_data.get('pseudo_style', [None])[row_idx]
    confidence = cv_data.get('pseudo_style_confidence', [None])[row_idx]
    if style is None or confidence is None:
        return None, None
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None, None
    if style not in UNIFIED_STYLES:
        return None, confidence
    if confidence < threshold_map.get(style, 0.0):
        return None, confidence
    return style, confidence


def select_commonvoice_rows(cv_data, args, style_caps, threshold_map):
    speaker_to_rows = defaultdict(list)
    for row_idx, speaker_id in enumerate(cv_data['speaker_ids']):
        speaker_to_rows[str(speaker_id)].append(row_idx)

    speakers = list(speaker_to_rows.keys())
    random.Random(args.seed).shuffle(speakers)
    if args.commonvoice_max_speakers is not None:
        speakers = speakers[:args.commonvoice_max_speakers]

    rng = random.Random(args.seed)
    selected = []
    selected_style_counts = Counter()
    skipped_by_cap_counts = Counter()

    for speaker_id in speakers:
        candidates = speaker_to_rows[speaker_id][:]
        rng.shuffle(candidates)

        def sort_key(row_idx):
            style, conf = accepted_commonvoice_style(cv_data, row_idx, threshold_map)
            has_label = style is not None
            is_neutral = style == 'neutral'
            priority = 0
            if args.commonvoice_prefer_pseudo:
                if has_label and not is_neutral:
                    priority = 0
                elif has_label and is_neutral:
                    priority = 1
                else:
                    priority = 2
            return (priority, -(conf or -1.0), row_idx)

        ordered = sorted(candidates, key=sort_key)
        target_count = len(ordered)
        if args.commonvoice_max_clips_per_speaker is not None:
            target_count = min(target_count, args.commonvoice_max_clips_per_speaker)
        if target_count <= 0:
            continue

        chosen = []
        used = set()
        for row_idx in ordered:
            if len(chosen) >= target_count:
                break
            style, confidence = accepted_commonvoice_style(cv_data, row_idx, threshold_map)
            if style is not None and style in style_caps and selected_style_counts[style] >= style_caps[style]:
                skipped_by_cap_counts[style] += 1
                continue
            chosen.append((row_idx, style, confidence))
            used.add(row_idx)
            if style is not None:
                selected_style_counts[style] += 1

        if len(chosen) < target_count:
            for row_idx in ordered:
                if row_idx in used:
                    continue
                style, confidence = accepted_commonvoice_style(cv_data, row_idx, threshold_map)
                if style is not None and style in style_caps and selected_style_counts[style] >= style_caps[style]:
                    skipped_by_cap_counts[style] += 1
                    style, confidence = None, None
                chosen.append((row_idx, style, confidence))
                used.add(row_idx)
                if style is not None:
                    selected_style_counts[style] += 1
                if len(chosen) >= target_count:
                    break

        for row_idx, style, confidence in chosen:
            selected.append({
                'dataset': 'CommonVoice',
                'row_idx': row_idx,
                'speaker_id': str(cv_data['speaker_ids'][row_idx]),
                'clip_path': cv_data['clip_paths'][row_idx] if 'clip_paths' in cv_data else None,
                'style': style,
                'label_source': 'pseudo' if style is not None else 'none',
                'label_confidence': float(confidence) if confidence is not None else 0.0,
            })

    selection_report = {
        'skipped_by_cap_counts': dict(skipped_by_cap_counts),
    }
    return selected, selection_report


def build_rows(args):
    cv_data = torch.load(args.commonvoice, weights_only=False, map_location='cpu')
    cremad_data = torch.load(args.cremad, weights_only=False, map_location='cpu')
    expresso_data = torch.load(args.expresso, weights_only=False, map_location='cpu')
    parquet_dir = resolve_expresso_parquet_dir(args.parquet_dir)
    expresso_df = load_expresso_metadata(expresso_data, parquet_dir)

    style_caps = parse_style_caps(args.commonvoice_style_caps)
    threshold_map = parse_style_thresholds(
        args.pseudo_style_thresholds,
        args.pseudo_style_threshold,
    )
    commonvoice_rows, commonvoice_selection_report = select_commonvoice_rows(
        cv_data,
        args,
        style_caps,
        threshold_map,
    )
    cremad_rows = resolve_cremad_rows(cremad_data)
    expresso_rows = resolve_expresso_rows(expresso_data, expresso_df, args.expresso_only_cap, args.seed)

    rows = commonvoice_rows + cremad_rows + expresso_rows
    payloads = {
        'CommonVoice': cv_data,
        'CREMA-D': cremad_data,
        'Expresso': expresso_data,
    }
    return rows, payloads, parquet_dir, threshold_map, commonvoice_selection_report


def compute_style_row_weights(rows, args):
    style_counts = Counter(row['style'] for row in rows if row['style'] is not None)
    total = sum(style_counts.values())
    present = len(style_counts) or 1
    for row in rows:
        if row['style'] is None:
            row['style_row_weight'] = 0.0
            continue
        base = total / (present * style_counts[row['style']])
        if row['label_source'] == 'pseudo':
            base *= args.pseudo_row_weight
            if args.pseudo_confidence_scale:
                base *= row['label_confidence']
        else:
            base *= args.true_row_weight
        row['style_row_weight'] = float(base)


def build_save_dict(rows, payloads, args, parquet_dir, threshold_map, commonvoice_selection_report):
    compute_style_row_weights(rows, args)

    embeddings = []
    targets = defaultdict(list)
    speaker_ids = []
    clip_paths = []
    source_datasets = []
    style_sources = []
    style_confidences = []
    style_mask = []
    style_row_weights = []

    for row in rows:
        source_payload = payloads[row['dataset']]
        emb = source_payload['data'][row['row_idx']]
        embeddings.append(emb)

        if row['style'] is None:
            target = build_empty_target()
            mask = 0.0
        else:
            target = onehot_target(row['style'])
            mask = 1.0

        for idx, style_name in enumerate(UNIFIED_STYLES):
            targets[f'label_{style_name}'].append(torch.tensor(target[idx], dtype=torch.float32))

        speaker_ids.append(row['speaker_id'])
        clip_paths.append(row['clip_path'])
        source_datasets.append(row['dataset'])
        style_sources.append(row['label_source'])
        style_confidences.append(float(row['label_confidence']))
        style_mask.append(mask)
        style_row_weights.append(float(row['style_row_weight']))

    save_dict = {
        'data': torch.stack(embeddings, dim=0),
        'supported_styles': list(UNIFIED_STYLES),
        'speaker_ids': speaker_ids,
        'clip_paths': clip_paths,
        'source_dataset': source_datasets,
        'style_label_source': style_sources,
        'style_label_confidence': torch.tensor(style_confidences, dtype=torch.float32).unsqueeze(1),
        'style_label_mask': torch.tensor(style_mask, dtype=torch.float32).unsqueeze(1),
        'style_label_row_weight': torch.tensor(style_row_weights, dtype=torch.float32).unsqueeze(1),
    }
    for key, values in targets.items():
        save_dict[key] = torch.stack(values).unsqueeze(1)

    dataset_counts = Counter(source_datasets)
    dataset_labeled_counts = Counter(
        row['dataset'] for row in rows if row['style'] is not None
    )
    dataset_unique_speakers = {
        dataset: len({row['speaker_id'] for row in rows if row['dataset'] == dataset})
        for dataset in DATASETS
        if dataset_counts.get(dataset)
    }
    raw_pseudo_counts = Counter(
        style
        for style in payloads['CommonVoice'].get('pseudo_style', [])
        if style in UNIFIED_STYLES
    )
    threshold_accepted_counts = Counter()
    threshold_rejected_counts = Counter()
    for style, confidence in zip(
        payloads['CommonVoice'].get('pseudo_style', []),
        payloads['CommonVoice'].get('pseudo_style_confidence', []),
    ):
        if style not in UNIFIED_STYLES or confidence is None:
            continue
        confidence = float(confidence)
        if confidence >= threshold_map.get(style, 0.0):
            threshold_accepted_counts[style] += 1
        else:
            threshold_rejected_counts[style] += 1
    selected_pseudo_counts = Counter(
        row['style'] for row in rows if row['dataset'] == 'CommonVoice' and row['style'] is not None
    )
    save_dict['mixture_report'] = {
        'seed': args.seed,
        'parquet_dir': str(parquet_dir),
        'commonvoice_source_artifact': args.commonvoice,
        'cremad_source_artifact': args.cremad,
        'expresso_source_artifact': args.expresso,
        'pseudo_style_threshold': args.pseudo_style_threshold,
        'pseudo_style_thresholds': dict(threshold_map),
        'commonvoice_style_caps': parse_style_caps(args.commonvoice_style_caps),
        'commonvoice_selection': {
            'max_speakers': args.commonvoice_max_speakers,
            'min_clips_per_speaker': args.commonvoice_min_clips_per_speaker,
            'max_clips_per_speaker': args.commonvoice_max_clips_per_speaker,
            'prefer_pseudo': args.commonvoice_prefer_pseudo,
        },
        'row_weight_config': {
            'pseudo_confidence_scale': bool(args.pseudo_confidence_scale),
            'pseudo_row_weight': float(args.pseudo_row_weight),
            'true_row_weight': float(args.true_row_weight),
        },
        'dataset_counts': dict(dataset_counts),
        'dataset_labeled_counts': dict(dataset_labeled_counts),
        'dataset_unique_speakers': dataset_unique_speakers,
        'commonvoice_raw_pseudo_style_counts': dict(raw_pseudo_counts),
        'commonvoice_threshold_accepted_style_counts': dict(threshold_accepted_counts),
        'commonvoice_threshold_rejected_style_counts': dict(threshold_rejected_counts),
        'commonvoice_selected_pseudo_style_counts': dict(selected_pseudo_counts),
        'commonvoice_skipped_by_cap_counts': dict(
            commonvoice_selection_report.get('skipped_by_cap_counts', {})
        ),
        'style_counts_all_labeled_rows': dict(Counter(row['style'] for row in rows if row['style'] is not None)),
        'label_source_counts': dict(Counter(style_sources)),
    }
    if 'metadata_report' in payloads['CommonVoice']:
        save_dict['commonvoice_metadata_report'] = payloads['CommonVoice']['metadata_report']
    if 'pseudo_style_report' in payloads['CommonVoice']:
        save_dict['commonvoice_pseudo_style_report'] = payloads['CommonVoice']['pseudo_style_report']
    return save_dict


def print_report(save_dict):
    report = save_dict['mixture_report']
    print('Mixed-data training artifact report')
    print(f"  total rows: {save_dict['data'].shape[0]}")
    print('  dataset counts:')
    for dataset, count in report['dataset_counts'].items():
        labeled = report['dataset_labeled_counts'].get(dataset, 0)
        speakers = report['dataset_unique_speakers'].get(dataset, 0)
        print(f"    {dataset:11s} rows={count:4d} labeled={labeled:4d} speakers={speakers:4d}")
    print('  label sources:')
    for name, count in report['label_source_counts'].items():
        print(f"    {name:11s} {count:4d}")
    print('  labeled style counts:')
    for style in UNIFIED_STYLES:
        count = report['style_counts_all_labeled_rows'].get(style, 0)
        if count:
            print(f"    {style:11s} {count:4d}")
    pseudo_counts = report.get('commonvoice_selected_pseudo_style_counts', {})
    if pseudo_counts:
        print('  selected CommonVoice pseudo-style counts:')
        for style in UNIFIED_STYLES:
            count = pseudo_counts.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")
    rejected_counts = report.get('commonvoice_threshold_rejected_style_counts', {})
    if rejected_counts:
        print('  threshold-rejected CommonVoice pseudo-style counts:')
        for style in UNIFIED_STYLES:
            count = rejected_counts.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")
    skipped_by_cap = report.get('commonvoice_skipped_by_cap_counts', {})
    if skipped_by_cap:
        print('  cap-skipped CommonVoice pseudo-style counts:')
        for style in UNIFIED_STYLES:
            count = skipped_by_cap.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows, payloads, parquet_dir, threshold_map, commonvoice_selection_report = build_rows(args)
    save_dict = build_save_dict(
        rows,
        payloads,
        args,
        parquet_dir,
        threshold_map,
        commonvoice_selection_report,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    print_report(save_dict)
    print(f"Saved mixed-data artifact to {output_path}")


if __name__ == '__main__':
    main()
