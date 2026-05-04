"""
Annotate CommonVoice embedding artifacts with confidence-scored pseudo-style labels.

This uses emotion2vec to assign pseudo labels for the overlapping emotional
styles supported by the controllable OpenVoice pipeline:

  angry -> anger
  disgusted -> disgust
  fearful -> fear
  happy -> happy
  neutral -> neutral
  sad -> sad

Rows with other / <unk> predictions remain unlabeled. The raw mapped label and
confidence are always stored; downstream pretraining can apply its own
confidence threshold without re-running the annotation pass.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from funasr import AutoModel
from tqdm import tqdm

from dpvc import utils


E2V_TO_STYLE = {
    'angry': 'anger',
    'disgusted': 'disgust',
    'fearful': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
}


def canonical_label(raw):
    if '/' in raw:
        raw = raw.split('/')[-1]
    return raw.strip().lower()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--embeddings',
        default='embeddings/openvoice_commonvoice_cv500_emb.pt',
        help='Path to CommonVoice embedding artifact',
    )
    ap.add_argument(
        '--output',
        default='embeddings/openvoice_commonvoice_cv500_pseudo.pt',
        help='Output path for enriched artifact',
    )
    ap.add_argument(
        '--model',
        default='iic/emotion2vec_plus_large',
        help='funasr model id (default: iic/emotion2vec_plus_large)',
    )
    ap.add_argument(
        '--report-threshold',
        type=float,
        default=0.60,
        help='Confidence threshold used only for the printed/saved coverage report (default: 0.60)',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional limit for smoke runs',
    )
    return ap.parse_args()


def resolve_clip_path(corpus_path, clip_rel):
    clip_rel_path = Path(clip_rel)
    if clip_rel_path.is_absolute():
        return clip_rel_path
    return Path(corpus_path) / clip_rel_path


def main():
    args = parse_args()
    data = torch.load(args.embeddings, weights_only=False)

    clip_paths = data['clip_paths']
    corpus_path = data['corpus_path']
    total_rows = len(clip_paths)
    if args.limit is not None:
        total_rows = min(total_rows, args.limit)

    print(f'Loading {args.model} for pseudo-label annotation...')
    model = AutoModel(model=args.model, hub='hf', disable_update=True)
    print('Model loaded.')

    pseudo_style = [None] * len(clip_paths)
    pseudo_style_confidence = [None] * len(clip_paths)
    pseudo_style_raw_label = [None] * len(clip_paths)

    raw_counts = Counter()
    mapped_counts = Counter()
    accepted_counts = Counter()

    for idx, clip_rel in enumerate(tqdm(clip_paths[:total_rows], desc='Pseudo labels')):
        clip_path = resolve_clip_path(corpus_path, clip_rel)
        rec = model.generate(str(clip_path), granularity='utterance', extract_embedding=False)
        row = rec[0]
        labels = [canonical_label(label) for label in row['labels']]
        scores = row['scores']
        top_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        raw_label = labels[top_idx]
        confidence = float(scores[top_idx])
        mapped_style = E2V_TO_STYLE.get(raw_label)

        pseudo_style[idx] = mapped_style
        pseudo_style_confidence[idx] = confidence
        pseudo_style_raw_label[idx] = raw_label

        raw_counts[raw_label] += 1
        if mapped_style:
            mapped_counts[mapped_style] += 1
            if confidence >= args.report_threshold:
                accepted_counts[mapped_style] += 1

    metadata_report = data.get('metadata_report')
    if metadata_report is None:
        metadata_report = utils.build_commonvoice_metadata_report(
            data.get('age', []),
            data.get('gender', []),
            data.get('accent', []),
        )

    pseudo_style_report = {
        'model': args.model,
        'report_threshold': args.report_threshold,
        'rows_annotated': total_rows,
        'raw_label_counts': dict(raw_counts),
        'mapped_style_counts': dict(mapped_counts),
        'accepted_style_counts': dict(accepted_counts),
    }

    enriched = dict(data)
    enriched['pseudo_style'] = pseudo_style
    enriched['pseudo_style_confidence'] = pseudo_style_confidence
    enriched['pseudo_style_raw_label'] = pseudo_style_raw_label
    enriched['pseudo_style_source'] = args.model
    enriched['pseudo_style_report'] = pseudo_style_report
    enriched['metadata_report'] = metadata_report

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, output_path)

    print(f'Saved pseudo-labeled CommonVoice artifact to {output_path}')
    print(f'Rows annotated: {total_rows}/{len(clip_paths)}')
    print('Accepted pseudo-style counts at report threshold:')
    for style, count in sorted(accepted_counts.items()):
        print(f'  {style:8s}: {count}')


if __name__ == '__main__':
    main()
