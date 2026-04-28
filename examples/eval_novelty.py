"""
Speaker novelty evaluation for generated voice outputs.

Computes cosine similarity and cosine distance between the source speaker
embedding and each generated output embedding, using OpenVoice's native speaker
embedding extractor. In manifest mode it also compares each style output
against the same-speaker baseline conversion, so we can tell whether a style
pushes the voice farther from the source than plain voice conversion already
does.

Usage:
    # Evaluate a manifest-driven corpus
    python examples/eval_novelty.py \
        --manifest output/diverse_speakers/generation_manifest.jsonl \
        --out results/eval_novelty.csv

    # Evaluate a single source/generated pair
    python examples/eval_novelty.py \
        --source examples/trump_0.wav \
        --generated output/whisper.wav \
        --style whisper \
        --out /tmp/single_novelty.csv
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import dpvc


STYLES = ['baseline', 'anger', 'confused', 'disgust', 'enunciated',
          'fear', 'happy', 'neutral', 'sad', 'whisper']


def cosine(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    na = a / (np.linalg.norm(a) + 1e-8)
    nb = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(na, nb))


def cosine_distance(a, b):
    return float(1.0 - cosine(a, b))


def clean_float(value):
    if value is None:
        return ''
    return f"{value:.4f}"


def normalize_record(record):
    normalized = dict(record)
    normalized['source_file'] = str(Path(record['source_file']).expanduser().resolve())
    normalized['generated_file'] = str(Path(record['generated_file']).expanduser().resolve())
    if 'output_file' in normalized:
        normalized.pop('output_file')
    return normalized


def load_manifest(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append({
                'source_file': raw['source_file'],
                'generated_file': raw.get('output_file', raw.get('generated_file')),
                'source_stem': raw.get('source_stem') or Path(raw['source_file']).stem,
                'style': raw['style'],
                'style_index': raw.get('style_index'),
                'style_strength': raw.get('style_strength'),
                'noise_level': raw.get('noise_level'),
                'seed': raw.get('seed'),
                'vae_checkpoint': raw.get('vae_checkpoint'),
                'latent_dims': raw.get('latent_dims'),
            })
    return [normalize_record(row) for row in rows]


def build_single_record(args):
    return [{
        'source_file': str(Path(args.source).expanduser().resolve()),
        'generated_file': str(Path(args.generated).expanduser().resolve()),
        'source_stem': Path(args.source).stem,
        'style': args.style,
        'style_index': '',
        'style_strength': args.style_strength,
        'noise_level': args.noise_level,
        'seed': args.seed,
        'vae_checkpoint': '',
        'latent_dims': '',
    }]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--manifest',
                              help='generation_manifest.jsonl from openvoice_infer_controllable.py')
    source_group.add_argument('--source',
                              help='Single source file for one-off evaluation')
    ap.add_argument('--generated',
                    help='Single generated file for one-off evaluation')
    ap.add_argument('--style', default='single',
                    help='Style label for one-off evaluation (default: single)')
    ap.add_argument('--style-strength', type=float, default=None,
                    help='Optional style strength metadata for one-off evaluation')
    ap.add_argument('--noise-level', type=float, default=None,
                    help='Optional noise metadata for one-off evaluation')
    ap.add_argument('--seed', type=int, default=None,
                    help='Optional seed metadata for one-off evaluation')
    ap.add_argument('--out', default='results/eval_novelty.csv',
                    help='CSV output path (default: results/eval_novelty.csv)')
    ap.add_argument('--prefix', default=None,
                    help='Only process records whose source_stem starts with this prefix')
    args = ap.parse_args()

    if args.source and not args.generated:
        ap.error('--generated is required when using --source')
    if args.generated and not args.source:
        ap.error('--source is required when using --generated')

    return args


def resolve_records(args):
    if args.manifest:
        records = load_manifest(args.manifest)
    else:
        records = build_single_record(args)

    if args.prefix:
        records = [row for row in records if row['source_stem'].startswith(args.prefix)]
    if not records:
        raise SystemExit('No records matched the provided inputs.')
    return records


def build_baseline_map(records):
    baseline_map = {}
    for row in records:
        if row['style'] == 'baseline':
            baseline_map[row['source_stem']] = row['generated_file']
    return baseline_map


def collect_unique_paths(records, baseline_map):
    unique_paths = set()
    for row in records:
        unique_paths.add(row['source_file'])
        unique_paths.add(row['generated_file'])
        baseline_path = baseline_map.get(row['source_stem'])
        if baseline_path:
            unique_paths.add(baseline_path)
    return sorted(unique_paths)


def main():
    args = parse_args()
    records = resolve_records(args)
    baseline_map = build_baseline_map(records)
    unique_sources = sorted({row['source_stem'] for row in records})
    print(f"Resolved {len(records)} records across {len(unique_sources)} sources")
    if args.manifest:
        print(f"Manifest mode         : {Path(args.manifest).resolve()}")
        print(f"Baselines available   : {len(baseline_map)}/{len(unique_sources)}")
    else:
        print("Manifest mode         : no (single-pair evaluation)")

    wrapper = dpvc.OpenVoiceWrapper()
    unique_paths = collect_unique_paths(records, baseline_map)
    print(f"Unique audio files    : {len(unique_paths)}")
    print("Extracting OpenVoice speaker embeddings...")

    embedding_cache = {}
    for path_str in tqdm(unique_paths, desc='Embeddings'):
        path = Path(path_str)
        embedding_cache[path_str] = np.asarray(
            wrapper.extract_embedding(str(path)).detach().cpu()
        ).flatten()

    rows = []
    sims_by_style = defaultdict(list)
    gains_by_style = defaultdict(list)
    baseline_sims = []
    nonbaseline_sims = []

    for row in records:
        source_emb = embedding_cache[row['source_file']]
        generated_emb = embedding_cache[row['generated_file']]
        similarity = cosine(source_emb, generated_emb)
        distance = 1.0 - similarity

        baseline_file = baseline_map.get(row['source_stem'])
        baseline_similarity = None
        baseline_distance = None
        similarity_delta = None
        distance_delta = None
        novelty_gain = None

        if baseline_file:
            baseline_emb = embedding_cache[baseline_file]
            baseline_similarity = cosine(source_emb, baseline_emb)
            baseline_distance = 1.0 - baseline_similarity
            similarity_delta = similarity - baseline_similarity
            distance_delta = distance - baseline_distance
            novelty_gain = baseline_similarity - similarity

        out_row = {
            'speaker': row['source_stem'],
            'style': row['style'],
            'source_file': row['source_file'],
            'generated_file': row['generated_file'],
            'baseline_file': baseline_file or '',
            'similarity': f"{similarity:.4f}",
            'distance': f"{distance:.4f}",
            'baseline_similarity': clean_float(baseline_similarity),
            'baseline_distance': clean_float(baseline_distance),
            'similarity_delta_vs_baseline': clean_float(similarity_delta),
            'distance_delta_vs_baseline': clean_float(distance_delta),
            'novelty_gain_vs_baseline': clean_float(novelty_gain),
            'noise_level': row.get('noise_level', ''),
            'style_strength': row.get('style_strength', ''),
            'seed': row.get('seed', ''),
            'vae_checkpoint': row.get('vae_checkpoint', ''),
            'latent_dims': row.get('latent_dims', ''),
        }
        rows.append(out_row)

        sims_by_style[row['style']].append(similarity)
        if row['style'] == 'baseline':
            baseline_sims.append(similarity)
        else:
            nonbaseline_sims.append(similarity)
            if novelty_gain is not None:
                gains_by_style[row['style']].append(novelty_gain)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Rows written          : {len(rows)}")
    print(f"Unique sources        : {len(unique_sources)}")

    def print_stats(label, values):
        if not values:
            return
        print(f"{label:22s}: mean={statistics.mean(values):.4f}  "
              f"median={statistics.median(values):.4f}  "
              f"min={min(values):.4f}  max={max(values):.4f}  n={len(values)}")

    print_stats('Baseline similarity', baseline_sims)
    print_stats('Styled similarity', nonbaseline_sims)
    if baseline_sims and nonbaseline_sims:
        print(f"{'Mean novelty gain':22s}: "
              f"{statistics.mean(baseline_sims) - statistics.mean(nonbaseline_sims):.4f} "
              "(positive = styles are farther from source than baseline conversion)")

    print("\nPer-style similarity to source (lower = more novel):")
    for style in STYLES:
        vals = sims_by_style.get(style, [])
        if vals:
            print_stats(f"  {style}", vals)

    if gains_by_style:
        print("\nPer-style novelty gain vs baseline (positive = more novel):")
        for style in STYLES:
            vals = gains_by_style.get(style, [])
            if vals:
                print_stats(f"  {style}", vals)

    print(f"\nCSV written to: {out_path}")


if __name__ == '__main__':
    main()
