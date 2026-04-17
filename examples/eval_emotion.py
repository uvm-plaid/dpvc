"""
Emotion evaluation for generated voice outputs.

Runs emotion2vec_plus_large (ACL 2024) on a directory of generated .wav files
and computes two metrics per the EmoVoice paper (arxiv 2504.12867):

  - Recall Rate: does the predicted emotion match the intended style label?
  - emo_sim   : cosine similarity of emotion2vec embeddings between each
                generated file and the same-speaker baseline (no style control).

Filename convention expected:   <speaker_id>_<style>.wav
(matches output from openvoice_infer_controllable.py --all-styles)

Only 6 of the 9 styles have emotion2vec counterparts and so contribute to
Recall Rate: anger, disgust, fear, happy, neutral, sad. The remaining
styles (confused, enunciated, whisper) are still evaluated via emo_sim.

Usage:
    # Evaluate all files in a directory
    python examples/eval_emotion.py \
        --input output/diverse_speakers/ \
        --out output/eval_emotion.csv

    # Limit to one speaker
    python examples/eval_emotion.py \
        --input output/diverse_speakers/ \
        --prefix cremad_f1_1002 \
        --out /tmp/single_speaker.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from funasr import AutoModel


# Our 9 style labels (order matches the VAE's controllable latent dims)
STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
          'happy', 'neutral', 'sad', 'whisper']

# Map our style -> emotion2vec category label (None = no comparable category)
STYLE_TO_E2V = {
    'anger':      'angry',
    'confused':   None,
    'disgust':    'disgusted',
    'enunciated': None,
    'fear':       'fearful',
    'happy':      'happy',
    'neutral':    'neutral',
    'sad':        'sad',
    'whisper':    None,
}


def parse_filename(path):
    """Return (speaker_id, style) parsed from stem, or (None, None)."""
    stem = path.stem
    for style in STYLES + ['baseline']:
        suffix = f"_{style}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], style
    return None, None


def cosine(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    na = a / (np.linalg.norm(a) + 1e-8)
    nb = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(na, nb))


def canonical_label(raw):
    """Strip any Chinese-language prefix from emotion2vec labels (format: 'zh/en').

    emotion2vec returns labels like 'angry', '<unk>', or sometimes bilingual
    e.g. '生气/angry'. We always take the English side, lowercased.
    """
    if '/' in raw:
        raw = raw.split('/')[-1]
    return raw.strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Directory of generated .wav files")
    ap.add_argument("--out", default="output/eval_emotion.csv",
                    help="CSV output path (default: output/eval_emotion.csv)")
    ap.add_argument("--prefix", default=None,
                    help="Only process files whose name starts with this prefix")
    ap.add_argument("--model", default="iic/emotion2vec_plus_large",
                    help="funasr model id (default: iic/emotion2vec_plus_large)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    files = sorted(in_dir.glob("*.wav"))
    if args.prefix:
        files = [f for f in files if f.name.startswith(args.prefix)]
    if not files:
        raise SystemExit(f"No .wav files found in {in_dir}"
                         + (f" with prefix {args.prefix!r}" if args.prefix else ""))

    # Parse filenames. Skip anything that doesn't match our naming convention.
    per_file = []
    by_speaker = defaultdict(dict)
    for path in files:
        speaker, style = parse_filename(path)
        if speaker is None:
            print(f"  skipping (no style suffix): {path.name}")
            continue
        per_file.append((path, speaker, style))
        by_speaker[speaker][style] = path

    speakers = sorted(by_speaker.keys())
    speakers_with_baseline = [s for s in speakers if 'baseline' in by_speaker[s]]
    print(f"Parsed {len(per_file)} files across {len(speakers)} speakers "
          f"({len(speakers_with_baseline)} with a baseline for emo_sim)")

    # Load emotion2vec
    print(f"\nLoading {args.model}...")
    model = AutoModel(model=args.model, hub='hf', disable_update=True)
    print("Model loaded.\n")

    # Run inference once per file, keeping both the top label and the embedding
    preds = {}
    for i, (path, speaker, style) in enumerate(per_file, 1):
        rec = model.generate(str(path), granularity="utterance",
                             extract_embedding=True)
        r = rec[0]
        labels = [canonical_label(x) for x in r['labels']]
        scores = r['scores']
        top_i = int(np.argmax(scores))
        preds[path] = {
            'label': labels[top_i],
            'score': float(scores[top_i]),
            'embedding': np.asarray(r['feats']).flatten(),
        }
        print(f"  [{i:3d}/{len(per_file)}] {path.name:50s} -> "
              f"{preds[path]['label']:12s} ({preds[path]['score']:.2f})")

    # Compute per-file metrics
    rows = []
    correct, evaluable = 0, 0
    for path, speaker, style in per_file:
        p = preds[path]
        target = STYLE_TO_E2V.get(style)

        if style == 'baseline' or target is None:
            match = ''  # not part of recall denominator
        else:
            evaluable += 1
            is_correct = (p['label'] == target)
            correct += int(is_correct)
            match = '1' if is_correct else '0'

        emo_sim = ''
        baseline_path = by_speaker[speaker].get('baseline')
        if baseline_path is not None and path != baseline_path:
            emo_sim = f"{cosine(p['embedding'], preds[baseline_path]['embedding']):.4f}"

        rows.append({
            'speaker': speaker,
            'style': style,
            'predicted': p['label'],
            'target': target or '',
            'match': match,
            'score': f"{p['score']:.4f}",
            'emo_sim': emo_sim,
            'file': path.name,
        })

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files evaluated       : {len(per_file)}")
    print(f"Speakers              : {len(speakers)}")
    if evaluable:
        print(f"Overall recall rate   : {correct}/{evaluable} "
              f"= {100 * correct / evaluable:.1f}%")

    # Per-style recall
    per_style = defaultdict(lambda: [0, 0])
    for row in rows:
        if row['match'] in ('0', '1'):
            per_style[row['style']][1] += 1
            per_style[row['style']][0] += int(row['match'])
    print("\nPer-style recall (only styles with emotion2vec counterparts):")
    for style in STYLES:
        c, t = per_style.get(style, (0, 0))
        if t > 0:
            print(f"  {style:12s}: {c}/{t} = {100 * c / t:3.0f}%")
        elif STYLE_TO_E2V.get(style) is None:
            print(f"  {style:12s}: n/a (no emotion2vec counterpart)")

    # Mean emo_sim per style
    per_style_sim = defaultdict(list)
    for row in rows:
        if row['emo_sim']:
            per_style_sim[row['style']].append(float(row['emo_sim']))
    print("\nMean emo_sim vs baseline (higher = closer to baseline emotion):")
    for style in STYLES:
        sims = per_style_sim.get(style, [])
        if sims:
            print(f"  {style:12s}: mean={np.mean(sims):.3f}  "
                  f"min={min(sims):.3f}  max={max(sims):.3f}  n={len(sims)}")

    print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
