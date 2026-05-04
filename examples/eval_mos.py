"""
Predicted MOS (naturalness) evaluation for generated voice outputs.

Predicts subjective Mean Opinion Score for each generated .wav file using
torchaudio's SQUIM_SUBJECTIVE model (Meta, 2023) — a learned MOS predictor
trained on BVCC + DAPS subjective ratings. Output is on the standard 1-5
MOS scale (higher = more natural).

Note on UTMOS: the EmoVoice paper (arxiv 2504.12867) uses UTMOS (Saeki et al.
2022) for this role. UTMOS is distributed via sarulab-speech's `utmos` pip
package which requires a from-source fairseq build that conflicts with the
fairseq monkey-patches in this codebase. SQUIM_SUBJECTIVE measures the same
quantity (predicted subjective MOS on 1-5 scale) on comparable training data,
so we substitute it here.

Filename convention: <speaker_id>_<style>.wav

SQUIM_SUBJECTIVE is a "non-matching reference" predictor: it takes the test
audio and any clean-speech waveform as a reference for model conditioning.
By default we use the same-speaker baseline as the reference, so each
file is scored with its most natural-sounding cousin. Pass --reference to
override.

Usage:
    # Score every file in a directory (default: baseline-as-reference)
    python examples/eval_mos.py \
        --input output/diverse_speakers/ \
        --out output/eval_mos.csv

    # Use a fixed reference wav for all scoring
    python examples/eval_mos.py \
        --input output/diverse_speakers/ \
        --reference examples/source_speakers/cremad_1007.wav \
        --out output/eval_mos_fixed.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as TAF
from torchaudio.pipelines import SQUIM_SUBJECTIVE


STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
          'happy', 'neutral', 'sad', 'whisper']


def parse_filename(path):
    stem = path.stem
    for style in STYLES + ['baseline']:
        suffix = f"_{style}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], style
    return None, None


def load_wav(path, target_sr):
    """Load wav to mono float32 tensor at target_sr, shape [1, T]."""
    audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    wav = torch.from_numpy(audio).unsqueeze(0)
    if sr != target_sr:
        wav = TAF.resample(wav, sr, target_sr)
    return wav


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Directory of generated .wav files")
    ap.add_argument("--out", default="output/eval_mos.csv",
                    help="CSV output path (default: output/eval_mos.csv)")
    ap.add_argument("--prefix", default=None,
                    help="Only process files whose name starts with this prefix")
    ap.add_argument("--reference", default=None,
                    help="Fixed non-matching reference .wav for all files. "
                         "If unset, uses each file's same-speaker baseline as reference.")
    args = ap.parse_args()

    in_dir = Path(args.input)
    files = sorted(in_dir.glob("*.wav"))
    if args.prefix:
        files = [f for f in files if f.name.startswith(args.prefix)]
    if not files:
        raise SystemExit(f"No .wav files found in {in_dir}"
                         + (f" with prefix {args.prefix!r}" if args.prefix else ""))

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
    use_fixed_ref = args.reference is not None
    if use_fixed_ref:
        print(f"Parsed {len(per_file)} files. Using fixed reference: {args.reference}")
    else:
        speakers_with_baseline = [s for s in speakers if 'baseline' in by_speaker[s]]
        missing = set(speakers) - set(speakers_with_baseline)
        print(f"Parsed {len(per_file)} files across {len(speakers)} speakers "
              f"({len(speakers_with_baseline)} have a baseline for reference).")
        if missing:
            print(f"  Warning: no baseline for {sorted(missing)} — those files will be skipped.")

    # Load SQUIM
    print("\nLoading SQUIM_SUBJECTIVE model...")
    model = SQUIM_SUBJECTIVE.get_model()
    model.eval()
    target_sr = SQUIM_SUBJECTIVE.sample_rate
    print(f"Model loaded. Target SR: {target_sr} Hz.\n")

    # Cache waveforms keyed by path so we don't re-load baselines
    wav_cache = {}
    def get_wav(path):
        if path not in wav_cache:
            wav_cache[path] = load_wav(path, target_sr)
        return wav_cache[path]

    fixed_ref_wav = load_wav(args.reference, target_sr) if use_fixed_ref else None

    # Score every file
    rows = []
    scores_by_style = defaultdict(list)
    baseline_score = {}  # speaker -> baseline MOS (for delta computation)

    with torch.inference_mode():
        for i, (path, speaker, style) in enumerate(per_file, 1):
            wav = get_wav(path)

            if use_fixed_ref:
                ref_wav = fixed_ref_wav
                ref_source = Path(args.reference).name
            else:
                baseline_path = by_speaker[speaker].get('baseline')
                if baseline_path is None:
                    rows.append({
                        'speaker': speaker, 'style': style,
                        'mos': '', 'delta_vs_baseline': '',
                        'ref_source': 'no-baseline',
                        'file': path.name,
                    })
                    continue
                ref_wav = get_wav(baseline_path)
                ref_source = 'baseline' if path != baseline_path else 'self'

            mos = float(model(wav, ref_wav).detach())
            rows.append({
                'speaker': speaker, 'style': style,
                'mos': f"{mos:.4f}", 'delta_vs_baseline': '',
                'ref_source': ref_source,
                'file': path.name,
            })
            if style == 'baseline':
                baseline_score[speaker] = mos
            scores_by_style[style].append(mos)
            print(f"  [{i:3d}/{len(per_file)}] {path.name:50s} MOS={mos:.3f}")

    # Fill in delta_vs_baseline now that baseline_score is populated
    for row in rows:
        spk = row['speaker']
        if row['mos'] and spk in baseline_score and row['style'] != 'baseline':
            delta = float(row['mos']) - baseline_score[spk]
            row['delta_vs_baseline'] = f"{delta:+.4f}"

    # CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print("\n" + "=" * 60)
    print("Summary (scale: 1=bad, 5=excellent)")
    print("=" * 60)
    print(f"Reference mode        : "
          f"{'fixed reference' if use_fixed_ref else 'same-speaker baseline'}")
    print(f"Files scored          : {sum(1 for r in rows if r['mos'])}")

    all_vals = [float(r['mos']) for r in rows if r['mos']]
    if all_vals:
        print(f"Mean MOS              : {statistics.mean(all_vals):.3f}")
        print(f"Median MOS            : {statistics.median(all_vals):.3f}")
        print(f"Min / Max             : {min(all_vals):.3f} / {max(all_vals):.3f}")

    print("\nPer-style MOS:")
    for style in ['baseline'] + STYLES:
        vals = scores_by_style.get(style, [])
        if vals:
            print(f"  {style:12s}: mean={statistics.mean(vals):.3f}  "
                  f"median={statistics.median(vals):.3f}  "
                  f"min={min(vals):.3f}  max={max(vals):.3f}  n={len(vals)}")

    if not use_fixed_ref and baseline_score:
        print("\nPer-style mean MOS delta vs same-speaker baseline (negative = less natural):")
        per_style_delta = defaultdict(list)
        for row in rows:
            if row['delta_vs_baseline']:
                per_style_delta[row['style']].append(float(row['delta_vs_baseline']))
        for style in STYLES:
            deltas = per_style_delta.get(style, [])
            if deltas:
                print(f"  {style:12s}: mean={statistics.mean(deltas):+.3f}  "
                      f"n={len(deltas)}")

    print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
