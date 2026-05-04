"""
Word Error Rate (WER) evaluation for generated voice outputs.

Runs OpenAI Whisper on each generated .wav file and computes WER against a
reference transcription. Two reference modes:

  (default) baseline mode: the same-speaker baseline (no style control) is
      transcribed, and each style output for that speaker is compared to its
      baseline transcription. This measures how much style control degrades
      intelligibility *relative to no control*.

  --reference-text "..." : compare every file's Whisper transcription against a
      fixed ground-truth string. Measures absolute WER.

Filename convention expected: <speaker_id>_<style>.wav
(matches output from openvoice_infer_controllable.py --all-styles)

Usage:
    # Drift-from-baseline WER (default)
    python examples/eval_wer.py \
        --input output/diverse_speakers/ \
        --out output/eval_wer.csv

    # Absolute WER against known source text
    python examples/eval_wer.py \
        --input output/trump_styles/ \
        --reference-text "Our great movement. We will make America great again." \
        --out output/eval_wer_trump.csv
"""

import argparse
import csv
import string
from collections import defaultdict
from pathlib import Path

import jiwer
import whisper


STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
          'happy', 'neutral', 'sad', 'whisper']


# jiwer normalisation: lowercase, strip punctuation, collapse whitespace
TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def parse_filename(path):
    """Return (speaker_id, style) parsed from stem, or (None, None)."""
    stem = path.stem
    for style in STYLES + ['baseline']:
        suffix = f"_{style}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)], style
    return None, None


def clean(text):
    """Light normalisation for display; WER computation handles its own."""
    return ' '.join(text.strip().split())


def wer_safe(ref, hyp):
    """Compute WER, returning None if reference is empty (jiwer raises)."""
    r = clean(ref)
    h = clean(hyp)
    if not r:
        return None
    try:
        return jiwer.wer(
            r, h,
            reference_transform=TRANSFORM,
            hypothesis_transform=TRANSFORM,
        )
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Directory of generated .wav files")
    ap.add_argument("--out", default="output/eval_wer.csv",
                    help="CSV output path (default: output/eval_wer.csv)")
    ap.add_argument("--prefix", default=None,
                    help="Only process files whose name starts with this prefix")
    ap.add_argument("--model", default="base",
                    help="Whisper model: tiny / base / small / medium / large-v3 "
                         "(default: base)")
    ap.add_argument("--reference-text", default=None,
                    help="Ground-truth transcript string. If set, every file is "
                         "compared to this text (absolute WER). If unset, each "
                         "file is compared to its same-speaker baseline transcription.")
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
    use_baseline = args.reference_text is None
    if use_baseline:
        speakers_with_baseline = [s for s in speakers if 'baseline' in by_speaker[s]]
        missing = set(speakers) - set(speakers_with_baseline)
        print(f"Parsed {len(per_file)} files across {len(speakers)} speakers "
              f"({len(speakers_with_baseline)} have a baseline).")
        if missing:
            print(f"  Warning: no baseline for {sorted(missing)} — those files "
                  "will have empty WER.")
    else:
        print(f"Parsed {len(per_file)} files. Using fixed reference text "
              f"({len(args.reference_text.split())} words).")

    # Load Whisper
    print(f"\nLoading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)
    print("Model loaded.\n")

    # Transcribe each file once
    transcripts = {}
    for i, (path, speaker, style) in enumerate(per_file, 1):
        result = model.transcribe(str(path), fp16=False, verbose=False)
        txt = clean(result.get('text', ''))
        transcripts[path] = txt
        preview = txt[:60] + ('...' if len(txt) > 60 else '')
        print(f"  [{i:3d}/{len(per_file)}] {path.name:50s} -> {preview}")

    # Compute per-file WER
    rows = []
    wers_by_style = defaultdict(list)
    for path, speaker, style in per_file:
        hyp = transcripts[path]

        if args.reference_text is not None:
            ref = args.reference_text
            ref_source = 'reference-text'
        else:
            baseline_path = by_speaker[speaker].get('baseline')
            if baseline_path is None:
                rows.append({
                    'speaker': speaker, 'style': style,
                    'wer': '', 'ref_source': 'no-baseline',
                    'reference': '', 'hypothesis': hyp,
                    'file': path.name,
                })
                continue
            if path == baseline_path:
                # baseline vs itself = 0.0 by definition; report for completeness
                rows.append({
                    'speaker': speaker, 'style': style,
                    'wer': '0.0000', 'ref_source': 'self',
                    'reference': transcripts[baseline_path],
                    'hypothesis': hyp,
                    'file': path.name,
                })
                continue
            ref = transcripts[baseline_path]
            ref_source = 'baseline'

        w = wer_safe(ref, hyp)
        if w is None:
            rows.append({
                'speaker': speaker, 'style': style,
                'wer': '', 'ref_source': ref_source,
                'reference': ref, 'hypothesis': hyp,
                'file': path.name,
            })
            continue

        rows.append({
            'speaker': speaker, 'style': style,
            'wer': f"{w:.4f}", 'ref_source': ref_source,
            'reference': ref, 'hypothesis': hyp,
            'file': path.name,
        })
        if style != 'baseline':
            wers_by_style[style].append(w)

    # CSV
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
    print(f"Reference mode        : "
          f"{'fixed reference text' if not use_baseline else 'same-speaker baseline'}")
    print(f"Whisper model         : {args.model}")
    print(f"Files transcribed     : {len(per_file)}")

    scored = [r for r in rows if r['wer'] not in ('', '0.0000')]
    if scored:
        all_wers = [float(r['wer']) for r in scored]
        import statistics
        print(f"WER values scored     : {len(all_wers)}")
        print(f"Mean WER              : {statistics.mean(all_wers):.3f}")
        print(f"Median WER            : {statistics.median(all_wers):.3f}")
        print(f"Min / Max             : {min(all_wers):.3f} / {max(all_wers):.3f}")

    if wers_by_style:
        print("\nPer-style WER (lower = more intelligible relative to reference):")
        for style in STYLES:
            vals = wers_by_style.get(style, [])
            if vals:
                import statistics
                print(f"  {style:12s}: mean={statistics.mean(vals):.3f}  "
                      f"median={statistics.median(vals):.3f}  "
                      f"min={min(vals):.3f}  max={max(vals):.3f}  n={len(vals)}")

    print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
