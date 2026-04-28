"""
Build a filtered Common Voice validated.tsv for a local clip subset.

This is useful when you download one or more Common Voice audio shards locally and
want a corpus directory that still matches the repo's expected layout:
  <corpus-path>/validated.tsv
  <corpus-path>/clips/

Usage:
    python scripts/prepare_commonvoice_subset.py \
        --validated-tsv /path/to/full_validated.tsv \
        --clips-dir /path/to/clips \
        --output-tsv /path/to/subset/validated.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(
        description="Filter Common Voice validated.tsv to locally available clips"
    )
    ap.add_argument("--validated-tsv", required=True,
                    help="Path to the full Common Voice validated.tsv")
    ap.add_argument("--clips-dir", required=True,
                    help="Directory containing locally available clip files")
    ap.add_argument("--output-tsv", required=True,
                    help="Output path for the filtered validated.tsv")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed for optional speaker sampling")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="Optional cap on number of speakers to keep")
    ap.add_argument("--max-clips-per-speaker", type=int, default=10,
                    help="Optional cap on validated clips per speaker (default: 10)")
    return ap.parse_args()


def main():
    args = parse_args()
    validated_tsv = Path(args.validated_tsv).expanduser().resolve()
    clips_dir = Path(args.clips_dir).expanduser().resolve()
    output_tsv = Path(args.output_tsv).expanduser().resolve()

    if not validated_tsv.exists():
        raise FileNotFoundError(f"validated.tsv not found: {validated_tsv}")
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"clips directory not found: {clips_dir}")

    print(f"Loading metadata from {validated_tsv}")
    df = pd.read_csv(validated_tsv, sep="\t", low_memory=False)
    if "client_id" not in df.columns or "path" not in df.columns:
        raise ValueError("validated.tsv must contain 'client_id' and 'path' columns")

    available = {path.name for path in clips_dir.rglob("*") if path.is_file()}
    print(f"Found {len(available)} local clip files in {clips_dir}")

    df = df[df["client_id"].notna() & df["path"].notna()].copy()
    df["client_id"] = df["client_id"].astype(str)
    df = df[df["path"].astype(str).isin(available)].copy()
    if df.empty:
        raise RuntimeError("No validated rows matched the local clip subset")

    speaker_ids = sorted(df["client_id"].unique())
    if args.max_speakers is not None and args.max_speakers < len(speaker_ids):
        rng = np.random.default_rng(args.seed)
        speaker_ids = sorted(
            rng.choice(speaker_ids, size=args.max_speakers, replace=False).tolist()
        )
        df = df[df["client_id"].isin(speaker_ids)].copy()

    if args.max_clips_per_speaker is not None:
        parts = []
        for speaker_id, speaker_rows in df.groupby("client_id", sort=False):
            if len(speaker_rows) > args.max_clips_per_speaker:
                speaker_rows = speaker_rows.sample(
                    n=args.max_clips_per_speaker,
                    random_state=args.seed,
                )
            parts.append(speaker_rows.sort_values("path"))
        df = pd.concat(parts, ignore_index=True)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False)

    print(f"Wrote filtered validated.tsv to {output_tsv}")
    print(f"Rows kept: {len(df)}")
    print(f"Unique speakers kept: {df['client_id'].nunique()}")


if __name__ == "__main__":
    main()
