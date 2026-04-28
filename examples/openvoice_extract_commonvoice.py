"""
Extract speaker embeddings from a local Common Voice corpus using OpenVoice.

This script expects a local corpus layout with:
  <corpus-path>/validated.tsv
  <corpus-path>/clips/<audio files>

Usage:
    python examples/openvoice_extract_commonvoice.py \
        --corpus-path /path/to/cv-corpus-21.0/en \
        --output embeddings/openvoice_commonvoice_emb.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dpvc import OpenVoiceWrapper
from dpvc import utils


def clean_optional(value):
    return None if pd.isna(value) else value


def resolve_accent_column(df):
    for column in ("accent", "accents"):
        if column in df.columns:
            return column
    return None


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract OpenVoice embeddings from a local Common Voice corpus")
    ap.add_argument("--corpus-path", required=True,
                    help="Path to Common Voice language directory with validated.tsv and clips/")
    ap.add_argument("--output", default="embeddings/openvoice_commonvoice_emb.pt",
                    help="Output file path")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed (default: 42)")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="Optional cap on number of speakers to process")
    ap.add_argument("--max-clips-per-speaker", type=int, default=10,
                    help="Maximum validated clips per speaker (default: 10)")
    ap.add_argument("--checkpoint-every", type=int, default=1000,
                    help="Save a checkpoint every N extracted embeddings (default: 1000)")
    return ap.parse_args()


def save_checkpoint(path, save_dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, path)


def build_save_dict(embeddings, speaker_ids, clip_paths, ages, genders, accents,
                    corpus_path, seed, max_clips_per_speaker, missing_files,
                    unreadable_files):
    return {
        "data": torch.vstack(embeddings),
        "speaker_ids": speaker_ids,
        "clip_paths": clip_paths,
        "age": ages,
        "gender": genders,
        "accent": accents,
        "corpus_path": str(corpus_path),
        "seed": seed,
        "max_clips_per_speaker": max_clips_per_speaker,
        "missing_files": missing_files,
        "unreadable_files": unreadable_files,
    }


def build_clip_index(clips_dir):
    return {path.name: path for path in clips_dir.rglob("*") if path.is_file()}


def resolve_clip_path(clips_dir, clip_rel, clip_index):
    clip_path = clips_dir / clip_rel
    if clip_path.exists():
        return clip_path
    return clip_index.get(Path(clip_rel).name)


def main():
    args = parse_args()
    utils.set_seed(args.seed)

    corpus_path = Path(args.corpus_path).expanduser().resolve()
    validated_path = corpus_path / "validated.tsv"
    clips_dir = corpus_path / "clips"

    if not validated_path.exists():
        raise FileNotFoundError(f"Missing validated.tsv: {validated_path}")
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing clips directory: {clips_dir}")

    df = pd.read_csv(validated_path, sep="\t", low_memory=False)
    if "client_id" not in df.columns or "path" not in df.columns:
        raise ValueError("validated.tsv must contain at least 'client_id' and 'path' columns")

    df = df[df["client_id"].notna() & df["path"].notna()].copy()
    df["client_id"] = df["client_id"].astype(str)
    speaker_ids = sorted(df["client_id"].unique())
    if args.max_speakers is not None and args.max_speakers < len(speaker_ids):
        rng = np.random.default_rng(args.seed)
        speaker_ids = sorted(
            rng.choice(speaker_ids, size=args.max_speakers, replace=False).tolist()
        )
    grouped = df.groupby("client_id", sort=False)

    accent_column = resolve_accent_column(df)
    print(f"Loaded {len(df)} validated rows from {validated_path}")
    print(f"Processing {len(speaker_ids)} speakers with up to {args.max_clips_per_speaker} clips each")

    wrapper = OpenVoiceWrapper()

    embeddings = []
    out_speaker_ids = []
    clip_paths = []
    ages = []
    genders = []
    accents = []

    missing_files = 0
    unreadable_files = 0
    clip_index = build_clip_index(clips_dir)

    for speaker_id in tqdm(speaker_ids, desc="Speakers"):
        speaker_rows = grouped.get_group(speaker_id)
        if len(speaker_rows) > args.max_clips_per_speaker:
            speaker_rows = speaker_rows.sample(
                n=args.max_clips_per_speaker,
                random_state=args.seed,
            )
        speaker_rows = speaker_rows.sort_values("path")
        for _, row in speaker_rows.iterrows():
            clip_rel = str(row["path"])
            clip_path = resolve_clip_path(clips_dir, clip_rel, clip_index)
            if clip_path is None:
                missing_files += 1
                continue

            try:
                embedding = wrapper.extract_embedding(str(clip_path))
            except Exception as exc:
                unreadable_files += 1
                print(f"Error processing {clip_path}: {exc}")
                continue

            embeddings.append(embedding)
            out_speaker_ids.append(speaker_id)
            clip_paths.append(str(Path("clips") / clip_rel))
            ages.append(clean_optional(row["age"]) if "age" in row else None)
            genders.append(clean_optional(row["gender"]) if "gender" in row else None)
            accents.append(clean_optional(row[accent_column]) if accent_column else None)

            if args.checkpoint_every and len(embeddings) % args.checkpoint_every == 0:
                checkpoint_path = f"{args.output}.checkpoint.pt"
                save_checkpoint(
                    checkpoint_path,
                    build_save_dict(
                        embeddings=embeddings,
                        speaker_ids=out_speaker_ids,
                        clip_paths=clip_paths,
                        ages=ages,
                        genders=genders,
                        accents=accents,
                        corpus_path=corpus_path,
                        seed=args.seed,
                        max_clips_per_speaker=args.max_clips_per_speaker,
                        missing_files=missing_files,
                        unreadable_files=unreadable_files,
                    ),
                )
                print(f"Checkpoint saved to {checkpoint_path} ({len(embeddings)} embeddings)")

    if not embeddings:
        raise RuntimeError("No embeddings were extracted. Check corpus path and audio files.")

    save_dict = build_save_dict(
        embeddings=embeddings,
        speaker_ids=out_speaker_ids,
        clip_paths=clip_paths,
        ages=ages,
        genders=genders,
        accents=accents,
        corpus_path=corpus_path,
        seed=args.seed,
        max_clips_per_speaker=args.max_clips_per_speaker,
        missing_files=missing_files,
        unreadable_files=unreadable_files,
    )
    save_checkpoint(args.output, save_dict)

    print(f"Saved {len(embeddings)} embeddings to {args.output}")
    print(f"Unique speakers extracted: {len(set(out_speaker_ids))}")
    print(f"Missing clip files skipped: {missing_files}")
    print(f"Unreadable clip files skipped: {unreadable_files}")


if __name__ == "__main__":
    main()
