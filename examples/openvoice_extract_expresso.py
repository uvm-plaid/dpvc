"""
Extract speaker embeddings and style labels from the Expresso dataset using OpenVoice.

OpenVoice embeds F0/prosody in the speaker embedding, so style differences
should be captured in the extracted embeddings (unlike ControlVC's D_VECTOR).

Usage:
    # From locally cached parquet files (recommended):
    python examples/openvoice_extract_expresso.py \
        --output embeddings/openvoice_expresso_emb.pt \
        --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/.../read

    # Or download from HuggingFace:
    python examples/openvoice_extract_expresso.py \
        --output embeddings/openvoice_expresso_emb.pt

Requires: pip install pandas soundfile openvoice
"""

import argparse
import os
import tempfile
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import io
from dpvc import OpenVoiceWrapper

STYLES = ['default', 'confused', 'enunciated', 'happy', 'laughing', 'sad', 'whisper',
          'emphasis', 'essentials', 'longform', 'singing']


def encode_style_onehot(style_name):
    """Encode a style as a one-hot vector with +1 for active, -1 for inactive."""
    vec = [-1.0] * len(STYLES)
    if style_name in STYLES:
        vec[STYLES.index(style_name)] = 1.0
    return vec


def load_expresso(parquet_dir=None, max_samples=None):
    """Load Expresso dataset from parquet files or HuggingFace."""
    if parquet_dir:
        print(f"Loading Expresso from local parquet files in {parquet_dir}...")
        files = sorted([os.path.join(parquet_dir, f)
                        for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        print("Loading Expresso dataset from HuggingFace...")
        from datasets import load_dataset, Audio
        dataset = load_dataset("ylacombe/expresso", "read", split="train")
        dataset = dataset.cast_column("audio", Audio(decode=False))
        df = dataset.to_pandas()

    if max_samples:
        df = df.head(max_samples)

    print(f"Total samples: {len(df)}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Extract OpenVoice embeddings from Expresso dataset")
    ap.add_argument("--output", default="embeddings/openvoice_expresso_emb.pt",
                    help="Output file path (default: embeddings/openvoice_expresso_emb.pt)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit number of samples to process (default: all)")
    ap.add_argument("--parquet-dir", default=None,
                    help="Load from local parquet files instead of downloading")
    args = ap.parse_args()

    df = load_expresso(args.parquet_dir, args.max_samples)

    # Initialize OpenVoice wrapper
    wrapper = OpenVoiceWrapper()

    all_emb = []
    all_ids = []
    all_styles = {f'style_{s}': [] for s in STYLES}
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        style = row.get('style', '')

        if style not in STYLES:
            skipped += 1
            continue

        try:
            audio_data = row['audio']
            audio_bytes = audio_data['bytes']
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')

            # Write to temp WAV file for the wrapper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_array, sample_rate)
                embedding = wrapper.extract_embedding(tmp.name)
                os.unlink(tmp.name)

            all_emb.append(embedding)
            all_ids.append(torch.tensor(i))

            style_vec = encode_style_onehot(style)
            for j, s in enumerate(STYLES):
                all_styles[f'style_{s}'].append(torch.tensor(style_vec[j]))

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            skipped += 1

        # Periodic checkpoint
        if (len(all_emb)) % 1000 == 0 and all_emb:
            print(f"  {len(all_emb)} embeddings extracted, {skipped} skipped")
            _save_checkpoint(all_emb, all_ids, all_styles, f"{args.output}.checkpoint.pt")

    if not all_emb:
        print("No embeddings extracted. Check dataset and wrapper configuration.")
        return

    print(f"\nDone: {len(all_emb)} embeddings extracted, {skipped} samples skipped")
    _save_checkpoint(all_emb, all_ids, all_styles, args.output)
    print(f"Saved to {args.output}")


def _save_checkpoint(all_emb, all_ids, all_styles, path):
    """Save embeddings and labels to a .pt file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'data': torch.vstack(all_emb),
        'ids': torch.vstack(all_ids),
    }
    for key, vals in all_styles.items():
        save_dict[key] = torch.vstack(vals)
    print(f"  Saving: embeddings shape {save_dict['data'].shape}")
    torch.save(save_dict, path)


if __name__ == "__main__":
    main()
