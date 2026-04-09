"""
Extract speaker embeddings and style labels from the Expresso dataset using OpenVoice.

OpenVoice embeds F0/prosody in the speaker embedding, so style differences
should be captured in the extracted embeddings (unlike ControlVC's D_VECTOR).

Note: If downloads are slow, set HF_HUB_DISABLE_XET=1 to bypass the xet storage backend.

Usage:
    python examples/openvoice_extract_expresso.py \
        --output embeddings/openvoice_expresso_emb.pt

    # From locally cached parquet files:
    python examples/openvoice_extract_expresso.py \
        --output embeddings/openvoice_expresso_emb.pt \
        --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/.../read

Requires: pip install datasets soundfile openvoice
"""

import argparse
import os
import tempfile
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import datasets
from datasets import load_dataset
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


def main():
    ap = argparse.ArgumentParser(description="Extract OpenVoice embeddings from Expresso dataset")
    ap.add_argument("--output", default="embeddings/openvoice_expresso_emb.pt",
                    help="Output file path (default: embeddings/openvoice_expresso_emb.pt)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit number of samples to process (default: all)")
    ap.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    ap.add_argument("--parquet-dir", default=None,
                    help="Load from local parquet files instead of downloading")
    args = ap.parse_args()

    # Load Expresso dataset
    if args.parquet_dir:
        print(f"Loading Expresso from local parquet files in {args.parquet_dir}...")
        files = sorted([os.path.join(args.parquet_dir, f)
                        for f in os.listdir(args.parquet_dir) if f.endswith('.parquet')])
        slice_spec = f'{args.split}[:{args.max_samples}]' if args.max_samples else args.split
        dataset = load_dataset('parquet', data_files=files, split=slice_spec)
    else:
        print("Loading Expresso dataset from HuggingFace...")
        dataset = load_dataset("ylacombe/expresso", "read", split=args.split)

    if args.max_samples and not args.parquet_dir:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Disable automatic audio decoding (avoids torchcodec dependency)
    dataset = dataset.cast_column("audio", datasets.Audio(decode=False))

    print(f"Total samples: {len(dataset)}")

    # Initialize OpenVoice wrapper
    wrapper = OpenVoiceWrapper()

    all_emb = []
    all_ids = []
    all_styles = {f'style_{s}': [] for s in STYLES}
    skipped = 0

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        style = sample.get('style', sample.get('style_tag', ''))

        if style not in STYLES:
            skipped += 1
            continue

        try:
            audio_data = sample['audio']
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
        if (i + 1) % 1000 == 0 and all_emb:
            print(f"  Processed {i + 1} samples, {len(all_emb)} embeddings extracted, {skipped} skipped")
            _save_checkpoint(all_emb, all_ids, all_styles, f"{args.output}.checkpoint_{i}.pt")

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
