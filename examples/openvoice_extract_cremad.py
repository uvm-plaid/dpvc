"""
Extract speaker embeddings and emotion labels from the CREMA-D dataset using OpenVoice.

CREMA-D has 91 speakers × 6 emotions × ~13 sentences = 7,442 clips.
By default, extracts one sample per speaker per emotion (546 samples)
to maximize speaker diversity per Joe's recommendation.

Usage:
    python examples/openvoice_extract_cremad.py \
        --output embeddings/openvoice_cremad_emb.pt

    # Extract all samples (not just one per speaker per emotion):
    python examples/openvoice_extract_cremad.py \
        --output embeddings/openvoice_cremad_emb_all.pt --all-samples

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
from datasets import load_dataset, Audio
from collections import defaultdict
import io
from dpvc import OpenVoiceWrapper

EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']


def encode_emotion_onehot(emotion):
    """Encode an emotion as a one-hot vector with +1 for active, -1 for inactive."""
    vec = [-1.0] * len(EMOTIONS)
    if emotion in EMOTIONS:
        vec[EMOTIONS.index(emotion)] = 1.0
    return vec


def main():
    ap = argparse.ArgumentParser(description="Extract OpenVoice embeddings from CREMA-D dataset")
    ap.add_argument("--output", default="embeddings/openvoice_cremad_emb.pt",
                    help="Output file path")
    ap.add_argument("--all-samples", action="store_true",
                    help="Extract all samples (default: one per speaker per emotion)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit total samples to process")
    args = ap.parse_args()

    print("Loading CREMA-D dataset from HuggingFace...")
    dataset = load_dataset('AbstractTTS/CREMA-D', split='train')
    dataset = dataset.cast_column('audio', Audio(decode=False))
    print(f"Total samples in dataset: {len(dataset)}")

    # Filter to one sample per speaker per emotion for diversity
    if not args.all_samples:
        print("Filtering to one sample per speaker per emotion...")
        seen = set()
        indices = []
        for i, row in enumerate(dataset):
            speaker = row['file'].split('_')[0]
            emotion = row['major_emotion']
            key = (speaker, emotion)
            if key not in seen and emotion in EMOTIONS:
                seen.add(key)
                indices.append(i)
        dataset = dataset.select(indices)
        print(f"Filtered to {len(dataset)} samples ({len(seen)} speaker-emotion combos)")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Initialize OpenVoice wrapper
    wrapper = OpenVoiceWrapper()

    all_emb = []
    all_ids = []
    all_speakers = []
    all_emotions = {f'emotion_{e}': [] for e in EMOTIONS}
    skipped = 0

    for i, row in tqdm(enumerate(dataset), total=len(dataset)):
        emotion = row['major_emotion']
        speaker = row['file'].split('_')[0]

        if emotion not in EMOTIONS:
            skipped += 1
            continue

        try:
            audio_bytes = row['audio']['bytes']
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_array, sample_rate)
                embedding = wrapper.extract_embedding(tmp.name)
                os.unlink(tmp.name)

            all_emb.append(embedding)
            all_ids.append(torch.tensor(i))
            all_speakers.append(int(speaker))

            emo_vec = encode_emotion_onehot(emotion)
            for j, e in enumerate(EMOTIONS):
                all_emotions[f'emotion_{e}'].append(torch.tensor(emo_vec[j]))

        except Exception as e:
            print(f"Error processing sample {i} ({row['file']}): {e}")
            skipped += 1

        if len(all_emb) % 100 == 0 and all_emb:
            print(f"  {len(all_emb)} embeddings extracted, {skipped} skipped")

    if not all_emb:
        print("No embeddings extracted.")
        return

    print(f"\nDone: {len(all_emb)} embeddings extracted, {skipped} skipped")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'data': torch.vstack(all_emb),
        'ids': torch.vstack(all_ids),
        'speakers': torch.tensor(all_speakers),
    }
    for key, vals in all_emotions.items():
        save_dict[key] = torch.vstack(vals)

    print(f"Saving: embeddings shape {save_dict['data'].shape}, {len(set(all_speakers))} unique speakers")
    torch.save(save_dict, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
