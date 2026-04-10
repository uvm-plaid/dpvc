"""
Combine CREMA-D and Expresso embeddings into a unified training set.

Unified labels (9): anger, confused, disgust, enunciated, fear, happy, neutral, sad, whisper
- CREMA-D (91 speakers): anger, disgust, fear, happy, neutral, sad
- Expresso (3 speakers, capped): confused, enunciated, whisper
- Shared labels use CREMA-D + 1-per-speaker from Expresso

Usage:
    python examples/openvoice_combine_datasets.py \
        --cremad embeddings/openvoice_cremad_emb.pt \
        --expresso embeddings/openvoice_expresso_emb.pt \
        --output embeddings/openvoice_combined_emb.pt \
        --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/.../read
"""

import argparse
import torch
import numpy as np
import pandas as pd
import os
from collections import Counter, defaultdict

UNIFIED = ['anger', 'confused', 'disgust', 'enunciated', 'fear', 'happy', 'neutral', 'sad', 'whisper']

EXPRESSO_STYLES = ['default', 'confused', 'enunciated', 'happy', 'laughing', 'sad', 'whisper',
                   'emphasis', 'essentials', 'longform', 'singing']
EXPRESSO_MAP = {'default': 'neutral', 'confused': 'confused', 'enunciated': 'enunciated',
                'happy': 'happy', 'sad': 'sad', 'whisper': 'whisper'}
EXPRESSO_ONLY = {'confused', 'enunciated', 'whisper'}

CREMAD_EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']


def encode_onehot(label):
    vec = [-1.0] * len(UNIFIED)
    if label in UNIFIED:
        vec[UNIFIED.index(label)] = 1.0
    return vec


def main():
    ap = argparse.ArgumentParser(description="Combine CREMA-D + Expresso embeddings")
    ap.add_argument("--cremad", default="embeddings/openvoice_cremad_emb.pt")
    ap.add_argument("--expresso", default="embeddings/openvoice_expresso_emb.pt")
    ap.add_argument("--output", default="embeddings/openvoice_combined_emb.pt")
    ap.add_argument("--parquet-dir", required=True,
                    help="Expresso parquet dir (for speaker IDs)")
    ap.add_argument("--cap", type=int, default=90,
                    help="Cap Expresso-only styles to this many samples (default: 90)")
    args = ap.parse_args()

    # Load CREMA-D
    cremad = torch.load(args.cremad, weights_only=True)
    cremad_embs = cremad['data']
    cremad_labels = []
    for i in range(len(cremad_embs)):
        for e in CREMAD_EMOTIONS:
            if cremad[f'emotion_{e}'][i].item() > 0:
                cremad_labels.append(e)
                break
    print(f"CREMA-D: {len(cremad_embs)} samples")

    # Load Expresso
    expresso = torch.load(args.expresso, weights_only=True)
    expresso_embs = expresso['data']

    files = sorted([os.path.join(args.parquet_dir, f)
                    for f in os.listdir(args.parquet_dir) if f.endswith('.parquet')])
    df = pd.concat([pd.read_parquet(f, columns=['speaker_id', 'style']) for f in files],
                   ignore_index=True)

    # Collect Expresso candidates
    expresso_by_label = defaultdict(list)
    seen_shared = set()
    for i in range(len(expresso_embs)):
        orig_style = None
        for s in EXPRESSO_STYLES:
            if expresso[f'style_{s}'][i].item() > 0:
                orig_style = s
                break
        if orig_style not in EXPRESSO_MAP:
            continue
        unified = EXPRESSO_MAP[orig_style]
        if unified in EXPRESSO_ONLY:
            expresso_by_label[unified].append(i)
        else:
            speaker = df.iloc[i]['speaker_id']
            key = (speaker, unified)
            if key not in seen_shared:
                seen_shared.add(key)
                expresso_by_label[unified].append(i)

    # Cap and select
    np.random.seed(42)
    expresso_indices = []
    expresso_labels_out = []
    for label, indices in expresso_by_label.items():
        if label in EXPRESSO_ONLY and len(indices) > args.cap:
            indices = list(np.random.choice(indices, args.cap, replace=False))
        for idx in indices:
            expresso_indices.append(idx)
            expresso_labels_out.append(label)

    expresso_embs_sel = expresso_embs[expresso_indices]
    print(f"Expresso: {len(expresso_indices)} samples (filtered)")

    # Combine
    all_embs = torch.cat([cremad_embs, expresso_embs_sel], dim=0)
    all_labels = cremad_labels + expresso_labels_out

    label_dict = {f'label_{l}': [] for l in UNIFIED}
    for label in all_labels:
        vec = encode_onehot(label)
        for j, l in enumerate(UNIFIED):
            label_dict[f'label_{l}'].append(torch.tensor(vec[j]))

    save_dict = {'data': all_embs}
    for key, vals in label_dict.items():
        save_dict[key] = torch.vstack(vals)

    print(f"\nCombined: {all_embs.shape[0]} samples, {len(UNIFIED)} labels")
    dist = Counter(all_labels)
    for l in UNIFIED:
        print(f"  {l:12s} {dist.get(l, 0):4d}")

    torch.save(save_dict, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
