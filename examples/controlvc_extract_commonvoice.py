"""
Extract speaker embeddings from Common Voice corpus using ControlVC.

Usage:
    python examples/controlvc_extract_commonvoice.py \
        --repo-root /path/to/control-vc \
        --corpus-path /data/cv-corpus-21.0-2025-03-14/en \
        --output embeddings/controlvc_emb.pt
"""

import argparse
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dpvc import ControlVCWrapper


def main():
    ap = argparse.ArgumentParser(description="Extract ControlVC embeddings from Common Voice")
    ap.add_argument("--repo-root", required=True, help="Path to control-vc repository root")
    ap.add_argument("--corpus-path", required=True, help="Path to Common Voice corpus (e.g. /data/cv-corpus-.../en)")
    ap.add_argument("--device", default=None, help="Device (default: auto-detect)")
    ap.add_argument("--output", default="all_emb_labeled_cv_full.pt", help="Output file path")
    ap.add_argument("--max-clips-per-speaker", type=int, default=10, help="Max clips per speaker")
    args = ap.parse_args()

    base_path = args.corpus_path
    df = pd.read_csv(f'{base_path}/validated.tsv', sep='\t')
    print('number of clips:', len(df))
    clients = list(df['client_id'].unique())
    print('number of clients:', len(clients))

    wrapper_kwargs = {"repo_root": Path(args.repo_root)}
    if args.device:
        wrapper_kwargs["device"] = args.device
    wrapper = ControlVCWrapper(**wrapper_kwargs)

    all_emb = []
    all_labels = []
    for i, ident in tqdm(enumerate(clients), total=len(clients)):
        for source_path in df[df['client_id'] == ident]['path'][:args.max_clips_per_speaker]:
            try:
                source = f'{base_path}/clips/{source_path}'
                target_se = wrapper.extract_embedding(source)
                all_emb.append(target_se)
                all_labels.append(torch.tensor(i))
            except Exception as e:
                print(e)
        if i % 1000 == 0 and all_emb:
            all_emb_t = torch.vstack(all_emb)
            all_labels_t = torch.vstack(all_labels)
            print('saving checkpoint:', all_emb_t.shape)
            torch.save({'data': all_emb_t, 'labels': all_labels_t}, f'{args.output}.checkpoint_{i}.pt')

    all_emb_t = torch.vstack(all_emb)
    all_labels_t = torch.vstack(all_labels)
    print(all_emb_t.shape)
    print(all_labels_t.shape)
    torch.save({'data': all_emb_t, 'labels': all_labels_t}, args.output)


if __name__ == "__main__":
    main()
