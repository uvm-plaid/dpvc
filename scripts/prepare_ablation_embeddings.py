"""
Prepare fair single-dataset ablation embeddings in the unified OpenVoice label space.

These ablation datasets intentionally mirror the current combined-pipeline
preprocessing assumptions:

- `cremad_only`: uses the existing one-sample-per-speaker-per-emotion CREMA-D
  extraction and maps its 6 emotions directly into unified `label_*` columns.
- `expresso_only`: keeps only the Expresso styles that map into the unified
  label space and applies the same balancing logic as
  `examples/openvoice_combine_datasets.py` so the comparison isolates speaker
  diversity rather than dataset-specific sample-count inflation.

Outputs are `.pt` files compatible with `examples/openvoice_train_vae_combined.py`.
Only the supported `label_*` keys are saved, so the trainer supervises exactly
the styles each ablation condition actually contains.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch


UNIFIED_STYLES = [
    "anger",
    "confused",
    "disgust",
    "enunciated",
    "fear",
    "happy",
    "neutral",
    "sad",
    "whisper",
]

CREMAD_STYLES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EXPRESSO_STYLES = [
    "default",
    "confused",
    "enunciated",
    "happy",
    "laughing",
    "sad",
    "whisper",
    "emphasis",
    "essentials",
    "longform",
    "singing",
]
EXPRESSO_MAP = {
    "default": "neutral",
    "confused": "confused",
    "enunciated": "enunciated",
    "happy": "happy",
    "sad": "sad",
    "whisper": "whisper",
}
EXPRESSO_ONLY = {"confused", "enunciated", "whisper"}
EXPRESSO_SUPPORTED = ["confused", "enunciated", "happy", "neutral", "sad", "whisper"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--condition",
        required=True,
        choices=["cremad_only", "expresso_only"],
        help="Which ablation dataset to prepare",
    )
    ap.add_argument(
        "--cremad",
        default="embeddings/openvoice_cremad_emb.pt",
        help="Path to CREMA-D embeddings checkpoint",
    )
    ap.add_argument(
        "--expresso",
        default="embeddings/openvoice_expresso_emb.pt",
        help="Path to Expresso embeddings checkpoint",
    )
    ap.add_argument(
        "--parquet-dir",
        default=None,
        help="Expresso parquet directory. Auto-detected from the HF cache if omitted.",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=90,
        help="Cap Expresso-only styles to this many samples (default: 90)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for capped sampling (default: 42)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output .pt path. Defaults to embeddings/openvoice_<condition>_ablation_emb.pt",
    )
    return ap.parse_args()


def resolve_output_path(args):
    if args.output:
        return Path(args.output)
    return Path(f"embeddings/openvoice_{args.condition}_ablation_emb.pt")


def resolve_expresso_parquet_dir(parquet_dir):
    if parquet_dir:
        path = Path(parquet_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Expresso parquet dir not found: {path}")
        return path

    matches = sorted(
        glob(
            str(
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "datasets--ylacombe--expresso"
                / "snapshots"
                / "*"
                / "read"
            )
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-detect an Expresso parquet cache. Pass --parquet-dir explicitly."
        )
    return Path(matches[-1])


def label_tensor(labels, supported_styles):
    save_dict = {"supported_styles": supported_styles}
    for style in supported_styles:
        values = [1.0 if label == style else -1.0 for label in labels]
        save_dict[f"label_{style}"] = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    return save_dict


def build_cremad_only(input_path):
    data = torch.load(input_path, weights_only=True)
    embeddings = data["data"]
    labels = []
    for row_idx in range(len(embeddings)):
        for style in CREMAD_STYLES:
            key = f"emotion_{style}"
            if data[key][row_idx].item() > 0:
                labels.append(style)
                break
    if len(labels) != len(embeddings):
        raise ValueError("Failed to resolve a CREMA-D label for every embedding row.")

    save_dict = {
        "data": embeddings,
        "source_dataset": "CREMA-D",
    }
    if "ids" in data:
        save_dict["ids"] = data["ids"]
    if "speakers" in data:
        save_dict["speakers"] = data["speakers"]
    save_dict.update(label_tensor(labels, CREMAD_STYLES))
    return save_dict, Counter(labels)


def build_expresso_only(input_path, parquet_dir, cap, seed):
    data = torch.load(input_path, weights_only=True)
    embeddings = data["data"]

    files = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    df = pd.concat(
        [pd.read_parquet(path, columns=["speaker_id", "style"]) for path in files],
        ignore_index=True,
    )
    if len(df) != len(embeddings):
        if "ids" not in data:
            raise ValueError(
                "Expresso parquet metadata length does not match the extracted embeddings, "
                "and no ids field is available to realign them."
            )
        row_indices = [int(x) for x in data["ids"].view(-1).tolist()]
        df = df.iloc[row_indices].reset_index(drop=True)
        if len(df) != len(embeddings):
            raise ValueError(
                "Failed to realign Expresso parquet metadata with the extracted embeddings."
            )

    expresso_by_label = defaultdict(list)
    seen_shared = set()
    for row_idx in range(len(embeddings)):
        source_style = None
        for style in EXPRESSO_STYLES:
            if data[f"style_{style}"][row_idx].item() > 0:
                source_style = style
                break
        if source_style not in EXPRESSO_MAP:
            continue
        unified = EXPRESSO_MAP[source_style]
        if unified in EXPRESSO_ONLY:
            expresso_by_label[unified].append(row_idx)
            continue
        speaker_id = df.iloc[row_idx]["speaker_id"]
        key = (speaker_id, unified)
        if key not in seen_shared:
            seen_shared.add(key)
            expresso_by_label[unified].append(row_idx)

    rng = np.random.default_rng(seed)
    selected_indices = []
    selected_labels = []
    selected_speakers = []
    for style in EXPRESSO_SUPPORTED:
        indices = list(expresso_by_label.get(style, []))
        if style in EXPRESSO_ONLY and len(indices) > cap:
            indices = sorted(rng.choice(indices, cap, replace=False).tolist())
        for row_idx in indices:
            selected_indices.append(row_idx)
            selected_labels.append(style)
            selected_speakers.append(df.iloc[row_idx]["speaker_id"])

    if not selected_indices:
        raise ValueError("No Expresso rows were selected for the ablation dataset.")

    selected_tensor = torch.tensor(selected_indices, dtype=torch.long)
    save_dict = {
        "data": embeddings[selected_tensor],
        "source_dataset": "Expresso",
        "ids": data["ids"][selected_tensor] if "ids" in data else selected_tensor.unsqueeze(1),
        "speaker_ids": selected_speakers,
        "selected_indices": selected_indices,
    }
    save_dict.update(label_tensor(selected_labels, EXPRESSO_SUPPORTED))
    return save_dict, Counter(selected_labels)


def main():
    args = parse_args()
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.condition == "cremad_only":
        save_dict, counts = build_cremad_only(args.cremad)
    else:
        parquet_dir = resolve_expresso_parquet_dir(args.parquet_dir)
        print(f"Using Expresso parquet metadata from {parquet_dir}")
        save_dict, counts = build_expresso_only(args.expresso, parquet_dir, args.cap, args.seed)

    torch.save(save_dict, output_path)
    print(f"Saved {args.condition} ablation embeddings to {output_path}")
    print(f"Embeddings shape: {tuple(save_dict['data'].shape)}")
    print("Style distribution:")
    for style in UNIFIED_STYLES:
        if counts.get(style):
            print(f"  {style:12s} {counts[style]:4d}")


if __name__ == "__main__":
    main()
