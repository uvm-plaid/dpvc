"""
Generate evaluation corpora for the paper-strengthening branches.

This script keeps the main `examples/openvoice_infer_controllable.py` interface
stable and adds the extra research logic needed for the ablation matrix:

- condition-specific style maps for single-dataset checkpoints
- a deterministic naive baseline using unlabeled latent dimensions
- pass-specific checkpoint aliases for the CommonVoice and mixed-data branches
- manifest rows that record the ablation condition and control mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dpvc


FULL_STYLES = [
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

CONDITION_CONFIGS = {
    "combined": {
        "checkpoint": "embeddings/openvoice_vae_combined.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "commonvoice_cv500_init": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_ft_short": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_ft_short.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_ft_low_lr": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_ft_low_lr.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_ft_short_low_lr": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_ft_short_low_lr.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_obj_label2": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_obj_label2.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_obj_label4": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_obj_label4.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_obj_label_ramp": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_obj_label_ramp.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_obj_recon_half_label2": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_obj_recon_half_label2.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_rich_teacher_style": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_rich_teacher_style.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_rich_free_anchor": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_rich_free_anchor.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_rich_teacher_plus_anchor": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_rich_teacher_plus_anchor.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_pl_meta": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_pl_meta.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_pl_pseudo_style": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_pl_pseudo_style.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_pl_meta_plus_pseudo": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_pl_meta_plus_pseudo.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "mixed_static_balanced": {
        "checkpoint": "embeddings/openvoice_vae_mixed_static_balanced.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "mixed_cv_warmup": {
        "checkpoint": "embeddings/openvoice_vae_mixed_cv_warmup.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "mixed_labeled_finish": {
        "checkpoint": "embeddings/openvoice_vae_mixed_labeled_finish.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "mixed_quality_static_balanced": {
        "checkpoint": "embeddings/openvoice_vae_mixed_quality_static_balanced.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "mixed_quality_labeled_finish": {
        "checkpoint": "embeddings/openvoice_vae_mixed_quality_labeled_finish.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "mixed_quality_labeled_guarded": {
        "checkpoint": "embeddings/openvoice_vae_mixed_quality_labeled_guarded.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_ft_freeze_decoder": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_ft_freeze_decoder.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cv500_ft_freeze_encoder": {
        "checkpoint": "embeddings/openvoice_vae_combined_cv500_ft_freeze_encoder.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "labeled",
    },
    "cremad_only": {
        "checkpoint": "embeddings/openvoice_vae_cremad_ablation.pt",
        "styles": ["anger", "disgust", "fear", "happy", "neutral", "sad"],
        "style_to_index": {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
        },
        "control_mode": "labeled",
    },
    "expresso_only": {
        "checkpoint": "embeddings/openvoice_vae_expresso_ablation.pt",
        "styles": ["confused", "enunciated", "happy", "neutral", "sad", "whisper"],
        "style_to_index": {
            "confused": 0,
            "enunciated": 1,
            "happy": 2,
            "neutral": 3,
            "sad": 4,
            "whisper": 5,
        },
        "control_mode": "labeled",
    },
    "naive_noise_baseline": {
        "checkpoint": "embeddings/openvoice_vae_combined.pt",
        "styles": FULL_STYLES,
        "style_to_index": {style: idx for idx, style in enumerate(FULL_STYLES)},
        "control_mode": "random_free_dims",
        "free_dim_start": 9,
        "free_dim_count": 6,
    },
}

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", help="Single source audio file")
    source_group.add_argument("--source-dir", help="Directory of source audio files")
    ap.add_argument(
        "--condition",
        required=True,
        choices=sorted(CONDITION_CONFIGS.keys()),
        help="Ablation condition to generate",
    )
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults to <out>/generation_manifest.jsonl",
    )
    ap.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Optional checkpoint override for the selected condition",
    )
    ap.add_argument(
        "--style-strength",
        type=float,
        default=5.0,
        help="Control strength or matched free-dim L2 norm (default: 5.0)",
    )
    ap.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="DP noise level passed into the VAE (default: 0.0)",
    )
    ap.add_argument(
        "--latent-dims",
        type=int,
        default=15,
        help="VAE latent dimensionality (default: 15)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for generation and naive control vectors",
    )
    return ap.parse_args()


def collect_sources(source=None, source_dir=None):
    if source is not None:
        return [Path(source)]
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {root}")
    sources = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not sources:
        raise FileNotFoundError(f"No supported audio files found in {root}")
    return sources


def build_anonymizer(vae_checkpoint, latent_dims):
    wrapper = dpvc.OpenVoiceWrapper()
    vae_config = wrapper.get_vae_config()
    vae_config["checkpoint_path"] = vae_checkpoint
    vae_config["latent_dim"] = latent_dims
    return dpvc.Anonymizer(wrapper, vae_config=vae_config)


def resolve_manifest_path(out_dir, manifest):
    if manifest:
        return Path(manifest)
    return Path(out_dir) / "generation_manifest.jsonl"


def build_random_free_dim_controls(style_name, style_index, style_strength, seed, free_dim_start, free_dim_count):
    rng_seed = (seed if seed is not None else 0) + (style_index + 1) * 1009
    rng = np.random.default_rng(rng_seed)
    vec = rng.standard_normal(free_dim_count)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    vec = vec * style_strength
    return {
        free_dim_start + offset: float(value)
        for offset, value in enumerate(vec.tolist())
    }


def run_one(anonymizer, source, out_path, noise_level, seed, control_features):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anonymizer.anonymize(
        str(source),
        str(out_path),
        noise_level=noise_level,
        seed=seed,
        control_features=control_features,
    )


def build_record(source, output_file, condition, style, style_index, style_strength,
                 noise_level, seed, vae_checkpoint, latent_dims, control_mode,
                 control_features):
    return {
        "source_file": str(Path(source).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "source_stem": Path(source).stem,
        "condition": condition,
        "style": style,
        "style_index": style_index,
        "style_strength": style_strength,
        "noise_level": noise_level,
        "seed": seed,
        "vae_checkpoint": str(Path(vae_checkpoint).resolve()),
        "latent_dims": latent_dims,
        "control_mode": control_mode,
        "control_features": control_features,
    }


def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    cfg = dict(CONDITION_CONFIGS[args.condition])
    vae_checkpoint = args.vae_checkpoint or cfg["checkpoint"]
    sources = collect_sources(args.source, args.source_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_manifest_path(out_dir, args.manifest)

    anonymizer = build_anonymizer(vae_checkpoint, args.latent_dims)
    records = []

    for source in sources:
        src_stem = source.stem

        baseline_path = out_dir / f"{src_stem}_baseline.wav"
        print(f"[{args.condition}] baseline -> {baseline_path}")
        run_one(
            anonymizer,
            source,
            baseline_path,
            noise_level=args.noise_level,
            seed=args.seed,
            control_features=None,
        )
        records.append(
            build_record(
                source=source,
                output_file=baseline_path,
                condition=args.condition,
                style="baseline",
                style_index=None,
                style_strength=0.0,
                noise_level=args.noise_level,
                seed=args.seed,
                vae_checkpoint=vae_checkpoint,
                latent_dims=args.latent_dims,
                control_mode="baseline",
                control_features=None,
            )
        )

        for style in cfg["styles"]:
            style_index = cfg["style_to_index"][style]
            if cfg["control_mode"] == "random_free_dims":
                control_features = build_random_free_dim_controls(
                    style_name=style,
                    style_index=style_index,
                    style_strength=args.style_strength,
                    seed=args.seed,
                    free_dim_start=cfg["free_dim_start"],
                    free_dim_count=cfg["free_dim_count"],
                )
            else:
                control_features = {style_index: args.style_strength}

            out_path = out_dir / f"{src_stem}_{style}.wav"
            print(f"[{args.condition}] {style:12s} -> {out_path}")
            run_one(
                anonymizer,
                source,
                out_path,
                noise_level=args.noise_level,
                seed=args.seed,
                control_features=control_features,
            )
            records.append(
                build_record(
                    source=source,
                    output_file=out_path,
                    condition=args.condition,
                    style=style,
                    style_index=style_index,
                    style_strength=args.style_strength,
                    noise_level=args.noise_level,
                    seed=args.seed,
                    vae_checkpoint=vae_checkpoint,
                    latent_dims=args.latent_dims,
                    control_mode=cfg["control_mode"],
                    control_features=control_features,
                )
            )

    write_manifest(manifest_path, records)
    print(f"\nWrote manifest with {len(records)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
