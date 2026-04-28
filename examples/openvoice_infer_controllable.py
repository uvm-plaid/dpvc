"""
Controllable voice anonymization using OpenVoice with a style-trained VAE.

Applies a target emotion/style to a source voice while anonymizing speaker
identity with differential privacy noise. The first 9 latent dimensions of
the combined VAE correspond to:

  0: anger      1: confused    2: disgust     3: enunciated
  4: fear       5: happy       6: neutral     7: sad
  8: whisper

Usage:
    # Apply "happy" style with no DP noise:
    python examples/openvoice_infer_controllable.py \
        --source examples/trump_0.wav \
        --out output/happy.wav \
        --vae-checkpoint embeddings/openvoice_vae_combined.pt \
        --style happy

    # Generate all 9 styles for a single file:
    python examples/openvoice_infer_controllable.py \
        --source examples/trump_0.wav \
        --out output/trump_styles/ \
        --vae-checkpoint embeddings/openvoice_vae_combined.pt \
        --all-styles

    # Generate all 9 styles for a directory of sources:
    python examples/openvoice_infer_controllable.py \
        --source-dir examples/source_speakers/ \
        --out output/diverse_speakers/ \
        --vae-checkpoint embeddings/openvoice_vae_combined.pt \
        --all-styles
"""

import argparse
import json
from pathlib import Path

import dpvc


STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
          'happy', 'neutral', 'sad', 'whisper']
AUDIO_SUFFIXES = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}


def collect_sources(source=None, source_dir=None):
    if source is not None:
        return [Path(source)]

    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {root}")

    sources = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not sources:
        raise FileNotFoundError(
            f"No audio files with supported extensions found in {root}"
        )
    return sources


def build_anonymizer(vae_checkpoint, latent_dims):
    wrapper = dpvc.OpenVoiceWrapper()
    vae_config = wrapper.get_vae_config()
    vae_config['checkpoint_path'] = vae_checkpoint
    vae_config['latent_dim'] = latent_dims
    return dpvc.Anonymizer(wrapper, vae_config=vae_config)


def resolve_manifest_path(args, batch_mode):
    if args.manifest:
        return Path(args.manifest)

    out_path = Path(args.out)
    if batch_mode:
        return out_path / "generation_manifest.jsonl"

    return out_path.with_name(f"{out_path.stem}_manifest.jsonl")


def run_one(anonymizer, source, out_path, style_idx, strength, noise_level, seed):
    control_features = None
    if style_idx is not None:
        control_features = {style_idx: strength}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anonymizer.anonymize(
        str(source),
        str(out_path),
        noise_level=noise_level,
        seed=seed,
        control_features=control_features,
    )


def build_record(source, output_file, style, style_idx, strength, noise_level,
                 seed, vae_checkpoint, latent_dims):
    return {
        "source_file": str(Path(source).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "source_stem": Path(source).stem,
        "style": style,
        "style_index": style_idx,
        "style_strength": strength,
        "noise_level": noise_level,
        "seed": seed,
        "vae_checkpoint": str(Path(vae_checkpoint).resolve()),
        "latent_dims": latent_dims,
    }


def write_manifest(manifest_path, records):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Controllable OpenVoice voice anonymization")
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", help="Single source audio file path")
    source_group.add_argument(
        "--source-dir",
        help="Directory of source audio files for batch generation",
    )
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output file path for single-style single-source runs; output "
            "directory for --all-styles or --source-dir"
        ),
    )
    ap.add_argument("--vae-checkpoint", required=True,
                    help="Path to trained controllable VAE checkpoint")
    ap.add_argument("--style", default=None, choices=STYLES,
                    help="Target style (e.g., happy, whisper, anger)")
    ap.add_argument("--all-styles", action="store_true",
                    help="Generate all 9 styles plus a baseline")
    ap.add_argument("--style-strength", type=float, default=5.0,
                    help="Control strength (default: 5.0, higher = stronger style effect)")
    ap.add_argument("--noise-level", type=float, default=0.0,
                    help="DP noise level (default: 0.0, try 0.1 for light privacy)")
    ap.add_argument("--latent-dims", type=int, default=15,
                    help="VAE latent dimensions (default: 15, must match training)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42, use -1 for random)")
    ap.add_argument(
        "--manifest",
        help=(
            "Optional manifest path. Defaults to <out>/generation_manifest.jsonl "
            "for batch runs or <out_stem>_manifest.jsonl for single-file runs."
        ),
    )
    args = ap.parse_args()

    if args.style and args.all_styles:
        ap.error("Choose either --style <name> or --all-styles, not both")
    if not args.style and not args.all_styles:
        ap.error("Specify --style <name> or --all-styles")

    batch_mode = bool(args.source_dir) or args.all_styles
    if batch_mode:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()
    seed = args.seed if args.seed != -1 else None
    sources = collect_sources(args.source, args.source_dir)
    batch_mode = bool(args.source_dir) or args.all_styles
    manifest_path = resolve_manifest_path(args, batch_mode=batch_mode)

    anonymizer = build_anonymizer(args.vae_checkpoint, args.latent_dims)
    records = []

    if args.all_styles:
        out_dir = Path(args.out)
        for source in sources:
            src_stem = source.stem

            base_path = out_dir / f"{src_stem}_baseline.wav"
            print(f"Generating baseline for {source.name} -> {base_path}")
            run_one(anonymizer, source, base_path, None, 0.0, args.noise_level, seed)
            records.append(build_record(
                source=source,
                output_file=base_path,
                style="baseline",
                style_idx=None,
                strength=0.0,
                noise_level=args.noise_level,
                seed=seed,
                vae_checkpoint=args.vae_checkpoint,
                latent_dims=args.latent_dims,
            ))

            for idx, style in enumerate(STYLES):
                out_path = out_dir / f"{src_stem}_{style}.wav"
                print(f"Generating {style} for {source.name} -> {out_path}")
                run_one(
                    anonymizer,
                    source,
                    out_path,
                    idx,
                    args.style_strength,
                    args.noise_level,
                    seed,
                )
                records.append(build_record(
                    source=source,
                    output_file=out_path,
                    style=style,
                    style_idx=idx,
                    strength=args.style_strength,
                    noise_level=args.noise_level,
                    seed=seed,
                    vae_checkpoint=args.vae_checkpoint,
                    latent_dims=args.latent_dims,
                ))
    else:
        idx = STYLES.index(args.style)
        out_root = Path(args.out)
        if args.source_dir:
            for source in sources:
                out_path = out_root / f"{source.stem}_{args.style}.wav"
                print(f"Generating {args.style} for {source.name} -> {out_path}")
                run_one(
                    anonymizer,
                    source,
                    out_path,
                    idx,
                    args.style_strength,
                    args.noise_level,
                    seed,
                )
                records.append(build_record(
                    source=source,
                    output_file=out_path,
                    style=args.style,
                    style_idx=idx,
                    strength=args.style_strength,
                    noise_level=args.noise_level,
                    seed=seed,
                    vae_checkpoint=args.vae_checkpoint,
                    latent_dims=args.latent_dims,
                ))
        else:
            source = sources[0]
            print(f"Generating {args.style} for {source.name} -> {out_root}")
            print(f"Noise: {args.noise_level}")
            run_one(
                anonymizer,
                source,
                out_root,
                idx,
                args.style_strength,
                args.noise_level,
                seed,
            )
            records.append(build_record(
                source=source,
                output_file=out_root,
                style=args.style,
                style_idx=idx,
                strength=args.style_strength,
                noise_level=args.noise_level,
                seed=seed,
                vae_checkpoint=args.vae_checkpoint,
                latent_dims=args.latent_dims,
            ))

    write_manifest(manifest_path, records)
    print(f"\nWrote manifest with {len(records)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
