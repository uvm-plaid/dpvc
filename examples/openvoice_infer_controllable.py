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

    # Apply "whisper" with DP noise and stronger control:
    python examples/openvoice_infer_controllable.py \
        --source examples/trump_0.wav \
        --out output/whisper_dp.wav \
        --vae-checkpoint embeddings/openvoice_vae_combined.pt \
        --style whisper \
        --style-strength 5.0 \
        --noise-level 0.1

    # Generate all 9 styles at once:
    python examples/openvoice_infer_controllable.py \
        --source examples/trump_0.wav \
        --out output/ \
        --vae-checkpoint embeddings/openvoice_vae_combined.pt \
        --all-styles \
        --noise-level 0.0
"""

import argparse
from pathlib import Path
import dpvc


STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
          'happy', 'neutral', 'sad', 'whisper']


def run_one(anonymizer, source, out_path, style_idx, strength, noise_level, seed):
    control_features = None
    if style_idx is not None:
        control_features = {style_idx: strength}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anonymizer.anonymize(
        source, out_path,
        noise_level=noise_level,
        seed=seed,
        control_features=control_features,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Controllable OpenVoice voice anonymization")
    ap.add_argument("--source", required=True, help="Source audio file path")
    ap.add_argument("--out", required=True,
                    help="Output path. File path for single style, directory for --all-styles")
    ap.add_argument("--vae-checkpoint", required=True,
                    help="Path to trained controllable VAE checkpoint")
    ap.add_argument("--style", default=None, choices=STYLES,
                    help="Target style (e.g., happy, whisper, anger)")
    ap.add_argument("--all-styles", action="store_true",
                    help="Generate all 9 styles plus a baseline (output to directory)")
    ap.add_argument("--style-strength", type=float, default=5.0,
                    help="Control strength (default: 5.0, higher = stronger style effect)")
    ap.add_argument("--noise-level", type=float, default=0.0,
                    help="DP noise level (default: 0.0, try 0.1 for light privacy)")
    ap.add_argument("--latent-dims", type=int, default=15,
                    help="VAE latent dimensions (default: 15, must match training)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42, use None for random)")
    args = ap.parse_args()

    if not args.style and not args.all_styles:
        ap.error("Specify --style <name> or --all-styles")

    # Initialize
    wrapper = dpvc.OpenVoiceWrapper()
    vae_config = wrapper.get_vae_config()
    vae_config['checkpoint_path'] = args.vae_checkpoint
    vae_config['latent_dim'] = args.latent_dims
    anonymizer = dpvc.Anonymizer(wrapper, vae_config=vae_config)

    seed = args.seed if args.seed != -1 else None

    if args.all_styles:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        src_stem = Path(args.source).stem

        # Baseline (no style control)
        base_path = out_dir / f"{src_stem}_baseline.wav"
        print(f"Generating baseline -> {base_path}")
        run_one(anonymizer, args.source, str(base_path),
                None, 0, args.noise_level, seed)

        # Each style
        for style in STYLES:
            idx = STYLES.index(style)
            out_path = out_dir / f"{src_stem}_{style}.wav"
            print(f"Generating {style} -> {out_path}")
            run_one(anonymizer, args.source, str(out_path),
                    idx, args.style_strength, args.noise_level, seed)

        print(f"\nGenerated {len(STYLES) + 1} files in {out_dir}/")

    else:
        idx = STYLES.index(args.style)
        print(f"Style: {args.style} (dim {idx}, strength={args.style_strength})")
        print(f"Noise: {args.noise_level}")
        run_one(anonymizer, args.source, args.out,
                idx, args.style_strength, args.noise_level, seed)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
