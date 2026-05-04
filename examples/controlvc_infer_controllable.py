"""
Controllable voice anonymization using ControlVC with a style-trained VAE.

This script demonstrates controlling speaker style properties (e.g., happy,
whisper, sad) during DP voice anonymization. The first 7 latent dimensions
of the VAE correspond to Expresso styles:

  0: default     1: confused    2: enunciated    3: happy
  4: laughing    5: sad         6: whisper       7: emphasis
  8: essentials  9: longform   10: singing

Pass --style to push a specific dimension to +1 (active).
Values outside [-1, +1] can produce more extreme effects.

Usage:
    python examples/controlvc_infer_controllable.py \
        --repo-root /path/to/control-vc \
        --source source.wav \
        --out output.wav \
        --vae-checkpoint controlvc_vae_expresso.pt \
        --style happy \
        --noise-level 0.5
"""

import argparse
from pathlib import Path
import dpvc


STYLES = ['default', 'confused', 'enunciated', 'happy', 'laughing', 'sad', 'whisper',
          'emphasis', 'essentials', 'longform', 'singing']


def main():
    ap = argparse.ArgumentParser(description="Controllable ControlVC voice anonymization")
    ap.add_argument("--repo-root", required=True, help="Path to control-vc repository root")
    ap.add_argument("--source", required=True, help="Source audio file path")
    ap.add_argument("--out", required=True, help="Output audio file path")
    ap.add_argument("--vae-checkpoint", required=True, help="Path to trained controllable VAE checkpoint")
    ap.add_argument("--style", default=None, choices=STYLES,
                    help="Target style to apply (e.g., happy, whisper, sad)")
    ap.add_argument("--style-value", type=float, default=1.0,
                    help="Feature value for the target style (default: 1.0, range: -1 to beyond)")
    ap.add_argument("--noise-level", type=float, default=0.5,
                    help="DP noise level (default: 0.5, 0 = no noise)")
    ap.add_argument("--latent-dims", type=int, default=16,
                    help="Latent dimensions of the VAE (default: 16 = 11 style + 5 free)")
    ap.add_argument("--device", default=None, help="Device (default: auto-detect)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Initialize wrapper
    wrapper_kwargs = {"repo_root": Path(args.repo_root)}
    if args.device:
        wrapper_kwargs["device"] = args.device
    wrapper = dpvc.ControlVCWrapper(**wrapper_kwargs)

    # Configure VAE
    vae_config = wrapper.get_vae_config()
    vae_config['checkpoint_path'] = args.vae_checkpoint
    vae_config['latent_dim'] = args.latent_dims

    anonymizer = dpvc.Anonymizer(wrapper, vae_config=vae_config)

    # Build control features
    control_features = None
    if args.style:
        idx = STYLES.index(args.style)
        control_features = {idx: args.style_value}
        print(f"Controlling style: {args.style} (dim {idx} = {args.style_value})")

    # Run anonymization
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anonymizer.anonymize(
        args.source, args.out,
        noise_level=args.noise_level,
        seed=args.seed,
        control_features=control_features,
    )
    print(f"Saved anonymized audio to: {args.out}")


if __name__ == "__main__":
    main()
