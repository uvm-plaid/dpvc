"""
Controllable voice anonymization using ControlVC with F0-based style control.

Style is controlled via prosody (F0) manipulation, which produces audible
differences in pitch, intonation range, and expressiveness.

Available F0 presets:
  happy   — pitch up 20%, wider intonation range
  sad     — pitch down 15%, narrower range
  whisper — flattened pitch, compressed range
  confused — pitch up 10%, exaggerated range
  enunciated — slight pitch up, wider range
  laughing — pitch up 15%, very wide range

Custom F0 control via --pitch-shift, --range-scale, --flatten.

Usage:
    python examples/controlvc_infer_controllable.py \\
        --repo-root /path/to/control-vc \\
        --source source.wav \\
        --out output.wav \\
        --vae-checkpoint controlvc_vae_expresso.pt \\
        --style happy \\
        --noise-level 0.5
"""

import argparse
from pathlib import Path
import dpvc


STYLES = ['default', 'confused', 'enunciated', 'happy', 'laughing', 'sad', 'whisper',
          'emphasis', 'essentials', 'longform', 'singing']

# F0 transform presets: each style maps to prosody modifications
# Designed to be audibly distinct from each other
F0_PRESETS = {
    'happy':      {'pitch_shift': 1.25, 'range_scale': 1.6},
    'sad':        {'pitch_shift': 0.80, 'range_scale': 0.4, 'flatten': 0.3},
    'whisper':    {'pitch_shift': 0.90, 'flatten': 0.8, 'range_scale': 0.2},
    'confused':   {'pitch_shift': 1.10, 'range_scale': 2.0},
    'enunciated': {'pitch_shift': 1.0,  'range_scale': 1.8},
    'laughing':   {'pitch_shift': 1.35, 'range_scale': 2.2},
    'emphasis':   {'pitch_shift': 0.95, 'range_scale': 1.6},
    'default':    {},
}


def main():
    ap = argparse.ArgumentParser(description="Controllable ControlVC voice anonymization")
    ap.add_argument("--repo-root", required=True, help="Path to control-vc repository root")
    ap.add_argument("--source", required=True, help="Source audio file path")
    ap.add_argument("--out", required=True, help="Output audio file path")
    ap.add_argument("--vae-checkpoint", required=True, help="Path to trained controllable VAE checkpoint")
    ap.add_argument("--style", default=None, choices=STYLES,
                    help="Target style to apply (e.g., happy, whisper, sad)")
    ap.add_argument("--noise-level", type=float, default=0.5,
                    help="DP noise level (default: 0.5, 0 = no noise)")
    ap.add_argument("--latent-dims", type=int, default=16,
                    help="Latent dimensions of the VAE (default: 16 = 11 style + 5 free)")
    ap.add_argument("--device", default=None, help="Device (default: auto-detect)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Custom F0 control (overrides preset if provided)
    ap.add_argument("--pitch-shift", type=float, default=None,
                    help="F0 pitch multiplier (e.g., 1.2 = 20%% higher)")
    ap.add_argument("--range-scale", type=float, default=None,
                    help="F0 range scale (>1 = more expressive, <1 = flatter)")
    ap.add_argument("--flatten", type=float, default=None,
                    help="F0 flatten amount (0=none, 1=monotone)")
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

    # Build F0 transform from style preset or custom args
    f0_transform = None
    if args.style and args.style in F0_PRESETS:
        f0_transform = dict(F0_PRESETS[args.style])
        print(f"Style '{args.style}' F0 preset: {f0_transform}")

    # Custom args override preset values
    if args.pitch_shift is not None:
        f0_transform = f0_transform or {}
        f0_transform['pitch_shift'] = args.pitch_shift
    if args.range_scale is not None:
        f0_transform = f0_transform or {}
        f0_transform['range_scale'] = args.range_scale
    if args.flatten is not None:
        f0_transform = f0_transform or {}
        f0_transform['flatten'] = args.flatten

    if f0_transform:
        print(f"F0 transform: {f0_transform}")

    # Run anonymization
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anonymizer.anonymize(
        args.source, args.out,
        noise_level=args.noise_level,
        seed=args.seed,
        f0_transform=f0_transform,
    )
    print(f"Saved anonymized audio to: {args.out}")


if __name__ == "__main__":
    main()
