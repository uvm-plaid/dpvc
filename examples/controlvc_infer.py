"""
Example of using ControlVCWrapper for voice conversion.

This script demonstrates the two-stage API:
1. Extract speaker embedding from reference audio
2. Perform voice conversion with the target embedding

Usage:
    python examples/controlvc_infer.py
        --repo-root /path/to/control-vc
        --checkpoints /path/to/checkpoints
        --source source.wav
        --reference target_speaker.wav
        --out output.wav
        --device cuda
"""

from pathlib import Path
import argparse
import torchaudio
from dpvc import ControlVCWrapper


def main():
    """Run ControlVC voice conversion inference."""
    ap = argparse.ArgumentParser(description="ControlVC voice conversion example")
    ap.add_argument(
        "--repo-root",
        required=True,
        help="Path to control-vc repository root"
    )
    ap.add_argument(
        "--checkpoints",
        default=None,
        help="Path to checkpoints directory (default: repo-root/checkpoints)"
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to use for inference"
    )
    ap.add_argument(
        "--source",
        required=True,
        help="Source audio file path"
    )
    ap.add_argument(
        "--reference",
        required=True,
        help="Reference audio to extract target speaker embedding (DP noise can be injected later)"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output audio file path"
    )
    ap.add_argument(
        "--pitch-shift",
        type=float,
        default=1.0,
        help="Pitch shift multiplier (1.0 = no change)"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = ap.parse_args()

    # Initialize wrapper
    print(f"Initializing ControlVC wrapper...")
    wrapper = ControlVCWrapper(
        repo_root=Path(args.repo_root),
        checkpoints_dir=Path(args.checkpoints) if args.checkpoints else None,
        device=args.device,
        verbose=args.verbose,
    )

    # Extract target speaker embedding
    print(f"Extracting target embedding from: {args.reference}")
    target_embedding = wrapper.extract_embedding(Path(args.reference))
    print(f"  Embedding shape: {target_embedding.shape}")

    # Perform voice conversion
    print(f"Converting voice from: {args.source}")
    converted_wav = wrapper.infer(
        Path(args.source),
        target_embedding=target_embedding,
        out_sr=16000,
        pitch_shift=args.pitch_shift
    )
    print(f"  Output shape: {converted_wav.shape}")

    # Save output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, converted_wav.cpu(), 16000)
    print(f"Saved output to: {args.out}")


if __name__ == "__main__":
    main()


# Example run:
# python examples/controlvc_infer.py \
#     --repo-root /Users/steve/repos/control-vc \
#     --checkpoints /Users/steve/repos/control-vc/checkpoints \
#     --source input.wav \
#     --reference target_speaker.wav \
#     --out output.wav \
#     --device cpu \
#     --verbose
