"""
Example of using ControlVCWrapper for voice conversion.

This script demonstrates the two-stage API:
1. Extract speaker embedding from reference audio
2. Perform voice conversion with the target embedding

Usage:
    python examples/controlvc_infer.py
        --repo-root /path/to/control-vc
        --source source.wav
        --reference target_speaker.wav
        --out output.wav
        --device cpu
"""

from pathlib import Path
import argparse
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
        help="Reference audio to extract target speaker embedding"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output audio file path"
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
        device=args.device,
        verbose=args.verbose,
    )

    # Extract embeddings
    print(f"Extracting source embedding from: {args.source}")
    source_embedding = wrapper.extract_embedding(Path(args.source))

    print(f"Extracting target embedding from: {args.reference}")
    target_embedding = wrapper.extract_embedding(Path(args.reference))
    print(f"  Embedding shape: {target_embedding.shape}")

    # Perform voice conversion (writes directly to output file)
    print(f"Converting voice from: {args.source}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wrapper.inference(
        source_file=Path(args.source),
        output_file=args.out,
        source_embedding=source_embedding,
        target_embedding=target_embedding,
    )
    print(f"Saved output to: {args.out}")


if __name__ == "__main__":
    main()


# Example run:
# python examples/controlvc_infer.py \
#     --repo-root ~/repos/control-vc \
#     --source examples/trump_0.wav \
#     --reference examples/trump_0.wav \
#     --out output.wav \
#     --device cpu \
#     --verbose
