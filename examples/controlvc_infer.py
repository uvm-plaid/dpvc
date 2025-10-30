# examples/controlvc_infer.py
from pathlib import Path
import argparse
import torchaudio
from dpvc import ControlVCWrapper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--checkpoints", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference", required=True, help="temporary reference for target embedding (no DP yet)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = ControlVCWrapper(
        repo_root=Path(args.repo_root),
        checkpoints_dir=Path(args.checkpoints) if args.checkpoints else None,
        device=args.device,
    )
    tgt_emb = w.extract_embedding(Path(args.reference))
    wav = w.infer(Path(args.source), target_embedding=tgt_emb)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, wav.cpu(), 16000)

if __name__ == "__main__":
    main()
    
    
# Run shape

# python examples/controlvc_infer.py \
#   --repo-root /ABS/PATH/to/control-vc \
#   --checkpoints /ABS/PATH/to/control-vc/checkpoints \
#   --device cpu \
#   --source /ABS/PATH/src.wav \
#   --reference /ABS/PATH/ref.wav \
#   --out /ABS/PATH/out.wav