"""
Reconstruction-only VAE pretraining on Common Voice embeddings.

Usage:
    python examples/openvoice_pretrain_vae_commonvoice.py \
        --embeddings embeddings/openvoice_commonvoice_emb.pt \
        --output embeddings/openvoice_vae_commonvoice.pt
"""

import argparse

import torch

import dpvc


def resolve_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(
        description="Pretrain VAE on Common Voice embeddings with reconstruction loss only")
    ap.add_argument("--embeddings", default="embeddings/openvoice_commonvoice_emb.pt",
                    help="Path to Common Voice embeddings .pt file")
    ap.add_argument("--output", default="embeddings/openvoice_vae_commonvoice.pt",
                    help="Output checkpoint path")
    ap.add_argument("--epochs", type=int, default=3000,
                    help="Training epochs (default: 3000)")
    ap.add_argument("--latent-dims", type=int, default=15,
                    help="Latent dimensions (default: 15)")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="Learning rate (default: 1e-6)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed (default: 42)")
    args = ap.parse_args()

    dpvc.utils.set_seed(args.seed)
    device = resolve_device()

    data = torch.load(args.embeddings, weights_only=True)
    embeddings = data["data"].to(device).squeeze()
    print(f"Embeddings shape: {embeddings.shape}")
    if "speaker_ids" in data:
        print(f"Unique speakers: {len(set(data['speaker_ids']))}")
    print(f"Latent dims: {args.latent_dims}")

    AE = dpvc.VariationalAutoencoder(
        latent_dims=args.latent_dims,
        input_dim=embeddings.shape[-1],
    ).to(device)
    dpvc.utils.train_autoencoder(
        AE,
        embeddings,
        epochs=args.epochs,
        labels=None,
        lr=args.lr,
    )

    torch.save(AE.state_dict(), args.output)
    print(f"Saved pretrained VAE checkpoint to {args.output}")


if __name__ == "__main__":
    main()
