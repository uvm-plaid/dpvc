"""
Train a controllable VAE using Expresso-extracted embeddings and style labels.

The trained VAE maps the first 7 latent dimensions to Expresso styles
(default, confused, enunciated, happy, laughing, sad, whisper), enabling
controllable voice anonymization.

Usage:
    python examples/controlvc_train_vae_expresso.py \
        --embeddings embeddings/controlvc_expresso_emb.pt \
        --output controlvc_vae_expresso.pt \
        --epochs 2000
"""

import argparse
import torch
import dpvc


STYLES = ['default', 'confused', 'enunciated', 'happy', 'laughing', 'sad', 'whisper',
          'emphasis', 'essentials', 'longform', 'singing']


def main():
    ap = argparse.ArgumentParser(description="Train controllable VAE with Expresso style labels")
    ap.add_argument("--embeddings", required=True,
                    help="Path to extracted embeddings .pt file from controlvc_extract_expresso.py")
    ap.add_argument("--output", default="controlvc_vae_expresso.pt",
                    help="Output VAE checkpoint path (default: controlvc_vae_expresso.pt)")
    ap.add_argument("--epochs", type=int, default=2000, help="Training epochs (default: 2000)")
    ap.add_argument("--latent-dims", type=int, default=16,
                    help="Latent dimensions (default: 16 = 11 style dims + 5 free dims)")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="Learning rate (default: 1e-6, slower for label-aware training)")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data = torch.load(args.embeddings, weights_only=True)
    embeddings = data['data'].to(device).squeeze()
    print(f"Shape of extracted embeddings: {embeddings.shape}")

    # Build labels dict from style columns
    labels = {}
    for style in STYLES:
        key = f'style_{style}'
        if key in data:
            labels[key] = data[key]
        else:
            print(f"Warning: '{key}' not found in embeddings file, skipping")

    if not labels:
        print("Error: No style labels found in embeddings file.")
        return

    print(f"Training with {len(labels)} style labels: {list(labels.keys())}")
    print(f"Latent dims: {args.latent_dims} ({len(labels)} for styles, "
          f"{args.latent_dims - len(labels)} free for speaker identity)")

    # Train the controllable VAE
    AE = dpvc.VariationalAutoencoder(latent_dims=args.latent_dims, input_dim=256).to(device)
    dpvc.utils.train_autoencoder(AE, embeddings, epochs=args.epochs, labels=labels, lr=args.lr)
    torch.save(AE.state_dict(), args.output)
    print(f"Saved VAE checkpoint to {args.output}")


if __name__ == "__main__":
    main()
