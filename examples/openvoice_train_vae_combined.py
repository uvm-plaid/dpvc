"""
Train a controllable VAE on combined CREMA-D + Expresso embeddings.

The trained VAE maps the first 9 latent dimensions to unified style labels:
  0: anger      1: confused    2: disgust     3: enunciated
  4: fear       5: happy       6: neutral     7: sad
  8: whisper

Remaining latent dimensions (9-14) are free for speaker identity encoding.

Prerequisites:
    1. Extract Expresso embeddings:
       python examples/openvoice_extract_expresso.py
    2. Extract CREMA-D embeddings:
       python examples/openvoice_extract_cremad.py
    3. Combine datasets:
       python examples/openvoice_combine_datasets.py --parquet-dir <path>

Usage:
    python examples/openvoice_train_vae_combined.py \
        --embeddings embeddings/openvoice_combined_emb.pt \
        --output embeddings/openvoice_vae_combined.pt

    # Custom settings:
    python examples/openvoice_train_vae_combined.py \
        --embeddings embeddings/openvoice_combined_emb.pt \
        --output embeddings/openvoice_vae_combined.pt \
        --epochs 3000 --latent-dims 15 --lr 1e-6

    # Finetune from Common Voice pretraining:
    python examples/openvoice_train_vae_combined.py \
        --embeddings embeddings/openvoice_combined_emb.pt \
        --output embeddings/openvoice_vae_combined_finetuned.pt \
        --init-checkpoint embeddings/openvoice_vae_commonvoice.pt
"""

import argparse
import torch
import dpvc

UNIFIED_STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
                  'happy', 'neutral', 'sad', 'whisper']


def resolve_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_module_requires_grad(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def format_param_count(model):
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def main():
    ap = argparse.ArgumentParser(
        description="Train controllable VAE on combined CREMA-D + Expresso embeddings")
    ap.add_argument("--embeddings", default="embeddings/openvoice_combined_emb.pt",
                    help="Path to combined embeddings .pt file (default: embeddings/openvoice_combined_emb.pt)")
    ap.add_argument("--output", default="embeddings/openvoice_vae_combined.pt",
                    help="Output VAE checkpoint path (default: embeddings/openvoice_vae_combined.pt)")
    ap.add_argument("--epochs", type=int, default=3000,
                    help="Training epochs (default: 3000)")
    ap.add_argument("--latent-dims", type=int, default=15,
                    help="Latent dimensions (default: 15 = 9 style dims + 6 free dims)")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="Learning rate (default: 1e-6)")
    ap.add_argument("--init-checkpoint", default=None,
                    help="Optional checkpoint to load before finetuning")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Freeze the full encoder during training")
    ap.add_argument("--freeze-decoder", action="store_true",
                    help="Freeze the full decoder during training")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed (default: 42)")
    args = ap.parse_args()

    if args.freeze_encoder and args.freeze_decoder:
        ap.error("Refusing to freeze both encoder and decoder; nothing would remain trainable")

    dpvc.utils.set_seed(args.seed)
    device = resolve_device()

    # Load combined embeddings
    data = torch.load(args.embeddings, weights_only=True)
    embeddings = data['data'].to(device).squeeze()
    print(f"Embeddings shape: {embeddings.shape}")

    # Build labels dict from label columns
    labels = {}
    for style in UNIFIED_STYLES:
        key = f'label_{style}'
        if key in data:
            labels[key] = data[key]
        else:
            print(f"Warning: '{key}' not found in embeddings file, skipping")

    if not labels:
        print("Error: No style labels found. Did you use openvoice_combine_datasets.py?")
        return

    print(f"Training with {len(labels)} style labels: {list(labels.keys())}")
    print(f"Latent dims: {args.latent_dims} ({len(labels)} for styles, "
          f"{args.latent_dims - len(labels)} free)")

    # Train
    AE = dpvc.VariationalAutoencoder(
        latent_dims=args.latent_dims, input_dim=embeddings.shape[-1]).to(device)
    if args.init_checkpoint:
        print(f"Loading init checkpoint from {args.init_checkpoint}")
        AE.load_state_dict(
            torch.load(args.init_checkpoint, weights_only=True, map_location=device)
        )

    if args.freeze_encoder:
        set_module_requires_grad(AE.encoder, trainable=False)
    if args.freeze_decoder:
        set_module_requires_grad(AE.decoder, trainable=False)

    trainable_params, total_params = format_param_count(AE)
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Freeze decoder: {args.freeze_decoder}")
    print(f"Trainable parameters: {trainable_params}/{total_params}")

    dpvc.utils.train_autoencoder(
        AE, embeddings, epochs=args.epochs, labels=labels, lr=args.lr)

    torch.save(AE.state_dict(), args.output)
    print(f"Saved VAE checkpoint to {args.output}")


if __name__ == "__main__":
    main()
