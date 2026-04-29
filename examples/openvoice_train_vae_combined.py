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


def load_frozen_reference(checkpoint_path, latent_dims, input_dim, device):
    model = dpvc.VariationalAutoencoder(
        latent_dims=latent_dims, input_dim=input_dim).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location=device)
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def has_schedule(args):
    return any(
        value is not None for value in (
            args.recon_weight_final,
            args.kl_weight_final,
            args.label_weight_final,
            args.style_teacher_weight_final,
            args.free_anchor_weight_final,
        )
    )


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
    ap.add_argument("--recon-weight", type=float, default=1.0,
                    help="Initial reconstruction-loss weight (default: 1.0)")
    ap.add_argument("--kl-weight", type=float, default=1.0,
                    help="Initial KL-loss weight (default: 1.0)")
    ap.add_argument("--label-weight", type=float, default=1.0,
                    help="Initial style-label loss weight (default: 1.0)")
    ap.add_argument("--recon-weight-final", type=float, default=None,
                    help="Optional final reconstruction-loss weight for a linear schedule")
    ap.add_argument("--kl-weight-final", type=float, default=None,
                    help="Optional final KL-loss weight for a linear schedule")
    ap.add_argument("--label-weight-final", type=float, default=None,
                    help="Optional final style-label loss weight for a linear schedule")
    ap.add_argument("--style-teacher-checkpoint", default=None,
                    help="Optional teacher checkpoint for style-dimension distillation")
    ap.add_argument("--style-teacher-weight", type=float, default=0.0,
                    help="Initial weight for teacher style-dim distillation loss (default: 0.0)")
    ap.add_argument("--style-teacher-weight-final", type=float, default=None,
                    help="Optional final teacher style-dim loss weight for a linear schedule")
    ap.add_argument("--free-anchor-checkpoint", default=None,
                    help="Optional checkpoint used to anchor free latent dims. Defaults to --init-checkpoint when free-anchor loss is enabled.")
    ap.add_argument("--free-anchor-weight", type=float, default=0.0,
                    help="Initial weight for anchoring free latent dims to a frozen reference (default: 0.0)")
    ap.add_argument("--free-anchor-weight-final", type=float, default=None,
                    help="Optional final free-anchor loss weight for a linear schedule")
    ap.add_argument("--schedule-epochs", type=int, default=0,
                    help="Number of epochs over which *_final weights are reached (default: full training run if any *_final is set)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed (default: 42)")
    args = ap.parse_args()

    if args.freeze_encoder and args.freeze_decoder:
        ap.error("Refusing to freeze both encoder and decoder; nothing would remain trainable")
    if args.schedule_epochs < 0:
        ap.error("--schedule-epochs must be non-negative")
    if args.style_teacher_weight < 0 or args.free_anchor_weight < 0:
        ap.error("Auxiliary objective weights must be non-negative")
    if has_schedule(args) and args.schedule_epochs == 0:
        args.schedule_epochs = args.epochs

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
    style_dims = list(range(len(labels)))
    free_dims = list(range(len(labels), args.latent_dims))

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

    style_teacher_model = None
    if args.style_teacher_weight > 0 or args.style_teacher_weight_final is not None:
        if not args.style_teacher_checkpoint:
            ap.error("--style-teacher-checkpoint is required when teacher-style loss is enabled")
        print(f"Loading style teacher checkpoint from {args.style_teacher_checkpoint}")
        style_teacher_model = load_frozen_reference(
            args.style_teacher_checkpoint,
            latent_dims=args.latent_dims,
            input_dim=embeddings.shape[-1],
            device=device,
        )

    free_anchor_model = None
    needs_free_anchor = args.free_anchor_weight > 0 or args.free_anchor_weight_final is not None
    if needs_free_anchor:
        free_anchor_checkpoint = args.free_anchor_checkpoint or args.init_checkpoint
        if not free_anchor_checkpoint:
            ap.error("Free-anchor loss requires --free-anchor-checkpoint or --init-checkpoint")
        print(f"Loading free-anchor checkpoint from {free_anchor_checkpoint}")
        free_anchor_model = load_frozen_reference(
            free_anchor_checkpoint,
            latent_dims=args.latent_dims,
            input_dim=embeddings.shape[-1],
            device=device,
        )

    trainable_params, total_params = format_param_count(AE)
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Freeze decoder: {args.freeze_decoder}")
    print(f"Trainable parameters: {trainable_params}/{total_params}")
    print(f"Loss weights: recon={args.recon_weight}, kl={args.kl_weight}, "
          f"label={args.label_weight}, teacher={args.style_teacher_weight}, "
          f"free_anchor={args.free_anchor_weight}")
    if style_teacher_model is not None:
        print(f"Style teacher checkpoint: {args.style_teacher_checkpoint}")
        print(f"Style dims: {style_dims}")
    if free_anchor_model is not None:
        print(f"Free-anchor checkpoint: {args.free_anchor_checkpoint or args.init_checkpoint}")
        print(f"Free dims: {free_dims}")
    if has_schedule(args):
        print("Loss-weight schedule: "
              f"recon->{args.recon_weight_final if args.recon_weight_final is not None else args.recon_weight}, "
              f"kl->{args.kl_weight_final if args.kl_weight_final is not None else args.kl_weight}, "
              f"label->{args.label_weight_final if args.label_weight_final is not None else args.label_weight}, "
              f"teacher->{args.style_teacher_weight_final if args.style_teacher_weight_final is not None else args.style_teacher_weight}, "
              f"free_anchor->{args.free_anchor_weight_final if args.free_anchor_weight_final is not None else args.free_anchor_weight} "
              f"over {args.schedule_epochs} epochs")

    dpvc.utils.train_autoencoder(
        AE,
        embeddings,
        epochs=args.epochs,
        labels=labels,
        lr=args.lr,
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        label_weight=args.label_weight,
        recon_weight_final=args.recon_weight_final,
        kl_weight_final=args.kl_weight_final,
        label_weight_final=args.label_weight_final,
        schedule_epochs=args.schedule_epochs,
        style_teacher_model=style_teacher_model,
        style_teacher_weight=args.style_teacher_weight,
        style_teacher_weight_final=args.style_teacher_weight_final,
        free_anchor_model=free_anchor_model,
        free_anchor_weight=args.free_anchor_weight,
        free_anchor_weight_final=args.free_anchor_weight_final,
        style_dims=style_dims,
        free_dims=free_dims,
    )

    torch.save(AE.state_dict(), args.output)
    print(f"Saved VAE checkpoint to {args.output}")


if __name__ == "__main__":
    main()
