"""
Train a controllable VAE on a sampled mixed-data artifact.

This script is designed for the first CommonVoice + CREMA-D + Expresso mixed
training experiments. It expects an artifact built by:

    python scripts/build_mixed_training_set.py --output embeddings/openvoice_mixed_base.pt

Usage:
    python examples/openvoice_train_vae_mixed.py \
        --embeddings embeddings/openvoice_mixed_base.pt \
        --output embeddings/openvoice_vae_mixed_static_balanced.pt \
        --schedule static_balanced

    python examples/openvoice_train_vae_mixed.py \
        --embeddings embeddings/openvoice_mixed_base.pt \
        --output embeddings/openvoice_vae_mixed_cv_warmup.pt \
        --schedule cv_warmup

    python examples/openvoice_train_vae_mixed.py \
        --embeddings embeddings/openvoice_mixed_base.pt \
        --output embeddings/openvoice_vae_mixed_labeled_finish.pt \
        --schedule labeled_finish
"""

from __future__ import annotations

import argparse

import torch

import dpvc


DEFAULT_SCHEDULES = ["static_balanced", "cv_warmup", "labeled_finish"]
DATASET_NAMES = ["CommonVoice", "CREMA-D", "Expresso"]


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


def parse_dataset_masses(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    masses = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected dataset mass in name=value form, got: {item}")
        name, value = item.split('=', 1)
        dataset = name.strip()
        if dataset not in DATASET_NAMES:
            raise ValueError(f"Unknown dataset in mass config: {dataset}")
        masses[dataset] = float(value.strip())
    return masses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--embeddings",
        default="embeddings/openvoice_mixed_base.pt",
        help="Path to mixed-data embeddings artifact",
    )
    ap.add_argument(
        "--output",
        default="embeddings/openvoice_vae_mixed_static_balanced.pt",
        help="Output VAE checkpoint path",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=3000,
        help="Training epochs (default: 3000)",
    )
    ap.add_argument(
        "--latent-dims",
        type=int,
        default=15,
        help="Latent dimensions (default: 15)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed (default: 42)",
    )
    ap.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to load before mixed-data training",
    )
    ap.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the full encoder during training",
    )
    ap.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze the full decoder during training",
    )
    ap.add_argument(
        "--recon-weight",
        type=float,
        default=1.0,
        help="Reconstruction-loss weight (default: 1.0)",
    )
    ap.add_argument(
        "--kl-weight",
        type=float,
        default=1.0,
        help="KL-loss weight (default: 1.0)",
    )
    ap.add_argument(
        "--label-weight",
        type=float,
        default=1.0,
        help="Style-label loss weight (default: 1.0)",
    )
    ap.add_argument(
        "--schedule",
        default="static_balanced",
        choices=DEFAULT_SCHEDULES,
        help="Dataset-mixture schedule (default: static_balanced)",
    )
    ap.add_argument(
        "--schedule-epochs",
        type=int,
        default=0,
        help="Epochs over which a non-static schedule interpolates (default: full run)",
    )
    ap.add_argument(
        "--static-masses",
        default="",
        help="Optional dataset masses for static schedule, e.g. CommonVoice=0.2,CREMA-D=0.4,Expresso=0.4",
    )
    ap.add_argument(
        "--schedule-start-masses",
        default="",
        help="Optional dataset masses at the start of a non-static schedule",
    )
    ap.add_argument(
        "--schedule-end-masses",
        default="",
        help="Optional dataset masses at the end of a non-static schedule",
    )
    args = ap.parse_args()

    if args.freeze_encoder and args.freeze_decoder:
        ap.error("Refusing to freeze both encoder and decoder; nothing would remain trainable")

    dpvc.utils.set_seed(args.seed)
    device = resolve_device()
    static_masses = parse_dataset_masses(args.static_masses)
    schedule_start_masses = parse_dataset_masses(args.schedule_start_masses)
    schedule_end_masses = parse_dataset_masses(args.schedule_end_masses)

    data = torch.load(args.embeddings, weights_only=False)
    embeddings = data['data'].to(device).squeeze()
    print(f"Embeddings shape: {embeddings.shape}")

    supported_styles = data.get('supported_styles')
    if not supported_styles:
        supported_styles = [
            key.replace('label_', '')
            for key in sorted(data.keys())
            if key.startswith('label_')
        ]
    label_keys = [f'label_{style}' for style in supported_styles]
    missing_keys = [key for key in label_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing label keys in mixed artifact: {missing_keys}")

    style_targets = torch.cat([data[key] for key in label_keys], dim=1).to(device)
    style_label_mask = data.get('style_label_mask')
    if style_label_mask is None:
        raise ValueError("Mixed artifact is missing style_label_mask")
    style_label_mask = style_label_mask.to(device)

    style_label_row_weights = data.get('style_label_row_weight')
    if style_label_row_weights is not None:
        style_label_row_weights = style_label_row_weights.to(device)

    source_datasets = data.get('source_dataset')
    if source_datasets is None:
        raise ValueError("Mixed artifact is missing source_dataset")

    labeled_rows = int((style_label_mask.view(-1) > 0).sum().item())
    print(f"Supported styles: {supported_styles}")
    print(f"Labeled rows: {labeled_rows}/{len(embeddings)}")

    model = dpvc.VariationalAutoencoder(
        latent_dims=args.latent_dims,
        input_dim=embeddings.shape[-1],
    ).to(device)
    if args.init_checkpoint:
        print(f"Loading init checkpoint from {args.init_checkpoint}")
        model.load_state_dict(
            torch.load(args.init_checkpoint, weights_only=True, map_location=device)
        )

    if args.freeze_encoder:
        set_module_requires_grad(model.encoder, trainable=False)
    if args.freeze_decoder:
        set_module_requires_grad(model.decoder, trainable=False)

    trainable_params, total_params = format_param_count(model)
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Freeze decoder: {args.freeze_decoder}")
    print(f"Trainable parameters: {trainable_params}/{total_params}")
    print(f"Schedule: {args.schedule}")
    if args.schedule != 'static_balanced':
        print(
            f"Schedule epochs: {args.schedule_epochs if args.schedule_epochs > 0 else args.epochs}"
        )
    if static_masses:
        print(f"Static masses: {static_masses}")
    if schedule_start_masses:
        print(f"Schedule start masses: {schedule_start_masses}")
    if schedule_end_masses:
        print(f"Schedule end masses: {schedule_end_masses}")

    dpvc.utils.train_mixed_autoencoder(
        model,
        embeddings,
        style_targets,
        style_label_mask,
        source_datasets=source_datasets,
        epochs=args.epochs,
        lr=args.lr,
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        label_weight=args.label_weight,
        schedule=args.schedule,
        schedule_epochs=args.schedule_epochs,
        style_label_row_weights=style_label_row_weights,
        static_masses=static_masses,
        schedule_start_masses=schedule_start_masses,
        schedule_end_masses=schedule_end_masses,
    )

    torch.save(model.state_dict(), args.output)
    print(f"Saved mixed-data VAE checkpoint to {args.output}")


if __name__ == '__main__':
    main()
