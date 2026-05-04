"""
CommonVoice VAE pretraining with optional weak supervision.

Usage:
    python examples/openvoice_pretrain_vae_commonvoice.py \
        --embeddings embeddings/openvoice_commonvoice_emb.pt \
        --output embeddings/openvoice_vae_commonvoice.pt
"""

import argparse

import torch

import dpvc


UNIFIED_STYLES = ['anger', 'confused', 'disgust', 'enunciated', 'fear',
                  'happy', 'neutral', 'sad', 'whisper']
STYLE_TO_INDEX = {style: idx for idx, style in enumerate(UNIFIED_STYLES)}
SUPPORTED_METADATA_TARGETS = {'gender', 'age_bucket', 'accent'}
AGE_BUCKETS = {
    'teens': 'young',
    'twenties': 'young',
    'thirties': 'adult',
    'fourties': 'adult',
    'forties': 'adult',
    'fifties': 'adult',
    'sixties': 'senior',
    'seventies': 'senior',
    'eighties': 'senior',
    'nineties': 'senior',
}


def resolve_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_metadata_targets(raw):
    if not raw:
        return []
    targets = [item.strip() for item in raw.split(',') if item.strip()]
    invalid = [item for item in targets if item not in SUPPORTED_METADATA_TARGETS]
    if invalid:
        raise ValueError(
            f"Unsupported metadata targets: {invalid}. "
            f"Supported: {sorted(SUPPORTED_METADATA_TARGETS)}"
        )
    return targets


def parse_dims(raw, expected_default):
    if not raw:
        return list(expected_default)
    return [int(item.strip()) for item in raw.split(',') if item.strip()]


def normalize_gender(value):
    if dpvc.utils.is_missing_metadata(value):
        return None
    value = str(value).strip().lower()
    if value == 'male_masculine':
        return 'male'
    if value == 'female_feminine':
        return 'female'
    return value


def normalize_age_bucket(value):
    if dpvc.utils.is_missing_metadata(value):
        return None
    value = str(value).strip().lower()
    return AGE_BUCKETS.get(value)


def normalize_accent(value):
    if dpvc.utils.is_missing_metadata(value):
        return None
    return str(value).strip()


def build_metadata_specs(data, target_names, free_dim_count, device, min_count):
    target_values = {
        'gender': [normalize_gender(value) for value in data.get('gender', [])],
        'age_bucket': [normalize_age_bucket(value) for value in data.get('age', [])],
        'accent': [normalize_accent(value) for value in data.get('accent', [])],
    }

    specs = []
    for name in target_names:
        values = target_values[name]
        counts = {}
        for value in values:
            if value is None:
                continue
            counts[value] = counts.get(value, 0) + 1
        classes = sorted([value for value, count in counts.items() if count >= min_count])
        if not classes:
            print(f"Skipping metadata target '{name}' because no classes met min_count={min_count}")
            continue
        class_to_index = {value: idx for idx, value in enumerate(classes)}
        indices = torch.full((len(values),), -1, dtype=torch.long, device=device)
        for idx, value in enumerate(values):
            if value in class_to_index:
                indices[idx] = class_to_index[value]
        known_count = int((indices >= 0).sum().item())
        class_weights = torch.tensor(
            [known_count / (len(classes) * counts[class_name]) for class_name in classes],
            dtype=torch.float32,
            device=device,
        )
        head = torch.nn.Linear(free_dim_count, len(classes)).to(device)
        specs.append({
            'name': name,
            'classes': classes,
            'indices': indices,
            'known_count': known_count,
            'class_weights': class_weights,
            'head': head,
        })
    return specs


def build_pseudo_style_targets(data, device, threshold):
    pseudo_style = data.get('pseudo_style')
    pseudo_conf = data.get('pseudo_style_confidence')
    if pseudo_style is None or pseudo_conf is None:
        raise ValueError(
            "Pseudo-style pretraining requires 'pseudo_style' and "
            "'pseudo_style_confidence' in the embeddings artifact. "
            "Run scripts/annotate_commonvoice_pseudolabels.py first."
        )

    targets = torch.zeros((len(pseudo_style), len(UNIFIED_STYLES)), device=device)
    mask = torch.zeros(len(pseudo_style), dtype=torch.bool, device=device)
    accepted = {}
    for idx, (style, confidence) in enumerate(zip(pseudo_style, pseudo_conf)):
        if style is None or confidence is None:
            continue
        if float(confidence) < threshold:
            continue
        if style not in STYLE_TO_INDEX:
            continue
        mask[idx] = True
        targets[idx, STYLE_TO_INDEX[style]] = 1.0
        accepted[style] = accepted.get(style, 0) + 1
    row_weights = torch.zeros(len(pseudo_style), dtype=torch.float32, device=device)
    total = sum(accepted.values())
    if total:
        num_classes = len(accepted)
        per_style_weight = {
            style: total / (num_classes * count)
            for style, count in accepted.items()
        }
        for idx, style in enumerate(pseudo_style):
            if mask[idx]:
                row_weights[idx] = per_style_weight[style]
    return targets, mask, accepted, row_weights


def main():
    ap = argparse.ArgumentParser(
        description="Pretrain VAE on Common Voice embeddings with optional weak supervision")
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
    ap.add_argument("--recon-weight", type=float, default=1.0,
                    help="Reconstruction-loss weight (default: 1.0)")
    ap.add_argument("--kl-weight", type=float, default=1.0,
                    help="KL-loss weight (default: 1.0)")
    ap.add_argument("--metadata-targets", default="",
                    help="Comma-separated metadata targets to supervise (supported: gender, age_bucket, accent)")
    ap.add_argument("--metadata-weight", type=float, default=0.0,
                    help="Metadata auxiliary-loss weight (default: 0.0)")
    ap.add_argument("--metadata-min-count", type=int, default=10,
                    help="Minimum known-row count required for a metadata class to be used (default: 10)")
    ap.add_argument("--pseudo-style-weight", type=float, default=0.0,
                    help="Pseudo-style auxiliary-loss weight (default: 0.0)")
    ap.add_argument("--pseudo-style-threshold", type=float, default=None,
                    help="Confidence threshold for pseudo-style supervision (default: use report threshold or 0.60)")
    ap.add_argument("--style-dims", default="",
                    help="Comma-separated style dims to supervise (default: first 9 dims)")
    ap.add_argument("--free-dims", default="",
                    help="Comma-separated free dims to supervise (default: dims 9..latent_dims-1)")
    args = ap.parse_args()

    dpvc.utils.set_seed(args.seed)
    device = resolve_device()

    data = torch.load(args.embeddings, weights_only=False)
    embeddings = data["data"].to(device).squeeze()
    print(f"Embeddings shape: {embeddings.shape}")
    if "speaker_ids" in data:
        print(f"Unique speakers: {len(set(data['speaker_ids']))}")
    print(f"Latent dims: {args.latent_dims}")

    metadata_report = data.get('metadata_report')
    if metadata_report is None:
        metadata_report = dpvc.utils.build_commonvoice_metadata_report(
            data.get('age', []),
            data.get('gender', []),
            data.get('accent', []),
        )
    print('Metadata coverage report:')
    for field, summary in metadata_report.items():
        print(
            f"  {field:>6s}: known={summary['known']}/{summary['total']} "
            f"missing={summary['missing']} unique={summary['unique_known']}"
        )

    style_dims = parse_dims(args.style_dims, range(min(9, args.latent_dims)))
    free_dims = parse_dims(args.free_dims, range(len(style_dims), args.latent_dims))
    if args.metadata_weight > 0 and not free_dims:
        raise ValueError("Metadata supervision requires at least one free dim")
    if args.pseudo_style_weight > 0 and not style_dims:
        raise ValueError("Pseudo-style supervision requires at least one style dim")

    AE = dpvc.VariationalAutoencoder(
        latent_dims=args.latent_dims,
        input_dim=embeddings.shape[-1],
    ).to(device)

    metadata_targets = parse_metadata_targets(args.metadata_targets)
    metadata_specs = build_metadata_specs(
        data=data,
        target_names=metadata_targets,
        free_dim_count=len(free_dims),
        device=device,
        min_count=args.metadata_min_count,
    )
    if args.metadata_weight > 0 and not metadata_specs:
        raise ValueError(
            "Metadata supervision was requested, but no metadata targets had enough labeled rows."
        )

    pseudo_style_targets = None
    pseudo_style_mask = None
    if args.pseudo_style_weight > 0:
        default_threshold = data.get('pseudo_style_report', {}).get('report_threshold', 0.60)
        threshold = args.pseudo_style_threshold if args.pseudo_style_threshold is not None else default_threshold
        pseudo_style_targets, pseudo_style_mask, accepted, pseudo_style_row_weights = build_pseudo_style_targets(
            data=data,
            device=device,
            threshold=threshold,
        )
        print(
            f"Pseudo-style supervision threshold: {threshold:.2f} "
            f"({int(pseudo_style_mask.sum().item())} accepted rows)"
        )
        for style, count in sorted(accepted.items()):
            print(f"  pseudo-style {style:8s}: {count}")
        if int(pseudo_style_mask.sum().item()) == 0:
            raise ValueError("Pseudo-style supervision was requested, but no rows met the threshold")
    else:
        pseudo_style_row_weights = None

    if args.metadata_weight == 0 and args.pseudo_style_weight == 0:
        dpvc.utils.train_autoencoder(
            AE,
            embeddings,
            epochs=args.epochs,
            labels=None,
            lr=args.lr,
            recon_weight=args.recon_weight,
            kl_weight=args.kl_weight,
        )
    else:
        dpvc.utils.train_commonvoice_pretrain(
            AE,
            embeddings,
            epochs=args.epochs,
            lr=args.lr,
            recon_weight=args.recon_weight,
            kl_weight=args.kl_weight,
            metadata_specs=metadata_specs,
            metadata_weight=args.metadata_weight,
            pseudo_style_targets=pseudo_style_targets,
            pseudo_style_mask=pseudo_style_mask,
            pseudo_style_row_weights=pseudo_style_row_weights,
            pseudo_style_weight=args.pseudo_style_weight,
            style_dims=style_dims,
            free_dims=free_dims,
        )

    torch.save(AE.state_dict(), args.output)
    print(f"Saved pretrained VAE checkpoint to {args.output}")


if __name__ == "__main__":
    main()
