import torch
import random
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import requests
import zipfile
import contextlib
from collections import Counter
from typing import List, Optional, Sequence


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

def extract_embeddings(vc_wrapper, dataset: List[str]) -> torch.Tensor:
    """Extract speaker embeddings from many source .wav files"""
    embeddings = []
    print('Extracting embeddings...')
    for wav_file in tqdm(dataset):
        try:
            with contextlib.redirect_stdout(None):
                embedding = vc_wrapper.extract_embedding(wav_file)
                embeddings.append(embedding)
        except Exception as e:
            print('Error extracting embedding:', e)

    return torch.vstack(embeddings).squeeze()


def _interpolate_weight(start, end, epoch, schedule_epochs):
    if end is None or schedule_epochs <= 0:
        return start
    progress = min(max(epoch, 0), schedule_epochs) / schedule_epochs
    return start + (end - start) * progress


def _slice_or_none(tensor: torch.Tensor, dims: Optional[Sequence[int]]):
    if dims is None:
        return None
    if not dims:
        return tensor.new_zeros((tensor.shape[0], 0))
    return tensor[:, list(dims)]


def is_missing_metadata(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def summarize_categorical_values(values, top_n=10):
    total = len(values)
    normalized = [str(value) for value in values if not is_missing_metadata(value)]
    counts = Counter(normalized)
    return {
        'total': total,
        'known': len(normalized),
        'missing': total - len(normalized),
        'unique_known': len(counts),
        'top_values': [
            {'value': value, 'count': count}
            for value, count in counts.most_common(top_n)
        ],
    }


def build_commonvoice_metadata_report(age_values, gender_values, accent_values):
    return {
        'age': summarize_categorical_values(age_values),
        'gender': summarize_categorical_values(gender_values),
        'accent': summarize_categorical_values(accent_values),
    }


def train_autoencoder(model, embeddings, epochs=1000, labels=None, lr=1e-5,
                      recon_weight=1.0, kl_weight=1.0, label_weight=1.0,
                      recon_weight_final=None, kl_weight_final=None,
                      label_weight_final=None, schedule_epochs=0,
                      style_teacher_model=None, style_teacher_weight=0.0,
                      style_teacher_weight_final=None,
                      free_anchor_model=None, free_anchor_weight=0.0,
                      free_anchor_weight_final=None,
                      style_dims=None, free_dims=None):
    BATCH_SIZE = min(256, len(embeddings))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain in the model")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if labels is not None:
        num_labels = len(labels.keys())
        print('Label ordering:', [f for f in labels])
        label_vals = [list(labels[f]) for f in labels]
        label_tensor = torch.tensor(label_vals).to(embeddings.device).squeeze().T
    else:
        num_labels = 0
        label_tensor = None

    if schedule_epochs <= 0:
        schedule_epochs = 0

    print('Loss weights:')
    print(f'  recon: {recon_weight}'
          + (f' -> {recon_weight_final}' if recon_weight_final is not None else ''))
    print(f'  kl   : {kl_weight}'
          + (f' -> {kl_weight_final}' if kl_weight_final is not None else ''))
    print(f'  label: {label_weight}'
          + (f' -> {label_weight_final}' if label_weight_final is not None else ''))
    print(f'  teacher-style: {style_teacher_weight}'
          + (f' -> {style_teacher_weight_final}' if style_teacher_weight_final is not None else ''))
    print(f'  free-anchor : {free_anchor_weight}'
          + (f' -> {free_anchor_weight_final}' if free_anchor_weight_final is not None else ''))
    if schedule_epochs:
        print(f'  schedule epochs: {schedule_epochs}')
    if style_teacher_model is not None:
        print(f'  style dims for teacher loss: {list(style_dims or [])}')
    if free_anchor_model is not None:
        print(f'  free dims for anchor loss: {list(free_dims or [])}')

    print(f'Training autoencoder for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        current_recon_weight = _interpolate_weight(
            recon_weight, recon_weight_final, epoch, schedule_epochs)
        current_kl_weight = _interpolate_weight(
            kl_weight, kl_weight_final, epoch, schedule_epochs)
        current_label_weight = _interpolate_weight(
            label_weight, label_weight_final, epoch, schedule_epochs)
        current_style_teacher_weight = _interpolate_weight(
            style_teacher_weight, style_teacher_weight_final, epoch, schedule_epochs)
        current_free_anchor_weight = _interpolate_weight(
            free_anchor_weight, free_anchor_weight_final, epoch, schedule_epochs)

        with torch.no_grad():
            indexes = torch.randperm(embeddings.shape[0])
            embeddings_batches = torch.split(embeddings[indexes], BATCH_SIZE)
            if label_tensor is not None:
                labels_batches = torch.split(label_tensor[indexes], BATCH_SIZE)
            else:
                labels_batches = [None] * len(embeddings_batches)

        for embeddings_b, labels_b in zip(embeddings_batches, labels_batches):
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = model.kl
            student_mu = model.last_mu

            if labels_b is not None:
                label_loss = ((model.last_z[:, :num_labels] - labels_b)**2).sum()
            else:
                label_loss = embeddings_b.new_tensor(0.0)

            if style_teacher_model is not None and style_dims:
                with torch.no_grad():
                    teacher_mu, _ = style_teacher_model.encoder(embeddings_b)
                teacher_style_loss = (
                    (_slice_or_none(student_mu, style_dims) - _slice_or_none(teacher_mu, style_dims)) ** 2
                ).sum()
            else:
                teacher_style_loss = embeddings_b.new_tensor(0.0)

            if free_anchor_model is not None and free_dims:
                with torch.no_grad():
                    anchor_mu, _ = free_anchor_model.encoder(embeddings_b)
                free_anchor_loss = (
                    (_slice_or_none(student_mu, free_dims) - _slice_or_none(anchor_mu, free_dims)) ** 2
                ).sum()
            else:
                free_anchor_loss = embeddings_b.new_tensor(0.0)

            weighted_recon = current_recon_weight * recon_loss
            weighted_kl = current_kl_weight * kl_loss
            weighted_label = current_label_weight * label_loss
            weighted_teacher_style = current_style_teacher_weight * teacher_style_loss
            weighted_free_anchor = current_free_anchor_weight * free_anchor_loss
            loss = (
                weighted_recon
                + weighted_kl
                + weighted_label
                + weighted_teacher_style
                + weighted_free_anchor
            )

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(
                f'loss: {loss.item():.2f}  '
                f'recon: {recon_loss.item():.2f} (w={current_recon_weight:.2f})  '
                f'kl: {kl_loss.item():.2f} (w={current_kl_weight:.2f})  '
                f'label: {label_loss.item():.2f} (w={current_label_weight:.2f})  '
                f'teacher: {teacher_style_loss.item():.2f} (w={current_style_teacher_weight:.2f})  '
                f'anchor: {free_anchor_loss.item():.2f} (w={current_free_anchor_weight:.2f})'
            )

    print('Ending loss:', loss.item())


def _normalize_dataset_masses(raw_masses, present_datasets):
    filtered = {
        dataset: max(float(raw_masses.get(dataset, 0.0)), 0.0)
        for dataset in present_datasets
    }
    total = sum(filtered.values())
    if total <= 0:
        uniform = 1.0 / max(len(present_datasets), 1)
        return {dataset: uniform for dataset in present_datasets}
    return {dataset: value / total for dataset, value in filtered.items()}


def _dataset_epoch_masses(schedule, epoch, schedule_epochs, present_datasets):
    if schedule == "static_balanced":
        return _normalize_dataset_masses(
            {"CommonVoice": 1.0, "CREMA-D": 1.0, "Expresso": 1.0},
            present_datasets,
        )

    if schedule_epochs <= 0:
        schedule_epochs = 1
    progress = min(max(epoch, 0), schedule_epochs) / schedule_epochs

    if schedule == "cv_warmup":
        start = {"CommonVoice": 0.70, "CREMA-D": 0.15, "Expresso": 0.15}
        end = {"CommonVoice": 1.0, "CREMA-D": 1.0, "Expresso": 1.0}
    elif schedule == "labeled_finish":
        start = {"CommonVoice": 1.0, "CREMA-D": 1.0, "Expresso": 1.0}
        end = {"CommonVoice": 0.20, "CREMA-D": 0.40, "Expresso": 0.40}
    else:
        raise ValueError(f"Unsupported mixed-data schedule: {schedule}")

    raw = {
        dataset: start.get(dataset, 0.0) + (end.get(dataset, 0.0) - start.get(dataset, 0.0)) * progress
        for dataset in present_datasets
    }
    return _normalize_dataset_masses(raw, present_datasets)


def train_mixed_autoencoder(model, embeddings, style_targets, style_label_mask,
                            source_datasets, epochs=1000, lr=1e-5,
                            recon_weight=1.0, kl_weight=1.0,
                            label_weight=1.0, schedule="static_balanced",
                            schedule_epochs=0, style_label_row_weights=None):
    BATCH_SIZE = min(256, len(embeddings))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain in the model")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if schedule_epochs <= 0 and schedule != "static_balanced":
        schedule_epochs = epochs

    source_datasets = [str(dataset) for dataset in source_datasets]
    dataset_counts = Counter(source_datasets)
    present_datasets = [dataset for dataset in ["CommonVoice", "CREMA-D", "Expresso"] if dataset_counts.get(dataset)]
    if not present_datasets:
        raise ValueError("No source datasets found for mixed-data training")

    print("Mixed-data training config:")
    print(f"  recon weight : {recon_weight}")
    print(f"  kl weight    : {kl_weight}")
    print(f"  label weight : {label_weight}")
    print(f"  schedule     : {schedule}")
    if schedule != "static_balanced":
        print(f"  schedule epochs: {schedule_epochs}")
    print(f"  styles       : {style_targets.shape[1]}")
    for dataset in present_datasets:
        print(f"  dataset rows {dataset:11s}: {dataset_counts[dataset]}")
    labeled_rows = int((style_label_mask.view(-1) > 0).sum().item())
    print(f"  labeled rows : {labeled_rows}/{len(embeddings)}")

    print(f"Training mixed-data autoencoder for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        dataset_masses = _dataset_epoch_masses(schedule, epoch, schedule_epochs, present_datasets)
        per_row_probs = torch.tensor(
            [dataset_masses[dataset] / dataset_counts[dataset] for dataset in source_datasets],
            dtype=torch.float32,
            device=embeddings.device,
        )
        per_row_probs = per_row_probs / per_row_probs.sum()
        sampled_indexes = torch.multinomial(
            per_row_probs,
            num_samples=embeddings.shape[0],
            replacement=True,
        )
        index_batches = torch.split(sampled_indexes, BATCH_SIZE)

        for batch_indexes in index_batches:
            embeddings_b = embeddings[batch_indexes]
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = model.kl

            batch_mask = style_label_mask[batch_indexes].view(-1) > 0
            if batch_mask.any():
                target_b = style_targets[batch_indexes][batch_mask]
                student_b = model.last_z[batch_mask, :style_targets.shape[1]]
                row_loss = ((student_b - target_b)**2).sum(dim=1)
                if style_label_row_weights is not None:
                    weights_b = style_label_row_weights[batch_indexes].view(-1)[batch_mask]
                    label_loss = (row_loss * weights_b).sum()
                else:
                    label_loss = row_loss.sum()
            else:
                label_loss = embeddings_b.new_tensor(0.0)

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + label_weight * label_loss
            )
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            masses_str = ", ".join(
                f"{dataset}={dataset_masses[dataset]:.2f}" for dataset in present_datasets
            )
            print(
                f"loss: {loss.item():.2f}  "
                f"recon: {recon_loss.item():.2f} (w={recon_weight:.2f})  "
                f"kl: {kl_loss.item():.2f} (w={kl_weight:.2f})  "
                f"label: {label_loss.item():.2f} (w={label_weight:.2f})  "
                f"mix: {masses_str}"
            )

    print('Ending loss:', loss.item())


def train_commonvoice_pretrain(model, embeddings, epochs=1000, lr=1e-5,
                               recon_weight=1.0, kl_weight=1.0,
                               metadata_specs=None, metadata_weight=0.0,
                               pseudo_style_targets=None,
                               pseudo_style_mask=None,
                               pseudo_style_row_weights=None,
                               pseudo_style_weight=0.0,
                               style_dims=None, free_dims=None):
    BATCH_SIZE = min(256, len(embeddings))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain in the model")

    metadata_specs = metadata_specs or []
    head_params = []
    for spec in metadata_specs:
        head_params.extend(spec['head'].parameters())

    optimizer = torch.optim.Adam(trainable_params + head_params, lr=lr)

    print('CommonVoice pretraining loss weights:')
    print(f'  recon        : {recon_weight}')
    print(f'  kl           : {kl_weight}')
    print(f'  metadata     : {metadata_weight}')
    print(f'  pseudo-style : {pseudo_style_weight}')
    if style_dims is not None:
        print(f'  style dims   : {list(style_dims)}')
    if free_dims is not None:
        print(f'  free dims    : {list(free_dims)}')
    for spec in metadata_specs:
        print(
            f"  metadata target {spec['name']}: "
            f"{spec['known_count']} labeled rows across {len(spec['classes'])} classes"
        )
    if pseudo_style_mask is not None:
        print(f'  pseudo-style labeled rows: {int(pseudo_style_mask.sum().item())}')

    print(f'Training CommonVoice pretrain model for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            indexes = torch.randperm(embeddings.shape[0], device=embeddings.device)
            index_batches = torch.split(indexes, BATCH_SIZE)

        for batch_indexes in index_batches:
            embeddings_b = embeddings[batch_indexes]
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = model.kl

            metadata_loss = embeddings_b.new_tensor(0.0)
            if metadata_specs and free_dims:
                free_repr = _slice_or_none(model.last_mu, free_dims)
                for spec in metadata_specs:
                    target_indices = spec['indices'][batch_indexes]
                    mask = target_indices >= 0
                    if mask.any():
                        logits = spec['head'](free_repr[mask])
                        metadata_loss = metadata_loss + torch.nn.functional.cross_entropy(
                            logits,
                            target_indices[mask],
                            reduction='sum',
                            weight=spec.get('class_weights'),
                        )

            pseudo_style_loss = embeddings_b.new_tensor(0.0)
            if (
                pseudo_style_targets is not None
                and pseudo_style_mask is not None
                and style_dims
            ):
                batch_mask = pseudo_style_mask[batch_indexes]
                if batch_mask.any():
                    style_target = pseudo_style_targets[batch_indexes][batch_mask]
                    student_style = _slice_or_none(model.last_mu[batch_mask], style_dims)
                    row_loss = ((student_style - style_target[:, list(style_dims)]) ** 2).sum(dim=1)
                    if pseudo_style_row_weights is not None:
                        row_weights = pseudo_style_row_weights[batch_indexes][batch_mask]
                        pseudo_style_loss = (row_loss * row_weights).sum()
                    else:
                        pseudo_style_loss = row_loss.sum()

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + metadata_weight * metadata_loss
                + pseudo_style_weight * pseudo_style_loss
            )

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(
                f'loss: {loss.item():.2f}  '
                f'recon: {recon_loss.item():.2f} (w={recon_weight:.2f})  '
                f'kl: {kl_loss.item():.2f} (w={kl_weight:.2f})  '
                f'metadata: {metadata_loss.item():.2f} (w={metadata_weight:.2f})  '
                f'pseudo-style: {pseudo_style_loss.item():.2f} (w={pseudo_style_weight:.2f})'
            )

    print('Ending loss:', loss.item())

def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extracts a zip file if not already extracted."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # You can check if already extracted by verifying members
        existing = all((extract_dir / name).exists() for name in zf.namelist())
        if not existing:
            print(f"Extracting {zip_path} -> {extract_dir}")
            zf.extractall(extract_dir)

def download_file(url, dest):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def ensure_checkpoint(checkpoint_url):
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "openvoice_checkpoint"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "checkpoints.zip"
    ckpt_path = cache_dir / "checkpoints"

    if not zip_path.exists():
        print(f"Downloading model checkpoints to {zip_path}...")
        download_file(checkpoint_url, zip_path)
        extract_zip(zip_path, ckpt_path)

    return ckpt_path
