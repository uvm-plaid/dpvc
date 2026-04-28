import torch
import random
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import requests
import zipfile
import contextlib
from typing import List


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


def train_autoencoder(model, embeddings, epochs=1000, labels=None, lr=1e-5,
                      recon_weight=1.0, kl_weight=1.0, label_weight=1.0,
                      recon_weight_final=None, kl_weight_final=None,
                      label_weight_final=None, schedule_epochs=0):
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
    if schedule_epochs:
        print(f'  schedule epochs: {schedule_epochs}')

    print(f'Training autoencoder for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        current_recon_weight = _interpolate_weight(
            recon_weight, recon_weight_final, epoch, schedule_epochs)
        current_kl_weight = _interpolate_weight(
            kl_weight, kl_weight_final, epoch, schedule_epochs)
        current_label_weight = _interpolate_weight(
            label_weight, label_weight_final, epoch, schedule_epochs)

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

            if labels_b is not None:
                label_loss = ((model.last_z[:, :num_labels] - labels_b)**2).sum()
            else:
                label_loss = embeddings_b.new_tensor(0.0)

            weighted_recon = current_recon_weight * recon_loss
            weighted_kl = current_kl_weight * kl_loss
            weighted_label = current_label_weight * label_loss
            loss = weighted_recon + weighted_kl + weighted_label

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(
                f'loss: {loss.item():.2f}  '
                f'recon: {recon_loss.item():.2f} (w={current_recon_weight:.2f})  '
                f'kl: {kl_loss.item():.2f} (w={current_kl_weight:.2f})  '
                f'label: {label_loss.item():.2f} (w={current_label_weight:.2f})'
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
