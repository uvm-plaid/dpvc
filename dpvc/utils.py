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


def train_autoencoder(model, embeddings, epochs=1000):
    BATCH_SIZE = min(64, len(embeddings))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)#, weight_decay=1e-7)
    outputs = []
    losses = []
    beta = 1

    print(f'Training autoencoder for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            indexes = torch.randperm(embeddings.shape[0])
            embeddings = embeddings[indexes]
            embeddings_batches = torch.split(embeddings, BATCH_SIZE)

        for embeddings_b in embeddings_batches:
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = beta*model.encoder.kl
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

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
