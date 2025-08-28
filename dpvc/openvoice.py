import torch
import random
import os
import numpy as np
from typing import List
import contextlib
from tqdm import tqdm
import requests
import zipfile
from pathlib import Path

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

from .model_embedding_vae import VariationalAutoencoder
from . import utils

CHECKPOINT_URL = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

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

def ensure_checkpoint():
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "openvoice_checkpoint"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "checkpoints.zip"
    ckpt_path = cache_dir / "checkpoints"
    
    if not zip_path.exists():
        print(f"Downloading model checkpoints to {zip_path}...")
        download_file(CHECKPOINT_URL, zip_path)
        extract_zip(zip_path, ckpt_path)

    return ckpt_path


class OpenVoiceDPWrapper:
    def __init__(self):
        local_path = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = ensure_checkpoint()

        ckpt_base = f'{ckpt_path}/checkpoints_v2/base_speakers/EN'
        ckpt_converter = f'{ckpt_path}/checkpoints_v2/converter'
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        ae_path = f'{local_path}/openvoice_embedding_vae.pt'
        AE = VariationalAutoencoder(latent_dims=6).to(device)
        AE.load_state_dict(torch.load(ae_path, weights_only=True))
        AE.eval()
        # AE.set_noise_mult(1.0)
        AE.clip_threshold = 3.0
        self.AE = AE

        emb_path = f'{local_path}/openvoice_random_embeddings_cv.pt'
        emb = torch.load(emb_path).to(device).squeeze()
        self.emb = emb

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        self.tone_color_converter = tone_color_converter

    def extract_embeddings(self, dataset: List[str]) -> torch.Tensor:
        embeddings = []
        print('Extracting embeddings...')
        for wav_file in tqdm(dataset):
            try:
                with contextlib.redirect_stdout(None):
                    embedding, _ = se_extractor.get_se(wav_file, self.tone_color_converter,
                                                       target_dir='processed', vad=True)

                    embeddings.append(embedding)
            except Exception as e:
                print('Error extracting embedding:', e)

        return torch.vstack(embeddings).squeeze()

    def anonymize(self, source_file, output_file, noise_level, seed=None):
        self.AE.set_noise_mult(noise_level)

        utils.set_seed(seed)

        source_se, _ = se_extractor.get_se(source_file, self.tone_color_converter, target_dir='processed', vad=True)

        num_emb, dim_emb = self.emb.shape
        idx = random.randint(0, num_emb)
        random_se = self.emb[idx].unsqueeze(0)
        #target_se = random_se.unsqueeze(-1) # (to just use a random embedding)
        target_se = self.AE(random_se, seed=seed).unsqueeze(-1)

        self.tone_color_converter.convert(
            audio_src_path=source_file, 
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_file)
