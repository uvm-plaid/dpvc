import torch
import os
import random

from .model_embedding_vae import VariationalAutoencoder
from . import utils

class Anonymizer:
    def __init__(self, vc_wrapper):
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        self.vc_wrapper = vc_wrapper

        # TODO: generalize beyond openvoice
        local_path = os.path.dirname(os.path.abspath(__file__))
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

    def anonymize(self, source_file, output_file, noise_level, seed=None):
        self.AE.set_noise_mult(noise_level)

        utils.set_seed(seed)

        source_embedding = self.vc_wrapper.extract_embedding(source_file)

        num_emb, dim_emb = self.emb.shape
        idx = random.randint(0, num_emb)
        random_embedding = self.emb[idx].unsqueeze(0)
        #target_embedding = random_embedding.unsqueeze(-1) # (to just use a random embedding)
        target_embedding = self.AE(random_embedding, seed=seed).unsqueeze(-1)

        self.vc_wrapper.inference(
            source_file,
            output_file,
            source_embedding,
            target_embedding)
