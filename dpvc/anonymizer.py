import torch
import os
import random

from .model_embedding_vae import VariationalAutoencoder
from . import utils

class Anonymizer:
    def __init__(self, vc_wrapper, vae_checkpoint_path=None, vae_input_dim=256, vae_latent_dim=6):
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        self.vc_wrapper = vc_wrapper

        if vae_checkpoint_path is None:
            local_path = os.path.dirname(os.path.abspath(__file__))
            ae_path = f'{local_path}/openvoice_embedding_vae.pt'
        else:
            ae_path = vae_checkpoint_path

        AE = VariationalAutoencoder(latent_dims=vae_latent_dim, input_dim=vae_input_dim).to(device)
        AE.load_state_dict(torch.load(ae_path, weights_only=True, map_location=device))
        AE.eval()
        AE.clip_threshold = getattr(vc_wrapper, 'clip_threshold', 3.0)
        self.AE = AE

        # Only needed if we also want to select a random speaker to start from
        # emb_path = f'{local_path}/openvoice_random_embeddings_cv.pt'
        # emb = torch.load(emb_path).to(device).squeeze()
        # self.emb = emb

    @torch.inference_mode()
    def anonymize(self, source_file, output_file, noise_level, seed=None):
        """Anonymize the source file, using the specified noise level, writing
        to the output file"""
        self.AE.set_noise_mult(noise_level)

        utils.set_seed(seed)

        source_embedding = self.vc_wrapper.extract_embedding(source_file)

        # Only needed if we also want to select a random speaker to start from
        # num_emb, dim_emb = self.emb.shape
        # idx = random.randint(0, num_emb)
        # random_embedding = self.emb[idx].unsqueeze(0)
        # target_embedding = random_embedding.unsqueeze(-1) # (to just use a random embedding)
        # target_embedding = self.AE(random_embedding, seed=seed).unsqueeze(-1)

        target_embedding = self.AE(source_embedding.squeeze(-1), seed=seed).unsqueeze(-1)

        self.vc_wrapper.inference(
            source_file,
            output_file,
            source_embedding,
            target_embedding)
