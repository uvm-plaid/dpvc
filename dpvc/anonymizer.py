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
            ae_path = vc_wrapper.default_vae_path
        else:
            ae_path = vae_checkpoint_path

        AE = VariationalAutoencoder(latent_dims=vae_latent_dim,
                                    input_dim=vae_input_dim,
                                    clip_threshold=vc_wrapper.clip_threshold,
                                    post_clip_threshold=vc_wrapper.post_clip_threshold
                                    ).to(device)
        AE.load_state_dict(torch.load(ae_path, weights_only=True, map_location=device))
        AE.eval()
        self.AE = AE

    def anonymize(self, source_file, output_file, noise_level, seed=None):
        """Anonymize the source file, using the specified noise level, writing
        to the output file"""
        self.AE.set_noise_mult(noise_level)

        utils.set_seed(seed)

        source_embedding = self.vc_wrapper.extract_embedding(source_file)
        target_embedding = self.AE(source_embedding.squeeze(-1), seed=seed)#.unsqueeze(-1)

        self.vc_wrapper.inference(
            source_file,
            output_file,
            source_embedding,
            target_embedding)
