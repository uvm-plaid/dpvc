import torch
import os
import random

from .model_embedding_vae import VariationalAutoencoder
from . import utils

class Anonymizer:
    #def __init__(self, vc_wrapper, vae_checkpoint_path=None, vae_input_dim=256, vae_latent_dim=6):
    def __init__(self, vc_wrapper, vae_config=None):
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        self.vc_wrapper = vc_wrapper

        if vae_config is None:
            vae_config = vc_wrapper.get_vae_config()

        ae_path = vae_config['checkpoint_path']

        AE = VariationalAutoencoder(latent_dims=vae_config['latent_dim'],
                                    input_dim=vae_config['input_dim'],
                                    clip_threshold=vae_config['clip_threshold'],
                                    post_clip_threshold=vae_config['post_clip_threshold']
                                    ).to(device)
        AE.load_state_dict(torch.load(ae_path, weights_only=True, map_location=device))
        AE.eval()
        self.AE = AE

    @torch.inference_mode()
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
