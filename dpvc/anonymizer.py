import torch

from .model_embedding_vae import VariationalAutoencoder
from . import utils

class Anonymizer:
    def __init__(self, vc_wrapper, vae_config=None, vae_checkpoint_path=None):
        """Create an anonymizer for a wrapper-backed VC system.

        `vae_config` is the canonical interface. `vae_checkpoint_path` is kept
        as a temporary compatibility alias so older examples continue to work
        while the docs migrate to the config-dict pattern.
        """
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        self.vc_wrapper = vc_wrapper

        if vae_config is None:
            vae_config = vc_wrapper.get_vae_config()
        else:
            vae_config = dict(vae_config)

        if vae_checkpoint_path is not None:
            vae_config['checkpoint_path'] = vae_checkpoint_path

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
    def anonymize(self, source_file, output_file, noise_level, seed=None,
                  control_features=None):
        """Anonymize the source file, using the specified noise level, writing
        to the output file"""
        self.AE.set_noise_mult(noise_level)

        utils.set_seed(seed)

        source_embedding = self.vc_wrapper.extract_embedding(source_file)
        target_embedding = self.AE(source_embedding.squeeze(-1),
                                   seed=seed,
                                   control_features=control_features)

        self.vc_wrapper.inference(
            source_file,
            output_file,
            source_embedding,
            target_embedding)
