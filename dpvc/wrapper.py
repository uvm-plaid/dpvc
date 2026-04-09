import torch

class VoiceControlWrapper:
    def __init__(self):
        raise NotImplementedError

    def extract_embedding(self, source_file: str) -> torch.Tensor:
        """Extract the speaker embedding from a source .wav file"""
        raise NotImplementedError

    def inference(self, source_file: str, output_file: str,
                  source_embedding: torch.Tensor, target_embedding: torch.Tensor,
                  f0_transform: dict = None):
        """Perform inference with a source file and target speaker embedding,
        writing to the output file. Optionally apply f0_transform for prosody control."""
        raise NotImplementedError

    def get_vae_config(self) -> dict:
        """Return default VAE configuration for this wrapper.

        Returns a dict with keys: checkpoint_path, latent_dim, input_dim,
        clip_threshold, post_clip_threshold.
        """
        raise NotImplementedError
