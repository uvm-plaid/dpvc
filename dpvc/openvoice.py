import torch
import random
import os

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

from . import utils
from . import VoiceControlWrapper

CHECKPOINT_URL = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

class OpenVoiceWrapper(VoiceControlWrapper):
    def __init__(self):
        ckpt_path = utils.ensure_checkpoint(CHECKPOINT_URL)

        ckpt_base = f'{ckpt_path}/checkpoints_v2/base_speakers/EN'
        ckpt_converter = f'{ckpt_path}/checkpoints_v2/converter'
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        self.tone_color_converter = tone_color_converter

    def extract_embedding(self, source_file) -> torch.Tensor:
        """Extract the speaker embedding from a source .wav file"""
        source_se, _ = se_extractor.get_se(source_file, self.tone_color_converter,
                                           target_dir='processed', vad=True)
        return source_se

    def get_vae_config(self):
        """Return default VAE configuration for OpenVoice."""
        local_path = os.path.dirname(os.path.abspath(__file__))
        vae_path = f'{local_path}/openvoice_embedding_vae.pt'
        return {
            'checkpoint_path': vae_path,
            'latent_dim': 6,
            'input_dim': 256,
            'clip_threshold': 10.0,
            'post_clip_threshold': 10.0,
        }

    def inference(self, source_file, output_file, source_embedding, target_embedding):
        """Perform inference with a source file and target speaker embedding,
        writing to the output file"""
        self.tone_color_converter.convert(
            audio_src_path=source_file,
            src_se=source_embedding,
            tgt_se=target_embedding,
            output_path=output_file)
