import torch
import random
import os
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from .model_embedding_vae import VariationalAutoencoder

class OpenVoiceDPWrapper:
    def __init__(self):
        local_path = os.path.dirname(os.path.abspath(__file__))

        ckpt_base = '../checkpoints/checkpoints_v2/base_speakers/EN'
        ckpt_converter = '../checkpoints/checkpoints_v2/converter'
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        ae_path = f'{local_path}/embedding_vae.pt'
        AE = torch.load(ae_path, map_location=torch.device('cpu')).to(device)
        # AE.set_noise_mult(1.0)
        AE.clip_threshold = 3.0
        self.AE = AE

        emb_path = f'{local_path}/all_emb_labeled_cv_full.pt'
        emb = torch.load(emb_path)['data'].to(device).squeeze()
        self.emb = emb

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        self.tone_color_converter = tone_color_converter

    def anonymize(self, source_file, output_file, noise_level):
        self.AE.set_noise_mult(noise_level)
        source_se, _ = se_extractor.get_se(source_file, self.tone_color_converter, target_dir='processed', vad=True)

        num_emb, dim_emb = self.emb.shape
        idx = random.randint(0, num_emb)
        random_se = self.emb[idx].unsqueeze(0)
        #target_se = random_se.unsqueeze(-1) # (to just use a random embedding)
        target_se = self.AE(random_se).unsqueeze(-1)

        self.tone_color_converter.convert(
            audio_src_path=source_file, 
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_file)
