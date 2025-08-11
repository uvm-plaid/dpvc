import os
import torch
import random
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter


class OpenVoiceDPWrapper:
    def __init__(self):
        ckpt_base = 'checkpoints/checkpoints_v2/base_speakers/EN'
        ckpt_converter = 'checkpoints/checkpoints_v2/converter'
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        AE = torch.load('embedding_vae.pt', map_location=torch.device('cpu')).to(device)
        AE.set_noise_mult(1.0)
        AE.clip_threshold = 3.0
        self.AE = AE

        emb = torch.load('all_emb_labeled_cv_full.pt')['data'].to(device).squeeze()
        self.emb = emb

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        self.tone_color_converter = tone_color_converter

        # reference_speaker = 'vc_systems/seed-vc/examples/source/source_s2.wav'
        # target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

    def anonymize(self, source_file, output_file, noise_level):
        self.AE.set_noise_mult(noise_level)
        source_se, _ = se_extractor.get_se(source_file, self.tone_color_converter, target_dir='processed', vad=True)

        num_emb, dim_emb = self.emb.shape
        idx = random.randint(0, num_emb)
        random_se = self.emb[idx].unsqueeze(0)
        #target_se = random_se.unsqueeze(-1)
        target_se = self.AE(random_se).unsqueeze(-1)

        self.tone_color_converter.convert(
            audio_src_path=source_file, 
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_file)
    
def main():
    src_path = 'vc_systems/seed-vc/examples/source/source_s1.wav'
    src_path = '/data/cv-corpus-21.0-2025-03-14/en/clips/common_voice_en_17870026.mp3'

    anonymizer = OpenVoiceDPWrapper()

    for i in range(10):
        save_path = f'output/openvoice_noisy_{i}.wav'
        anonymizer.anonymize(src_path, save_path, noise_level=1.0)

if __name__ == '__main__':
    main()
