import torch
import random
import os
import numpy as np

from ns3_codec import FACodecEncoderV2, FACodecDecoderV2
from huggingface_hub import hf_hub_download

import librosa
import soundfile as sf

from . import utils
from . import VoiceControlWrapper

class NaturalSpeech3Wrapper(VoiceControlWrapper):
    def __init__(self):
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        self._load_models()

    def get_vae_config(self):
        local_path = os.path.dirname(os.path.abspath(__file__))
        vae_path = f'{local_path}/naturalspeech3_embedding_vae.pt'

        config = {
            'checkpoint_path': vae_path,
            'latent_dim': 6,
            'input_dim': 256,
            'clip_threshold': 3.0,
            'post_clip_threshold': 1.0
        }
        return config

    def _load_models(self):
        self.fa_encoder_v2 = FACodecEncoderV2(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        self.fa_encoder_v2.to(self.device)

        self.fa_decoder_v2 = FACodecDecoderV2(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        self.fa_decoder_v2.to(self.device)

        encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec",
                                       filename="ns3_facodec_encoder_v2.bin")
        decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec",
                                       filename="ns3_facodec_decoder_v2.bin")

        self.fa_encoder_v2.load_state_dict(torch.load(encoder_ckpt, weights_only=True))
        self.fa_decoder_v2.load_state_dict(torch.load(decoder_ckpt, weights_only=True))

        self.fa_encoder_v2.eval()
        self.fa_decoder_v2.eval()

    def _load_audio(self, source_file):
        wav, sr = librosa.load(source_file, sr=16000)
        wav = np.pad(wav, (0, 200 - len(wav) % 200))
        wav = torch.tensor(wav).to(self.device).unsqueeze(0).unsqueeze(0)
        return wav

    @torch.no_grad
    def extract_embedding(self, source_file) -> torch.Tensor:
        """Extract the speaker embedding from a source .wav file"""
        wav = self._load_audio(source_file)

        enc_out = self.fa_encoder_v2(wav)
        prosody = self.fa_encoder_v2.get_prosody_feature(wav)

        _, _, _, _, spk_embs = self.fa_decoder_v2(
            enc_out, prosody, eval_vq=False, vq=True
        )

        return spk_embs

    @torch.no_grad
    def inference(self, source_file, output_file, source_embedding, target_embedding):
        """Perform inference with a source file and target speaker embedding,
        writing to the output file"""
        target_embedding = target_embedding.squeeze(-1)

        wav = self._load_audio(source_file)

        enc_out = self.fa_encoder_v2(wav)
        prosody = self.fa_encoder_v2.get_prosody_feature(wav)

        _, vq_id, _, _, _ = self.fa_decoder_v2(
            enc_out, prosody, eval_vq=False, vq=True
        )

        vq_post_emb_conv = self.fa_decoder_v2.vq2emb(vq_id, use_residual=False)
        recon_wav_conv = self.fa_decoder_v2.inference(vq_post_emb_conv, target_embedding)

        sf.write(output_file, recon_wav_conv[0, 0].cpu().numpy(), 16000)
