import torch
import random
import os
import sys
sys.path.append('/home/jnear/co/vec2wav2.0')


import vec2wav2
from vec2wav2.ssl_models.vqw2v_extractor import Extractor as VQW2VExtractor
from vec2wav2.ssl_models.wavlm_extractor import Extractor as WavLMExtractor
# from vec2wav2.ssl_models.w2v2_extractor import Extractor as W2V2Extractor
from vec2wav2.utils.utils import load_model, load_feat_codebook, idx2vec, read_wav_16k
import soundfile as sf
import yaml

from . import utils
from . import VoiceControlWrapper

expdir = '/home/jnear/co/vec2wav2.0/pretrained'

class Vec2Wav2Wrapper(VoiceControlWrapper):
    clip_threshold = 5.0
    post_clip_threshold = 3.0

    def __init__(self):
        local_path = os.path.dirname(os.path.abspath(__file__))
        self.default_vae_path = f'{local_path}/naturalspeech3_embedding_vae.pt'

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # set up token extractor
        token_extractor = expdir + '/vq-wav2vec_kmeans.pt'
        self.token_extractor = VQW2VExtractor(checkpoint=token_extractor, device=self.device)
        feat_codebook, feat_codebook_numgroups = load_feat_codebook(self.token_extractor.get_codebook(), self.device)
        self.feat_codebook = feat_codebook
        self.feat_codebook_numgroups = feat_codebook_numgroups
        
        # set up prompt extractor
        prompt_extractor = expdir + '/WavLM-Large.pt'
        prompt_output_layer = 6
        self.prompt_extractor = WavLMExtractor(prompt_extractor, device=self.device, output_layer=prompt_output_layer)
        
        # load VC model
        self.config_path = os.path.join(expdir, "config.yml")
        with open(self.config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)
        checkpoint = os.path.join(expdir, "generator.ckpt")
        self.model = load_model(checkpoint, self.config)
        
        self.model.backend.remove_weight_norm()
        self.model.eval().to(self.device)

    @torch.no_grad
    def extract_embedding(self, source_file) -> torch.Tensor:
        """Extract the speaker embedding from a source .wav file"""

        target_wav = read_wav_16k(source_file)

        prompt = self.prompt_extractor.extract(target_wav).unsqueeze(0).to(self.device)
        new_prompt = prompt.mean(dim=1)

        return new_prompt

    @torch.no_grad
    def inference(self, source_file, output_file, source_embedding, target_embedding):
        """Perform inference with a source file and target speaker embedding,
        writing to the output file"""

        source_wav = read_wav_16k(source_file).squeeze()
        vq_idx = self.token_extractor.extract(source_wav).long().to(self.device)
        vqvec = idx2vec(self.feat_codebook, vq_idx, self.feat_codebook_numgroups).unsqueeze(0)

        prompt = target_embedding.unsqueeze(1)
        converted = self.model.inference(vqvec, prompt)[-1].view(-1)
        sf.write(output_file, converted.cpu().numpy(), 24000)
