import dpvc
from dpvc import model_embedding_vae

src_path = 'vc_systems/seed-vc/examples/source/source_s1.wav'
src_path = '/data/cv-corpus-21.0-2025-03-14/en/clips/common_voice_en_17870026.mp3'

anonymizer = dpvc.OpenVoiceDPWrapper()

for i in range(10):
    save_path = f'output/openvoice_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=1.0)
