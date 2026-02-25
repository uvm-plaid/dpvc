import dpvc
from pathlib import Path

src_path = 'wavs/example1.wav'

vc_wrapper = dpvc.ControlVCWrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)

for i in range(10):
    save_path = f'output/controlvc_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=2.0, seed=None)
