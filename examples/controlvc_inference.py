import dpvc
from pathlib import Path

src_path = 'trump_0.wav'
ae_path = 'controlvc_vae.pt'
# ae_path = None

vc_wrapper = dpvc.ControlVCWrapper(repo_root=Path("/home/jnear/co/cvc/control-vc"))
anonymizer = dpvc.Anonymizer(vc_wrapper, vae_checkpoint_path=ae_path)

for i in range(10):
    save_path = f'output/controlvc_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=2.0, seed=None)
