import dpvc
from pathlib import Path

src_path = 'trump_0.wav'
ae_path = 'controlvc_vae.pt'
# ae_path = None

import os
_repo = Path(os.environ.get("CONTROL_VC_DIR", "~/repos/control-vc")).expanduser()
vc_wrapper = dpvc.ControlVCWrapper(repo_root=_repo)
anonymizer = dpvc.Anonymizer(vc_wrapper, vae_checkpoint_path=ae_path)

for i in range(10):
    save_path = f'output/controlvc_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=2.0, seed=None)
