import dpvc

src_path = 'trump_0.wav'
# ae_path = 'example_openvoice_vae.pt'
ae_path = None

vc_wrapper = dpvc.OpenVoiceWrapper()

if ae_path:
    vae_config = vc_wrapper.get_vae_config()
    vae_config['checkpoint_path'] = ae_path
    anonymizer = dpvc.Anonymizer(vc_wrapper, vae_config=vae_config)
else:
    anonymizer = dpvc.Anonymizer(vc_wrapper)

for i in range(10):
    save_path = f'output/openvoice_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=2.0, seed=None)
