import dpvc

src_path = 'wavs/joe.wav'

vc_wrapper = dpvc.OpenVoiceWrapper()
vae_config = vc_wrapper.get_vae_config()
vae_config['checkpoint_path'] = 'openvoice_vae_features2.pt'
vae_config['latent_dim'] = 8

anonymizer = dpvc.Anonymizer(vc_wrapper, vae_config=vae_config)

# 0 is age, 1 is gender, 3 is accent
# feature values are between -1 and 1
control_features = {0: 2}

for i in range(1):
    save_path = f'output/openvoice_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=0, control_features=control_features)
