import dpvc

#src_path = 'trump_0.wav'
src_path = 'joe.wav'
#ae_path = 'openvoice_vae.pt'
ae_path = None

vc_wrapper = dpvc.OpenVoiceWrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper, vae_checkpoint_path=ae_path)

for i in range(10):
    save_path = f'output/openvoice_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=10.0, seed=None)
