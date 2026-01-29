import dpvc

#src_path = 'trump_0.wav'
src_path = 'joe.wav'
ae_path = 'vec2wav2_vae.pt'
#ae_path = None

vc_wrapper = dpvc.Vec2Wav2Wrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper,
                             vae_checkpoint_path=ae_path,
                             vae_latent_dim=16,
                             vae_input_dim=1024)

for i in range(10):
    save_path = f'output/vec2wav2_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=1.0, seed=None)
