import dpvc

#src_path = 'trump_0.wav'
src_path = 'joe.wav'
ae_path = 'naturalspeech3_vae.pt'
#ae_path = None

vc_wrapper = dpvc.NaturalSpeech3Wrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)#, vae_checkpoint_path=ae_path)

for i in range(10):
    save_path = f'output/naturalspeech3_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=5.0, seed=None)
