import dpvc

#src_path = 'trump_0.wav'
src_path = 'joe.wav'

vc_wrapper = dpvc.OpenVoiceWrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)

for i in range(10):
    save_path = f'output/openvoice_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=1.0, seed=None)
