import dpvc

src_path = 'wavs/example1.wav'

vc_wrapper = dpvc.Vec2Wav2Wrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)

for i in range(10):
    save_path = f'output/vec2wav2_noisy_{i}.wav'
    anonymizer.anonymize(src_path, save_path, noise_level=1.0, seed=None)
