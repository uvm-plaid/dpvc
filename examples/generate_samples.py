# This file generates all of the samples submitted in the supplementary material

import dpvc
import os

NUM_SAMPLES = 5
vc_wrapper = dpvc.OpenVoiceWrapper()

def emit_samples(system_name, wrapper_cls):
    print('Generating samples for', system_name)

    os.makedirs(f'output/{system_name}', exist_ok=True)
    noise_levels = [0.5, 1.0, 2.0, 10.0]
    wavs = [f'example{i}' for i in range(1,6)]

    vc_wrapper = wrapper_cls()
    anonymizer = dpvc.Anonymizer(vc_wrapper)

    for wav in wavs:
        print(' Working on wav:', wav)
        for noise_level in noise_levels:
            for i in range(NUM_SAMPLES):
                src_path = f'wavs/{wav}.wav'
                save_path = f'output/{system_name}/{wav}_noise{noise_level}_{i}.wav'
                anonymizer.anonymize(src_path, save_path, noise_level=noise_level, seed=None)

emit_samples('openvoice', dpvc.OpenVoiceWrapper)
emit_samples('naturalspeech3', dpvc.NaturalSpeech3Wrapper)
emit_samples('vec2wav2', dpvc.Vec2Wav2Wrapper)
emit_samples('controlvc', dpvc.ControlVCWrapper)
