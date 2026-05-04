# Performing Anonymization

The DPVC library is organized around two classes: `Anonymizer`, which
performs anonymization, and `VoiceControlWrapper`, a base class for
objects that wrap voice control systems. The anonymizer provides an
`anonymize` method to perform anonymization; extensions of
`VoiceControlWrapper` provide `inference` and `extract_embedding`
methods based on the underlying voice control system.

## Example: OpenVoice

DPVC provides a wrapper around the
[OpenVoice](https://github.com/myshell-ai/OpenVoice) voice control
system. The `OpenVoiceWrapper` class extends `VoiceControlWrapper` and
exposes methods for inference and extraction of speaker embeddings
using OpenVoice. A minimal example of using it is as follows:

``` py
import dpvc

# Construct the wrapper
vc_wrapper = dpvc.OpenVoiceWrapper()

# Construct the anonymizer
anonymizer = dpvc.Anonymizer(vc_wrapper)

# Perform anonymization
anonymizer.anonymize(src_path, output_path, noise_level=1.0)
```

Here, `src_path` should be an input .wav file name, and `output_path`
should be the output .wav file name. The `noise_level` parameter
controls how much noise is added in the differential privacy step. The
`OpenVoiceWrapper` object encapsulates the OpenVoice models, and the
`anonymize` method performs the anonymization via differential
privacy.

If you want to use a custom VAE checkpoint, the canonical interface is
to start from the wrapper's config dict and override the checkpoint:

``` py
vae_config = vc_wrapper.get_vae_config()
vae_config["checkpoint_path"] = "example_openvoice_vae.pt"
anonymizer = dpvc.Anonymizer(vc_wrapper, vae_config=vae_config)
```

`vae_checkpoint_path=` is still accepted as a temporary compatibility
alias for older scripts, but new code should use `vae_config`.
