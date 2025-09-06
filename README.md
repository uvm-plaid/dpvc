# Differentially Private Anonymization via Voice Control

This repository provides a library for defining differentially private speaker anonymization systems using existing voice control models. The approach works for any voice control system that separates utterance information into constant-length speaker information (e.g. a speaker embedding) and time-varying content information (e.g. semantic features).

## Installation

Install the library by cloning this repository and then running:

```
pip install .
```

The library currently does not specify any dependencies, so you'll need to manually ensure that dependencies (e.g. OpenVoice) are installed.

## Example: OpenVoice

The library provides a wrapper around the OpenVoice voice control system. A minimal example of using it is as follows:

```
import dpvc
vc_wrapper = dpvc.OpenVoiceDPWrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)
anonymizer.anonymize(src_path, output_path, noise_level=1.0)
```

Here, `src_path` should be an input .wav file name, and `output_path` should be the output .wav file name. The `noise_level` parameter controls how much noise is added in the differential privacy step. The `OpenVoiceDPWrapper` object encapsulates the OpenVoice models, and the `anonymize` method performs the anonymization via differential privacy.

See the following files for examples of use:

- `examples/openvoice_inference.py` contains a more complete example of anonymization using the OpenVoice wrapper
- `examples/openvoice_train_vae.py` contains an example of how to train a custom DP-VAE for use in the anonymizer

