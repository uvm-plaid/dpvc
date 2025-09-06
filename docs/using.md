# Performing Anonymization

## Example: OpenVoice

The library provides a wrapper around the OpenVoice voice control system. A minimal example of using it is as follows:

``` py
import dpvc
vc_wrapper = dpvc.OpenVoiceDPWrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)
anonymizer.anonymize(src_path, output_path, noise_level=1.0)
```

Here, `src_path` should be an input .wav file name, and `output_path` should be the output .wav file name. The `noise_level` parameter controls how much noise is added in the differential privacy step. The `OpenVoiceDPWrapper` object encapsulates the OpenVoice models, and the `anonymize` method performs the anonymization via differential privacy.
