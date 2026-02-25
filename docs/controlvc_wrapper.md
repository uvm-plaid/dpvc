# ControlVC Wrapper Documentation

## Overview

The `ControlVCWrapper` provides a clean, two-stage API for integrating the ControlVC voice conversion system into the differential privacy anonymization pipeline.

## Architecture

The wrapper directly loads and manages ControlVC models without subprocess execution, providing:

1. **Speaker Embedding Extraction**: Extract 256-dimensional speaker embeddings using the D_VECTOR model
2. **Voice Conversion**: Convert source audio to target speaker using CodeGenerator (HiFi-GAN based)

## Required Checkpoints

The wrapper expects the following checkpoint structure:

```
checkpoints/
├── embed_f0stat2/              # Main VC model
│   ├── config.json             # Model configuration
│   └── g_XXXXXXXX              # Generator checkpoint
├── 3000000-BL.ckpt             # Speaker embedding model (D_VECTOR)
├── hubert_base_ls960.pt        # HuBERT model (optional, for content extraction)
└── km.bin                      # K-means quantizer (optional, for HuBERT codes)
```

### Downloading Checkpoints

Download the pre-trained models from the ControlVC Google Drive:
https://drive.google.com/drive/folders/1APVHQFIb1871UhvymdK_oewWKJWrInYK

## Installation

See `docs/controlvc_setup.md` for the full setup guide. The quick path is:

```bash
bash scripts/setup.sh
source .venv310/bin/activate
```

## API Reference

### Initialization

```python
from pathlib import Path
from dpvc import ControlVCWrapper

wrapper = ControlVCWrapper(
    repo_root=Path("~/repos/control-vc").expanduser(),
    device="cpu",    # or "cuda"
    verbose=True     # Enable debug messages
)
```

### Extract Speaker Embedding

```python
embedding = wrapper.extract_embedding(wav_path)
# Returns: torch.Tensor of shape (1, 256)
```

**Parameters:**
- `wav_path` (Path): Path to audio file
- `num_utterances` (int): Number of utterances to average (default: 1)

**Returns:**
- `torch.Tensor`: Speaker embedding of shape (1, 256)

### Voice Conversion

```python
wrapper.inference(source_file, output_file, source_embedding, target_embedding)
# Writes converted audio to output_file (16 kHz WAV)
```

**Parameters:**
- `source_file` (Path or str): Path to source audio file
- `output_file` (Path or str): Path to write converted audio
- `source_embedding` (torch.Tensor): Embedding of the source speaker
- `target_embedding` (torch.Tensor): Embedding of the target speaker — accepts shapes `(256,)`, `(1, 256)`, or `(256, 1)`

## Usage Examples

### Basic Voice Conversion

```python
from pathlib import Path
from dpvc import ControlVCWrapper

# Initialize wrapper
wrapper = ControlVCWrapper(
    repo_root=Path("~/repos/control-vc").expanduser(),
    device="cpu"
)

# Extract target speaker embedding
embedding = wrapper.extract_embedding(Path("target_speaker.wav"))

# Convert voice (writes directly to output.wav)
wrapper.inference(
    source_file=Path("source.wav"),
    output_file=Path("output.wav"),
    source_embedding=embedding,
    target_embedding=embedding
)
```

### With Differential Privacy

```python
from pathlib import Path
from dpvc import ControlVCWrapper, Anonymizer

vc_wrapper = ControlVCWrapper(
    repo_root=Path("~/repos/control-vc").expanduser(),
    device="cpu"
)

# Pass the ControlVC-specific VAE checkpoint
anonymizer = Anonymizer(vc_wrapper, vae_checkpoint_path="examples/controlvc_vae.pt")

anonymizer.anonymize(
    source_file="source.wav",
    output_file="anonymized.wav",
    noise_level=1.0  # higher = more privacy, less speaker diversity
)
```

## Command-Line Interface

The wrapper includes an example CLI script:

```bash
python examples/controlvc_infer.py \
    --repo-root ~/repos/control-vc \
    --source source.wav \
    --reference target_speaker.wav \
    --out output.wav \
    --device cpu \
    --verbose
```

## Technical Details

### Model Architecture

1. **D_VECTOR (Speaker Embedding)**
   - 3-layer LSTM with 768 hidden units
   - Input: 80-dimensional mel-spectrogram
   - Output: 256-dimensional normalized embedding

2. **CodeGenerator (Voice Conversion)**
   - HiFi-GAN based architecture
   - Input: Content codes + F0 + speaker embedding
   - Output: 16kHz waveform
   - Features:
     - Content code embedding (100 vocab, 128 dims)
     - F0 quantization for pitch control
     - Multi-scale residual blocks
     - 5-stage upsampling (5×4×4×2×2 = 320× upsampling)

### Content Extraction

The wrapper supports two modes for content code extraction:

1. **HuBERT + K-means** (Recommended):
   - Requires `hubert_base_ls960.pt` and `km.bin` checkpoints
   - Extracts semantic content features
   - Quantizes to 100-vocabulary discrete codes

2. **Dummy Codes** (Fallback):
   - Used when HuBERT checkpoints are not available
   - Creates zero-codes (placeholder)
   - **Note**: Will not produce good conversion results

### F0 Extraction

- Uses YAAPT (Yet Another Audio Pitch Tracker)
- Frame length: 20ms, Frame spacing: 5ms
- Interpolates unvoiced regions
- Optional speaker-specific normalization

## Limitations

1. **Content Codes**: Currently uses dummy codes if HuBERT is not available. For production use, download HuBERT checkpoints.

2. **Sample Rate**: Models are trained at 16kHz. Input audio is resampled automatically.

3. **Audio Length**: Very long audio files may require chunking to avoid memory issues.

4. **Language**: Models are primarily trained on English speech.

## Troubleshooting

### Import Error: "Failed to import ControlVC modules"

**Solution**: Ensure the control-vc repository contains:
- `models.py`
- `dataset.py`
- `utils.py`
- `fairseq_feature_reader.py` (if using HuBERT)

### Warning: "Using dummy content codes"

**Solution**: Download HuBERT and K-means checkpoints:
- `hubert_base_ls960.pt`
- `km.bin`

Place them in the checkpoints directory.

### RuntimeError: "Speaker embedding model not loaded"

**Solution**: Ensure `3000000-BL.ckpt` exists in the checkpoints directory.

### CUDA Out of Memory

**Solutions**:
- Use `device="cpu"` instead of CUDA
- Process shorter audio segments

## Integration with DP Pipeline

The wrapper is designed to work seamlessly with the differential privacy pipeline:

```python
# 1. Extract embedding
embedding = wrapper.extract_embedding(source_wav)

# 2. Apply DP noise (via VAE + Anonymizer)
anonymizer = Anonymizer(wrapper, vae_checkpoint_path="examples/controlvc_vae.pt")
anonymizer.anonymize(source_wav, output_wav, noise_level=1.0)
```

This ensures privacy-preserving voice conversion where the target speaker identity is differentially private.

## References

- ControlVC Paper: https://arxiv.org/abs/2209.11866
- ControlVC Repository: https://github.com/zuruoke/control-vc
- HiFi-GAN: https://github.com/jik876/hifi-gan
- HuBERT: https://github.com/facebookresearch/fairseq

## License

See LICENSE file in the parent directory.
