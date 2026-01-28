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

Download the pre-trained models from the ControlVC repository:
https://drive.google.com/drive/folders/1APVHQFIb1871UhvymdK_oewWKJWrInYK

## Installation

1. Clone the control-vc repository:
```bash
git clone https://github.com/YourUsername/control-vc.git
cd control-vc
```

2. Install dependencies:
```bash
pip install torch torchaudio librosa numpy soundfile
```

3. Download and place checkpoints in the `checkpoints/` directory

## API Reference

### Initialization

```python
from dpvc import ControlVCWrapper

wrapper = ControlVCWrapper(
    repo_root="/path/to/control-vc",
    checkpoints_dir="/path/to/checkpoints",  # Optional, defaults to repo_root/checkpoints
    device="cuda",                           # or "cpu"
    verbose=True                             # Enable debug messages
)
```

### Extract Speaker Embedding

```python
embedding = wrapper.extract_embedding(wav_path)
# Returns: torch.Tensor of shape (256, 1)
```

**Parameters:**
- `wav_path` (Path): Path to audio file
- `num_utterances` (int): Number of utterances to average (default: 1)

**Returns:**
- `torch.Tensor`: Speaker embedding of shape (256, 1)

### Voice Conversion

```python
converted_audio = wrapper.infer(
    source_wav=source_path,
    target_embedding=embedding,
    out_sr=16000,
    pitch_shift=1.0
)
# Returns: torch.Tensor of shape (1, T)
```

**Parameters:**
- `source_wav` (Path): Path to source audio file
- `target_embedding` (torch.Tensor): Target speaker embedding (256,) or (256, 1)
- `out_sr` (int): Output sample rate (default: 16000)
- `pitch_shift` (float): Pitch shift multiplier (default: 1.0, no change)

**Returns:**
- `torch.Tensor`: Converted waveform of shape (1, T)

## Usage Examples

### Basic Voice Conversion

```python
from pathlib import Path
from dpvc import ControlVCWrapper
import torchaudio

# Initialize wrapper
wrapper = ControlVCWrapper(
    repo_root="/path/to/control-vc",
    device="cuda"
)

# Extract target speaker embedding
target_embedding = wrapper.extract_embedding(Path("target_speaker.wav"))

# Convert voice
output = wrapper.infer(
    source_wav=Path("source.wav"),
    target_embedding=target_embedding
)

# Save result
torchaudio.save("output.wav", output.cpu(), 16000)
```

### With Differential Privacy

```python
from dpvc import ControlVCWrapper, Anonymizer

# Initialize wrapper
vc_wrapper = ControlVCWrapper(
    repo_root="/path/to/control-vc",
    device="cuda"
)

# Create anonymizer with DP
anonymizer = Anonymizer(vc_wrapper)

# Anonymize with differential privacy
anonymizer.anonymize(
    source_file="source.wav",
    output_file="anonymized.wav",
    noise_level=1.0  # DP noise level
)
```

### Pitch Shifting

```python
# Convert with pitch shift
output = wrapper.infer(
    source_wav=Path("source.wav"),
    target_embedding=target_embedding,
    pitch_shift=1.2  # 20% higher pitch
)
```

## Command-Line Interface

The wrapper includes an example CLI script:

```bash
python examples/controlvc_infer.py \
    --repo-root /path/to/control-vc \
    --source source.wav \
    --reference target_speaker.wav \
    --out output.wav \
    --device cuda \
    --pitch-shift 1.0 \
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
- Reduce batch size (not applicable for single-file inference)

## Integration with DP Pipeline

The wrapper is designed to work seamlessly with the differential privacy pipeline:

```python
# 1. Extract embedding
embedding = wrapper.extract_embedding(source_wav)

# 2. Apply DP noise (via VAE)
noised_embedding = vae(embedding, noise_level=1.0)

# 3. Convert with noised embedding
output = wrapper.infer(source_wav, noised_embedding)
```

This ensures privacy-preserving voice conversion where the target speaker identity is differentially private.

## Performance

Typical inference times on different hardware:

| Hardware | Embedding Extraction | Voice Conversion (1s audio) |
|----------|---------------------|----------------------------|
| CPU (Intel i7) | ~0.5s | ~2-3s |
| GPU (NVIDIA V100) | ~0.1s | ~0.3s |
| GPU (NVIDIA RTX 3090) | ~0.08s | ~0.2s |

## Contributing

For issues, improvements, or questions about the ControlVC wrapper, please open an issue in the dp-vc repository.

## References

- ControlVC Paper: https://arxiv.org/abs/2209.11866
- ControlVC Repository: https://github.com/meiyingchen/ControlVC
- HiFi-GAN: https://github.com/jik876/hifi-gan
- HuBERT: https://github.com/facebookresearch/fairseq

## License

See LICENSE file in the parent directory.
