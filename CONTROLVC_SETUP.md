# ControlVC Wrapper Setup Guide

## Quick Start

This guide will help you set up and use the ControlVC wrapper in the dp-vc repository.

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support (optional, for GPU acceleration)
- Control-VC repository cloned locally

## Step 1: Clone Control-VC Repository

```bash
cd ~/repos  # or your preferred location
git clone https://github.com/meiyingchen/ControlVC.git control-vc
cd control-vc
```

## Step 2: Download Checkpoints

Download all required checkpoints from the ControlVC Google Drive:
https://drive.google.com/drive/folders/1APVHQFIb1871UhvymdK_oewWKJWrInYK

Place the checkpoints in the following structure:

```
control-vc/
└── checkpoints/
    ├── embed_f0stat2/
    │   ├── config.json
    │   └── g_00400000
    ├── 3000000-BL.ckpt
    ├── hubert_base_ls960.pt
    ├── km.bin
    └── vctk_f0_vq/
        └── g_00400000
```

**Minimum Required (for basic functionality):**
- `embed_f0stat2/` directory with config and checkpoint
- `3000000-BL.ckpt` (speaker embedding model)

**Optional (for better quality):**
- `hubert_base_ls960.pt` (HuBERT model for content extraction)
- `km.bin` (K-means quantizer)

## Step 3: Apply Critical Fix to Control-VC

**IMPORTANT**: The control-vc repository needs a small fix for Apple Silicon (M1/M2) compatibility.

Edit `control-vc/fairseq_feature_reader.py` line 22-24:

**Before:**
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
```

**After:**
```python
# Force CPU for compatibility - MPS doesn't support all fairseq operators
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This prevents MPS device errors on Apple Silicon Macs.

## Step 4: Install Dependencies

```bash
# In the dp-vc repository
cd ~/UVM-plaid/dp-vc
pip install -e .

# Install critical dependencies for HuBERT extraction
# Note: Use older pip version to avoid omegaconf metadata issues
pip install 'pip<24.1'
pip install 'omegaconf==2.0.6'
pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2

# Install audio processing dependencies
pip install torchaudio==2.0.2 librosa soundfile joblib amfm_decompy

# Upgrade pip back if desired
pip install --upgrade pip
```

**Note on fairseq**: Version 0.12.2 is required for HuBERT support. Earlier versions lack the `hubert_pretraining` task.

## Step 5: Test the Wrapper

### Python API Test

Create a test script `test_wrapper.py`:

```python
from pathlib import Path
from dpvc import ControlVCWrapper
import torchaudio

# Initialize wrapper
wrapper = ControlVCWrapper(
    repo_root=Path("/Users/steve/repos/control-vc"),
    device="cpu",  # or "cuda" if available
    verbose=True
)

# Test embedding extraction
print("Testing embedding extraction...")
embedding = wrapper.extract_embedding(Path("examples/trump_0.wav"))
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")

# Test voice conversion
print("\nTesting voice conversion...")
output = wrapper.infer(
    source_wav=Path("examples/trump_0.wav"),
    target_embedding=embedding,
    out_sr=16000
)
print(f"Output shape: {output.shape}")

# Save result
torchaudio.save("test_output.wav", output.cpu(), 16000)
print("Saved to test_output.wav")
```

Run the test:
```bash
python test_wrapper.py
```

### Command-Line Test

```bash
python examples/controlvc_infer.py \
    --repo-root /Users/steve/repos/control-vc \
    --source examples/trump_0.wav \
    --reference examples/trump_0.wav \
    --out test_output_cli.wav \
    --device cpu \
    --verbose
```

## Step 5: Use with Differential Privacy

```python
from dpvc import ControlVCWrapper, Anonymizer

# Initialize wrapper
vc_wrapper = ControlVCWrapper(
    repo_root="/Users/steve/repos/control-vc",
    device="cuda"
)

# Create anonymizer (uses VAE for DP noise)
anonymizer = Anonymizer(vc_wrapper)

# Anonymize with differential privacy
anonymizer.anonymize(
    source_file="input.wav",
    output_file="anonymized_output.wav",
    noise_level=1.0  # Adjust DP noise level
)
```

## Troubleshooting

### Issue: "Failed to import ControlVC modules"

**Cause**: The wrapper can't find the Control-VC repository modules.

**Solution**:
- Verify the `repo_root` path points to the correct directory
- Ensure the repository contains: `models.py`, `dataset.py`, `utils.py`, etc.

### Issue: "Speaker embedding model not loaded"

**Cause**: Missing `3000000-BL.ckpt` checkpoint.

**Solution**: Download the speaker embedding checkpoint and place it in `checkpoints/`

### Issue: "Using dummy content codes" warning

**Cause**: HuBERT checkpoint not found.

**Solution**:
- Download `hubert_base_ls960.pt` and `km.bin`
- Place them in the `checkpoints/` directory
- **Note**: The wrapper will still work but with reduced quality

### Issue: CUDA Out of Memory

**Solutions**:
- Switch to CPU: `device="cpu"`
- Process shorter audio clips
- Use a GPU with more memory

### Issue: Import error on `fairseq_feature_reader`

**Cause**: Missing `fairseq_feature_reader.py` in control-vc repo.

**Solution**:
- Ensure you have the complete ControlVC repository
- The wrapper will fall back to dummy codes if this file is missing

## File Locations

### In control-vc repository (`~/repos/control-vc/`)
- `models.py` - Model definitions
- `dataset.py` - Data loading and preprocessing
- `utils.py` - Utility functions
- `fairseq_feature_reader.py` - HuBERT feature extraction
- `checkpoints/` - All model checkpoints

### In dp-vc repository (`~/UVM-plaid/dp-vc/`)
- `dpvc/controlvc.py` - ControlVC wrapper implementation
- `dpvc/anonymizer.py` - DP anonymization pipeline
- `examples/controlvc_infer.py` - Example CLI script
- `docs/controlvc_wrapper.md` - Full documentation

## Performance Tips

1. **Use GPU**: Set `device="cuda"` for 5-10x speedup
2. **Batch Processing**: Use the wrapper in a loop for multiple files
3. **Checkpoint Caching**: The wrapper loads models once during initialization
4. **Audio Format**: Use 16kHz mono WAV files for best performance

## Example Workflow

```python
from pathlib import Path
from dpvc import ControlVCWrapper
import torchaudio

# Initialize once
wrapper = ControlVCWrapper(
    repo_root="/Users/steve/repos/control-vc",
    device="cuda",
    verbose=True
)

# Extract reference embeddings (do this once per target speaker)
target_speakers = {
    "speaker_a": wrapper.extract_embedding(Path("refs/speaker_a.wav")),
    "speaker_b": wrapper.extract_embedding(Path("refs/speaker_b.wav")),
}

# Convert multiple files
source_files = Path("inputs").glob("*.wav")
for source in source_files:
    for speaker_name, embedding in target_speakers.items():
        output = wrapper.infer(source, embedding)
        output_path = f"outputs/{source.stem}_{speaker_name}.wav"
        torchaudio.save(output_path, output.cpu(), 16000)
        print(f"Saved: {output_path}")
```

## Next Steps

- Read the full documentation: [docs/controlvc_wrapper.md](docs/controlvc_wrapper.md)
- Explore the example: [examples/controlvc_infer.py](examples/controlvc_infer.py)
- Integrate with your DP pipeline using the `Anonymizer` class

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the full documentation
3. Open an issue in the dp-vc repository
