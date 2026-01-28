# ControlVC Wrapper - Quick Reference

## Installation

```bash
# Clone control-vc
git clone https://github.com/meiyingchen/ControlVC.git ~/repos/control-vc

# Download checkpoints to ~/repos/control-vc/checkpoints/
# From: https://drive.google.com/drive/folders/1APVHQFIb1871UhvymdK_oewWKJWrInYK

# Install dp-vc
cd ~/UVM-plaid/dp-vc
pip install -e .
```

## Basic Usage

```python
from pathlib import Path
from dpvc import ControlVCWrapper
import torchaudio

# Initialize
wrapper = ControlVCWrapper(
    repo_root="/Users/steve/repos/control-vc",
    device="cuda"  # or "cpu"
)

# Extract embedding
target_emb = wrapper.extract_embedding(Path("target.wav"))

# Convert voice
output = wrapper.infer(
    source_wav=Path("source.wav"),
    target_embedding=target_emb
)

# Save
torchaudio.save("output.wav", output.cpu(), 16000)
```

## With Differential Privacy

```python
from dpvc import ControlVCWrapper, Anonymizer

wrapper = ControlVCWrapper(repo_root="...")
anonymizer = Anonymizer(wrapper)

anonymizer.anonymize(
    "source.wav",
    "anonymized.wav",
    noise_level=1.0
)
```

## Command Line

```bash
python examples/controlvc_infer.py \
    --repo-root ~/repos/control-vc \
    --source input.wav \
    --reference target.wav \
    --out output.wav \
    --device cuda \
    --verbose
```

## Pitch Shifting

```python
output = wrapper.infer(
    source_wav=Path("source.wav"),
    target_embedding=embedding,
    pitch_shift=1.2  # 20% higher
)
```

## Batch Processing

```python
# Extract embeddings once
speakers = {
    "alice": wrapper.extract_embedding("refs/alice.wav"),
    "bob": wrapper.extract_embedding("refs/bob.wav")
}

# Convert multiple files
for source in Path("inputs").glob("*.wav"):
    for name, emb in speakers.items():
        output = wrapper.infer(source, emb)
        torchaudio.save(f"outputs/{source.stem}_{name}.wav",
                       output.cpu(), 16000)
```

## Checkpoint Structure

```
control-vc/checkpoints/
├── embed_f0stat2/
│   ├── config.json          # Required
│   └── g_00400000           # Required
├── 3000000-BL.ckpt          # Required
├── hubert_base_ls960.pt     # Optional (recommended)
└── km.bin                   # Optional (recommended)
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "Failed to import ControlVC modules" | Check `repo_root` path |
| "Speaker embedding model not loaded" | Download `3000000-BL.ckpt` |
| "Using dummy content codes" | Download HuBERT checkpoints |
| CUDA OOM | Use `device="cpu"` |

## API Reference

### ControlVCWrapper.__init__()

```python
ControlVCWrapper(
    repo_root: Path,              # Path to control-vc repo
    checkpoints_dir: Path = None, # Optional: custom checkpoint path
    device: str = "cpu",          # "cpu", "cuda", "cuda:0"
    verbose: bool = False         # Enable debug output
)
```

### extract_embedding()

```python
embedding = wrapper.extract_embedding(
    wav_path: Path,              # Audio file path
    num_utterances: int = 1      # Currently only supports 1
)
# Returns: torch.Tensor of shape (256, 1)
```

### infer()

```python
output = wrapper.infer(
    source_wav: Path,            # Source audio path
    target_embedding: Tensor,    # Speaker embedding (256, 1)
    out_sr: int = 16000,        # Output sample rate
    pitch_shift: float = 1.0    # Pitch multiplier
)
# Returns: torch.Tensor of shape (1, T)
```

## Performance

| Hardware | Embedding | Conversion (1s) |
|----------|-----------|-----------------|
| CPU (i7) | ~0.5s | ~2-3s |
| GPU (V100) | ~0.1s | ~0.3s |
| GPU (3090) | ~0.08s | ~0.2s |

## Documentation

- Full Guide: `docs/controlvc_wrapper.md`
- Setup: `CONTROLVC_SETUP.md`
- Implementation: `WRAPPER_IMPLEMENTATION_SUMMARY.md`

## Example Output

```
[ControlVC] Initializing ControlVC wrapper with device: cuda
[ControlVC] Loaded generator: g_00400000
[ControlVC] Loaded speaker model: 3000000-BL.ckpt
[ControlVC] All models loaded successfully
[ControlVC] Extracting embedding from target.wav
[ControlVC] Converting source.wav
```
