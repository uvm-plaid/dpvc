# ControlVC Baseline - Quick Reference

ControlVC is kept in this repo as a **DP baseline and wrapper reference**.
For controllable style work, use the OpenVoice pipeline in
`examples/README.md`.

## Installation

```bash
bash scripts/setup.sh
source .venv310/bin/activate
```

See `docs/controlvc_setup.md` for full details.

## Basic Usage

```python
from pathlib import Path
from dpvc import ControlVCWrapper

# Initialize
wrapper = ControlVCWrapper(
    repo_root=Path("~/repos/control-vc").expanduser(),
    device="cpu"  # or "cuda"
)

# Extract embedding
embedding = wrapper.extract_embedding(Path("target.wav"))
# Returns: torch.Tensor of shape (1, 256)

# Convert voice (writes directly to output.wav)
wrapper.inference(
    source_file=Path("source.wav"),
    output_file=Path("output.wav"),
    source_embedding=embedding,
    target_embedding=embedding
)
```

## With Differential Privacy

```python
from pathlib import Path
from dpvc import ControlVCWrapper, Anonymizer

wrapper = ControlVCWrapper(
    repo_root=Path("~/repos/control-vc").expanduser(),
    device="cpu"
)
vae_config = wrapper.get_vae_config()
vae_config["checkpoint_path"] = "examples/controlvc_vae.pt"
anonymizer = Anonymizer(wrapper, vae_config=vae_config)

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
    --device cpu \
    --verbose
```

## Checkpoint Structure

```
control-vc/checkpoints/
├── embed_f0stat2/
│   ├── config.json          # Required
│   └── g_00400000           # Required
├── 3000000-BL.ckpt          # Required
├── hubert_base_ls960.pt     # Strongly recommended (~1.1 GB)
└── km.bin                   # Strongly recommended
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "Failed to import ControlVC modules" | Check `repo_root` path |
| "Speaker embedding model not loaded" | Download `3000000-BL.ckpt` |
| "Using dummy content codes" / garbled audio | Download HuBERT checkpoints |
| CUDA OOM | Use `device="cpu"` |
| Output sounds the same every run | Lower `noise_level` to 0.5–1.0 |

## API Reference

### ControlVCWrapper.__init__()

```python
ControlVCWrapper(
    repo_root: Path,       # Path to control-vc repo
    device: str = "cpu",   # "cpu", "cuda", "cuda:0"
    verbose: bool = False  # Enable debug output
)
```

### extract_embedding()

```python
embedding = wrapper.extract_embedding(
    wav_path: Path,          # Audio file path
    num_utterances: int = 1
)
# Returns: torch.Tensor of shape (1, 256)
```

### inference()

```python
wrapper.inference(
    source_file,        # Source audio path
    output_file,        # Output WAV path (written at 16 kHz)
    source_embedding,   # Source speaker embedding
    target_embedding    # Target speaker embedding — (256,), (1,256), or (256,1)
)
```

## Documentation

- Full Guide: `docs/controlvc_wrapper.md`
- Setup: `docs/controlvc_setup.md`
