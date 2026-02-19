# ControlVC Wrapper Setup Guide

## Quick Setup (Recommended)

Run the provided setup script — it handles everything automatically:

```bash
bash setup.sh
```

Then activate the environment and verify:

```bash
source .venv310/bin/activate
python test_wrapper.py
```

Expected output ends with `All tests passed! ✓`. Listen to `test_output.wav` — it should be clear, intelligible speech with a different voice from the original speaker.

**Custom control-vc path** (if you already have the repo elsewhere):
```bash
bash setup.sh --control-vc-dir /path/to/control-vc
```

**`CONTROL_VC_DIR` env var** is also respected by `test_wrapper.py`:
```bash
CONTROL_VC_DIR=/path/to/control-vc python test_wrapper.py
```

---

## Manual Setup

Use this if `setup.sh` fails at a specific step, or if you need more control.

### Prerequisites

- **Python 3.10** (required — fairseq 0.12.2 is incompatible with Python 3.11+)
- PyTorch (CPU is fine; CUDA optional for GPU acceleration)

If you use pyenv, the project includes a `.python-version` file that selects 3.10 automatically.

### Step 1: Clone Control-VC Repository

Use the `zuruoke` fork, which removes pinned library versions to allow newer torchaudio/librosa:

```bash
git clone https://github.com/zuruoke/control-vc.git ~/repos/control-vc
```

### Step 2: Download Checkpoints

Download all required checkpoints from the ControlVC Google Drive:
https://drive.google.com/drive/folders/1APVHQFIb1871UhvymdK_oewWKJWrInYK

Place them in `~/repos/control-vc/checkpoints/`:

```
checkpoints/
├── embed_f0stat2/
│   ├── config.json
│   └── g_00400000        # (or g_00350000)
├── 3000000-BL.ckpt       # Required: speaker embedding model
├── hubert_base_ls960.pt  # Strongly recommended (~1.1 GB)
├── km.bin                # Strongly recommended
└── vctk_f0_vq/
    └── g_00400000
```

**Without `hubert_base_ls960.pt` and `km.bin`**, the wrapper falls back to dummy content codes and produces garbled audio.

### Step 3: Apply Fix for Apple Silicon (MPS)

If you are on an M1/M2/M3 Mac, edit `~/repos/control-vc/fairseq_feature_reader.py` lines 22–26:

**Before:**
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
```

**After:**
```python
# Force CPU - MPS does not support all fairseq operators
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Step 4: Install Dependencies

```bash
cd ~/UVM-plaid/dp-vc

# Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv310
source .venv310/bin/activate

# Install dp-vc package in editable mode
pip install -e .

# fairseq requires an older pip to handle omegaconf metadata
pip install 'pip<24.1'
pip install 'omegaconf==2.0.6'
pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2

# Audio processing dependencies
pip install torchaudio librosa soundfile joblib amfm_decompy matplotlib gdown
```

**Note on fairseq**: version 0.12.2 is required for HuBERT support. It requires Python 3.10 — Python 3.11+ changed dataclass validation in a way that breaks fairseq's internal configs.

### Step 4b: Patch fairseq for PyTorch 2.6+

PyTorch 2.6 changed `torch.load` to use `weights_only=True` by default, which blocks loading the HuBERT checkpoint (it contains `argparse.Namespace`). Apply this one-line fix after installation:

```bash
python - <<'EOF'
import site, pathlib
for d in site.getsitepackages():
    f = pathlib.Path(d) / "fairseq/checkpoint_utils.py"
    if f.exists():
        txt = f.read_text()
        old = "torch.load(f, map_location=torch.device(\"cpu\"))"
        new = "torch.load(f, map_location=torch.device(\"cpu\"), weights_only=False)"
        if old in txt:
            f.write_text(txt.replace(old, new))
            print(f"Patched {f}")
        else:
            print(f"Already patched or not found in {f}")
EOF
```

### Step 5: Verify the Setup

```bash
python test_wrapper.py
```

Expected output:
```
==================================================
Testing ControlVC Wrapper
==================================================
1. Initializing wrapper...
[ControlVC] Loaded generator: g_00350000.pth
[ControlVC] Loaded speaker model: 3000000-BL.ckpt
✓ Initialization successful!

2. Testing embedding extraction with examples/trump_0.wav...
✓ Embedding extracted: shape=torch.Size([256, 1])

3. Testing voice conversion...
[ControlVC] Loaded HuBERT and K-means models
✓ Conversion successful: shape=torch.Size([1, 124480])
✓ Saved output to test_output.wav

==================================================
All tests passed! ✓
==================================================
```

Listen to `test_output.wav` — it should be clear, intelligible speech. Garbled audio means HuBERT/k-means is not loading correctly.

---

## Use with Differential Privacy

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
    source_file="examples/trump_0.wav",
    output_file="output_anonymized.wav",
    noise_level=1.0   # higher = more privacy, less speaker diversity
)
```

**Note on `noise_level`**: At high values (e.g., 2.0), the DP noise dominates the 6-dimensional latent space and all outputs converge to a similar-sounding anonymous voice. Lower values (0.5–1.0) preserve more speaker-to-speaker variation while still providing privacy.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Using dummy content codes" / garbled audio | HuBERT or km.bin not found | Verify checkpoints; check `import fairseq` works |
| `omegaconf has invalid metadata` | pip >= 24.1 | `pip install 'pip<24.1'` before fairseq |
| MPS device error on Apple Silicon | fairseq_feature_reader.py uses MPS | Apply Step 3 fix (or re-run `setup.sh`) |
| Output sounds the same every run | `noise_level` too high | Try `noise_level=0.5` |
| `gdown` fails during setup | Google Drive quota exceeded | Download checkpoints manually (Step 2) |

## Command-Line Usage

```bash
python examples/controlvc_infer.py \
    --repo-root ~/repos/control-vc \
    --source examples/trump_0.wav \
    --reference examples/trump_0.wav \
    --out output.wav \
    --device cpu \
    --verbose
```
