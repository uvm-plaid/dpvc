# ControlVC Wrapper - Meeting Preparation for Joe

**Date**: 2026-01-29
**Attendees**: Steve, Joe
**Branch**: `feat/controlvc`

## Meeting Objective

Test and verify the ControlVC wrapper implementation for the differential privacy voice anonymization pipeline. Ensure the setup is reproducible across different machines.

---

## What Was Implemented

We've completed a full integration of ControlVC into the dp-vc repository:

1. **Complete rewrite of ControlVCWrapper** (`dpvc/controlvc.py`)
   - Direct PyTorch model loading (no subprocess execution)
   - Two-stage API: `extract_embedding()` and `infer()`
   - Proper HuBERT content extraction with fairseq v0.12.2
   - Automatic device management (CPU/CUDA)

2. **Working voice conversion pipeline**
   - HuBERT + K-means for content codes (72 discrete codes)
   - D_VECTOR for speaker embeddings (256-dimensional)
   - YAAPT for F0 (pitch) extraction
   - HiFi-GAN vocoder for waveform generation

3. **Documentation**
   - `CONTROLVC_SETUP.md` - Step-by-step setup guide
   - `QUICK_REFERENCE.md` - Quick API reference
   - `docs/controlvc_wrapper.md` - Full technical documentation
   - `WRAPPER_IMPLEMENTATION_SUMMARY.md` - Implementation details

---

## Pre-Meeting Setup (Do This Before Meeting)

### 1. Clone Control-VC Repository

```bash
cd ~/repos  # or your preferred location
git clone https://github.com/MelissaChen15/control-vc.git control-vc
cd control-vc
```

### 2. Download Checkpoints

**Critical**: Download all checkpoints from:
https://drive.google.com/drive/folders/1APVHQFIb1871UhvymdK_oewWKJWrInYK

Place them in `control-vc/checkpoints/` with this structure:

```
control-vc/checkpoints/
├── embed_f0stat2/
│   ├── config.json
│   └── g_00350000.pth (or g_00400000)
├── 3000000-BL.ckpt
├── hubert_base_ls960.pt    # Critical for quality!
├── km.bin                   # Critical for quality!
└── vctk_f0_vq/
    └── g_00400000
```

**Note**: Without `hubert_base_ls960.pt` and `km.bin`, the wrapper will fall back to dummy codes and produce garbled audio.

### 3. Apply Critical Fix (Apple Silicon Only)

**If you're on an M1/M2/M3 Mac**, edit `control-vc/fairseq_feature_reader.py` line 22-26:

**Change from:**
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
```

**Change to:**
```python
# Force CPU for compatibility - MPS doesn't support all fairseq operators
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 4. Install Dependencies

```bash
# Switch to dp-vc repository
cd ~/UVM-plaid/dp-vc

# Check out the feature branch
git fetch origin
git checkout feat/controlvc
git pull origin feat/controlvc

# Install dp-vc package
pip install -e .

# Install fairseq v0.12.2 (required for HuBERT)
# Note: Requires older pip version to handle omegaconf metadata
pip install 'pip<24.1'
pip install 'omegaconf==2.0.6'
pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2

# Install audio processing dependencies
pip install torchaudio==2.0.2 librosa soundfile joblib amfm_decompy

# Optional: Upgrade pip back
pip install --upgrade pip
```

**Expected install time**: 5-10 minutes (fairseq takes the longest)

### 5. Verify Setup

```bash
cd ~/UVM-plaid/dp-vc

# Run the test script
python test_wrapper.py
```

**Expected output**:
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

**Listen to `test_output.wav`** - it should be clear, intelligible speech.

---

## What We'll Test During the Meeting

### Test 1: Basic Wrapper Functionality

Verify that the wrapper loads models correctly and performs voice conversion.

```bash
cd ~/UVM-plaid/dp-vc
python test_wrapper.py
```

**Success criteria:**
- ✅ No errors during initialization
- ✅ Models load successfully
- ✅ HuBERT and K-means models load (not dummy codes)
- ✅ Output audio is intelligible

### Test 2: Command-Line Interface

Test the example CLI script for voice conversion.

```bash
python examples/controlvc_infer.py \
    --repo-root ~/repos/control-vc \
    --source examples/trump_0.wav \
    --reference examples/trump_0.wav \
    --out test_cli_output.wav \
    --device cpu \
    --verbose
```

**Success criteria:**
- ✅ Script runs without errors
- ✅ Output file is created
- ✅ Audio quality is good

### Test 3: Python API Integration

Test programmatic usage in Python.

```python
from pathlib import Path
from dpvc import ControlVCWrapper

# Initialize wrapper
wrapper = ControlVCWrapper(
    repo_root=Path("~/repos/control-vc").expanduser(),
    device="cpu",
    verbose=True
)

# Extract speaker embedding
target_emb = wrapper.extract_embedding(Path("examples/trump_0.wav"))
print(f"Embedding shape: {target_emb.shape}")  # Should be (256, 1)

# Perform voice conversion
output = wrapper.infer(
    source_wav=Path("examples/trump_0.wav"),
    target_embedding=target_emb
)
print(f"Output shape: {output.shape}")  # Should be (1, T)

# Save result
import soundfile as sf
sf.write("test_api_output.wav", output.cpu().squeeze().numpy(), 16000)
```

**Success criteria:**
- ✅ Embedding extraction works
- ✅ Voice conversion produces audio
- ✅ Output can be saved to file

### Test 4: Content Code Verification

Verify that real HuBERT codes are being extracted (not dummy zeros).

```python
from pathlib import Path
from dpvc import ControlVCWrapper
import librosa
import numpy as np

wrapper = ControlVCWrapper(
    repo_root=Path("~/repos/control-vc").expanduser(),
    device="cpu"
)

# Load audio
audio, sr = librosa.load("examples/trump_0.wav", sr=16000, mono=True)
audio = librosa.util.normalize(audio) * 0.95

# Extract content codes
codes = wrapper._extract_content_codes(audio, sr)

print(f"Content codes shape: {codes.shape}")
print(f"Unique codes: {len(np.unique(codes))}")  # Should be ~50-80
print(f"First 20 codes: {codes[:20]}")
print(f"All zeros?: {np.all(codes == 0)}")  # Should be False
```

**Success criteria:**
- ✅ 50-80 unique code values (not all zeros)
- ✅ Codes vary across the sequence
- ✅ No warnings about "Using dummy content codes"

---

## Troubleshooting Guide

### Issue: "HuBERT extraction failed" warning

**Symptoms:**
- Warning: "Using dummy content codes"
- Audio output is garbled/noisy

**Cause:** HuBERT checkpoint or K-means model not found, or fairseq not installed correctly.

**Solution:**
1. Verify checkpoints exist:
   ```bash
   ls ~/repos/control-vc/checkpoints/hubert_base_ls960.pt
   ls ~/repos/control-vc/checkpoints/km.bin
   ```
2. Test fairseq import:
   ```bash
   python -c "import fairseq; print(fairseq.__version__)"
   ```
   Should output: `0.12.2`
3. If fairseq not installed, reinstall following Step 4 above

### Issue: "Buffer dtype mismatch" error

**Symptoms:**
- Error: `Buffer dtype mismatch, expected 'const double' but got 'float'`

**Cause:** Fixed in the latest code, but if you see this, ensure you have the latest version.

**Solution:**
```bash
cd ~/UVM-plaid/dp-vc
git pull origin feat/controlvc
```

The fix is in `dpvc/controlvc.py` line 419: `codes = self._kmeans.predict(feats.cpu().numpy())`

### Issue: MPS device error (Apple Silicon)

**Symptoms:**
- Error: `The operator 'aten::_weight_norm_interface' is not currently implemented for the MPS device`

**Cause:** fairseq_feature_reader.py tries to use MPS, which doesn't support all operations.

**Solution:** Apply the fix in Step 3 above to force CPU for HuBERT.

### Issue: Fairseq installation fails

**Symptoms:**
- `error: metadata-generation-failed`
- `omegaconf has invalid metadata`

**Solution:**
```bash
# Use older pip version
pip install 'pip<24.1'
pip install 'omegaconf==2.0.6'
pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2
```

### Issue: Audio still garbled despite "HuBERT loaded" message

**Symptoms:**
- Message says "Loaded HuBERT and K-means models"
- But audio is still garbled

**Cause:** The wrapper attempted HuBERT extraction but fell back to dummy codes due to a hidden error.

**Solution:** Run the content code verification test (Test 4 above) to check if codes are actually non-zero.

---

## Expected Performance

### Timing (on CPU)
- Model initialization: ~3-5 seconds
- Embedding extraction: ~0.5-1 second
- Voice conversion (1 second of audio): ~2-3 seconds
- HuBERT feature extraction: ~1-2 seconds

### Audio Quality
- **With HuBERT**: Clear, intelligible speech with natural voice quality
- **Without HuBERT (dummy codes)**: Garbled noise (unusable)

### Key Success Indicator
The most important test: **Can you understand the words in the output audio?**
- ✅ YES = HuBERT is working correctly
- ❌ NO = HuBERT extraction is failing (see troubleshooting)

---

## Questions We'll Address During Meeting

1. **Reproducibility**: Did the setup work smoothly on Joe's machine?
2. **Audio quality**: Does the output sound good with real HuBERT codes?
3. **Integration**: How should this integrate with the differential privacy pipeline?
4. **Next steps**: What additional features or tests are needed?

---

## Repository Structure

```
dp-vc/
├── dpvc/
│   ├── __init__.py          # Package initialization
│   ├── controlvc.py         # Main wrapper implementation
│   └── anonymizer.py        # DP anonymization (existing)
├── examples/
│   ├── controlvc_infer.py   # CLI example script
│   └── trump_0.wav          # Test audio file
├── docs/
│   └── controlvc_wrapper.md # Full technical documentation
├── CONTROLVC_SETUP.md       # Setup guide (this is the main one)
├── QUICK_REFERENCE.md       # Quick API reference
├── WRAPPER_IMPLEMENTATION_SUMMARY.md  # Implementation details
├── MEETING_PREP_JOE.md      # This document
└── test_wrapper.py          # Test script
```

---

## Post-Meeting Action Items

- [ ] Document any platform-specific issues encountered
- [ ] Update troubleshooting guide with new issues
- [ ] Merge `feat/controlvc` branch to main (if tests pass)
- [ ] Integrate with differential privacy anonymization pipeline
- [ ] Set up automated testing (CI/CD)

---

## Additional Resources

- ControlVC Paper: https://arxiv.org/abs/2305.18145
- ControlVC GitHub: https://github.com/meiyingchen/ControlVC
- HuBERT Paper: https://arxiv.org/abs/2106.07447
- Full wrapper documentation: `docs/controlvc_wrapper.md`

---

**Contact**: If you encounter any issues before the meeting, ping Steve on Teams or email.
