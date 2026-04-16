# Controllable DP Voice Conversion — OpenVoice Pipeline

End-to-end guide for reproducing the controllable voice anonymization results
using OpenVoice as the VC backend.

## Quick Start

If you just want to run inference with a pre-trained VAE:

```bash
python examples/openvoice_infer_controllable.py \
    --source examples/trump_0.wav \
    --out output/happy.wav \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --style happy
```

Generate all 9 styles at once:

```bash
python examples/openvoice_infer_controllable.py \
    --source examples/trump_0.wav \
    --out output/ \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --all-styles
```

## Full Pipeline

### 0. Environment Setup

```bash
# Create venv (Python 3.9-3.12)
python -m venv .venv
source .venv/bin/activate

# Install dpvc with OpenVoice support
pip install -e ".[openvoice]"

# Additional dependencies for dataset extraction
pip install datasets pandas
```

OpenVoice model checkpoints download automatically on first use to
`~/.cache/openvoice_checkpoint/`.

### 1. Extract Expresso Embeddings

Expresso provides 11 styles from 4 speakers. We use 7 styles that map to
our unified label scheme.

```bash
python examples/openvoice_extract_expresso.py \
    --output embeddings/openvoice_expresso_emb.pt
```

This downloads from HuggingFace (~8GB). If you've already cached the dataset,
use `--parquet-dir` to skip re-downloading:

```bash
python examples/openvoice_extract_expresso.py \
    --output embeddings/openvoice_expresso_emb.pt \
    --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/*/read
```

Output: `embeddings/openvoice_expresso_emb.pt` (~8700 embeddings with style labels)

### 2. Extract CREMA-D Embeddings

CREMA-D provides 6 emotions from 91 speakers. The speaker diversity is
critical — without it, the VAE memorizes speaker identity instead of learning
style (see FINDINGS.md, Finding 2).

```bash
python examples/openvoice_extract_cremad.py \
    --output embeddings/openvoice_cremad_emb.pt
```

By default, extracts one sample per speaker per emotion (546 samples) to
maximize diversity. Use `--all-samples` for the full set.

Output: `embeddings/openvoice_cremad_emb.pt` (~546 embeddings with emotion labels)

### 3. Combine Datasets

Merges CREMA-D and Expresso into a unified label scheme with 9 styles:
anger, confused, disgust, enunciated, fear, happy, neutral, sad, whisper.

```bash
python examples/openvoice_combine_datasets.py \
    --cremad embeddings/openvoice_cremad_emb.pt \
    --expresso embeddings/openvoice_expresso_emb.pt \
    --output embeddings/openvoice_combined_emb.pt \
    --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/*/read
```

The `--parquet-dir` is needed to look up Expresso speaker IDs for balanced
sampling. Styles unique to Expresso (confused, enunciated, whisper) are capped
at ~90 samples to balance with CREMA-D's ~91 per emotion.

Output: `embeddings/openvoice_combined_emb.pt` (~825 samples, ~90 per style)

### 4. Train the Controllable VAE

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined.pt
```

Defaults: 3000 epochs, 15 latent dims (9 style + 6 free), lr=1e-6. Training
takes ~1-2 minutes on CPU.

Watch the loss output — label loss should drop below ~35 by the end. If it
plateaus above 50, try increasing epochs or lowering lr.

Output: `embeddings/openvoice_vae_combined.pt`

### 5. Run Controllable Inference

Single style:

```bash
python examples/openvoice_infer_controllable.py \
    --source examples/trump_0.wav \
    --out output/whisper.wav \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --style whisper
```

All styles with DP noise:

```bash
python examples/openvoice_infer_controllable.py \
    --source examples/trump_0.wav \
    --out output/ \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --all-styles \
    --noise-level 0.1
```

Key parameters:
- `--style-strength` (default 5.0): How hard to push the style. Higher = more
  pronounced but risks output collapse on some speakers. Try 3.0 for safer results.
- `--noise-level` (default 0.0): DP noise. 0.1 = light privacy with good style
  preservation. 0.5+ degrades style and can make baseline unintelligible.
- `--seed` (default 42): For reproducible outputs. Use -1 for random.

## Style Reference

| Dim | Style | Acoustic signature |
|-----|-------|-------------------|
| 0 | anger | Bright, sharp, edgy |
| 1 | confused | Hesitant, narrow F0 range |
| 2 | disgust | Low, withdrawn |
| 3 | enunciated | Crisp, bright articulation |
| 4 | fear | Tense, highly variable F0 |
| 5 | happy | Most animated, 2x F0 variation |
| 6 | neutral | Flat, subdued, dimmest |
| 7 | sad | Darker, more varied |
| 8 | whisper | Near-zero F0, breathy/airy |

## File Inventory

### Scripts (run in order)
| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `openvoice_extract_expresso.py` | HuggingFace / parquet | `openvoice_expresso_emb.pt` |
| 2 | `openvoice_extract_cremad.py` | HuggingFace | `openvoice_cremad_emb.pt` |
| 3 | `openvoice_combine_datasets.py` | Step 1 + 2 outputs | `openvoice_combined_emb.pt` |
| 4 | `openvoice_train_vae_combined.py` | Step 3 output | `openvoice_vae_combined.pt` |
| 5 | `openvoice_infer_controllable.py` | Step 4 output + audio | `.wav` files |

### Other files
- `openvoice_train_vae.py` — Trains basic (non-controllable) VAE. Not needed for style control.
- `openvoice_inference.py` — Basic DP inference without style control.
- `openvoice_extract_commonvoice.py` — CommonVoice extraction (for future pre-training work).
- `source_speakers/` — CREMA-D audio clips used for diverse speaker evaluation.
- `trump_0.wav` — Default test source audio.

### ControlVC equivalents (require separate control-vc repo)
The `controlvc_*` scripts are the ControlVC equivalents. These require cloning
the control-vc repo and passing `--repo-root`. Note: ControlVC's D_VECTOR
embedding does not encode style (see FINDINGS.md, Finding 1), so style
control does not work with ControlVC.

## Troubleshooting

**"Audio too short, fail to add watermark"**: Source audio is under ~3 seconds.
OpenVoice needs at least 3-4 seconds. Concatenate clips or use longer audio.

**Output sounds identical to baseline**: Style control strength may be too low.
Try `--style-strength 5.0` or higher.

**Output is unintelligible / silence**: The speaker-style combination pushed
the embedding into a degenerate region. Try lower `--style-strength` (3.0) or
a different source speaker. This affects ~9% of speaker-style combinations.

**"size mismatch" loading VAE**: The VAE checkpoint was trained with different
`--latent-dims` than what you're loading with. The combined VAE uses 15 dims.
Pass `--latent-dims 15`.

**fairseq errors on Apple Silicon**: fairseq may fail on MPS. Force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
```

**HuggingFace rate limiting**: Set `HF_TOKEN` for faster downloads:
```bash
export HF_TOKEN=hf_xxxxx
```
