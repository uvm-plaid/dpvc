# Controllable DP Voice Conversion — Work Log

**Last updated:** 2026-04-16
**Branches:** `feat/controlvc`, `feat/openvoice-expresso`, `feat/f0-style-control`, `feat/cremad-experiments`
**Author:** Stephen Oladele (with Claude, and Joe Near's upstream work)

---

## 0. Roadmap

### Phase 1: Core Controllability (prove it works)
- [x] End-to-end pipeline: extract → train VAE → infer → intelligible speech
- [x] Demonstrate style control with OpenVoice (not ControlVC — Joe confirmed, April 9)
- [x] Speaker diversity: train on 91+ speakers (CREMA-D) so VAE learns style, not speaker
- [x] Combined dataset (CREMA-D + Expresso): 9 styles, 94 speakers, all perceptually distinct
- [x] Acoustic analysis confirming style differences (F0, energy, spectral centroid)
- [x] Test style control survives DP noise (noise_level 0–1.0, styles persist at 0.1, degrade gracefully)
- [x] Test on diverse source speakers (5 speakers: brightness consistent 7/9 styles, F0 inconsistent, 9% collapse rate)
- [x] Per-speaker style evaluation: brightness is the reliable cross-speaker metric, not F0

### Phase 1.5: Scale Up Speaker Diversity (Joe's suggestion, April 16)
- [ ] Extract OpenVoice embeddings from CommonVoice (20k+ speakers, no style labels)
- [ ] Pre-train VAE on CommonVoice (reconstruction loss only)
- [ ] Finetune on CREMA-D + Expresso (style labels, label loss)
- [ ] Compare with current combined-only VAE: does pre-training reduce collapse rate?
- [ ] Add age/gender control dims using CommonVoice metadata (dims 9-10)
- [ ] Test orthogonality: does pushing emotion dims shift perceived age/gender?

### Phase 2: Paper-Ready Results
- [ ] Quantitative style evaluation: pretrained emotion classifier on outputs (precision/recall per style)
- [ ] Speaker verification experiments: does anonymization actually reduce re-identification accuracy?
- [ ] Privacy-utility curves: plot speaker verification vs. emotion classification at different noise levels
- [ ] Ablation study: CREMA-D only vs. Expresso only vs. combined (already have data for this)
- [ ] Compare with naive baseline: random noise without style control
- [ ] Investigate F0-based re-identification attack: can F0 alone re-identify speakers after embedding anonymization?
- [ ] Write up negative result: ControlVC D_VECTOR doesn't encode style (separability ratio 0.88)
- [ ] Address collapse issue: 9% of speaker-style combinations produce unintelligible output

### Phase 3: Reproducibility & Collaboration (for Joe)
- [ ] Clean up repo: document pipeline commands, parquet paths, environment setup
- [ ] Share Expresso download instructions with Joe
- [ ] Ensure Joe can run extraction + training + inference from scratch
- [ ] Pin dependencies (fairseq compat, OpenVoice install steps)

### Phase 4: Application / Demo
- [ ] Design interactive demo: upload voice → choose style → choose privacy level → download anonymized output
- [ ] Web UI or CLI tool that non-researchers can use
- [ ] Real-time voice conversion mode (stretch goal)
- [ ] Package as installable tool (pip install dpvc or similar)

### Key Insight from Joe (April 9 call)
> "If I want to invent a completely hypothetical speaker who has some properties, everybody's bad at that. And in some ways, that's what we're trying to do. If we can make this work, that's a big deal."

The paper contribution is **controllable speaker profile synthesis with formal privacy guarantees** — something the speech community has struggled with. Even without the privacy angle, demonstrating controllability over speaker profiles is itself a significant result.

---

## 1. Project Overview

**dpvc** is a Python library for **differentially private voice conversion** — it anonymizes a speaker's identity by passing their voice embedding through a VAE with calibrated DP noise, then reconstructs audio with a modified (anonymized) speaker embedding.

The new work on `feat/controlvc` extends this with **controllable anonymization**: instead of just adding noise, we can steer specific perceptual attributes (happy, sad, whisper, etc.) in the anonymized output by manipulating labeled latent dimensions of the VAE.

### Core Pipeline

```
Source Audio
    │
    ├─► HuBERT (content codes)  ─────────────────────────┐
    ├─► D_VECTOR (256-dim speaker embedding) ─► VAE ─► modified embedding
    ├─► YAAPT (F0 pitch contour)  ────────────────────────┤
    │                                                      │
    └──────────────────────────────────────────────────────┴─► CodeGenerator (vocoder) ─► Output Audio
```

**Key insight:** The VAE sits on the speaker embedding only. Content (HuBERT codes) and prosody (F0) pass through unchanged. So style control operates on *who it sounds like*, not *what they say*.

---

## 2. What Joe Near Built (Pre-April 2026)

Joe's work existed on two branches:

### `feat/controlvc` (merged ~Feb 2026)
- Added `ControlVCWrapper` in `dpvc/controlvc.py` (~500 lines) wrapping the [control-vc](https://github.com/auspicious3000/control-vc) system
- Integrated it alongside the existing `OpenVoiceWrapper`
- **Issues found:** hardcoded device `"cuda"` (broke on Mac), hardcoded paths to Joe's machine, fixed `INPUT_DIM=256` in VAE

### `controllable_vae` branch (never merged)
- Upgraded `model_embedding_vae.py` with a more sophisticated architecture: GELU activations, LayerNorm, proper reparameterization trick, `control_features` dict for overriding specific latent dims at inference
- Updated `utils.py` `train_autoencoder()` to support label-aware training (MSE loss on first K latent dims)
- Updated `anonymizer.py` to use a `vae_config` dict pattern
- All of this was OpenVoice-specific; needed porting to work with ControlVC

---

## 3. What We Built (April 8-9, 2026)

### 3.1 Portability Fixes (commit `139d0e5`)
- Changed ControlVC device default from `"cuda"` to auto-detect (`cuda` if available, else `cpu`)
- Added `get_vae_config()` to `ControlVCWrapper` and abstract `Wrapper` base class
- Fixed hardcoded paths in example scripts with argparse

### 3.2 Controllable VAE Port (commit `dfcbf9a`)
- Ported the full controllable VAE architecture from `origin/controllable_vae` into the main codebase
- Made it work with ControlVC (was OpenVoice-only)
- **Files replaced/updated:**
  - `dpvc/model_embedding_vae.py` — full rewrite with GELU/LayerNorm encoder, reparameterization trick, control_features support
  - `dpvc/anonymizer.py` — rewritten with vae_config dict pattern
  - `dpvc/utils.py` — `train_autoencoder()` now supports `labels` dict and label MSE loss
  - `dpvc/wrapper.py` — added `get_vae_config()` to abstract interface
  - `dpvc/openvoice.py` — added `get_vae_config()` for consistency

### 3.3 Expresso Pipeline (commit `dfcbf9a`)
Three new example scripts for end-to-end controllable training:

1. **`examples/controlvc_extract_expresso.py`** — Extracts ControlVC speaker embeddings from HuggingFace's `ylacombe/expresso` dataset
   - Handles 11 styles: default, confused, enunciated, happy, laughing, sad, whisper, emphasis, essentials, longform, singing
   - One-hot style encoding: active style = +1, all others = -1
   - Workarounds for: xet storage stalls (`HF_HUB_DISABLE_XET=1`), torchcodec incompatibility (uses `soundfile` instead), dataset download failures (supports `--parquet-dir` for local cache)

2. **`examples/controlvc_train_vae_expresso.py`** — Trains controllable VAE with style labels
   - Default: latent_dims=16 (11 style + 5 free), lr=1e-6
   - Label loss forces first 11 latent dims to match one-hot style encoding via MSE

3. **`examples/controlvc_infer_controllable.py`** — CLI for controllable voice anonymization
   - Maps style name to latent dim index
   - Sets all 11 style dims at inference (active = style_value, others = -1) to match training encoding

### 3.4 Fairseq/HuBERT Fix (April 9, uncommitted)
The HuBERT content encoder is critical — without it, the vocoder produces unintelligible noise. Fairseq 0.12.2 is incompatible with Python 3.11 due to mutable dataclass defaults.

**Fix applied in `dpvc/controlvc.py`:**
- Before importing fairseq, monkey-patches `dataclasses._get_field` to convert mutable defaults to `default_factory` calls
- Adds the control-vc repo to `sys.path` so `fairseq_feature_reader` is importable
- Also patched `fairseq/dataclass/configs.py` (manual `field(default_factory=...)` edits) and `fairseq/dataclass/initialize.py` (handle MISSING defaults) and `fairseq/checkpoint_utils.py` (`weights_only=False`) — these are in the venv, not version-controlled

**Result:** HuBERT now loads and produces real content codes, generating intelligible speech output.

### 3.6 F0 Prosody-Based Style Control (April 9, branch `feat/f0-style-control`)

After confirming that D_VECTOR embeddings don't carry style information (Section 5), pivoted to controlling style via F0 (pitch) manipulation.

**Changes:**
- `dpvc/controlvc.py` — Added `f0_transform` parameter to `inference()`. Supports three operations applied to the F0 contour before it hits the vocoder:
  - `pitch_shift`: multiply voiced F0 values (e.g., 1.25 = 25% higher pitch)
  - `range_scale`: expand/compress variation around mean (e.g., 0.4 = flatter, 2.0 = more expressive)
  - `flatten`: interpolate toward mean (0.0 = no change, 1.0 = completely monotone)
- `dpvc/anonymizer.py` — Pass-through `f0_transform` to wrapper
- `dpvc/wrapper.py` — Added `f0_transform` to abstract base class signature
- `examples/controlvc_infer_controllable.py` — Rewritten with F0 preset system and custom CLI args

**F0 Presets (tuned for audible distinction):**
| Style | pitch_shift | range_scale | flatten |
|-------|------------|-------------|---------|
| happy | 1.25 | 1.6 | — |
| sad | 0.80 | 0.4 | 0.3 |
| whisper | 0.90 | 0.2 | 0.8 |
| confused | 1.10 | 2.0 | — |
| laughing | 1.35 | 2.2 | — |
| enunciated | 1.00 | 1.8 | — |

**Result:** Audible style differences confirmed. Tested with trump_0.wav as source. Happy and laughing are clearly higher/more animated, sad is lower/flatter, whisper is near-monotone. The VC pipeline itself changes voice identity (expected — HuBERT quantization is lossy), but prosody differences are clearly distinguishable.

### 3.5 Inference Style Encoding Fix (April 9, uncommitted)
Original inference script only set 1 latent dim (the active style). But training used a full one-hot encoding (+1 active, -1 all others across all 11 dims). Fixed to set all 11 style dims at inference.

---

## 4. Training Runs

### Extraction
- **4,840 samples** extracted from Expresso dataset (5 of 6 shards — shard 5 consistently failed to download)
- 256-dim speaker embeddings + 11-dim one-hot style labels
- Saved to `embeddings/controlvc_expresso_emb.pt`
- Took ~9.5 minutes on CPU

### VAE Training

| Version | System | Epochs | LR   | Final Label Loss | Final Recon | Notes |
|---------|--------|--------|------|-----------------|-------------|-------|
| v1      | ControlVC | 2000   | 1e-6 | ~590            | 0.20        | LR too low — label dims barely learned |
| v2      | ControlVC | 2000   | 1e-4 | ~55             | 0.20        | 10x improvement, but still high |
| v3      | ControlVC | 5000   | 1e-4 | ~15             | 0.18        | Label loss improved but no audible style diff — even at style_value=5.0 |
| **v4**  | **OpenVoice** | **2000** | **1e-4** | **~1.5** | **~67** | **Label loss 10x lower than ControlVC v3! OpenVoice embeddings encode style.** |
| **v5**  | **OpenVoice+CREMA-D** | **2000** | **1e-4** | **~18** | **~200** | **91 speakers, 6 emotions. Higher label loss but all emotions perceptually distinct.** |

Checkpoints saved in `embeddings/`:
- `controlvc_vae_expresso.pt` (v1, ControlVC)
- `controlvc_vae_expresso_v2.pt` (v2, ControlVC)
- `controlvc_vae_expresso_v3.pt` (v3, ControlVC)
- `openvoice_vae_expresso.pt` (v4, OpenVoice+Expresso)
- `openvoice_vae_cremad.pt` (v5, OpenVoice+CREMA-D — best diversity)

---

## 5. Current Status & Open Problems

### Working
- End-to-end pipeline: extract → train → infer produces intelligible speech
- HuBERT content encoder loads and produces real content codes
- VAE reconstructs speaker embeddings with low reconstruction loss
- Style control infrastructure is in place

### OpenVoice Style Control Results (April 9)

Trained controllable VAE on 8,712 OpenVoice embeddings (v4, label loss ~1.5 — 10x better than ControlVC). Tested with `trump_0.wav` as source.

**Perceptual evaluation (amplified style values):**

| Style | x1 | x3 | x5 |
|-------|-----|-----|-----|
| whisper | Subtle | **Sounds like a real whisper** | Over-exaggerated, background noise |
| sad | Subtle shift | **Sounds like an actual sad person** (some distortion) | Sad but more distortion |
| happy | Barely perceptible | Jovial but close to original, not pronounced | Distorted, not clearly happy |
| laughing | No difference | No meaningful change | Minimal change |

**Key findings:**
- **x3 is the sweet spot** — x5 overdrives the decoder and introduces artifacts
- **Whisper and sad work well** via embedding alone — these styles map to strong tonal shifts (pitch reduction, energy reduction, flattening)
- **Happy and laughing need F0 augmentation** — their perceptual qualities (pitch variation, rhythmic changes) aren't fully captured in the 256-dim embedding
- **Massive improvement over ControlVC** — went from zero audible difference to clearly recognizable style shifts

**Numerical differences from baseline (mean absolute):**
| Style | x1 | x3 | x5 |
|-------|-----|-----|-----|
| whisper | 0.039 | 0.062 | **0.099** |
| sad | 0.036 | 0.038 | 0.056 |
| happy | 0.035 | 0.037 | 0.037 |
| laughing | 0.034 | 0.030 | 0.037 |

**F0 post-processing attempt (failed):**
Tried combining embedding control (x3) with F0 pitch shifting on the output audio for happy (+20%), laughing (+30%), confused (+10%). Result: voices became thin and high-pitched, not perceptibly "happy" or "laughing." Pitch shifting output audio is too crude — happiness and laughter are speech *behaviors* (varied intonation patterns, rhythm, breath) not achievable by shifting a single signal. Reverted F0 post-processing from OpenVoice wrapper.

**Conclusion:** Embedding-based style control with OpenVoice at x3 works for **tonal/energy styles** (whisper, sad) but not for **behavioral styles** (happy, laughing, confused). This is a meaningful finding — it reveals which style dimensions are capturable in a 256-dim speaker embedding vs. which require fundamentally different representations.

### CREMA-D Results (April 10, branch `feat/cremad-experiments`)

**Motivation:** Expresso has only 3 speakers — within-style variance is dominated by speaker identity, not emotion. Joe recommended maximizing speaker diversity. CREMA-D has 91 speakers × 6 emotions × ~13 sentences = 7,442 clips.

**Dataset:** [AbstractTTS/CREMA-D on HuggingFace](https://huggingface.co/datasets/AbstractTTS/CREMA-D). 6 emotions: anger, disgust, fear, happy, neutral, sad. Extracted 1 sample per speaker per emotion = 546 samples (per Joe's recommendation).

**Embedding separability (CREMA-D, 91 speakers):**

| Emotion | Dist from global | Within var | Ratio |
|---------|-----------------|------------|-------|
| anger | 0.2712 | 0.4875 | 0.56 |
| sad | 0.2006 | 0.4875 | 0.41 |
| happy | 0.1612 | 0.4978 | 0.32 |
| neutral | 0.1430 | 0.4768 | 0.30 |
| disgust | 0.1483 | 0.4891 | 0.30 |
| fear | 0.1321 | 0.5180 | 0.26 |

Overall ratio: 0.54. Speaker separability: 1.43 (speakers are well-separated, emotions are not). The embedding fundamentally prioritizes speaker identity over emotion — but the VAE can still learn nonlinear separations.

**Perceptual evaluation (x3 amplification):**

| Emotion | Listener assessment |
|---------|-------------------|
| anger | Subtly different, noticeable |
| disgust | Similar to anger |
| fear | Distinct but not "fearful" — sounds timid |
| happy | Does sound happy, especially toward end of speech |
| neutral | Neutral sounding |
| sad | Sounds sad |

**Acoustic analysis (librosa F0/RMS/spectral centroid):**

| Style | dF0 Mean | dF0 Std | dF0 Range | dEnergy | dBrightness |
|-------|----------|---------|-----------|---------|-------------|
| anger | +12.4 | +2.1 | -12.7 | -0.001 | +123 |
| disgust | -38.9 | +3.8 | -36.3 | +0.001 | -27 |
| fear | -5.4 | +7.6 | -6.4 | +0.017 | -26 |
| happy | -14.4 | +12.1 | +12.9 | +0.001 | -49 |
| neutral | -53.4 | +0.1 | -43.7 | +0.007 | -178 |
| sad | -12.9 | +2.7 | -8.0 | +0.008 | -223 |

**Key findings:**
- **Happy has highest F0 variance** (+12.1 over baseline) — exactly matches how happy speech sounds
- **Anger is brightest** (+123) — sharper, edgier tone
- **Sad is darkest** (-223 brightness) — muffled, lower quality
- **Neutral is flattest** (F0 std +0.1, smallest range) — monotone
- **All 6 emotions are acoustically distinct** — major improvement over Expresso where only whisper/sad worked
- **Speaker diversity matters more than label separability** — despite lower raw separability (0.54 vs 0.62), the VAE learned better generalizable patterns from 91 speakers vs 3

**Why CREMA-D works better than Expresso for emotion control:**
1. 91 speakers forces the VAE to find emotion patterns that generalize across voices
2. 1 sample per speaker per emotion prevents speaker memorization
3. CREMA-D emotions are acted with clear intent (professional actors), while Expresso styles are more subtle reading variations

### Combined CREMA-D + Expresso Results (April 10, branch `feat/cremad-experiments`)

**Motivation:** CREMA-D provides speaker diversity (91 speakers, 6 emotions) but lacks Expresso's unique styles (whisper, confused, enunciated). Combining both gives the best of both worlds: 9 unified labels with strong speaker diversity.

**Unified label scheme (825 samples, ~90-94 per label):**
- From CREMA-D (91 speakers): anger, disgust, fear, happy, neutral, sad
- From Expresso (3 speakers, capped at 90): confused, enunciated, whisper
- Shared (CREMA-D + 1-per-speaker Expresso): happy, neutral, sad

**VAE v6:** 3000 epochs, lr=1e-4, latent_dims=15 (9 labels + 6 free). Final label loss ~30.

**Acoustic analysis (all deltas from baseline):**

| Style | dF0 Mean | dF0 Std | dF0 Range | dBrightness | Signature |
|-------|----------|---------|-----------|-------------|-----------|
| anger | +13.2 | -3.9 | +4.4 | +369 | Sharp, edgy |
| confused | +2.5 | -3.8 | -29.5 | +111 | Hesitant, narrow range |
| disgust | -25.6 | +6.1 | +24.3 | -156 | Low, withdrawn |
| enunciated | -10.5 | -3.9 | +5.2 | +274 | Crisp, bright |
| fear | +17.4 | +4.8 | +80.8 | -155 | Tense, variable |
| **happy** | **+26.8** | **+33.2** | **+240.7** | +84 | **Most expressive — 2x F0 variation** |
| neutral | -46.0 | +6.7 | +38.8 | -533 | Flat, subdued |
| sad | -0.3 | +11.2 | +44.3 | -272 | Darker tone |
| **whisper** | **-128.6** | **-30.8** | **-143.6** | **+945** | **F0 near zero, breathy/airy** |

**Perceptual evaluation:** All 9 styles perceptually distinct and matching acoustic expectations. This is the first time happy has produced a convincingly animated output.

**Why the combined model works:**
1. **Speaker diversity from CREMA-D** (91 speakers) prevents speaker memorization
2. **Style richness from Expresso** adds whisper/confused/enunciated which CREMA-D lacks
3. **Balanced sampling** (~90 per label) prevents any label from dominating
4. **Cross-dataset generalization** — the model learns emotion patterns that hold across two completely different recording conditions

**Progression of results:**
| Model | Dataset | Speakers | Working styles | Happy? |
|-------|---------|----------|---------------|--------|
| v1-v3 | Expresso+ControlVC | 3 | 0 | No |
| v4 | Expresso+OpenVoice | 3 | 2 (whisper, sad) | No |
| v5 | CREMA-D+OpenVoice | 91 | 6 (all emotions) | Yes |
| **v6** | **Combined** | **94** | **9 (all labels)** | **Yes** |

---

### ControlVC Style Differentiation (April 9 — superseded by OpenVoice)
**The core problem with ControlVC:** All style-controlled outputs sounded identical. Setting different style dims at inference produced no audible differences.

**Root cause identified (April 9):** Empirical analysis of the raw 256-dim D_VECTOR embeddings shows that **styles are not separable in embedding space:**

```
Embedding analysis (4,840 samples, unit-normalized):
  Between-style mean centroid distance: 0.0298
  Within-style mean distance to centroid: 0.0338
  Separability ratio: 0.88 (needs >>1)

  Only whisper shows any separation (~0.06 from others)
  All other styles overlap almost completely
```

**What this means:** The D_VECTOR speaker embedding encodes speaker identity, not speaking style. Within any given style, samples from different speakers vary MORE than the style signal itself. The VAE cannot learn to control what the input representation doesn't encode.

**Style distribution in training data:**
| Style | Samples | Notes |
|-------|---------|-------|
| default | 759 | |
| confused | 760 | |
| enunciated | 760 | |
| happy | 760 | |
| laughing | 557 | |
| sad | 379 | |
| whisper | 379 | Only style with measurable embedding separation |
| emphasis | 400 | |
| essentials | 80 | Too few samples |
| longform | 3 | Effectively zero |
| singing | 3 | Effectively zero |

**Training code observation:** `beta = 1` and losses use `.sum()` not `.mean()`. Label loss sums over batch×11 dims = 2,816 terms. Recon loss sums over batch×256 dims = 65,536 terms. Label loss is inherently ~23x smaller in gradient magnitude — the VAE heavily prioritizes reconstruction over label matching.

**Confirmed April 9:** Tested v3 checkpoint (label loss ~15) with style_value=5.0 (5x normal). Zero audible difference from baseline. The decoder is simply not sensitive to these latent dims because the input representation doesn't carry style.

**Possible paths forward:**

1. **Control F0 instead of (or in addition to) speaker embedding:** Style differences primarily manifest in prosody (F0 contour), not speaker identity. Applying the controllable VAE to F0 features might yield audible style differences. The ControlVC pipeline already extracts F0 via YAAPT — we just don't route it through the VAE.

2. **Use a style-aware embedding model:** Replace D_VECTOR with an embedding model that jointly encodes identity and style (e.g., emotion-aware speaker encoder, or a multi-task model trained on both speaker ID and emotion).

3. **Switch to per-speaker style control:** Within a single speaker, style differences may be more detectable (no cross-speaker variance to drown the signal). Train speaker-specific VAEs or condition on speaker ID.

4. **Operate on a concatenated representation:** Instead of just the 256-dim speaker embedding, pass [speaker_emb; F0_stats; energy_stats] through the VAE. This gives the model more style-relevant information to work with.

5. **~~Ask Joe~~** ✓ **ANSWERED (April 9):** Joe confirmed that OpenVoice embeds F0 in the speaker embedding, so style changes ARE audible with OpenVoice. He noted that some VC systems (like ControlVC) treat F0 separately, so randomizing the speaker embedding alone doesn't produce audible style changes. He also reminded us that the original plan was to use Expresso with **OpenVoice, not ControlVC**, since OpenVoice has been easier to work with.

**Conclusion: Pivot to OpenVoice for controllable style work.** The ControlVC pipeline is still valuable for DP anonymization (speaker embedding noise), but style control requires a system where F0 is part of the embedding — which OpenVoice provides.

---

## 6. Architecture Details

### VAE (model_embedding_vae.py)

```
Encoder: Linear(256→512) → GELU → LayerNorm(512) → Linear(512→256) → GELU → Linear(256→64) → GELU
         → to_mu(64→latent_dim), to_logvar(64→latent_dim)

Decoder: Linear(latent_dim→64) → GELU → Linear(64→256) → GELU → Linear(256→512) → GELU → Linear(512→256)
```

Reparameterization: `z = mu + eps * exp(0.5 * logvar)`

DP noise (at inference): L2 clip → Gaussian noise → post-clip → clamp

Control features: After computing z, override specific dims: `z[:, idx] = value`

### Training Loss
```
total_loss = recon_loss + kl_loss + beta * label_mse_loss
```
Where:
- `recon_loss`: MSE between input and reconstructed embedding
- `kl_loss`: KL divergence (standard VAE)
- `label_mse_loss`: MSE between first K latent dims and target labels (one-hot style encoding)
- `beta`: weighting factor (needs investigation — see `dpvc/utils.py`)

### Anonymizer Flow
```python
source_embedding = wrapper.extract_embedding(source_file)   # 256-dim
target_embedding = VAE(source_embedding, noise, control_features)  # 256-dim (modified)
wrapper.inference(source_file, output_file, source_embedding, target_embedding)
```

The `inference()` call uses the source audio's HuBERT codes and F0, but replaces the speaker embedding with the VAE-modified one.

---

## 7. File Inventory

### Core Library (`dpvc/`)
| File | Purpose | Status |
|------|---------|--------|
| `controlvc.py` | ControlVC wrapper (HuBERT, D_VECTOR, F0, vocoder) | Updated: fairseq compat patch, device auto-detect |
| `anonymizer.py` | DP pipeline orchestrator | Rewritten: vae_config dict pattern |
| `model_embedding_vae.py` | VAE model (GELU/LayerNorm, control_features) | Rewritten from controllable_vae branch |
| `utils.py` | Training utilities (train_autoencoder with labels) | Updated: label-aware training |
| `wrapper.py` | Abstract base class | Updated: get_vae_config() |
| `openvoice.py` | OpenVoice wrapper | Updated: get_vae_config() |
| `__init__.py` | Package exports | Unchanged |

### Examples (`examples/`)
| File | Purpose | Status |
|------|---------|--------|
| `controlvc_extract_expresso.py` | Extract embeddings from Expresso | New |
| `controlvc_train_vae_expresso.py` | Train controllable VAE | New |
| `controlvc_infer_controllable.py` | Controllable inference CLI | New, updated (full style encoding) |
| `controlvc_extract_commonvoice.py` | CommonVoice extraction | Updated (argparse) |
| `openvoice_inference.py` | OpenVoice demo | Updated (vae_config API) |

### Generated Artifacts (not committed)
| Path | Description |
|------|-------------|
| `embeddings/controlvc_expresso_emb.pt` | 4,840 extracted embeddings + style labels |
| `embeddings/controlvc_vae_expresso.pt` | VAE v1 (lr=1e-6, 2000 epochs) |
| `embeddings/controlvc_vae_expresso_v2.pt` | VAE v2 (lr=1e-4, 2000 epochs) |
| `embeddings/controlvc_vae_expresso_v3.pt` | VAE v3 (lr=1e-4, 5000 epochs, in progress) |
| `output/*.wav` | Generated audio samples |

---

## 8. Dependencies & Environment

- **Python:** 3.11.9 (pyenv)
- **PyTorch:** 2.0.1 (via venv)
- **torchaudio:** 2.9.1
- **fairseq:** 0.12.2 (requires dataclass monkey-patch for Python 3.11)
- **hydra-core:** 1.3.2 (upgraded from 1.0.7)
- **omegaconf:** 2.3.0 (upgraded from 2.0.6)
- **control-vc repo:** `/Users/steve/repos/control-vc` (contains checkpoints and fairseq_feature_reader.py)
- **HuBERT checkpoint:** `control-vc/checkpoints/hubert_base_ls960.pt`
- **K-means model:** `control-vc/checkpoints/km.bin`
- **Vocoder:** `control-vc/checkpoints/embed_f0stat2/g_00350000.pth`

### Venv patches (not version-controlled)
These files in `.venv/lib/python3.11/site-packages/fairseq/` were manually patched:
- `dataclass/configs.py` — `field(default_factory=...)` for FairseqConfig fields
- `dataclass/initialize.py` — handle MISSING defaults in hydra_init loop
- `checkpoint_utils.py` — `weights_only=False` for torch.load

---

## 9. Lingering Questions

### Resolved
1. ~~**Is speaker embedding the right place to control style?**~~ **No.** Confirmed empirically — D_VECTOR doesn't encode style. Pivoted to F0.
2. ~~**What is beta in the label loss?**~~ beta=1. Loss uses `.sum()` not `.mean()`, so label loss is ~23x smaller in gradient magnitude than recon loss.
3. ~~**Should we validate style presence in embeddings first?**~~ Yes, did this. Separability ratio = 0.88 (not separable).

### Still Open

1. **How do we make F0 control DP-compatible?** This is the central research question. See Section 10.1 below for a full analysis.

2. **Did the controllable VAE ever work with OpenVoice?** If Joe saw audible style differences with OpenVoice embeddings, the embedding-based approach might work for some VC systems. Need to ask.

3. **What F0 features are speaker-identifying?** Mean F0, F0 range, speaking rate, and intonation patterns are all biometric. We need to understand which features carry identity vs. style to design the right DP mechanism.

4. **Do we need the full 11 styles?** Longform (3 samples) and singing (3 samples) are useless. essentials (80 samples) is marginal. Probably should reduce to 7-8 styles for cleaner training.

5. **How does the noise budget compose across embedding + F0?** If we apply DP noise to both speaker embedding AND F0 features, the total privacy cost combines via composition. What's the right epsilon split?

---

## 10. Novelty & Paper Potential

### 10.1 The DP-Compatible F0 Problem (Key Research Question)

**The gap in our current system:**

Right now, the pipeline has two independent pieces:
1. **Speaker embedding** → VAE + DP noise → anonymized embedding (has formal privacy guarantee)
2. **F0 contour** → deterministic preset transform → modified pitch (no privacy guarantee)

The F0 contour is **biometric information**. Your pitch patterns, speaking rhythm, and intonation help identify you as a speaker. If we apply DP noise to the speaker embedding but leave F0 unprotected (or only apply a fixed transform), an attacker could potentially re-identify the speaker from the F0 alone — defeating the privacy guarantee on the embedding side.

**What "DP-compatible F0 control" means:**

Instead of hardcoded presets like `pitch_shift=1.25`, we need a mechanism where:
- F0 features pass through a noise mechanism with a formal privacy guarantee (calibrated epsilon)
- Style can still be controlled by overriding specific dimensions
- The privacy budget accounts for BOTH the embedding and F0 channels

**Three possible approaches:**

**Approach A: F0 statistics through the existing VAE**
- Extract F0 summary statistics (mean, std, range, slope) from the source audio — say 4-6 features
- Concatenate with the 256-dim speaker embedding → 260-262 dim input
- Train a single VAE on the combined representation
- First K latent dims → style labels (which now have F0 information to learn from!)
- DP noise applied to the full latent space
- Reconstruct both embedding + F0 stats; use reconstructed stats to reshape the F0 contour
- **Pros:** Single privacy budget, one model, clean architecture
- **Cons:** Different feature scales (unit-norm embedding vs. raw Hz F0), may need careful normalization

**Approach B: Separate F0 VAE**
- Train a second, smaller VAE specifically on F0 statistics
- Apply DP noise independently to each VAE
- Use composition theorem to compute total privacy cost
- **Pros:** Each model focuses on its domain, easier to tune
- **Cons:** Two models to maintain, composition increases total epsilon

**Approach C: Direct DP mechanism on F0 statistics**
- Skip the VAE for F0 — just L2-clip the F0 stats and add Gaussian noise directly
- Override specific stats with style targets (e.g., force mean_F0 = 200 Hz for "happy")
- Reconstruct the F0 contour from the (noisy) stats
- **Pros:** Simplest implementation, no training needed
- **Cons:** Less expressive, can't learn nuanced style-identity disentanglement

**Recommendation for discussion with Joe:** Approach A is the most elegant and publishable — a single VAE that jointly protects identity (via embedding) and prosody (via F0 stats), while separating style-controllable dims from identity dims. The key experiment would be: do F0 statistics form separable clusters by style? (We already know the answer should be yes — we proved F0 control produces audible differences.)

**What this enables for a paper:**
- "We show that speaker embeddings alone are insufficient for style control (separability ratio 0.88)"
- "We demonstrate that F0 prosody features carry style information that is audibly distinguishable"
- "We propose a joint embedding-prosody VAE that provides DP guarantees across both channels while maintaining controllable style"
- This would be a genuine contribution — most voice anonymization work ignores prosody as an identity channel

---

### Core Contribution
**Controllable differentially private voice conversion** — to our knowledge, no prior work combines:
- Differential privacy on speaker embeddings
- Explicit control over perceptual attributes in the anonymized output
- A VAE architecture that separates controllable (labeled) and free (identity/privacy) latent dimensions

### Why This Matters
Standard DP voice anonymization destroys all speaker information indiscriminately. Controllable DP-VC lets you *choose* what to reveal: "anonymize the speaker, but make them sound happy" or "anonymize but preserve the emotional tone." This has applications in:
- **Call center anonymization:** Strip identity but preserve customer sentiment
- **Witness protection recordings:** Change voice but maintain emotional authenticity
- **Accessible media:** Re-voice content with specific style properties

### Paper Framing Ideas
1. **"Prosody-Aware Differentially Private Voice Conversion"** — argue that existing DP voice anonymization leaks identity through prosody, then show a joint embedding+F0 mechanism that protects both while enabling style control
2. **"Controllable Differential Privacy for Voice Conversion"** — frame as a privacy-utility tradeoff where "utility" includes perceptual style control
3. **"Expressive Voice Anonymization with Formal Privacy Guarantees"** — emphasize the practical application angle

### Key Results to Include
- **Negative result (important!):** Speaker embeddings (D_VECTOR) don't encode style. Separability ratio = 0.88. This is worth reporting — it tells the community that embedding-only style control doesn't work.
- **Positive result:** F0 prosody manipulation produces audibly distinct styles through the VC pipeline. Style lives in prosody, not in speaker embeddings.
- **The gap:** F0 is an unprotected identity channel in current DP voice anonymization systems.

### What Would Strengthen a Paper
- **Quantitative style evaluation:** Use a pretrained emotion classifier on outputs to measure if controlled styles are detectable
- **Speaker verification experiments:** Show that anonymization actually reduces speaker re-identification accuracy
- **F0-based re-identification attack:** Demonstrate that F0 alone can re-identify speakers even after embedding anonymization — this motivates the need for F0 protection
- **Privacy-utility curves:** Plot speaker verification accuracy vs. emotion classification accuracy at different noise levels, for embedding-only vs. embedding+F0 protection
- **Joint VAE ablation:** Compare Approach A (joint VAE) vs. Approach B (separate VAEs) vs. Approach C (direct mechanism)
- **Comparison with naive approach:** Show that just adding noise (without style control) cannot achieve the same style preservation

### Related Work to Position Against
- Voice Privacy Challenge (VPC) 2020-2024 — DP voice anonymization baselines (embedding-only, no F0 protection)
- SpeechFlow / NaturalSpeech — controllable speech synthesis (but no privacy)
- FHVAE / SpeechSplit — disentangled speech representations (but no DP)
- Prosody-based speaker recognition literature — establishes that F0 IS identifying (motivates our work)

---

## 11. Meeting Prep Notes (for April 10)

### What to Demo
- The end-to-end pipeline works: extract → train → infer → audible speech
- HuBERT content encoder is now functional (was broken, produced noise)
- **F0-based style control produces audible differences** (validated April 9)
- Can play back original Trump audio → baseline VC → happy/sad/whisper/laughing variants

### What to Discuss
- **ControlVC D_VECTOR doesn't encode style** — confirmed empirically (separability ratio 0.88) and by Joe ("some systems treat F0 differently"). Not a dead end for the project, just the wrong VC system for style control.
- **OpenVoice is the right target for controllable style** — its embedding captures F0/prosody, so the controllable VAE should produce audible differences. This was the original plan per Joe.
- **Concrete next step:** Re-run the Expresso extraction + VAE training pipeline with OpenVoice instead of ControlVC. The code is already set up for this (`dpvc/openvoice.py` has `get_vae_config()`).
- **F0 prosody control on ControlVC works as a fallback** — we proved direct F0 manipulation produces audible style differences. This could still be useful for ControlVC-based anonymization even if the VAE-based approach moves to OpenVoice.
- **DP question for longer term:** If we do both embedding-based style control (OpenVoice) AND F0 manipulation, how do the privacy budgets compose? Is there a unified approach?
- **The fairseq compat situation** is fragile (monkey-patches for Python 3.11). Worth discussing whether to pin Python 3.10 or migrate to torchaudio's HuBERT.

### Joe's Feedback (April 9, pre-meeting)
> "Some systems treat f0 differently and so randomizing the speaker embedding doesn't sound like a big change. OpenVoice embeds the f0 profile in the speaker embedding, so with OpenVoice you do tend to get audible differences. I think we discussed doing that with OpenVoice (not ControlVC) since OpenVoice has been the easiest to work with in general."

**Implication:** The controllable VAE architecture is sound — the problem is ControlVC's D_VECTOR, not the approach. Switching to OpenVoice should produce audible style differences because its embedding captures F0/prosody.

### Prior Meeting Notes (March 18 call — Expresso plan)

**The agreed-upon plan was always OpenVoice + Expresso:**
1. **Extraction:** Read Expresso wav files, extract speaker embeddings using the OpenVoice wrapper, save embeddings + style labels to a .pt file
   - Reference: `controllable_vae` branch → `examples/openvoice_extract_commonvoice_features.py`
2. **Train VAE with labeled features:** For K labeled features, force the first K latent dims to match the labels via MSE loss during training
   - Reference: `controllable_vae` branch → `examples/openvoice_train_vae_features.py`

**The mechanism (from Joe):**
- Latent representation has N dimensions (e.g., 8)
- If we have K labeled features (e.g., age, gender, accent), the first K dims are forced to equal those labels
- Training loss includes MSE between feature values and corresponding latent dims
- At inference, override those K dims to control the output

**Key URLs from that meeting:**
- Expresso dataset: https://speechbot.github.io/expresso/
- NaturalSpeech3 extraction example: `controllable_vae` branch → `examples/naturalspeech3_extract_commonvoice...`
- OpenVoice extraction: `controllable_vae` branch → `examples/openvoice_extract_commonvoice_features.py`
- OpenVoice VAE training: `controllable_vae` branch → `examples/openvoice_train_vae_features.py`

**What we did instead:** Built the pipeline for ControlVC (which we now know doesn't embed F0 in the speaker embedding). Need to redo with OpenVoice as originally planned.

### 3.7 OpenVoice Expresso Extraction (April 9, branch `feat/openvoice-expresso`)

Per Joe's feedback and the original plan, pivoted to extracting Expresso embeddings with OpenVoice instead of ControlVC. OpenVoice embeds F0/prosody in its speaker embedding, so style control via the controllable VAE should produce audible differences.

**Changes:**
- `dpvc/openvoice.py` — Rewrote `extract_embedding()` to call `tone_color_converter.extract_se()` directly, bypassing OpenVoice's `get_se()` which runs VAD-based audio splitting. The VAD splitting asserts `num_splits > 0` (requires ~10s of speech after VAD), causing 75% of short Expresso utterances to fail. Direct extraction works on any length audio.
- `dpvc/__init__.py` — Uncommented and fixed OpenVoiceWrapper export
- `examples/openvoice_extract_expresso.py` — New extraction script adapted from ControlVC version:
  - Uses pandas for parquet loading (bypasses HuggingFace datasets library issues with incomplete cache)
  - Falls back to HuggingFace `load_dataset` when no `--parquet-dir` given
  - Same one-hot style encoding (+1 active, -1 others) across 11 styles
  - ~15-20 samples/sec on CPU (faster than ControlVC extraction)

**Extraction results:**
- 8,712 total samples across 9 parquet files (only 9 of 12 downloaded — 3 missing shards)
- 0% skip rate (vs. 75% with the old VAD-based extraction)
- Embedding shape: [N, 256, 1] (256-dim, same as ControlVC D_VECTOR)
- Saved to `embeddings/openvoice_expresso_emb.pt`

**Installation notes:**
- OpenVoice installed via `pip install git+https://github.com/myshell-ai/OpenVoice --no-deps` (PyPI package doesn't exist)
- Manual deps: unidecode, inflect, cn2an, pypinyin, jieba, eng_to_ipa, langid, whisper-timestamped
- Checkpoint auto-downloads to `~/.cache/openvoice_checkpoint/` (122 MB)

### What's Committed vs. Uncommitted
- **Committed (feat/controlvc):** VAE port, Expresso pipeline scripts, portability fixes, .gitignore
- **Committed (feat/f0-style-control):** F0 transform in controlvc.py, anonymizer.py, wrapper.py; inference CLI with F0 presets; fairseq Python 3.11 compat patch
- **Committed (feat/openvoice-expresso):** OpenVoice extraction script (initial version)
- **Not committed:** openvoice.py VAD bypass fix, __init__.py export fix, extraction script pandas update, WORKLOG.md, generated artifacts

---

## 12. Reproduction Commands

```bash
# Setup
cd /Users/steve/UVM-plaid/dp-vc
source .venv/bin/activate
export PYTHONPATH=/Users/steve/UVM-plaid/dp-vc

# ===== OpenVoice Pipeline (recommended) =====

# 1. Extract OpenVoice embeddings from Expresso (~8 min on CPU)
python examples/openvoice_extract_expresso.py \
  --output embeddings/openvoice_expresso_emb.pt \
  --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/9fb79a189698de3255eff48edd2bc0d9e487adc0/read

# 2. Train controllable VAE on OpenVoice embeddings
python examples/controlvc_train_vae_expresso.py \
  --embeddings embeddings/openvoice_expresso_emb.pt \
  --output embeddings/openvoice_vae_expresso.pt \
  --epochs 2000 --lr 1e-4

# 3. Run controllable inference (TODO: adapt for OpenVoice)
# python examples/openvoice_infer_controllable.py ...

# ===== ControlVC Pipeline (F0 style control) =====

# 1. Extract ControlVC embeddings from Expresso (~10 min on CPU)
export HF_HUB_DISABLE_XET=1
python examples/controlvc_extract_expresso.py \
  --repo-root /Users/steve/repos/control-vc \
  --output embeddings/controlvc_expresso_emb.pt \
  --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/9fb79a189698de3255eff48edd2bc0d9e487adc0/read

# 2. Run F0-based style inference (no VAE needed for F0 control)
python examples/controlvc_infer_controllable.py \
  --repo-root /Users/steve/repos/control-vc \
  --source examples/trump_0.wav \
  --out output/trump_happy.wav \
  --style happy --noise-level 0.5
```
