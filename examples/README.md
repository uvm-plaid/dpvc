# Controllable DP Voice Conversion — OpenVoice Pipeline

End-to-end guide for reproducing the controllable voice anonymization results
using OpenVoice as the VC backend.

This is the repository's **active** controllable pipeline. ControlVC is still
available as a DP baseline, but style-control work should start here.

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

Batch-generate the same 10-file set (baseline + 9 styles) for every `.wav`
in `examples/source_speakers/`:

```bash
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/diverse_speakers/ \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --all-styles
```

Each run writes a JSONL manifest by default. Single-file runs create
`<out_stem>_manifest.jsonl`; batch runs create
`<out>/generation_manifest.jsonl`.

## Full Pipeline

### 0. Environment Setup

```bash
# Create venv (tested with Python 3.12 in .venv)
python -m venv .venv
source .venv/bin/activate

# Install dpvc with OpenVoice support
pip install -e ".[openvoice]"

# Dataset extraction
pip install -e ".[expresso]"

# Optional evaluation stack
pip install -e ".[eval]"
```

OpenVoice model checkpoints download automatically on first use to
`~/.cache/openvoice_checkpoint/`.

Tested Pass 1 environment:

- `torch==2.9.1`
- `torchaudio==2.9.1`
- `numpy==2.3.5`
- `librosa==0.9.1`
- `soundfile==0.13.1`
- `datasets==4.8.4`
- `pandas==3.0.2`

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

### 4b. CommonVoice pretraining extension

If you want to attack the training-coverage bottleneck directly, pretrain the
VAE on a local Common Voice corpus before finetuning on the combined
CREMA-D + Expresso labels.

If you only have downloaded Common Voice audio shards plus the full
`validated.tsv`, first build a filtered local subset that matches the repo's
expected layout:

```bash
python scripts/prepare_commonvoice_subset.py \
    --validated-tsv /path/to/full_validated.tsv \
    --clips-dir /path/to/cv-corpus-21.0-2025-03-14-subset/en/clips \
    --output-tsv /path/to/cv-corpus-21.0-2025-03-14-subset/en/validated.tsv \
    --max-speakers 500 \
    --max-clips-per-speaker 10 \
    --seed 42
```

The validation-scale Pass 2 run used a local `cv500` subset: `500` speakers,
`1,202` clips, extracted from a larger local pool of `6,795` speakers.

Then extract OpenVoice embeddings from a local Common Voice layout with
`validated.tsv` and `clips/`:

```bash
python examples/openvoice_extract_commonvoice.py \
    --corpus-path /path/to/cv-corpus-21.0-2025-03-14/en \
    --output embeddings/openvoice_commonvoice_cv500_emb.pt
```

Then run reconstruction-only pretraining:

```bash
python examples/openvoice_pretrain_vae_commonvoice.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_emb.pt \
    --output embeddings/openvoice_vae_commonvoice_cv500.pt
```

Finally finetune the combined controllable model from that checkpoint:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt
```

This preserves the existing controllable training path while letting the model
start from a broader speaker-distribution prior.

To reproduce the validation-scale comparison from Pass 2, generate the same
11-speaker evaluation corpus with the finetuned checkpoint:

```bash
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/pass2_cv500_eval/ \
    --vae-checkpoint embeddings/openvoice_vae_combined_cv500.pt \
    --all-styles \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42
```

Then evaluate and compare against the combined-only baseline:

```bash
python examples/eval_emotion.py --input output/pass2_cv500_eval/ --out results/eval_emotion_pass2_cv500.csv
python examples/eval_novelty.py --manifest output/pass2_cv500_eval/generation_manifest.jsonl --out results/eval_novelty_pass2_cv500.csv
python examples/eval_wer.py     --input output/pass2_cv500_eval/ --out results/eval_wer_pass2_cv500.csv
python scripts/compare_emotion_eval.py \
    --baseline results/eval_emotion_pass2_combined.csv \
    --candidate results/eval_emotion_pass2_cv500.csv
```

Current result from that `cv500` pass: content preservation improves sharply,
but emotion control collapses toward `neutral`, and the novelty metric shows
that most style outputs no longer move much farther from the source than
baseline conversion already does. See [`FINDINGS.md`](../FINDINGS.md)
Findings 10 and 11 before scaling this recipe up.

### 4c. Pass 4 ablation matrix

Pass 4 is the paper-strengthening comparison pass. It asks a sharper question
than "does the model work?":

**Which training condition gives the best balance of target alignment, speaker novelty, intelligibility, and naturalness?**

The official matrix compares:

- `combined`
- `commonvoice_cv500_init`
- `cremad_only`
- `expresso_only`
- `naive_noise_baseline`

Prepare the two single-dataset ablation embedding sets:

```bash
python scripts/prepare_ablation_embeddings.py --condition cremad_only
python scripts/prepare_ablation_embeddings.py --condition expresso_only \
    --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/*/read
```

Train the two ablation checkpoints:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_cremad_only_ablation_emb.pt \
    --output embeddings/openvoice_vae_cremad_ablation.pt

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_expresso_only_ablation_emb.pt \
    --output embeddings/openvoice_vae_expresso_ablation.pt
```

Generate the three new Pass 4 corpora:

```bash
python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition cremad_only \
    --out output/pass4_cremad_only_eval/ \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition expresso_only \
    --out output/pass4_expresso_only_eval/ \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition naive_noise_baseline \
    --out output/pass4_naive_noise_baseline_eval/ \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42
```

Notes:

- The `combined` and `commonvoice_cv500_init` conditions reuse the already generated
  validation corpora from Pass 2:
  - `output/pass2_combined_eval/`
  - `output/pass2_cv500_eval/`
- The naive baseline is **not** a plain noising baseline. It applies
  deterministic, L2-matched random control vectors only to the six *free*
  latent dims (`9-14`) of the combined checkpoint. That makes it a fair test
  of whether arbitrary latent perturbation can mimic structured style control.

Run the Pass 4 metrics:

```bash
python examples/eval_emotion.py --input output/pass4_cremad_only_eval --out results/eval_emotion_pass4_cremad_only.csv
python examples/eval_novelty.py --manifest output/pass4_cremad_only_eval/generation_manifest.jsonl --out results/eval_novelty_pass4_cremad_only.csv
python examples/eval_wer.py     --input output/pass4_cremad_only_eval --out results/eval_wer_pass4_cremad_only.csv
python examples/eval_mos.py     --input output/pass4_cremad_only_eval --out results/eval_mos_pass4_cremad_only.csv

python examples/eval_emotion.py --input output/pass4_expresso_only_eval --out results/eval_emotion_pass4_expresso_only.csv
python examples/eval_novelty.py --manifest output/pass4_expresso_only_eval/generation_manifest.jsonl --out results/eval_novelty_pass4_expresso_only.csv
python examples/eval_wer.py     --input output/pass4_expresso_only_eval --out results/eval_wer_pass4_expresso_only.csv
python examples/eval_mos.py     --input output/pass4_expresso_only_eval --out results/eval_mos_pass4_expresso_only.csv

python examples/eval_emotion.py --input output/pass4_naive_noise_baseline_eval --out results/eval_emotion_pass4_naive_noise_baseline.csv
python examples/eval_novelty.py --manifest output/pass4_naive_noise_baseline_eval/generation_manifest.jsonl --out results/eval_novelty_pass4_naive_noise_baseline.csv
python examples/eval_wer.py     --input output/pass4_naive_noise_baseline_eval --out results/eval_wer_pass4_naive_noise_baseline.csv
python examples/eval_mos.py     --input output/pass4_naive_noise_baseline_eval --out results/eval_mos_pass4_naive_noise_baseline.csv
```

Then summarize the matrix:

```bash
python scripts/summarize_ablation_results.py
```

This writes:

- `results/eval_ablation_summary_pass4.csv`
- `results/eval_ablation_collapse_pass4.csv`

Current Pass 4 conclusion from the checked-in artifacts:

- the **combined** model remains the best overall tradeoff
- `cv500`, `cremad_only`, and `expresso_only` are all more stable but collapse toward baseline identity and/or neutral emotion
- the naive baseline proves that **raw novelty is not enough** — it creates a larger identity shift than the combined model, but with worse target alignment and much worse naturalness

### 4d. Pass 5 CommonVoice finetune ablation

Pass 5 asks a narrower question than Pass 4:

**Can we keep the WER/MOS gains of the CommonVoice `cv500` init while recovering controllability and novelty through gentler finetuning?**

The Pass 5 matrix compares:

- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short`
- `cv500_ft_low_lr`
- `cv500_ft_short_low_lr`
- `cv500_ft_freeze_decoder`
- `cv500_ft_freeze_encoder`

Train the five new finetune variants from the unchanged CommonVoice init:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_short.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 1e-6

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_low_lr.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 3000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_short_low_lr.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_freeze_decoder.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 3000 --lr 1e-6 --freeze-decoder

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_freeze_encoder.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 3000 --lr 1e-6 --freeze-encoder
```

Generate the five matched evaluation corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_short --out output/pass5_cv500_ft_short_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_low_lr --out output/pass5_cv500_ft_low_lr_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_short_low_lr --out output/pass5_cv500_ft_short_low_lr_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_freeze_decoder --out output/pass5_cv500_ft_freeze_decoder_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_freeze_encoder --out output/pass5_cv500_ft_freeze_encoder_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the four metrics on each corpus:

```bash
python examples/eval_emotion.py --input output/pass5_cv500_ft_short_eval --out results/eval_emotion_pass5_cv500_ft_short.csv
python examples/eval_novelty.py --manifest output/pass5_cv500_ft_short_eval/generation_manifest.jsonl --out results/eval_novelty_pass5_cv500_ft_short.csv
python examples/eval_wer.py     --input output/pass5_cv500_ft_short_eval --out results/eval_wer_pass5_cv500_ft_short.csv
python examples/eval_mos.py     --input output/pass5_cv500_ft_short_eval --out results/eval_mos_pass5_cv500_ft_short.csv
```

Repeat those four commands for:

- `cv500_ft_low_lr`
- `cv500_ft_short_low_lr`
- `cv500_ft_freeze_decoder`
- `cv500_ft_freeze_encoder`

The unchanged `combined` and `commonvoice_cv500_init` references can be reused
from Pass 4 by copying their metric CSVs into the Pass 5 naming scheme.

Then summarize the matrix:

```bash
python scripts/summarize_commonvoice_finetune_ablation.py
```

This writes:

- `results/eval_commonvoice_finetune_summary_pass5.csv`
- `results/eval_commonvoice_finetune_collapse_pass5.csv`

Current Pass 5 conclusion from the checked-in artifacts:

- none of the simple gentler-finetune variants recovers the **combined** model's controllability/novelty tradeoff
- `cv500_ft_short_low_lr` is the best partial recovery recipe: novelty improves to `0.0692` and identity-collapse count drops to `44`, but recall stays stuck at `16.7%`
- `cv500_ft_freeze_encoder` is the only variant with any recall gain (`18.2%`), but it gives back too much naturalness (`-0.1261` MOS delta)
- `cv500_ft_freeze_decoder` is a strong negative result: it worsens identity collapse (`88` rows) and drops novelty below the original `cv500` init
- the next CommonVoice experiments should focus on better objectives, finer-grained adaptation, or larger-scale data, not just lighter finetuning

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

Batch-generate for a whole directory of sources:

```bash
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/diverse_speakers/ \
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
- `--manifest` (optional): Override the default manifest path. The manifest
  records source path, output path, style, style strength, noise level, seed,
  checkpoint, and latent dimensions for each generated file.

## Evaluation

Once you have generated outputs (e.g. via `--all-styles`), evaluate emotion
controllability with `eval_emotion.py`. This runs emotion2vec_plus_large
(ACL 2024) on each file and reports Recall Rate + emo_sim per the EmoVoice
paper.

```bash
# Install the emotion model backend (one-time)
pip install funasr

# Run on a directory of .wav files (expects <speaker>_<style>.wav naming)
python examples/eval_emotion.py \
    --input output/ \
    --out output/eval_emotion.csv
```

The script expects files named `<speaker_id>_<style>.wav` — which is what
`openvoice_infer_controllable.py --all-styles` produces. A file named
`<speaker_id>_baseline.wav` per speaker is used as the reference for emo_sim.

**Metrics:**
- **Recall Rate**: predicted emotion == target emotion (only 6 styles have
  emotion2vec counterparts: anger, disgust, fear, happy, neutral, sad)
- **emo_sim**: cosine similarity to same-speaker baseline embedding (0–1)

Output is a CSV with one row per file plus a printed summary table.

### Speaker novelty (proof-of-novelty metric)

`eval_novelty.py` measures how far each generated output moves from the source
speaker in OpenVoice's native speaker-embedding space. In manifest mode it also
computes a **novelty gain vs baseline** score:

- positive value: the style output moved farther from the source than plain
  baseline voice conversion did
- near-zero value: the style output is not meaningfully more novel than the
  baseline conversion

```bash
# Manifest-driven novelty eval (recommended for batch corpora)
python examples/eval_novelty.py \
    --manifest output/diverse_speakers/generation_manifest.jsonl \
    --out output/eval_novelty.csv

# One-off spot check
python examples/eval_novelty.py \
    --source examples/source_speakers/cremad_1003.wav \
    --generated output/diverse_speakers/cremad_1003_whisper.wav \
    --style whisper \
    --out /tmp/eval_novelty_oneoff.csv
```

Key columns in the CSV:

- `similarity`: cosine similarity(source, generated)
- `distance`: `1 - similarity`
- `baseline_similarity`: cosine similarity(source, same-speaker baseline conversion)
- `novelty_gain_vs_baseline`: `baseline_similarity - similarity`

This is a **speaker novelty metric**, not yet a full privacy metric. It is the
right tool for the paper question "did we generate a genuinely shifted speaker
identity?" before adding an external speaker-verification / EER pipeline.

### WER (intelligibility sanity check)

`eval_wer.py` transcribes each generated .wav with OpenAI Whisper and computes
Word Error Rate against a reference. Two reference modes:

```bash
pip install -U openai-whisper jiwer

# Drift-from-baseline (default): compares each <speaker>_<style>.wav to its
# <speaker>_baseline.wav transcription. Measures how much the style control
# changes what Whisper hears.
python examples/eval_wer.py \
    --input output/diverse_speakers/ \
    --out output/eval_wer.csv

# Absolute WER against a known ground-truth transcript
python examples/eval_wer.py \
    --input output/trump_styles/ \
    --reference-text "Our great movement. We will make America great again." \
    --out output/eval_wer_trump.csv
```

The default (drift) mode is right for the research question "does style
control degrade intelligibility?". The fixed-reference mode is right for
reporting absolute WER on a held-out test set.

### MOS (predicted naturalness)

`eval_mos.py` predicts a 1–5 Mean Opinion Score for each generated .wav
using torchaudio's `SQUIM_SUBJECTIVE` model (Meta, 2023). This is the same
quantity UTMOS (Saeki 2022, used by EmoVoice) measures; we use SQUIM
because the `utmos` pip package requires a from-source fairseq build that
conflicts with this codebase's fairseq monkey-patches.

```bash
# Score every file; reference = same-speaker baseline (for model conditioning)
python examples/eval_mos.py \
    --input output/diverse_speakers/ \
    --out output/eval_mos.csv

# Use a single fixed reference wav instead
python examples/eval_mos.py \
    --input output/diverse_speakers/ \
    --reference examples/source_speakers/cremad_1007.wav \
    --out output/eval_mos_fixed.csv
```

SQUIM_SUBJECTIVE is a **non-matching reference** predictor — it just uses
the reference waveform to condition the model, not as a comparison target.
Any clean-speech file works; using the same-speaker baseline makes the
scores more interpretable.

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
| 4b | `openvoice_extract_commonvoice.py` | local Common Voice `validated.tsv` + `clips/` | `openvoice_commonvoice_emb.pt` |
| 4c | `openvoice_pretrain_vae_commonvoice.py` | Step 4b output | `openvoice_vae_commonvoice.pt` |
| 4d | `openvoice_train_vae_combined.py --init-checkpoint ...` | Step 3 output + Step 4c checkpoint | `openvoice_vae_combined_finetuned.pt` |
| 4e | `../scripts/prepare_ablation_embeddings.py` | Step 1 or 2 outputs | `openvoice_*_ablation_emb.pt` |
| 4f | `openvoice_train_vae_combined.py` | Step 4e output | `openvoice_vae_*_ablation.pt` |
| 4g | `openvoice_train_vae_combined.py --freeze-* ...` | Step 3 output + Step 4c checkpoint | `openvoice_vae_combined_cv500_ft_*.pt` |
| 5 | `openvoice_infer_controllable.py` | Step 4 or 4d output + audio | `.wav` files |
| 5b | `../scripts/run_ablation_inference.py` | Step 4 / 4d / 4f / 4g output + audio | Pass 4 or Pass 5 evaluation corpora + manifest |
| 6 | `eval_emotion.py` | Step 5 output directory | `eval_emotion.csv` |
| 7 | `eval_novelty.py` | Step 5 manifest or explicit source/generated pair | `eval_novelty.csv` |
| 8 | `eval_wer.py` | Step 5 output directory | `eval_wer.csv` |
| 9 | `eval_mos.py` | Step 5 output directory | `eval_mos.csv` |
| 10 | `../scripts/summarize_ablation_results.py` | Pass 4 eval CSVs | `eval_ablation_summary_pass4.csv` + `eval_ablation_collapse_pass4.csv` |
| 10b | `../scripts/summarize_commonvoice_finetune_ablation.py` | Pass 5 eval CSVs | `eval_commonvoice_finetune_summary_pass5.csv` + `eval_commonvoice_finetune_collapse_pass5.csv` |

### Other files
- `openvoice_train_vae.py` — Trains basic (non-controllable) VAE. Not needed for style control.
- `openvoice_inference.py` — Basic DP inference without style control.
- `openvoice_extract_commonvoice.py` — Local Common Voice extraction for pretraining.
- `openvoice_pretrain_vae_commonvoice.py` — Reconstruction-only VAE pretraining on Common Voice.
- `eval_novelty.py` — Measures source-vs-generated speaker novelty in OpenVoice embedding space.
- `../scripts/prepare_commonvoice_subset.py` — Filters a full Common Voice `validated.tsv` down to the locally available clip subset.
- `../scripts/prepare_ablation_embeddings.py` — Builds the Pass 4 `cremad_only` / `expresso_only` embedding sets.
- `../scripts/run_ablation_inference.py` — Generates Pass 4 ablation corpora and the naive free-dim baseline.
- `../scripts/summarize_ablation_results.py` — Aggregates Pass 4 metrics into a condition table and a collapse taxonomy.
- `source_speakers/` — CREMA-D audio clips used for diverse speaker evaluation.
- `trump_0.wav` — Default test source audio.

### ControlVC equivalents (require separate control-vc repo)
The `controlvc_*` scripts are the ControlVC equivalents. These require cloning
the control-vc repo and passing `--repo-root`. Note: ControlVC's D_VECTOR
embedding does not encode style (see FINDINGS.md, Finding 1), so style
control does not work with ControlVC. Treat those scripts as the ControlVC DP
baseline path, not as the main controllable pipeline. Setup details live in
`docs/controlvc_setup.md`.

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

**fairseq errors on Apple Silicon**: this only applies to the ControlVC
baseline path. fairseq may fail on MPS; force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
```

**HuggingFace rate limiting**: Set `HF_TOKEN` for faster downloads:
```bash
export HF_TOKEN=hf_xxxxx
```
