# Differentially Private Anonymization via Voice Control

This repository provides a library for defining differentially private speaker anonymization systems using existing voice control models. The approach works for any voice control system that separates utterance information into constant-length speaker information (e.g. a speaker embedding) and time-varying content information (e.g. semantic features).

[Click here for full documentation](https://jnear.w3.uvm.edu/dpvc/)

## Current work — controllable DP voice conversion

Current paper-strengthening branch: **`research/commonvoice-partial-label-pretrain`**. We've extended the library with a **controllable** VAE that exposes 9 style knobs (anger, confused, disgust, enunciated, fear, happy, neutral, sad, whisper) on top of the DP anonymization pipeline. Primary entry points:

- **[`examples/README.md`](examples/README.md)** — end-to-end reproduction guide (extraction → training → controllable inference → evaluation).
- **[`FINDINGS.md`](FINDINGS.md)** — 16 paper-facing findings with methodology and per-row takeaways.
- **[`WORKLOG.md`](WORKLOG.md)** — roadmap and progress tracking.
- **[`results/`](results/)** — raw evaluation CSVs (emotion2vec Recall/emo_sim, WER, predicted MOS) backing the findings.

OpenVoice is the **canonical controllable pipeline**. ControlVC remains in the
repository as a useful DP baseline and wrapper reference, but not as the
recommended path for style control.

Current best checked-in result: the **combined** OpenVoice model remains the
best tradeoff across controllability, speaker novelty, intelligibility, and
naturalness. Passes 5-8 sharpened that conclusion by showing that neither
simple gentler CommonVoice finetuning, nor simple scalar objective
reweighting, nor the first richer teacher/anchor CommonVoice objectives, nor
validation-scale weak-label CommonVoice pretraining recover the combined
model's tradeoff. The main summary artifacts are:

- [`results/eval_ablation_summary_pass4.csv`](results/eval_ablation_summary_pass4.csv)
- [`results/eval_commonvoice_finetune_summary_pass5.csv`](results/eval_commonvoice_finetune_summary_pass5.csv)
- [`results/eval_commonvoice_objective_summary_pass6.csv`](results/eval_commonvoice_objective_summary_pass6.csv)
- [`results/eval_commonvoice_rich_objectives_summary_pass7.csv`](results/eval_commonvoice_rich_objectives_summary_pass7.csv)
- [`results/eval_commonvoice_partial_label_summary_pass8.csv`](results/eval_commonvoice_partial_label_summary_pass8.csv)

## Installation

Clone this repository, then install with the extras you need. The active
OpenVoice path is covered by package extras; the ControlVC baseline has a
separate setup guide because it depends on an external repo plus Python 3.10 /
fairseq compatibility work.

```bash
# Core library only
pip install -e .

# + OpenVoice backend (required for the controllable pipeline)
pip install -e ".[openvoice]"

# + Expresso dataset extraction
pip install -e ".[openvoice,expresso]"

# + Evaluation pipeline (emotion2vec, novelty, Whisper WER, predicted MOS)
pip install -e ".[openvoice,expresso,eval]"
```

Tested Pass 1 environment in `.venv`:

- `torch==2.9.1`
- `torchaudio==2.9.1`
- `numpy==2.3.5`
- `librosa==0.9.1`
- `soundfile==0.13.1`
- `datasets==4.8.4`
- `pandas==3.0.2`
- `funasr==1.3.1`
- `openai-whisper==20250625`
- `jiwer==4.0.0`

For ControlVC-specific setup, use [`docs/controlvc_setup.md`](docs/controlvc_setup.md).

## Example: basic DP anonymization (OpenVoice)

```python
import dpvc
vc_wrapper = dpvc.OpenVoiceWrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)
anonymizer.anonymize(src_path, output_path, noise_level=1.0)
```

`src_path` is an input .wav, `output_path` is the anonymized output, and `noise_level` controls the magnitude of DP noise added to the speaker embedding.

See also:

- `examples/openvoice_inference.py` — basic anonymization (no style control).
- `examples/openvoice_train_vae.py` — train a custom DP-VAE for the anonymizer.
- `examples/openvoice_infer_controllable.py` — **controllable** style-aware inference (the current headline flow; see [`examples/README.md`](examples/README.md) for the full pipeline).
- `examples/openvoice_extract_commonvoice.py` + `examples/openvoice_pretrain_vae_commonvoice.py` — Common Voice pretraining path, including validation-scale weak supervision from metadata and pseudo labels.
- `scripts/prepare_commonvoice_subset.py` — helper for turning downloaded Common Voice shards into a filtered local `validated.tsv` + `clips/` subset.
- `scripts/annotate_commonvoice_pseudolabels.py` — adds confidence-scored pseudo-style labels to a Common Voice embedding artifact.
- `scripts/prepare_ablation_embeddings.py` — builds the `cremad_only` and `expresso_only` Pass 4 ablation datasets in the unified label format.
- `scripts/run_ablation_inference.py` — generates the Pass 4 corpora, including the naive unlabeled-latent baseline.
- `scripts/summarize_ablation_results.py` — builds the condition summary table and collapse taxonomy for the paper.
- `scripts/summarize_commonvoice_finetune_ablation.py` — compares the Pass 5 CommonVoice finetune variants against `combined` and the original `cv500` init.
- `scripts/summarize_commonvoice_objective_ablation.py` — compares the Pass 6 CommonVoice objective variants against `combined`, the raw `cv500` init, and the best Pass 5 finetune recipe.
- `scripts/summarize_commonvoice_rich_objectives.py` — compares the Pass 7 teacher/anchor CommonVoice variants against `combined`, the raw `cv500` init, and the best Pass 5 finetune recipe.
- `scripts/summarize_commonvoice_partial_label.py` — compares the Pass 8 metadata / pseudo-label CommonVoice variants against `combined`, the raw `cv500` init, the best Pass 5 finetune recipe, and the best Pass 7 rich-objective recipe.
- `docs/controlvc_setup.md` — ControlVC baseline setup and smoke-test path.

## Evaluation

The evaluation scripts under `examples/` cover the three EmoVoice-style axes
plus our speaker-novelty proof:

- `examples/eval_emotion.py` — emotion2vec_plus_large Recall Rate + emo_sim (target alignment)
- `examples/eval_novelty.py` — OpenVoice native speaker-embedding novelty vs source and vs baseline conversion (speaker shift / proof of novelty)
- `examples/eval_wer.py` — OpenAI Whisper drift-from-baseline Word Error Rate (content preservation)
- `examples/eval_mos.py` — torchaudio SQUIM_SUBJECTIVE predicted MOS (naturalness)

CSV outputs from our runs live in [`results/`](results/). Schemas and reproduction steps are in [`results/README.md`](results/README.md).

For the current paper matrix, start with:

- [`results/eval_ablation_summary_pass4.csv`](results/eval_ablation_summary_pass4.csv)
- [`results/eval_ablation_collapse_pass4.csv`](results/eval_ablation_collapse_pass4.csv)
- [`results/eval_commonvoice_finetune_summary_pass5.csv`](results/eval_commonvoice_finetune_summary_pass5.csv)
- [`results/eval_commonvoice_objective_summary_pass6.csv`](results/eval_commonvoice_objective_summary_pass6.csv)
- [`results/eval_commonvoice_rich_objectives_summary_pass7.csv`](results/eval_commonvoice_rich_objectives_summary_pass7.csv)
- [`results/eval_commonvoice_partial_label_summary_pass8.csv`](results/eval_commonvoice_partial_label_summary_pass8.csv)

## Building Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/):

```bash
pip install mkdocs "mkdocstrings[python]" mkdocs-material
mkdocs build
```
