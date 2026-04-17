# Differentially Private Anonymization via Voice Control

This repository provides a library for defining differentially private speaker anonymization systems using existing voice control models. The approach works for any voice control system that separates utterance information into constant-length speaker information (e.g. a speaker embedding) and time-varying content information (e.g. semantic features).

[Click here for full documentation](https://jnear.w3.uvm.edu/dpvc/)

## Current work — controllable DP voice conversion

Active branch: **`feat/cremad-experiments`**. We've extended the library with a **controllable** VAE that exposes 9 style knobs (anger, confused, disgust, enunciated, fear, happy, neutral, sad, whisper) on top of the DP anonymization pipeline. Primary entry points:

- **[`examples/README.md`](examples/README.md)** — end-to-end reproduction guide (extraction → training → controllable inference → evaluation).
- **[`FINDINGS.md`](FINDINGS.md)** — 9 key findings with methodology and per-row takeaways.
- **[`WORKLOG.md`](WORKLOG.md)** — roadmap and progress tracking.
- **[`results/`](results/)** — raw evaluation CSVs (emotion2vec Recall/emo_sim, WER, predicted MOS) backing the findings.

## Installation

Clone this repository, then install with the extras you need:

```bash
# Core library only
pip install -e .

# + OpenVoice backend (required for the controllable pipeline)
pip install -e ".[openvoice]"

# + Expresso dataset extraction
pip install -e ".[openvoice,expresso]"

# + Evaluation pipeline (emotion2vec, Whisper WER, predicted MOS)
pip install -e ".[openvoice,expresso,eval]"
```

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

## Evaluation

The evaluation scripts under `examples/` measure the three axes the EmoVoice paper uses:

- `examples/eval_emotion.py` — emotion2vec_plus_large Recall Rate + emo_sim (target alignment)
- `examples/eval_wer.py` — OpenAI Whisper drift-from-baseline Word Error Rate (content preservation)
- `examples/eval_mos.py` — torchaudio SQUIM_SUBJECTIVE predicted MOS (naturalness)

CSV outputs from our runs live in [`results/`](results/). Schemas and reproduction steps are in [`results/README.md`](results/README.md).

## Building Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/):

```bash
pip install mkdocs "mkdocstrings[python]" mkdocs-material
mkdocs build
```
