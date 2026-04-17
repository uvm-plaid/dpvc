# Evaluation Results

Raw per-file evaluation outputs backing the numbers in [`FINDINGS.md`](../FINDINGS.md).
These are the full 258-file sweep over 27 speaker/variant configurations run on
2026-04-17 using the combined CREMA-D + Expresso VAE (`embeddings/openvoice_vae_combined.pt`).

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `eval_emotion_full.csv` | 258 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | Finding 7 (Recall Rate, emo_sim) |
| `eval_wer_full.csv`     | 258 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | Finding 8 (drift-from-baseline WER) |
| `eval_mos_full.csv`     | 258 | [`examples/eval_mos.py`](../examples/eval_mos.py)         | Finding 9 (SQUIM_SUBJECTIVE predicted MOS) |

## Schema

### `eval_emotion_full.csv`
`speaker, style, predicted, target, match, score, emo_sim, file`

- `predicted`: emotion2vec_plus_large argmax label (9 classes)
- `target`: what the generated file is supposed to be
- `match`: 1 if `predicted == mapped(target)`, 0 otherwise, empty for non-emotional styles (confused/enunciated/whisper) and for baseline rows
- `score`: softmax probability of the predicted label
- `emo_sim`: cosine similarity of emotion2vec embedding to same-speaker baseline (empty on baseline rows)

### `eval_wer_full.csv`
`speaker, style, wer, ref_source, reference, hypothesis, file`

- `wer`: word error rate between hypothesis and reference (lower = better, 0 = identical after normalization)
- `ref_source`: `baseline` (same-speaker baseline transcript), `self` (baseline file compared against itself), or `fixed` (when run with `--reference-text`)

### `eval_mos_full.csv`
`speaker, style, mos, delta_vs_baseline, ref_source, file`

- `mos`: SQUIM_SUBJECTIVE predicted MOS, 1–5 scale
- `delta_vs_baseline`: `mos(row) − mos(same-speaker baseline row)`, empty for baseline rows
- `ref_source`: what SQUIM used as the non-matching reference (`baseline` or `self`)

## Reproducing

From a clone with the model checkpoint built (see [`examples/README.md`](../examples/README.md) steps 1–4):

```bash
# Generate the 258-file corpus (steps 1-5 of examples/README.md)
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/diverse_speakers/ \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --all-styles

# Install eval deps (one-time)
pip install -e ".[eval]"

# Re-run the three evaluations
python examples/eval_emotion.py --input output/diverse_speakers/ --out results/eval_emotion_full.csv
python examples/eval_wer.py     --input output/diverse_speakers/ --out results/eval_wer_full.csv
python examples/eval_mos.py     --input output/diverse_speakers/ --out results/eval_mos_full.csv
```

Numbers should reproduce within rounding.
