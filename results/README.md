# Evaluation Results

Raw per-file evaluation outputs backing the numbers in [`FINDINGS.md`](../FINDINGS.md).
These are the full 258-file sweep over 27 speaker/variant configurations run on
2026-04-17 using the combined CREMA-D + Expresso VAE (`embeddings/openvoice_vae_combined.pt`).

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `eval_emotion_full.csv` | 258 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | Finding 7 (Recall Rate, emo_sim) |
| `eval_wer_full.csv`     | 258 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | Finding 8 (drift-from-baseline WER) |
| `eval_mos_full.csv`     | 258 | [`examples/eval_mos.py`](../examples/eval_mos.py)         | Finding 9 (SQUIM_SUBJECTIVE predicted MOS) |

Validation-scale CommonVoice pretraining comparison artifacts from 2026-04-28:

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `eval_emotion_pass2_combined.csv` | 110 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | Pass 2 combined-only baseline for Finding 10 |
| `eval_wer_pass2_combined.csv`     | 110 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | Pass 2 combined-only baseline for Finding 10 |
| `eval_emotion_pass2_cv500.csv`    | 110 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | Finding 10 (`cv500` CommonVoice init candidate) |
| `eval_wer_pass2_cv500.csv`        | 110 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | Finding 10 (`cv500` CommonVoice init candidate) |
| `eval_novelty_pass2_combined.csv` | 110 | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | Pass 3 novelty baseline for Finding 11 |
| `eval_novelty_pass2_cv500.csv`    | 110 | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | Finding 11 (`cv500` novelty candidate) |

Pass 4 ablation-matrix artifacts from 2026-04-28:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass4_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 combined reference condition |
| `pass4_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 CommonVoice-init condition |
| `pass4_cremad_only` | 77 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 CREMA-D-only condition |
| `pass4_expresso_only` | 77 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 Expresso-only condition |
| `pass4_naive_noise_baseline` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 naive unlabeled-latent baseline |
| `eval_ablation_summary_pass4.csv` | 5 conditions | [`scripts/summarize_ablation_results.py`](../scripts/summarize_ablation_results.py) | Finding 12 top-line matrix |
| `eval_ablation_collapse_pass4.csv` | per generated file | [`scripts/summarize_ablation_results.py`](../scripts/summarize_ablation_results.py) | Finding 12 collapse taxonomy |

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

### `eval_novelty_pass2_combined.csv` / `eval_novelty_pass2_cv500.csv`
`speaker, style, source_file, generated_file, baseline_file, similarity, distance, baseline_similarity, baseline_distance, similarity_delta_vs_baseline, distance_delta_vs_baseline, novelty_gain_vs_baseline, noise_level, style_strength, seed, vae_checkpoint, latent_dims`

- `similarity`: cosine similarity between the source speaker embedding and the generated output embedding in OpenVoice's native embedding space
- `distance`: `1 - similarity`
- `baseline_similarity`: cosine similarity between the source speaker embedding and the same-speaker baseline conversion
- `novelty_gain_vs_baseline`: `baseline_similarity - similarity`; positive means the style output is farther from the source than baseline conversion already was

### `eval_ablation_summary_pass4.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse`

- `emotion_recall`: fraction of emotional rows whose predicted label matches target
- `mean_novelty_gain_vs_baseline`: average `baseline_similarity - similarity`
- `mean_mos_delta_vs_baseline`: average style-row MOS minus same-speaker baseline MOS
- collapse counts use the Pass 4 taxonomy implemented in `scripts/summarize_ablation_results.py`

### `eval_ablation_collapse_pass4.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- `content_collapse`: `WER >= 0.8`
- `style_collapse_to_neutral`: emotional target predicted as `neutral`
- `identity_collapse_to_baseline`: novelty gain vs baseline `<= 0.05`
- `mixed_collapse`: at least two collapse axes on the same row

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

The generation step also writes `output/diverse_speakers/generation_manifest.jsonl`.
That manifest records the exact source file, output file, style, noise level,
style strength, seed, and checkpoint used for each row in the evaluation
corpus.

Numbers should reproduce within rounding.

## Pass 2 Reproduction (`cv500`)

This is the validation-scale CommonVoice pretraining comparison used in
Finding 10. It compares the current combined-only checkpoint against a
CommonVoice-initialized checkpoint trained on a local `cv500` subset
(`500` speakers / `1,202` clips).

```bash
# Combined-only baseline corpus
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/pass2_combined_eval/ \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --all-styles \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python examples/eval_emotion.py --input output/pass2_combined_eval/ --out results/eval_emotion_pass2_combined.csv
python examples/eval_novelty.py --manifest output/pass2_combined_eval/generation_manifest.jsonl --out results/eval_novelty_pass2_combined.csv
python examples/eval_wer.py     --input output/pass2_combined_eval/ --out results/eval_wer_pass2_combined.csv

# CommonVoice-pretrained candidate corpus
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/pass2_cv500_eval/ \
    --vae-checkpoint embeddings/openvoice_vae_combined_cv500.pt \
    --all-styles \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python examples/eval_emotion.py --input output/pass2_cv500_eval/ --out results/eval_emotion_pass2_cv500.csv
python examples/eval_novelty.py --manifest output/pass2_cv500_eval/generation_manifest.jsonl --out results/eval_novelty_pass2_cv500.csv
python examples/eval_wer.py     --input output/pass2_cv500_eval/ --out results/eval_wer_pass2_cv500.csv

# Emotion recall delta
python scripts/compare_emotion_eval.py \
    --baseline results/eval_emotion_pass2_combined.csv \
    --candidate results/eval_emotion_pass2_cv500.csv
```

Expected qualitative outcome from the checked-in CSVs:

- emotion recall gets worse (`25.8%` -> `16.7%`)
- mean novelty gain vs baseline collapses (`0.2599` -> `0.0369`)
- predicted labels collapse almost entirely to `neutral` (`70/110` -> `109/110`)
- WER improves substantially (`0.235` -> `0.084` mean across all non-baseline style rows)

That makes the `cv500` run a useful negative result: better content
preservation, weaker style controllability.

## Pass 4 Reproduction (ablation matrix)

This is the paper-strengthening ablation pass used in Finding 12. The matrix
compares:

- `combined`
- `commonvoice_cv500_init`
- `cremad_only`
- `expresso_only`
- `naive_noise_baseline`

Prepare and train the two single-dataset ablation conditions:

```bash
python scripts/prepare_ablation_embeddings.py --condition cremad_only
python scripts/prepare_ablation_embeddings.py --condition expresso_only \
    --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/*/read

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_cremad_only_ablation_emb.pt \
    --output embeddings/openvoice_vae_cremad_ablation.pt

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_expresso_only_ablation_emb.pt \
    --output embeddings/openvoice_vae_expresso_ablation.pt
```

Generate the three new evaluation corpora:

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

The unchanged `combined` and `commonvoice_cv500_init` conditions reuse the
already generated Pass 2 corpora:

- `output/pass2_combined_eval/`
- `output/pass2_cv500_eval/`

Run the metrics:

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

Then aggregate:

```bash
python scripts/summarize_ablation_results.py
```

Expected qualitative outcome from the checked-in CSVs:

- `combined` is still the best overall tradeoff
- `cv500`, `cremad_only`, and `expresso_only` are all stability-biased failures that collapse back toward neutral emotion and/or baseline identity
- the naive baseline produces **more novelty** than the combined model, but with worse recall and a much worse MOS delta
