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
| `eval_emotion_pass2_combined.csv` | 110 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | CommonVoice pretraining pipeline combined-only baseline for Finding 10 |
| `eval_wer_pass2_combined.csv`     | 110 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | CommonVoice pretraining pipeline combined-only baseline for Finding 10 |
| `eval_emotion_pass2_cv500.csv`    | 110 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | Finding 10 (`cv500` CommonVoice init candidate) |
| `eval_wer_pass2_cv500.csv`        | 110 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | Finding 10 (`cv500` CommonVoice init candidate) |
| `eval_novelty_pass2_combined.csv` | 110 | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | speaker novelty metric work novelty baseline for Finding 11 |
| `eval_novelty_pass2_cv500.csv`    | 110 | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | Finding 11 (`cv500` novelty candidate) |

Evaluation ablation matrix artifacts from 2026-04-28:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass4_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 combined reference condition |
| `pass4_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 CommonVoice-init condition |
| `pass4_cremad_only` | 77 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 CREMA-D-only condition |
| `pass4_expresso_only` | 77 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 Expresso-only condition |
| `pass4_naive_noise_baseline` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 naive unlabeled-latent baseline |
| `eval_ablation_summary_pass4.csv` | 5 conditions | [`scripts/summarize_ablation_results.py`](../scripts/summarize_ablation_results.py) | Finding 12 top-line matrix |
| `eval_ablation_collapse_pass4.csv` | per generated file | [`scripts/summarize_ablation_results.py`](../scripts/summarize_ablation_results.py) | Finding 12 collapse taxonomy |

CommonVoice finetune ablation artifacts from 2026-04-28:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass5_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 combined reference condition |
| `pass5_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 raw `cv500` reference condition |
| `pass5_cv500_ft_short` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 short-finetune condition |
| `pass5_cv500_ft_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 low-LR condition |
| `pass5_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 best partial-recovery condition |
| `pass5_cv500_ft_freeze_decoder` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 decoder-freeze negative result |
| `pass5_cv500_ft_freeze_encoder` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 encoder-freeze condition |
| `eval_commonvoice_finetune_summary_pass5.csv` | 7 conditions | [`scripts/summarize_commonvoice_finetune_ablation.py`](../scripts/summarize_commonvoice_finetune_ablation.py) | Finding 13 top-line matrix |
| `eval_commonvoice_finetune_collapse_pass5.csv` | per generated file | [`scripts/summarize_commonvoice_finetune_ablation.py`](../scripts/summarize_commonvoice_finetune_ablation.py) | Finding 13 collapse taxonomy |

CommonVoice objective ablation artifacts from 2026-04-28:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass6_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 combined reference condition |
| `pass6_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 raw `cv500` reference condition |
| `pass6_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 best CommonVoice finetune reference |
| `pass6_cv500_obj_label2` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 label-upweight condition |
| `pass6_cv500_obj_label4` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 high-label-weight condition |
| `pass6_cv500_obj_label_ramp` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 label-ramp condition |
| `pass6_cv500_obj_recon_half_label2` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 reduced-reconstruction condition |
| `eval_commonvoice_objective_summary_pass6.csv` | 7 conditions | [`scripts/summarize_commonvoice_objective_ablation.py`](../scripts/summarize_commonvoice_objective_ablation.py) | Finding 14 top-line matrix |
| `eval_commonvoice_objective_collapse_pass6.csv` | per generated file | [`scripts/summarize_commonvoice_objective_ablation.py`](../scripts/summarize_commonvoice_objective_ablation.py) | Finding 14 collapse taxonomy |

CommonVoice rich-objective ablation artifacts from 2026-04-29:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass7_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 combined reference condition |
| `pass7_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 raw `cv500` reference condition |
| `pass7_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 best CommonVoice finetune reference |
| `pass7_cv500_rich_teacher_style` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 style-teacher distillation condition |
| `pass7_cv500_rich_free_anchor` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 free-dim anchor condition |
| `pass7_cv500_rich_teacher_plus_anchor` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 combined teacher+anchor condition |
| `eval_commonvoice_rich_objectives_summary_pass7.csv` | 6 conditions | [`scripts/summarize_commonvoice_rich_objectives.py`](../scripts/summarize_commonvoice_rich_objectives.py) | Finding 15 top-line matrix |
| `eval_commonvoice_rich_objectives_collapse_pass7.csv` | per generated file | [`scripts/summarize_commonvoice_rich_objectives.py`](../scripts/summarize_commonvoice_rich_objectives.py) | Finding 15 collapse taxonomy |

CommonVoice partial-label pretraining artifacts from 2026-04-29:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass8_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 combined reference condition |
| `pass8_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 raw `cv500` reference condition |
| `pass8_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 best CommonVoice finetune reference |
| `pass8_cv500_rich_free_anchor` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 best CommonVoice rich-objective reference |
| `pass8_cv500_pl_meta` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 metadata-only weak-label condition |
| `pass8_cv500_pl_pseudo_style` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 pseudo-style weak-label condition |
| `pass8_cv500_pl_meta_plus_pseudo` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 hybrid weak-label condition |
| `eval_commonvoice_partial_label_summary_pass8.csv` | 7 conditions | [`scripts/summarize_commonvoice_partial_label.py`](../scripts/summarize_commonvoice_partial_label.py) | Finding 16 top-line matrix |
| `eval_commonvoice_partial_label_collapse_pass8.csv` | per generated file | [`scripts/summarize_commonvoice_partial_label.py`](../scripts/summarize_commonvoice_partial_label.py) | Finding 16 collapse taxonomy |

Mixed-data pseudolabel mix schedule matrix from 2026-04-30:

- The full mixed-data result bundle is now checked in for the three new schedule conditions:
  - `pass9_mixed_static_balanced`
  - `pass9_mixed_cv_warmup`
  - `pass9_mixed_labeled_finish`
- The mixed-data summary also reuses copied reference CSVs for:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `cv500_pl_meta`
- Top-line matrix:
  - `mixed_static_balanced`: recall `16.7%`, novelty `0.0865`, mean WER `0.0825`, MOS delta `-0.1316`
  - `mixed_cv_warmup`: recall `16.7%`, novelty `0.0738`, mean WER `0.0700`, MOS delta `-0.1341`
  - `mixed_labeled_finish`: recall `16.7%`, novelty `0.0828`, mean WER `0.0606`, MOS delta `-0.1425`
- Current conclusion:
  - schedule choice changes the WER/novelty tradeoff modestly
  - none of the three schedules improves recall beyond `16.7%`
  - mixed-data training helps the CommonVoice line more on intelligibility than on controllability

Mixed-data pseudolabel quality follow-up from 2026-05-03:

- The full quality-follow-up result bundle is now checked in for:
  - `mixed_quality_static_balanced`
  - `mixed_quality_labeled_finish`
  - `mixed_quality_labeled_guarded`
- The quality summary also reuses copied reference CSVs for:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `cv500_pl_meta`
  - `mixed_static_balanced`
  - `mixed_cv_warmup`
  - `mixed_labeled_finish`
- Top-line matrix:
  - `mixed_quality_static_balanced`: recall `16.7%`, novelty `0.0770`, mean WER `0.0911`, MOS delta `-0.1130`
  - `mixed_quality_labeled_finish`: recall `16.7%`, novelty `0.0763`, mean WER `0.0649`, MOS delta `-0.1232`
  - `mixed_quality_labeled_guarded`: recall `18.2%`, novelty `0.0764`, mean WER `0.0978`, MOS delta `-0.1234`
- Current conclusion:
  - stricter pseudo-label filtering plus stronger labeled-data protection can move recall a little
  - `mixed_quality_labeled_guarded` is the first mixed-data condition to improve recall above `16.7%`
  - the gain is not a clean win, because it gives back WER versus `mixed_labeled_finish` and novelty versus `mixed_static_balanced`
  - `mixed_quality_labeled_guarded` is still the right checkpoint to carry into the pending non-Trump style-strength sweep

Non-Trump style-strength sweep from 2026-05-03:

- The full strength-sweep result bundle is now checked in for:
  - `5p0`
  - `7p5`
  - `10p0`
  - `12p5`
- Top-line overall matrix:
  - `5.0`: recall `20.8%`, novelty `0.0789`, mean WER `0.1472`, MOS delta `-0.1312`
  - `7.5`: recall `16.7%`, novelty `0.1246`, mean WER `0.1681`, MOS delta `-0.1701`
  - `10.0`: recall `16.7%`, novelty `0.1570`, mean WER `0.2028`, MOS delta `-0.2287`
  - `12.5`: recall `16.7%`, novelty `0.1761`, mean WER `0.2444`, MOS delta `-0.2119`
- Focus conclusion:
  - `5.0` remains the safest default
  - `7.5` is the best stronger-than-default compromise on the checked-in non-Trump panel
  - `whisper` and `confused` gain the most novelty from higher strength
  - `10.0-12.5` are better treated as higher-risk, higher-novelty style-specific settings than as new defaults

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
- collapse counts use the evaluation ablation matrix taxonomy implemented in `scripts/summarize_ablation_results.py`

### `eval_ablation_collapse_pass4.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- `content_collapse`: `WER >= 0.8`
- `style_collapse_to_neutral`: emotional target predicted as `neutral`
- `identity_collapse_to_baseline`: novelty gain vs baseline `<= 0.05`
- `mixed_collapse`: at least two collapse axes on the same row

### `eval_commonvoice_finetune_summary_pass5.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_combined, delta_novelty_vs_combined`

- same core fields as `eval_ablation_summary_pass4.csv`
- `delta_*_vs_cv500`: direct comparison against the original `commonvoice_cv500_init` condition
- `delta_*_vs_combined`: direct comparison against the main paper checkpoint

### `eval_commonvoice_finetune_collapse_pass5.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as evaluation ablation matrix, but applied to the CommonVoice finetune matrix

### `eval_commonvoice_objective_summary_pass6.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_best_ft, delta_novelty_vs_best_ft, delta_wer_vs_best_ft, delta_mos_delta_vs_best_ft, delta_recall_vs_combined, delta_novelty_vs_combined`

- same core fields as `eval_ablation_summary_pass4.csv`
- `delta_*_vs_cv500`: direct comparison against the original `commonvoice_cv500_init` condition
- `delta_*_vs_best_ft`: direct comparison against `cv500_ft_short_low_lr`, the best CommonVoice finetune recipe
- `delta_*_vs_combined`: direct comparison against the main paper checkpoint

### `eval_commonvoice_objective_collapse_pass6.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as Passes 4-5, but applied to the CommonVoice objective matrix

### `eval_commonvoice_rich_objectives_summary_pass7.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_best_ft, delta_novelty_vs_best_ft, delta_wer_vs_best_ft, delta_mos_delta_vs_best_ft, delta_recall_vs_combined, delta_novelty_vs_combined`

- same core fields as `eval_ablation_summary_pass4.csv`
- `delta_*_vs_cv500`: direct comparison against the original `commonvoice_cv500_init` condition
- `delta_*_vs_best_ft`: direct comparison against `cv500_ft_short_low_lr`, the best CommonVoice finetune recipe
- `delta_*_vs_combined`: direct comparison against the main paper checkpoint

### `eval_commonvoice_rich_objectives_collapse_pass7.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as Passes 4-6, but applied to the CommonVoice rich-objective matrix

### `eval_commonvoice_partial_label_summary_pass8.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_best_ft, delta_novelty_vs_best_ft, delta_wer_vs_best_ft, delta_mos_delta_vs_best_ft, delta_recall_vs_best_rich, delta_novelty_vs_best_rich, delta_wer_vs_best_rich, delta_mos_delta_vs_best_rich, delta_recall_vs_combined, delta_novelty_vs_combined, delta_wer_vs_combined, delta_mos_delta_vs_combined`

- same core fields as the CommonVoice finetune, objective, and rich-objective ablation result bundles
- `best_ft` is `cv500_ft_short_low_lr`
- `best_rich` is `cv500_rich_free_anchor`
- deltas make the CommonVoice partial-label pretraining weak-label conditions directly comparable to the strongest earlier CommonVoice baselines

### `eval_commonvoice_partial_label_collapse_pass8.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as Passes 4-7, but applied to the CommonVoice weak-label matrix

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

## CommonVoice pretraining pipeline Reproduction (`cv500`)

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

## evaluation ablation matrix Reproduction (ablation matrix)

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
already generated CommonVoice pretraining pipeline corpora:

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

## CommonVoice finetune ablation Reproduction (CommonVoice finetune ablation)

This is the narrow follow-up to Finding 10. It keeps the CommonVoice-pretrained
checkpoint fixed and varies only the finetuning recipe.

Train the five new finetune variants:

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

Generate the five evaluation corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_short --out output/pass5_cv500_ft_short_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_low_lr --out output/pass5_cv500_ft_low_lr_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_short_low_lr --out output/pass5_cv500_ft_short_low_lr_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_freeze_decoder --out output/pass5_cv500_ft_freeze_decoder_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_freeze_encoder --out output/pass5_cv500_ft_freeze_encoder_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the metrics for each corpus:

```bash
python examples/eval_emotion.py --input output/pass5_cv500_ft_short_eval --out results/eval_emotion_pass5_cv500_ft_short.csv
python examples/eval_novelty.py --manifest output/pass5_cv500_ft_short_eval/generation_manifest.jsonl --out results/eval_novelty_pass5_cv500_ft_short.csv
python examples/eval_wer.py     --input output/pass5_cv500_ft_short_eval --out results/eval_wer_pass5_cv500_ft_short.csv
python examples/eval_mos.py     --input output/pass5_cv500_ft_short_eval --out results/eval_mos_pass5_cv500_ft_short.csv
```

Repeat that four-metric block for:

- `cv500_ft_low_lr`
- `cv500_ft_short_low_lr`
- `cv500_ft_freeze_decoder`
- `cv500_ft_freeze_encoder`

Reuse the unchanged `combined` and `commonvoice_cv500_init` references by
copying their evaluation ablation matrix CSVs into the CommonVoice finetune ablation naming scheme, then summarize:

```bash
python scripts/summarize_commonvoice_finetune_ablation.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the simple gentler CommonVoice finetune variants recovers the **combined** model's controllability/novelty tradeoff
- `cv500_ft_short_low_lr` is the best partial-recovery condition, mainly by reducing identity collapse rather than improving recall
- `cv500_ft_freeze_encoder` gives the only recall bump, but it loses too much naturalness
- `cv500_ft_freeze_decoder` is a strong negative result and should not be treated as the path forward

## CommonVoice objective ablation Reproduction (CommonVoice objective ablation)

This is the objective-design follow-up to CommonVoice finetune ablation. It keeps the CommonVoice
pretrained checkpoint, the combined embeddings, and the evaluation corpus fixed,
and changes only the loss weighting during finetuning.

Train the four new objective variants:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_label2.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --label-weight 2.0

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_label4.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --label-weight 4.0

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_label_ramp.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --label-weight 1.0 \
    --label-weight-final 4.0 \
    --schedule-epochs 1000

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_recon_half_label2.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --recon-weight 0.5 \
    --label-weight 2.0
```

Generate the four evaluation corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_label2 --out output/pass6_cv500_obj_label2_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_label4 --out output/pass6_cv500_obj_label4_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_label_ramp --out output/pass6_cv500_obj_label_ramp_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_recon_half_label2 --out output/pass6_cv500_obj_recon_half_label2_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the metrics for each corpus:

```bash
python examples/eval_emotion.py --input output/pass6_cv500_obj_label2_eval --out results/eval_emotion_pass6_cv500_obj_label2.csv
python examples/eval_novelty.py --manifest output/pass6_cv500_obj_label2_eval/generation_manifest.jsonl --out results/eval_novelty_pass6_cv500_obj_label2.csv
python examples/eval_wer.py     --input output/pass6_cv500_obj_label2_eval --out results/eval_wer_pass6_cv500_obj_label2.csv
python examples/eval_mos.py     --input output/pass6_cv500_obj_label2_eval --out results/eval_mos_pass6_cv500_obj_label2.csv
```

Repeat that four-metric block for:

- `cv500_obj_label4`
- `cv500_obj_label_ramp`
- `cv500_obj_recon_half_label2`

Reuse the unchanged `combined`, `commonvoice_cv500_init`, and
`cv500_ft_short_low_lr` references by copying their existing metric CSVs into
the CommonVoice objective ablation naming scheme, then summarize:

```bash
python scripts/summarize_commonvoice_objective_ablation.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the simple scalar objective variants recovers the **combined** model's controllability/novelty tradeoff
- none of the four new variants improves recall beyond `16.7%`
- `cv500_obj_label2` is the strongest of the new objective variants, but it still trails `cv500_ft_short_low_lr` on novelty and identity collapse
- `cv500_obj_label_ramp` preserves MOS closest to the raw `cv500` init, but only by staying near the same conservative collapse basin
- the next CommonVoice experiments should focus on richer supervision or larger-scale training, not more scalar loss-weight sweeps

## CommonVoice rich-objective ablation Reproduction (CommonVoice rich objectives)

This is the richer-supervision follow-up to CommonVoice objective ablation. It keeps the CommonVoice
init checkpoint, the combined embeddings, and the evaluation corpus fixed, and
changes only the finetune-time supervision by adding style-teacher and
free-anchor losses in latent space.

Train the three new rich-objective variants:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_rich_teacher_style.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
    --style-teacher-weight 2.0 \
    --epochs 1000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_rich_free_anchor.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --free-anchor-weight 2.0 \
    --epochs 1000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_rich_teacher_plus_anchor.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
    --style-teacher-weight 2.0 \
    --free-anchor-weight 1.0 \
    --epochs 1000 --lr 3e-7
```

Generate the three evaluation corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_rich_teacher_style --out output/pass7_cv500_rich_teacher_style_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_rich_free_anchor --out output/pass7_cv500_rich_free_anchor_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_rich_teacher_plus_anchor --out output/pass7_cv500_rich_teacher_plus_anchor_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the metrics for each corpus:

```bash
python examples/eval_emotion.py --input output/pass7_cv500_rich_teacher_style_eval --out results/eval_emotion_pass7_cv500_rich_teacher_style.csv
python examples/eval_novelty.py --manifest output/pass7_cv500_rich_teacher_style_eval/generation_manifest.jsonl --out results/eval_novelty_pass7_cv500_rich_teacher_style.csv
python examples/eval_wer.py     --input output/pass7_cv500_rich_teacher_style_eval --out results/eval_wer_pass7_cv500_rich_teacher_style.csv
python examples/eval_mos.py     --input output/pass7_cv500_rich_teacher_style_eval --out results/eval_mos_pass7_cv500_rich_teacher_style.csv
```

Repeat that four-metric block for:

- `cv500_rich_free_anchor`
- `cv500_rich_teacher_plus_anchor`

Reuse the unchanged `combined`, `commonvoice_cv500_init`, and
`cv500_ft_short_low_lr` references by copying their existing metric CSVs into
the CommonVoice rich-objective ablation naming scheme, then summarize:

```bash
python scripts/summarize_commonvoice_rich_objectives.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the richer teacher/anchor variants recovers the **combined** model's controllability/novelty tradeoff
- none of the three new variants improves recall beyond `16.7%`
- `cv500_rich_free_anchor` is the strongest CommonVoice rich-objective ablation variant, mainly by improving WER and MOS while still trailing `cv500_ft_short_low_lr` on novelty and identity collapse
- `cv500_rich_teacher_style` and `cv500_rich_teacher_plus_anchor` stay in the same conservative neutral-collapse basin
- the next CommonVoice experiments should focus on richer supervision earlier in the pipeline, partial-label/pseudo-label CommonVoice objectives, or larger-scale training once a stronger objective survives on the validation corpus

---

## CommonVoice partial-label pretraining Reproduction (CommonVoice partial-label / pseudo-label pretraining)

CommonVoice partial-label pretraining tests whether adding weak supervision during the CommonVoice stage
itself helps before combined finetuning begins.

First annotate the CommonVoice embedding artifact with pseudo labels:

```bash
python scripts/annotate_commonvoice_pseudolabels.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_emb.pt \
    --output embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --report-threshold 0.6
```

Train the three weak-supervision CommonVoice pretraining variants:

```bash
python examples/openvoice_pretrain_vae_commonvoice.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --output embeddings/openvoice_vae_commonvoice_cv500_pl_meta.pt \
    --epochs 3000 \
    --metadata-targets gender,age_bucket \
    --metadata-weight 0.5

python examples/openvoice_pretrain_vae_commonvoice.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --output embeddings/openvoice_vae_commonvoice_cv500_pl_pseudo_style.pt \
    --epochs 3000 \
    --pseudo-style-weight 1.0 \
    --pseudo-style-threshold 0.6

python examples/openvoice_pretrain_vae_commonvoice.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --output embeddings/openvoice_vae_commonvoice_cv500_pl_meta_plus_pseudo.pt \
    --epochs 3000 \
    --metadata-targets gender,age_bucket \
    --metadata-weight 0.5 \
    --pseudo-style-weight 1.0 \
    --pseudo-style-threshold 0.6
```

Finetune on the combined labeled embeddings:

```bash
python examples/openvoice_train_vae_combined.py --embeddings embeddings/openvoice_combined_emb.pt --output embeddings/openvoice_vae_combined_cv500_pl_meta.pt --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500_pl_meta.pt --epochs 1000 --lr 3e-7
python examples/openvoice_train_vae_combined.py --embeddings embeddings/openvoice_combined_emb.pt --output embeddings/openvoice_vae_combined_cv500_pl_pseudo_style.pt --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500_pl_pseudo_style.pt --epochs 1000 --lr 3e-7
python examples/openvoice_train_vae_combined.py --embeddings embeddings/openvoice_combined_emb.pt --output embeddings/openvoice_vae_combined_cv500_pl_meta_plus_pseudo.pt --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500_pl_meta_plus_pseudo.pt --epochs 1000 --lr 3e-7
```

Generate the three matched corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_pl_meta --out output/pass8_cv500_pl_meta_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_pl_pseudo_style --out output/pass8_cv500_pl_pseudo_style_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_pl_meta_plus_pseudo --out output/pass8_cv500_pl_meta_plus_pseudo_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the four metrics on each corpus, reuse the unchanged reference CSVs in the
CommonVoice partial-label pretraining naming scheme, then summarize:

```bash
python examples/eval_emotion.py --input output/pass8_cv500_pl_meta_eval --out results/eval_emotion_pass8_cv500_pl_meta.csv
python examples/eval_novelty.py --manifest output/pass8_cv500_pl_meta_eval/generation_manifest.jsonl --out results/eval_novelty_pass8_cv500_pl_meta.csv
python examples/eval_wer.py     --input output/pass8_cv500_pl_meta_eval --out results/eval_wer_pass8_cv500_pl_meta.csv
python examples/eval_mos.py     --input output/pass8_cv500_pl_meta_eval --out results/eval_mos_pass8_cv500_pl_meta.csv
python scripts/summarize_commonvoice_partial_label.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the weak-label variants improves recall beyond `16.7%`
- `cv500_pl_meta` is the best novelty result of the new variants (`0.0570`), but it still trails `cv500_ft_short_low_lr` (`0.0692`) and `cv500_rich_free_anchor` (`0.0646`)
- `cv500_pl_pseudo_style` and `cv500_pl_meta_plus_pseudo` produce the best WER of any CommonVoice variants tested so far (`0.0263` and `0.0285`), but they do so by collapsing toward baseline identity (`85-90` identity-collapse rows)
- the next CommonVoice work should focus on better pseudo-label quality, stronger teacher/prototype targets during CommonVoice pretraining, or curriculum strategies rather than simply adding these weak labels at the current validation scale
