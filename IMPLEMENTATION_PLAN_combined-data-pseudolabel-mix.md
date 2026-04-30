# Implementation Plan: `research/combined-data-pseudolabel-mix`

## 1. Why this branch exists

The April 30, 2026 meeting with Joe clarified that the most important missing
experiment is **not another CommonVoice-only refinement**.

The most important missing experiment is the first real sampled mixed-data run
that combines:
- CommonVoice,
- CREMA-D,
- and Expresso,

while preserving:
- broad speaker diversity,
- emotion/style controllability,
- intelligibility,
- and speaker novelty.

Joe also explicitly endorsed pseudo-labeled CommonVoice as the most promising
way to bootstrap the emotion side of the mixed-data run.

## 2. Branch name

- `research/combined-data-pseudolabel-mix`

## 3. Core question

**Can a sampled mixed-data training setup recover more emotion controllability
than the CommonVoice-only line while keeping enough of the CommonVoice speaker
breadth and intelligibility benefits?**

## 4. What stays fixed

To keep the experiment interpretable, this branch should keep fixed:
- the OpenVoice backend,
- the current evaluation stack:
  - emotion recall / emo_sim
  - speaker novelty
  - WER
  - MOS
- the standard 11-speaker validation corpus used in the later CommonVoice
  passes,
- the current controllable VAE framing unless the data experiment clearly fails
  in a way that justifies architecture changes.

## 5. Main hypotheses

### Hypothesis 1
The most important missing ingredient is **mixed supervision**, not more
CommonVoice-only tuning.

### Hypothesis 2
A naive full merge will likely underperform because CommonVoice will dominate
CREMA-D and Expresso unless the mixture is explicitly controlled.

### Hypothesis 3
Speaker breadth in CommonVoice matters more than raw clip count, so the next
CommonVoice sample should prioritize **at least one clip per speaker first**.

### Hypothesis 4
Pseudo-labeled CommonVoice helps only if the pseudo labels are good enough and
not so imbalanced that they push the model back into the conservative neutral /
baseline-identity basin.

## 6. Scope of the branch

This branch should stay focused on the **first mixed-data experiment family**.

It should not try to solve all of these at once:
- a new architecture,
- a privacy metric redesign,
- full large-scale CommonVoice training,
- or product/demo work.

The goal is to answer whether **sampled mixed-data training** changes the story.

## 7. Implementation steps

### Step 1. Build the mixed-data training artifact

Create a new combined embedding artifact that merges:
- CommonVoice embeddings,
- CREMA-D embeddings,
- Expresso embeddings.

Recommended new script:
- `scripts/build_mixed_training_set.py`

Expected inputs:
- `embeddings/openvoice_commonvoice_cv500_pseudo.pt`
- `embeddings/openvoice_cremad_emb.pt`
- `embeddings/openvoice_expresso_emb.pt`

Expected output:
- `embeddings/openvoice_mixed_base.pt`

Artifact schema should preserve:
- `data`
- `labels` / style targets when present
- `speaker_ids`
- `source_dataset`
- `clip_paths` when available
- pseudo-label fields for CommonVoice
- metadata coverage summary
- mixture report

### Step 2. Sample CommonVoice for speaker breadth

Implement CommonVoice sampling that prioritizes:
- at least one clip per speaker first,
- then optional extra clips only after breadth is satisfied.

Recommended controls:
- `--max-speakers`
- `--min-clips-per-speaker`
- `--max-clips-per-speaker`
- `--speaker-first`
- `--seed`

This should be explicit in the mixture report so later runs can be compared
fairly.

### Step 3. Preserve the labeled datasets in the mix

The branch should explicitly prevent CommonVoice from drowning out the labeled
sets.

Add mixture controls such as:
- target row quotas by dataset,
- oversampling of CREMA-D / Expresso,
- and per-dataset weighting metadata in the saved artifact.

At minimum, the first run family should include:
- one **static balanced mix** schedule,
- one **CommonVoice-heavy warmup** schedule,
- one **labeled-data-heavy finish** schedule.

### Step 4. Improve pseudo-label quality enough for the first mixed run

Do not treat pseudo-label improvement as a separate future branch only.

This branch should include modest quality improvements that are necessary for a
fair mixed-data run:
- better label mapping cleanup,
- optional per-class confidence thresholds,
- optional cap on accepted `neutral` pseudo labels,
- optional class-balanced acceptance,
- better reporting of pseudo-label class distribution before training.

Recommended extension points:
- `scripts/annotate_commonvoice_pseudolabels.py`
- or a new helper like `scripts/filter_commonvoice_pseudolabels.py`

### Step 5. Improve Expresso label mapping

Joe explicitly liked the idea of using Expresso's richer label space better.

This branch should include:
- a reviewed mapping from Expresso labels into the current controllable style
  space,
- documentation of any dropped / merged labels,
- and a report that shows which styles are uniquely coming from Expresso.

### Step 6. Train the first mixed-data models

Recommended condition family:
- `mixed_static_balanced`
- `mixed_cv_warmup`
- `mixed_labeled_finish`

Possible checkpoint names:
- `embeddings/openvoice_vae_mixed_static_balanced.pt`
- `embeddings/openvoice_vae_mixed_cv_warmup.pt`
- `embeddings/openvoice_vae_mixed_labeled_finish.pt`

The current implementation uses a **single mixed artifact plus schedule-aware
training**, so the exact schedule must be preserved in logs and in the result
summary.

### Step 7. Generate matched evaluation corpora

Recommended output directories:
- `output/pass9_mixed_static_balanced_eval/`
- `output/pass9_mixed_cv_warmup_eval/`
- `output/pass9_mixed_labeled_finish_eval/`

### Step 8. Run the full evaluation stack

Run:
- `examples/eval_emotion.py`
- `examples/eval_novelty.py`
- `examples/eval_wer.py`
- `examples/eval_mos.py`

Compare against these references:
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short_low_lr`
- `cv500_rich_free_anchor`
- `cv500_pl_meta`

### Step 9. Add one mixed-data summarizer

Recommended new script:
- `scripts/summarize_mixed_data_results.py`

Expected outputs:
- `results/eval_mixed_data_summary_pass9.csv`
- `results/eval_mixed_data_collapse_pass9.csv`

This summary should explicitly answer:
- did mixed-data training beat raw CommonVoice?
- did it beat the best CommonVoice recovery recipe?
- did it recover recall?
- did it retain novelty?
- did it preserve enough WER/MOS?

### Step 10. Add a strength sweep on representative non-Trump speakers

Joe's qualitative feedback means we should stop treating `5.0` as a hard
practical ceiling.

Recommended small additional experiment:
- sweep `style_strength` across values above `5.0`
- on a small non-Trump speaker panel
- at least for whisper and one or two emotional styles

This does not need to become the main branch output, but it should update user
facing guidance if the results hold.

## 7A. Current implementation status

The branch now has a validated scaffold in code:

- `scripts/build_mixed_training_set.py`
- `examples/openvoice_train_vae_mixed.py`
- `dpvc.utils.train_mixed_autoencoder()`
- `scripts/summarize_mixed_data_results.py`
- mixed-condition support in `scripts/run_ablation_inference.py`

Validated so far:

- syntax checks passed for the new training / builder / summary scripts
- the builder was smoke-tested successfully on a small artifact
- all three schedule families ran end to end on that smoke artifact
- mixed-condition inference was smoke-tested with a one-file source run
- the first real mixed artifact was built locally:
  - `embeddings/openvoice_mixed_base.pt`
  - current composition: `1325` rows = `500` CommonVoice + `546` CREMA-D + `279` Expresso
  - labeled rows: `1287/1325`

What is still missing is the **first real Pass 9 result family**.

## 7B. Immediate execution order for the next work phase

This is the concrete run order we should follow next on this branch.

### Task 1. Train the three real mixed-data checkpoints

Train from:
- `embeddings/openvoice_mixed_base.pt`

Produce:
- `embeddings/openvoice_vae_mixed_static_balanced.pt`
- `embeddings/openvoice_vae_mixed_cv_warmup.pt`
- `embeddings/openvoice_vae_mixed_labeled_finish.pt`

Recommended first commands:

```bash
python examples/openvoice_train_vae_mixed.py \
    --embeddings embeddings/openvoice_mixed_base.pt \
    --output embeddings/openvoice_vae_mixed_static_balanced.pt \
    --schedule static_balanced

python examples/openvoice_train_vae_mixed.py \
    --embeddings embeddings/openvoice_mixed_base.pt \
    --output embeddings/openvoice_vae_mixed_cv_warmup.pt \
    --schedule cv_warmup \
    --schedule-epochs 1000

python examples/openvoice_train_vae_mixed.py \
    --embeddings embeddings/openvoice_mixed_base.pt \
    --output embeddings/openvoice_vae_mixed_labeled_finish.pt \
    --schedule labeled_finish \
    --schedule-epochs 1000
```

### Task 2. Generate the Pass 9 evaluation corpora

Produce:
- `output/pass9_mixed_static_balanced_eval/`
- `output/pass9_mixed_cv_warmup_eval/`
- `output/pass9_mixed_labeled_finish_eval/`

Use:
- `scripts/run_ablation_inference.py`

### Task 3. Run the full metric stack on each Pass 9 corpus

Run:
- `examples/eval_emotion.py`
- `examples/eval_novelty.py`
- `examples/eval_wer.py`
- `examples/eval_mos.py`

Expected result artifacts:
- `results/eval_emotion_pass9_<condition>.csv`
- `results/eval_novelty_pass9_<condition>.csv`
- `results/eval_wer_pass9_<condition>.csv`
- `results/eval_mos_pass9_<condition>.csv`

### Task 4. Summarize the Pass 9 matrix

Run:
- `scripts/summarize_mixed_data_results.py`

Expected outputs:
- `results/eval_mixed_data_summary_pass9.csv`
- `results/eval_mixed_data_collapse_pass9.csv`

### Task 5. Update the paper-facing docs only if the results are verified

If the full Pass 9 run completes:
- update `WORKLOG.md`
- update `FINDINGS.md`
- update `results/README.md`
- update `examples/README.md` if workflow guidance changes

If the run is partial or blocked:
- update `WORKLOG.md`
- do **not** add a new finding yet

### Task 6. Run a small style-strength sweep on the best Pass 9 condition

After the three conditions are scored, pick the best checkpoint and test a
small non-Trump strength sweep, especially for:
- `whisper`
- one strong emotional style such as `happy` or `sad`

This is a branch follow-up, not a prerequisite for the first Pass 9 matrix.

## 8. Validation criteria

These should be recorded explicitly in `WORKLOG.md` before the branch closes.

- `Validation`: The first mixed-data training artifact is reproducible and
  documents dataset composition, speaker counts, and pseudo-label coverage.
- `Validation`: The branch compares at least three explicit mixture strategies,
  not just one naive combined run.
- `Validation`: The three real mixed-data checkpoints train successfully from
  `embeddings/openvoice_mixed_base.pt`.
- `Validation`: Each Pass 9 checkpoint has a matched evaluation corpus and
  complete metric bundle (emotion, novelty, WER, MOS).
- `Validation`: The CommonVoice sampling policy is speaker-breadth aware and
  documented.
- `Validation`: The mixed-data result explicitly answers whether combining
  CommonVoice with CREMA-D and Expresso improves controllability beyond the
  CommonVoice-only line.
- `Validation`: The branch preserves comparison against the established
  references (`combined`, `cv500`, best Pass 5 / 7 / 8 variants).
- `Validation`: Strength guidance is updated if the non-Trump sweep shows that
  values above `5.0` are consistently useful.

## 9. Success criteria

The branch is a success if at least one mixed-data condition:
- improves recall beyond `16.7%`,
- improves novelty beyond the stronger CommonVoice-only variants,
- and keeps enough of the CommonVoice intelligibility / MOS story to remain
  attractive.

The branch is still useful even if it fails, provided it tells us one of these
clearly:
- naive mixing collapses into CommonVoice,
- labeled-data protection is insufficient,
- pseudo-label quality is still the main blocker,
- or the current architecture cannot exploit the mixed-data setup.

## 10. What not to overclaim

Do not claim success just because:
- WER improves,
- or MOS improves,
- or novelty moves a little.

The whole point of this branch is to test whether we can improve **emotion
controllability and useful speaker shift** at the same time.

## 11. Future upgrades to preserve in the roadmap

These should stay in `WORKLOG.md` even if this branch does not reach them.

### After the first mixed-data run
- scale the best mixed-data recipe to a larger CommonVoice slice,
- compare one-clip-per-speaker and two-clips-per-speaker CommonVoice sampling
  inside the mixed-data setup, because the current real artifact uses one clip
  per speaker and Joe's heuristic still needs a direct empirical check,
- archive one saved `mixture_report` snapshot per Pass 9 condition next to the
  result bundle so later runs can be compared without reconstructing builder
  arguments from memory,
- try prototype- or teacher-space CommonVoice pretraining targets,
- add per-style recovery plots for mixed-data conditions,
- add repeated-seed uncertainty for the best mixed-data recipe,
- add stronger pseudo-label calibration and agreement filters,
- add per-class CommonVoice pseudo-label thresholds or caps, because the first
  real `openvoice_mixed_base.pt` artifact is still heavily skewed toward
  `neutral` (`207`) and `sad` (`170`) among the accepted pseudo-labeled
  CommonVoice rows,
- revisit architecture only after the mixed-data story is clear.

## 12. Expected next update points

When this branch is eventually implemented, we should update:
- `WORKLOG.md`
- `FINDINGS.md`
- `README.md`
- `examples/README.md`
- `results/README.md`

## 13. Bottom line

This branch exists to test the most important remaining missing experiment from
Joe's April 30 feedback:

**Can pseudo-labeled CommonVoice become useful for controllable speaker
creation once it is trained together with CREMA-D and Expresso under a
carefully controlled mixture, instead of being explored only in isolation?**
