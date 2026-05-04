# Implementation Plan: `research/mixed-data-pseudolabel-teacher`

## 1. Why this branch exists

The first mixed-data branch and the mixed-data quality follow-up answered two
important questions:

- schedule choice alone is not enough,
- and first-pass pseudo-label filtering / labeled-data protection is still not
  enough.

The best mixed-data condition so far is:
- `mixed_quality_labeled_guarded`
- recall `18.2%`
- novelty `0.0764`
- WER `0.0978`
- MOS delta `-0.1234`

That is the first mixed-data line to move recall above `16.7%`, but it is still
far from the `combined` checkpoint on controllability and speaker novelty.

So the next highest-value branch is to improve the **teacher that produces the
pseudo labels**, and the **class-balanced acceptance policy** that decides which
pseudo-labeled CommonVoice rows enter the mixed-data artifact.

## 2. Branch name

- `research/mixed-data-pseudolabel-teacher`

## 3. Core question

**Can a stronger pseudo-label teacher plus better class-balanced acceptance move
mixed-data recall meaningfully above `18.2%` without giving back the WER gains
that made the mixed-data line interesting?**

## 4. What stays fixed

To keep this branch interpretable, hold fixed:
- the OpenVoice backend
- the 15-dim controllable VAE framing
- the current mixed-data artifact schema
- the 11-speaker evaluation corpus
- the 4-metric stack:
  - emotion recall / emo_sim
  - novelty
  - WER
  - MOS
- the current comparison references:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `mixed_static_balanced`
  - `mixed_labeled_finish`
  - `mixed_quality_labeled_guarded`

## 5. Working hypotheses

### Hypothesis 1
The current pseudo-label teacher is too conservative or too noisy in the wrong
way, which leaves CommonVoice pseudo-style rows clustered around a narrow
neutral/sad basin.

### Hypothesis 2
Acceptance logic needs to be class-aware, not just confidence-thresholded.
Even a decent teacher can fail if `neutral` and `sad` dominate the accepted
pool.

### Hypothesis 3
Better pseudo labels should help more than another pure schedule tweak, because
Finding 18 already showed that schedule + first-pass filtering only nudges the
result.

## 6. Implementation steps

### Step 1. Freeze the current mixed-data baseline

Use the current best mixed-data references as explicit baselines in the branch
summary:
- `results/eval_mixed_data_summary_pass9.csv`
- `results/eval_mixed_quality_summary.csv`
- `results/eval_nontrump_strength_sweep.csv`

These should remain unchanged so the new branch answers one clean question.

### Step 2. Extend pseudo-label generation

Primary target:
- `scripts/annotate_commonvoice_pseudolabels.py`

Add support for:
- teacher selection / checkpoint selection
- saved teacher metadata in the artifact
- class probability logging instead of only top-1 labels
- optional top-k retention for later filtering decisions

Recommended new controls:
- `--teacher-checkpoint`
- `--teacher-config`
- `--save-logits` or `--save-probs`
- `--top-k`
- `--batch-size`

If the current script becomes too messy, split out:
- `scripts/score_commonvoice_pseudolabels.py`
- `scripts/filter_commonvoice_pseudolabels.py`

Preferred design:
- score first
- filter second

That keeps the teacher outputs reusable across multiple acceptance policies.

### Step 3. Add better acceptance logic to the mixed-data builder

Primary target:
- `scripts/build_mixed_training_set.py`

Add or strengthen support for:
- per-class confidence thresholds
- per-class target counts or caps
- per-class fallback logic when a rare class has too few confident rows
- agreement rules if more than one teacher signal is available later
- a clear accepted vs rejected report per style

Recommended new controls:
- `--pseudo-style-thresholds anger=...,happy=...,sad=...`
- `--commonvoice-style-caps neutral=...,sad=...`
- `--commonvoice-style-targets anger=...,fear=...`
- `--acceptance-policy` with values like:
  - `confidence_only`
  - `threshold_plus_caps`
  - `balanced_targets`

### Step 4. Make the artifact reports stronger

Every mixed-data artifact should keep a richer `mixture_report` with:
- teacher identity / checkpoint
- threshold by style
- accepted count by style
- rejected count by style
- cap-skipped count by style
- fallback-selected count by style
- final pseudo-label distribution
- CommonVoice speaker count
- CommonVoice labeled-row count
- dataset masses used in training

That makes later comparisons much easier and reduces branch archaeology.

### Step 5. Train a focused teacher-quality matrix

Minimum recommended condition family:

1. `mixed_teacher_threshold_balanced`
- improved teacher outputs
- threshold + per-class balancing
- static balanced masses

2. `mixed_teacher_labeled_finish`
- improved teacher outputs
- threshold + per-class balancing
- labeled-finish masses

3. `mixed_teacher_labeled_guarded`
- improved teacher outputs
- threshold + per-class balancing
- strongest labeled-data protection

Optional stretch condition:
4. `mixed_teacher_rare_class_guarded`
- explicitly protects rare pseudo classes such as `anger` and `fear`
- only if the accepted pseudo pool is large enough to justify it

Suggested checkpoints:
- `embeddings/openvoice_vae_mixed_teacher_threshold_balanced.pt`
- `embeddings/openvoice_vae_mixed_teacher_labeled_finish.pt`
- `embeddings/openvoice_vae_mixed_teacher_labeled_guarded.pt`

### Step 6. Generate matched evaluation corpora

Suggested output dirs:
- `output/mixed_teacher_threshold_balanced_eval/`
- `output/mixed_teacher_labeled_finish_eval/`
- `output/mixed_teacher_labeled_guarded_eval/`

### Step 7. Run the full metric stack

Run:
- `examples/eval_emotion.py`
- `examples/eval_novelty.py`
- `examples/eval_wer.py`
- `examples/eval_mos.py`

Compare directly against:
- `mixed_static_balanced`
- `mixed_labeled_finish`
- `mixed_quality_labeled_guarded`
- `combined`

### Step 8. Add a dedicated summarizer for the branch

Recommended new script:
- `scripts/summarize_mixed_teacher_results.py`

Expected outputs:
- `results/eval_mixed_teacher_summary.csv`
- `results/eval_mixed_teacher_collapse.csv`

The summary should explicitly answer:
- did the stronger teacher beat `mixed_quality_labeled_guarded` on recall?
- did it preserve or improve novelty?
- did it preserve enough WER / MOS?
- did it reduce identity collapse or just move the error somewhere else?

## 7. Success criteria

A successful branch should do at least one of these clearly:
- push recall meaningfully above `18.2%`
- improve novelty without losing the mixed-data WER gains
- reduce identity collapse while preserving the current best mixed-data recall

If it fails, that is still useful if the failure is cleanly characterized.

## 8. Validation

Record these explicitly in `WORKLOG.md` at branch closeout:
- `Validation`: The stronger pseudo-label teacher path is reproducible from checked-in scripts.
- `Validation`: Every teacher-quality condition has a named artifact, checkpoint, corpus, and result bundle.
- `Validation`: The comparison explicitly answers whether teacher quality beats the current mixed-data quality baseline.
- `Validation`: The branch isolates pseudo-label teacher / acceptance changes rather than mixing in unrelated architecture changes.

## 9. Follow-up tasks to preserve

If this branch helps only a little:
- compare one-clip-per-speaker vs two-clips-per-speaker CommonVoice sampling
- add prototype-space or teacher-space style targets during mixed-data training
- revisit Expresso label mapping and rare-style preservation again

If this branch works well:
- rerun the non-Trump style-strength sweep on the new best checkpoint
- expand the sweep to a larger panel
- add style presets backed by metrics
- add repeated-seed uncertainty before freezing any paper table
