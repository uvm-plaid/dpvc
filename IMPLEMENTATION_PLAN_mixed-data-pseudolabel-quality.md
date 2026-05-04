# Implementation Plan: `research/mixed-data-pseudolabel-quality`

## 1. Why this branch exists

The first real **CommonVoice + CREMA-D + Expresso** mixed-data run answered the
schedule question narrowly:

- `mixed_labeled_finish` improved WER the most,
- `mixed_static_balanced` improved novelty the most,
- and none of the three schedules improved recall beyond `16.7%`.

That means the next meaningful intervention is **not another simple schedule
variant**.

The next meaningful intervention is to improve:

- CommonVoice pseudo-label quality,
- class balance inside the accepted pseudo-labeled pool,
- and the degree to which the smaller labeled datasets are protected during
  training.

This branch should answer whether those changes can recover control without
giving back the mixed-data intelligibility gains.

## 2. Branch sequence

### Step 1. Mixed-data quality branch

- **Branch:** `research/mixed-data-pseudolabel-quality`
- **Purpose:** improve pseudo-label filtering, class balance, and labeled-data
  protection inside the mixed-data training line

### Step 2. Non-Trump strength-sweep branch

- **Branch:** `research/nontrump-style-strength-sweep`
- **Purpose:** run the pending strength sweep above `5.0` on representative
  non-Trump speakers after the quality branch identifies the best checkpoint to
  test

These should stay as **separate branches**:

- the first changes training artifacts and model-selection logic,
- the second is a smaller inference/evaluation study with user-facing guidance
  updates.

## 3. Core question

**Can better pseudo-label filtering plus stronger labeled-data protection
improve mixed-data recall and novelty beyond the current mixed-data line while
keeping the WER gains that made the mixed-data result interesting in the first
place?**

## 4. What stays fixed

To keep the next comparison interpretable, keep fixed:

- the OpenVoice backend,
- the 15-dim controllable VAE framing,
- the 11-speaker evaluation corpus,
- the evaluation stack:
  - `examples/eval_emotion.py`
  - `examples/eval_novelty.py`
  - `examples/eval_wer.py`
  - `examples/eval_mos.py`
- the existing mixed-data reference checkpoints:
  - `embeddings/openvoice_vae_mixed_static_balanced.pt`
  - `embeddings/openvoice_vae_mixed_cv_warmup.pt`
  - `embeddings/openvoice_vae_mixed_labeled_finish.pt`
- the comparison references already used in the mixed-data schedule matrix:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `cv500_pl_meta`

## 5. Workstream A: Mixed-data pseudo-label quality

### Goal

Build a stronger mixed-data artifact and rerun the mixed-data training matrix
with:

- better pseudo-label acceptance,
- lower `neutral` / `sad` dominance,
- and more aggressive protection for CREMA-D and Expresso during training.

### Why this is the right next step

The current `openvoice_mixed_base.pt` artifact still accepts a heavily skewed
CommonVoice pseudo-style mix:

- `neutral=207`
- `sad=170`
- `happy=37`
- `disgust=34`
- `anger=10`
- `fear=4`

The first mixed-data result suggests that this skew plus limited labeled-data
protection keeps the model in a conservative basin. The next branch should test
that directly instead of treating schedule choice as the only lever.

### Implementation steps

#### Step A1. Audit the current artifact as the baseline

Use the checked-in `mixture_report` and result bundle to freeze the current
baseline before changing anything:

- `embeddings/openvoice_mixed_base.pt`
- `results/eval_mixed_data_summary_pass9.csv`
- `results/eval_mixed_data_collapse_pass9.csv`

Record the baseline in the new branch summary so later reruns do not require
digging through multiple commits.

#### Step A2. Extend `scripts/build_mixed_training_set.py`

Keep the current speaker-first CommonVoice sampling, but add explicit controls
for per-class pseudo-label filtering and reporting.

Recommended additions:

- `--pseudo-style-thresholds anger=...,happy=...,sad=...`
  - per-style acceptance thresholds instead of one global threshold
- `--commonvoice-style-caps neutral=...,sad=...`
  - this already exists; the new branch should make it part of the default
    branch experiment instead of a buried optional knob
- `--commonvoice-style-mins`
  - optional floor for rare accepted pseudo classes if we need to avoid
    eliminating them completely
- richer reporting in `mixture_report`:
  - accepted vs rejected pseudo-label counts per style
  - threshold used per style
  - selected CommonVoice speaker count
  - selected CommonVoice labeled-row count
  - final class mix after caps

Expected artifact:

- `embeddings/openvoice_mixed_quality_base.pt`

#### Step A3. Preserve stronger labeled-data protection during training

Extend:

- `examples/openvoice_train_vae_mixed.py`
- `dpvc.utils.train_mixed_autoencoder()`

Add configurable dataset-mass controls rather than relying only on the three
hard-coded schedules.

Recommended new controls:

- `--static-masses CommonVoice=...,CREMA-D=...,Expresso=...`
- `--schedule-start-masses ...`
- `--schedule-end-masses ...`

The point is to make **labeled-data protection explicit and logged**, not just
implicit inside the source code.

#### Step A4. Train a focused quality-protection matrix

Recommended condition family:

1. `mixed_quality_static_balanced`
   - improved artifact
   - equal dataset masses

2. `mixed_quality_labeled_finish`
   - improved artifact
   - current labeled-finish logic

3. `mixed_quality_labeled_guarded`
   - improved artifact
   - more aggressive labeled protection, for example:
     - `CommonVoice=0.10`
     - `CREMA-D=0.45`
     - `Expresso=0.45`

Possible checkpoint names:

- `embeddings/openvoice_vae_mixed_quality_static_balanced.pt`
- `embeddings/openvoice_vae_mixed_quality_labeled_finish.pt`
- `embeddings/openvoice_vae_mixed_quality_labeled_guarded.pt`

#### Step A5. Generate matched evaluation corpora

Recommended output directories:

- `output/mixed_quality_static_balanced_eval/`
- `output/mixed_quality_labeled_finish_eval/`
- `output/mixed_quality_labeled_guarded_eval/`

#### Step A6. Run the full evaluation stack

Run the same four metrics as the current mixed-data line:

- emotion recall / emo_sim
- novelty gain
- WER
- MOS

Compare against:

- the three original mixed-data schedules,
- `combined`,
- `commonvoice_cv500_init`,
- `cv500_ft_short_low_lr`,
- `cv500_rich_free_anchor`,
- `cv500_pl_meta`

#### Step A7. Summarize the branch with descriptive artifact names

Add or extend a summarizer so the follow-up writes its own standalone result
bundle.

Recommended outputs:

- `results/eval_mixed_quality_summary.csv`
- `results/eval_mixed_quality_collapse.csv`

The summary should answer:

- did better pseudo-label filtering help recall?
- did stronger labeled-data protection help recall?
- which intervention moved novelty most?
- which intervention preserved WER/MOS best?
- which intervention is the best next checkpoint for a user-facing strength
  sweep?

### Validation

These should be written explicitly into `WORKLOG.md` before we leave the
branch.

- `Validation`: The mixed-data quality branch preserves a reproducible
  artifact-level `mixture_report` with per-style pseudo-label acceptance and
  rejection counts.
- `Validation`: The branch makes labeled-data protection explicit in the
  training interface rather than leaving it hard-coded.
- `Validation`: Every new condition has a named checkpoint and matched
  evaluation corpus.
- `Validation`: The comparison explicitly answers whether better pseudo-label
  filtering plus stronger labeled-data protection improve recall beyond the
  original mixed-data line.

### Success criteria

The branch is a meaningful improvement if at least one new condition:

- improves recall above `16.7%`,
- improves novelty above `0.0865` or at least reduces identity collapse
  substantially,
- and keeps mean WER close to the current mixed-data line, ideally `<= 0.09`.

If no condition clears that bar, that is still useful:

- we will have ruled out another concrete family of data-side interventions,
- and we will know the next move should likely be stronger supervision or a
  deeper architecture change rather than more filtering tweaks.

## 6. Workstream B: Non-Trump style-strength sweep

### Goal

Run the pending style-strength sweep above `5.0` on representative non-Trump
speakers and update user-facing guidance if the results hold.

### Branch recommendation

- **Branch:** `research/nontrump-style-strength-sweep`

This is worth a separate branch because it is:

- smaller in scope,
- mostly inference/evaluation plus docs,
- and easier to review separately from the mixed-data training changes.

### Implementation steps

#### Step B1. Choose the checkpoint to sweep

Use the best checkpoint from Workstream A if one clearly wins.

If Workstream A does **not** produce a new best checkpoint, fall back to the
current mixed-data references:

- `mixed_static_balanced` for novelty-oriented guidance
- `mixed_labeled_finish` for intelligibility-oriented guidance

#### Step B2. Freeze a non-Trump source panel

Document a small source panel explicitly in the repo:

- at least 4 representative non-Trump files
- ideally with both male and female speakers if the current corpus allows it
- preserve the exact file list in the sweep result artifact

#### Step B3. Sweep style strengths above `5.0`

Recommended strengths:

- `5.0`
- `7.5`
- `10.0`

Optional:

- `12.5` if `10.0` remains stable on the first panel

Recommended styles:

- `whisper`
- `happy`
- `sad`
- optionally `confused`

#### Step B4. Score the sweep

Minimum metrics:

- emotion recall / emo_sim
- novelty
- WER

Add MOS if the sweep remains small enough to score cheaply.

Recommended outputs:

- `results/eval_nontrump_strength_sweep.csv`
- `results/eval_nontrump_strength_sweep_summary.md`

#### Step B5. Update usage guidance

If the sweep shows that values above `5.0` remain useful for specific
speaker/style pairs, update:

- `examples/README.md`
- `results/README.md`
- `README.md` if the high-level guidance changes materially

### Validation

- `Validation`: The sweep uses a documented non-Trump source panel, not a loose
  ad hoc selection.
- `Validation`: The result clearly answers whether `5.0` is too conservative
  for at least some practical speaker/style pairs.
- `Validation`: Any updated strength guidance is backed by checked-in metrics
  and a documented speaker/style panel.

## 7. Documentation and context-preservation plan

This follow-up should preserve the next questions explicitly in the repo so
they survive context compaction.

At branch closeout, update:

- `WORKLOG.md`
- `FINDINGS.md` only if a new paper-facing result is verified
- `README.md`
- `examples/README.md`
- `results/README.md`

`WORKLOG.md` should explicitly record:

- branch name
- improved artifact name
- accepted/rejected pseudo-label counts
- per-class thresholds and caps used
- dataset-mass schedule values used
- checkpoint names
- evaluation corpus directories
- summary artifact names
- top-line metrics
- whether any condition beat the original mixed-data line
- which checkpoint should feed the non-Trump strength sweep

## 8. Future upgrades to preserve in the roadmap

These should stay in `WORKLOG.md` even if this branch does not reach them.

### After the mixed-data quality branch

- compare one-clip-per-speaker versus two-clips-per-speaker CommonVoice
  sampling inside the mixed-data setup,
- save one checked-in `mixture_report` snapshot next to each mixed-data result
  bundle,
- test stricter agreement filters or multi-model pseudo-label acceptance,
- compare per-class caps against prototype- or teacher-space supervision during
  pretraining,
- add per-style recovery plots for the mixed-data follow-up conditions,
- add repeated-seed uncertainty before freezing paper tables,
- cross-check the best mixed-data condition with an independent speaker
  verifier / EER pipeline,
- build novelty-vs-strength curves once the non-Trump sweep is complete,
- revisit architecture only if the improved data-side intervention still leaves
  recall flat.

## 9. Bottom line

The mixed-data branch proved that **schedule choice alone** is not enough.

This next branch exists to answer the sharper question:

**If we improve CommonVoice pseudo-label quality and protect the labeled
datasets more aggressively, can the mixed-data line finally recover recall
without giving back the WER gains that made it promising?**

## 10. Current implementation status

This branch has now been implemented and validated on the validation-scale
mixed-data setup.

Implemented artifact:

- `embeddings/openvoice_mixed_quality_base.pt`

Implemented conditions:

- `mixed_quality_static_balanced`
- `mixed_quality_labeled_finish`
- `mixed_quality_labeled_guarded`

Checked-in result artifacts:

- `results/eval_mixed_quality_summary.csv`
- `results/eval_mixed_quality_collapse.csv`

Top-line outcome:

- `mixed_quality_static_balanced`: recall `16.7%`, novelty `0.0770`, WER `0.0911`, MOS delta `-0.1130`
- `mixed_quality_labeled_finish`: recall `16.7%`, novelty `0.0763`, WER `0.0649`, MOS delta `-0.1232`
- `mixed_quality_labeled_guarded`: recall `18.2%`, novelty `0.0764`, WER `0.0978`, MOS delta `-0.1234`

Interpretation:

- better pseudo-label filtering plus stronger labeled-data protection finally
  moves mixed-data recall above `16.7%`
- the best new control-capable condition is `mixed_quality_labeled_guarded`
- the gain is still narrow and costly, because that condition gives back WER
  versus `mixed_labeled_finish` and gives back novelty versus
  `mixed_static_balanced`
- this branch does not replace the `combined` model as the best overall result

## 11. Confirmed next branch

- **Branch:** `research/nontrump-style-strength-sweep`

Why this follows naturally:

- Joe explicitly said higher strengths than `5.0` can still work well on
  non-Trump examples, especially whisper
- `mixed_quality_labeled_guarded` is now the most control-capable mixed-data
  checkpoint, so it is the right model to stress-test next on a documented
  non-Trump speaker panel
