# Implementation Plan: `research/nontrump-style-strength-sweep`

## Goal

Test whether style strengths above `5.0` remain useful on a documented non-Trump speaker panel, especially for whisper, using the best current mixed-data checkpoint:

- `embeddings/openvoice_vae_mixed_quality_labeled_guarded.pt`

## Fixed panel

- `examples/nontrump_strength_panel/female_1_cremad_1002.wav`
- `examples/nontrump_strength_panel/female_2_cremad_1012.wav`
- `examples/nontrump_strength_panel/male_1_cremad_1003.wav`
- `examples/nontrump_strength_panel/male_2_cremad_1051.wav`

## Sweep settings

- strengths: `5.0`, `7.5`, `10.0`, `12.5`
- checkpoint: `embeddings/openvoice_vae_mixed_quality_labeled_guarded.pt`
- source dir: `examples/nontrump_strength_panel/`
- noise level: `0.0`
- seed: `42`
- generation mode: `--all-styles`

## Outputs to produce

- `output/nontrump_strength_5p0_eval/`
- `output/nontrump_strength_7p5_eval/`
- `output/nontrump_strength_10p0_eval/`
- `output/nontrump_strength_12p5_eval/`
- `results/eval_nontrump_strength_5p0_emotion.csv`
- `results/eval_nontrump_strength_5p0_novelty.csv`
- `results/eval_nontrump_strength_5p0_wer.csv`
- `results/eval_nontrump_strength_5p0_mos.csv`
- same four metric CSVs for `7p5`, `10p0`, and `12p5`
- `results/eval_nontrump_strength_sweep.csv`
- `results/eval_nontrump_strength_sweep_summary.md`

## Validation

- `Validation`: The sweep uses a fixed non-Trump source panel checked into the repo.
- `Validation`: The results compare at least `5.0`, `7.5`, and `10.0`, with `12.5` included if generation remains stable enough to score.
- `Validation`: Any updated strength guidance is backed by checked-in metrics and an explicit panel definition.

## Context preservation

At branch closeout, record in `WORKLOG.md`:

- branch name
- chosen checkpoint
- fixed panel files
- strengths tested
- result artifact names
- top-line overall and whisper-specific takeaways
- whether the guidance for `style_strength > 5.0` changed materially

## Current implementation status

This branch has now been implemented and validated.

Checked-in panel:

- `examples/nontrump_strength_panel/female_1_cremad_1002.wav`
- `examples/nontrump_strength_panel/female_2_cremad_1012.wav`
- `examples/nontrump_strength_panel/male_1_cremad_1003.wav`
- `examples/nontrump_strength_panel/male_2_cremad_1051.wav`

Checked-in outputs:

- `results/eval_nontrump_strength_sweep.csv`
- `results/eval_nontrump_strength_sweep_summary.md`

Top-line result:

- `5.0`: recall `20.8%`, novelty `0.0789`, WER `0.1472`, MOS delta `-0.1312`
- `7.5`: recall `16.7%`, novelty `0.1246`, WER `0.1681`, MOS delta `-0.1701`
- `10.0`: recall `16.7%`, novelty `0.1570`, WER `0.2028`, MOS delta `-0.2287`
- `12.5`: recall `16.7%`, novelty `0.1761`, WER `0.2444`, MOS delta `-0.2119`

Interpretation:

- `5.0` remains the safest global default
- `7.5` is the best stronger-than-default compromise on this panel
- `10.0-12.5` increase novelty further, especially for `whisper` and
  `confused`, but the overall WER/MOS costs are too high to treat them as new
  defaults

## Recommended next branch

- `research/mixed-data-pseudolabel-teacher`

Reason:

- the sweep changed inference guidance, but it did not solve the training-side
  controllability bottleneck
- the next highest-value research move is to improve the mixed-data pseudo-label
  teacher and class-balanced acceptance logic rather than keep pushing the
  strength dial harder
