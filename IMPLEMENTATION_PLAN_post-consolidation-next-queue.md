# Implementation Plan: Post-Consolidation Next Queue

## 1. Why this plan exists

The repository now has a single consolidation PR:

- `integration/research-rollup` -> `main`
- PR: `https://github.com/uvm-plaid/dpvc/pull/3`

That PR solves the collaboration problem Joe raised about branch sprawl, but it
also changes the shape of the next work.

The next queue is no longer just “pick the next experiment.” It now has two
separate goals that should happen in a deliberate order:

1. finish repository consolidation cleanly,
2. then resume research on the strongest remaining bottleneck.

This file records that order so the project stays coherent after context
compaction.

## 2. Execution order

### Step 1. Merge the consolidation PR

**Goal:** make `main` the single readable review surface for the accepted
research line.

Branch / PR:
- branch: `integration/research-rollup`
- PR: `#3` into `main`

Validation:
- PR is mergeable
- PR body clearly states what is included and what is intentionally excluded
- current research branch tip is preserved in the merged history

Notes:
- This step should happen before new research work starts, otherwise the branch
  cleanup benefit gets delayed and new work immediately re-fragments the repo.

### Step 2. Prune merged sequential branches

**Goal:** reduce collaborator overhead after the rollup lands.

Delete the merged sequential research branches from the remote after `main`
contains them.

Recommended branches to delete after merge:
- `feat/openvoice-pipeline-stabilization`
- `feat/commonvoice-pretrain`
- `feat/speaker-novelty-metric`
- `research/eval-ablations`
- `research/commonvoice-finetune-ablation`
- `research/commonvoice-objective-ablation`
- `research/commonvoice-rich-objectives`
- `research/commonvoice-partial-label-pretrain`
- `research/combined-data-pseudolabel-mix`
- `research/mixed-data-pseudolabel-quality`
- `research/nontrump-style-strength-sweep`

Keep unmerged side branches for now:
- `feat/controlvc`
- `feat/openvoice-expresso`
- `feat/f0-style-control`

Reason:
- those branches are not part of the accepted linear research line in PR `#3`
- they should be evaluated separately rather than silently folded into `main`

Validation:
- `main` contains the merged integration branch
- remote branch list is materially simpler for Joe
- `WORKLOG.md` / `README.md` continue to point to `FINDINGS.md` instead of old
  branch-by-branch review patterns

### Step 3. Start the next research branch

**Branch:** `research/mixed-data-pseudolabel-teacher`

**Goal:** improve the mixed-data pseudo-label teacher and class-balanced
acceptance logic, because the best current mixed-data condition only reached
`18.2%` recall and inference-time strength changes did not solve the training
supervision bottleneck.

Dedicated technical plan:
- `IMPLEMENTATION_PLAN_mixed-data-pseudolabel-teacher.md`

Validation:
- new branch starts from merged `main`, not from a stale pre-rollup branch
- the branch isolates training-supervision changes rather than mixing in
  unrelated repo cleanup

### Step 4. Add a short metric-reading guide for Joe

**Goal:** reduce interpretation overhead when Joe reviews results.

Add a concise explanation of:
- emotion recall / emo_sim
- novelty
- WER
- MOS
- how to read tradeoffs across them

Recommended doc locations:
- primary: `FINDINGS.md`
- optional short version: `results/README.md`

This should explain:
- what “good” means for each metric
- why novelty alone is insufficient
- why a lower WER / better MOS result can still be a style-control failure
- how to read the collapse taxonomy

Validation:
- a collaborator can interpret the ablation tables without digging through old
  branches or raw CSVs first

### Step 5. Extend the non-Trump strength sweep

**Goal:** decide whether the current `style_strength` guidance generalizes.

The first non-Trump sweep showed:
- `5.0` remains the safest default
- `7.5` is a defensible stronger setting for `whisper` / `confused`
- `10.0-12.5` are high-novelty, higher-cost settings

The next sweep should:
- expand beyond the current 4-speaker panel
- compare the `combined` checkpoint against `mixed_quality_labeled_guarded`
- test whether the whisper/confused pattern is stable across a broader panel

Validation:
- updated guidance is based on a broader panel than the initial 4-speaker run
- the repo can justify either keeping the current default or adding style
  presets such as `default`, `strong-whisper`, and `strong-confused`

### Step 6. Finish Joe-facing reproducibility cleanup

**Goal:** make the repo easier for Joe to rerun end to end.

Required tasks:
- share Expresso download instructions with Joe
- ensure extraction + training + inference can be rerun from scratch
- pin dependencies / constraints for the tested OpenVoice path
- keep `FINDINGS.md` as the single async review document

Recommended implementation outputs:
- one checked-in dependency constraints file or lockfile for the tested stack
- one smoke-tested README path from install -> extraction -> train -> infer ->
  eval
- one short reproducibility checklist for Joe

Validation:
- Joe can rerun the active OpenVoice pipeline without needing branch-specific
  tribal knowledge

## 3. Why this order is best

This order avoids two common problems:

1. starting a new research branch before consolidation lands,
2. doing user-facing reproducibility polish before the next real research
   bottleneck is scoped.

The queue above keeps the project synchronized across:
- collaboration hygiene,
- research progress,
- and paper-readiness.

## 4. Future upgrades to preserve

### Mixed-data follow-up ideas
- compare one-clip-per-speaker vs two-clips-per-speaker CommonVoice sampling
  inside the mixed-data setup, because Joe's speaker-breadth heuristic still
  needs a direct empirical check
- enrich the Expresso label mapping further, because Expresso is still one of
  the best sources of labeled emotion structure
- add stronger teacher-space or prototype-space supervision if the next teacher
  branch still fails to move recall enough

### Evaluation / paper upgrades
- add confidence intervals or repeated-seed uncertainty before freezing paper
  tables
- add condition-by-style plots and a manual collapse-audit sheet
- cross-check novelty with an independent speaker verifier / EER pipeline

### Inference / demo upgrades
- extend the non-Trump sweep into style presets if the larger panel supports it
- add a broader novelty-vs-strength curve once the next mixed-data teacher
  branch stabilizes
- revisit the demo use cases after the training bottleneck improves
