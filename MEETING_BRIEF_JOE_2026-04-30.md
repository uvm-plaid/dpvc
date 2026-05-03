# Meeting Brief and Follow-up for Professor Joe

**Project:** DPVC / controllable differentially private voice conversion  
**Meeting date:** April 30, 2026  
**Prepared by:** Codex collaboration notes for Steve

## 1. Purpose of this document

This document now serves two roles:
- a concise summary of what we have done so far,
- and a post-meeting record of what Joe confirmed is most important next.

It is intended to make async follow-up easier than reading branch-by-branch notes.

## 2. One-sentence project summary

The project is best understood as a **controllable voice-to-voice speaker generation system** that aims to:
- anonymize speaker identity,
- preserve intelligibility and naturalness,
- and deliberately control speaking style or emotion.

## 3. Post-meeting update: what Joe confirmed

The April 30 call sharpened the next step in a useful way.

Joe's main takeaways were:
- the biggest untried experiment is still the first real training run that combines **CommonVoice + CREMA-D + Expresso together**,
- using a pretrained emotion model to pseudo-label CommonVoice still feels like the most promising way to bootstrap emotion control,
- data mixture and sampling likely matter more right now than another small CommonVoice-only tweak,
- architecture may matter, but it should come after we get the data composition right,
- and the current combined OpenVoice model is already impressive enough that the direction looks worthwhile.

So the next step changed from:
- “keep refining CommonVoice in isolation”

to:
- “run the first sampled mixed-data experiment with pseudo-labeled CommonVoice and explicit mixture control.”

## 4. Current high-level status

The repository is in much better shape than before for both engineering and paper work:
- the OpenVoice path is now the canonical active pipeline,
- the docs and scripts are much more reproducible,
- we now have a novelty metric in addition to emotion / WER / MOS,
- and we have a clearer picture of why CommonVoice pretraining is promising but not yet solving the controllability problem.

## 5. Important terminology

In my internal notes, I used the word "pass" to mean a **focused work phase on its own branch**, with one main question and one validated outcome.

For discussion, it is better to call these **work phases** or **experiment phases**.

## 6. Work phases completed so far

### Phase 1. OpenVoice stabilization and reproducibility
- **Branch:** `feat/openvoice-pipeline-stabilization`
- **Main question:** can we make the active OpenVoice path coherent and rerunnable?
- **What changed:**
  - OpenVoice was made the explicit main path.
  - ControlVC was reframed as a DP baseline / negative result for style control.
  - Docs, interfaces, and scripts were aligned.
  - Batch generation and manifest output were added.
- **Outcome:** success on engineering/reproducibility, not a new scientific finding.

### Phase 2. CommonVoice pretraining pipeline
- **Branch:** `feat/commonvoice-pretrain`
- **Main question:** does broader speaker coverage from CommonVoice help the model?
- **What changed:**
  - built a local CommonVoice extraction path,
  - added CommonVoice reconstruction-only pretraining,
  - finetuned the combined OpenVoice model from that initialization.
- **Main result:**
  - CommonVoice pretraining improved WER strongly,
  - but reduced emotion recall and speaker novelty,
  - pushing the model toward a conservative neutral / baseline-identity basin.

### Phase 3. Speaker novelty metric
- **Branch:** `feat/speaker-novelty-metric`
- **Main question:** are generated voices actually moving away from the source speaker identity?
- **What changed:**
  - added a native OpenVoice embedding-space novelty metric.
- **Main result:**
  - the combined OpenVoice model really does shift speaker identity,
  - some CommonVoice-initialized models collapse back toward baseline identity even when they sound clean.

### Phase 4. Evaluation ablation matrix
- **Branch:** `research/eval-ablations`
- **Main question:** which training condition actually gives the best overall tradeoff?
- **What changed:**
  - compared combined, single-dataset, CommonVoice-initialized, and naive baselines,
  - across emotion, novelty, WER, and MOS.
- **Main result:**
  - the combined OpenVoice model remains the best overall condition,
  - single-dataset and CommonVoice variants are often cleaner or more stable,
  - but they collapse toward neutral emotion and/or baseline identity,
  - and the naive baseline proves that raw speaker movement alone is not the objective.

### Phase 5. CommonVoice finetuning ablation
- **Branch:** `research/commonvoice-finetune-ablation`
- **Main question:** is the CommonVoice collapse just because finetuning was too aggressive?
- **What changed:**
  - tried shorter finetuning,
  - lower learning rate,
  - and encoder/decoder freezing variants.
- **Main result:**
  - some settings recovered a bit of novelty,
  - but none recovered recall beyond `16.7%`.

### Phase 6. CommonVoice objective ablation
- **Branch:** `research/commonvoice-objective-ablation`
- **Main question:** can simple loss reweighting fix the CommonVoice collapse?
- **What changed:**
  - varied reconstruction vs label loss weights and schedules.
- **Main result:**
  - simple scalar reweighting did not beat the best gentler-finetuning recipe.

### Phase 7. CommonVoice richer-objective ablation
- **Branch:** `research/commonvoice-rich-objectives`
- **Main question:** can stronger finetune-time supervision help, beyond scalar weight tuning?
- **What changed:**
  - added style-teacher supervision,
  - added free-dimension anchor losses.
- **Main result:**
  - this preserved stability somewhat,
  - but recall still stayed flat,
  - and it still did not beat the best earlier CommonVoice recovery recipe.

### Phase 8. CommonVoice partial-label / pseudo-label pretraining
- **Branch:** `research/commonvoice-partial-label-pretrain`
- **Main question:** is the collapse already being baked in during CommonVoice pretraining itself, and can weak supervision there help?
- **What changed:**
  - added sparse metadata supervision,
  - added pseudo-style labels from a frozen emotion model,
  - tested metadata-only, pseudo-style-only, and hybrid pretraining.
- **Main result:**
  - metadata-only supervision slightly improved novelty,
  - but still did not recover recall,
  - pseudo-style supervision greatly improved WER,
  - but collapsed identity even harder toward baseline.

## 7. Most important scientific findings so far

### Finding A. ControlVC is useful as a DP baseline, but not the right backend for controllable style
- **Relevant branches:** `feat/controlvc`, `research/eval-ablations`
- **Meaning:** ControlVC can stay in the repo as a baseline, but it is not where the controllable-style story is strongest.

### Finding B. OpenVoice remains the best main model
- **Relevant branch:** `research/eval-ablations`
- **Meaning:** the combined OpenVoice model still gives the best tradeoff across:
  - style control,
  - speaker novelty,
  - intelligibility,
  - and naturalness.

### Finding C. CommonVoice is promising for intelligibility, but not yet for controllable style
- **Relevant branch:** `feat/commonvoice-pretrain`
- **Meaning:** CommonVoice clearly helps WER, but the naive recipe weakens emotion recall and speaker shift.

### Finding D. We now have a novelty metric that confirms real identity movement
- **Relevant branch:** `feat/speaker-novelty-metric`
- **Meaning:** the best model is not just sounding different; it is measurably farther from the source speaker in embedding space.

### Finding E. The CommonVoice collapse has survived multiple increasingly informed fixes
- **Relevant branches:**
  - `research/commonvoice-finetune-ablation`
  - `research/commonvoice-objective-ablation`
  - `research/commonvoice-rich-objectives`
  - `research/commonvoice-partial-label-pretrain`
- **Meaning:** the current bottleneck is probably deeper than:
  - finetuning aggressiveness,
  - scalar loss balancing,
  - first-pass teacher/anchor finetune-time supervision,
  - or first-pass weak-label CommonVoice pretraining.

## 8. Best current reference points

| Condition | Branch | Recall | Novelty | Mean WER | Meaning |
|-----------|--------|--------|---------|----------|---------|
| Combined OpenVoice model | `research/eval-ablations` | `25.8%` | `0.2599` | `0.2353` | Best overall controllable model |
| Raw CommonVoice `cv500` init | `feat/commonvoice-pretrain` | `16.7%` | `0.0369` | `0.0844` | Better intelligibility, but conservative collapse |
| Best CommonVoice recovery so far (`cv500_ft_short_low_lr`) | `research/commonvoice-finetune-ablation` | `16.7%` | `0.0692` | `0.0791` | Best partial novelty recovery on CommonVoice line |
| Best richer-objective CommonVoice variant (`cv500_rich_free_anchor`) | `research/commonvoice-rich-objectives` | `16.7%` | `0.0646` | `0.0724` | Better stability, still no recall recovery |
| Best CommonVoice partial-label pretraining metadata-only weak-label variant (`cv500_pl_meta`) | `research/commonvoice-partial-label-pretrain` | `16.7%` | `0.0570` | `0.0918` | Slight novelty improvement, but not enough |
| CommonVoice partial-label pretraining pseudo-style weak-label variants | `research/commonvoice-partial-label-pretrain` | `16.7%` | `0.0181-0.0190` | `0.0263-0.0285` | Excellent WER, but much stronger identity collapse |

## 9. What is especially interesting right now

### 1. The repo is finally in a paper-usable state
We now have:
- a reproducible active path,
- documented experiments,
- a novelty metric,
- ablations,
- negative results,
- and clearer next-step logic.

### 2. The CommonVoice story is scientifically interesting even though it is not a success yet
This is not just "it failed."

It is more precise than that:
- broader pretraining improves intelligibility,
- but repeatedly washes out the controllable structure,
- and increasingly strong fixes have not recovered that structure yet.

That is a meaningful research result.

### 3. The failure modes are now much better understood
We can now distinguish between:
- content preservation,
- style control,
- and speaker novelty.

Some CommonVoice variants help one axis while harming another. That makes the research story sharper and more defensible.

## 10. What the current bottleneck appears to be

The strongest current interpretation is:

**The CommonVoice pretraining objective is preserving broad speaker/content structure, but it keeps pushing the model into a conservative basin that is hard to escape later when we want controllable style and meaningful speaker shift.**

After the April 30 meeting, the most important added nuance is:

**we still have not tested the first real sampled mixed-data run that combines CommonVoice, CREMA-D, and Expresso together with explicit mixture control.**

That means the main issue is probably not just:
- optimizer settings,
- simple scalar weighting,
- or one more small finetuning tweak,

but we also should not pretend we have already tested the most important combined-data experiment.

## 11. Recommended next step

### Completed after this meeting
- `research/combined-data-pseudolabel-mix`

That branch has now been run and it answered the schedule question narrowly:
- WER improved,
- novelty improved modestly,
- recall stayed fixed at `16.7%`,
- and better schedule choice alone did not escape the conservative
  CommonVoice-like basin.

### Completed follow-up branch
- `research/mixed-data-pseudolabel-quality`

That follow-up tightened pseudo-label filtering and labeled-data protection in
the mixed-data line. It produced the first mixed-data recall bump (`18.2%` in
`mixed_quality_labeled_guarded`), but not a clean overall win because WER and
novelty still regressed versus the best earlier mixed schedules.

### Current best next branch
- `research/nontrump-style-strength-sweep`

### Why this is the best current next move
The latest checked-in evidence now suggests:
- CommonVoice-only refinement was informative but not enough,
- the first sampled mixed-data run was worth doing but not sufficient,
- pseudo-labeled CommonVoice still looks like the best bootstrap path,
- and the highest-value remaining data-side intervention is now better
  pseudo-label filtering plus stronger labeled-data protection.

So the next branch should focus on:
- using `mixed_quality_labeled_guarded` as the current best mixed-data
  checkpoint,
- sweeping style strengths above `5.0` on a documented non-Trump source panel,
- checking whether Joe's qualitative impression about higher strengths,
  especially for whisper, is supported by metrics,
- and updating usage guidance only if the sweep holds up.

## 12. Decisions / questions to discuss with Joe next

### A. Mixed-data experiment design
- Should the first combined-data run use a static balanced mix?
- Or should we explicitly try a curriculum, such as CommonVoice-heavy warmup followed by labeled-data-heavy finish?

### B. Sampling policy
- Is "at least one clip per speaker first" the right CommonVoice heuristic?
- Or should we allow more clips per speaker earlier if that improves stability?

### C. Pseudo-label quality
- Which teacher model or label mapping should we trust most for CommonVoice pseudo labels?
- Can Expresso's richer labels be used to strengthen the mapping before the mixed-data run?

### D. Architecture timing
- Do we want to postpone architecture changes until after the first mixed-data run?
- Or is there one minimal architecture change worth pairing with it?

### E. Strength calibration
- Since Joe found that values above `5.0` can still work well for some styles, should we add a broader non-Trump strength sweep before freezing any user-facing guidance?

## 13. If Joe asks "what have we really learned?"

A concise answer would be:

> We have learned that OpenVoice is still the right main backend for controllable voice-to-voice anonymized generation; that CommonVoice helps intelligibility; but that every CommonVoice-based variant we have tested so far tends to weaken controllability and speaker novelty. We now have enough evaluation and ablation evidence to say that this is a real bottleneck, not just noise. After the April 30 meeting, the clearest next step is not another isolated CommonVoice tweak, but the first sampled mixed-data run using CommonVoice together with CREMA-D and Expresso, with pseudo-labeled CommonVoice and explicit mixture control.

## 14. Useful repo files

If you want to point Joe to concrete artifacts:
- `/Users/steve/UVM-plaid/dp-vc/FINDINGS.md`
- `/Users/steve/UVM-plaid/dp-vc/WORKLOG.md`
- `/Users/steve/UVM-plaid/dp-vc/results/eval_ablation_summary_pass4.csv`
- `/Users/steve/UVM-plaid/dp-vc/results/eval_commonvoice_finetune_summary_pass5.csv`
- `/Users/steve/UVM-plaid/dp-vc/results/eval_commonvoice_objective_summary_pass6.csv`
- `/Users/steve/UVM-plaid/dp-vc/results/eval_commonvoice_rich_objectives_summary_pass7.csv`
- `/Users/steve/UVM-plaid/dp-vc/results/eval_commonvoice_partial_label_summary_pass8.csv`

## 15. Bottom line

If we want the cleanest post-meeting summary:

- **OpenVoice combined is still the best overall model.**
- **ControlVC is useful as a DP baseline, but not for the controllable-style story.**
- **CommonVoice is clearly helping intelligibility, but still hurting controllability and novelty.**
- **We now understand that bottleneck much better than before.**
- **Joe agrees the most important missing experiment is the first sampled mixed-data run with CommonVoice + CREMA-D + Expresso.**
- **The next serious research move should focus on that combined-data experiment, with pseudo-labeled CommonVoice and explicit mixture/schedule control.**
