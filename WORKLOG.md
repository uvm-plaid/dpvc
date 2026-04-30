# Controllable DP Voice Conversion — Work Log

**Last updated:** 2026-04-30
**Branches:** `feat/controlvc`, `feat/openvoice-expresso`, `feat/f0-style-control`, `feat/cremad-experiments`, `feat/openvoice-pipeline-stabilization`, `feat/commonvoice-pretrain`, `feat/speaker-novelty-metric`, `research/eval-ablations`, `research/commonvoice-finetune-ablation`, `research/commonvoice-objective-ablation`, `research/commonvoice-rich-objectives`, `research/commonvoice-partial-label-pretrain`, `research/combined-data-pseudolabel-mix`
**Author:** Stephen Oladele (with Claude, and Joe Near's upstream work)

---

## 0. Roadmap

Priority tags:
- `NOW` = immediate next work; unblocks reproducibility or addresses the main research bottleneck
- `SOON` = important after the `NOW` items are stable
- `LATER` = downstream, stretch, or productization work

### Phase 0.5: Consolidate Active Path
- [x] `[NOW]` Consolidate the active controllable pipeline around OpenVoice; treat ControlVC as a useful DP baseline and negative result for style control

### Phase 1: Core Controllability (prove it works)
- [x] End-to-end pipeline: extract → train VAE → infer → intelligible speech
- [x] Demonstrate style control with OpenVoice (not ControlVC — Joe confirmed, April 9)
- [x] Speaker diversity: train on 91+ speakers (CREMA-D) so VAE learns style, not speaker
- [x] Combined dataset (CREMA-D + Expresso): 9 styles, 94 speakers, all perceptually distinct
- [x] Acoustic analysis confirming style differences (F0, energy, spectral centroid)
- [x] Test style control survives DP noise (noise_level 0–1.0, styles persist at 0.1, degrade gracefully)
- [x] Test on diverse source speakers (5 speakers: brightness consistent 7/9 styles, F0 inconsistent, 9% collapse rate)
- [x] Per-speaker style evaluation: brightness is the reliable cross-speaker metric, not F0

### Phase 1.5: Scale Up Speaker Diversity (Joe's suggestion, April 16)
- [x] `[NOW]` Extract OpenVoice embeddings from CommonVoice (20k+ speakers, no style labels)
- [x] `[NOW]` Pre-train VAE on CommonVoice (reconstruction loss only)
- [x] `[NOW]` Finetune on CREMA-D + Expresso (style labels, label loss)
- [x] `[NOW]` Compare with current combined-only VAE: does pre-training reduce collapse rate?
- [ ] `[SOON]` Scale the validated `cv500` CommonVoice recipe to a much larger local slice or a full English mirror before drawing a final conclusion about pre-training
- [x] `[SOON]` Test gentler finetuning from the CommonVoice init checkpoint (fewer finetune epochs, lower LR, or partial-freeze) to see whether the `cv500` neutral-collapse failure is an over-regularization problem rather than a dead-end. **Pass 5 result:** simple recipe changes recover only small novelty gains (`0.0369 -> 0.0692` best case) and do not recover the combined model's recall/novelty tradeoff
- [ ] `[SOON]` Scale the best Pass 5 recipe (`cv500_ft_short_low_lr`) to a larger local CommonVoice slice and test whether its partial novelty recovery survives at higher speaker counts
- [x] `[SOON]` Test layer-granular freezing or loss-weight schedules instead of full encoder/decoder freezes; Pass 6 result: simple CommonVoice loss reweighting (`label2`, `label4`, label-ramp, reduced-reconstruction) still leaves recall flat at `16.7%` and does not beat `cv500_ft_short_low_lr`
- [ ] `[SOON]` Compare reconstruction-only CommonVoice pretraining against partially labeled or multi-objective pretraining, because Passes 6-7 suggest finetune-time reweighting and teacher/anchor finetuning are still too weak when the CommonVoice stage itself remains purely reconstruction-driven
- [x] `[SOON]` Test richer CommonVoice adaptation objectives (teacher or latent anchoring, curriculum label emphasis, or partial-label pretraining) because Pass 6 shows simple loss-weight ramps do not restore recall or beat the best Pass 5 novelty recovery. **Pass 7 result:** teacher-style distillation and free-dim anchoring preserve WER/MOS better than some earlier variants, but recall stays flat at `16.7%` and none beats `cv500_ft_short_low_lr` on novelty or identity collapse
- [x] `[SOON]` Test richer CommonVoice adaptation objectives that use partial labels, pseudo-labels, or a better pretraining objective on CommonVoice itself; Pass 8 result: metadata-only weak supervision gives modest novelty recovery (`0.0570`) but no recall gain, while pseudo-style supervision drives WER down sharply (`0.0263-0.0285`) at the cost of even lower novelty (`0.0190-0.0181`) and much higher identity collapse (`85-90`)
- [ ] `[SOON]` Compare teacher-style distillation against prototype-style or curriculum supervision in the labeled style subspace, because Pass 7's latent teacher/anchor losses improved stability more than control
- [ ] `[SOON]` Improve CommonVoice pseudo-label quality and calibration (better teacher, confidence filtering, or class-balanced acceptance), because Pass 8 suggests the current pseudo-style labels over-regularize the model into a conservative neutral / baseline-identity basin
- [ ] `[SOON]` Test prototype-style or teacher-embedding targets during CommonVoice pretraining itself, not just combined finetuning, because Pass 8's label-space pseudo supervision preserved intelligibility much more than controllability
- [ ] `[SOON]` Compare metadata-only weak supervision against stronger free-dim supervision (for example: age/gender/accent + auxiliary speaker-structure constraints), because Pass 8 suggests metadata shapes novelty a little but does not recover recall
- [ ] `[SOON]` Add per-style recovery plots for the CommonVoice finetune and objective variants; Passes 5-7 suggest novelty returns first for a few conservative styles, not as broad emotion recovery
- [ ] `[SOON]` Add age/gender control dims using CommonVoice metadata (dims 9-10)
- [ ] `[SOON]` Test orthogonality: does pushing emotion dims shift perceived age/gender?
- [ ] `[SOON]` Test whether CommonVoice-broad pretraining preserves age/gender control more easily than emotion control; Pass 5 suggests different attribute families may survive broad speaker priors differently
- [ ] `[SOON]` **Open question (Joe, April 16):** Can we train all knobs at once when labels come from different datasets? CommonVoice has age/gender, CREMA-D has emotion — each stage only trains a subset of latent dims

### Phase 1.6: Mixed-Data Bootstrap (Joe's April 30 call)
- [ ] `[NOW]` Run the first sampled mixed-data experiment that trains on **CommonVoice + CREMA-D + Expresso together**, because Joe confirmed on April 30 that this is still the biggest untried experiment and the most important missing check after the CommonVoice-only follow-up passes
- [ ] `[NOW]` Build the mixed-data corpus around **speaker breadth first**, not raw CommonVoice clip count, because Joe's current heuristic is that getting at least one clip per CommonVoice speaker may matter more than maximizing total volume
- [ ] `[NOW]` Compare at least three mixture schedules for the first mixed-data run: a static balanced mix, a CommonVoice-heavy warmup followed by mixed training, and a labeled-data-heavy finish, because Joe explicitly flagged the schedule as an open empirical question
- [ ] `[NOW]` Protect the small labeled datasets with quotas or upsampling in the mixed-data run, because Joe warned that a naive full merge may simply behave like CommonVoice training and wash out the CREMA-D / Expresso effect
- [ ] `[SOON]` Improve CommonVoice pseudo-label quality and calibration **as part of the mixed-data setup**, not only as a CommonVoice-only refinement, because Joe endorsed pseudo-labeled CommonVoice + combined-data training as the most promising bootstrap path
- [ ] `[SOON]` Clean up and enrich the Expresso label mapping before the mixed-data run, because Joe agreed the richer Expresso label space is one of the best ways to inject emotion structure into the broader CommonVoice speaker prior
- [ ] `[SOON]` Run style-strength sweeps above `5.0` on representative non-Trump speakers, because Joe qualitatively found that higher strengths can work well, especially for whisper, so `5.0` should not be treated as a hard ceiling
- [ ] `[SOON]` Add a concise "how to read the metrics" guide for Joe covering emotion recall / emo_sim, novelty, WER, and MOS, because he explicitly said the branch and metric layout is hard to interpret quickly

### Phase 2: Evaluation (Joe: emotion eval is #1 priority)
- [x] **Research TTS controllability evaluation metrics** — settled on EmoVoice pipeline (arxiv 2504.12867, Joe's suggestion): emotion2vec Recall Rate + emo_sim (primary), UTMOS (naturalness), WER (intelligibility)
- [x] Build emotion evaluation pipeline: `examples/eval_emotion.py` runs emotion2vec_plus_large on a directory of generated audio and writes a CSV with per-file Recall Rate and emo_sim vs. same-speaker baseline (April 17)
- [x] Run emotion eval on full 258-file diverse-speaker corpus: **36/162 = 22.2% overall recall** at strength=5.0; per-style neutral 67%, anger 30%, sad 26%, disgust 11%, fear 0%, happy 0%. Whisper has the largest emo_sim deviation (0.875 mean, 0.616 min) — emo_sim validated as a secondary signal for styles with no emotion2vec counterpart
- [x] Word error rate via Whisper: `examples/eval_wer.py` runs Whisper `base` on a directory and computes per-file WER against the same-speaker baseline (April 17). **Result: 6 of 9 styles have median WER ≤ 0.20; whisper is the only style with systemic loss (mean 0.356). Style control is orthogonal to the content channel.**
- [x] Predicted MOS via torchaudio SQUIM_SUBJECTIVE (UTMOS substitute — the `utmos` pip package conflicts with our fairseq monkey-patches). `examples/eval_mos.py`, runs on directory, outputs MOS + delta-vs-baseline per file. **Result: baseline MOS 4.05; 6 of 9 styles stay within 0.12 MOS of baseline; whisper/confused/anger degrade most. emo_sim + WER + MOS converge on the same three hardest styles — cross-metric triangulation validates the evaluation pipeline.** (April 17)
- [x] Speaker novelty metric — `examples/eval_novelty.py` computes source-vs-generated cosine similarity in native OpenVoice speaker-embedding space, with delta-vs-baseline conversion. **Pass 3 result (11-speaker validation corpus):** combined-only model mean novelty gain vs baseline = `0.2599`; `cv500` CommonVoice checkpoint mean novelty gain vs baseline = `0.0369`, confirming the CommonVoice neutral-collapse result on a second axis
- [ ] `[SOON]` Speaker verification / privacy metric (secondary — Joe: "not sure we want to focus on privacy as the main thing")
- [ ] `[SOON]` Calibrate novelty thresholds from source-vs-baseline and source-vs-style distributions, so the metric can support a more explicit "novel enough" claim instead of raw cosine values alone
- [ ] `[SOON]` Cross-check the novelty metric with an independent speaker encoder / EER pipeline, not just OpenVoice's native embedding space
- [ ] `[SOON]` Run novelty-vs-noise and novelty-vs-style-strength sweeps once the speaker novelty metric is stable, to add a privacy/utility-style novelty curve
- [x] Ablation study: CREMA-D only vs. Expresso only vs. combined, extended with the validation-scale CommonVoice `cv500` init and a naive baseline. **Pass 4 result:** the combined model remains the best tradeoff across controllability, novelty, intelligibility, and naturalness (`results/eval_ablation_summary_pass4.csv`)
- [x] Compare with naive baseline: random unlabeled latent control without style supervision. **Pass 4 result:** it creates more novelty than the combined model (`0.4708` vs. `0.2599`) but with worse recall (`18.2%`) and much worse MOS delta (`-0.6453`)
- [x] Write up negative result: ControlVC D_VECTOR doesn't encode style (separability ratio 0.88) — now explicitly carried by Finding 1 and the Pass 4 paper-strengthening pass
- [x] Collapse taxonomy across ablations: content collapse (`WER >= 0.8`), style collapse to neutral, identity collapse to baseline, and mixed collapse. **Pass 4 result:** `cv500`, `cremad_only`, and `expresso_only` are dominated by identity/style collapse; the combined model has fewer but more diverse failures
- [ ] `[LATER]` Investigate F0-based re-identification attack: can F0 alone re-identify speakers after embedding anonymization?
- [ ] `[SOON]` Address collapse issue: 9% of speaker-style combinations produce unintelligible output (Joe says expected, no perfect fix needed, but worth tracking)
- [ ] `[SOON]` Add bootstrap confidence intervals or repeated-seed uncertainty to the Pass 4 ablation matrix before freezing paper tables
- [ ] `[SOON]` Add condition-by-style plots for emotion recall, novelty gain, WER, and MOS so the paper can show where each condition fails, not just overall means
- [ ] `[SOON]` Add a manual collapse-audit sheet for rows where metrics disagree (for example: high novelty but low emotional alignment, or good WER but strong neutral collapse)
- [ ] `[SOON]` Cross-check the ablation matrix with an independent speaker verifier / EER pipeline, not just OpenVoice's native embedding space

### Phase 2.5: Framing-driven tasks (from Joe's April 16 evening message)

Joe clarified that our problem is **controllable speaker generation for voice-to-voice** — a more general problem than VoicePrivacy (preserves emotion) or TTS (generates from text). The VAE enables multiple use cases; we've been showcasing only one.

- [ ] `[SOON]` **Demo use case #2: emotion change without identity change** — modify style latent dims while keeping the rest of the embedding fixed. Same-sounding person, different mood. Needs a new inference mode in `openvoice_infer_controllable.py` (or a new script) that takes a *source* speaker and only perturbs style dims instead of re-sampling the whole latent.
- [ ] `[SOON]` **Demo use case #3: fully random speaker with style control** — sample the VAE from prior (no source speaker reference) and apply style. Produces a brand-new speaker targeting a specific emotion.
- [ ] `[SOON]` **Demo use case #4: random speaker near an existing one** — sample in a neighborhood of the source speaker's latent code (small sigma) rather than from the full prior.
- [ ] `[LATER]` **Literature search for SOTA comparison** — what systems claim controllable speaker generation for voice-to-voice? Need to know what we're being compared against for Joe's "might be SOTA" claim. *Blocked: waiting for Joe's reply on April 17 message.*

### Phase 3: Reproducibility & Collaboration (for Joe)
- [x] Commit training script + inference CLI + README (done April 16, during call)
- [x] Add evaluation script (`eval_emotion.py`) with README docs — reproducible emotion2vec + emo_sim pipeline (April 17)
- [x] Add WER evaluation script (`eval_wer.py`) — Whisper + jiwer, drift-from-baseline mode by default, fixed-reference mode via `--reference-text` (April 17)
- [x] Add MOS evaluation script (`eval_mos.py`) — torchaudio SQUIM_SUBJECTIVE, baseline-as-reference mode by default, fixed-reference mode via `--reference` (April 17)
- [ ] `[NOW]` Share Expresso download instructions with Joe
- [ ] `[NOW]` Ensure Joe can run extraction + training + inference from scratch
- [ ] `[NOW]` Pin dependencies (fairseq compat, OpenVoice install steps)
- [ ] `[NOW]` Keep `FINDINGS.md` as the single async review document for Joe and point branch-heavy result dumps back to it, because the April 30 meeting confirmed that direct branch-by-branch review is slowing interpretation
- [ ] `[SOON]` Add a manifest-driven multi-metric eval helper so emotion, WER, novelty, and future privacy metrics can be rerun together on the same corpus without ad hoc command reconstruction
- [ ] `[SOON]` Add a reusable experiment runner that records checkpoint -> corpus -> metrics -> summary for CommonVoice follow-up passes, so future finetune and objective sweeps are less manual than Passes 5-7
- [ ] `[SOON]` Add a checked-in OpenVoice constraints file or lockfile matching the tested `.venv` stack, so setup is copy-paste reproducible beyond the README version notes
- [ ] `[SOON]` Add an automated smoke test for `openvoice_infer_controllable.py --source-dir` + manifest generation against cached local checkpoints

### Phase 4: Application / Demo
- [ ] `[LATER]` Design interactive demo: upload voice → choose style → choose privacy level → download anonymized output
- [ ] `[LATER]` Web UI or CLI tool that non-researchers can use
- [ ] `[LATER]` Real-time voice conversion mode (stretch goal)
- [ ] `[LATER]` Package as installable tool (pip install dpvc or similar)

### Key Insight from Joe (April 9 call)
> "If I want to invent a completely hypothetical speaker who has some properties, everybody's bad at that. And in some ways, that's what we're trying to do. If we can make this work, that's a big deal."

The paper contribution is **controllable speaker profile synthesis with formal privacy guarantees** — something the speech community has struggled with. Even without the privacy angle, demonstrating controllability over speaker profiles is itself a significant result.

### 0.6 Pass 1 Closeout (April 28, branch `feat/openvoice-pipeline-stabilization`)

- Reframed the public docs so **OpenVoice is the active controllable pipeline** and **ControlVC is the DP baseline / negative-result path for style control**
- Added a temporary `vae_checkpoint_path` compatibility alias back to `dpvc.Anonymizer`, while keeping `vae_config` as the canonical interface for new docs and scripts
- Extended `examples/openvoice_infer_controllable.py` with:
  - `--source-dir` batch generation
  - default JSONL manifest output (`generation_manifest.jsonl` for batch runs)
  - manifest rows containing source path, output path, style, style strength, noise level, seed, checkpoint, and latent dims
- Updated `README.md`, `examples/README.md`, `results/README.md`, `docs/using.md`, `docs/training.md`, and ControlVC docs to match the real interfaces and current project framing
- Fixed packaging drift for the active OpenVoice path: `requests` is now a base dependency, `pandas` is included in the `expresso` extra

**Verification**
- `./.venv/bin/python examples/openvoice_infer_controllable.py --help` shows the new `--source-dir` and `--manifest` interface
- `./.venv/bin/python` compatibility smoke test confirmed both `Anonymizer(..., vae_checkpoint_path=...)` and `Anonymizer(..., vae_config=...)` load the same VAE class successfully
- Real OpenVoice run completed on a one-file source directory: baseline + 9 style outputs written to `/tmp/dpvc_pass1_out/`, with a 10-row manifest at `/tmp/dpvc_pass1_out/generation_manifest.jsonl`

**FINDINGS.md review**
- Reviewed after Pass 1. No new paper-facing scientific finding was added because this pass stabilized interfaces and reproducibility rather than producing new experimental evidence.

### 0.7 Pass 2 Closeout (April 28, branch `feat/commonvoice-pretrain`)

- Added a local-corpus Common Voice extraction CLI: `examples/openvoice_extract_commonvoice.py`
  - expects `<corpus-path>/validated.tsv` + `clips/`
  - deterministic seed `42`
  - optional speaker and clip caps
  - recursive clip discovery, missing/unreadable-file accounting, and checkpointed extraction
- Added `scripts/prepare_commonvoice_subset.py` so downloaded Common Voice shards can be turned into a local subset without hardcoded paths
- Added reconstruction-only Common Voice pretraining via `examples/openvoice_pretrain_vae_commonvoice.py`
- Extended `examples/openvoice_train_vae_combined.py` with `--init-checkpoint` so the combined controllable VAE can finetune from Common Voice pretrained weights
- Added `scripts/compare_emotion_eval.py` to compare baseline vs candidate emotion-eval CSVs directly
- Built and validated a local English Common Voice subset at `/Users/steve/datasets/cv-corpus-21.0-2025-03-14-subset/en`
  - local clips matched: `28,116`
  - local speakers available: `6,795`
  - validation-scale training subset used for this pass: `500` speakers / `1,202` clips (`cv500`)
- Ran the full `cv500` comparison end to end:
  - extracted `embeddings/openvoice_commonvoice_cv500_emb.pt`
  - pretrained `embeddings/openvoice_vae_commonvoice_cv500.pt`
  - finetuned `embeddings/openvoice_vae_combined_cv500.pt`
  - generated `output/pass2_cv500_eval/` + manifest
  - evaluated `results/eval_emotion_pass2_cv500.csv` and `results/eval_wer_pass2_cv500.csv`

**Validation**
- [x] A small local CommonVoice subset can be extracted and pretrained without hardcoded paths
  - validated with the local `cv500` subset: `500` speakers, `1,202` embeddings, `0` missing files, `0` unreadable files
- [x] Finetuning from the pretrained checkpoint is supported by the existing combined-training path
  - validated by training `embeddings/openvoice_vae_combined_cv500.pt` from `--init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt`
- [x] The comparison explicitly answers whether pre-training reduces collapse rate and improves recall/generalization
  - answer on the validation-scale `cv500` run: **no**
  - emotion recall dropped from `25.8%` to `16.7%` (`-9.1 pts`)
  - neutral predictions rose from `70/110` files to `109/110` files
  - WER improved from `0.235` mean to `0.084` mean across all `99` scored style rows
  - interpretation: reconstruction-only Common Voice pretraining improved intelligibility/content preservation but over-regularized the model toward baseline/neutral speech and weakened controllable style expression

**FINDINGS.md review**
- Updated after Pass 2 with a new paper-facing result: validation-scale Common Voice pretraining (`cv500`) improved WER substantially but caused near-total neutral collapse in emotion classification, so the current naive pretraining recipe is not yet a win for controllable style generalization.

### 0.8 Pass 3 Closeout (April 28, branch `feat/speaker-novelty-metric`)

- Added `examples/eval_novelty.py`
  - manifest-driven corpus evaluation via `generation_manifest.jsonl`
  - one-off `--source` / `--generated` mode for spot checks
  - native OpenVoice speaker-embedding cosine similarity + cosine distance
  - same-speaker baseline conversion delta when a baseline row exists in the manifest
- Ran novelty evaluation on both validation corpora:
  - `results/eval_novelty_pass2_combined.csv`
  - `results/eval_novelty_pass2_cv500.csv`
- Updated docs so novelty is part of the reproducible evaluation stack alongside emotion, WER, and MOS

**Validation**
- [x] The novelty CLI runs on manifest-driven corpora without manual file mapping
  - validated on `output/pass2_combined_eval/generation_manifest.jsonl` and `output/pass2_cv500_eval/generation_manifest.jsonl`
- [x] The novelty CSV is reproducible and includes source, generated, style, and similarity fields needed for analysis
  - validated by writing both checked-in CSV artifacts plus a one-off smoke CSV from explicit `--source` / `--generated`
- [x] The summary clearly answers whether generated outputs differ from the source speaker, and whether that differs across styles or checkpoints
  - combined-only checkpoint: mean novelty gain vs baseline = `0.2599`
  - `cv500` CommonVoice checkpoint: mean novelty gain vs baseline = `0.0369`
  - strongest novelty on combined-only: `whisper` (`0.6561`) and `confused` (`0.5589`)
  - interpretation: the native novelty metric confirms that the `cv500` checkpoint's style control largely collapses back toward baseline voice identity

**FINDINGS.md review**
- Updated after Pass 3 with a new paper-facing result: the novelty metric confirms that the combined-only model creates genuinely shifted speakers relative to the source, while the `cv500` CommonVoice checkpoint suppresses that shift and aligns with the neutral-collapse story from Pass 2.

### 0.9 Pass 4 Closeout (April 28, branch `research/eval-ablations`)

- Added `scripts/prepare_ablation_embeddings.py`
  - prepares `cremad_only` and `expresso_only` ablation datasets directly in the unified `label_*` format expected by the OpenVoice trainer
  - uses saved Expresso row ids to realign parquet metadata with the extracted embedding file, so the ablation dataset is reproducible even when extraction skipped rows
- Added `scripts/run_ablation_inference.py`
  - generates condition-specific evaluation corpora without overloading the main public inference CLI
  - supports the single-dataset style maps and the naive free-dimension baseline
- Added `scripts/summarize_ablation_results.py`
  - reads `results/eval_*_pass4_*.csv`
  - writes `results/eval_ablation_summary_pass4.csv`
  - writes `results/eval_ablation_collapse_pass4.csv`
- Trained the two missing single-dataset ablation checkpoints:
  - `embeddings/openvoice_vae_cremad_ablation.pt`
  - `embeddings/openvoice_vae_expresso_ablation.pt`
- Generated the three new Pass 4 evaluation corpora:
  - `output/pass4_cremad_only_eval/`
  - `output/pass4_expresso_only_eval/`
  - `output/pass4_naive_noise_baseline_eval/`
- Reused the already validated Pass 2 generation corpora for the unchanged `combined` and `commonvoice_cv500_init` conditions, then copied their existing emotion/WER/novelty CSVs into the Pass 4 naming scheme and added the missing MOS layer

**Pass 4 condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cremad_only`
- `expresso_only`
- `naive_noise_baseline`

**Top-line result (`results/eval_ablation_summary_pass4.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cremad_only`: recall `16.7%`, novelty gain `0.0158`, mean WER `0.0534`, mean MOS delta `+0.0059`
- `expresso_only`: recall `36.4%` on `33` emotional rows only, novelty gain `-0.0008`, mean WER `0.0549`, mean MOS delta `-0.0120`
- `naive_noise_baseline`: recall `18.2%`, novelty gain `0.4708`, mean WER `0.1579`, mean MOS delta `-0.6453`

**Interpretation**
- The combined checkpoint remains the best overall tradeoff. It is the only condition with both non-trivial controllability and non-trivial identity shift while keeping WER and MOS within a survivable range.
- The `cv500` CommonVoice init, `cremad_only`, and `expresso_only` conditions all expose the same research trap from different angles: they can sound stable and transcribe well while still collapsing back toward baseline identity and/or neutral emotion.
- The naive baseline proves that **novelty alone is not the paper objective**. Random unlabeled latent pushes can create larger identity shifts than the combined model, but they do not create clean target emotion control and they degrade naturalness badly.

**Validation**
- [x] Every ablation condition has a reproducible command path and named artifacts
  - generation: `scripts/run_ablation_inference.py`
  - aggregation: `scripts/summarize_ablation_results.py`
  - artifact bundle: `results/eval_*_pass4_*.csv`
- [x] The summary table clearly shows which condition best balances controllability, novelty, intelligibility, and naturalness
  - `results/eval_ablation_summary_pass4.csv`
- [x] The naive baseline is concrete, reproducible, and fair
  - deterministic random control vectors are applied only to the six free latent dims (`9-14`) and are L2-matched to the style-control strength
- [x] The collapse summary distinguishes failure modes rather than flattening them into one bucket
  - `results/eval_ablation_collapse_pass4.csv`
- [x] The ControlVC negative-result writeup is backed by concrete evidence already present in the repo
  - Finding 1 plus the explicit Pass 4 matrix framing

**FINDINGS.md review**
- Updated after Pass 4 with a new paper-facing result: the combined model remains the best overall tradeoff in the ablation matrix, and the naive baseline shows that raw novelty is not enough without target alignment and naturalness.

### 0.10 Pass 5 Closeout (April 28, branch `research/commonvoice-finetune-ablation`)

- Extended `examples/openvoice_train_vae_combined.py` with coarse finetune controls:
  - `--freeze-encoder`
  - `--freeze-decoder`
  - explicit reporting of trainable vs total parameter counts
- Extended `scripts/run_ablation_inference.py` with the new CommonVoice finetune conditions:
  - `cv500_ft_short`
  - `cv500_ft_low_lr`
  - `cv500_ft_short_low_lr`
  - `cv500_ft_freeze_decoder`
  - `cv500_ft_freeze_encoder`
- Added `scripts/summarize_commonvoice_finetune_ablation.py`
  - reads `results/eval_*_pass5_*.csv`
  - writes `results/eval_commonvoice_finetune_summary_pass5.csv`
  - writes `results/eval_commonvoice_finetune_collapse_pass5.csv`
- Trained the five new finetune variants from `embeddings/openvoice_vae_commonvoice_cv500.pt` against the unchanged combined embedding set:
  - `embeddings/openvoice_vae_combined_cv500_ft_short.pt` (`1000` epochs, `1e-6`)
  - `embeddings/openvoice_vae_combined_cv500_ft_low_lr.pt` (`3000` epochs, `3e-7`)
  - `embeddings/openvoice_vae_combined_cv500_ft_short_low_lr.pt` (`1000` epochs, `3e-7`)
  - `embeddings/openvoice_vae_combined_cv500_ft_freeze_decoder.pt` (`3000` epochs, `1e-6`, decoder frozen)
  - `embeddings/openvoice_vae_combined_cv500_ft_freeze_encoder.pt` (`3000` epochs, `1e-6`, encoder frozen)
- Generated the five matched Pass 5 evaluation corpora:
  - `output/pass5_cv500_ft_short_eval/`
  - `output/pass5_cv500_ft_low_lr_eval/`
  - `output/pass5_cv500_ft_short_low_lr_eval/`
  - `output/pass5_cv500_ft_freeze_decoder_eval/`
  - `output/pass5_cv500_ft_freeze_encoder_eval/`
- Reused the already validated `combined` and `commonvoice_cv500_init` result CSVs from Pass 4 by copying them into the Pass 5 naming scheme, so the changed variable stayed strictly on finetuning policy

**Pass 5 condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short`
- `cv500_ft_low_lr`
- `cv500_ft_short_low_lr`
- `cv500_ft_freeze_decoder`
- `cv500_ft_freeze_encoder`

**Top-line result (`results/eval_commonvoice_finetune_summary_pass5.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short`: recall `16.7%`, novelty gain `0.0558`, mean WER `0.0390`, mean MOS delta `-0.0108`
- `cv500_ft_low_lr`: recall `16.7%`, novelty gain `0.0495`, mean WER `0.0340`, mean MOS delta `-0.0142`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_ft_freeze_decoder`: recall `16.7%`, novelty gain `0.0192`, mean WER `0.0330`, mean MOS delta `-0.0154`
- `cv500_ft_freeze_encoder`: recall `18.2%`, novelty gain `0.0594`, mean WER `0.0905`, mean MOS delta `-0.1261`

**Interpretation**
- None of the simple finetune-policy changes restored the combined model's tradeoff. The best novelty recovery came from `cv500_ft_short_low_lr`, but it still stayed far below the combined model on both novelty (`0.0692` vs. `0.2599`) and recall (`16.7%` vs. `25.8%`).
- `cv500_ft_short_low_lr` was the strongest partial recovery recipe:
  - best novelty among the CommonVoice finetune variants
  - lowest identity-collapse count among the variants (`44`, down from `72` on the original `cv500`)
  - but no recall improvement over the original `cv500`
- `cv500_ft_freeze_encoder` was the only variant to improve recall at all (`18.2%` vs. `16.7%`), but it gave back most of the CommonVoice stability story by worsening MOS delta to `-0.1261`
- `cv500_ft_freeze_decoder` was the clearest negative result in the sweep: it reduced novelty below the original `cv500` (`0.0192`) and increased identity-collapse count to `88`
- All five variants preserved the same basic Pass 2 failure shape: very low content collapse, but dominant collapse back toward `neutral` emotion and/or baseline identity

**Validation**
- [x] New finetuning controls are reproducible from the checked-in training interface
  - validated via `examples/openvoice_train_vae_combined.py --freeze-encoder/--freeze-decoder`
  - each run logged its checkpoint source, freeze settings, and trainable parameter count
- [x] Every finetuning variant has a named checkpoint and matched evaluation corpus
  - five checkpoints under `embeddings/`
  - five 110-row corpora + manifests under `output/pass5_*`
- [x] The comparison explicitly answers whether gentler finetuning preserves CommonVoice's WER/MOS gains while recovering novelty and emotion control
  - answer: **not with these simple recipe changes**
  - novelty improves modestly for `cv500_ft_short`, `cv500_ft_low_lr`, `cv500_ft_short_low_lr`, and `cv500_ft_freeze_encoder`
  - recall remains flat at `16.7%` for four of five variants and only rises to `18.2%` once, still well below the combined model's `25.8%`
- [x] The pass isolates finetuning strategy as the changed variable rather than mixing in new datasets or new metrics
  - same CommonVoice pretrained init checkpoint
  - same combined fine-tuning embeddings
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack

**FINDINGS.md review**
- Updated after Pass 5 with a new paper-facing result: simple gentler finetuning helps only marginally and does not fix the CommonVoice collapse, which narrows the next research step to better objectives or larger-scale training rather than just lighter fine-tuning.

### 0.11 Pass 6 Closeout (April 28, branch `research/commonvoice-objective-ablation`)

- Extended `dpvc/utils.py::train_autoencoder` with explicit objective controls:
  - `recon_weight`
  - `kl_weight`
  - `label_weight`
  - optional `*_final` targets plus `schedule_epochs` for linear ramps
- Extended `examples/openvoice_train_vae_combined.py` with the matching CLI flags:
  - `--recon-weight`
  - `--kl-weight`
  - `--label-weight`
  - `--recon-weight-final`
  - `--kl-weight-final`
  - `--label-weight-final`
  - `--schedule-epochs`
- Extended `scripts/run_ablation_inference.py` with four Pass 6 objective conditions:
  - `cv500_obj_label2`
  - `cv500_obj_label4`
  - `cv500_obj_label_ramp`
  - `cv500_obj_recon_half_label2`
- Added `scripts/summarize_commonvoice_objective_ablation.py`
  - reads `results/eval_*_pass6_*.csv`
  - writes `results/eval_commonvoice_objective_summary_pass6.csv`
  - writes `results/eval_commonvoice_objective_collapse_pass6.csv`
- Trained the four new objective variants from the unchanged CommonVoice init checkpoint `embeddings/openvoice_vae_commonvoice_cv500.pt`:
  - `embeddings/openvoice_vae_combined_cv500_obj_label2.pt`
  - `embeddings/openvoice_vae_combined_cv500_obj_label4.pt`
  - `embeddings/openvoice_vae_combined_cv500_obj_label_ramp.pt`
  - `embeddings/openvoice_vae_combined_cv500_obj_recon_half_label2.pt`
- Generated the four matched Pass 6 evaluation corpora:
  - `output/pass6_cv500_obj_label2_eval/`
  - `output/pass6_cv500_obj_label4_eval/`
  - `output/pass6_cv500_obj_label_ramp_eval/`
  - `output/pass6_cv500_obj_recon_half_label2_eval/`
- Reused the already validated `combined`, `commonvoice_cv500_init`, and `cv500_ft_short_low_lr` CSVs by copying them into the Pass 6 naming scheme, so the changed variable stayed strictly on objective design

**Pass 6 condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short_low_lr`
- `cv500_obj_label2`
- `cv500_obj_label4`
- `cv500_obj_label_ramp`
- `cv500_obj_recon_half_label2`

**Top-line result (`results/eval_commonvoice_objective_summary_pass6.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_obj_label2`: recall `16.7%`, novelty gain `0.0589`, mean WER `0.0862`, mean MOS delta `-0.0300`
- `cv500_obj_label4`: recall `16.7%`, novelty gain `0.0496`, mean WER `0.1222`, mean MOS delta `-0.0228`
- `cv500_obj_label_ramp`: recall `16.7%`, novelty gain `0.0442`, mean WER `0.0832`, mean MOS delta `-0.0126`
- `cv500_obj_recon_half_label2`: recall `16.7%`, novelty gain `0.0500`, mean WER `0.1146`, mean MOS delta `-0.0226`

**Interpretation**
- None of the simple objective variants beat the best Pass 5 recipe (`cv500_ft_short_low_lr`) on novelty, recall, or collapse counts.
- `cv500_obj_label2` was the strongest of the new objective variants, but it still trailed `cv500_ft_short_low_lr` on novelty (`0.0589` vs. `0.0692`) and identity collapse (`53` vs. `44`).
- `cv500_obj_label_ramp` preserved MOS closest to the raw `cv500` init, but it did so by staying close to the same conservative failure shape rather than recovering controllability.
- The CommonVoice bottleneck now looks deeper than both simple finetune-policy changes (Pass 5) and simple scalar objective reweighting (Pass 6).

**Validation**
- [x] New objective controls are reproducible from the checked-in training interface
  - validated via `examples/openvoice_train_vae_combined.py --recon-weight/--label-weight/...`
  - smoke-tested with short scheduled-weight runs and then used for the four full Pass 6 checkpoints
- [x] Every objective variant has a named checkpoint and matched evaluation corpus
  - four checkpoints under `embeddings/`
  - four 110-row corpora + manifests under `output/pass6_*`
- [x] The comparison explicitly answers whether objective changes recover controllability and novelty without surrendering the CommonVoice WER/MOS gains
  - answer: **not with simple scalar reweighting or label ramps**
  - all four objective variants stay flat at `16.7%` recall and none beats `cv500_ft_short_low_lr` on novelty
- [x] The pass isolates objective design as the changed variable rather than mixing in new datasets or new metrics
  - same CommonVoice init checkpoint
  - same combined fine-tuning embeddings
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack

**FINDINGS.md review**
- Updated after Pass 6 with a new paper-facing result: simple loss reweighting and label-weight schedules do not fix the CommonVoice collapse either, which narrows the next research step further toward richer objectives or larger-scale supervision rather than more scalar tuning.

### 0.12 Pass 7 Closeout (April 29, branch `research/commonvoice-rich-objectives`)

- Extended `dpvc/model_embedding_vae.py` so the VAE caches `last_mu` and `last_logvar` during `forward`, which lets training-time auxiliary losses supervise the latent geometry directly instead of only the decoded embedding or sampled latent
- Extended `dpvc/utils.py::train_autoencoder` with richer CommonVoice adaptation controls:
  - frozen style-teacher loss on style dims (`0-8`)
  - frozen free-anchor loss on non-style dims (`9-14`)
  - scheduled weights for both auxiliary terms
- Extended `examples/openvoice_train_vae_combined.py` with the matching CLI flags:
  - `--style-teacher-checkpoint`
  - `--style-teacher-weight`
  - `--style-teacher-weight-final`
  - `--free-anchor-checkpoint`
  - `--free-anchor-weight`
  - `--free-anchor-weight-final`
- Extended `scripts/run_ablation_inference.py` with three Pass 7 rich-objective conditions:
  - `cv500_rich_teacher_style`
  - `cv500_rich_free_anchor`
  - `cv500_rich_teacher_plus_anchor`
- Added `scripts/summarize_commonvoice_rich_objectives.py`
  - reads `results/eval_*_pass7_*.csv`
  - writes `results/eval_commonvoice_rich_objectives_summary_pass7.csv`
  - writes `results/eval_commonvoice_rich_objectives_collapse_pass7.csv`
- Trained the three new rich-objective variants from the unchanged CommonVoice init checkpoint `embeddings/openvoice_vae_commonvoice_cv500.pt`:
  - `embeddings/openvoice_vae_combined_cv500_rich_teacher_style.pt`
  - `embeddings/openvoice_vae_combined_cv500_rich_free_anchor.pt`
  - `embeddings/openvoice_vae_combined_cv500_rich_teacher_plus_anchor.pt`
- Generated the three matched Pass 7 evaluation corpora:
  - `output/pass7_cv500_rich_teacher_style_eval/`
  - `output/pass7_cv500_rich_free_anchor_eval/`
  - `output/pass7_cv500_rich_teacher_plus_anchor_eval/`
- Reused the already validated `combined`, `commonvoice_cv500_init`, and `cv500_ft_short_low_lr` CSVs by copying them into the Pass 7 naming scheme, so the changed variable stayed strictly on rich supervision during CommonVoice adaptation

**Pass 7 condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short_low_lr`
- `cv500_rich_teacher_style`
- `cv500_rich_free_anchor`
- `cv500_rich_teacher_plus_anchor`

**Top-line result (`results/eval_commonvoice_rich_objectives_summary_pass7.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_rich_teacher_style`: recall `16.7%`, novelty gain `0.0574`, mean WER `0.0851`, mean MOS delta `-0.0200`
- `cv500_rich_free_anchor`: recall `16.7%`, novelty gain `0.0646`, mean WER `0.0724`, mean MOS delta `-0.0079`
- `cv500_rich_teacher_plus_anchor`: recall `16.7%`, novelty gain `0.0566`, mean WER `0.0809`, mean MOS delta `-0.0161`

**Interpretation**
- None of the three rich-objective variants improved recall beyond `16.7%`.
- None beat the best Pass 5 recipe (`cv500_ft_short_low_lr`) on novelty or identity collapse.
- `cv500_rich_free_anchor` was the strongest of the new variants: it improved WER and MOS beyond `cv500_ft_short_low_lr`, but novelty still trailed slightly (`0.0646` vs. `0.0692`) and identity collapse stayed worse (`50` vs. `44`).
- The style-teacher and teacher-plus-anchor variants also stayed in the same conservative failure shape: recall flat, style collapse near `54-55`, and identity collapse still well above the combined model.
- The CommonVoice bottleneck now looks deeper than simple finetune-policy changes (Pass 5), scalar loss reweighting (Pass 6), and the first richer-teacher/anchor objective family (Pass 7).

**Validation**
- [x] Rich-objective controls are reproducible from the checked-in training interface
  - validated via `examples/openvoice_train_vae_combined.py --style-teacher-* / --free-anchor-*`
  - smoke-tested with short runs and then used for the three full Pass 7 checkpoints
- [x] Every rich-objective variant has a named checkpoint and matched evaluation corpus
  - three checkpoints under `embeddings/`
  - three 110-row corpora + manifests under `output/pass7_*`
- [x] The comparison explicitly answers whether richer supervision beats raw `cv500` and the best Pass 5 recipe
  - answer: **not with these teacher/anchor losses**
  - all three variants stay flat at `16.7%` recall and none beats `cv500_ft_short_low_lr` on novelty or identity collapse
- [x] The pass isolates supervision/objective design as the changed variable
  - same CommonVoice init checkpoint
  - same combined fine-tuning embeddings
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack
- [x] The pass yields a clear next-step decision
  - the next CommonVoice work should move either to richer pretraining supervision on CommonVoice itself, partial-label / pseudo-label objectives, or a larger-scale follow-up once a stronger objective survives on this validation-scale setup

**FINDINGS.md review**
- Updated after Pass 7 with a new paper-facing result: richer teacher/anchor supervision during combined finetuning still does not recover recall after CommonVoice pretraining, which suggests the remaining bottleneck is deeper than finetune-time scalar tuning or the first richer-objective family tried here.

### 0.13 Pass 8 Closeout (April 29, branch `research/commonvoice-partial-label-pretrain`)

- Extended `examples/openvoice_extract_commonvoice.py` so saved CommonVoice embedding artifacts now include a `metadata_report` summarizing field coverage and categorical distributions
- Extended `dpvc/utils.py` with CommonVoice weak-supervision helpers:
  - metadata coverage summarization utilities
  - `train_commonvoice_pretrain(...)` for reconstruction + masked metadata + pseudo-style losses
  - inverse-frequency balancing for metadata classes and pseudo-style rows
- Extended `examples/openvoice_pretrain_vae_commonvoice.py` with weak-label CLI support:
  - `--metadata-targets`
  - `--metadata-weight`
  - `--metadata-min-count`
  - `--pseudo-style-weight`
  - `--pseudo-style-threshold`
  - `--style-dims`
  - `--free-dims`
- Added `scripts/annotate_commonvoice_pseudolabels.py`
  - reads a CommonVoice embedding artifact
  - adds `pseudo_style`, `pseudo_style_confidence`, `pseudo_style_raw_label`, `pseudo_style_report`
  - keeps all pseudo labels in the artifact and uses thresholds only at training/report time so later threshold sweeps do not require relabeling
- Extended `scripts/run_ablation_inference.py` with three Pass 8 weak-supervision conditions:
  - `cv500_pl_meta`
  - `cv500_pl_pseudo_style`
  - `cv500_pl_meta_plus_pseudo`
- Added `scripts/summarize_commonvoice_partial_label.py`
  - reads `results/eval_*_pass8_*.csv`
  - writes `results/eval_commonvoice_partial_label_summary_pass8.csv`
  - writes `results/eval_commonvoice_partial_label_collapse_pass8.csv`
- Recorded CommonVoice metadata sparsity on the `cv500` subset artifact:
  - age known on `193/1202` clips
  - gender known on `184/1202`
  - accent known on `190/1202`
  - only two age buckets (`young`, `adult`) had enough support for the metadata-weighted run at the default minimum-count filter
- Generated a pseudo-labeled CommonVoice artifact:
  - `embeddings/openvoice_commonvoice_cv500_pseudo.pt`
  - accepted pseudo-style counts at threshold `0.60`: `neutral=640`, `sad=336`, `happy=51`, `disgust=46`, `anger=11`, `fear=5`
  - this class imbalance is why Pass 8 used inverse-frequency row weighting for pseudo-style supervision
- Trained the three new CommonVoice pretraining variants:
  - `embeddings/openvoice_vae_commonvoice_cv500_pl_meta.pt`
  - `embeddings/openvoice_vae_commonvoice_cv500_pl_pseudo_style.pt`
  - `embeddings/openvoice_vae_commonvoice_cv500_pl_meta_plus_pseudo.pt`
- Finetuned each one on the unchanged combined labeled embeddings:
  - `embeddings/openvoice_vae_combined_cv500_pl_meta.pt`
  - `embeddings/openvoice_vae_combined_cv500_pl_pseudo_style.pt`
  - `embeddings/openvoice_vae_combined_cv500_pl_meta_plus_pseudo.pt`
- Generated the three matched Pass 8 evaluation corpora:
  - `output/pass8_cv500_pl_meta_eval/`
  - `output/pass8_cv500_pl_pseudo_style_eval/`
  - `output/pass8_cv500_pl_meta_plus_pseudo_eval/`
- Reused the already validated `combined`, `commonvoice_cv500_init`, `cv500_ft_short_low_lr`, and `cv500_rich_free_anchor` CSVs by copying them into the Pass 8 naming scheme, so the changed variable stayed strictly on weak supervision during CommonVoice pretraining

**Pass 8 condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short_low_lr`
- `cv500_rich_free_anchor`
- `cv500_pl_meta`
- `cv500_pl_pseudo_style`
- `cv500_pl_meta_plus_pseudo`

**Top-line result (`results/eval_commonvoice_partial_label_summary_pass8.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_rich_free_anchor`: recall `16.7%`, novelty gain `0.0646`, mean WER `0.0724`, mean MOS delta `-0.0079`
- `cv500_pl_meta`: recall `16.7%`, novelty gain `0.0570`, mean WER `0.0918`, mean MOS delta `-0.0505`
- `cv500_pl_pseudo_style`: recall `16.7%`, novelty gain `0.0190`, mean WER `0.0263`, mean MOS delta `-0.0229`
- `cv500_pl_meta_plus_pseudo`: recall `16.7%`, novelty gain `0.0181`, mean WER `0.0285`, mean MOS delta `-0.0148`

**Interpretation**
- None of the three weak-label variants improved recall beyond `16.7%`.
- `cv500_pl_meta` was the best novelty result of the new variants, but it still trailed both `cv500_ft_short_low_lr` and `cv500_rich_free_anchor`, and it did so with worse WER and MOS than those stronger earlier CommonVoice baselines.
- `cv500_pl_pseudo_style` and `cv500_pl_meta_plus_pseudo` dramatically improved WER, but only by collapsing much harder toward baseline speaker identity: identity-collapse rows jumped to `85` and `90`.
- The pseudo-label path therefore looks conservative rather than expressive on this validation-scale setup: it preserves content and naturalness, but not controllability or speaker shift.
- The CommonVoice bottleneck now looks deeper than finetune-policy tuning (Pass 5), scalar weight schedules (Pass 6), finetune-time teacher/anchor supervision (Pass 7), and this first weak-label pretraining family (Pass 8).

**Validation**
- [x] CommonVoice metadata coverage is measured and reported from the checked-in extraction interface
  - validated through the `metadata_report` saved by `examples/openvoice_extract_commonvoice.py`
  - confirmed sparse coverage on the checked-in `cv500` subset artifact, which now informs how we interpret the metadata-only run
- [x] Partial-label CommonVoice pretraining is reproducible from the checked-in training interface
  - validated through `examples/openvoice_pretrain_vae_commonvoice.py --metadata-* / --pseudo-style-*`
  - smoke-tested with short runs and then used for the three full Pass 8 pretraining checkpoints
- [x] Every Pass 8 condition has a named pretrain checkpoint, finetuned checkpoint, and matched evaluation corpus
  - three CommonVoice pretrain checkpoints under `embeddings/`
  - three combined finetune checkpoints under `embeddings/`
  - three 110-row corpora + manifests under `output/pass8_*`
- [x] The comparison explicitly answers whether weak supervision during CommonVoice pretraining beats raw `cv500`, the best Pass 5 recipe, and the best Pass 7 recipe
  - answer: **not on this validation-scale setup**
  - metadata-only supervision improves novelty over raw `cv500`, but not enough to beat the best earlier CommonVoice variants
  - pseudo-style supervision improves WER sharply, but collapses identity shift even harder
- [x] The pass isolates pretraining supervision as the changed variable
  - same `cv500` CommonVoice subset scale
  - same combined finetune data
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack
- [x] The pass yields a clear next-step decision
  - the next CommonVoice work should focus on pseudo-label quality, prototype/teacher-space targets during CommonVoice pretraining, or stronger curricula rather than simply adding the current weak labels at this scale

**FINDINGS.md review**
- Updated after Pass 8 with a new paper-facing result: validation-scale weak supervision during CommonVoice pretraining still does not recover recall, and the pseudo-style variants appear to trade controllability/novelty for stronger intelligibility rather than solving the underlying collapse.

---

### 0.14 April 30 Meeting with Joe — Direction Update

- Joe confirmed that the **biggest untried experiment** is still the first real sampled mixed-data run that combines CommonVoice, CREMA-D, and Expresso in one training setup
- Joe explicitly endorsed **pseudo-labeled CommonVoice + mixed-data training** as the most promising current path, more than another isolated CommonVoice-only refinement
- Joe warned that a naive combined run may just behave like CommonVoice unless we manage the mixture carefully; he called out the training schedule itself as an open empirical question
- Joe's current sampling intuition is that **speaker breadth may matter more than raw clip count** on CommonVoice, so the next mixed-data corpus should prioritize at least one clip per speaker before adding more clips
- Joe agrees that architecture may matter, but he put **data mixing and supervision quality ahead of architecture changes** for the immediate next step
- Joe qualitatively found that style strengths **above `5.0` can still work well**, especially for whisper on non-Trump examples, so our docs should not treat `5.0` as a hard ceiling
- Joe said the branch stack is getting hard to interpret directly and asked for a **single document** that summarizes findings and how to read the metrics, which reinforces keeping `FINDINGS.md` as the async review hub
- Joe confirmed the current **12GB GPU machine is sufficient for these VAE experiments**, while larger cluster access may be possible later but should not be assumed for the next branch
- **Planned next branch:** `research/combined-data-pseudolabel-mix`

### 0.15 Pass 9 Bootstrap Implementation (April 30, branch `research/combined-data-pseudolabel-mix`)

- Added `scripts/build_mixed_training_set.py`
  - builds the first sampled **CommonVoice + CREMA-D + Expresso** training artifact
  - preserves `source_dataset`, `style_label_mask`, `style_label_row_weight`, pseudo-label confidence, and a saved `mixture_report`
  - prioritizes CommonVoice **speaker breadth first**, with optional pseudo-label preference and per-style caps
- Added `examples/openvoice_train_vae_mixed.py`
  - trains directly from the mixed artifact
  - supports the first three schedule families Joe called out:
    - `static_balanced`
    - `cv_warmup`
    - `labeled_finish`
- Added `dpvc.utils.train_mixed_autoencoder()`
  - samples rows each epoch according to dataset-mass schedules rather than treating the merged artifact as a flat dataset
  - keeps the labeled-row loss masked so unlabeled CommonVoice rows do not act like fake negatives
- Added `scripts/summarize_mixed_data_results.py`
  - reserved for the upcoming Pass 9 result matrix so the mixed-data branch has a checked-in summary format before the full run

**Smoke validation**
- [x] Mixed-data artifact builder compiles and runs
  - validated with a small smoke artifact at `/private/tmp/openvoice_mixed_smoke.pt`
  - smoke artifact summary: `627` total rows = `36` CommonVoice + `546` CREMA-D + `45` Expresso
  - labeled rows: `622/627`
- [x] First real mixed-data artifact built for the branch
  - wrote `embeddings/openvoice_mixed_base.pt`
  - current branch artifact summary: `1,325` total rows = `500` CommonVoice + `546` CREMA-D + `279` Expresso
  - labeled rows: `1,287/1,325`
  - current CommonVoice pseudo-label mix in the artifact: `neutral=207`, `sad=170`, `happy=37`, `disgust=34`, `anger=10`, `fear=4`
- [x] CommonVoice speaker-first sampling, pseudo-label filtering, and mixture reporting execute end to end
  - validated with `20` CommonVoice speakers, `max 2` clips per speaker, pseudo-label preference on, and explicit style caps
- [x] All three mixed-data schedules execute end to end on the smoke artifact
  - `static_balanced`
  - `cv_warmup`
  - `labeled_finish`
- [x] The shared evaluation-corpus generator now accepts the first mixed-data conditions
  - smoke-validated with `mixed_static_balanced` plus a one-file source run that wrote 10 outputs and a manifest to `/private/tmp/pass9_mixed_infer_smoke/`
- [x] New scripts pass syntax validation
  - `py_compile` passed for `dpvc/utils.py`, `examples/openvoice_train_vae_mixed.py`, `scripts/build_mixed_training_set.py`, `scripts/run_ablation_inference.py`, and `scripts/summarize_mixed_data_results.py`

**What is still not done**
- The first **full Pass 9 scientific run** is still pending
- No new paper-facing result belongs in `FINDINGS.md` yet from this branch
- Phase 1.6 roadmap items stay open until we train the real mixed-data checkpoints, generate matched corpora, and run the full metric stack

**Immediate next execution target**
- Build the first real `embeddings/openvoice_mixed_base.pt`
- Train:
  - `embeddings/openvoice_vae_mixed_static_balanced.pt`
  - `embeddings/openvoice_vae_mixed_cv_warmup.pt`
  - `embeddings/openvoice_vae_mixed_labeled_finish.pt`
- Then generate the Pass 9 corpora and evaluate them against the strongest earlier CommonVoice references

---

## 1. Project Overview

**dpvc** is a Python library for **differentially private voice conversion** — it anonymizes a speaker's identity by passing their voice embedding through a VAE with calibrated DP noise, then reconstructs audio with a modified (anonymized) speaker embedding.

The new work on `feat/controlvc` extends this with **controllable anonymization**: instead of just adding noise, we can steer specific perceptual attributes (happy, sad, whisper, etc.) in the anonymized output by manipulating labeled latent dimensions of the VAE.

### Core Pipeline

```
Source Audio
    │
    ├─► HuBERT (content codes)  ─────────────────────────┐
    ├─► D_VECTOR (256-dim speaker embedding) ─► VAE ─► modified embedding
    ├─► YAAPT (F0 pitch contour)  ────────────────────────┤
    │                                                      │
    └──────────────────────────────────────────────────────┴─► CodeGenerator (vocoder) ─► Output Audio
```

**Key insight:** The VAE sits on the speaker embedding only. Content (HuBERT codes) and prosody (F0) pass through unchanged. So style control operates on *who it sounds like*, not *what they say*.

---

## 2. What Joe Near Built (Pre-April 2026)

Joe's work existed on two branches:

### `feat/controlvc` (merged ~Feb 2026)
- Added `ControlVCWrapper` in `dpvc/controlvc.py` (~500 lines) wrapping the [control-vc](https://github.com/auspicious3000/control-vc) system
- Integrated it alongside the existing `OpenVoiceWrapper`
- **Issues found:** hardcoded device `"cuda"` (broke on Mac), hardcoded paths to Joe's machine, fixed `INPUT_DIM=256` in VAE

### `controllable_vae` branch (never merged)
- Upgraded `model_embedding_vae.py` with a more sophisticated architecture: GELU activations, LayerNorm, proper reparameterization trick, `control_features` dict for overriding specific latent dims at inference
- Updated `utils.py` `train_autoencoder()` to support label-aware training (MSE loss on first K latent dims)
- Updated `anonymizer.py` to use a `vae_config` dict pattern
- All of this was OpenVoice-specific; needed porting to work with ControlVC

---

## 3. What We Built (April 8-9, 2026)

### 3.1 Portability Fixes (commit `139d0e5`)
- Changed ControlVC device default from `"cuda"` to auto-detect (`cuda` if available, else `cpu`)
- Added `get_vae_config()` to `ControlVCWrapper` and abstract `Wrapper` base class
- Fixed hardcoded paths in example scripts with argparse

### 3.2 Controllable VAE Port (commit `dfcbf9a`)
- Ported the full controllable VAE architecture from `origin/controllable_vae` into the main codebase
- Made it work with ControlVC (was OpenVoice-only)
- **Files replaced/updated:**
  - `dpvc/model_embedding_vae.py` — full rewrite with GELU/LayerNorm encoder, reparameterization trick, control_features support
  - `dpvc/anonymizer.py` — rewritten with vae_config dict pattern
  - `dpvc/utils.py` — `train_autoencoder()` now supports `labels` dict and label MSE loss
  - `dpvc/wrapper.py` — added `get_vae_config()` to abstract interface
  - `dpvc/openvoice.py` — added `get_vae_config()` for consistency

### 3.3 Expresso Pipeline (commit `dfcbf9a`)
Three new example scripts for end-to-end controllable training:

1. **`examples/controlvc_extract_expresso.py`** — Extracts ControlVC speaker embeddings from HuggingFace's `ylacombe/expresso` dataset
   - Handles 11 styles: default, confused, enunciated, happy, laughing, sad, whisper, emphasis, essentials, longform, singing
   - One-hot style encoding: active style = +1, all others = -1
   - Workarounds for: xet storage stalls (`HF_HUB_DISABLE_XET=1`), torchcodec incompatibility (uses `soundfile` instead), dataset download failures (supports `--parquet-dir` for local cache)

2. **`examples/controlvc_train_vae_expresso.py`** — Trains controllable VAE with style labels
   - Default: latent_dims=16 (11 style + 5 free), lr=1e-6
   - Label loss forces first 11 latent dims to match one-hot style encoding via MSE

3. **`examples/controlvc_infer_controllable.py`** — CLI for controllable voice anonymization
   - Maps style name to latent dim index
   - Sets all 11 style dims at inference (active = style_value, others = -1) to match training encoding

### 3.4 Fairseq/HuBERT Fix (April 9, uncommitted)
The HuBERT content encoder is critical — without it, the vocoder produces unintelligible noise. Fairseq 0.12.2 is incompatible with Python 3.11 due to mutable dataclass defaults.

**Fix applied in `dpvc/controlvc.py`:**
- Before importing fairseq, monkey-patches `dataclasses._get_field` to convert mutable defaults to `default_factory` calls
- Adds the control-vc repo to `sys.path` so `fairseq_feature_reader` is importable
- Also patched `fairseq/dataclass/configs.py` (manual `field(default_factory=...)` edits) and `fairseq/dataclass/initialize.py` (handle MISSING defaults) and `fairseq/checkpoint_utils.py` (`weights_only=False`) — these are in the venv, not version-controlled

**Result:** HuBERT now loads and produces real content codes, generating intelligible speech output.

### 3.6 F0 Prosody-Based Style Control (April 9, branch `feat/f0-style-control`)

After confirming that D_VECTOR embeddings don't carry style information (Section 5), pivoted to controlling style via F0 (pitch) manipulation.

**Changes:**
- `dpvc/controlvc.py` — Added `f0_transform` parameter to `inference()`. Supports three operations applied to the F0 contour before it hits the vocoder:
  - `pitch_shift`: multiply voiced F0 values (e.g., 1.25 = 25% higher pitch)
  - `range_scale`: expand/compress variation around mean (e.g., 0.4 = flatter, 2.0 = more expressive)
  - `flatten`: interpolate toward mean (0.0 = no change, 1.0 = completely monotone)
- `dpvc/anonymizer.py` — Pass-through `f0_transform` to wrapper
- `dpvc/wrapper.py` — Added `f0_transform` to abstract base class signature
- `examples/controlvc_infer_controllable.py` — Rewritten with F0 preset system and custom CLI args

**F0 Presets (tuned for audible distinction):**
| Style | pitch_shift | range_scale | flatten |
|-------|------------|-------------|---------|
| happy | 1.25 | 1.6 | — |
| sad | 0.80 | 0.4 | 0.3 |
| whisper | 0.90 | 0.2 | 0.8 |
| confused | 1.10 | 2.0 | — |
| laughing | 1.35 | 2.2 | — |
| enunciated | 1.00 | 1.8 | — |

**Result:** Audible style differences confirmed. Tested with trump_0.wav as source. Happy and laughing are clearly higher/more animated, sad is lower/flatter, whisper is near-monotone. The VC pipeline itself changes voice identity (expected — HuBERT quantization is lossy), but prosody differences are clearly distinguishable.

### 3.5 Inference Style Encoding Fix (April 9, uncommitted)
Original inference script only set 1 latent dim (the active style). But training used a full one-hot encoding (+1 active, -1 all others across all 11 dims). Fixed to set all 11 style dims at inference.

---

## 4. Training Runs

### Extraction
- **4,840 samples** extracted from Expresso dataset (5 of 6 shards — shard 5 consistently failed to download)
- 256-dim speaker embeddings + 11-dim one-hot style labels
- Saved to `embeddings/controlvc_expresso_emb.pt`
- Took ~9.5 minutes on CPU

### VAE Training

| Version | System | Epochs | LR   | Final Label Loss | Final Recon | Notes |
|---------|--------|--------|------|-----------------|-------------|-------|
| v1      | ControlVC | 2000   | 1e-6 | ~590            | 0.20        | LR too low — label dims barely learned |
| v2      | ControlVC | 2000   | 1e-4 | ~55             | 0.20        | 10x improvement, but still high |
| v3      | ControlVC | 5000   | 1e-4 | ~15             | 0.18        | Label loss improved but no audible style diff — even at style_value=5.0 |
| **v4**  | **OpenVoice** | **2000** | **1e-4** | **~1.5** | **~67** | **Label loss 10x lower than ControlVC v3! OpenVoice embeddings encode style.** |
| **v5**  | **OpenVoice+CREMA-D** | **2000** | **1e-4** | **~18** | **~200** | **91 speakers, 6 emotions. Higher label loss but all emotions perceptually distinct.** |

Checkpoints saved in `embeddings/`:
- `controlvc_vae_expresso.pt` (v1, ControlVC)
- `controlvc_vae_expresso_v2.pt` (v2, ControlVC)
- `controlvc_vae_expresso_v3.pt` (v3, ControlVC)
- `openvoice_vae_expresso.pt` (v4, OpenVoice+Expresso)
- `openvoice_vae_cremad.pt` (v5, OpenVoice+CREMA-D — best diversity)

---

## 5. Current Status & Open Problems

### Working
- End-to-end pipeline: extract → train → infer produces intelligible speech
- HuBERT content encoder loads and produces real content codes
- VAE reconstructs speaker embeddings with low reconstruction loss
- Style control infrastructure is in place

### OpenVoice Style Control Results (April 9)

Trained controllable VAE on 8,712 OpenVoice embeddings (v4, label loss ~1.5 — 10x better than ControlVC). Tested with `trump_0.wav` as source.

**Perceptual evaluation (amplified style values):**

| Style | x1 | x3 | x5 |
|-------|-----|-----|-----|
| whisper | Subtle | **Sounds like a real whisper** | Over-exaggerated, background noise |
| sad | Subtle shift | **Sounds like an actual sad person** (some distortion) | Sad but more distortion |
| happy | Barely perceptible | Jovial but close to original, not pronounced | Distorted, not clearly happy |
| laughing | No difference | No meaningful change | Minimal change |

**Key findings:**
- **x3 is the sweet spot** — x5 overdrives the decoder and introduces artifacts
- **Whisper and sad work well** via embedding alone — these styles map to strong tonal shifts (pitch reduction, energy reduction, flattening)
- **Happy and laughing need F0 augmentation** — their perceptual qualities (pitch variation, rhythmic changes) aren't fully captured in the 256-dim embedding
- **Massive improvement over ControlVC** — went from zero audible difference to clearly recognizable style shifts

**Numerical differences from baseline (mean absolute):**
| Style | x1 | x3 | x5 |
|-------|-----|-----|-----|
| whisper | 0.039 | 0.062 | **0.099** |
| sad | 0.036 | 0.038 | 0.056 |
| happy | 0.035 | 0.037 | 0.037 |
| laughing | 0.034 | 0.030 | 0.037 |

**F0 post-processing attempt (failed):**
Tried combining embedding control (x3) with F0 pitch shifting on the output audio for happy (+20%), laughing (+30%), confused (+10%). Result: voices became thin and high-pitched, not perceptibly "happy" or "laughing." Pitch shifting output audio is too crude — happiness and laughter are speech *behaviors* (varied intonation patterns, rhythm, breath) not achievable by shifting a single signal. Reverted F0 post-processing from OpenVoice wrapper.

**Conclusion:** Embedding-based style control with OpenVoice at x3 works for **tonal/energy styles** (whisper, sad) but not for **behavioral styles** (happy, laughing, confused). This is a meaningful finding — it reveals which style dimensions are capturable in a 256-dim speaker embedding vs. which require fundamentally different representations.

### CREMA-D Results (April 10, branch `feat/cremad-experiments`)

**Motivation:** Expresso has only 3 speakers — within-style variance is dominated by speaker identity, not emotion. Joe recommended maximizing speaker diversity. CREMA-D has 91 speakers × 6 emotions × ~13 sentences = 7,442 clips.

**Dataset:** [AbstractTTS/CREMA-D on HuggingFace](https://huggingface.co/datasets/AbstractTTS/CREMA-D). 6 emotions: anger, disgust, fear, happy, neutral, sad. Extracted 1 sample per speaker per emotion = 546 samples (per Joe's recommendation).

**Embedding separability (CREMA-D, 91 speakers):**

| Emotion | Dist from global | Within var | Ratio |
|---------|-----------------|------------|-------|
| anger | 0.2712 | 0.4875 | 0.56 |
| sad | 0.2006 | 0.4875 | 0.41 |
| happy | 0.1612 | 0.4978 | 0.32 |
| neutral | 0.1430 | 0.4768 | 0.30 |
| disgust | 0.1483 | 0.4891 | 0.30 |
| fear | 0.1321 | 0.5180 | 0.26 |

Overall ratio: 0.54. Speaker separability: 1.43 (speakers are well-separated, emotions are not). The embedding fundamentally prioritizes speaker identity over emotion — but the VAE can still learn nonlinear separations.

**Perceptual evaluation (x3 amplification):**

| Emotion | Listener assessment |
|---------|-------------------|
| anger | Subtly different, noticeable |
| disgust | Similar to anger |
| fear | Distinct but not "fearful" — sounds timid |
| happy | Does sound happy, especially toward end of speech |
| neutral | Neutral sounding |
| sad | Sounds sad |

**Acoustic analysis (librosa F0/RMS/spectral centroid):**

| Style | dF0 Mean | dF0 Std | dF0 Range | dEnergy | dBrightness |
|-------|----------|---------|-----------|---------|-------------|
| anger | +12.4 | +2.1 | -12.7 | -0.001 | +123 |
| disgust | -38.9 | +3.8 | -36.3 | +0.001 | -27 |
| fear | -5.4 | +7.6 | -6.4 | +0.017 | -26 |
| happy | -14.4 | +12.1 | +12.9 | +0.001 | -49 |
| neutral | -53.4 | +0.1 | -43.7 | +0.007 | -178 |
| sad | -12.9 | +2.7 | -8.0 | +0.008 | -223 |

**Key findings:**
- **Happy has highest F0 variance** (+12.1 over baseline) — exactly matches how happy speech sounds
- **Anger is brightest** (+123) — sharper, edgier tone
- **Sad is darkest** (-223 brightness) — muffled, lower quality
- **Neutral is flattest** (F0 std +0.1, smallest range) — monotone
- **All 6 emotions are acoustically distinct** — major improvement over Expresso where only whisper/sad worked
- **Speaker diversity matters more than label separability** — despite lower raw separability (0.54 vs 0.62), the VAE learned better generalizable patterns from 91 speakers vs 3

**Why CREMA-D works better than Expresso for emotion control:**
1. 91 speakers forces the VAE to find emotion patterns that generalize across voices
2. 1 sample per speaker per emotion prevents speaker memorization
3. CREMA-D emotions are acted with clear intent (professional actors), while Expresso styles are more subtle reading variations

### Combined CREMA-D + Expresso Results (April 10, branch `feat/cremad-experiments`)

**Motivation:** CREMA-D provides speaker diversity (91 speakers, 6 emotions) but lacks Expresso's unique styles (whisper, confused, enunciated). Combining both gives the best of both worlds: 9 unified labels with strong speaker diversity.

**Unified label scheme (825 samples, ~90-94 per label):**
- From CREMA-D (91 speakers): anger, disgust, fear, happy, neutral, sad
- From Expresso (3 speakers, capped at 90): confused, enunciated, whisper
- Shared (CREMA-D + 1-per-speaker Expresso): happy, neutral, sad

**VAE v6:** 3000 epochs, lr=1e-4, latent_dims=15 (9 labels + 6 free). Final label loss ~30.

**Acoustic analysis (all deltas from baseline):**

| Style | dF0 Mean | dF0 Std | dF0 Range | dBrightness | Signature |
|-------|----------|---------|-----------|-------------|-----------|
| anger | +13.2 | -3.9 | +4.4 | +369 | Sharp, edgy |
| confused | +2.5 | -3.8 | -29.5 | +111 | Hesitant, narrow range |
| disgust | -25.6 | +6.1 | +24.3 | -156 | Low, withdrawn |
| enunciated | -10.5 | -3.9 | +5.2 | +274 | Crisp, bright |
| fear | +17.4 | +4.8 | +80.8 | -155 | Tense, variable |
| **happy** | **+26.8** | **+33.2** | **+240.7** | +84 | **Most expressive — 2x F0 variation** |
| neutral | -46.0 | +6.7 | +38.8 | -533 | Flat, subdued |
| sad | -0.3 | +11.2 | +44.3 | -272 | Darker tone |
| **whisper** | **-128.6** | **-30.8** | **-143.6** | **+945** | **F0 near zero, breathy/airy** |

**Perceptual evaluation:** All 9 styles perceptually distinct and matching acoustic expectations. This is the first time happy has produced a convincingly animated output.

**Why the combined model works:**
1. **Speaker diversity from CREMA-D** (91 speakers) prevents speaker memorization
2. **Style richness from Expresso** adds whisper/confused/enunciated which CREMA-D lacks
3. **Balanced sampling** (~90 per label) prevents any label from dominating
4. **Cross-dataset generalization** — the model learns emotion patterns that hold across two completely different recording conditions

**Progression of results:**
| Model | Dataset | Speakers | Working styles | Happy? |
|-------|---------|----------|---------------|--------|
| v1-v3 | Expresso+ControlVC | 3 | 0 | No |
| v4 | Expresso+OpenVoice | 3 | 2 (whisper, sad) | No |
| v5 | CREMA-D+OpenVoice | 91 | 6 (all emotions) | Yes |
| **v6** | **Combined** | **94** | **9 (all labels)** | **Yes** |

---

### ControlVC Style Differentiation (April 9 — superseded by OpenVoice)
**The core problem with ControlVC:** All style-controlled outputs sounded identical. Setting different style dims at inference produced no audible differences.

**Root cause identified (April 9):** Empirical analysis of the raw 256-dim D_VECTOR embeddings shows that **styles are not separable in embedding space:**

```
Embedding analysis (4,840 samples, unit-normalized):
  Between-style mean centroid distance: 0.0298
  Within-style mean distance to centroid: 0.0338
  Separability ratio: 0.88 (needs >>1)

  Only whisper shows any separation (~0.06 from others)
  All other styles overlap almost completely
```

**What this means:** The D_VECTOR speaker embedding encodes speaker identity, not speaking style. Within any given style, samples from different speakers vary MORE than the style signal itself. The VAE cannot learn to control what the input representation doesn't encode.

**Style distribution in training data:**
| Style | Samples | Notes |
|-------|---------|-------|
| default | 759 | |
| confused | 760 | |
| enunciated | 760 | |
| happy | 760 | |
| laughing | 557 | |
| sad | 379 | |
| whisper | 379 | Only style with measurable embedding separation |
| emphasis | 400 | |
| essentials | 80 | Too few samples |
| longform | 3 | Effectively zero |
| singing | 3 | Effectively zero |

**Training code observation:** `beta = 1` and losses use `.sum()` not `.mean()`. Label loss sums over batch×11 dims = 2,816 terms. Recon loss sums over batch×256 dims = 65,536 terms. Label loss is inherently ~23x smaller in gradient magnitude — the VAE heavily prioritizes reconstruction over label matching.

**Confirmed April 9:** Tested v3 checkpoint (label loss ~15) with style_value=5.0 (5x normal). Zero audible difference from baseline. The decoder is simply not sensitive to these latent dims because the input representation doesn't carry style.

**Possible paths forward:**

1. **Control F0 instead of (or in addition to) speaker embedding:** Style differences primarily manifest in prosody (F0 contour), not speaker identity. Applying the controllable VAE to F0 features might yield audible style differences. The ControlVC pipeline already extracts F0 via YAAPT — we just don't route it through the VAE.

2. **Use a style-aware embedding model:** Replace D_VECTOR with an embedding model that jointly encodes identity and style (e.g., emotion-aware speaker encoder, or a multi-task model trained on both speaker ID and emotion).

3. **Switch to per-speaker style control:** Within a single speaker, style differences may be more detectable (no cross-speaker variance to drown the signal). Train speaker-specific VAEs or condition on speaker ID.

4. **Operate on a concatenated representation:** Instead of just the 256-dim speaker embedding, pass [speaker_emb; F0_stats; energy_stats] through the VAE. This gives the model more style-relevant information to work with.

5. **~~Ask Joe~~** ✓ **ANSWERED (April 9):** Joe confirmed that OpenVoice embeds F0 in the speaker embedding, so style changes ARE audible with OpenVoice. He noted that some VC systems (like ControlVC) treat F0 separately, so randomizing the speaker embedding alone doesn't produce audible style changes. He also reminded us that the original plan was to use Expresso with **OpenVoice, not ControlVC**, since OpenVoice has been easier to work with.

**Conclusion: Pivot to OpenVoice for controllable style work.** The ControlVC pipeline is still valuable for DP anonymization (speaker embedding noise), but style control requires a system where F0 is part of the embedding — which OpenVoice provides.

---

## 6. Architecture Details

### VAE (model_embedding_vae.py)

```
Encoder: Linear(256→512) → GELU → LayerNorm(512) → Linear(512→256) → GELU → Linear(256→64) → GELU
         → to_mu(64→latent_dim), to_logvar(64→latent_dim)

Decoder: Linear(latent_dim→64) → GELU → Linear(64→256) → GELU → Linear(256→512) → GELU → Linear(512→256)
```

Reparameterization: `z = mu + eps * exp(0.5 * logvar)`

DP noise (at inference): L2 clip → Gaussian noise → post-clip → clamp

Control features: After computing z, override specific dims: `z[:, idx] = value`

### Training Loss
```
total_loss = recon_loss + kl_loss + beta * label_mse_loss
```
Where:
- `recon_loss`: MSE between input and reconstructed embedding
- `kl_loss`: KL divergence (standard VAE)
- `label_mse_loss`: MSE between first K latent dims and target labels (one-hot style encoding)
- `beta`: weighting factor (needs investigation — see `dpvc/utils.py`)

### Anonymizer Flow
```python
source_embedding = wrapper.extract_embedding(source_file)   # 256-dim
target_embedding = VAE(source_embedding, noise, control_features)  # 256-dim (modified)
wrapper.inference(source_file, output_file, source_embedding, target_embedding)
```

The `inference()` call uses the source audio's HuBERT codes and F0, but replaces the speaker embedding with the VAE-modified one.

---

## 7. File Inventory

### Core Library (`dpvc/`)
| File | Purpose | Status |
|------|---------|--------|
| `controlvc.py` | ControlVC wrapper (HuBERT, D_VECTOR, F0, vocoder) | Updated: fairseq compat patch, device auto-detect |
| `anonymizer.py` | DP pipeline orchestrator | Rewritten: vae_config dict pattern |
| `model_embedding_vae.py` | VAE model (GELU/LayerNorm, control_features) | Rewritten from controllable_vae branch |
| `utils.py` | Training utilities (train_autoencoder with labels) | Updated: label-aware training |
| `wrapper.py` | Abstract base class | Updated: get_vae_config() |
| `openvoice.py` | OpenVoice wrapper | Updated: get_vae_config() |
| `__init__.py` | Package exports | Unchanged |

### Examples (`examples/`)
| File | Purpose | Status |
|------|---------|--------|
| `controlvc_extract_expresso.py` | Extract embeddings from Expresso | New |
| `controlvc_train_vae_expresso.py` | Train controllable VAE | New |
| `controlvc_infer_controllable.py` | Controllable inference CLI | New, updated (full style encoding) |
| `controlvc_extract_commonvoice.py` | CommonVoice extraction | Updated (argparse) |
| `openvoice_inference.py` | OpenVoice demo | Updated (vae_config API) |

### Generated Artifacts (not committed)
| Path | Description |
|------|-------------|
| `embeddings/controlvc_expresso_emb.pt` | 4,840 extracted embeddings + style labels |
| `embeddings/controlvc_vae_expresso.pt` | VAE v1 (lr=1e-6, 2000 epochs) |
| `embeddings/controlvc_vae_expresso_v2.pt` | VAE v2 (lr=1e-4, 2000 epochs) |
| `embeddings/controlvc_vae_expresso_v3.pt` | VAE v3 (lr=1e-4, 5000 epochs, in progress) |
| `output/*.wav` | Generated audio samples |

---

## 8. Dependencies & Environment

- **Python:** 3.11.9 (pyenv)
- **PyTorch:** 2.0.1 (via venv)
- **torchaudio:** 2.9.1
- **fairseq:** 0.12.2 (requires dataclass monkey-patch for Python 3.11)
- **hydra-core:** 1.3.2 (upgraded from 1.0.7)
- **omegaconf:** 2.3.0 (upgraded from 2.0.6)
- **control-vc repo:** `/Users/steve/repos/control-vc` (contains checkpoints and fairseq_feature_reader.py)
- **HuBERT checkpoint:** `control-vc/checkpoints/hubert_base_ls960.pt`
- **K-means model:** `control-vc/checkpoints/km.bin`
- **Vocoder:** `control-vc/checkpoints/embed_f0stat2/g_00350000.pth`

### Venv patches (not version-controlled)
These files in `.venv/lib/python3.11/site-packages/fairseq/` were manually patched:
- `dataclass/configs.py` — `field(default_factory=...)` for FairseqConfig fields
- `dataclass/initialize.py` — handle MISSING defaults in hydra_init loop
- `checkpoint_utils.py` — `weights_only=False` for torch.load

---

## 9. Lingering Questions

### Resolved
1. ~~**Is speaker embedding the right place to control style?**~~ **No.** Confirmed empirically — D_VECTOR doesn't encode style. Pivoted to F0.
2. ~~**What is beta in the label loss?**~~ beta=1. Loss uses `.sum()` not `.mean()`, so label loss is ~23x smaller in gradient magnitude than recon loss.
3. ~~**Should we validate style presence in embeddings first?**~~ Yes, did this. Separability ratio = 0.88 (not separable).

### Still Open

1. **How do we make F0 control DP-compatible?** This is the central research question. See Section 10.1 below for a full analysis.

2. **Did the controllable VAE ever work with OpenVoice?** If Joe saw audible style differences with OpenVoice embeddings, the embedding-based approach might work for some VC systems. Need to ask.

3. **What F0 features are speaker-identifying?** Mean F0, F0 range, speaking rate, and intonation patterns are all biometric. We need to understand which features carry identity vs. style to design the right DP mechanism.

4. **Do we need the full 11 styles?** Longform (3 samples) and singing (3 samples) are useless. essentials (80 samples) is marginal. Probably should reduce to 7-8 styles for cleaner training.

5. **How does the noise budget compose across embedding + F0?** If we apply DP noise to both speaker embedding AND F0 features, the total privacy cost combines via composition. What's the right epsilon split?

---

## 10. Novelty & Paper Potential

### 10.1 The DP-Compatible F0 Problem (Key Research Question)

**The gap in our current system:**

Right now, the pipeline has two independent pieces:
1. **Speaker embedding** → VAE + DP noise → anonymized embedding (has formal privacy guarantee)
2. **F0 contour** → deterministic preset transform → modified pitch (no privacy guarantee)

The F0 contour is **biometric information**. Your pitch patterns, speaking rhythm, and intonation help identify you as a speaker. If we apply DP noise to the speaker embedding but leave F0 unprotected (or only apply a fixed transform), an attacker could potentially re-identify the speaker from the F0 alone — defeating the privacy guarantee on the embedding side.

**What "DP-compatible F0 control" means:**

Instead of hardcoded presets like `pitch_shift=1.25`, we need a mechanism where:
- F0 features pass through a noise mechanism with a formal privacy guarantee (calibrated epsilon)
- Style can still be controlled by overriding specific dimensions
- The privacy budget accounts for BOTH the embedding and F0 channels

**Three possible approaches:**

**Approach A: F0 statistics through the existing VAE**
- Extract F0 summary statistics (mean, std, range, slope) from the source audio — say 4-6 features
- Concatenate with the 256-dim speaker embedding → 260-262 dim input
- Train a single VAE on the combined representation
- First K latent dims → style labels (which now have F0 information to learn from!)
- DP noise applied to the full latent space
- Reconstruct both embedding + F0 stats; use reconstructed stats to reshape the F0 contour
- **Pros:** Single privacy budget, one model, clean architecture
- **Cons:** Different feature scales (unit-norm embedding vs. raw Hz F0), may need careful normalization

**Approach B: Separate F0 VAE**
- Train a second, smaller VAE specifically on F0 statistics
- Apply DP noise independently to each VAE
- Use composition theorem to compute total privacy cost
- **Pros:** Each model focuses on its domain, easier to tune
- **Cons:** Two models to maintain, composition increases total epsilon

**Approach C: Direct DP mechanism on F0 statistics**
- Skip the VAE for F0 — just L2-clip the F0 stats and add Gaussian noise directly
- Override specific stats with style targets (e.g., force mean_F0 = 200 Hz for "happy")
- Reconstruct the F0 contour from the (noisy) stats
- **Pros:** Simplest implementation, no training needed
- **Cons:** Less expressive, can't learn nuanced style-identity disentanglement

**Recommendation for discussion with Joe:** Approach A is the most elegant and publishable — a single VAE that jointly protects identity (via embedding) and prosody (via F0 stats), while separating style-controllable dims from identity dims. The key experiment would be: do F0 statistics form separable clusters by style? (We already know the answer should be yes — we proved F0 control produces audible differences.)

**What this enables for a paper:**
- "We show that speaker embeddings alone are insufficient for style control (separability ratio 0.88)"
- "We demonstrate that F0 prosody features carry style information that is audibly distinguishable"
- "We propose a joint embedding-prosody VAE that provides DP guarantees across both channels while maintaining controllable style"
- This would be a genuine contribution — most voice anonymization work ignores prosody as an identity channel

---

### Core Contribution
**Controllable differentially private voice conversion** — to our knowledge, no prior work combines:
- Differential privacy on speaker embeddings
- Explicit control over perceptual attributes in the anonymized output
- A VAE architecture that separates controllable (labeled) and free (identity/privacy) latent dimensions

### Why This Matters
Standard DP voice anonymization destroys all speaker information indiscriminately. Controllable DP-VC lets you *choose* what to reveal: "anonymize the speaker, but make them sound happy" or "anonymize but preserve the emotional tone." This has applications in:
- **Call center anonymization:** Strip identity but preserve customer sentiment
- **Witness protection recordings:** Change voice but maintain emotional authenticity
- **Accessible media:** Re-voice content with specific style properties

### Paper Framing Ideas
1. **"Prosody-Aware Differentially Private Voice Conversion"** — argue that existing DP voice anonymization leaks identity through prosody, then show a joint embedding+F0 mechanism that protects both while enabling style control
2. **"Controllable Differential Privacy for Voice Conversion"** — frame as a privacy-utility tradeoff where "utility" includes perceptual style control
3. **"Expressive Voice Anonymization with Formal Privacy Guarantees"** — emphasize the practical application angle

### Key Results to Include
- **Negative result (important!):** Speaker embeddings (D_VECTOR) don't encode style. Separability ratio = 0.88. This is worth reporting — it tells the community that embedding-only style control doesn't work.
- **Positive result:** F0 prosody manipulation produces audibly distinct styles through the VC pipeline. Style lives in prosody, not in speaker embeddings.
- **The gap:** F0 is an unprotected identity channel in current DP voice anonymization systems.

### What Would Strengthen a Paper
- **Quantitative style evaluation:** Use a pretrained emotion classifier on outputs to measure if controlled styles are detectable
- **Speaker verification experiments:** Show that anonymization actually reduces speaker re-identification accuracy
- **F0-based re-identification attack:** Demonstrate that F0 alone can re-identify speakers even after embedding anonymization — this motivates the need for F0 protection
- **Privacy-utility curves:** Plot speaker verification accuracy vs. emotion classification accuracy at different noise levels, for embedding-only vs. embedding+F0 protection
- **Joint VAE ablation:** Compare Approach A (joint VAE) vs. Approach B (separate VAEs) vs. Approach C (direct mechanism)
- **Comparison with naive approach:** Show that just adding noise (without style control) cannot achieve the same style preservation

### Related Work to Position Against
- Voice Privacy Challenge (VPC) 2020-2024 — DP voice anonymization baselines (embedding-only, no F0 protection)
- SpeechFlow / NaturalSpeech — controllable speech synthesis (but no privacy)
- FHVAE / SpeechSplit — disentangled speech representations (but no DP)
- Prosody-based speaker recognition literature — establishes that F0 IS identifying (motivates our work)

---

## 11. Meeting Prep Notes (for April 10)

### What to Demo
- The end-to-end pipeline works: extract → train → infer → audible speech
- HuBERT content encoder is now functional (was broken, produced noise)
- **F0-based style control produces audible differences** (validated April 9)
- Can play back original Trump audio → baseline VC → happy/sad/whisper/laughing variants

### What to Discuss
- **ControlVC D_VECTOR doesn't encode style** — confirmed empirically (separability ratio 0.88) and by Joe ("some systems treat F0 differently"). Not a dead end for the project, just the wrong VC system for style control.
- **OpenVoice is the right target for controllable style** — its embedding captures F0/prosody, so the controllable VAE should produce audible differences. This was the original plan per Joe.
- **Concrete next step:** Re-run the Expresso extraction + VAE training pipeline with OpenVoice instead of ControlVC. The code is already set up for this (`dpvc/openvoice.py` has `get_vae_config()`).
- **F0 prosody control on ControlVC works as a fallback** — we proved direct F0 manipulation produces audible style differences. This could still be useful for ControlVC-based anonymization even if the VAE-based approach moves to OpenVoice.
- **DP question for longer term:** If we do both embedding-based style control (OpenVoice) AND F0 manipulation, how do the privacy budgets compose? Is there a unified approach?
- **The fairseq compat situation** is fragile (monkey-patches for Python 3.11). Worth discussing whether to pin Python 3.10 or migrate to torchaudio's HuBERT.

### Joe's Feedback (April 9, pre-meeting)
> "Some systems treat f0 differently and so randomizing the speaker embedding doesn't sound like a big change. OpenVoice embeds the f0 profile in the speaker embedding, so with OpenVoice you do tend to get audible differences. I think we discussed doing that with OpenVoice (not ControlVC) since OpenVoice has been the easiest to work with in general."

**Implication:** The controllable VAE architecture is sound — the problem is ControlVC's D_VECTOR, not the approach. Switching to OpenVoice should produce audible style differences because its embedding captures F0/prosody.

### Prior Meeting Notes (March 18 call — Expresso plan)

**The agreed-upon plan was always OpenVoice + Expresso:**
1. **Extraction:** Read Expresso wav files, extract speaker embeddings using the OpenVoice wrapper, save embeddings + style labels to a .pt file
   - Reference: `controllable_vae` branch → `examples/openvoice_extract_commonvoice_features.py`
2. **Train VAE with labeled features:** For K labeled features, force the first K latent dims to match the labels via MSE loss during training
   - Reference: `controllable_vae` branch → `examples/openvoice_train_vae_features.py`

**The mechanism (from Joe):**
- Latent representation has N dimensions (e.g., 8)
- If we have K labeled features (e.g., age, gender, accent), the first K dims are forced to equal those labels
- Training loss includes MSE between feature values and corresponding latent dims
- At inference, override those K dims to control the output

**Key URLs from that meeting:**
- Expresso dataset: https://speechbot.github.io/expresso/
- NaturalSpeech3 extraction example: `controllable_vae` branch → `examples/naturalspeech3_extract_commonvoice...`
- OpenVoice extraction: `controllable_vae` branch → `examples/openvoice_extract_commonvoice_features.py`
- OpenVoice VAE training: `controllable_vae` branch → `examples/openvoice_train_vae_features.py`

**What we did instead:** Built the pipeline for ControlVC (which we now know doesn't embed F0 in the speaker embedding). Need to redo with OpenVoice as originally planned.

### 3.7 OpenVoice Expresso Extraction (April 9, branch `feat/openvoice-expresso`)

Per Joe's feedback and the original plan, pivoted to extracting Expresso embeddings with OpenVoice instead of ControlVC. OpenVoice embeds F0/prosody in its speaker embedding, so style control via the controllable VAE should produce audible differences.

**Changes:**
- `dpvc/openvoice.py` — Rewrote `extract_embedding()` to call `tone_color_converter.extract_se()` directly, bypassing OpenVoice's `get_se()` which runs VAD-based audio splitting. The VAD splitting asserts `num_splits > 0` (requires ~10s of speech after VAD), causing 75% of short Expresso utterances to fail. Direct extraction works on any length audio.
- `dpvc/__init__.py` — Uncommented and fixed OpenVoiceWrapper export
- `examples/openvoice_extract_expresso.py` — New extraction script adapted from ControlVC version:
  - Uses pandas for parquet loading (bypasses HuggingFace datasets library issues with incomplete cache)
  - Falls back to HuggingFace `load_dataset` when no `--parquet-dir` given
  - Same one-hot style encoding (+1 active, -1 others) across 11 styles
  - ~15-20 samples/sec on CPU (faster than ControlVC extraction)

**Extraction results:**
- 8,712 total samples across 9 parquet files (only 9 of 12 downloaded — 3 missing shards)
- 0% skip rate (vs. 75% with the old VAD-based extraction)
- Embedding shape: [N, 256, 1] (256-dim, same as ControlVC D_VECTOR)
- Saved to `embeddings/openvoice_expresso_emb.pt`

**Installation notes:**
- OpenVoice installed via `pip install git+https://github.com/myshell-ai/OpenVoice --no-deps` (PyPI package doesn't exist)
- Manual deps: unidecode, inflect, cn2an, pypinyin, jieba, eng_to_ipa, langid, whisper-timestamped
- Checkpoint auto-downloads to `~/.cache/openvoice_checkpoint/` (122 MB)

### What's Committed vs. Uncommitted
- **Committed (feat/controlvc):** VAE port, Expresso pipeline scripts, portability fixes, .gitignore
- **Committed (feat/f0-style-control):** F0 transform in controlvc.py, anonymizer.py, wrapper.py; inference CLI with F0 presets; fairseq Python 3.11 compat patch
- **Committed (feat/openvoice-expresso):** OpenVoice extraction script (initial version)
- **Not committed:** openvoice.py VAD bypass fix, __init__.py export fix, extraction script pandas update, WORKLOG.md, generated artifacts

---

## 12. Reproduction Commands

```bash
# Setup
cd /Users/steve/UVM-plaid/dp-vc
source .venv/bin/activate
export PYTHONPATH=/Users/steve/UVM-plaid/dp-vc

# ===== OpenVoice Pipeline (recommended) =====

# 1. Extract OpenVoice embeddings from Expresso (~8 min on CPU)
python examples/openvoice_extract_expresso.py \
  --output embeddings/openvoice_expresso_emb.pt \
  --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/9fb79a189698de3255eff48edd2bc0d9e487adc0/read

# 2. Train controllable VAE on OpenVoice embeddings
python examples/controlvc_train_vae_expresso.py \
  --embeddings embeddings/openvoice_expresso_emb.pt \
  --output embeddings/openvoice_vae_expresso.pt \
  --epochs 2000 --lr 1e-4

# 3. Run controllable inference (TODO: adapt for OpenVoice)
# python examples/openvoice_infer_controllable.py ...

# ===== ControlVC Pipeline (F0 style control) =====

# 1. Extract ControlVC embeddings from Expresso (~10 min on CPU)
export HF_HUB_DISABLE_XET=1
python examples/controlvc_extract_expresso.py \
  --repo-root /Users/steve/repos/control-vc \
  --output embeddings/controlvc_expresso_emb.pt \
  --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/9fb79a189698de3255eff48edd2bc0d9e487adc0/read

# 2. Run F0-based style inference (no VAE needed for F0 control)
python examples/controlvc_infer_controllable.py \
  --repo-root /Users/steve/repos/control-vc \
  --source examples/trump_0.wav \
  --out output/trump_happy.wav \
  --style happy --noise-level 0.5
```
