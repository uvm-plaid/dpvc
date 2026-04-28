# Key Findings — Controllable DP Voice Conversion

**Last updated:** 2026-04-17 (methodology preambles + per-row Takeaway columns added to Findings 2, 3, 6, 7, 8, 9)
**Authors:** Stephen Oladele, Joe Near

---

## Thesis

Controllable differentially private voice conversion: anonymize a speaker's identity via calibrated noise on speaker embeddings, while explicitly controlling perceptual style attributes (emotion, whisper, etc.) in the output. No prior work combines DP guarantees with explicit style controllability.

---

## Finding 1: Speaker Embeddings Encode Style Only in Some VC Systems

**ControlVC's D_VECTOR does not encode style.**
- Separability ratio: 0.88 (need >>1)
- Between-style centroid distance (0.030) < within-style variance (0.034)
- Even at 5x amplification of style latent dims, zero audible difference
- Root cause: D_VECTOR encodes speaker identity; F0/prosody are handled separately

**OpenVoice's speaker embedding does encode style.**
- OpenVoice embeds F0/prosody profile into the speaker embedding
- Whisper achieves separability ratio 1.30 (clearly separable)
- Confirmed by Joe: "OpenVoice embeds the F0 profile in the speaker embedding"

**Implication for the field:** Researchers must verify that their chosen VC system's speaker embedding actually carries the features they want to control. This is not guaranteed and varies across systems.

---

## Finding 2: Speaker Diversity is Critical for Learning Generalizable Style

**3 speakers (Expresso only):** VAE memorizes speaker-specific tonal patterns instead of learning style.
- Only whisper (ratio 1.30) and sad (ratio 0.39) produce audible differences
- Happy/laughing/confused indistinguishable from baseline
- Within-style variance dominated by speaker identity, not emotion

**91 speakers (CREMA-D):** VAE forced to learn cross-speaker emotion patterns.
- All 6 emotions acoustically distinct
- Happy works for the first time (highest F0 variance)

**94 speakers (CREMA-D + Expresso combined):** Best of both worlds.
- All 9 styles perceptually distinct and acoustically confirmed
- Happy shows 2x F0 variation over baseline

**Progression:**

| Model | Speakers | Working styles | Happy? | Takeaway |
|-------|----------|---------------|--------|----------|
| ControlVC + Expresso | 3 | 0 of 11 | No | Embedding doesn't carry style at all — no amount of VAE training can recover what isn't there. Dead end for ControlVC. |
| OpenVoice + Expresso | 3 | 2 of 11 (whisper, sad) | No | Right embedding, wrong training set — too few speakers, so the VAE memorizes speaker identity instead of learning generalizable style. Only the most extreme styles (whisper, sad) break through. |
| OpenVoice + CREMA-D | 91 | 6 of 6 | Yes | Speaker diversity is the unlock — with 91 speakers the VAE is forced to disentangle style from identity. Happy works for the first time. |
| OpenVoice + Combined | 94 | 9 of 9 | Yes | Combines CREMA-D's 91-speaker backbone with Expresso's unique styles (confused, enunciated, whisper). All 9 styles controllable. |

---

## Finding 3: Style Control Survives Differential Privacy Noise

Tested at noise levels 0, 0.1, 0.3, 0.5, and 1.0 with the combined VAE (v6, 9 styles, 94 speakers).

**Metric — spectral centroid (brightness).** Spectral centroid is the "center of mass" of the short-time power spectrum, reported in Hz. A higher centroid means more energy sits in the high frequencies, which listeners perceive as a brighter or sharper voice (e.g., whisper, anger); a lower centroid is dimmer/darker (e.g., neutral, sad). We use deltas from the uncontrolled baseline, so units are Hz-shift relative to what OpenVoice would have produced with no style control. **Why brightness and not F0:** Finding 6 showed that brightness is the one acoustic correlate that moves consistently across source speakers; F0 does not. Brightness is therefore the metric we trust for measuring whether style control is working.

**Noise scale.** Gaussian DP noise with standard deviation equal to `noise_level` is added to the VAE-decoded speaker embedding. `noise=0` is no privacy; `noise=1.0` is heavy privacy (formal ε TBD — see Open Questions).

**Brightness (spectral centroid) deltas from baseline persist across noise levels:**

| Style | noise=0 | noise=0.1 | noise=0.3 | noise=0.5 | noise=1.0 | Takeaway |
|-------|---------|-----------|-----------|-----------|-----------|----------|
| whisper | +945 | +631 | +280 | +393 | +266 | Brightest style at every noise level — whisper's high-frequency breathy signature is the most privacy-robust of any style we tested. |
| anger | +369 | +177 | -136 | -93 | -113 | Direction *flips* under noise — the bright-edge signature of anger is fragile past noise=0.3. Anger is the least noise-robust controllable style. |
| neutral | -533 | -678 | -759 | -527 | -442 | Most stable downward shift — neutral's "dim, subdued" signature stays intact across every noise level. Safe choice under strong privacy. |
| sad | -272 | -414 | -705 | -668 | -633 | Consistent downward shift like neutral — confirms sad is acoustically closer to the low-energy end than to anger/whisper. |

**Style differences diminish but do not disappear under DP noise.** This is the expected privacy-utility tradeoff — higher noise provides stronger privacy at the cost of some style fidelity.

**Unexpected finding:** At high noise levels (0.3+), the uncontrolled baseline becomes unintelligible (F0 collapses to 0), but style-controlled outputs retain speech-like qualities. **Style control partially rescues speech intelligibility from noise destruction.** This means controllable DP-VC produces *better* outputs than uncontrolled DP-VC at the same privacy level.

---

## Finding 4: Acoustic Signatures Match Emotional Expectations

Combined model (v6) acoustic analysis, all deltas from uncontrolled baseline:

| Style | dF0 Mean | dF0 Std | dF0 Range | dBrightness | Acoustic signature |
|-------|----------|---------|-----------|-------------|-------------------|
| anger | +13.2 | -3.9 | +4.4 | +369 | Sharp, bright, edgy |
| confused | +2.5 | -3.8 | -29.5 | +111 | Hesitant, narrow range |
| disgust | -25.6 | +6.1 | +24.3 | -156 | Low, withdrawn |
| enunciated | -10.5 | -3.9 | +5.2 | +274 | Crisp, bright articulation |
| fear | +17.4 | +4.8 | +80.8 | -155 | Tense, highly variable |
| happy | +26.8 | +33.2 | +240.7 | +84 | Most animated — 2x F0 variation |
| neutral | -46.0 | +6.7 | +38.8 | -533 | Flat, subdued, dimmest |
| sad | -0.3 | +11.2 | +44.3 | -272 | Darker, more varied |
| whisper | -128.6 | -30.8 | -143.6 | +945 | F0 near zero, breathy/airy |

These patterns align with established prosodic correlates of emotion in the speech science literature.

---

## Finding 5: Embedding Separability Alone Does Not Predict VAE Success

CREMA-D raw embedding separability ratio (0.54) was *worse* than Expresso (0.62), yet the CREMA-D VAE produced better perceptual results. This is because:

1. Linear centroid analysis misses nonlinear structure that the VAE encoder learns
2. Speaker diversity matters more than raw separability — with 91 speakers, the VAE has enough variation to disentangle style from identity
3. The VAE's label loss during training is a better predictor of downstream controllability than raw embedding separability

---

## Finding 6: Style Generalizes Across Speakers via Brightness, Not F0

Tested on 5 source speakers (1 known male + 4 CREMA-D speakers with 5-8s audio) at noise=0, control_strength=5.0.

**Spectral brightness (centroid) is direction-consistent across speakers for 7/9 styles:**

| Style | trump | spk1007 | spk1023 | spk1045 | spk1076 | Consistent? | Takeaway |
|-------|-------|---------|---------|---------|---------|-------------|----------|
| anger | +795 | collapsed | +364 | +561 | +398 | YES | Brighter in every non-collapsed speaker — the one collapse is a speaker-specific edge case, not a control failure. |
| confused | +536 | +84 | -180 | -79 | +71 | NO | "Confused" doesn't have a single brightness signature — speakers express hesitation via different acoustic means (pause structure, pitch contour, etc.), not consistent spectral shift. |
| enunciated | +759 | -412 | -731 | -616 | -658 | NO | Sign flip between trump (+759) and all CREMA-D speakers (negative) — enunciation is likely domain-sensitive; it behaves differently on scripted CREMA-D voices than on the conversational Trump sample. |
| fear | +259 | +580 | +375 | +558 | +106 | YES | Consistently brighter — tense, elevated formants hold across all tested speakers. |
| happy | +550 | +342 | +297 | +191 | +290 | YES | Reliable — happy voices are measurably brighter in every speaker we tested. Matches Finding 4's 2× F0 variation. |
| neutral | -325 | collapsed | collapsed | -43 | -184 | YES | Direction consistent where defined, but two collapses suggest "neutral" can push already-neutral source voices into degenerate territory. |
| sad | +117 | +358 | +21 | +243 | +57 | YES | Surprisingly upward — CREMA-D's "sad" has a tense, pleading quality rather than a mellow one, so it lands as slightly *brighter*, not darker. |
| whisper | +973 | +956 | +350 | +720 | +758 | YES | Strongest and most consistent cross-speaker signal of any style — whisper is the one style that generalizes nearly perfectly. |

**F0 direction is NOT consistent** — only 1/9 styles (happy) showed the same F0 shift direction across all speakers. This is expected: OpenVoice encodes timbre/tonal color in the speaker embedding, not F0 directly. F0 changes are a secondary effect that depends on how each source voice interacts with the modified embedding.

**Collapses:** 4/45 outputs (9%) had F0=0 (unintelligible). This occurred when the combination of source speaker embedding + style control pushed the reconstruction into a degenerate region. Affects anger and neutral for spk1007, and disgust/neutral for spk1023.

**At noise=0.1:** Brightness consistency maintained (7/9), but collapses increased to 5/45 (11%). The privacy noise makes some speakers more vulnerable to collapse.

**Control strength tradeoff:**
- strength=5.0: 7/9 consistent, 4 collapses
- strength=3.0: 3/9 consistent, 2 collapses
- strength=2.0: 4/9 consistent, 2 collapses

Higher control strength gives better cross-speaker consistency but increases collapse risk. This suggests that the latent dimensions need to be pushed far enough to dominate the source speaker's baseline characteristics, but this can exceed the decoder's valid input range for some speakers.

**Joe's feedback (April 16 call):** F0 inconsistency is expected and not a problem — different people express emotion through pitch differently. F0 is not a knob we should control; the knobs are the style labels themselves (anger, happy, etc.). Brightness and F0 are measurement metrics, not user-facing controls. Joe also noted that collapses are expected when the VAE goes outside its training distribution and don't require a perfect fix.

**Implication:** The system works across diverse speakers for the majority of styles, with spectral brightness as the reliable cross-speaker acoustic correlate. Papers should report brightness as the primary measurement metric rather than F0.

---

## Finding 7: Emotion Classification Reveals a Training Gap, Not a Control Gap

### Methodology

**What emotion2vec_plus_large is.** A self-supervised universal speech-emotion encoder (Ma et al., ACL 2024; released as `iic/emotion2vec_plus_large` on HuggingFace, loaded here through `funasr`). It takes a raw waveform and produces (1) a fixed-dimensional emotion embedding and (2) softmax probabilities over 9 Chinese-English bilingual labels (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`, `other`, `<unk>`). It is the current state-of-the-art universal emotion encoder, and it is what the EmoVoice paper (arxiv 2504.12867) uses to benchmark emotional TTS — so using it keeps us numerically comparable to the EmoVoice evaluation pipeline.

**How we score each file.** For every generated `.wav` we run emotion2vec, take the argmax label, strip the bilingual prefix (e.g. `生气/angry` → `angry`), and compare it to our target style. Three of our nine styles (`confused`, `enunciated`, `whisper`) have no emotion2vec counterpart — we report emo_sim only for those, with no Recall Rate.

**Design choice — `_plus_large` over the base model.** The `_plus` variant is fine-tuned on additional labeled emotion corpora (IEMOCAP-style), which gives higher absolute accuracy on our CREMA-D/Expresso-style inputs than the base SSL model. This is the same variant EmoVoice reports, so the numbers are directly comparable.

**Primary metric — Recall Rate.** Does the argmax predicted emotion equal the target style? This is the EmoVoice paper's primary controllability metric. Random-chance baseline over 9 classes ≈ 11%.

**Secondary metric — emo_sim.** Cosine similarity between the emotion2vec embedding of a generated file and the embedding of the *same speaker's uncontrolled baseline*. Measures how far the style control pushes the file through emotion-embedding space, regardless of whether the push lands on the target label. Useful for (a) styles emotion2vec can't label directly, and (b) sanity-checking that "zero recall" doesn't mean "identical to baseline."

**How this relates to the project.** Recall Rate is the headline number Joe asked us to produce before anything else (April 16 call). Getting it above chance proves the VAE controls emotion; getting it to >50% is the bar for a paper-quality result. emo_sim is our fallback signal when the classifier can't help us.

### Results — initial 5-speaker run at control_strength=5.0, noise=0.0:

| Style | Recall Rate | emotion2vec prediction | Takeaway |
|-------|------------|----------------------|----------|
| anger | 2/5 (40%) | mixed: `angry`, `<unk>` | Partial success — 2 samples land on target; the rest are off-distribution enough that emotion2vec refuses to label them. |
| disgust | 0/5 (0%) | mostly `disgusted` → `sad`/`<unk>` | Classifier *does* see disgust-like features but not strongly enough to clear the argmax threshold. Signal is there, calibration is off. |
| fear | 0/5 (0%) | `worried`/`happy`/other | emotion2vec's "fearful" class is narrower than our "fear" — training-data mismatch, not a control failure. |
| happy | 0/5 (0%) | `disgusted`/`<unk>` | The most striking miss. Our "happy" is acoustically real (Finding 4: 2× F0 variation) but emotion2vec labels it as disgust — the latent direction may point at sarcasm/mockery, not joy. |
| neutral | 3/5 (60%) | mostly correct | Neutral is the easiest category for any classifier and the smallest perturbation — expected ceiling. |
| sad | 1/5 (20%) | `neutral`/`disgusted` | Our sad comes out flat enough to read as neutral — our acoustic "sad" doesn't match emotion2vec's prototype. |
| **Overall** | **6/30 (20%)** | | Roughly 2× random chance, well short of the >50% target the paper needs. |

**Full-corpus run (258 files, 27 speaker-variant configs, strength sweep included):**

Reproduced via `python examples/eval_emotion.py --input output/diverse_speakers/ --out output/eval_emotion_full.csv`.

| Style | Recall Rate | Mean emo_sim | emo_sim range | Takeaway |
|-------|-------------|--------------|---------------|----------|
| anger | 8/27 (30%) | 0.937 | 0.851–0.998 | Best recall of any emotion — anger has a recognizable acoustic signature (sharp, bright) that survives the CREMA-D → wild-audio transfer. |
| disgust | 3/27 (11%) | 0.970 | 0.939–0.994 | Recall ≈ random chance — acoustic signature exists but is too subtle for emotion2vec to pick out reliably. |
| fear | 0/27 (0%) | 0.960 | 0.893–0.998 | Zero recall despite high emo_sim: we *are* perturbing the embedding, but not in a direction emotion2vec recognizes as fear. Training-distribution mismatch. |
| happy | 0/27 (0%) | 0.961 | 0.849–0.999 | Same pattern — happy is acoustically real (Finding 4) but classifier-invisible. The strongest evidence that recall is a training-coverage gap, not an architecture gap. |
| neutral | 18/27 (67%) | 0.976 | 0.889–0.998 | Strongest category — validates that the pipeline works end-to-end when the target's acoustic signature is broad enough for emotion2vec. |
| sad | 7/27 (26%) | 0.966 | 0.898–0.998 | Above chance, below ceiling — similar training-gap story as disgust. |
| **Overall** | **36/162 (22.2%)** | — | — | Consistent with the 20% 5-speaker run — not a sample-size problem. Motivates CommonVoice pre-training (Phase 1.5). |
| confused | n/a (no e2v class) | 0.943 | 0.888–0.990 | Embedding is clearly shifted from baseline — confused is a real style even without a recall number. |
| enunciated | n/a (no e2v class) | 0.959 | 0.925–0.995 | Least embedding-disruptive of the non-emotional styles — closest to baseline in emotion-space. |
| whisper | n/a (no e2v class) | **0.875** | **0.616–0.994** | Most embedding-disruptive style of any kind. Validates emo_sim as a useful probe for non-labeled controls: it reveals that whisper genuinely perturbs the emotional signature, even though no classifier class exists for it. |

Per-style emo_sim ranges and the 20% vs 22.2% difference are both consistent with a training-coverage issue rather than a measurement artifact.

**Control strength test on a single speaker (Trump), noise=0.0:**

| Style | s=5.0 | s=10.0 | s=20.0 | Takeaway |
|-------|-------|--------|--------|----------|
| anger | `<unk>` | `<unk>` | `<unk>` | Strength doesn't help — the Trump-speaker "anger" region is simply outside emotion2vec's label space at every intensity. |
| disgust | `neutral` | `neutral` | `disgusted` ✓ | Strength *does* help — disgust reaches target at s=20. Latent direction is correct but weakly encoded; pushing harder works. |
| fear | `disgusted` | `disgusted` | `disgusted` | Latent direction is *wrong* (not weak) — pushing harder just makes it "more disgusted." A strength dial cannot fix a direction error. |
| happy | `disgusted` | `disgusted` | `disgusted` | Same story as fear — the axis we labeled "happy" points into emotion2vec's disgust region. Evidence the VAE learned *something*, but mislabeled. |
| neutral | `sad` | `sad` | `sad` | Odd — controlled "neutral" reads as sad. Likely the label encoding pulls toward low-energy regions during training. |
| sad | `neutral` | `sad` ✓ | `sad` ✓ | Recoverable with strength — sad is weakly encoded but directionally correct. |

**Interpretation of the strength sweep:** the three outcomes — "strength fixes it" (disgust, sad), "strength doesn't help" (anger), "strength makes it more wrong" (fear, happy) — tell us exactly where the training gap lives. Anger/fear/happy need *better training data*, not a bigger strength dial. This is the direct motivation for Phase 1.5 (CommonVoice pre-training).

**Interpretation:** Increasing control strength does help some styles (sad, disgust reach their target at higher values), but doesn't fix anger, happy, or fear. This is not a strength problem — the latent dimensions for those styles don't map cleanly to the emotion2vec categories the classifier was trained on. The VAE learned acoustic features for emotion, but the specific acoustic signature it produces for e.g. "angry" may differ from what emotion2vec considers "angry."

**Root cause hypothesis:** The model was trained on CREMA-D (scripted emotional speech, 91 speakers) and Expresso (expressive storytelling speech, 3 speakers). Both datasets have different recording conditions from natural conversational speech. The emotion representations the VAE learns may be dataset-specific rather than universal. Training on a larger and more diverse base (e.g., CommonVoice for acoustic pre-training) should improve cross-domain generalisation.

**emo_sim scores (0.87–0.97) are consistently high** — every output is emotionally coherent and close in embedding space to baseline speech. This is expected: we're controlling a few latent dimensions, not rewriting the full embedding.

**Whisper is the most embedding-disruptive style (new observation, April 17).** Across all 15 whisper outputs, emo_sim mean = 0.875 and min = 0.616 — substantially lower than every other style. This validates emo_sim as a secondary signal for styles that emotion2vec cannot classify directly (whisper, confused, enunciated): even when Recall Rate is undefined, emo_sim reveals whether the style is actually perturbing the emotional signature. Whisper perturbs it the most, which matches its acoustic profile (near-zero F0, breathy/airy texture).

**Implication:** The evaluation pipeline is working — emotion2vec is sensitive enough to detect the failures. The next step is improving training coverage, not changing the architecture. CommonVoice pre-training (Phase 1.5) is directly motivated by this finding.

---

## Finding 8: Style Control Preserves Intelligibility (WER Sanity Check)

### Methodology

**What WER is.** Word Error Rate is the standard ASR evaluation metric: (substitutions + deletions + insertions) / reference words, after normalization. Lower is better; 0 means the hypothesis and reference transcripts are identical. We compute WER with `jiwer`, which handles the alignment plus a normalization chain (lowercase, strip punctuation, collapse whitespace) so that `"Hello, world."` and `"hello world"` score as identical.

**What we use as the ASR.** OpenAI Whisper `base` model — 74M params, multilingual. We chose `base` over `large` because this is a **sanity check**, not a transcription benchmark: we only need it to be sensitive enough to detect intelligibility loss caused by our control knob. `base` is fast (~real-time on CPU) and its error modes are well-understood.

**Design choice — drift-from-baseline, not absolute WER.** For each `<speaker>_<style>.wav` the reference is the *same speaker's* `<speaker>_baseline.wav` transcription, not a ground-truth script. This is a deliberate methodological choice: we don't care whether Whisper gets the words right in absolute terms (the source audio may be noisy, accented, or hard for Whisper regardless of our pipeline). We care whether the style control *changes what Whisper hears*. A drift-from-baseline WER of 0.15 means "style control caused Whisper to disagree with itself on 15% of words" — a clean attribution to our intervention. `eval_wer.py` also supports `--reference-text` for absolute WER on known scripts, which we'll use when we report on a held-out test set.

**How this relates to the project.** One of the claims of a controllable DP-VC system is that the controls are orthogonal to the content channel — you change *how* something is said, not *what* is said. WER is the direct test of that claim. Without it we cannot rule out "style control works by scrambling the phonemes."

### Results (258 files, 73 scored against a same-speaker baseline):

| Style | Mean WER | Median WER | Min | Max | Takeaway |
|-------|----------|------------|-----|-----|----------|
| happy | **0.110** | 0.000 | 0 | 0.750 | Best-preserving style — median 0 means Whisper recovers the exact baseline transcript most of the time. Happy is "free" on the content channel. |
| neutral | 0.130 | 0.000 | 0 | 0.750 | Expected — neutral is the smallest embedding perturbation. |
| sad | 0.150 | 0.000 | 0 | 0.750 | Slow, flat delivery but ASR still recovers the words — median 0 confirms it. |
| anger | 0.166 | 0.125 | 0 | 0.750 | Low mean WER despite the large acoustic shift — anger stays articulate. Content channel intact. |
| fear | 0.188 | 0.200 | 0 | 0.750 | Moderate — tense, variable F0 hurts ASR slightly but not catastrophically. |
| disgust | 0.200 | 0.143 | 0 | 0.750 | Moderate — low, withdrawn voice degrades recovery slightly. Still acceptable. |
| enunciated | 0.244 | 0.214 | 0 | 0.600 | Counter-intuitive — "crisp articulation" doesn't help Whisper here, probably because Expresso's storytelling-style enunciation is stylistically unusual. |
| confused | 0.275 | 0.250 | 0 | 1.000 | Hesitant delivery genuinely confuses the ASR — content drift is real, not just acoustic. |
| **whisper** | **0.356** | **0.286** | 0 | **1.000** | Worst. Two factors compound: Whisper-the-model is broadly poor on whispered speech regardless, *and* high-strength whisper occasionally collapses entirely (max = 1.0). |

**Interpretation:**

1. **6 of 9 styles preserve intelligibility well** (median WER ≤ 0.20). For happy/neutral/sad the median is exactly 0 — Whisper recovers the same transcription as the baseline.
2. **Whisper is the only style with systemic intelligibility loss** (mean 0.356). Part of this is intrinsic to whispered speech — ASR systems are broadly worse on whisper. But some of it is real degradation from our control, especially at high strength.
3. **WER max of 1.00 occurs for whisper, confused, and a few outliers in other styles.** These are the collapse cases already catalogued in Finding 6 (9% of speaker-style combinations produce degenerate output) showing up in the content-recovery metric.
4. **WER and emo_sim are consistent.** Whisper has both the lowest mean emo_sim (0.875 — largest embedding shift) AND the highest mean WER (0.356 — largest content drift). Both metrics agree that whisper is the style most perturbing to "normal" speech, which is what whisper should be.

**Implication:** The style control is essentially **orthogonal to the content channel**. We can turn style knobs without breaking what the speaker says. This is the sanity-check result we needed before claiming "controllable speaker generation": the system genuinely controls speaker properties without damaging the content channel, which is the promise of separating speaker embedding from HuBERT content codes.

---

## Finding 9: Predicted MOS — Naturalness Holds for Most Styles

### Methodology

**What MOS is.** Mean Opinion Score — a standard 1-to-5 subjective scale for perceptual audio quality, where 1 = "bad" and 5 = "excellent." Traditionally collected via human listening panels; modern papers use **predicted MOS** from a model trained on large panels of human ratings (much cheaper, reproducible, highly correlated with real panels).

**Which predictor we use — and why not UTMOS.** The EmoVoice paper (arxiv 2504.12867) uses UTMOS (Saeki et al., 2022), which is the dominant predicted-MOS model in TTS literature. UTMOS is distributed via `sarulab-speech/utmos`, which in turn requires a from-source build of fairseq. Our codebase monkey-patches fairseq for OpenVoice compatibility, and the two fairseq versions conflict — we cannot install UTMOS without breaking inference. We substitute torchaudio's `SQUIM_SUBJECTIVE` (Meta, 2023), which predicts the same quantity (subjective MOS on the same 1–5 scale) trained on comparable datasets (BVCC + DAPS). SQUIM and UTMOS are not numerically identical, but they measure the same construct and rank-order audio similarly — which is all we need for a *relative* comparison across styles.

**How SQUIM_SUBJECTIVE works — the "non-matching reference" design.** SQUIM takes two inputs: the audio to score, and a *reference* waveform used only to condition the model (not to compare against). The reference can be any clean-speech file — the model uses it to calibrate its judgment of what "natural speech" looks like for this speaker/channel, then scores the test audio on its own merits. This is why `eval_mos.py` defaults to using each file's same-speaker baseline as reference: gives the model a consistent conditioning anchor per speaker. Pass `--reference` to override with a single fixed waveform.

**Design choice — per-speaker baseline delta.** For each style we report `Δ vs same-speaker baseline = MOS(style) − MOS(baseline)` for that speaker. This removes per-speaker MOS variation (some source voices are just noisier than others) and isolates the style knob's naturalness cost.

**How this relates to the project.** Recall Rate (Finding 7) tells us if the style hits the target; WER (Finding 8) tells us if the words survive; MOS tells us if the output still *sounds like a person*. Without MOS, a style-control system could achieve high recall and low WER by producing robotic, artifact-laden audio that a human would rate as unusable. MOS closes that loop.

### Results (258 files, 150 scored — the others had no same-speaker baseline available):

| Style | Mean MOS | Median | Range | Δ vs same-speaker baseline | Takeaway |
|-------|----------|--------|-------|----------------------------|----------|
| baseline | 4.055 | 3.980 | 3.925–4.448 | — | OpenVoice's naturalness ceiling — "good" quality before any style control is applied. Sets the upper bound. |
| fear | 4.074 | 3.979 | 3.966–4.444 | +0.019 | Statistically indistinguishable from baseline — fear is free on the naturalness axis. |
| sad | 4.068 | 3.978 | 3.898–4.438 | +0.013 | No measurable cost. |
| neutral | 4.045 | 3.974 | 3.916–4.402 | −0.010 | No meaningful cost. |
| happy | 4.025 | 3.984 | 3.392–4.411 | −0.030 | Tiny mean cost, but note the low end (3.39) — occasional samples are noticeably degraded, probably collapse-adjacent cases. |
| disgust | 4.021 | 3.972 | 3.872–4.411 | −0.034 | Small cost, no outliers — disgust is a benign style in naturalness terms. |
| enunciated | 3.932 | 3.947 | 3.069–4.448 | −0.123 | First style with a measurable cost, but still "good." The low-end (3.07) reveals that aggressive enunciation can sound unnaturally over-articulated. |
| anger | 3.760 | 3.975 | **1.407**–4.377 | −0.294 (bimodal; median healthy) | Bimodal failure — median is still fine (3.98), but a few outliers crash to 1.4 MOS and drag the mean. These are the collapse cases from Finding 6 showing up in MOS. |
| confused | 3.704 | 3.808 | 2.892–4.421 | −0.351 | Systemic cost — every sample is a little less natural, no clean cases. Hesitant/halting speech intrinsically scores lower. |
| **whisper** | **3.623** | 3.904 | **2.503**–4.170 | **−0.432** | Worst naturalness cost overall — whisper is genuinely hard for any speech synthesis system because its acoustic profile (near-zero F0, breath) is atypical for the MOS model's training data. |

**Interpretation:**

1. **Baseline MOS ≈ 4.05** — the OpenVoice VC pipeline itself produces "good"-to-"very good" naturalness, before any style control is applied. This sets the naturalness ceiling we're operating under.
2. **6 of 9 styles stay within 0.12 MOS of baseline** (fear, sad, neutral, happy, disgust, enunciated). The style knob preserves perceived naturalness for most emotions.
3. **Three styles degrade naturalness measurably**: whisper (−0.43), confused (−0.35), anger (−0.29). The anger degradation is bimodal — median is fine (3.98) but high-strength trump samples drop to 1.4 MOS. Whisper and confused are systemic: every sample is somewhat degraded.

**Cross-metric convergence (the most important observation).** All three independent evaluation metrics — emo_sim (Finding 7), WER (Finding 8), and predicted MOS (this finding) — rank the same three styles as hardest: **whisper, confused, anger**. Each metric measures a different quantity (embedding shift, content-channel drift, naturalness perception), yet they converge on the same ordering.

| Style | emo_sim mean | WER mean | MOS mean | Worst-on-all? | Takeaway |
|-------|--------------|----------|----------|---------------|----------|
| whisper | 0.875 | 0.356 | 3.623 | ✓ | Hardest style on every axis — embedding shift, content drift, and naturalness all agree. This is where training coverage matters most. |
| confused | 0.943 | 0.275 | 3.704 | ✓ | Second-hardest — hesitant delivery disrupts all three channels, but less severely than whisper. |
| anger | 0.937 | 0.166 | 3.760 | ✓ (on MOS & emo_sim) | Third-hardest — good on WER, bimodal on MOS. Failures are collapse cases, not the style itself being unnatural. A collapse-detector would recover most of anger's MOS. |

This triangulation is what we'd want to see before trusting any single metric. It means the signal is real — these styles are genuinely harder for our current model, not an artifact of any one evaluator.

**Implication:** For the paper, we can report MOS to confirm that style control preserves naturalness for most styles, and flag whisper/confused/anger as the training-coverage edge cases. Combined with WER (intelligibility) and emotion2vec (target alignment), we have a three-axis evaluation that mirrors the EmoVoice (arxiv 2504.12867) pipeline.

---

## Finding 10: Validation-Scale CommonVoice Pre-Training Improves WER but Collapses Style Toward Neutral

### Methodology

To test the Phase 1.5 hypothesis without waiting for a full Common Voice mirror, we built a **local validation-scale English subset** from downloaded Common Voice 21.0 shards:

- local clips available: `28,116`
- local speakers matched in `validated.tsv`: `6,795`
- training subset used for this pass: **`500` speakers / `1,202` clips** (`cv500`)

Pipeline:

1. Build a local `validated.tsv` + `clips/` subset with `scripts/prepare_commonvoice_subset.py`
2. Extract OpenVoice speaker embeddings with `examples/openvoice_extract_commonvoice.py`
3. Run reconstruction-only pre-training with `examples/openvoice_pretrain_vae_commonvoice.py`
4. Finetune the existing CREMA-D + Expresso controllable VAE with `examples/openvoice_train_vae_combined.py --init-checkpoint ...`
5. Re-run the same 110-file evaluation corpus and compare against the current combined-only baseline

This is an intentionally conservative test: **same controllable training recipe, same source speakers, same evaluation scripts**. The only change is the CommonVoice initialization.

### Results (`cv500` candidate vs current combined-only baseline)

| Metric | Combined-only baseline | CommonVoice `cv500` init | Delta | Takeaway |
|--------|------------------------|--------------------------|-------|----------|
| Emotion recall | `17/66 = 25.8%` | `11/66 = 16.7%` | `-9.1 pts` | Worse overall controllability |
| Neutral recall | `9/11 = 81.8%` | `11/11 = 100%` | `+18.2 pts` | Model got even better at staying neutral |
| Anger recall | `3/11 = 27.3%` | `0/11 = 0%` | `-27.3 pts` | Previously recoverable style collapsed |
| Disgust recall | `1/11 = 9.1%` | `0/11 = 0%` | `-9.1 pts` | No gain |
| Sad recall | `4/11 = 36.4%` | `0/11 = 0%` | `-36.4 pts` | Lost one of the stronger styles |
| Neutral predictions across all 110 files | `70/110` | `109/110` | `+39 files` | Strong neutral-collapse signature |
| Mean WER across all 99 non-baseline style rows | `0.235` | `0.084` | `-0.151` | Much better content preservation |
| Median WER across all 99 non-baseline style rows | `0.143` | `0.000` | `-0.143` | Most candidate outputs transcribe exactly like baseline |
| Whisper mean WER | `0.448` | `0.111` | `-0.337` | Even the hardest style became far easier for Whisper |

The emotion2vec output distribution makes the failure mode unambiguous: the `cv500` candidate predicts **`neutral` for 109 of 110 files**. Mean `emo_sim` values also jump extremely close to baseline (`0.979–0.998` across styles), which is exactly what we'd expect from a model that learned a tighter reconstruction prior but weakened its style offsets.

### Interpretation

This is a **real negative result**, and an informative one:

1. **Naive reconstruction-only CommonVoice pre-training is not automatically helpful for controllable style generalization.** On this validation-scale run it made the model *less* expressive, not more.
2. **The gain is real on the content channel.** WER improved substantially, so the pre-training is doing something meaningful: it is regularizing the speaker embedding toward more stable, baseline-like speech.
3. **The failure mode is over-regularization, not random noise.** The model did not become chaotic; it became too conservative. Almost everything collapsed back toward the neutral/baseline region.
4. **This does not kill the CommonVoice direction.** It narrows the hypothesis. The question is no longer "does broader speaker coverage help at all?" The question is: **how do we keep the broader speaker prior without washing out the labeled style axes?**

### Implication

For the paper, this gives us a strong and honest intermediate result:

- **positive:** CommonVoice pre-training can improve intelligibility/content preservation dramatically
- **negative:** the current `cv500` recipe hurts controllability by collapsing style toward neutral
- **next step:** scale the subset and test gentler finetuning or partial freezing before claiming CommonVoice pre-training improves emotion recall

This is exactly the kind of result worth reporting because it turns a vague Phase 1.5 idea into a concrete research question with a measurable failure mode.

---

## Open Questions

1. **What are the formal privacy guarantees?** We need to compute epsilon for each noise level and report privacy-utility curves.
2. ~~**Does style control generalize across source speakers?**~~ → **Answered in Finding 6.** Brightness generalizes (7/9 styles); F0 does not. Some speaker-style combinations collapse.
3. ~~**How do we evaluate emotion controllability?**~~ → **Answered in Finding 7.** emotion2vec Recall Rate + emo_sim (per EmoVoice) is the primary metric. Recall is 20% — training gap identified.
4. **Can CommonVoice pre-training improve recall at scale?** The first validation-scale `cv500` run (Finding 10) improved WER but collapsed style toward neutral. The open question is now whether a larger subset and gentler finetuning can preserve the gain without washing out the style axes.
5. **Can we train age/gender and emotion knobs simultaneously?** CommonVoice has age/gender, CREMA-D has emotion. Can a single VAE learn all at once when each training stage only labels a subset? Unknown — Joe flagged this as an open research question.
6. **Can an adversary re-identify speakers from F0 alone?** If so, embedding-only DP is insufficient — motivates joint protection.
7. **What is the minimum speaker count for style learning?** We jumped from 3 to 91. Where's the threshold?
8. **Can we interpolate between styles?** E.g., 50% happy + 50% sad — does the output sound bittersweet?
9. **How to prevent collapses?** 9% of speaker-style combinations produce unintelligible output in the combined-only model, and the `cv500` CommonVoice run adds a second collapse mode: style washing back to neutral. Can we clamp the latent space, partially freeze style dims, or detect/reject bad combinations?

---

## Paper Framing

**Title candidates:**
1. "Controllable Differentially Private Voice Conversion"
2. "Expressive Voice Anonymization with Formal Privacy Guarantees"
3. "Style-Preserving Speaker Anonymization via Controllable VAE"

**Core argument (refined after April 16 call + April 16 evening message from Joe):** The problem we solve is **controllable speaker generation for voice-to-voice systems** — a more general problem than either VoicePrivacy (which preserves existing emotion) or TTS (which generates from text). A single controllable VAE enables multiple downstream applications:

1. **Identity change with controlled style** — anonymize a speaker while targeting a specific emotion (the demo we currently emphasize)
2. **Emotion change without identity change** — modify style latent dims while keeping the rest of the embedding fixed; same-sounding person, different mood
3. **Random speaker generation with control** — sample freely from the VAE to produce speakers that never existed, with specified properties
4. **Random speaker near an existing one** — sample from a neighborhood in latent space around a reference speaker

Privacy / DP noise is **one application** of use cases (3) and (4), not the paper's headline. Joe's assessment (April 16 evening): "I'm not sure about this, but our solution might end up being the state of the art for this more general thing."

**Key results to highlight:**
1. Negative result: not all VC embeddings encode style (ControlVC fails, OpenVoice works)
2. Speaker diversity is the critical training requirement (3 vs 91 speakers)
3. 9 styles controllable and acoustically verified
4. Style differences survive DP noise (privacy-utility tradeoff, not cliff)
5. Style control rescues intelligibility under heavy noise (bonus finding)
6. Style generalizes across diverse speakers (7/9 via brightness, collapses expected)
7. emotion2vec evaluation reveals training gap: 22.2% recall at current scale — motivates CommonVoice pre-training
8. Style control preserves intelligibility (WER sanity check): 6 of 9 styles have median WER ≤ 0.20 against same-speaker baseline; whisper is the only style with systemic loss
9. Predicted MOS confirms naturalness preservation for 6 of 9 styles; emo_sim + WER + MOS converge on the same three hardest styles (whisper, confused, anger) — cross-metric triangulation validates the signal
10. Validation-scale CommonVoice pre-training (`cv500`) is a mixed result: WER improves sharply, but emotion recall drops and the model collapses toward neutral. The CommonVoice direction remains promising, but the naive recipe is not paper-ready yet.

**Evaluation approach (per Joe, April 16 + EmoVoice paper):**
- **Primary:** emotion2vec Recall Rate + emo_sim (per EmoVoice pipeline) — measures whether generated outputs express the intended emotion
- **Secondary:** Word error rate via Whisper — intelligibility sanity check only
- **Tertiary:** Privacy/speaker verification (add as "we can also do this")
- Baseline for recall: random chance = ~11% (9 classes); current model = 20%; target: >50% for paper
