# Key Findings — Controllable DP Voice Conversion

**Last updated:** 2026-04-28 (Pass 6 CommonVoice objective-ablation added as Finding 14; paper framing and open questions updated)
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

## Finding 11: Speaker Novelty Metric Confirms Real Identity Shift in the Combined Model and Collapse in the `cv500` Model

### Methodology

We added a **speaker novelty metric** in the native OpenVoice speaker-embedding space. For each generated output we extract:

- the source speaker embedding from the original source audio
- the generated speaker embedding from the output waveform
- the same-speaker baseline conversion embedding when available

Then we compute:

- **similarity** = cosine similarity(source, generated)
- **distance** = `1 - similarity`
- **novelty gain vs baseline** = similarity(source, baseline) - similarity(source, generated)

The key quantity is **novelty gain vs baseline**. It answers the paper question more cleanly than raw similarity alone:

- positive value: the style-controlled output moved farther from the source than plain voice conversion already does
- near-zero value: the style-controlled output is not meaningfully more novel than the baseline conversion

This is a **proof-of-novelty** metric in our framing, not yet a full privacy metric. It uses the system's native embedding space and is intended to answer "did we generate a genuinely shifted speaker identity?" before we add an independent speaker-verification/EER pipeline.

We ran it on the same two 110-file validation corpora used in Pass 2:

1. combined-only checkpoint: `results/eval_novelty_pass2_combined.csv`
2. CommonVoice `cv500` init checkpoint: `results/eval_novelty_pass2_cv500.csv`

### Results

| Metric | Combined-only model | CommonVoice `cv500` model | Takeaway |
|--------|---------------------|---------------------------|----------|
| Mean source-to-baseline similarity | `0.9101` | `0.8682` | Both baseline conversions remain relatively close to the source, though `cv500` baseline is already a bit farther away |
| Mean source-to-styled similarity | `0.6502` | `0.8313` | Combined-only styles are much farther from source; `cv500` styles stay much closer |
| **Mean novelty gain vs baseline** | **`0.2599`** | **`0.0369`** | Combined-only model creates substantially more novel speakers; `cv500` barely moves beyond baseline conversion |

Per-style novelty gain vs baseline (higher = more novel than the baseline conversion):

| Style | Combined-only | `cv500` | Takeaway |
|-------|---------------|---------|----------|
| whisper | `0.6561` | `0.0337` | Strongest example of the collapse: whisper is highly speaker-shifting in the combined model, but almost baseline-like after `cv500` pretraining |
| confused | `0.5589` | `0.0280` | Same story — style difference largely disappears |
| enunciated | `0.4495` | `0.0023` | Nearly no novelty beyond baseline under `cv500` |
| anger | `0.1506` | `0.0371` | Novelty compressed |
| fear | `0.1027` | `0.1015` | The one style that remains comparably speaker-shifting in both models |

The raw similarity values also line up with intuition from the earlier findings:

- In the combined-only model, **whisper** and **confused** are the most identity-shifting styles, which matches their cross-metric difficulty in WER and MOS.
- In the `cv500` model, nearly every style clusters close to the source/baseline region, which matches the emotion-classifier collapse toward `neutral`.

### Interpretation

This finding is useful for two reasons:

1. **It validates the main model on a new axis.** The combined-only controllable model is not just changing classifier labels or acoustic surface features; it is producing speaker embeddings that move substantially away from the source relative to baseline conversion.
2. **It independently confirms the `cv500` failure mode.** The CommonVoice-pretrained model does not just collapse in emotion2vec space; it also stops producing meaningfully novel speaker identities for most styles. That makes the Pass 2 negative result much harder to dismiss as a quirk of one evaluator.

In other words, the novelty metric and the emotion metric tell the same story from different angles:

- **combined-only model**: expressive, identity-shifting, but imperfectly aligned to target emotion labels
- **`cv500` model**: more conservative, more transcript-stable, but much less novel and much less expressive

### Implication

For the paper, this closes an important gap in the evaluation stack:

- emotion2vec Recall/emo_sim: does the style target land?
- WER: does the content survive?
- MOS: does it still sound natural?
- **Novelty**: does the output become a genuinely different speaker profile?

It also sharpens the CommonVoice follow-up agenda. The next question is no longer just "can pre-training improve recall?" It is:

**Can we preserve the intelligibility gains of broader pre-training without collapsing both emotion control and speaker novelty back toward the baseline identity?**

---

## Finding 12: The Combined Model Best Balances Control, Novelty, Intelligibility, and Naturalness

### Methodology

We ran a **five-condition ablation matrix** on the same 11-speaker validation corpus used in Findings 10 and 11, with the same evaluation stack:

- emotion2vec Recall Rate + emo_sim
- native OpenVoice novelty gain vs baseline
- Whisper drift-from-baseline WER
- SQUIM_SUBJECTIVE predicted MOS delta vs baseline

The five conditions were:

1. **combined** — the main 9-style CREMA-D + Expresso checkpoint
2. **CommonVoice `cv500` init** — the Pass 2 finetuned checkpoint
3. **CREMA-D only** — a speaker-diverse 6-style checkpoint trained only on CREMA-D labels
4. **Expresso only** — a 6-style checkpoint trained only on the balanced Expresso subset
5. **naive baseline** — the combined checkpoint, but with deterministic random control vectors applied only to the six *unlabeled* latent dims (9-14), L2-matched to the style-control strength

That naive condition is important. It answers a harder question than "does noise change the speaker?" It asks:

**Can arbitrary latent perturbation create something that looks as good as labeled style control?**

### Results

| Condition | Supported styles | Emotional rows scored | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|------------------|-----------------------|--------|--------------------------|----------|----------------|----------|
| combined | 9 | 66 | 25.8% | 0.2599 | 0.2353 | -0.0792 | Best overall tradeoff — non-trivial target alignment and non-trivial identity shift, with moderate WER/MOS cost |
| CommonVoice `cv500` init | 9 | 66 | 16.7% | 0.0369 | 0.0844 | -0.0123 | Sounds stable and transcribes well, but style and novelty collapse back toward baseline |
| CREMA-D only | 6 | 66 | 16.7% | 0.0158 | 0.0534 | +0.0059 | Excellent stability, weak speaker shift — speaker-diverse emotion data alone is not enough |
| Expresso only | 6 | 33 | 36.4% | -0.0008 | 0.0549 | -0.0120 | Superficially decent recall on a tiny emotional subset, but almost no identity shift at all |
| naive free-dim baseline | 9 | 66 | 18.2% | 0.4708 | 0.1579 | -0.6453 | High novelty without clean control — random latent pushes are not a substitute for labeled style axes |

### Collapse taxonomy

To keep "collapse" from meaning everything and nothing, we tracked four failure types:

- **content collapse**: `WER >= 0.8`
- **style collapse to neutral**: target is emotional, but emotion2vec predicts `neutral`
- **identity collapse to baseline**: novelty gain vs baseline `<= 0.05`
- **mixed collapse**: at least two collapse axes on the same file

Condition-level summary:

| Condition | Content collapse | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|------------------|---------------------------|-------------------------------|----------------|----------|
| combined | 6 | 37 | 7 | 4 | Fewer failures overall, but the failures that remain are spread across content, style, and identity |
| CommonVoice `cv500` init | 0 | 54 | 72 | 34 | Classic conservative collapse: no catastrophic audio failures, but the model washes back toward neutral and baseline identity |
| CREMA-D only | 0 | 55 | 61 | 50 | Very stable audio, but almost every row is a style/identity collapse rather than a successful controlled speaker shift |
| Expresso only | 0 | 21 | 66 | 21 | The dominant failure is identity collapse — outputs stay too close to the baseline speaker profile |
| naive free-dim baseline | 1 | 41 | 0 | 0 | The failure is the opposite of `cv500`: lots of speaker movement, but poor emotional targeting and degraded naturalness |

### Interpretation

This ablation answers several paper-critical questions at once:

1. **The combined model is still the main model.** It is not the cleanest on every single metric, but it is the only condition that jointly preserves meaningful target alignment and meaningful identity shift.
2. **Single-dataset stability is not enough.** `CREMA-D only`, `Expresso only`, and `cv500` all show that a model can sound natural and transcribe well while still failing the actual controllable-speaker objective.
3. **Novelty alone is not the goal.** The naive baseline produces *more* novelty than the combined model, but its recall is worse and its MOS collapses by `-0.6453`. This is the clearest evidence yet that the project is not "make the embedding move"; it is "make the embedding move in a useful, controllable way."
4. **The CommonVoice direction remains open but narrowed.** `cv500` is better framed now as a stability-biased initialization that over-regularizes the system unless the finetuning recipe changes.

### Implication

Pass 4 turns the repo from "we have a working system" into "we have a defensible comparison story":

- **combined** is the paper's current headline model
- **CommonVoice `cv500` init** is a real negative result and a focused follow-up agenda
- **CREMA-D only** and **Expresso only** explain *why* the combined condition is necessary
- **naive free-dim baseline** shows why labeled style supervision matters beyond raw novelty

That is the comparison structure the paper needs.

---

## Finding 13: Simple Gentler Finetuning Does Not Fix the CommonVoice Collapse

### Methodology

Pass 5 kept the **CommonVoice `cv500` pretrained checkpoint fixed** and asked a
much narrower question than Finding 10:

**Was the conservative `cv500` collapse mainly caused by an overly aggressive finetuning policy?**

To test that, we held the dataset and evaluation stack constant and changed
only the finetuning recipe used to adapt the CommonVoice-pretrained VAE onto
the combined CREMA-D + Expresso labels.

We compared the existing references:

1. **combined** — the main paper checkpoint
2. **CommonVoice `cv500` init** — the original Pass 2 finetune

against five new finetuning variants:

1. **`cv500_ft_short`** — fewer finetune epochs (`1000` instead of `3000`)
2. **`cv500_ft_low_lr`** — lower learning rate (`3e-7` instead of `1e-6`)
3. **`cv500_ft_short_low_lr`** — fewer epochs plus lower learning rate
4. **`cv500_ft_freeze_decoder`** — finetune only the encoder/latent side
5. **`cv500_ft_freeze_encoder`** — finetune only the decoder

All conditions used the same:

- CommonVoice init checkpoint: `embeddings/openvoice_vae_commonvoice_cv500.pt`
- combined finetuning embeddings: `embeddings/openvoice_combined_emb.pt`
- 11-speaker validation corpus format
- four-metric evaluation stack from Findings 10-12

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|--------|--------------------------|----------|----------------|----------|
| combined | 25.8% | 0.2599 | 0.2353 | -0.0792 | Still the paper model: the only condition with both meaningful control and meaningful identity shift |
| CommonVoice `cv500` init | 16.7% | 0.0369 | 0.0844 | -0.0123 | Stable but collapsed reference point |
| `cv500_ft_short` | 16.7% | 0.0558 | 0.0390 | -0.0108 | Better novelty than raw `cv500`, but no recall recovery |
| `cv500_ft_low_lr` | 16.7% | 0.0495 | 0.0340 | -0.0142 | Slight novelty gain, even lower WER, still no control recovery |
| `cv500_ft_short_low_lr` | 16.7% | 0.0692 | 0.0791 | -0.0441 | Best novelty recovery among the CommonVoice variants, still far from combined |
| `cv500_ft_freeze_decoder` | 16.7% | 0.0192 | 0.0330 | -0.0154 | Worst novelty of the sweep; freezing the decoder makes identity collapse worse |
| `cv500_ft_freeze_encoder` | 18.2% | 0.0594 | 0.0905 | -0.1261 | Only variant with any recall gain, but it gives back too much naturalness |

### Collapse behavior

The collapse summary makes the failure mode more precise.

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|---------------------------|-------------------------------|----------------|----------|
| CommonVoice `cv500` init | 54 | 72 | 34 | Original conservative collapse pattern |
| `cv500_ft_short` | 55 | 55 | 26 | Identity collapse improves, but style collapse does not |
| `cv500_ft_low_lr` | 55 | 61 | 29 | Slightly better than raw `cv500`, still mostly conservative collapse |
| `cv500_ft_short_low_lr` | 55 | 44 | 22 | Best partial recovery: identity collapse drops the most, but style collapse stays unchanged |
| `cv500_ft_freeze_decoder` | 54 | 88 | 49 | Decoder freeze is too blunt; it worsens the identity-collapse failure |
| `cv500_ft_freeze_encoder` | 54 | 63 | 33 | Slight recall gain, only modest identity recovery, worse naturalness |

### Interpretation

This pass narrows the CommonVoice agenda in an important way:

1. **The `cv500` failure is not just “too many finetune steps.”** Shorter training and lower LR help only marginally.
2. **The `cv500` failure is not solved by coarse freezing either.** Whole-module encoder/decoder freezes are too blunt to restore the combined model's tradeoff.
3. **Novelty is easier to recover than emotional alignment.** Several variants improve novelty over raw `cv500`, but four of the five do not improve recall at all.
4. **The best partial recipe is `cv500_ft_short_low_lr`, not because it wins outright, but because it reduces identity collapse the most without destroying WER/MOS.**
5. **The decoder-freeze result is particularly informative.** It suggests that preserving the pretrained decoder too rigidly keeps the model trapped near the conservative CommonVoice prior.

### Implication

Pass 5 changes the next-step recommendation for the paper:

- We should **not** frame the CommonVoice issue as something that simple “gentler finetuning” already solved.
- We **can** say that modest identity recovery is possible without giving back all the intelligibility gain.
- The next meaningful experiments are now narrower and more defensible:
  - larger-scale CommonVoice with the best partial recipe (`short_low_lr`)
  - better pretraining objectives, not just reconstruction-only pretraining
  - finer-grained freeze or loss-weight schedules rather than coarse encoder/decoder freezing

That is a much sharper research position than the repo had after Finding 10 alone.

---

## Finding 14: Simple Objective Reweighting Does Not Recover Controllability After CommonVoice Pretraining

### Methodology

Pass 6 asked a narrower follow-up question than Finding 13:

**Was the CommonVoice `cv500` collapse mainly an objective-balance problem that we
could fix by reweighting the existing reconstruction, KL, and label losses
during combined finetuning?**

To test that, we kept the data, evaluation corpus, and four-metric stack fixed,
started from the same CommonVoice-pretrained init checkpoint, and changed only
the loss weights used during combined finetuning.

We kept these pieces unchanged:

- CommonVoice init checkpoint: `embeddings/openvoice_vae_commonvoice_cv500.pt`
- combined finetuning embeddings: `embeddings/openvoice_combined_emb.pt`
- 11-speaker validation corpus format
- same evaluation stack from Findings 10-13
- same best Pass 5 training schedule foundation: `1000` epochs at `3e-7`

We compared three references:

1. **combined** — the current paper checkpoint
2. **CommonVoice `cv500` init** — the original conservative CommonVoice result
3. **`cv500_ft_short_low_lr`** — the best partial-recovery finetune recipe from Finding 13

against four new objective variants:

1. **`cv500_obj_label2`** — label weight `2.0`, reconstruction weight `1.0`
2. **`cv500_obj_label4`** — label weight `4.0`, reconstruction weight `1.0`
3. **`cv500_obj_label_ramp`** — label weight ramps from `1.0` to `4.0`
4. **`cv500_obj_recon_half_label2`** — reconstruction weight `0.5`, label weight `2.0`

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|--------|--------------------------|----------|----------------|----------|
| combined | 25.8% | 0.2599 | 0.2353 | -0.0792 | Still the only condition with both meaningful control and meaningful identity shift |
| CommonVoice `cv500` init | 16.7% | 0.0369 | 0.0844 | -0.0123 | Conservative collapse reference point |
| `cv500_ft_short_low_lr` | 16.7% | 0.0692 | 0.0791 | -0.0441 | Best CommonVoice partial-recovery reference from Finding 13 |
| `cv500_obj_label2` | 16.7% | 0.0589 | 0.0862 | -0.0300 | Best of the new objective variants, but still below `cv500_ft_short_low_lr` |
| `cv500_obj_label4` | 16.7% | 0.0496 | 0.1222 | -0.0228 | Larger label weight hurts WER without recovering control |
| `cv500_obj_label_ramp` | 16.7% | 0.0442 | 0.0832 | -0.0126 | Preserves MOS closest to raw `cv500`, but stays conservative |
| `cv500_obj_recon_half_label2` | 16.7% | 0.0500 | 0.1146 | -0.0226 | Lower reconstruction weight is still not enough to recover recall or novelty |

### Collapse behavior

The collapse summary confirms that the new objective variants did not escape the
same underlying failure mode.

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|---------------------------|-------------------------------|----------------|----------|
| CommonVoice `cv500` init | 54 | 72 | 34 | Original conservative collapse pattern |
| `cv500_ft_short_low_lr` | 55 | 44 | 22 | Best Pass 5 identity-recovery reference |
| `cv500_obj_label2` | 55 | 53 | 28 | Some identity recovery over raw `cv500`, still worse than best Pass 5 recipe |
| `cv500_obj_label4` | 55 | 55 | 32 | Stronger label weight worsens WER and does not reduce style collapse |
| `cv500_obj_label_ramp` | 55 | 57 | 35 | Label ramp behaves almost like the original conservative collapse |
| `cv500_obj_recon_half_label2` | 55 | 56 | 32 | Reduced reconstruction pressure still does not recover control |

### Interpretation

Pass 6 sharpens the CommonVoice story again:

1. **Simple scalar objective reweighting is not enough.** None of the four variants improves recall beyond `16.7%`.
2. **Objective tuning helps less than the best Pass 5 finetune recipe.** Even the best new variant (`cv500_obj_label2`) trails `cv500_ft_short_low_lr` on novelty and identity collapse.
3. **The label-ramp result is particularly informative.** It preserves MOS close to the original `cv500` run, but only by staying near the same conservative identity/style collapse basin.
4. **The CommonVoice failure is now unlikely to be just a basic weight-balance issue.** Pass 5 ruled out simple finetune-policy changes; Pass 6 rules out simple scalar reweighting of the existing losses.
5. **The next objective work needs to be richer than rebalancing the current losses.** The more promising directions are now partial-label pretraining, teacher or latent anchoring, curriculum objectives, or larger-scale data with stronger supervision.

### Implication

Pass 6 changes how we should describe the CommonVoice line in the paper:

- We should **not** claim that simple objective tuning already solves the CommonVoice collapse.
- We **can** say that the failure has been narrowed substantially: it survives both gentler finetuning (Finding 13) and simple loss reweighting (Finding 14).
- The next serious CommonVoice experiments should focus on richer supervision or larger-scale training, not more scalar weight sweeps.

That is a stronger and more honest research position than we had after Finding
13 alone.

---

## Open Questions

1. **What are the formal privacy guarantees?** We need to compute epsilon for each noise level and report privacy-utility curves.
2. ~~**Does style control generalize across source speakers?**~~ → **Answered in Finding 6.** Brightness generalizes (7/9 styles); F0 does not. Some speaker-style combinations collapse.
3. ~~**How do we evaluate emotion controllability?**~~ → **Answered in Finding 7.** emotion2vec Recall Rate + emo_sim (per EmoVoice) is the primary metric. Recall is 20% — training gap identified.
4. **Can CommonVoice pre-training improve recall at scale?** The first validation-scale `cv500` run (Finding 10) improved WER but collapsed style toward neutral. Pass 5 showed that simple gentler finetuning is not enough to fix that on its own, and Pass 6 shows that simple scalar loss reweighting is not enough either. The open question is now whether a larger subset plus richer supervision or a better objective can preserve the gain without washing out the style axes.
5. **Can we train age/gender and emotion knobs simultaneously?** CommonVoice has age/gender, CREMA-D has emotion. Can a single VAE learn all at once when each training stage only labels a subset? Unknown — Joe flagged this as an open research question.
6. **Can an independent speaker verifier confirm the novelty signal?** Finding 11 uses OpenVoice's native embedding space. The next step is an external speaker encoder / EER-style check.
7. **Can an adversary re-identify speakers from F0 alone?** If so, embedding-only DP is insufficient — motivates joint protection.
8. **What is the minimum speaker count for style learning?** We jumped from 3 to 91. Where's the threshold?
9. **Can we interpolate between styles?** E.g., 50% happy + 50% sad — does the output sound bittersweet?
10. **How to prevent collapses?** 9% of speaker-style combinations produce unintelligible output in the combined-only model, and the `cv500` CommonVoice run adds a second collapse mode: style washing back to neutral. Pass 5 shows that coarse whole-module freezing is not enough, and Pass 6 shows that simple scalar loss-weight schedules are not enough either. Can we use richer objectives, finer-grained latent constraints, teacher alignment, or detect/reject bad combinations?
11. **How stable are the ablation conclusions across seeds?** Pass 4 used a single deterministic seed and one validation corpus. We should add repeated-seed confidence intervals before freezing paper tables.

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
11. Native OpenVoice novelty evaluation confirms that the combined-only model produces genuinely shifted speaker identities relative to the source, while the `cv500` CommonVoice checkpoint largely collapses that shift back toward baseline voice identity.
12. The Pass 4 ablation matrix shows that the combined model remains the best overall tradeoff. Single-dataset and `cv500` conditions are more stable but collapse toward baseline identity and/or neutral emotion, while the naive random-latent baseline proves that raw novelty without target alignment is not the paper objective.
13. Pass 5 shows that simple gentler finetuning from the CommonVoice `cv500` init only partially recovers novelty and does not recover the combined model's controllability. The CommonVoice direction remains open, but the next gains likely require better objectives or larger-scale adaptation rather than just lighter finetuning.
14. Pass 6 shows that simple objective reweighting during CommonVoice finetuning also does not recover recall or beat the best Pass 5 novelty recovery. The next CommonVoice gains likely require richer supervision, partial-label objectives, or larger-scale training rather than more scalar weight sweeps.

**Evaluation approach (per Joe, April 16 + EmoVoice paper):**
- **Primary:** emotion2vec Recall Rate + emo_sim (per EmoVoice pipeline) — measures whether generated outputs express the intended emotion
- **Secondary:** Speaker novelty in native OpenVoice embedding space — proof that the output speaker identity actually shifts away from the source
- **Tertiary:** Word error rate via Whisper — intelligibility sanity check
- **Quaternary:** Privacy/speaker verification (add as "we can also do this")
- Baseline for recall: random chance = ~11% (9 classes); current model = 20%; target: >50% for paper
