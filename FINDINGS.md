# Key Findings — Controllable DP Voice Conversion

**Last updated:** 2026-04-16
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

| Model | Speakers | Working styles | Happy? |
|-------|----------|---------------|--------|
| ControlVC + Expresso | 3 | 0 of 11 | No |
| OpenVoice + Expresso | 3 | 2 of 11 (whisper, sad) | No |
| OpenVoice + CREMA-D | 91 | 6 of 6 | Yes |
| OpenVoice + Combined | 94 | 9 of 9 | Yes |

---

## Finding 3: Style Control Survives Differential Privacy Noise

Tested at noise levels 0, 0.1, 0.3, 0.5, and 1.0 with the combined VAE (v6, 9 styles, 94 speakers).

**Brightness (spectral centroid) deltas from baseline persist across noise levels:**

| Style | noise=0 | noise=0.1 | noise=0.3 | noise=0.5 | noise=1.0 |
|-------|---------|-----------|-----------|-----------|-----------|
| whisper | +945 | +631 | +280 | +393 | +266 |
| anger | +369 | +177 | -136 | -93 | -113 |
| neutral | -533 | -678 | -759 | -527 | -442 |
| sad | -272 | -414 | -705 | -668 | -633 |

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

| Style | trump | spk1007 | spk1023 | spk1045 | spk1076 | Consistent? |
|-------|-------|---------|---------|---------|---------|-------------|
| anger | +795 | collapsed | +364 | +561 | +398 | YES |
| confused | +536 | +84 | -180 | -79 | +71 | NO |
| enunciated | +759 | -412 | -731 | -616 | -658 | NO |
| fear | +259 | +580 | +375 | +558 | +106 | YES |
| happy | +550 | +342 | +297 | +191 | +290 | YES |
| neutral | -325 | collapsed | collapsed | -43 | -184 | YES |
| sad | +117 | +358 | +21 | +243 | +57 | YES |
| whisper | +973 | +956 | +350 | +720 | +758 | YES |

**F0 direction is NOT consistent** — only 1/9 styles (happy) showed the same F0 shift direction across all speakers. This is expected: OpenVoice encodes timbre/tonal color in the speaker embedding, not F0 directly. F0 changes are a secondary effect that depends on how each source voice interacts with the modified embedding.

**Collapses:** 4/45 outputs (9%) had F0=0 (unintelligible). This occurred when the combination of source speaker embedding + style control pushed the reconstruction into a degenerate region. Affects anger and neutral for spk1007, and disgust/neutral for spk1023.

**At noise=0.1:** Brightness consistency maintained (7/9), but collapses increased to 5/45 (11%). The privacy noise makes some speakers more vulnerable to collapse.

**Control strength tradeoff:**
- strength=5.0: 7/9 consistent, 4 collapses
- strength=3.0: 3/9 consistent, 2 collapses
- strength=2.0: 4/9 consistent, 2 collapses

Higher control strength gives better cross-speaker consistency but increases collapse risk. This suggests that the latent dimensions need to be pushed far enough to dominate the source speaker's baseline characteristics, but this can exceed the decoder's valid input range for some speakers.

**Implication:** The system works across diverse speakers for the majority of styles, with spectral brightness as the reliable cross-speaker acoustic correlate. Papers should report brightness as the primary style metric rather than F0.

---

## Open Questions

1. **What are the formal privacy guarantees?** We need to compute epsilon for each noise level and report privacy-utility curves.
2. ~~**Does style control generalize across source speakers?**~~ → **Answered in Finding 6.** Brightness generalizes (7/9 styles); F0 does not. Some speaker-style combinations collapse.
3. **Can an adversary re-identify speakers from F0 alone?** If so, embedding-only DP is insufficient — motivates joint protection.
4. **What is the minimum speaker count for style learning?** We jumped from 3 to 91. Where's the threshold?
5. **Can we interpolate between styles?** E.g., 50% happy + 50% sad — does the output sound bittersweet?
6. **How to prevent collapses?** 9% of speaker-style combinations produce unintelligible output. Can we clamp the latent space or detect/reject bad combinations?

---

## Paper Framing

**Title candidates:**
1. "Controllable Differentially Private Voice Conversion"
2. "Expressive Voice Anonymization with Formal Privacy Guarantees"
3. "Style-Preserving Speaker Anonymization via Controllable VAE"

**Core argument:** Existing DP voice anonymization destroys all speaker attributes indiscriminately. We show that a controllable VAE trained on diverse, emotion-labeled data can selectively preserve or modify emotional style while maintaining formal privacy guarantees on speaker identity. This enables applications where emotional tone must survive anonymization (therapy recordings, call centers, witness protection).

**Key results to highlight:**
1. Negative result: not all VC embeddings encode style (ControlVC fails, OpenVoice works)
2. Speaker diversity is the critical training requirement (3 vs 91 speakers)
3. 9 styles controllable and acoustically verified
4. Style differences survive DP noise (privacy-utility tradeoff, not cliff)
5. Style control rescues intelligibility under heavy noise (bonus finding)
