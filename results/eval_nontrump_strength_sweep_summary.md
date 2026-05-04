# Non-Trump Style-Strength Sweep Summary

## Fixed panel

- `female_1_cremad_1002`
- `female_2_cremad_1012`
- `male_1_cremad_1003`
- `male_2_cremad_1051`

## Overall metrics by strength

| Strength | Recall | emo_sim | Novelty gain | Mean WER | Mean MOS delta |
|----------|--------|---------|--------------|----------|----------------|
| `5.0` | 0.2083 | 0.9874 | 0.0789 | 0.1472 | -0.1312 |
| `7.5` | 0.1667 | 0.9773 | 0.1246 | 0.1681 | -0.1701 |
| `10.0` | 0.1667 | 0.9726 | 0.1570 | 0.2028 | -0.2287 |
| `12.5` | 0.1667 | 0.9728 | 0.1761 | 0.2444 | -0.2119 |

## Focus styles

| Strength | Style | Recall | emo_sim | Novelty gain | WER | MOS delta |
|----------|-------|--------|---------|--------------|-----|-----------|
| `5.0` | `confused` | - | 0.9796 | 0.2871 | 0.1125 | -0.5253 |
| `5.0` | `happy` | 0.0000 | 0.9951 | -0.0325 | 0.0625 | -0.0182 |
| `5.0` | `sad` | 0.2500 | 0.9845 | 0.0338 | 0.0625 | -0.0166 |
| `5.0` | `whisper` | - | 0.9617 | 0.3651 | 0.3000 | -0.6416 |
| `7.5` | `confused` | - | 0.9625 | 0.4549 | 0.1125 | -0.6577 |
| `7.5` | `happy` | 0.0000 | 0.9897 | -0.0239 | 0.1875 | -0.0211 |
| `7.5` | `sad` | 0.0000 | 0.9887 | 0.0367 | 0.0625 | -0.0182 |
| `7.5` | `whisper` | - | 0.8953 | 0.5123 | 0.3000 | -0.7248 |
| `10.0` | `confused` | - | 0.9691 | 0.5341 | 0.1125 | -0.9787 |
| `10.0` | `happy` | 0.0000 | 0.9902 | -0.0326 | 0.2500 | -0.0233 |
| `10.0` | `sad` | 0.0000 | 0.9857 | 0.0437 | 0.0625 | -0.0179 |
| `10.0` | `whisper` | - | 0.8750 | 0.6042 | 0.3000 | -0.7918 |
| `12.5` | `confused` | - | 0.9600 | 0.5694 | 0.1125 | -1.0095 |
| `12.5` | `happy` | 0.0000 | 0.9895 | -0.0332 | 0.2500 | -0.0208 |
| `12.5` | `sad` | 0.0000 | 0.9866 | 0.0649 | 0.0625 | -0.0175 |
| `12.5` | `whisper` | - | 0.8867 | 0.6521 | 0.3000 | -0.7141 |

## Notes

- `Recall` is only defined for emotional styles that have a mapped target in the emotion classifier.
- `Novelty gain` is positive when the styled output moves farther from the source than plain baseline conversion did.
- `MOS delta` is relative to the same-speaker baseline conversion; values closer to `0.0` are better.
