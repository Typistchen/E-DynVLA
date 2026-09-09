# EVIS v2/v3 aligned 10-episode evaluation

This report compares balanced v2 with the latest photometric v3 on ten paired
DOM episodes. Event timestamps were shifted by the measured Isaac episode time
origin (`0.07 s`) before temporal, ROI, and voxel metrics were calculated.

## Protocol

- State/action arrays are exact in all 10 episode pairs.
- Semantic segmentation is pixel-exact in all 10 episode pairs.
- One episode is one independent sample; cameras are first averaged within an
  episode, then paired episode differences are summarized.
- Confidence metrics are unavailable because these photometric event files do
  not contain `q`.
- Bootstrap intervals and paired Wilcoxon p-values are exploratory at n=10.

## Episode-level results

| Metric | Better | v2 | v3 | v3-v2 | Improved episodes | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Events/camera | neutral | 7,584,483 | 7,931,603 | +347,120 | n/a | 0.00195 |
| ON/OFF imbalance | lower | 0.0508 | 0.0494 | -0.0013 | 9/10 | 0.00586 |
| 10 ms count CV | lower | 2.3394 | 1.6663 | -0.6730 | 10/10 | 0.00195 |
| Phase imbalance | lower | 2.0683 | 1.3841 | -0.6841 | 10/10 | 0.00195 |
| Harmonic power ratio | lower | 0.6428 | 0.6051 | -0.0377 | 5/10 | 0.32227 |
| 25 Hz keyframe dB | lower | 16.2328 | 16.2582 | +0.0254 | 3/10 | 0.69531 |
| Object-boundary coverage | higher | 0.5375 | 0.5839 | +0.0464 | 10/10 | 0.00195 |
| Object-boundary density | higher | 3.3648 | 3.8140 | +0.4492 | 10/10 | 0.00195 |
| Halo/boundary ratio | lower | 0.1637 | 0.2405 | +0.0768 | 0/10 | 0.00195 |
| Static events/MPix/s | lower | 891,559 | 1,203,827 | +312,268 | 0/10 | 0.00195 |

The paired bootstrap 95% intervals for boundary coverage, halo ratio, and
static rate changes are `[+0.0363, +0.0588]`, `[+0.0504, +0.1063]`, and
`[+165,394, +505,162]`, respectively. Their directions are consistent across
all ten episodes.

Fine occupied-voxel similarity (`1 px / 1 ms`) is F1 `0.493`, precision `0.464`,
and recall `0.528` versus v2. These values measure algorithmic change, not
accuracy against a ground-truth event stream.

## Wrist-camera diagnosis

The wrist view has the strongest task-relevant gain and the strongest leakage:

- Boundary coverage: `+0.0609`, improved in 10/10 episodes.
- Boundary density: `+0.8680`, improved in 10/10 episodes.
- Halo/boundary ratio: `+0.1712`, worse in 10/10 episodes.
- Static event rate: `+778,059 events/MPix/s`, worse in 10/10 episodes.
- Harmonic power: `+0.0510`, worse in 8/10 episodes.
- 25 Hz line: `+2.0623 dB`, worse in 8/10 episodes.

## Conclusion

v3 reliably distributes events more evenly and preserves substantially more of
the moving-object boundary. It does not reliably suppress the 25 Hz keyframe
line, and it consistently increases halo and static-region activity—especially
on the wrist camera used by a VLA policy. Balanced v2 should therefore remain
the default. The next v3 change should retain continuous threshold timing while
adding photometric/visibility gating for background and halo events; a
high-rate rendered oracle is still required for event F1 and timestamp error.
