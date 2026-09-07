# Lighting-suppressed motion separation v2: ten-demo report

## Scope

Ten deterministic DOM pick episodes (seeds 1--10) were regenerated with the
same v4-hybrid EVIS configuration and wrist geometry capture. The classifier
outputs four event confidences:

- `q_motion`: motion inconsistent with a static scene;
- `q_static`: ego-consistent static-scene events;
- `q_dynamic`: motion after illumination suppression;
- `q_illumination` / `q_unknown`: brightness-only or ambiguous events.

Semantic segmentation and object velocity are not used by calibration,
threshold estimation, or classification. They are loaded after inference only
to evaluate background leakage and target retention.

## Method

The static-world flow is predicted from metric depth, consecutive wrist-camera
poses, and intrinsics. Motion confidence is the maximum of:

1. absolute residual flow confidence; and
2. residual flow normalized by `1 + |ego flow|`.

This dual-scale rule preserves slow independent motion during fast wrist motion
without using a class mask. The illumination gate detects exposure-compensated
log-intensity changes that lack depth, chromatic, or short-term motion support.
It downweights these events instead of forcing them into the static or dynamic
class.

The simulator pilot still uses Isaac motion vectors as the observed-flow upper
bound. They must be replaced by estimated RGB/event flow or RGB-D scene flow
for real deployment.

## Per-demo results

| Demo | Dynamic events | Background retained | Target retained | Illumination events | Motion rejected by light gate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.68% | 8.95% | 82.77% | 3.04% | 0.64% |
| 2 | 28.56% | 26.61% | 58.95% | 5.16% | 8.84% |
| 3 | 24.14% | 22.17% | 56.01% | 0.60% | 0.88% |
| 4 | 18.99% | 11.34% | 88.30% | 0.46% | 1.44% |
| 5 | 11.73% | 5.20% | 81.15% | 1.21% | 1.43% |
| 6 | 17.17% | 13.33% | 70.49% | 2.60% | 2.07% |
| 7 | 28.29% | 19.50% | 86.39% | 5.74% | 4.29% |
| 8 | 32.86% | 32.60% | 86.91% | 2.79% | 4.89% |
| 9 | 20.33% | 18.80% | 51.12% | 9.31% | 10.23% |
| 10 | 34.11% | 28.76% | 81.44% | 5.98% | 6.33% |

## Ten-demo summary

| Metric | Mean +/- std | 95% CI |
| --- | ---: | ---: |
| Target-event retention | **74.35% +/- 14.11%** | [64.26%, 84.45%] |
| Background dynamic leakage | **18.73% +/- 9.01%** | [12.28%, 25.17%] |
| Moving-target pixel recall | 82.35% +/- 19.83% | [68.17%, 96.53%] |
| Dynamic-event fraction | 25.39% +/- 8.32% | [19.43%, 31.34%] |
| Static-event fraction | 59.41% +/- 10.29% | [52.05%, 66.77%] |
| Illumination-event fraction | 3.69% +/- 2.82% | [1.67%, 5.71%] |
| Motion candidates rejected by light gate | 4.10% +/- 3.44% | [1.65%, 6.56%] |

The retained target/background ratio improves from 3.78 before the lighting
gate to 3.97 after it. The gate removes 4.99% of background motion candidates
and 1.24% of target candidates on average. Background retention decreases by
1.16 percentage points (10/10 episodes, Wilcoxon p=0.0020); target retention
decreases by 0.83 percentage points (Wilcoxon p=0.0077).

## Interpretation

The dual-scale v2 result is the selected VLA-facing representation: compared
with the absolute-only pilot it raises mean target-event retention by 27.34
percentage points, with a 6.00-point increase in background leakage. Sampled
videos retain the manipulated object in the dynamic channel while moving most
ego-induced background structure into the static channel.

This does not prove illumination precision/recall because the simulator does
not provide per-event illumination-causality labels. The illumination channel
is a geometry/photometry proxy. Demo 10 also has low full-mask moving-target
pixel recall (28.11%) despite 81.44% target-event retention, showing that the
dense confidence map still needs improvement even though sparse target events
remain available.

## Reproduce

```bash
# Capture and process seeds 1--10.
./benchmark/run_motion_separation_10demo.sh 1 10

# Rebuild the aggregate JSON and Markdown report.
python scripts/aggregate_motion_separation.py \
  /home/typist/dataset/dom_simulation/datasets/motion_separation_10
```

The complete local dataset is stored under
`datasets/motion_separation_10/demo{1..10}/motion_separation_v2`.
