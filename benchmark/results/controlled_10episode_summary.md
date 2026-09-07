# Controlled 10-episode EVIS v2/v3/v4 benchmark

## Protocol

All three versions were regenerated from the same current source tree and the
same seeds (1--10). Scene/object roots, state machine, cameras, HDR input,
threshold (`0.15`), warp count (`4`), anti-aliasing, and rendering settings were
held fixed. Only the EVIS algorithm flags changed:

- v2: `evis_mode=v2_balanced`, adaptive off, hybrid gain `0`.
- v3: `evis_mode=v3_adaptive`, adaptive on, hybrid gain `0`.
- v4: `evis_mode=v4_hybrid`, adaptive on, hybrid gain `0.25`, support radius
  `2 px`.

Source revisions were DynamicVLA `58673d671e98c8571c5d83517dab2fb5a30e8f22`
and EVIS `3c4a5aa678fd80a165c0b04057d4d8d4dd207945`. Every capture records the same
event time origin (`0.0799999982 s`). All 20 pair reports passed exact
state/action and segmentation checks. The episode is the independent unit;
the three cameras are averaged inside each episode. Confidence intervals are
paired episode bootstraps, and p-values are exploratory paired two-sided
Wilcoxon tests.

## Three-version means

| Metric | Direction | v2 balanced | v3 adaptive | v4 hybrid |
| --- | --- | ---: | ---: | ---: |
| Events/camera | neutral | 7,931,382 | 9,251,616 | 9,172,555 |
| Polarity imbalance | lower | 0.0494 | **0.0476** | 0.0478 |
| 10 ms count CV | lower | **1.6663** | 1.7017 | 1.7051 |
| Phase imbalance | lower | **1.4002** | 1.4473 | 1.4547 |
| Harmonic power ratio | lower | **0.6071** | 0.6249 | 0.6269 |
| 25 Hz keyframe dB | lower | **16.2806** | 18.3258 | 18.3801 |
| Object-boundary coverage | higher | 0.5420 | **0.5573** | 0.5537 |
| Object-boundary density | higher | 3.8716 | **4.3670** | 4.3554 |
| Halo/boundary ratio | lower | **0.1971** | 0.2252 | 0.2248 |
| Static events (Mpix/s) | lower | **0.781 M** | 1.046 M | 1.004 M |

## v4 hybrid versus v3 adaptive

| Metric | v4-v3 | Relative | Improved demos | 95% CI | p | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Events/camera | -79,061 | -0.85% | n/a | [-98,356,-59,191] | 0.002 | Small event reduction |
| Static rate | **-42,465** | **-4.06%** | **10/10** | [-61,670,-24,863] | **0.002** | Consistent improvement |
| Halo/boundary | -0.00040 | -0.18% | 9/10 | [-0.00105,+0.00054] | 0.084 | Tiny, not significant |
| Boundary coverage | **-0.00358** | **-0.64%** | **0/10** | [-0.00506,-0.00248] | **0.002** | Small consistent loss |
| Boundary density | -0.01169 | -0.27% | 0/10 | [-0.01831,-0.00740] | 0.002 | Small consistent loss |
| 10 ms CV | +0.00336 | +0.20% | 1/10 | [+0.00185,+0.00495] | 0.004 | Slightly worse |
| Phase imbalance | +0.00744 | +0.51% | 0/10 | [+0.00545,+0.00960] | 0.002 | Slightly worse |
| Harmonic ratio | +0.00200 | +0.32% | 2/10 | [+0.00105,+0.00295] | 0.010 | Slightly worse |
| 25 Hz keyframe dB | +0.054 | +0.30% | 2/10 | [-0.096,+0.171] | 0.193 | No supported difference |

At 1 px/1 ms, v4 and v3 have voxel F1 `0.8377` (precision `0.8411`, recall
`0.8344`). Their event streams are therefore close but not identical. Mean
event confidence is effectively unchanged (`0.78268` to `0.78251`). The
confidence-weighted static rate falls from `741,637` to `706,645 Mpix/s`, a
`4.72%` reduction, consistent with the raw static proxy.

## v4 hybrid versus v2 balanced

V4 retains more object-boundary information than v2: coverage rises by
`+0.0117` (`10/10`, p=`0.002`) and boundary density by `+0.4837` (`10/10`,
p=`0.002`). However, it does not recover v2 cleanliness or temporal behavior:
halo/boundary increases by `+0.0277`, static rate by `+222,593 Mpix/s`, 10 ms
CV by `+0.0387`, phase imbalance by `+0.0545`, and 25 Hz keyframe dB by
`+2.0995` (all `0/10` improved and p=`0.002`).

## Conclusion

With `gate_gain=0.25` and `support_radius=2`, v4 is not an overall replacement
for v3. It buys a reproducible `4.1%` reduction in static leakage while losing
about `0.6%` object-boundary coverage and slightly worsening temporal
uniformity. The halo improvement is too small to distinguish from episode
variation. Relative to v2, v4 has stronger object-boundary support but remains
substantially less clean and less temporally uniform.

For a VLA that values motion/object boundaries most, v3 is currently the safer
default. V4 is defensible when static-event reduction is worth a very small
boundary loss, but it has not yet achieved the intended combination of v2
cleanliness and v3 completeness. Downstream VLA evaluation and a high-rate
rendering oracle remain necessary before making a task-level accuracy claim.

## Artifact locations

- Dataset root: `/home/typist/dataset/dom_simulation/datasets/evis_benchmark_10`
- v2-v4 aggregate: `metrics/v2_balanced_vs_v4_hybrid_10episodes.{json,md}`
- v3-v4 aggregate: `metrics/v3_adaptive_vs_v4_hybrid_10episodes.{json,md}`
- Individual reports: `metrics/<pair>/evis_demo{1..10}.{json,md}`
- Every demo contains the RGB video, event video, comparison video, event HDF5,
  episode HDF5, simulation log, and `validation.json`.
