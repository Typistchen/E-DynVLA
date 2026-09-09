# Seed-2 EVIS v2/v3 source-level metrics

> **Historical, superseded report:** this report was generated before event
> timestamps were shifted from absolute Isaac time to episode-relative time.
> Its boundary/halo/static ROI values are misaligned by approximately 70 ms and
> must not be used as final evidence. See
> [the corrected 10-episode report](ten_episode_v2_v3_aligned_metrics.md).

This report compares balanced v2 (fixed `K=4`) and experimental v3
(`K=4..8`, per-event confidence) on the same DOM cup episode. It does not use a
VLA downstream task.

## Controlled comparison

- 158 frames, 360x480, 25 Hz keyframes, 6.32 s.
- Robot/object state and action arrays are exactly equal.
- All three semantic-segmentation sequences are pixel-exact.
- RGB differs only by small renderer numerical variation (MAE: opst 0.050,
  side 0.037, wrist 0.191 on uint8 values).
- All six event streams have monotonic timestamps.

## Event and temporal metrics

| Camera | Version | Events | ON/OFF imbalance | 10 ms CV | Phase imbalance | Harmonic power | 25 Hz line (dB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| opst | v2 | 4,520,820 | 0.0123 | 1.4917 | 1.3476 | 0.6366 | 16.15 |
| opst | v3 | 4,895,733 | 0.0113 | 1.5355 | 1.3863 | 0.6522 | 19.15 |
| side | v2 | 2,322,899 | 0.0140 | 1.2048 | 1.4230 | 0.8168 | 21.45 |
| side | v3 | 2,458,885 | 0.0133 | 1.2408 | 1.4567 | 0.8012 | 22.13 |
| wrist | v2 | 9,802,022 | 0.0131 | 2.1178 | 1.1674 | 0.2592 | 5.24 |
| wrist | v3 | 11,509,404 | 0.0108 | 2.1076 | 1.2667 | 0.2660 | 7.40 |

Total events increase from 16,645,741 to 18,864,022 (+13.33%). The increase is
largest on the wrist camera (+17.42%). Polarity balance improves slightly, but
the 25 Hz spectral line is stronger in all cameras. Raw phase imbalance also
worsens by 2.88% / 2.37% / 8.51% for opst / side / wrist. Side-camera total
harmonic power is the one temporal artifact metric that improves (-1.90%).

## Cup-boundary and leakage proxies

The cup boundary is derived from DOM label 2 and evaluated only on intervals
where its segmentation changes. The halo annulus is background-only and lies
3--8 pixels outside the swept cup mask. Static leakage uses unchanged stored
LDR pixels away from semantic edges; because EVIS uses HDR, it is a controlled
proxy, not a false-positive oracle.

| Camera | Version | Cup boundary coverage | Boundary density | Halo/boundary density | Static events/MPix/s |
| --- | --- | ---: | ---: | ---: | ---: |
| opst | v2 | 0.2182 | 0.7113 | 2.4071 | 1,504,114 |
| opst | v3 | 0.2228 | 0.8461 | 2.2330 | 1,584,149 |
| side | v2 | 0.2197 | 1.6103 | 0.2511 | 756,486 |
| side | v3 | 0.2207 | 1.6725 | 0.2580 | 800,266 |
| wrist | v2 | 0.3859 | 1.5916 | 0.8136 | 2,664,835 |
| wrist | v3 | 0.3969 | 1.8811 | 0.7674 | 3,100,844 |

v3 raises cup-boundary coverage by 2.09% / 0.45% / 2.84% and boundary density
by 18.94% / 3.86% / 18.19%. Relative halo density improves on opst (-7.23%)
and wrist (-5.68%) but becomes 2.76% worse on side. Raw static-region event
rate increases by 5.32% / 5.79% / 16.36%.

With `q` used as a soft weight, v3 static rates become 1,387,501 / 717,939 /
2,116,492 events/MPix/s, respectively 7.75% / 5.10% / 20.58% below unweighted
v2. This is not proof of correctness: without an oracle, lowering every
uncertain event can trivially lower the metric. In particular, wrist boundary
mean `q=0.649` and halo mean `q=0.671`, so wrist `q` does not yet rank the halo
below the task boundary.

## Cross-version occupied-voxel agreement

| Camera | 1px/1ms F1 | 2px/2ms F1 | 4px/5ms F1 |
| --- | ---: | ---: | ---: |
| opst | 0.6334 | 0.7576 | 0.8134 |
| side | 0.6644 | 0.7938 | 0.8430 |
| wrist | 0.4666 | 0.6108 | 0.7196 |

This is a symmetric similarity score between versions, not accuracy. Wrist has
the largest fine-time change, consistent with adaptive sampling being triggered
most strongly by camera/arm/cup motion.

## Confidence and storage

Mean v3 confidence is 0.822 / 0.852 / 0.687 (opst / side / wrist). The fraction
at the visible-event floor `q=0.5` is 15.23% / 8.75% / 17.20%. The HDF5 event
file grows from 134,098,104 bytes (8.06 bytes/event) to 173,190,948 bytes
(9.18 bytes/event), mainly due to the new float16 `q` dataset and extra events.

## Conclusion

On this episode, v3 better preserves the manipulated-object boundary and
reduces the relative cup halo in two of three views. It also produces more
static-region events and stronger 25 Hz timing structure. The evidence supports
keeping v3 experimental and balanced v2 as the default. Promotion requires a
high-rate rendered oracle to compute event F1, timestamp MAE, log-intensity
RMSE, and calibration metrics (ECE/Brier/AUROC) for `q`.

The reusable evaluator is `scripts/evaluate_evis_versions.py`; its full JSON
output includes all raw counts, proxy accumulators, quality quantiles, and
occupied-voxel intersections.
