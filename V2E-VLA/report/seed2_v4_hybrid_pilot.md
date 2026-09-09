# Seed-2 v4 hybrid pilot

This controlled DOM cup pilot compares balanced v2, adaptive v3, and v4 hybrid
(`gate_gain=0.25`, same-polarity support radius `2 px`). State/action arrays and
semantic segmentation are exact across runs. All event streams are aligned
using their recorded Isaac episode time origin.

## Three-version result

Values below are means across opst, side, and wrist cameras for this episode.

| Metric | Better | v2 | adaptive v3 | v4 hybrid |
| --- | --- | ---: | ---: | ---: |
| Events/camera | neutral | 5,120,452 | 6,288,007 | 6,209,795 |
| ON/OFF imbalance | lower | 0.01465 | **0.01180** | 0.01246 |
| 10 ms count CV | lower | 1.8350 | **1.6275** | 1.6325 |
| Phase imbalance | lower | 2.6997 | **1.3613** | 1.3783 |
| Harmonic power ratio | lower | 0.7496 | **0.5773** | 0.5832 |
| 25 Hz keyframe dB | lower | 19.48 | **16.45** | 16.52 |
| Object-boundary coverage | higher | 0.4894 | **0.5396** | 0.4873 |
| Object-boundary density | higher | 2.5239 | 3.1226 | **3.1592** |
| Halo/boundary ratio | lower | 0.1516 | 0.1609 | **0.1291** |
| Static events/MPix/s | lower | **451,607** | 686,172 | 463,208 |

Relative to v2, v4 keeps essentially the same mean boundary coverage, raises
boundary density by 25%, lowers halo/boundary by 15%, and keeps static leakage
within 2.6%, while substantially improving every temporal-artifact proxy.

Relative to adaptive v3, v4 changes the temporal metrics by only 0.3--1.3%,
but lowers halo/boundary by 19.8% and static events by 32.5%. Mean boundary
coverage falls by 9.7%, so the camera-specific result matters for VLA use.

## Wrist-camera result

| Metric | Better | v2 | adaptive v3 | v4 hybrid |
| --- | --- | ---: | ---: | ---: |
| Boundary coverage | higher | 0.6752 | **0.7287** | 0.6994 |
| Halo/boundary ratio | lower | 0.3994 | 0.4163 | **0.3390** |
| Static events/MPix/s | lower | **960,701** | 1,338,034 | 979,647 |

For the VLA-facing wrist view, v4 retains more object boundary than v2 while
producing less halo than both v2 and v3; its static rate is close to v2 and far
below v3. This is the intended hybrid trade-off.

## Status

The pilot is good enough to proceed to the same 10-episode paired evaluation,
but one episode is not sufficient to promote v4. Balanced v2 remains the
default; v3 and v4 remain opt-in experiments.
