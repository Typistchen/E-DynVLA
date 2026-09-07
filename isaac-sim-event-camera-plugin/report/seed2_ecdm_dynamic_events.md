# Seed 2 ego-compensated dynamic-event pilot

## Goal

This pilot separates wrist-camera events caused by camera ego-motion from events
that are inconsistent with a static world. It produces three event streams:

1. the raw EVIS v4-hybrid stream;
2. an ego-motion-compensated dynamic stream (`q_dyn`);
3. a task-target dynamic stream (`q_target_dyn`).

The target-conditioned stream uses the simulator cup semantic label (`seg == 2`)
and temporal memory. It is an oracle diagnostic, not yet a real-world perception
module. The dynamic confidence itself does not use the target label.

## Controlled input

- Task: `pick_franka_cup06d_O02`
- Seed: `2`
- Camera: wrist camera, 480 x 360
- EVIS source: v4 hybrid
- Frames: 158 (6.32 s at 25 Hz)
- Raw events: 11,285,332
- Geometry capture: metric depth, Isaac motion vectors, per-frame ROS camera
  pose, and camera intrinsics (`--event_dynamic_gt`)

## Method

For every keyframe pair, the extractor back-projects metric depth, transforms the
3-D points with the measured camera poses, and reprojects them to obtain the
static-world optical flow expected from wrist-camera ego-motion. It then:

1. masks invalid reprojections, occlusions, and dilated depth edges;
2. calibrates the Isaac motion-vector temporal anchor, sign, and scale on static
   simulator pixels;
3. computes residual flow between observed and predicted ego flow;
4. converts residual magnitude into a soft dynamic confidence;
5. applies cup semantics plus decaying temporal memory only for the optional
   target-conditioned confidence.

For the balanced setting, the residual threshold is the 95th percentile of the
static-scene residual (`--static-quantile 0.95`).

## Result

| Metric | Balanced q95 |
| --- | ---: |
| Static residual median | 0.000116 px |
| Static residual p95 / threshold | 1.76297 px |
| Static residual p99 | 9.51264 px |
| Background-event suppression | **71.40%** |
| Target-event retention | **58.02%** |
| Robot-event retention | 28.40% |
| Moving-target pixel recall | **96.57%** |
| Dynamic-event fraction | 30.12% |
| Target-dynamic-event fraction | 3.54% |

The stricter q99 pilot suppresses 94.7% of background events, but retains only
30.5% of target events. q95 is therefore the selected VLA-facing setting: it
keeps substantially more of the moving cup while still removing most wrist
ego-motion background events.

## Reproduce extraction

```bash
python scripts/extract_ego_dynamic_events.py \
  --episode /path/to/seed2_episode.h5 \
  --events /path/to/events/env0_ep2.h5 \
  --camera wrist_cam \
  --out-dir /path/to/ecdm_q95 \
  --fps 25 \
  --event-window-ms 10 \
  --static-quantile 0.95
```

The command writes a derived event HDF5 file, JSON metrics, a six-panel video,
a raw/dynamic/target comparison video, and standalone dynamic and target-event
videos.

## Limitations and next evaluation

- This is a one-episode algorithm pilot. It does not establish statistical
  superiority over v2/v3/v4.
- The target stream currently uses simulator ground-truth semantics; deployment
  requires a language-conditioned predicted mask.
- Pose, depth, and simulator motion vectors are privileged signals. A real robot
  needs calibrated pose/depth estimates and uncertainty propagation.
- The next controlled experiment should run the same 10 seeds for raw v4,
  dynamic q95, and target-dynamic q95, followed by VLA success-rate and temporal
  ablations.
