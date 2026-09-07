# V2E_VLA

Event-camera simulation for DynamicVLA, combining the EVIS Isaac Sim plugin
with the DOM/DynamicVLA simulation pipeline.

## Repository layout

- `isaac-sim-event-camera-plugin/`: EVIS event generation, multi-threshold
  event model, noise/refractory handling, HDF5 recording, and visualization.
- `dynamic-vla/`: DynamicVLA DOM simulation integration and usage docs.

Both directories retain their original Git histories through subtree imports.
Generated datasets and videos are intentionally not tracked.

## Current checkpoints

- EVIS core multi-threshold event model: `a6eec8b`
- EVIS Isaac/H.264 compatibility: `50a1a7c`
- EVIS event-video tooling: `3967743`
- DynamicVLA EVIS integration: `df4526b`
- EVIS photometric/occlusion-aware warp fix: `8ac5418`
- Balanced dynamic-object-preserving warp mask: `e90175b`
- Seed-2 halo evaluation report: `3161a4f`
- DynamicVLA no-warp control support: `9dbea01`
- EVIS per-event soft confidence: `3e2b183`
- EVIS adaptive temporal knots: `ea838c5`
- Dynamic-target confidence calibration: `a09977d`
- DynamicVLA v3 runtime controls: `50eb6bd`

The `demo2` halo fix and its no-warp control are included. The initial
`8ac5418` mask is retained for comparison but deprecated because it can erase
the manipulated object; use the balanced `e90175b` checkpoint. See
`isaac-sim-event-camera-plugin/docs/photometric_warp_fix.md` for measured event
counts, temporal-burst metrics, limitations, and the next VLA evaluation step.

The experimental `feat/continuous-events-v3` history is also imported. It adds
adaptive `K=4..8` temporal sampling and an HDF5 `q` confidence dataset while
keeping all visible dynamic-object events. The seed-2 cup remains intact, but
v3 is not promoted over balanced v2 yet because its raw 10 ms phase-imbalance
metric is slightly worse on this episode. Fixed-K balanced v2 therefore remains
the default; pass `--event_adaptive_warp` to opt into v3. Do not hard-threshold
`q` in a VLA loader.

See `dynamic-vla/docs/event_camera.md` for setup and generation commands.
