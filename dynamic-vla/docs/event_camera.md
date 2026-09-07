# DOM event-camera generation

The DOM simulator can use the EVIS `dvs_gen` plugin while retaining its RGB,
semantic-segmentation, robot-state, and action outputs.

Install the local plugin into the IsaacLab environment:

```bash
/home/typist/miniconda3/envs/isaaclab/bin/python -m pip install \
  -e /home/typist/dataset/dom_simulation/isaac-sim-event-camera-plugin --no-deps
```

Generate one episode with three 360x480 DOM event cameras. The base cameras run
at 25 Hz and `--event_warp 4` requests at least four temporal intervals per
keyframe gap. EVIS v3 adaptively raises difficult motion/HDR gaps to at most
eight intervals (`--event_max_warp_factor 2`):

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/typist/miniconda3/envs/isaaclab/bin/python simulations/simulate.py \
  --headless --enable_cameras --num_envs 1 -n 1 --task pick --seed 44 \
  --save --debug --event_camera --event_warp 4 --event_adaptive_warp \
  --event_source hdr --event_threshold 0.15 \
  --scene_dir /home/typist/dataset/dom_simulation/tests/DOM-Test/scenes \
  --object_dir /home/typist/dataset/dom_simulation/objects/DOM-3D-Objects/objects \
  -o /home/typist/dataset/dom_simulation/datasets/dom_event
```

Event files are written under `<output_dir>/events/env<id>_ep<seed>.h5`. Each
contains `DVS/wrist_cam`, `DVS/opst_cam`, and `DVS/side_cam`, with `x`, `y`, `t`,
`p`, and `q` datasets. `q` is the per-event soft warp confidence: `0` is
unobservable, visible synthesized events are in `[0.5,1]`, and real samples
default to `1`. DynamicVLA can use it as a voxel weight or auxiliary channel;
do not hard-threshold it, because uncertain cup/robot edges are task-relevant.
Set `--no-event_adaptive_warp` for the fixed-K v2 comparison,
`--event_warp 1` to disable interpolation, or `--event_source ldr` to generate
events from tone-mapped RGB instead of HDR.
