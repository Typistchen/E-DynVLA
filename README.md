# E-DynVLA

E-DynVLA combines DOM robot-manipulation episodes with v4-hybrid event-camera
simulation, lighting-suppressed ego-motion separation, sparse static/dynamic
event tokens, DynamicVLA action prediction, and an optional Event-WAM
short-horizon prediction objective.

The repository starts from a clean, squashed source snapshot so its GitHub
contributor graph reflects development in this repository. The upstream
licenses and attribution remain in the corresponding source directories.

## New event-token path

- Architecture: [`dynamic-vla/docs/edynvla_architecture.md`](dynamic-vla/docs/edynvla_architecture.md)
- Training preset: [`dynamic-vla/configs/edynvla.yaml`](dynamic-vla/configs/edynvla.yaml)
- Event tokenizer: [`dynamic-vla/policies/edynvla/event_tokenizer.py`](dynamic-vla/policies/edynvla/event_tokenizer.py)
- Event-WAM head: [`dynamic-vla/policies/edynvla/event_wam.py`](dynamic-vla/policies/edynvla/event_wam.py)
- DOM/event adapter: [`dynamic-vla/policies/edynvla/data.py`](dynamic-vla/policies/edynvla/data.py)

## Repository layout

- `isaac-sim-event-camera-plugin/`: EVIS event generation, multi-threshold
  v4-hybrid event model, confidence, HDF5 recording, motion separation, and
  evaluation tools.
- `dynamic-vla/`: DOM simulation, DynamicVLA policy, paired DOM/event loader,
  sparse event tokenizer, and Event-WAM head.
- `benchmark/`: controlled EVIS results and reproducible dataset manifests.

Generated datasets and videos are intentionally not tracked. Set
`EDYNVLA_DATA_ROOT` to the external paired DOM/event dataset directory before
using `dynamic-vla/configs/edynvla.yaml`.

## Provenance

This project builds on DynamicVLA and the JHU Isaac Sim event-camera plugin.
Their licenses and author attribution are preserved inside the corresponding
directories. Git history is intentionally squashed at import so the GitHub
contributor graph records contributions made specifically in E-DynVLA.

See `dynamic-vla/docs/event_camera.md` for event generation commands and
`dynamic-vla/docs/edynvla_architecture.md` for the new model/data contract.

## Quick start for the paired ten-demo set

```bash
export EDYNVLA_DATA_ROOT=/path/to/motion_separation_10

python dynamic-vla/scripts/build_edynvla_manifest.py \
  --root "$EDYNVLA_DATA_ROOT" \
  --output benchmark/manifests/dom_event_10demo.json

cd dynamic-vla
python run.py --cfg configs/edynvla.yaml --gpus 0
```

The ten demos validate data alignment and model wiring. They are not treated as
sufficient data to train the Small VLM from scratch; the intended training set
is the full DOM corpus with matching eventized episodes.
