# V2E-VLA

V2E-VLA is the event-generation and event-processing module used by E-DynVLA.
It converts Isaac Sim RGB/HDR camera observations into raw event streams and
provides lighting-aware static/dynamic separation for downstream VLA models.

## Included components

- v2 balanced, v3 adaptive, and v4 confidence-gated hybrid event generation;
- bidirectional motion-vector frame interpolation;
- HDF5/AEDAT4-compatible event recording and visualization;
- ego-motion and illumination-aware static/dynamic event separation;
- quantitative comparison and aggregation scripts.

## Installation

Install into the Isaac Lab Python environment without replacing Isaac's own
dependencies:

```bash
python -m pip install -e . --no-deps
```

The distribution is named `v2e-vla`; the Python import remains `dvs_gen` so the
adapted Isaac camera code stays compatible:

```python
from dvs_gen.sensors import DVSCamera, DVSCameraCfg
```

## v4-hybrid configuration

```python
dvs = DVSCamera.from_scene(
    scene,
    camera_names,
    adaptive_warp=True,
    hybrid_gate_gain=0.25,
    hybrid_support_radius=2,
)
```

Evaluation and motion-separation entry points are under `scripts/`. See the
repository-level [benchmark documentation](../benchmark/README.md) for the
controlled ten-demo protocol.

This module retains the minimum EVIS core needed by E-DynVLA and preserves its
MIT license. See [../THIRD_PARTY.md](../THIRD_PARTY.md).
