# Controlled EVIS v2/v3/v4 benchmark

This benchmark regenerates all three algorithms from the same current source
tree. Seeds, scene/object roots, event threshold, HDR source, cameras, and warp
count remain fixed. Only the algorithm flags differ:

- `v2_balanced`: no adaptive or hybrid flag.
- `v3_adaptive`: `--event_adaptive_warp`.
- `v4_hybrid`: `--event_hybrid --event_hybrid_gate_gain 0.25
  --event_hybrid_support_radius 2`.

Every event HDF5 is checked against its expected `evis_mode`, adaptive flag,
hybrid parameters, time-origin metadata, camera datasets, and event lengths.
Existing captures are reused only after they pass this validation and are never
silently overwritten.

Generate captures and comparison videos:

```bash
./benchmark/run_evis_10demo_benchmark.sh all 1 10
```

Evaluate v4 against v2 and v3 with per-file HDF5 time origins:

```bash
./benchmark/evaluate_evis_10demo_benchmark.sh
```

Outputs are written outside Git to
`datasets/evis_benchmark_10/{v2_balanced,v3_adaptive,v4_hybrid}`. Pairwise JSON
and Markdown reports are written below `datasets/evis_benchmark_10/metrics`.
